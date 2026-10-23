"""Purged short-horizon HGB fits and causal paper-only opportunity discovery."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import fast_market
from .features import _validated_frame, forward_returns, make_features
from .fees import official_public_rates
from .horizon_benchmark import make_targets
from .model_signals import _sha
from .spot import canonical_symbol
from .trees import TrainClip


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _probability(classifier, calibrator, values):
    probability = classifier.predict_proba(values)[:, 1]
    if calibrator is not None:
        logit = np.log(np.clip(probability, 1e-6, 1-1e-6) / np.clip(1-probability, 1e-6, 1))
        probability = calibrator.predict_proba(logit[:, None])[:, 1]
    return probability


def _simulation(frame, prediction, start, end, horizon, threshold, config):
    from .engine import backtest
    target, exits = make_targets(frame, prediction, start, end, horizon, threshold, config["max_exposure"])
    return backtest(frame.iloc[:end], target[:end], {**config, "forced_exit_indices": exits}, start=start)["metrics"]


def fit_market(frame, symbol, quote, directory, config=None):
    """Fit TRAIN70%, select on purged VAL15%; final15% is diagnostic only."""
    config = dict(config or {})
    bars, times, gaps, interval = _validated_frame(frame)
    if interval != 300 or len(frame) < 240:
        raise ValueError("paper fitting requires at least 240 genuine M5 candles")
    if canonical_symbol(symbol).replace("/", "") != symbol or not symbol.endswith(quote):
        raise ValueError("canonical compact symbol and quote must agree")
    rates = official_public_rates()
    if quote not in rates or float(config.get("fee_bps", rates[quote])) != rates[quote]:
        raise ValueError("paper model fees must match the verified public first-tier taker schedule")
    fee, slip = rates[quote], float(config.get("slippage_bps", 0))
    if not np.isfinite(slip) or slip < 0:
        raise ValueError("explicit slippage must be finite and nonnegative")
    execution = dict(initial_cash=float(config.get("initial_cash", 1000)), max_exposure=float(config.get("max_exposure", .25)),
                     min_notional=float(config.get("min_notional", 1 if quote == "USDT" else 50000)),
                     fee_bps=fee, slippage_bps=slip)
    cost = 2 * (fee + slip) / 10000
    train_end, val_end = int(len(frame) * .70), int(len(frame) * .85)
    features = make_features(frame)
    X = features.to_numpy(dtype=np.float32)
    rows = np.arange(len(frame))
    cumulative = np.cumsum(gaps)
    valid = (rows >= 48) & (cumulative == cumulative[np.maximum(rows - 48, 0)])
    trace, fitted = [], {}
    iterations = int(config.get("tree_iterations", 70))
    if not 1 <= iterations <= 150:
        raise ValueError("tree_iterations must lie in [1,150]")
    with threadpool_limits(limits=2):
        for horizon in (3, 6, 12):
            labels = forward_returns(frame, horizon).to_numpy()
            train = valid & np.isfinite(labels) & (rows + horizon < train_end)
            validation = valid & np.isfinite(labels) & (rows >= train_end) & (rows + horizon < val_end)
            if train.sum() < 80 or validation.sum() < 15:
                raise ValueError("insufficient mature contiguous TRAIN/VAL rows")
            settings = dict(max_iter=iterations, max_leaf_nodes=15, min_samples_leaf=20,
                            learning_rate=.05, early_stopping=False, random_state=17)
            regressor = make_pipeline(TrainClip(), StandardScaler(), HistGradientBoostingRegressor(**settings))
            regressor.fit(X[train], labels[train])
            prediction = regressor.predict(X)
            prediction[~valid] = np.nan
            event = (labels > cost).astype(int)
            classifier, calibrator = None, None
            probability = {"available": False, "reason": "TRAIN has only one event class"}
            if len(np.unique(event[train])) == 2:
                classifier = make_pipeline(TrainClip(), StandardScaler(), HistGradientBoostingClassifier(**settings))
                classifier.fit(X[train], event[train])
                raw = classifier.predict_proba(X[validation])[:, 1]
                outcomes = event[validation]
                positives = int(outcomes.sum())
                if min(positives, len(outcomes) - positives) >= 5 and len(outcomes) >= 30:
                    logits = np.log(np.clip(raw, 1e-6, 1-1e-6) / np.clip(1-raw, 1e-6, 1))
                    calibrator = LogisticRegression(C=1., random_state=17).fit(logits[:, None], outcomes)
                calibrated = _probability(classifier, calibrator, X[validation])
                probability = dict(available=True, event="gross horizon close return exceeds stored roundtrip fee/slippage",
                    event_cost_fraction=cost, rows=int(validation.sum()), positive_rows=positives,
                    prevalence=float(outcomes.mean()), predicted_mean=float(calibrated.mean()),
                    mean_bias=float(np.mean(calibrated - outcomes)), brier=float(np.mean((calibrated-outcomes)**2)),
                    raw_brier=float(np.mean((raw-outcomes)**2)),
                    calibration="VAL-fitted sigmoid" if calibrator is not None else "uncalibrated TRAIN HGB probability",
                    caveat="Calibration diagnostics reuse the fitted validation block; no independent coverage or probability guarantee")
            residual = labels[validation] - prediction[validation]
            quantiles = np.quantile(residual, [.1, .5, .9]).tolist()
            fitted[horizon] = dict(regressor=regressor, classifier=classifier, calibrator=calibrator,
                                   residual_quantiles=quantiles, probability_validation=probability,
                                   prediction=prediction, last_train_label_endpoint=int(rows[train][-1] + horizon))
            for margin in (0., .001, .0025):
                metrics = _simulation(frame, prediction, train_end, val_end-horizon, horizon, cost+margin, execution)
                trace.append(dict(horizon=horizon, margin=margin, validation_net_return=float(metrics["total_return"]),
                    validation_trades=int(metrics["roundtrip_count"]),
                    last_validation_label_endpoint=int(rows[validation][-1]+horizon)))
    selected = max(trace, key=lambda item: (item["validation_net_return"], -item["validation_trades"]))
    horizon = selected["horizon"]
    chosen = fitted[horizon]
    diagnostic = _simulation(frame, chosen["prediction"], val_end, len(frame), horizon, cost+selected["margin"], execution)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "training_source.csv", index=False)
    bundle = {key: value for key, value in chosen.items() if key not in {"prediction", "probability_validation", "last_train_label_endpoint"}}
    bundle.update(feature_columns=list(features.columns), horizon=horizon)
    joblib.dump(bundle, directory / "bundle.joblib")
    metadata = dict(status="ok", symbol=symbol, quote_currency=quote, interval_seconds=interval,
        rows=len(frame), train_end=train_end, val_end=val_end, selected=selected,
        horizon_minutes=horizon*5, fee_bps_per_side=fee, slippage_bps_per_side=slip,
        roundtrip_cost_fraction=cost, execution_config=execution,
        frozen_at=(times[val_end-1] + pd.Timedelta(seconds=interval)).isoformat(),
        trained_at=datetime.now(timezone.utc).isoformat(), selection_trace=trace,
        probability_validation=chosen["probability_validation"],
        residual_quantile_levels=[.1,.5,.9], residual_quantiles=chosen["residual_quantiles"],
        last_train_label_endpoint=chosen["last_train_label_endpoint"],
        calibration_sample_warning=selected["validation_trades"] < 5,
        test_diagnostic=dict(used_for_selection=False, total_return=float(diagnostic["total_return"]),
                             trades=int(diagnostic["roundtrip_count"])),
        artifacts={"bundle.joblib": _sha(directory / "bundle.joblib"), "training_source.csv": _sha(directory / "training_source.csv")},
        source_sha256={Path(path).name:_sha(path) for path in (__file__, Path(__file__).with_name("features.py"))},
        validity=["Public fee scenario; actual private account fee unverified; zero extra slippage is an explicit unknown-cost baseline.",
                  "Final15% is exposed retrospective diagnostic, not a prospective profit estimate.",
                  "Fixed three-horizon, three-margin selection uses VAL only; no minimum trade quota or forced entry.",
                  "Residual quantiles and sigmoid calibration use VAL; independent coverage remains unverified."])
    _save(directory / "metadata.json", metadata)
    return metadata


class ReturnModelBank:
    def __init__(self, root="runs/paper-models-20261021"):
        self.root, self._loaded, self._cache = Path(root), {}, {}

    def _load(self, symbol):
        if symbol not in self._loaded:
            directory = self.root / symbol
            metadata_path = directory / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["status"] != "ok" or metadata["symbol"] != symbol:
                raise ValueError("model identity/status mismatch")
            for filename, expected in metadata["source_sha256"].items():
                if _sha(Path(__file__).with_name(filename)) != expected:
                    raise ValueError(f"model inference source hash mismatch: {filename}")
            for filename, expected in metadata["artifacts"].items():
                if _sha(directory / filename) != expected:
                    raise ValueError(f"model artifact hash mismatch: {filename}")
            bundle = joblib.load(directory / "bundle.joblib")
            self._loaded[symbol] = metadata, bundle, _sha(metadata_path)
        return self._loaded[symbol]

    def predict(self, frame, symbol):
        result = dict(status="unavailable", model_ready=False, models={}, errors={}, cache_hit=False)
        try:
            symbol = canonical_symbol(symbol).replace("/", "")
            metadata, bundle, metadata_hash = self._load(symbol)
            bars, times, gaps, interval = _validated_frame(frame)
            if interval != metadata["interval_seconds"] or len(frame) < 49 or gaps[-48:].any():
                raise ValueError("M5 cadence and latest49 contiguous completed candles required")
            if times[-1] + pd.Timedelta(seconds=interval) <= pd.Timestamp(metadata["frozen_at"]):
                raise ValueError("inference before TRAIN/VAL freeze")
            digest = hashlib.sha256(metadata_hash.encode())
            digest.update(bars.to_numpy(dtype=np.float64).tobytes())
            digest.update(times.as_unit("ns").asi8.tobytes())
            key = digest.hexdigest()
            if symbol in self._cache and self._cache[symbol][0] == key:
                result = deepcopy(self._cache[symbol][1])
                result["cache_hit"] = True
                return result
            features = make_features(frame)
            if list(features.columns) != bundle["feature_columns"]:
                raise ValueError("paper feature schema differs from checkpoint")
            X = features.to_numpy(dtype=np.float32)[-1:]
            with threadpool_limits(limits=2):
                gross = float(bundle["regressor"].predict(X)[0])
                probability = float(_probability(bundle["classifier"], bundle["calibrator"], X)[0]) if bundle["classifier"] is not None else None
            if not np.isfinite(gross) or probability is not None and not 0 <= probability <= 1:
                raise ValueError("nonfinite model forecast")
            quantiles = {name: float(gross + value) for name, value in zip(("q10","q50","q90"), bundle["residual_quantiles"])}
            artifact = dict(path=str(self.root / symbol / "bundle.joblib"), sha256=metadata["artifacts"]["bundle.joblib"], metadata_sha256=metadata_hash)
            result.update(status="ok", model_ready=True, models={"hgb":dict(gross_return=gross,
                net_return=gross-metadata["roundtrip_cost_fraction"], horizon_bars=bundle["horizon"], artifact=artifact)},
                required_models=["hgb"], aggregate_gross_return=gross, aggregate_net_return=gross-metadata["roundtrip_cost_fraction"],
                probability_gross_exceeds_cost=probability, probability_event_cost_fraction=metadata["roundtrip_cost_fraction"],
                probability_validation=metadata["probability_validation"], gross_return_quantiles=quantiles,
                horizon_bars=bundle["horizon"], horizon_minutes=bundle["horizon"]*interval/60,
                entry_margin=metadata["selected"]["margin"], fee_bps_per_side=metadata["fee_bps_per_side"],
                model_identity={"hgb":artifact["sha256"]}, provenance=metadata,
                research_candidate=dict(test_exposed=True, explicitly_configured=True, model="short-horizon HGB regression/classifier",
                    selection="TRAIN70/VAL15/final15 diagnostic; no TEST selection", artifact_sha256=artifact["sha256"]))
            self._cache[symbol] = key, deepcopy(result)
        except Exception as exc:
            result.update(status="unavailable", model_ready=False)
            result["errors"]["paper_models"] = f"{type(exc).__name__}: {exc}"
        return result


class PaperScanner:
    def __init__(self, bank=None, config=None):
        self.bank, self.config = bank or ReturnModelBank(), dict(config or {})

    def evaluate(self, frame, symbol, asof, book, config):
        config = {**self.config, **config}
        canonical = canonical_symbol(symbol)
        quote = canonical.split("/")[1]
        fee_map = config.get("fee_bps", official_public_rates())
        fee = float(fee_map[quote] if isinstance(fee_map, dict) else fee_map)
        slip = float(config.get("slippage_bps", 0))
        result = fast_market.evaluate(frame, symbol, asof, book, {**config, "fee_bps":fee, "slippage_bps":slip})
        eligible = result["status"] == "setup" or (result["status"] == "no_setup" and result["reason"] == "latest completed candle has no eligible setup")
        if not eligible:
            return result
        interval = result["clock"]["interval_seconds"]
        times = pd.to_datetime(frame.timestamp, utc=True)
        complete = frame.loc[times + pd.Timedelta(seconds=interval) <= pd.Timestamp(asof)].copy()
        complete.attrs["interval_seconds"] = interval
        prediction = self.bank.predict(complete, canonical.replace("/", ""))
        result.update(model_forecasts=prediction, setup=None)
        if not prediction["model_ready"]:
            result.update(status="unavailable", reason="trained short-horizon paper model unavailable")
            return result
        cost, margin = 2*(fee+slip)/10000, prediction["entry_margin"]
        gross, horizon = prediction["aggregate_gross_return"], prediction["horizon_bars"]
        net = gross-cost
        summary = result["summary"]
        summary.update(hgb_gross_return=gross, model_net_return=net, model_median_net_return=net,
            model_positive_votes=int(net>0), model_disagreement=0., model_horizon_bars=horizon,
            model_horizon_minutes=prediction["horizon_minutes"], model_entry_margin=margin,
            model_margin_after_cost=net-margin, model_gross_return_threshold=cost+margin,
            model_current_roundtrip_cost=cost, probability_event_cost_fraction=prediction["probability_event_cost_fraction"])
        for name, value in prediction["gross_return_quantiles"].items():
            summary["model_gross_"+name] = value
        if prediction["probability_gross_exceeds_cost"] is not None:
            summary["probability_gross_exceeds_training_cost"] = prediction["probability_gross_exceeds_cost"]
        prediction["aggregate_net_return"] = net
        prediction["models"]["hgb"]["net_return"] = net
        if net <= margin:
            result.update(status="no_setup", reason="trained return forecast does not clear current costs and frozen VAL margin")
            return result
        bars = complete.tail(48)
        positive_volume = bars.volume.iloc[-12:]
        positive_volume = positive_volume[positive_volume>0]
        if not len(positive_volume):
            result.update(status="no_setup", reason="positive observed confirmation volume required")
            return result
        close = float(bars.close.iloc[-1])
        stop = max(.003, min(.02, 2*summary["ATR_fraction"]))
        target = min(.08, max(cost+margin, gross, .65*summary["range_48"]))
        candle = result["clock"]["candle_close"]
        result.update(status="setup", reason="genuine short-horizon HGB forecast exceeds verified costs and frozen validation margin",
            fingerprint=hashlib.sha256(json.dumps([canonical,candle,prediction["model_identity"]],sort_keys=True).encode()).hexdigest(),
            setup=dict(template="paper_hgb_forecast",family="paper_hgb_forecast",symbol=canonical,
                reference_price=close,current_close=close,current_volume=float(bars.volume.iloc[-1]),
                stop_fraction=stop,target_fraction=target,roundtrip_cost_fraction=cost,
                potential_net_reward_fraction=target-cost,reward_risk_after_cost=(target-cost)/(stop+cost),
                confirmation_price=float(bars.high.iloc[-4:].max())*1.0005,
                reclaim_price=max(float(bars.close.iloc[-12:].mean())*1.0005,close*1.001),
                arm_price=close*1.001,minimum_volume=float(positive_volume.median()),
                timestamp=candle,expires_at=(pd.Timestamp(candle)+pd.Timedelta(seconds=horizon*interval)).isoformat(),
                max_hold_bars=horizon,horizon_bars=horizon,horizon_minutes=prediction["horizon_minutes"],
                research_candidate=prediction["research_candidate"],
                geometry_caveat="VAL-calibrated residual/probability estimates are not independently validated probabilities or promised fills"))
        return result
