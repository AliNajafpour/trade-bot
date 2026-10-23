"""Purged, cost-aware return models with chronological model/threshold selection.

TrainClip and the sklearn pipeline adapt Trade-Bot/scripts/run_gold_benchmark.py
without importing its paths, data panels, calibration or campaign runner.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import forward_returns, make_features


class TrainClip(TransformerMixin, BaseEstimator):
    """Learn clipping bounds only from the purged training block."""

    def fit(self, X, y=None):
        self.lower_, self.upper_ = np.quantile(X, (.005, .995), axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)


def generate(frame, train_end, val_end, config, run_dir):
    """Fit TRAIN once; choose model and long/cash threshold on purged VAL only.

    Scores predict horizon close-to-close fractional return minus round-trip
    fees and slippage. Execution remains the shared next-open spot simulation.
    The saved joblib bundle must only be loaded from a trusted local run.
    """
    from .engine import backtest

    horizon = config.get("horizon", 3)
    seed = config.get("seed", 17)
    iterations = config.get("tree_iterations", 150)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("tree_iterations must be a positive integer")
    n = len(frame)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (train_end, val_end)):
        raise ValueError("split boundaries must be integer positions")
    if not 0 < train_end < val_end < n:
        raise ValueError("require nonempty chronological TRAIN, VAL and TEST blocks")
    fee, slip, exposure = (float(config.get(k, default)) for k, default in
                           (("fee_bps", 10), ("slippage_bps", 5), ("max_exposure", .5)))
    if not np.isfinite([fee, slip, exposure]).all() or min(fee, slip) < 0 or not 0 <= exposure <= 1:
        raise ValueError("costs must be finite nonnegative and max_exposure in [0,1]")
    cost = 2 * (fee + slip) / 10000
    if cost >= 1:
        raise ValueError("round-trip costs must be below 100 percent")

    feature_frame = make_features(frame)
    X = feature_frame.to_numpy(dtype=float)
    if X.ndim != 2 or len(X) != n or not np.isfinite(X).all():
        raise ValueError("features must be finite and aligned to candles")
    gross_y = np.asarray(forward_returns(frame, horizon=horizon), dtype=float)
    indices = np.arange(n)
    train = (indices < train_end - horizon) & np.isfinite(gross_y)
    validation = (indices >= train_end) & (indices < val_end - horizon) & np.isfinite(gross_y)
    if train.sum() < 32 or validation.sum() < 8:
        raise ValueError("need at least 32 purged training labels and 8 purged validation labels")
    net_y = gross_y - cost
    fingerprint = hashlib.sha256(X[train].tobytes() + net_y[train].tobytes()).hexdigest()

    settings = {
        "ridge": {"alpha": 100.0, "random_state": int(seed)},
        "hgb": {"max_iter": int(iterations), "max_leaf_nodes": 15, "min_samples_leaf": 30,
                "learning_rate": .05, "early_stopping": False, "random_state": int(seed)},
    }
    estimators = {"ridge": Ridge(**settings["ridge"]),
                  "hgb": HistGradientBoostingRegressor(**settings["hgb"])}
    unavailable = []
    for package, name in (("lightgbm", "lgbm"), ("xgboost", "xgb")):
        if importlib.util.find_spec(package) is None:
            unavailable.append(package)
            continue
        settings[name] = {"n_estimators": int(iterations), "learning_rate": .03,
                          "n_jobs": 2, "random_state": int(seed)}
        if name == "lgbm":
            from lightgbm import LGBMRegressor
            settings[name].update(num_leaves=15, min_child_samples=30, verbosity=-1)
            estimators[name] = LGBMRegressor(**settings[name])
        else:
            from xgboost import XGBRegressor
            settings[name].update(max_depth=3, objective="reg:squarederror", tree_method="hist")
            estimators[name] = XGBRegressor(**settings[name])

    models, predictions, candidates = {}, {}, {}
    winner, best = None, None
    # Validation ends before its horizon purge, so fills/liquidation cannot use TEST.
    val_frame = frame.iloc[train_end:val_end - horizon].reset_index(drop=True)
    with threadpool_limits(limits=2):
        for name, estimator in estimators.items():
            pipeline = Pipeline([("clip", TrainClip()), ("scale", StandardScaler()), ("model", estimator)])
            pipeline.fit(X[train], net_y[train])
            prediction = np.asarray(pipeline.predict(X), dtype=float)
            if prediction.shape != (n,) or not np.isfinite(prediction).all():
                raise ValueError(f"{name} returned invalid predictions")
            models[name], predictions[name] = pipeline, prediction
            val_prediction = prediction[validation]
            error = val_prediction - net_y[validation]
            # This extra cutoff provides a genuine all-cash validation alternative.
            cash_cutoff = max(0.0, float(prediction[train_end:val_end - horizon].max())) + max(cost, 1e-12)
            thresholds = sorted(set([0.0, cost / 2, cost, cash_cutoff]))
            trials = []
            for threshold in thresholds:
                val_target = np.where(prediction[train_end:val_end - horizon] > threshold, exposure, 0.0)
                simulation = backtest(val_frame, val_target, config)
                metrics = simulation["metrics"]
                value = float(metrics["total_return"])
                if not np.isfinite(value):
                    raise ValueError("validation backtest returned nonfinite total_return")
                trial = {"threshold": float(threshold), "metrics": metrics}
                trials.append(trial)
                # Stable tie breaks prefer less exposure, then lower prediction error.
                ranking = (value, -float(val_target.mean()), -float(np.mean(error ** 2)))
                if best is None or ranking > best:
                    best, winner = ranking, (name, float(threshold))
            candidates[name] = {
                "config": settings[name], "validation_rmse_net_return": float(np.sqrt(np.mean(error ** 2))),
                "validation_mae_net_return": float(np.mean(np.abs(error))),
                "validation_direction_accuracy": float(np.mean((val_prediction > 0) == (net_y[validation] > 0))),
                "threshold_trials": trials,
            }

    name, threshold = winner
    score = predictions[name]
    target = np.where(score > threshold, exposure, 0.0)
    target[:train_end] = 0.0
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "ok", "strategy": "trees", "selected_model": name,
        "selected_threshold": threshold, "max_exposure": exposure,
        "seed": int(seed), "cpu_threads": 2, "horizon": int(horizon),
        "train_end": int(train_end), "val_end": int(val_end),
        "training_rows": int(train.sum()), "validation_rows": int(validation.sum()),
        "training_last_label_origin": int(indices[train][-1]),
        "validation_last_label_origin": int(indices[validation][-1]),
        "training_target_mean": float(net_y[train].mean()), "training_fingerprint": fingerprint,
        "target_definition": "forward close fractional return minus 2*(fee_bps+slippage_bps)/10000",
        "score_semantics": "predicted net fractional horizon return; long iff score > selected_threshold",
        "round_trip_cost_fraction": cost, "fee_bps": fee, "slippage_bps": slip,
        "preprocessing": "TRAIN-only 0.5/99.5 percentile clipping then StandardScaler",
        "feature_columns": list(feature_frame.columns), "candidates": candidates,
        "unavailable_optional_packages": unavailable,
        "selection": "maximize purged VAL next-open backtest total_return after costs; ties prefer less exposure then lower MSE",
        "refit": "none; all models and preprocessing stay fitted exclusively on purged TRAIN",
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]
                     + [p for p in ("lightgbm", "xgboost") if p not in unavailable]},
        "source_reuse": {"origin": "Trade-Bot/scripts/run_gold_benchmark.py:TrainClip,fit_predict",
                         "sha256": "d3ba14014f8d7de986bac0a16dddc902805f5fc37a050b4d1ade0333262a7533",
                         "adaptation": "train-only clipping/scaled estimator pipeline; local fractional targets and cost-aware chronological selection"},
        "validity_caveats": [
            "VAL metrics are selection diagnostics, not an independent performance estimate.",
            "TRAIN allocations are zero; no in-sample fitted scores are presented as executed training trades.",
            "A close-to-close forecast approximates the return available to a next-open execution policy.",
            "Costs are assumptions; bar data cannot establish historical order-book fills.",
            "Fixed booster iterations; no random internal validation or TEST-driven early stopping.",
            "joblib bundles are trusted local artifacts; never load an untrusted bundle.",
        ],
        "artifacts": {"bundle": str(output / "trees_bundle.joblib"),
                      "forecasts": str(output / "trees_forecasts.csv")},
    }
    # Validate metadata before persisting a partly unusable research run.
    encoded = json.dumps(metadata, indent=2, allow_nan=False)
    bundle = {"models": models, "selected_model": name, "selected_threshold": threshold,
              "feature_columns": list(feature_frame.columns), "metadata": metadata}
    joblib.dump(bundle, output / "trees_bundle.joblib")
    forecasts = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy(),
                              "split": np.where(indices < train_end, "TRAIN", np.where(indices < val_end, "VAL", "TEST")),
                              "prediction_net_return": score, "target": target})
    for model_name, prediction in predictions.items():
        forecasts[f"prediction_{model_name}_net_return"] = prediction
    forecasts.to_csv(output / "trees_forecasts.csv", index=False)
    (output / "trees_metadata.json").write_text(encoded, encoding="utf-8")
    return {"target": target, "score": score, "metadata": metadata}
