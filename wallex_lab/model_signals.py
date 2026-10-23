"""Fresh forecasts from pinned fitted artifacts. No fitting or saved-score replay."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .features import _validated_frame, make_features


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ForecastBank:
    def __init__(self, root=Path("runs/wallex-20261021-causal"), device="cpu", kronos_root=None,
                 manifest_path=None):
        self.root = Path(root).resolve()
        self.kronos_root = Path(kronos_root).resolve() if kronos_root is not None else self.root
        local_manifest = self.root / "model_manifest.json"
        self.manifest_path = Path(manifest_path) if manifest_path is not None else (
            local_manifest if local_manifest.exists() else Path("runs/orchestrator-model-bank/model_manifest.json"))
        self.device = str(device)
        self._models, self._sources, self._encoders, self._cache = {}, {}, {}, {}

    def _source(self, root, symbol):
        key = (str(root), symbol)
        if key not in self._sources:
            identity_path = root / symbol / "identity.json"
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            source = Path(identity["data_path"])
            if not source.stem.startswith(symbol + "_") or _sha(source) != identity["data_sha256"]:
                raise ValueError("original model data hash/symbol identity mismatch")
            source_frame = pd.read_csv(source)
            source_frame["timestamp"] = pd.to_datetime(source_frame["timestamp"], utc=True)
            _, times, _, cadence = _validated_frame(source_frame)
            self._sources[key] = identity, times, cadence, _sha(identity_path)
        return self._sources[key]

    def _load(self, symbol, name):
        if (symbol, name) in self._models:
            return self._models[symbol, name]
        strategy = {"hgb": "trees", "gru": "neural", "kronos": "kronos"}[name]
        root = self.root if name != "kronos" else self.kronos_root
        if name == "kronos" and not (root / symbol / strategy / "result.json").exists() and symbol == "BTCTMN":
            root = self.root
        directory = root / symbol / strategy
        receipt_path = directory / "result.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        metadata = receipt["metadata"]
        if receipt.get("status") != "ok" or metadata.get("status", "ok") != "ok" or receipt.get("symbol") != symbol:
            raise ValueError("model result is failed or belongs to another symbol")
        identity, source_times, cadence, identity_hash = self._source(root, symbol)
        if receipt["data_sha256"] != identity["data_sha256"]:
            raise ValueError("result and original training data hashes disagree")
        for filename in (strategy + ".py", "features.py"):
            if receipt["source_sha256"].get(filename) != _sha(Path(__file__).with_name(filename)):
                raise ValueError(f"inference source hash mismatch: {filename}")
        if _sha(directory / "outputs.npz") != receipt["outputs_sha256"]:
            raise ValueError("saved output evidence hash mismatch")
        horizon = int(metadata.get("horizon", metadata.get("label_horizon", identity["config"].get("horizon", 3))))
        if horizon != 3:
            raise ValueError("live bank requires the shared three-bar horizon")
        boundary = int(metadata["val_end"])
        if not 0 < boundary <= len(source_times):
            raise ValueError("invalid frozen model selection boundary")
        frozen_at = source_times[boundary - 1] + pd.Timedelta(seconds=cadence)
        artifact = directory / {"hgb": "trees_bundle.joblib", "gru": "gru.pt", "kronos": "head.npz"}[name]
        actual_hash = _sha(artifact)
        if name == "kronos":
            expected = metadata["head_sha256"]
        else:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            expected = manifest["files"][symbol][name]
        if actual_hash != expected:
            raise ValueError(f"pinned {name} checkpoint hash mismatch")
        if name == "hgb":
            import joblib
            bundle = joblib.load(artifact)
            model = bundle["models"]["hgb"]
            columns = bundle["feature_columns"]
            cost = float(metadata["round_trip_cost_fraction"])
            loaded = dict(model=model, columns=columns, cost=cost)
        elif name == "gru":
            import torch
            bundle = torch.load(artifact, map_location="cpu", weights_only=True)
            if bundle["name"] != "gru":
                raise ValueError("GRU checkpoint contains another architecture")
            cost = float(metadata["round_trip_hurdle_bps"]) / 10000
            loaded = dict(bundle=bundle, cost=cost)
        else:
            from . import kronos
            config = {"kronos_model": metadata.get("provenance", {}).get("family", "small")}
            family = config["kronos_model"]
            if family not in self._encoders:
                model_dir, tokenizer_dir, provenance = kronos._verify_assets(config)
                model, tokenizer = kronos._load_encoder(model_dir, tokenizer_dir, self.device)
                self._encoders[family] = model, tokenizer, provenance
            with np.load(artifact, allow_pickle=False) as stored:
                head = {key: stored[key].copy() for key in stored.files}
            if not all(np.isfinite(values).all() for values in head.values()) or (head["scaler_scale"] <= 0).any():
                raise ValueError("Kronos head contains invalid fitted parameters")
            execution = receipt.get("execution_config", identity["config"])
            cost = 2 * (float(execution.get("fee_bps", 10)) + float(execution.get("slippage_bps", 5))) / 10000
            loaded = dict(head=head, family=family, context=int(metadata.get("context", 64)), cost=cost)
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("invalid stored model cost convention")
        loaded.update(horizon=horizon, cadence=cadence, frozen_at=frozen_at,
                      artifact={"path": str(artifact), "sha256": actual_hash, "result_sha256": _sha(receipt_path),
                                "identity_sha256": identity_hash, "data_sha256": identity["data_sha256"],
                                "frozen_at": frozen_at.isoformat(), "trained_completed_utc": receipt.get("completed_utc")})
        self._models[symbol, name] = loaded
        return loaded

    def predict(self, frame, symbol):
        result = {"status": "unavailable", "models": {}, "aggregate_net_return": None,
                  "agreement_positive": 0, "count": 0, "model_ready": False, "errors": {}, "cache_hit": False}
        try:
            if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]+", symbol):
                raise ValueError("canonical uppercase symbol required")
            bars, times, gaps, cadence = _validated_frame(frame)
            if "symbol" in frame and not frame["symbol"].eq(symbol).all():
                raise ValueError("input candle symbol differs from requested model")
            if len(frame) < 64 or gaps[-63:].any():
                raise ValueError("64 contiguous completed candles required after the latest gap")
            digest = hashlib.sha256(symbol.encode())
            digest.update(bars.to_numpy(dtype=np.float64).tobytes())
            digest.update(times.as_unit("ns").asi8.tobytes())
            digest.update(str(cadence).encode())
            frame_hash = digest.hexdigest()
        except (ValueError, TypeError, KeyError) as exc:
            result["errors"]["input"] = str(exc)
            return result
        loaded = {}
        for name in ("hgb", "gru", "kronos"):
            try:
                model = self._load(symbol, name)
                if cadence != model["cadence"]:
                    raise ValueError("input cadence differs from fitted model cadence")
                if times[-1] + pd.Timedelta(seconds=cadence) <= model["frozen_at"]:
                    raise ValueError("latest candle is before frozen selection boundary")
                loaded[name] = model
            except Exception as exc:
                result["errors"][name] = f"{type(exc).__name__}: {exc}"
        model_ids = {name: model["artifact"]["sha256"] for name, model in loaded.items()}
        key = (frame_hash, json.dumps(model_ids, sort_keys=True))
        if key in self._cache:
            cached = deepcopy(self._cache[key])
            cached["cache_hit"] = True
            cached["errors"] = result["errors"]
            return cached
        for name, model in loaded.items():
            try:
                if name == "hgb":
                    features = make_features(frame)
                    if list(features.columns) != model["columns"]:
                        raise ValueError("HGB feature schema mismatch")
                    net = float(model["model"].predict(features.to_numpy()[-1:])[0])
                    gross = net + model["cost"]
                elif name == "gru":
                    from . import neural
                    bundle = model["bundle"]
                    X, sequences, valid, columns = neural._model_inputs(frame, bundle["sequence_length"])
                    if columns != bundle["feature_columns"] or not valid[-1]:
                        raise ValueError("GRU latest sequence is gapped or feature schema differs")
                    gross = float(neural.predict_bundle(bundle, X[-1:], sequences[-1:],
                                                       device=self.device, batch_size=1)[0]) / 10000
                else:
                    from . import kronos
                    encoder, tokenizer, _ = self._encoders[model["family"]]
                    latent = kronos._encode_windows(frame, np.asarray([len(frame) - 1]), model["context"], 1,
                                                    self.device, encoder, tokenizer)
                    head = model["head"]
                    projected = ((latent - head["scaler_mean"]) / head["scaler_scale"] - head["pca_mean"]) @ head["pca_components"].T
                    gross = float((projected @ head["coefficient"] + head["intercept"]).item())
                if not np.isfinite(gross):
                    raise ValueError("nonfinite fresh model forecast")
                artifact = dict(model["artifact"])
                trained_at = artifact["trained_completed_utc"]
                artifact["retrospective_before_training_wallclock"] = (
                    bool(times[-1] + pd.Timedelta(seconds=cadence) < pd.Timestamp(trained_at)) if trained_at else None)
                result["models"][name] = {"gross_return": gross, "net_return": gross - model["cost"],
                                            "horizon_bars": model["horizon"], "artifact": artifact}
            except Exception as exc:
                result["errors"][name] = f"{type(exc).__name__}: {exc}"
        values = [model["net_return"] for model in result["models"].values()]
        result.update(count=len(values), agreement_positive=sum(value > 0 for value in values),
                      aggregate_net_return=float(np.mean(values)) if values else None,
                      model_ready=len(values) == 3, status="ok" if len(values) == 3 else "unavailable",
                      completed_candle=times[-1].isoformat(), frame_sha256=frame_hash,
                      caveat="Fresh fitted-model inference; forecast magnitude is not a calibrated probability or execution guarantee.")
        # ponytail: retain one input forecast per symbol; add an LRU only if concurrent replay needs older prefixes.
        self._cache = {cache_key: value for cache_key, value in self._cache.items() if value.get("symbol") != symbol}
        result["symbol"] = symbol
        self._cache[key] = deepcopy(result)
        return result
