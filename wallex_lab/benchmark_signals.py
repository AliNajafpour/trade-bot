"""Explicit research-candidate inference from frozen horizon benchmark fits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import _validated_frame
from .model_signals import _sha


class BenchmarkBank:
    def __init__(self, case_dir, combo, device="cpu", margin=None, fee_profile="tmn_level1_taker"):
        if combo not in {"hgb", "gru", "mean_hgb_gru"}:
            raise ValueError("explicit benchmark_combo must be hgb, gru or mean_hgb_gru; no automatic winner selection")
        self.case = Path(case_dir).resolve()
        self.combo, self.device = combo, str(device)
        self.required = ("hgb", "gru") if combo == "mean_hgb_gru" else (combo,)
        self.margin, self.fee_profile = margin, fee_profile
        self._loaded, self._cache = None, None

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        symbol = self.case.parent.parent.name
        directory = self.case / "models"
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "ok":
            raise ValueError("frozen horizon models are not complete")
        for filename, expected in metadata["source_sha256"].items():
            if _sha(Path(__file__).with_name(filename)) != expected:
                raise ValueError(f"fitted inference source hash mismatch: {filename}")
        data_path = self.case.parent / "candles.csv"
        data_receipt = json.loads((self.case.parent / "data.json").read_text(encoding="utf-8"))
        data_hash = _sha(data_path)
        if data_hash != data_receipt["data_sha256"]:
            raise ValueError("original benchmark candle evidence hash mismatch")
        source = pd.read_csv(data_path)
        source.timestamp = pd.to_datetime(source.timestamp, utc=True)
        _, source_times, _, interval = _validated_frame(source)
        if self.case.parent.name != f"{int(interval // 60)}m":
            raise ValueError("benchmark directory cadence disagrees with source data")
        horizon = int(metadata["horizon"])
        if self.case.name != f"h{int(horizon * interval / 60)}" or not 0 < horizon * interval <= 21600:
            raise ValueError("configured research horizon must match the case and be at most six hours")
        end = int(metadata["val_end"])
        if not 0 < int(metadata["train_end"]) < end < len(source):
            raise ValueError("invalid fitted chronological boundaries")
        freeze = source_times[end - 1] + pd.Timedelta(seconds=interval)
        artifacts = {}
        for filename in ("forecasts.npz", *({"hgb": "hgb.joblib", "gru": "gru.pt"}[name] for name in self.required)):
            path = directory / filename
            if path.resolve() != Path(metadata["artifacts"][filename]).resolve():
                raise ValueError(f"artifact manifest path differs: {filename}")
            actual = _sha(path)
            if actual != metadata["artifact_sha256"][filename]:
                raise ValueError(f"frozen benchmark artifact hash mismatch: {filename}")
            artifacts[filename] = {"path": str(path), "sha256": actual}
        margin = self.margin
        margin_source = "explicit benchmark_margin config"
        if margin is None:
            # Read only the frozen VAL-selected threshold, never rank TEST outcomes.
            plan_path = self.case / self.fee_profile / self.combo / "test.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if plan["data_sha256"] != data_hash or plan["model_metadata"]["input_key"] != metadata["input_key"]:
                raise ValueError("frozen threshold is not bound to these data/model artifacts")
            execution = plan["execution_config"]
            old_cost = 2 * (float(execution["fee_bps"]) + float(execution.get("slippage_bps", 0))) / 10000
            margin = float(plan["row"]["threshold"]) - old_cost
            margin_source = str(plan_path) + ": frozen VALIDATION-selected margin, TEST metrics ignored"
        margin = float(margin)
        if not np.isfinite(margin) or margin < -1e-12:
            raise ValueError("research entry margin must be finite and nonnegative")
        loaded = {}
        if "hgb" in self.required:
            import joblib
            loaded["hgb"] = joblib.load(directory / "hgb.joblib")
            if int(loaded["hgb"]["horizon"]) != horizon:
                raise ValueError("HGB horizon differs from frozen metadata")
        if "gru" in self.required:
            import torch
            loaded["gru"] = torch.load(directory / "gru.pt", map_location="cpu", weights_only=True)
            if loaded["gru"]["name"] != "gru" or int(loaded["gru"]["horizon"]) != horizon:
                raise ValueError("GRU architecture/horizon differs from frozen metadata")
        provenance = {"case_dir": str(self.case), "combo": self.combo, "explicitly_configured": True,
                      "test_exposed": True, "selection_caveat": "Exploratory candidate configured after exposed benchmark results; no prospective validation or automatic test-winner selection.",
                      "metadata_sha256": _sha(metadata_path), "data_sha256": data_hash,
                      "input_key": metadata["input_key"], "artifact_manifest": artifacts,
                      "entry_margin": max(0., margin), "margin_source": margin_source,
                      "train_end": int(metadata["train_end"]), "val_end": end,
                      "frozen_at": freeze.isoformat()}
        self._loaded = dict(models=loaded, metadata=metadata, provenance=provenance, symbol=symbol,
                            horizon=horizon, interval=interval, freeze=freeze)
        return self._loaded

    def predict(self, frame, symbol):
        result = dict(status="unavailable", models={}, model_ready=False, errors={}, cache_hit=False,
                      aggregate_gross_return=None, aggregate_net_return=None, agreement_positive=0, count=0)
        try:
            loaded = self._load()
            bars, times, gaps, interval = _validated_frame(frame)
            if symbol != loaded["symbol"] or ("symbol" in frame and not frame.symbol.eq(symbol).all()):
                raise ValueError("configured benchmark symbol does not match input")
            if interval != loaded["interval"]:
                raise ValueError("input candle cadence differs from fitted benchmark")
            if times[-1] + pd.Timedelta(seconds=interval) <= loaded["freeze"]:
                raise ValueError("inference precedes frozen TRAIN/VALIDATION cutoff")
            length = loaded["models"].get("gru", {}).get("sequence_length", 16)
            if len(frame) < max(49, length) or gaps[-(max(49, length) - 1):].any():
                raise ValueError("contiguous completed history must rebuild after the latest gap")
            digest = hashlib.sha256(json.dumps(loaded["provenance"], sort_keys=True).encode())
            digest.update(bars.to_numpy(dtype=np.float64).tobytes())
            digest.update(times.as_unit("ns").asi8.tobytes())
            key = digest.hexdigest()
            if self._cache is not None and self._cache[0] == key:
                result = deepcopy(self._cache[1])
                result["cache_hit"] = True
                return result
            from . import neural
            X, sequences, valid, columns = neural._model_inputs(frame, length)
            if not valid[-1]:
                raise ValueError("latest model sequence crosses a gap")
            for name, bundle in loaded["models"].items():
                if columns != bundle["feature_columns"]:
                    raise ValueError(f"{name} feature schema differs from checkpoint")
                if name == "hgb":
                    gross = float(bundle["model"].predict(X[-1:])[0])
                else:
                    gross = float(neural.predict_bundle(bundle, X[-1:], sequences[-1:], device=self.device, batch_size=1)[0]) / 10000
                if not np.isfinite(gross):
                    raise ValueError("fresh fitted model returned a nonfinite gross return")
                filename = "hgb.joblib" if name == "hgb" else "gru.pt"
                result["models"][name] = dict(gross_return=gross, horizon_bars=loaded["horizon"],
                                              artifact=loaded["provenance"]["artifact_manifest"][filename])
            gross = float(np.mean([result["models"][name]["gross_return"] for name in self.required]))
            result.update(status="ok", model_ready=True, aggregate_gross_return=gross,
                          count=len(self.required), agreement_positive=sum(model["gross_return"] > 0 for model in result["models"].values()),
                          combo=self.combo, required_models=list(self.required), horizon_bars=loaded["horizon"],
                          horizon_minutes=loaded["horizon"] * interval / 60, interval_seconds=interval,
                          entry_margin=loaded["provenance"]["entry_margin"], research_candidate=loaded["provenance"],
                          model_identity={name: model["artifact"]["sha256"] for name, model in result["models"].items()})
            self._cache = key, deepcopy(result)
        except Exception as exc:
            result.update(status="unavailable", model_ready=False)
            result["errors"]["benchmark"] = f"{type(exc).__name__}: {exc}"
        return result
