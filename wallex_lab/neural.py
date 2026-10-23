"""MLP/GRU comparison with a purged holdout inside TRAIN only.

Portable model/scaler/training subset adapted from read-only Trade-Bot:
scripts/neural_gold.py, SHA256
07e364aee5535d47a269727d74c2169dac671e513b5790131c3f9d9600419837.
Architecture subset is restricted to MLP/GRU; thread limit changed to two.
No original data, trained models or gold calendars are used.
"""
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import make_features, forward_returns

DESCRIPTIONS = {
    "mlp": "Two-hidden-layer MLP on engineered features",
    "gru": "One-layer GRU with final-history-state regression",
}
SOURCE = {
    "path": "D:/ALI/Programming-AI/Personal Projects and Tests/Trade-Bot/scripts/neural_gold.py",
    "sha256": "07e364aee5535d47a269727d74c2169dac671e513b5790131c3f9d9600419837",
    "reuse": "MLP/GRU, robust train scalers, deterministic fitting, checkpoint inference",
}


class _ForecastNet(nn.Module):
    def __init__(self, name, n_features, seq_shape, hidden):
        super().__init__()
        self.name = name
        if name == "mlp":
            self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        elif name == "gru":
            self.encoder = nn.GRU(seq_shape[1], hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
        else:
            raise ValueError("Only mlp and gru are supported")

    def forward(self, values):
        if self.name == "mlp":
            return self.encoder(values).squeeze(-1)
        encoded, _ = self.encoder(values)
        return self.head(encoded[:, -1]).squeeze(-1)


def _array(values, ndim, label):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != ndim or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite {ndim}-dimensional array")
    return values


def _fit_scaler(values):
    flat = values.reshape(-1, values.shape[-1])
    lower, upper = np.quantile(flat, (0.005, 0.995), axis=0)
    clipped = np.clip(flat, lower, upper)
    scale = clipped.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"lower": lower.tolist(), "upper": upper.tolist(),
            "mean": clipped.mean(axis=0).tolist(), "scale": scale.tolist()}


def _transform(values, scaler):
    clipped = np.clip(values, scaler["lower"], scaler["upper"])
    return np.ascontiguousarray(np.clip((clipped - scaler["mean"]) / scaler["scale"], -10, 10), dtype=np.float32)


def _predict(model, values, device, batch_size):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def _inputs(X, seq, label, n_features=None, seq_shape=None):
    X = _array(X, 2, label + " X")
    seq = _array(seq, 3, label + " seq")
    if len(X) != len(seq) or X.shape[1] < 1 or min(seq.shape[1:]) < 1:
        raise ValueError(f"{label} X and seq need matching rows and nonempty features/history")
    if n_features is not None and (X.shape[1] != n_features or tuple(seq.shape[1:]) != tuple(seq_shape)):
        raise ValueError(f"{label} feature/history shapes differ from training")
    return X, seq


def fit_predict(name, X_train, seq_train, y_train, X_val, seq_val, y_val,
                X_test, seq_test, seed=17, config=None):
    """Fit using train labels, stop on validation RMSE, predict test returns in bp.

    Config supports epochs <= 20, patience, batch_size <= 1024, hidden <= 128,
    learning_rate, weight_decay and device. Defaults: 12/3/512/32/1e-3/1e-4/auto.
    Caller must enforce label maturity and period exclusions before calling.
    """
    start = time.perf_counter()
    name = name.lower().replace("-", "_")
    if name not in DESCRIPTIONS:
        raise ValueError(f"Unknown neural candidate {name!r}")
    settings = dict(epochs=12, patience=3, batch_size=512, hidden=32,
                    learning_rate=1e-3, weight_decay=1e-4, device="auto")
    config = dict(config or {})
    if set(config) - set(settings):
        raise ValueError(f"Unknown neural settings: {sorted(set(config) - set(settings))}")
    settings.update(config)
    for key, ceiling in (("epochs", 20), ("patience", 20), ("batch_size", 1024), ("hidden", 128)):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
            raise ValueError(f"{key} must be an integer in [1, {ceiling}]")
    for key in ("learning_rate", "weight_decay"):
        if not np.isfinite(settings[key]) or settings[key] < 0 or (key == "learning_rate" and settings[key] == 0):
            raise ValueError(f"{key} must be finite and {'positive' if key == 'learning_rate' else 'nonnegative'}")
    X_train, seq_train = _inputs(X_train, seq_train, "training")
    n_features, seq_shape = X_train.shape[1], tuple(seq_train.shape[1:])
    X_val, seq_val = _inputs(X_val, seq_val, "validation", n_features, seq_shape)
    X_test, seq_test = _inputs(X_test, seq_test, "test", n_features, seq_shape)
    y_train, y_val = _array(y_train, 1, "y_train"), _array(y_val, 1, "y_val")
    if not len(X_train) or not len(X_val) or len(y_train) != len(X_train) or len(y_val) != len(X_val):
        raise ValueError("Training and validation need nonempty matching input/target rows")
    device = settings["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    scalers = {"X": _fit_scaler(X_train), "seq": _fit_scaler(seq_train),
               "y_mean": float(y_train.mean(dtype=np.float64)),
               "y_scale": max(float(y_train.std(dtype=np.float64)), 1e-8)}
    source = "X" if name == "mlp" else "seq"
    train = _transform(X_train if source == "X" else seq_train, scalers[source])
    validation = _transform(X_val if source == "X" else seq_val, scalers[source])
    target = torch.from_numpy(((y_train - scalers["y_mean"]) / scalers["y_scale"]).astype(np.float32))
    model = _ForecastNet(name, n_features, seq_shape, settings["hidden"]).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters >= 1_000_000:
        raise ValueError("Candidate exceeds the protocol's one-million-parameter bound")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    best_rmse, stale, history, best_state, best_epoch = math.inf, 0, [], None, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        indices = torch.randperm(len(train), generator=generator).numpy()
        loss_sum = 0.0
        for offset in range(0, len(train), settings["batch_size"]):
            rows = indices[offset:offset + settings["batch_size"]]
            batch = torch.from_numpy(train[rows]).to(device)
            batch_target = target[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch), batch_target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in {name}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(rows)
        val_prediction = _predict(model, validation, device, settings["batch_size"]) * scalers["y_scale"] + scalers["y_mean"]
        rmse = float(np.sqrt(np.mean((val_prediction.astype(np.float64) - y_val) ** 2)))
        if not math.isfinite(rmse):
            raise FloatingPointError(f"Nonfinite validation RMSE in {name}, epoch {epoch}")
        history.append({"epoch": epoch, "training_mse_scaled": loss_sum / len(train), "validation_rmse_bp": rmse})
        if rmse < best_rmse:
            best_rmse, best_epoch, stale = rmse, epoch, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    test = _transform(X_test if source == "X" else seq_test, scalers[source])
    predictions = _predict(model, test, device, settings["batch_size"]) * scalers["y_scale"] + scalers["y_mean"]
    val_predictions = _predict(model, validation, device, settings["batch_size"]) * scalers["y_scale"] + scalers["y_mean"]
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite test predictions")
    bundle = {"format_version": 1, "name": name, "n_features": n_features,
              "seq_shape": list(seq_shape), "hidden": settings["hidden"], "scalers": scalers,
              "state_dict": best_state, "seed": int(seed)}
    training = {"architecture": DESCRIPTIONS[name], "config": settings, "seed": int(seed),
                "actual_epochs": epoch, "best_epoch": best_epoch, "parameters": parameters,
                "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "runtime_seconds": time.perf_counter() - start, "best_validation_rmse_bp": best_rmse,
                "validation_criterion": "return RMSE in bp", "training_rows": len(train),
                "validation_rows": len(validation), "torch_version": str(torch.__version__),
                "numpy_version": np.__version__, "history": history}
    return {"prediction": predictions.astype(np.float32), "validation_prediction": val_predictions.astype(np.float32),
            "training": training, "bundle": bundle}


def predict_bundle(bundle, X, seq, device="auto", batch_size=512):
    """Predict bp from a trusted fitted bundle without refitting scalers or weights."""
    if bundle["format_version"] != 1:
        raise ValueError("Unsupported neural bundle format")
    X, seq = _inputs(X, seq, "prediction", bundle["n_features"], bundle["seq_shape"])
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 1024:
        raise ValueError("batch_size must be an integer in [1, 1024]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    model = _ForecastNet(bundle["name"], bundle["n_features"], bundle["seq_shape"], bundle["hidden"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    source = "X" if bundle["name"] == "mlp" else "seq"
    scaler = bundle["scalers"]
    values = _transform(X if source == "X" else seq, scaler[source])
    prediction = _predict(model, values, device, batch_size) * scaler["y_scale"] + scaler["y_mean"]
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite reloaded predictions")
    return prediction.astype(np.float32)


def _model_inputs(frame, length):
    features = make_features(frame)
    X = np.ascontiguousarray(features.to_numpy(), dtype=np.float32)
    sequences = np.zeros((len(X), length, X.shape[1]), dtype=np.float32)
    if len(X) >= length:
        sequences[length - 1:] = np.lib.stride_tricks.sliding_window_view(X, length, axis=0).transpose(0, 2, 1)
    rows = np.arange(len(X))
    gaps = features["gap_before"].to_numpy() if "gap_before" in features else np.zeros(len(X))
    cumulative = np.cumsum(gaps)
    # The gap before the first window bar is allowed; gaps inside it are not.
    valid = (rows >= length - 1) & (cumulative == cumulative[np.maximum(rows - length + 1, 0)])
    return X, sequences, valid, list(features.columns)


def predict(frame, bundle, device="auto", batch_size=512):
    """Causal return forecasts in basis points from one saved MLP/GRU bundle."""
    X, sequences, valid, columns = _model_inputs(frame, bundle["sequence_length"])
    if columns != bundle["feature_columns"]:
        raise ValueError("Feature schema differs from the fitted checkpoint")
    forecasts = predict_bundle(bundle, X, sequences, device=device, batch_size=batch_size)
    forecasts[~valid] = 0
    return forecasts


def generate(frame, train_end, val_end, config, run_dir):
    """Compare actual MLP/GRU fits without inspecting outer VAL or TEST labels.

    Internal early stopping and architecture selection share the last 20% of
    TRAIN. That selection score is optimistic; outer VAL/TEST remain untouched.
    All positions before train_end are zero because these are retrospective fits.
    """
    config = dict(config or {})
    settings = {**config, **config.get("neural", {})}
    n = len(frame)
    if not 0 < train_end < val_end < n:
        raise ValueError("Require 0 < train_end < val_end < number of candles")
    horizon = settings.get("horizon", 3)
    length = settings.get("sequence_length", 16)
    max_rows = settings.get("max_train_rows", 12000)
    for key, value in (("horizon", horizon), ("sequence_length", length), ("max_train_rows", max_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if length > 128:
        raise ValueError("sequence_length must be at most 128")
    exposure = float(config.get("max_exposure", 1))
    fee, slippage = float(config.get("fee_bps", 10)), float(config.get("slippage_bps", 5))
    if not np.isfinite([exposure, fee, slippage]).all() or not 0 <= exposure <= 1 or min(fee, slippage) < 0:
        raise ValueError("Require exposure in [0,1] and finite nonnegative costs")

    X, sequences, valid, columns = _model_inputs(frame, length)
    # Only this prefix is passed to label construction. Labels cannot touch VAL/TEST.
    labels = np.full(n, np.nan, dtype=np.float32)
    labels[:train_end] = forward_returns(frame.iloc[:train_end], horizon).to_numpy() * 10000
    inner_end = int(train_end * .8)
    rows = np.arange(n)
    fit_rows = rows[valid & np.isfinite(labels) & (rows < inner_end - horizon)][-max_rows:]
    stop_rows = rows[valid & np.isfinite(labels) & (rows >= inner_end) & (rows < train_end - horizon)]
    if len(fit_rows) < 24 or len(stop_rows) < 8:
        raise ValueError("Neural fitting needs at least 24 purged fit rows and 8 internal validation rows")
    model_config = dict(epochs=8, patience=3, batch_size=256, hidden=32,
                        learning_rate=1e-3, weight_decay=1e-4, device="auto")
    model_config.update({key: settings[key] for key in model_config if key in settings})
    seed = settings.get("seed", 17)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    candidate_metadata, forecasts = {}, {}
    for name in ("mlp", "gru"):
        fitted = fit_predict(name, X[fit_rows], sequences[fit_rows], labels[fit_rows],
                             X[stop_rows], sequences[stop_rows], labels[stop_rows],
                             X, sequences, seed=seed, config=model_config)
        bundle = fitted["bundle"]
        bundle.update(sequence_length=length, feature_columns=columns, source=SOURCE)
        torch.save(bundle, directory / f"{name}.pt")
        candidate_metadata[name] = fitted["training"]
        forecasts[name] = fitted["prediction"]
        forecasts[name][~valid] = 0
    chosen = min(candidate_metadata, key=lambda name: candidate_metadata[name]["best_validation_rmse_bp"])
    forecast = forecasts[chosen]
    hurdle = 2 * (fee + slippage)
    target = np.where((rows >= train_end) & valid & (forecast > hurdle), exposure, 0.0)
    metadata = {
        "status": "ok", "selected_model": chosen, "candidates": candidate_metadata,
        "selection": "minimum return RMSE on purged internal TRAIN holdout; outer VAL/TEST labels unused",
        "train_end": int(train_end), "val_end": int(val_end), "internal_validation_start": inner_end,
        "fit_rows": len(fit_rows), "fit_last_row": int(fit_rows[-1]),
        "internal_validation_rows": len(stop_rows), "internal_validation_last_row": int(stop_rows[-1]),
        "horizon": horizon, "sequence_length": length, "max_train_rows": max_rows,
        "round_trip_hurdle_bps": hurdle, "score_semantics": "expected horizon fractional close return",
        "source": SOURCE, "checkpoints": [f"{name}.pt" for name in candidate_metadata],
        "caveats": ["Small local comparators; no profitability claim or probability calibration",
                    "Internal holdout reused for early stopping and architecture selection",
                    "Scores before train_end are retrospective; allocations there are zero",
                    "Cost hurdle approximates two-sided costs; next-open fills remain approximate"],
    }
    (directory / "config.json").write_text(json.dumps({"model": model_config, "metadata": metadata}, indent=2), encoding="utf-8")
    np.savez_compressed(directory / "forecasts.npz", mlp_bp=forecasts["mlp"], gru_bp=forecasts["gru"], target=target)
    return {"target": target, "score": forecast.astype(float) / 10000,
            "forecast_bp": forecast, "metadata": metadata}
