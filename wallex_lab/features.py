"""Causal crypto features adapted from Trade-Bot/scripts/prepare_gold.py.

Reuses its audit_minutes validation and prepare trailing-return/range ideas, with
UTC candle opens, fractional returns and volume. Gold sessions, units, scaler
state and models are deliberately absent. See the source reuse manifest.
"""
import numpy as np
import pandas as pd


def _validated_frame(frame):
    required = ["open", "high", "low", "close", "volume"]
    if not len(frame) or "timestamp" not in frame or any(k not in frame for k in required):
        raise ValueError("nonempty timestamp/OHLCV frame required")
    if pd.api.types.is_numeric_dtype(frame["timestamp"]):
        raise ValueError("timestamp must contain timezone-aware datetimes")
    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], errors="raise"))
        if timestamps.tz is None:
            raise ValueError("timestamps must be timezone-aware")
        timestamps = timestamps.tz_convert("UTC")
        data = frame[required].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid timestamp or OHLCV values") from exc
    if timestamps.hasnans or not timestamps.is_monotonic_increasing or not timestamps.is_unique:
        raise ValueError("timestamps must be valid, sorted and unique")
    values = data.to_numpy()
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("prices must be positive finite and volume nonnegative finite")
    if (data.high < data[["open", "close", "low"]].max(axis=1)).any() or (data.low > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("OHLC range is inconsistent")
    delta = np.diff(timestamps.as_unit("ns").asi8) / 1e9
    # With no manifest cadence, the first observed interval fixes the clock causally.
    interval = float(frame.attrs.get("interval_seconds", delta[0] if len(delta) else 60))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("interval_seconds must be positive finite")
    gaps = np.r_[False, ~np.isclose(delta, interval, rtol=0, atol=1e-6)]
    return data, timestamps, gaps, interval


def make_features(frame):
    """Finite trailing features; warmup is zero-filled and history resets at gaps.

    Each row is known at that candle close. No fit, future fill or global scale
    is used. gap_before=1 marks the first bar after a missing/off-cadence interval.
    """
    data, _, gaps, _ = _validated_frame(frame)
    pieces = []
    for _, bars in data.groupby(np.cumsum(gaps), sort=False):
        close = bars.close
        ret = close.pct_change(fill_method=None)
        values = {f"return_{lag}": close.pct_change(lag, fill_method=None) for lag in (1, 3, 6, 12)}
        for width in (6, 12, 24):
            values[f"volatility_{width}"] = ret.rolling(width, min_periods=2).std(ddof=0)
            values[f"ema_gap_{width}"] = close / close.ewm(span=width, adjust=False).mean() - 1
        spread = bars.high - bars.low
        values.update(range=spread / close, body=(close - bars.open) / bars.open,
                      close_location=((close - bars.low) / spread.where(spread.ne(0))).fillna(.5),
                      volume_change=np.log1p(bars.volume).diff(),
                      volume_ratio_12=bars.volume / bars.volume.rolling(12, min_periods=1).mean().replace(0, np.nan) - 1)
        pieces.append(pd.DataFrame(values, index=bars.index))
    features = pd.concat(pieces).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["gap_before"] = gaps.astype(float)
    return features


def forward_returns(frame, horizon=3):
    """close[t+h]/close[t]-1; immature or gap-crossing labels are NaN.

    Callers must additionally purge horizon rows at every train/validation/test
    boundary. A finite label is an observed future outcome, never a fill price.
    """
    data, _, gaps, _ = _validated_frame(frame)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    groups = pd.Series(np.cumsum(gaps), index=data.index)
    labels = data.close.shift(-horizon) / data.close - 1
    return labels.where(groups.eq(groups.shift(-horizon))).rename("forward_return")


def split_indices(n):
    """Chronological 60/20/20 boundaries; labels need caller-side purging."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return int(n * .6), int(n * .8)
