"""Bounded, causal numeric setup scan. No network, model calls or orders."""
import hashlib
import json
import math

import numpy as np
import pandas as pd

from . import opportunities
from .context import positive, timestamp
from .features import _validated_frame


def evaluate(frame, symbol: str, asof: str, book: dict | None, config: dict) -> dict:
    """Inspect the latest completed candle and a receipt-timed normalized quote."""
    result = {"status": "unavailable", "reason": "", "summary": {}, "setup": None,
              "fingerprint": "", "clock": {"symbol": symbol, "asof": asof}, "diagnostics": {}}
    family = "none"
    closed_at = None

    def finish(status, reason):
        if not all(math.isfinite(value) for value in result["summary"].values()):
            result["summary"], result["setup"] = {}, None
            status, reason = "unavailable", "invalid market input: nonfinite derived numeric summary"
        result["status"], result["reason"] = status, reason
        identity = [str(symbol).strip().upper(), closed_at.isoformat() if closed_at is not None else None, family]
        result["fingerprint"] = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        return result

    try:
        cutoff = pd.Timestamp(timestamp(asof))
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be nonempty")
        limits = {name: float(config.get(name, default)) for name, default in
                  (("max_candle_age_seconds", 1800), ("max_quote_age_seconds", 30), ("max_spread_bps", 100))}
        if not all(math.isfinite(v) and v > 0 for v in limits.values()):
            raise ValueError("freshness and spread limits must be positive finite")
        times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], errors="raise"))
        if times.tz is None or times.hasnans or not times.is_monotonic_increasing or not times.is_unique:
            raise ValueError("candle opens must be sorted unique aware timestamps")
        times = times.tz_convert("UTC")
        observable = times[times <= cutoff][-101:]
        if len(observable) < 2:
            return finish("unavailable", "insufficient candle history")
        interval = float(config.get("interval_seconds", frame.attrs.get("interval_seconds",
                         np.median(np.diff(observable.as_unit("ns").asi8)) / 1e9)))
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("positive finite candle interval required")
        # Drop unseen/unfinished bars before inspecting any OHLCV values.
        complete = times + pd.Timedelta(seconds=interval) <= cutoff
        window = frame.loc[complete].tail(100).copy().reset_index(drop=True)
        window.attrs["interval_seconds"] = interval
        if len(window) < 49:
            return finish("unavailable", "49 completed candles required")
        bars, opens_at, gaps, _ = _validated_frame(window)
        closed_at = opens_at[-1] + pd.Timedelta(seconds=interval)
        candle_age = float((cutoff - closed_at).total_seconds())
        result["clock"] = {"symbol": symbol.strip().upper(), "asof": cutoff.isoformat(),
                           "candle_open": opens_at[-1].isoformat(), "candle_close": closed_at.isoformat(),
                           "interval_seconds": interval}
        if candle_age > limits["max_candle_age_seconds"]:
            return finish("unavailable", "latest completed candle is stale")
        if gaps[-48:].any():
            return finish("unavailable", "latest 49 candles cross a gap; history must rebuild")
        close, high, low, volume = (bars[name].to_numpy() for name in ("close", "high", "low", "volume"))
        c = float(close[-1])
        returns = np.diff(np.log(close[-49:]))
        tr = np.maximum(high[-48:] - low[-48:], np.maximum(np.abs(high[-48:] - close[-49:-1]),
                                                        np.abs(low[-48:] - close[-49:-1])))
        change = np.diff(close[-15:])
        gain, loss = float(np.maximum(change, 0).mean()), float(np.maximum(-change, 0).mean())
        rsi = 100 * gain / (gain + loss) if gain + loss else 50.
        spread_range = float(high[-1] - low[-1])
        summary = {f"return_{lag}": float(c / close[-lag - 1] - 1) for lag in (1, 3, 12, 48)}
        summary.update(volatility_48=float(np.std(returns)),
                       volume_ratio_12=float(np.mean(volume[-12:]) / max(np.mean(volume[-48:]), 1e-12)),
                       close_location=float((c - low[-1]) / spread_range) if spread_range else .5,
                       range_48=float((max(high[-48:]) - min(low[-48:])) / c),
                       ATR_fraction=float(np.mean(tr[-14:]) / c),
                       trend_strength=float(abs(np.mean(returns)) / max(np.std(returns), 1e-12)),
                       RSI14=float(rsi), current_close=c, current_volume=float(volume[-1]),
                       candle_age_seconds=candle_age)
        result["summary"] = summary
        hold = min(24, math.floor(360 * 60 / interval))
        if hold < 1:
            return finish("unavailable", "candle interval exceeds six-hour holding limit")
        scanned = opportunities.scan_with_diagnostics(window, {**config, "scan_stride_bars": 1, "max_hold_bars": hold})
        result["diagnostics"] = scanned["diagnostics"]
        candidate = next((item for item in scanned["candidates"] if item["index"] == len(window) - 1), None)
        family = candidate["template"] if candidate else "none"
        if not isinstance(book, dict):
            return finish("unavailable", "fresh real order book required")
        quote_at = pd.Timestamp(timestamp(book["available_at"]))
        quote_age = float((cutoff - quote_at).total_seconds())
        if quote_age < 0 or quote_age > limits["max_quote_age_seconds"]:
            return finish("unavailable", "order book is future-dated or stale")
        bid, ask = positive(book["bid"]), positive(book["ask"])
        if bid >= ask:
            raise ValueError("order book must have bid below ask")
        spread = (ask - bid) / ((ask + bid) / 2) * 10000
        summary.update(best_bid=bid, best_ask=ask, spread_bps=spread, quote_age_seconds=quote_age)
        quantities = {}
        for side, price in (("bid", bid), ("ask", ask)):
            if side + "_quantity" in book:
                quantities[side] = positive(book[side + "_quantity"], allow_zero=True)
                summary[side + "_top_depth_notional"] = price * quantities[side]
        if "quantity_imbalance" in book:
            imbalance = float(book["quantity_imbalance"])
            if not math.isfinite(imbalance) or not -1 <= imbalance <= 1:
                raise ValueError("order-book imbalance must lie in [-1,1]")
            summary["quantity_imbalance"] = imbalance
        elif len(quantities) == 2 and sum(quantities.values()) > 0:
            summary["quantity_imbalance"] = (quantities["bid"] - quantities["ask"]) / sum(quantities.values())
        if spread > limits["max_spread_bps"]:
            return finish("no_setup", "order-book spread exceeds configured cap")
        if candidate is None:
            return finish("no_setup", "latest completed candle has no eligible setup")
        positive_volume = volume[-12:][volume[-12:] > 0]
        if not len(positive_volume):
            return finish("no_setup", "positive trailing median volume required")
        minimum_volume = float(np.median(positive_volume))
        expires = closed_at + pd.Timedelta(seconds=hold * interval)
        if cutoff >= expires:
            return finish("no_setup", "latest setup holding window has expired")
        confirmation = float(max(high[-4:]) * 1.0005)
        reclaim = float(max(np.mean(close[-12:]) * 1.0005, c * 1.001))
        result["setup"] = {**candidate, "family": family, "symbol": symbol.strip().upper(),
                           "reference_price": c, "current_close": c, "current_volume": float(volume[-1]),
                           "confirmation_price": confirmation, "reclaim_price": reclaim,
                           "trigger_price": confirmation if family == "trend_breakout" else reclaim,
                           "arm_price": c * 1.001, "minimum_volume": minimum_volume,
                           "timestamp": closed_at.isoformat(), "expires_at": expires.isoformat()}
        return finish("setup", "latest completed candle passes numeric and quote gates")
    except (KeyError, TypeError, ValueError, OverflowError, IndexError) as exc:
        result["setup"] = None
        return finish("unavailable", f"invalid market input: {exc}")
