"""Receipt-backed completed OHLCV chart pixels for council vision requests."""
import hashlib
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import aggregate_complete
from .features import _validated_frame
from .llm import _json
from .spot import canonical_symbol


def render_chart(candles, symbol, asof, path, signals=None, history=None):
    """Render the latest 120 complete bars per timeframe; retain exact source inputs.

    Input timestamps are aware one-minute candle opens. Missing minutes remain gaps;
    a higher-timeframe candle is drawn only when all of its native minutes exist.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle

    symbol = canonical_symbol(symbol)
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is None or pd.isna(cutoff):
        raise ValueError("An aware asof timestamp is required")
    cutoff = cutoff.tz_convert("UTC")
    _, times, _, interval = _validated_frame(candles)
    if interval != 60 or (times.second != 0).any() or (times.microsecond != 0).any():
        raise ValueError("Charts require minute-aligned one-minute candles")
    path = Path(path).resolve()
    if path.suffix.lower() != ".png":
        raise ValueError("Chart output must be PNG")
    closed = times + pd.Timedelta(minutes=1) <= cutoff
    frame = candles.loc[closed, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    frame["timestamp"] = times[closed]
    if not len(frame):
        raise ValueError("No completed candles available for chart")
    quote = candles.attrs.get("price_unit") or symbol.split("/")[1]
    if not isinstance(quote, str) or not re.fullmatch(r"[A-Za-z0-9 _-]{1,24}", quote):
        raise ValueError("Set candles.attrs['price_unit'] to the quote currency")
    signals = [] if signals is None else signals
    if not isinstance(signals, list) or len(signals) > 8:
        raise ValueError("A list of at most eight signal overlays is required")
    for signal in signals:
        if not isinstance(signal, dict):
            raise ValueError("Signal overlays must be objects")
        for name in ("stop_price", "target_price", "reference_price"):
            if name in signal and (isinstance(signal[name], bool) or not isinstance(signal[name], (int, float)) or not math.isfinite(signal[name]) or signal[name] <= 0):
                raise ValueError("Overlay prices must be positive finite numbers")
    # Bounded chart history limits receipt size while leaving 120 completed 15m bars.
    frame = frame.tail(120 * 15 + 15)
    bars, counts = [], {}
    for minutes in (1, 5, 15):
        grouped = frame if minutes == 1 else aggregate_complete(frame, minutes * 60)[0]
        grouped = grouped.loc[grouped.timestamp + pd.Timedelta(minutes=minutes) <= cutoff].tail(120)
        counts[f"{minutes}m"] = len(grouped)
        bars.append((minutes, grouped))
    figure = Figure(figsize=(18, 10), dpi=120, layout="constrained", facecolor="#ffffff")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3, gridspec_kw={"height_ratios": [4, 1]}, sharex="col")
    for column, (minutes, data) in enumerate(bars):
        price, volume = axes[0, column], axes[1, column]
        price.set_title(f"{symbol} | {minutes}m | {len(data)} completed bars", fontsize=12)
        price.set_ylabel(f"Price ({quote})")
        volume.set_ylabel("Base volume")
        volume.set_xlabel("Candle open, UTC")
        volume.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=cutoff.tzinfo))
        for axis in (price, volume):
            axis.grid(alpha=.2)
            axis.tick_params(labelsize=9)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        if not len(data):
            price.text(.5, .5, "No complete buckets", transform=price.transAxes, ha="center")
            continue
        x = mdates.date2num(pd.DatetimeIndex(data.timestamp).to_pydatetime())
        volume.set_xticks(np.linspace(x[0], x[-1], min(4, len(x))))
        width = minutes / 1440 * .7
        colors = np.where(data.close >= data.open, "#087f5b", "#c92a2a")
        price.vlines(x, data.low, data.high, color=colors, linewidth=1)
        for clock, row, color in zip(x, data.itertuples(index=False), colors):
            height = abs(row.close - row.open)
            price.add_patch(Rectangle((clock - width / 2, min(row.open, row.close)), width,
                                      max(height, row.close * 1e-7), facecolor=color, edgecolor=color))
        volume.bar(x, data.volume, width=width, color=colors)
        for index, signal in enumerate(signals):
            for name, label, color in (("stop_price", "SL", "#c92a2a"), ("target_price", "TP", "#087f5b"), ("reference_price", "Reference", "#555555")):
                if name in signal:
                    value = signal[name]
                    price.axhline(value, color=color, linestyle="--", linewidth=.9,
                                  label=f"{index + 1} {label} {value:.10g}")
        if signals:
            price.legend(fontsize=8, loc="best")
        low, high = float(data.low.min()), float(data.high.max())
        padding = max((high-low)*.08, high*.0001)
        price.set_ylim(low-padding, high+padding)
        price.set_xlim(x[0] - width, x[-1] + width * 2)
    figure.suptitle(f"As of {cutoff.isoformat()} | completed OHLCV only | quote {quote}\n"
                   "Price view fits observed candles; off-panel TP/SL values remain in legend", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png")
    figure.clear()
    records = frame.copy()
    records["timestamp"] = records.timestamp.map(lambda value: value.isoformat())
    inputs = {"symbol": symbol, "asof": cutoff.isoformat(), "price_unit": quote,
              "source_interval_seconds": 60, "candles": records.to_dict("records"),
              "signals": signals, "history": history}
    input_path = path.with_suffix(".inputs.json")
    input_path.write_text(_json(inputs), encoding="utf-8")
    receipt = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "symbol": symbol, "asof": cutoff.isoformat(), "price_unit": quote,
               "timeframes": list(counts), "bars": counts, "width": 2160, "height": 1200,
               "source_rows": len(candles), "retained_rows": len(frame), "unclosed_removed": int((~closed).sum()),
               "inputs_path": str(input_path), "inputs_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "timestamp_semantics": "candle_open_utc", "volume_unit": "base_asset"}
    receipt_path = path.with_suffix(".receipt.json")
    receipt["receipt_path"] = str(receipt_path)
    receipt_path.write_text(_json(receipt), encoding="utf-8")
    return receipt
