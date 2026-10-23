"""Public Wallex downloads with raw receipts and strict, unfilled UTC OHLCV bars."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://api.wallex.ir"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_history(payload, interval_seconds=60, closed_before=None):
    """Validate arrays, reject conflicting duplicates and remove unclosed/future bars."""
    if payload.get("s") == "no_data":
        return pd.DataFrame(columns=COLUMNS), {"no_data": True}
    if payload.get("s") != "ok":
        raise ValueError(f"History status: {payload}")
    keys = ["t", "o", "h", "l", "c", "v"]
    if any(not isinstance(payload.get(k), list) for k in keys):
        raise ValueError("Missing OHLCV arrays")
    if len({len(payload[k]) for k in keys}) != 1:
        raise ValueError("Unequal OHLCV arrays")
    frame = pd.DataFrame({name: payload[key] for name, key in zip(COLUMNS, keys)})
    numeric = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite OHLCV")
    if ((numeric.timestamp % interval_seconds) != 0).any():
        raise ValueError("Candle opens are not aligned to the requested native interval")
    if ((numeric[["open", "high", "low", "close"]] <= 0).any().any()
            or (numeric.volume < 0).any()
            or (numeric.high < numeric[["open", "close", "low"]].max(axis=1)).any()
            or (numeric.low > numeric[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC prices or negative volume")
    duplicates = numeric.duplicated("timestamp", keep=False)
    if duplicates.any() and (numeric[duplicates].groupby("timestamp").nunique() > 1).any().any():
        raise ValueError("Conflicting duplicate candles")
    before = len(numeric)
    numeric = numeric.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    duplicate_count = before - len(numeric)
    cutoff = time.time() if closed_before is None else closed_before
    closed = numeric.timestamp + interval_seconds <= cutoff
    quality = {"duplicates_removed": duplicate_count, "unclosed_removed": int((~closed).sum())}
    numeric = numeric.loc[closed].copy()
    numeric.timestamp = pd.to_datetime(numeric.timestamp, unit="s", utc=True)
    numeric.attrs["interval_seconds"] = interval_seconds
    return numeric.reset_index(drop=True), quality


def aggregate_complete(frame, interval_seconds=900, source_seconds=60):
    """Aggregate only buckets containing every native candle, preserving all gaps."""
    if interval_seconds % source_seconds or interval_seconds < source_seconds:
        raise ValueError("Target interval must be a positive multiple of source interval")
    source = frame.sort_values("timestamp").set_index("timestamp")
    groups = source.resample(f"{interval_seconds}s", origin="epoch", label="left", closed="left")
    result = groups.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = groups["close"].count()
    valid = counts == interval_seconds // source_seconds
    result = result.loc[valid].reset_index()
    result.attrs["interval_seconds"] = interval_seconds
    return result, {"incomplete_buckets_removed": int((~valid).sum())}


def quality_report(frame, interval_seconds):
    seconds = frame.timestamp.astype("datetime64[ns, UTC]").astype("int64").to_numpy() // 10**9
    deltas = np.diff(seconds)
    gap_positions = np.flatnonzero(deltas > interval_seconds)
    return {"rows": len(frame), "interval_seconds": interval_seconds,
            "first_open_utc": frame.timestamp.iloc[0].isoformat() if len(frame) else None,
            "last_open_utc": frame.timestamp.iloc[-1].isoformat() if len(frame) else None,
            "missing_intervals": int(np.sum(deltas // interval_seconds - 1)) if len(deltas) else 0,
            "gaps": [{"after": frame.timestamp.iloc[i].isoformat(),
                      "before": frame.timestamp.iloc[i + 1].isoformat(),
                      "missing": int(deltas[i] // interval_seconds - 1)} for i in gap_positions],
            "price_unit": "toman", "volume_unit": "base_asset", "timestamp_semantics": "candle_open_utc",
            "interpolated": False}


def load_market(path, interval_seconds=None):
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame.timestamp = pd.to_datetime(frame.timestamp, utc=True)
    if interval_seconds is None:
        suffix = Path(path).stem.split("_")[-1]
        interval_seconds = int(suffix[:-1]) * 60 if suffix.endswith("m") else None
    if interval_seconds:
        frame.attrs["interval_seconds"] = interval_seconds
    return frame


class PublicClient:
    """One session, bounded timeouts, every response body and failure persisted."""
    def __init__(self, root="data", timeout=25):
        self.root = Path(root)
        (self.root / "raw").mkdir(parents=True, exist_ok=True)
        (self.root / "market").mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WallexResearch/1.0 public-data-only", "Accept": "application/json"})

    def get(self, path, params=None, label="public"):
        url = requests.Request("GET", BASE + path, params=params).prepare().url
        for attempt in range(3):
            started = utc_now()
            receipt = {"url": url, "method": "GET", "request_started_utc": started,
                       "attempt": attempt + 1, "authenticated": False}
            name = f"{time.time_ns()}_{label}"
            response = None
            try:
                response = self.session.get(url, timeout=(8, self.timeout))
                raw = response.content
                body_path = self.root / "raw" / f"{name}.body"
                body_path.write_bytes(raw)
                receipt.update(status=response.status_code, sha256=hashlib.sha256(raw).hexdigest(),
                               body_representation="HTTP decoded response entity bytes",
                               bytes=len(raw), body_path=str(body_path), headers=dict(response.headers))
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                receipt["error"] = f"{type(exc).__name__}: {exc}"
                payload = None
            receipt["response_received_utc"] = utc_now()
            (self.root / "raw" / f"{name}.receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            if "error" not in receipt:
                return payload, receipt
            if response is not None and response.status_code < 500 and response.status_code != 429:
                break
            if attempt < 2:
                time.sleep(attempt + 1)
        raise RuntimeError(receipt["error"])


def download_history(client, symbol, start, end, interval_seconds=900, chunk_days=1):
    """One-day M1 pages: live two-day requests return misleading no_data."""
    if interval_seconds % 60 or interval_seconds < 60:
        raise ValueError("Use minute multiples")
    frames, receipts, pages = [], [], []
    start = int(start) // 60 * 60
    end = int(end) // 60 * 60
    cursor = start
    while cursor < end:
        stop = min(end, cursor + chunk_days * 86400)
        payload, receipt = client.get("/v1/udf/history", {"symbol": symbol, "resolution": "1", "from": cursor, "to": stop - 1}, f"{symbol}_1m")
        frame, detail = normalize_history(payload, 60, min(end, datetime.fromisoformat(receipt["request_started_utc"]).timestamp()))
        if len(frame):
            frame = frame.loc[(frame.timestamp >= pd.Timestamp(cursor, unit="s", tz="UTC")) &
                              (frame.timestamp < pd.Timestamp(stop, unit="s", tz="UTC"))]
            frames.append(frame)
        pages.append({"from": cursor, "to_exclusive": stop, "rows": len(frame),
                      "expected_native_rows": (stop - cursor) // 60,
                      "missing_native_rows": (stop - cursor) // 60 - len(frame), **detail})
        receipts.append(receipt)
        print(f"{symbol} {pd.Timestamp(cursor, unit='s', tz='UTC').isoformat()} {len(frame)} native bars", flush=True)
        cursor = stop
    if not frames:
        raise ValueError(f"No actual candles returned for {symbol}")
    all_rows = pd.concat(frames, ignore_index=True)
    duplicate_rows = all_rows.duplicated("timestamp", keep=False)
    if duplicate_rows.any() and (all_rows[duplicate_rows].groupby("timestamp").nunique() > 1).any().any():
        raise ValueError("Conflicting candles across pages")
    native = all_rows.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    native.attrs["interval_seconds"] = 60
    bars, aggregation = aggregate_complete(native, interval_seconds, 60)
    if not len(bars):
        raise ValueError(f"No complete {interval_seconds}s candles for {symbol}")
    native_path = client.root / "market" / f"{symbol}_1m.csv"
    path = client.root / "market" / f"{symbol}_{interval_seconds // 60}m.csv"
    native.to_csv(native_path, index=False)
    bars.to_csv(path, index=False)
    report = {"symbol": symbol, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "native_path": str(native_path), "native_quality": quality_report(native, 60),
              "quality": quality_report(bars, interval_seconds), "aggregation": aggregation, "pages": pages,
              "expected_aggregate_rows": (end - start) // interval_seconds,
              "missing_aggregate_rows_including_edges": (end - start) // interval_seconds - len(bars),
              "requested_from_utc": pd.Timestamp(start, unit="s", tz="UTC").isoformat(),
              "requested_to_exclusive_utc": pd.Timestamp(end, unit="s", tz="UTC").isoformat(),
              "history_receipts": receipts,
              "caveats": ["Public exchange bars are used as returned; no trade-level reconstruction.",
                          "No missing OHLC filled. Only fully observed native-minute buckets retained.",
                          "No historical bid/ask archive; snapshots do not establish historical fills."]}
    (client.root / "market" / f"{symbol}_{interval_seconds // 60}m.quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"READY {path} rows={len(bars)} gaps={report['quality']['missing_intervals']}", flush=True)
    return report


def acquire(symbols=("BTCTMN", "ETHTMN", "SOLTMN", "USDTTMN"), days=60, minutes=15, root="data"):
    if days <= 0 or minutes <= 0:
        raise ValueError("Days and minutes must be positive")
    client = PublicClient(root)
    end = int(time.time()) // (minutes * 60) * minutes * 60
    start = end - days * 86400
    manifest = {"created_utc": utc_now(), "source": BASE, "documentation": "https://api-docs.wallex.ir/",
                "symbols": {}, "discovery": [], "errors": [], "interval_seconds": minutes * 60,
                "authentication": "none", "resolution_note": "Live resolution=15 probe returns minute bars. Native resolution=1 is aggregated locally.",
                "pagination_note": "Live M1 requests spanning 2 or 3 days return no_data while one-day historical pages work. One-day pages used; all missing minutes reported."}
    manifest_path = Path(root) / "acquisition_manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Save metadata and live resolution evidence, including failures, before history.
    for path in ("/v1/markets", "/hector/web/v1/markets", "/v1/udf/config"):
        try:
            payload, receipt = client.get(path, label="metadata")
            manifest["discovery"].append(receipt)
            if path == "/v1/markets":
                (Path(root) / "market" / "exchange_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            elif "hector" in path:
                (Path(root) / "market" / "hector_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            manifest["errors"].append({"endpoint": path, "error": str(exc)})
        save()
    for resolution in ("5", "15", "60"):
        try:
            payload, receipt = client.get("/v1/udf/history", {"symbol": symbols[0], "resolution": resolution, "from": end-86400, "to": end-1}, f"probe_{resolution}")
            timestamps = payload.get("t", [])
            manifest["discovery"].append({**receipt, "requested_resolution": resolution, "payload_status": payload.get("s"),
                                          "bars": len(timestamps), "observed_deltas_seconds": sorted(set(np.diff(timestamps).tolist()))})
        except Exception as exc:
            manifest["errors"].append({"resolution": resolution, "error": str(exc)})
        save()
    for symbol in symbols:
        try:
            report = download_history(client, symbol, start, end, minutes * 60)
            manifest["symbols"][symbol] = report
            save()
            for endpoint in ("depth", "trades"):
                try:
                    payload, receipt = client.get(f"/v1/{endpoint}", {"symbol": symbol}, f"{symbol}_{endpoint}")
                    report[f"{endpoint}_receipt"] = receipt
                    (Path(root) / "market" / f"{symbol}_{endpoint}_snapshot.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    manifest["errors"].append({"symbol": symbol, "endpoint": endpoint, "error": str(exc)})
                save()
        except Exception as exc:
            manifest["errors"].append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            save()
    manifest["completed_utc"] = utc_now()
    save()
    if not manifest["symbols"]:
        raise RuntimeError("No market history acquired; see acquisition_manifest.json and raw receipts")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCTMN", "ETHTMN", "SOLTMN", "USDTTMN"])
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    result = acquire(args.symbols, args.days, args.minutes, args.root)
    print(json.dumps({"symbols": {k: v["quality"] for k, v in result["symbols"].items()}, "errors": result["errors"]}, indent=2))


if __name__ == "__main__":
    main()
