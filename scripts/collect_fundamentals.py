"""Collect prospective ETH news/book context plus receipt-grounded public facts."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wallex_lab.context import collect, fetch_public, save_json, timestamp
from wallex_lab.data import utc_now, PublicClient, download_history, load_market, aggregate_complete

ROOT = Path("data/context-fusion")
OUTPUT = ROOT / "current.json"
DOCUMENTATION = "https://api-docs.defillama.com/"
ENDPOINTS = {"chains": "https://api.llama.fi/v2/chains",
             "historical_tvl": "https://api.llama.fi/v2/historicalChainTvl/Ethereum",
             "stablecoins": "https://stablecoins.llama.fi/stablecoinchains"}


def parse_facts(kind, payload, receipt):
    """Validate vendor data and retain receipt availability, never backdate it."""
    if receipt.get("status") != 200 or receipt.get("error"):
        raise ValueError("Fundamental source request failed")
    if timestamp(receipt["response_received_utc"]) < timestamp(receipt["request_started_utc"]):
        raise ValueError("Backwards source receipt clocks")
    if not isinstance(payload, list) or not payload or not all(isinstance(x, dict) for x in payload):
        raise ValueError("Expected nonempty source rows")
    measurement = None
    if kind == "historical_tvl":
        row = max(payload, key=lambda x: int(x["date"]))
        measurement = datetime.fromtimestamp(int(row["date"]), timezone.utc).isoformat()
        if timestamp(measurement) > timestamp(receipt["response_received_utc"]):
            raise ValueError("Vendor measurement date is in the future")
        value = row["tvl"]
        metric = "ethereum_defi_tvl_latest_daily_usd"
        text = "DefiLlama latest dated Ethereum DeFi TVL point, excluding liquid staking and double-counted TVL."
        quality = ["Vendor measurement date is not this system's publication or availability time.",
                   "USD valuation can change with token prices and vendor methodology; no directional price signal established."]
    else:
        rows = [x for x in payload if x.get("name") == "Ethereum"]
        if len(rows) != 1:
            raise ValueError("Expected exactly one Ethereum chain")
        row = rows[0]
        if kind == "chains":
            if row.get("tokenSymbol") != "ETH" or str(row.get("chainId")) != "1":
                raise ValueError("Ethereum chain identity mismatch")
            value, metric = row["tvl"], "ethereum_defi_tvl_current_usd"
            text = "DefiLlama reports the current USD TVL of Ethereum DeFi under its chain TVL methodology."
        elif kind == "stablecoins":
            value, metric = row["totalCirculatingUSD"]["peggedUSD"], "ethereum_usd_pegged_stablecoin_circulation_usd"
            text = "DefiLlama reports USD-pegged stablecoins circulating on Ethereum valued in USD; other currency pegs excluded."
        else:
            raise ValueError("Unknown source metric")
        quality = ["Source omits a metric update timestamp; observation and availability use actual local receipt time.",
                   "Vendor aggregate is not independently verified on chain and establishes no directional ETH price prediction."]
    if isinstance(value, bool):
        raise ValueError("Boolean metric value")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Fundamental metric must be finite and nonnegative")
    received = receipt["response_received_utc"]
    url = receipt["url"]
    fact = {"id": hashlib.sha256(f"{url}|{metric}|{receipt['sha256']}".encode()).hexdigest(),
            "url": url, "source_url": url, "source": "DefiLlama", "documentation": DOCUMENTATION,
            "assets": ["ETH"], "observed_at": received, "available_at": received,
            "metric": metric, "value": value, "unit": "USD", "text": text[:600],
            "source_receipt": receipt, "quality": quality}
    if measurement:
        fact["measurement_at"] = measurement
        fact["measurement_age_seconds_at_observation"] = (timestamp(received) - timestamp(measurement)).total_seconds()
    return [fact]


def verify():
    context = json.loads(OUTPUT.read_text(encoding="utf-8"))
    rebuilt, receipt_count = [], 0
    for item in context["fundamental_sources"]:
        receipt = item["receipt"]
        if "error" not in receipt:
            raw = Path(receipt["body_path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
            rebuilt.extend(parse_facts(item["kind"], json.loads(raw), receipt))
    assert sorted(rebuilt, key=lambda x: x["id"]) == sorted(context["fundamentals"], key=lambda x: x["id"])
    for fact in rebuilt:
        assert timestamp(fact["available_at"]) <= timestamp(context["available_at"])
        assert fact["available_at"] == fact["observed_at"]
        assert fact["assets"] == ["ETH"] and fact["unit"] == "USD"
    for path in (ROOT / "raw").glob("*.receipt.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if "sha256" in receipt:
            assert hashlib.sha256(Path(receipt["body_path"]).read_bytes()).hexdigest() == receipt["sha256"]
            receipt_count += 1
    validation = {"validated_utc": utc_now(), "fundamental_facts": len(rebuilt), "raw_hashes_checked": receipt_count,
                  "fact_availability_not_backdated": True, "receipt_reconstruction_passed": True}
    if "technical_data" in context:
        technical = context["technical_data"]
        native = load_market(technical["path_1m"], 60)
        bars = load_market(technical["path_5m"], 300)
        derived, _ = aggregate_complete(native, 300, 60)
        import pandas as pd
        pd.testing.assert_frame_equal(bars, derived, check_dtype=False, check_exact=False, rtol=1e-12)
        cutoff = timestamp(technical["requested_to_exclusive_utc"])
        assert (native.timestamp + pd.Timedelta(seconds=60) <= cutoff).all()
        assert hashlib.sha256(Path(technical["path_1m"]).read_bytes()).hexdigest() == technical["sha256_1m"]
        assert hashlib.sha256(Path(technical["path_5m"]).read_bytes()).hexdigest() == technical["sha256_5m"]
        validation["technical_rows_1m"], validation["technical_rows_5m"] = len(native), len(bars)
        validation["technical_reconstruction_passed"] = True
    save_json(ROOT / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


def run(timeout=10):
    context = collect(str(OUTPUT), timeout=timeout, symbols=("ETHTMN",))
    context["fundamentals"], context["fundamental_sources"] = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch_public, url, ROOT, f"fundamental_{kind}", timeout): kind for kind, url in ENDPOINTS.items()}
        for future in as_completed(futures):
            kind = futures[future]
            raw, receipt = future.result()
            context["sources"].append(receipt)
            context["fundamental_sources"].append({"kind": kind, "receipt": receipt})
            try:
                context["fundamentals"].extend(parse_facts(kind, json.loads(raw), receipt))
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                context["errors"].append({"source": ENDPOINTS[kind], "error": f"{type(exc).__name__}: {exc}"})
    _, receipt = fetch_public(DOCUMENTATION, ROOT, "defillama_documentation", timeout)
    context["sources"].append(receipt)
    if receipt.get("error"):
        context["errors"].append({"source": DOCUMENTATION, "error": receipt["error"]})
    end = int(time.time()) // 60 * 60
    try:
        report = download_history(PublicClient(ROOT, timeout=timeout), "ETHTMN", end - 2 * 86400, end, interval_seconds=300)
        native_path, bars_path = ROOT / "ETHTMN_1m.csv", ROOT / "ETHTMN_5m.csv"
        shutil.copyfile(report["native_path"], native_path)
        shutil.copyfile(report["path"], bars_path)
        context["technical_data"] = {"symbol": "ETHTMN", "price_unit": "toman", "path_1m": str(native_path),
                                     "path_5m": str(bars_path), "source_interval_seconds": 60, "derived_interval_seconds": 300,
                                     "requested_from_utc": report["requested_from_utc"],
                                     "requested_to_exclusive_utc": report["requested_to_exclusive_utc"],
                                     "sha256_1m": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                                     "sha256_5m": hashlib.sha256(bars_path.read_bytes()).hexdigest(),
                                     "native_quality": report["native_quality"], "quality_5m": report["quality"],
                                     "available_at": utc_now(), "source_report": report}
        context["sources"].extend(report["history_receipts"])
    except (ValueError, RuntimeError, OSError) as exc:
        context["errors"].append({"source": "wallex/ETHTMN/current_candles", "error": f"{type(exc).__name__}: {exc}"})
    context["observed_at"] = context["available_at"] = utc_now()
    context["status"] = "partial" if context["errors"] else "ok"
    context["quality"].extend(["Fundamentals are vendor aggregate facts observed now; no historical availability or price-prediction skill inferred.",
                               "Ethereum TVL and USD-pegged stablecoin values are USD measurements, separate from Wallex ETHTMN prices in toman.",
                               "Daily metric measurement dates are not publication timestamps; only receipt-time availability is asserted."])
    save_json(OUTPUT, context)
    verify()
    summary = {"output": str(OUTPUT), "status": context["status"], "available_at": context["available_at"],
               "news": len(context["news"]), "eth_news": sum("ETH" in x["assets"] for x in context["news"]),
               "orderflow_symbols": list(context["orderflow"]), "fundamental_facts": len(context["fundamentals"]),
               "facts": [{k: fact[k] for k in ("metric", "value", "unit", "available_at")} for fact in context["fundamentals"]],
               "errors": context["errors"]}
    print(json.dumps(summary, indent=2), flush=True)
    if not context["fundamentals"]:
        raise RuntimeError("No usable fundamental facts collected; raw errors preserved")
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify saved receipts/facts offline")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if not 0 < args.timeout <= 20:
        parser.error("timeout must be in (0,20]")
    verify() if args.verify else run(args.timeout)
