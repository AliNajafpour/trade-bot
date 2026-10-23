"""Short public-book paper smoke test; quote assumptions are never exchange fills."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import PublicClient, load_market, utc_now

STRATEGIES = ("rules", "trees", "neural", "kronos", "agent", "fusion")


def stamp(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("UTC-aware observation/decision clocks required")
    return result.tz_convert("UTC")


def validate_quote(payload, receipt, asof, max_age_seconds=30):
    """Reject malformed, crossed, old and future observations before paper use."""
    now = stamp(asof)
    received = stamp(receipt["response_received_utc"])
    started = stamp(receipt["request_started_utc"])
    if received > now or not 0 <= (now - received).total_seconds() <= max_age_seconds:
        raise ValueError("Stale or future observation receipt")
    if not 0 <= (received - started).total_seconds() <= max_age_seconds:
        raise ValueError("Stale or backwards request clock")
    headers = {k.lower(): v for k, v in receipt.get("headers", {}).items()}
    if "date" not in headers:
        raise ValueError("No HTTP Date to check server freshness")
    server_time = stamp(parsedate_to_datetime(headers["date"]))
    if abs((received - server_time).total_seconds()) > max_age_seconds:
        raise ValueError("Stale HTTP server Date")
    if receipt.get("status") != 200 or payload.get("success") is False:
        raise ValueError("Unsuccessful book response")
    book = payload.get("result", {})
    try:
        bids = [(float(x["price"]), float(x["quantity"])) for x in book["bid"]]
        asks = [(float(x["price"]), float(x["quantity"])) for x in book["ask"]]
        if not bids or not asks or not all(math.isfinite(v) and v > 0 for level in bids + asks for v in level):
            raise ValueError("Empty/nonpositive/nonfinite book")
        bid, bid_qty = max(bids)
        ask, ask_qty = min(asks)
    except (TypeError, KeyError, OverflowError) as exc:
        raise ValueError("Malformed book levels") from exc
    if bid > ask:
        raise ValueError("Crossed book")
    return {"request_started_utc": started.isoformat(), "observed_utc": received.isoformat(),
            "server_http_date_utc": server_time.isoformat(), "bid": bid, "ask": ask,
            "bid_quantity": bid_qty, "ask_quantity": ask_qty, "mid": (bid + ask) / 2,
            "full_spread_bps": (ask - bid) / ((bid + ask) / 2) * 10000,
            "sha256": receipt["sha256"], "body_path": receipt.get("body_path"),
            "freshness_basis": "local receipt and HTTP Date; exchange event timestamp unavailable"}


def simulate_accounts(signals, observations, initial_cash=5_000_000., fee_bps=20., max_exposure=.5, min_notional=50_000.):
    """Independent long-only ledgers; modeled buys at observed ask and sells at bid."""
    fee = fee_bps / 10000
    if (not all(math.isfinite(v) for v in (initial_cash, fee, max_exposure, min_notional))
            or initial_cash <= 0 or not 0 <= fee < 1 or not 0 <= max_exposure <= 1 or min_notional < 0):
        raise ValueError("Invalid paper account settings")
    accounts = {}
    for signal in signals:
        name = signal["name"]
        if signal.get("status") == "blocked":
            accounts[name] = {"status": "blocked", "reason": signal["reason"], "signal": signal}
            continue
        target = float(signal["target"])
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Target must be finite in [0,1]")
        target = min(max_exposure, target)
        available = stamp(signal["available_utc"])
        cash, quantity, paid = initial_cash, 0., 0.
        events, marks = [], []
        for quote in observations:
            # Prices from a request already in flight at decision adoption are refused.
            if stamp(quote["request_started_utc"]) < available:
                continue
            bid, ask = quote["bid"], quote["ask"]
            equity_ask = cash + quantity * ask
            buy_value = (target * equity_ask - quantity * ask) / (1 + target * fee)
            equity_bid = cash + quantity * bid
            sell_value = (quantity * bid - target * equity_bid) / (1 - target * fee)
            if buy_value >= min_notional:
                amount = min(buy_value / ask, cash / (ask * (1 + fee)), quote["ask_quantity"])
                side, price = "buy", ask
            elif sell_value >= min_notional:
                amount = min(sell_value / bid, quantity, quote["bid_quantity"])
                side, price = "sell", bid
            else:
                amount = 0.
            if amount > 0 and amount * price >= min_notional:
                value = amount * price
                charge = value * fee
                if side == "buy":
                    cash -= value + charge
                    quantity += amount
                else:
                    cash += value - charge
                    quantity -= amount
                paid += charge
                events.append({"observed_utc": quote["observed_utc"], "side": side, "price": price,
                               "quantity": amount, "notional": value, "fee": charge,
                               "quote_sha256": quote["sha256"], "assumption": "paper top quote, no fill guarantee"})
            mark = cash + quantity * bid * (1 - fee)
            marks.append({"observed_utc": quote["observed_utc"], "cash": cash, "quantity": quantity,
                          "marked_liquidation_equity": mark, "hypothetical_exit_fee": quantity * bid * fee,
                          "top_bid_quantity_covers_inventory": quantity <= quote["bid_quantity"],
                          "bid": bid, "ask": ask, "quote_sha256": quote["sha256"]})
        if not marks:
            accounts[name] = {"status": "blocked", "reason": "No admitted quote requested after decision availability", "signal": signal,
                              "paper_events": [], "cash": initial_cash, "quantity": 0.}
            continue
        last = marks[-1]
        accounts[name] = {"status": "ok", "signal": signal, "target_applied": target, "initial_cash": initial_cash,
                          "cash": cash, "quantity": quantity, "paper_fees_paid": paid,
                          "paper_events": events, "marks": marks,
                          "marked_liquidation_equity": last["marked_liquidation_equity"],
                          "hypothetical_exit_fee": last["hypothetical_exit_fee"],
                          "marked_net_pnl": last["marked_liquidation_equity"] - initial_cash,
                          "marked_net_return": last["marked_liquidation_equity"] / initial_cash - 1,
                          "liquidation_is_hypothetical": True,
                          "top_bid_quantity_covers_inventory": last["top_bid_quantity_covers_inventory"]}
    return accounts


def read_signals(strategy_root, prospective_path, data_path, adopted_utc):
    """Freeze existing output targets only after collection starts, recording age."""
    signals = []
    frame = load_market(data_path)
    candle_close = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=900)
    adopted = stamp(adopted_utc)
    for name in STRATEGIES:
        path = Path(strategy_root) / name / "outputs.npz"
        try:
            raw = path.read_bytes()
            with np.load(path, allow_pickle=False) as output:
                targets = output["target"]
                if targets.shape != (len(frame),) or not np.isfinite(targets).all():
                    raise ValueError("Output length or target values do not match canonical dataset")
                target = float(targets[-1])
                if not 0 <= target <= 1:
                    raise ValueError("Target outside [0,1]")
            signals.append({"name": name, "status": "ok", "target": target, "source": str(path),
                            "source_sha256": hashlib.sha256(raw).hexdigest(), "available_utc": adopted.isoformat(),
                            "source_candle_close_utc": candle_close.isoformat(),
                            "source_candle_age_seconds_at_adoption": (adopted - candle_close).total_seconds(),
                            "signal_type": "delayed stale frozen backtest research target",
                            "inference_completion_time_unverified": True,
                            "source_file_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
        except (OSError, ValueError, KeyError) as exc:
            signals.append({"name": name, "status": "blocked", "reason": f"{type(exc).__name__}: {exc}", "source": str(path)})
    try:
        path = Path(prospective_path)
        raw = path.read_bytes()
        item = json.loads(raw)
        decision, metadata = item["decision"], item["metadata"]
        completed = stamp(metadata["completed_utc"])
        if metadata.get("status") != "ok" or not metadata.get("prospective") or completed > adopted:
            raise ValueError("Prospective decision has not completed successfully by adoption")
        target = float(decision["response"]["target_allocation"])
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Invalid prospective target")
        close = stamp(metadata["last_bar_close_utc"])
        signals.append({"name": "prospective_agent", "status": "ok", "target": target, "source": str(path),
                        "source_sha256": hashlib.sha256(raw).hexdigest(), "available_utc": adopted.isoformat(),
                        "original_inference_started_utc": decision["inference_started_utc"],
                        "original_inference_completed_utc": completed.isoformat(),
                        "source_candle_close_utc": close.isoformat(),
                        "source_candle_age_seconds_at_adoption": (adopted - close).total_seconds(),
                        "inference_age_seconds_at_adoption": (adopted - completed).total_seconds(),
                        "signal_type": "actual prospective LLM decision adopted after original completion",
                        "response": decision["response"]})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        signals.append({"name": "prospective_agent", "status": "blocked", "reason": f"{type(exc).__name__}: {exc}"})
    return signals


def collect(symbol="BTCTMN", samples=4, spacing_seconds=15., strategy_root=None,
            prospective_path=None, root="runs/shadow"):
    if samples < 4 or not 0 < spacing_seconds <= 15:
        raise ValueError("At least four observations and spacing in (0,15] seconds required")
    started = utc_now()
    run_dir = Path(root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    source = Path(strategy_root or f"runs/wallex-20261021-causal/{symbol}")
    current = Path(prospective_path or f"runs/prospective/{symbol}/current.json")
    signals = read_signals(source, current, f"data/market/{symbol}_15m.csv", utc_now())
    (run_dir / "signals.json").write_text(json.dumps(signals, indent=2), encoding="utf-8")
    client = PublicClient(run_dir, timeout=12)
    observations, failures = [], []
    first_tick = time.monotonic()
    for index in range(samples):
        delay = first_tick + index * spacing_seconds - time.monotonic()
        while delay > 0:
            time.sleep(min(delay, 15.))
            delay = first_tick + index * spacing_seconds - time.monotonic()
        try:
            payload, receipt = client.get("/v1/depth", {"symbol": symbol}, f"{symbol}_shadow_depth")
            quote = validate_quote(payload, receipt, utc_now())
            quote["sample_index"] = index
            observations.append(quote)
            print(f"observation {index + 1}/{samples} {quote['observed_utc']} bid={quote['bid']} ask={quote['ask']}", flush=True)
        except (RuntimeError, ValueError, KeyError) as exc:
            failures.append({"sample_index": index, "failed_utc": utc_now(), "reason": f"{type(exc).__name__}: {exc}"})
        (run_dir / "observations.json").write_text(json.dumps({"observations": observations, "failures": failures}, indent=2), encoding="utf-8")
    accounts = simulate_accounts(signals, observations)
    report = {"status": "ok" if len(observations) >= 4 else "blocked", "symbol": symbol,
              "collection_started_utc": started, "completed_utc": utc_now(), "requested_samples": samples,
              "valid_observations": len(observations), "spacing_seconds": spacing_seconds,
              "paper_only": True, "real_orders_submitted": 0, "initial_cash_per_independent_account": 5_000_000,
              "fee_bps_per_side_assumption": 20, "max_exposure": .5, "min_notional_toman": 50_000,
              "accounts": accounts, "observations": observations, "failures": failures,
              "limitations": ["Operational smoke only; a roughly45-second window cannot validate prospective returns.",
                              "Prices were requested after signal adoption; no historical price or backdated fills used.",
                              "Frozen historical targets are stale research signals, not fresh model inference.",
                              "Book quotes have local receipt and HTTP clocks, but no exchange event timestamp.",
                              "Paper buys/sells assume visible best ask/bid availability and cap event size by displayed top quantity; no fill guarantee.",
                              "Marked liquidation deducts assumed exit fees; valuation assumes whole inventory at last bid and reports displayed quantity coverage.",
                              "Fees are research assumptions; no private account fee schedule verified."]}
    if observations:
        report["price_evolution"] = {"first_mid": observations[0]["mid"], "last_mid": observations[-1]["mid"],
                                      "mid_return": observations[-1]["mid"] / observations[0]["mid"] - 1,
                                      "elapsed_observation_seconds": (stamp(observations[-1]["observed_utc"]) - stamp(observations[0]["observed_utc"])).total_seconds()}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "status": report["status"], "valid_observations": len(observations),
                      "accounts": {name: {k: value.get(k) for k in ("status", "target_applied", "quantity", "marked_net_pnl", "paper_fees_paid")} for name, value in accounts.items()}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCTMN")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--spacing-seconds", type=float, default=15.)
    parser.add_argument("--strategy-root")
    parser.add_argument("--prospective-path")
    parser.add_argument("--root", default="runs/shadow")
    args = parser.parse_args()
    report = collect(**vars(args))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
