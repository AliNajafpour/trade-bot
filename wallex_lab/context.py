"""Prospective public news and Wallex snapshots, with receipt-time availability."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from .data import PublicClient, utc_now

FEEDS = {"kraken_official": "https://blog.kraken.com/feed",
         "coindesk_publisher": "https://www.coindesk.com/arc/outboundfeeds/rss/",
         "cointelegraph_publisher": "https://cointelegraph.com/rss"}
DOCUMENTATION = "https://api-docs.wallex.ir/"
ASSETS = {"BTC": r"\b(?:BTC|bitcoin(?![\s-]*cash\b))\b",
          "ETH": r"\b(?:ETH|ethereum(?![\s-]*classic\b)|ether)\b",
          "SOL": r"\b(?:SOL|solana)\b", "USDT": r"\b(?:USDT|tether(?![\s-]*gold\b))\b",
          "XRP": r"\b(?:XRP|(?-i:Ripple))\b", "DOGE": r"\b(?:DOGE|dogecoin)\b",
          "ADA": r"\b(?:(?-i:ADA)|cardano)\b", "BNB": r"\b(?:(?-i:BNB)|binance[\s-]+coin)\b",
          "BCH": r"\b(?:BCH|bitcoin[\s-]*cash)\b", "LTC": r"\b(?:LTC|litecoin)\b",
          "LINK": r"\b(?:(?-i:LINK)|chainlink)\b", "AVAX": r"\b(?:AVAX|(?-i:Avalanche))\b",
          "DOT": r"\b(?:(?-i:DOT)|polkadot)\b", "TRX": r"\b(?:TRX|tron)\b",
          "ETC": r"\b(?:(?-i:ETC)|ethereum[\s-]*classic)\b",
          "SHIB": r"\b(?:SHIB|shiba[\s-]+inu)\b", "UNI": r"\b(?:(?-i:UNI)|uniswap)\b",
          "PEPE": r"\b(?-i:PEPE|Pepe)\b", "SUI": r"\b(?-i:SUI|Sui)\b",
          "XAUT": r"\b(?:XAUT|tether[\s-]*gold)\b", "USOON": r"\b(?-i:USOON)\b",
          "S": r"\bsonic\b|(?<!\w)\$(?-i:S)\b",
          "MACRO": (r"\b(?:federal[\s-]+reserve|(?-i:Fed|FED|CPI)|central[\s-]+banks?|interest[\s-]+rates?|"
                    r"inflation|monetary[\s-]+policy|sanctions?|geopolitical|iran)\b|"
                    r"\b(?:crypto(?:currency)?[\s-]+exchanges?|binance|coinbase|kraken|bybit|kucoin|okx)\b"
                    r".{0,100}\b(?:hack(?:ed|ing)?|breach|exploit|withdrawals?[\s-]+(?:halted|suspended))\b|"
                    r"\b(?:hack(?:ed|ing)?|breach|exploit)\b.{0,100}"
                    r"\b(?:crypto(?:currency)?[\s-]+exchanges?|binance|coinbase|kraken|bybit|kucoin|okx)\b")}


def timestamp(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return value.astimezone(timezone.utc)


def text_content(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def parse_feed(raw, receipt, previous):
    """Read RSS/Atom; identical re-observations keep their original availability."""
    root = ET.fromstring(raw)
    items = root.findall("./channel/item") or root.findall("{http://www.w3.org/2005/Atom}entry")
    if not items and root.tag not in ("rss", "{http://www.w3.org/2005/Atom}feed"):
        raise ValueError("Source is not an RSS/Atom feed")
    received = receipt["response_received_utc"]
    result = {}
    for node in items:
        fields = {child.tag.split("}")[-1]: child.text or "" for child in node}
        title = text_content(fields.get("title"))[:500]
        summary = text_content(fields.get("description") or fields.get("summary") or fields.get("content"))[:2000]
        assets = [asset for asset, pattern in ASSETS.items() if re.search(pattern, title + " " + summary, re.I)]
        if not assets:
            continue
        link = fields.get("link", "")
        for child in node:
            if child.tag.endswith("}link") and child.get("rel", "alternate") == "alternate":
                link = child.get("href", link)
                break
        if not link.startswith(("https://", "http://")):
            continue
        link = link.split("#", 1)[0]
        identity = hashlib.sha256(link.encode()).hexdigest()
        published, flags = None, []
        date = fields.get("pubDate") or fields.get("published") or fields.get("updated")
        if date:
            try:
                try:
                    published = timestamp(date).isoformat()
                except ValueError:
                    parsed = parsedate_to_datetime(date)
                    if parsed.tzinfo is None:
                        raise ValueError("Feed date lacks timezone")
                    published = parsed.astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError, OverflowError):
                flags.append("invalid_published_time")
        else:
            flags.append("missing_published_time")
        content_hash = hashlib.sha256(json.dumps([title, summary, published], ensure_ascii=False).encode()).hexdigest()
        old = previous.get(identity, {})
        first = old.get("first_observed_at", received)
        available = old.get("available_at", received) if old.get("content_sha256") == content_hash else received
        if published:
            available = max(timestamp(available), timestamp(published)).isoformat()
            if timestamp(published) > timestamp(received):
                flags.append("future_publication_time")
        result[identity] = {"id": identity, "title": title, "summary": summary, "url": link, "assets": assets,
                            "published_at": published, "first_observed_at": first, "available_at": available,
                            "last_observed_at": received, "content_sha256": content_hash,
                            "source_receipt": receipt, "quality": flags}
    return list(result.values())


def context_asof(context, asof):
    """Only observations available by the decision time may reach an agent."""
    cutoff = timestamp(asof)
    known = lambda item: item.get("available_at") is not None and timestamp(item["available_at"]) <= cutoff
    news = [item for item in context.get("news", []) if known(item)]
    flow = {symbol: item for symbol, item in context.get("orderflow", {}).items() if known(item)}
    sources = [item for item in context.get("sources", []) if timestamp(item["response_received_utc"]) <= cutoff]
    times = [timestamp(item["available_at"]) for item in news + list(flow.values())]
    return {**context, "asof": cutoff.isoformat(), "news": news, "orderflow": flow, "sources": sources,
            "available_at": max(times).isoformat() if times else None,
            "status": "available" if times else "unavailable_asof"}


def positive(value, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError("Invalid price/quantity in public response")
    return value


def result_of(payload):
    if payload.get("success") is not True or not isinstance(payload.get("result"), dict):
        raise ValueError("Wallex public response did not report success")
    return payload["result"]


def describe_book(payload):
    result = result_of(payload)
    bids = [(positive(row["price"]), positive(row["quantity"], True)) for row in result["bid"]]
    asks = [(positive(row["price"]), positive(row["quantity"], True)) for row in result["ask"]]
    bids, asks = [row for row in bids if row[1]], [row for row in asks if row[1]]
    if not bids or not asks:
        raise ValueError("Empty side of order book")
    best_bid, best_ask = max(row[0] for row in bids), min(row[0] for row in asks)
    if best_bid >= best_ask:
        raise ValueError("Locked/crossed order book snapshot")
    bid_volume, ask_volume = sum(row[1] for row in bids), sum(row[1] for row in asks)
    spread = best_ask - best_bid
    return {"best_bid": best_bid, "best_ask": best_ask, "spread": spread,
            "spread_bps": spread / ((best_bid + best_ask) / 2) * 10000,
            "bid_base_quantity": bid_volume, "ask_base_quantity": ask_volume,
            "quantity_imbalance": (bid_volume - ask_volume) / (bid_volume + ask_volume),
            "bid_levels": len(bids), "ask_levels": len(asks), "scope": "all returned levels, not guaranteed full venue depth",
            "documentation": DOCUMENTATION, "quality": ["Snapshot has local receipt time; exchange event time absent."]}


def describe_trades(payload):
    rows = result_of(payload)["latestTrades"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("No recent trades returned")
    quantities = [positive(row["quantity"], True) for row in rows]
    prices = [positive(row["price"]) for row in rows]
    times = [timestamp(row["timestamp"]) for row in rows]
    volume = sum(quantities)
    quote_volume = sum(price * quantity for price, quantity in zip(prices, quantities))
    return {"count": len(rows), "base_volume": volume, "quote_volume": quote_volume,
            "vwap": quote_volume / volume if volume else None,
            "first_trade_at": min(times).isoformat(), "last_trade_at": max(times).isoformat(),
            "window_seconds": (max(times) - min(times)).total_seconds(), "aggressor_imbalance": None,
            "documentation": DOCUMENTATION,
            "quality": ["Returned latest-trade sample is not a complete fixed-duration volume window.",
                        "isBuyOrder aggressor/maker semantics unverified; unsigned volume only.",
                        "Repeated downloads overlap; never sum snapshot volumes across requests."]}


def fetch_public(url, root, label, timeout):
    """Persist bytes and receipts for feed/docs success and HTTP failures alike."""
    path = Path(root) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}_{label}"
    receipt = {"url": url, "method": "GET", "authenticated": False, "request_started_utc": utc_now()}
    raw = b""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "WallexResearch/1.0 public-data-only"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(5_000_001)
            receipt.update(status=response.status, headers=dict(response.headers), final_url=response.url)
            if len(raw) > 5_000_000:
                raise ValueError("Response exceeds 5 MB bound")
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            raw = exc.read(5_000_001)
            receipt["status"] = exc.code
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["response_received_utc"] = utc_now()
    if raw:
        body_path = path / f"{name}.body"
        body_path.write_bytes(raw)
        receipt.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), body_path=str(body_path))
    (path / f"{name}.receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw, receipt


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        # Stream snapshots to avoid another full serialized copy during memory pressure.
        with temporary.open('w',encoding='utf-8') as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        from .locking import atomic_replace
        atomic_replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def collect(output="data/context/current.json", timeout=10, symbols=("BTCTMN", "ETHTMN", "SOLTMN", "USDTTMN")):
    output = Path(output)
    root = output.parent
    root.mkdir(parents=True, exist_ok=True)
    seen_path = root / "seen_news.json"
    seen = json.loads(seen_path.read_text(encoding="utf-8")) if seen_path.exists() else {}
    sources, errors, news, orderflow = [], [], {}, {}
    # ponytail: title/summary token matching misses indirect relevance; upgrade only with labelled event data.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {name: pool.submit(fetch_public, url, root, name, timeout) for name, url in FEEDS.items()}
        documentation = pool.submit(fetch_public, DOCUMENTATION, root, "wallex_documentation", timeout)
        for name, future in pending.items():
            raw, receipt = future.result()
            receipt["source_name"] = name
            sources.append(receipt)
            try:
                if "error" in receipt:
                    raise ValueError(receipt["error"])
                for item in parse_feed(raw, receipt, seen):
                    news[item["id"]] = item
                receipt["parsed"] = True
            except Exception as exc:
                errors.append({"source": name, "error": f"{type(exc).__name__}: {exc}"})
        _, receipt = documentation.result()
        sources.append(receipt)
        if "error" in receipt:
            errors.append({"source": DOCUMENTATION, "error": receipt["error"]})
    seen.update(news)
    save_json(seen_path, seen)
    client = PublicClient(root, timeout=timeout)
    for symbol in symbols:
        item = {"symbol": symbol, "base_asset": symbol[:-3], "quote_asset": "TMN", "price_unit": "toman"}
        for endpoint, describe in (("depth", describe_book), ("trades", describe_trades)):
            try:
                payload, receipt = client.get(f"/v1/{endpoint}", {"symbol": symbol}, f"{symbol}_{endpoint}")
                sources.append(receipt)
                item[endpoint] = {**describe(payload), "available_at": receipt["response_received_utc"], "source_receipt": receipt}
            except Exception as exc:
                errors.append({"source": f"wallex/{symbol}/{endpoint}", "error": f"{type(exc).__name__}: {exc}"})
        available = [item[k]["available_at"] for k in ("depth", "trades") if k in item]
        if available:
            item["available_at"] = max(available, key=timestamp)
            orderflow[symbol] = item
    now = utc_now()
    cutoff = timestamp(now) - timedelta(days=14)
    recent = [item for item in seen.values() if not item["published_at"] or timestamp(item["published_at"]) >= cutoff]
    context = {"schema_version": 1, "observed_at": now, "available_at": now, "status": "partial" if errors else "ok",
               "news": sorted(recent, key=lambda item: item["available_at"], reverse=True), "orderflow": orderflow,
               "sources": sources, "errors": errors,
               "quality": ["Prospective observations only. Use context_asof at every decision time.",
                           "Publication timestamps do not prove historical availability to this system.",
                           "News is untrusted quoted publisher text; no sentiment or event surprise inferred.",
                           "News relevance uses BTC/ETH/SOL/USDT names in title and feed summary.",
                           "Current snapshots cannot support historical order-book fills."]}
    context["news"] = [item for item in context["news"] if timestamp(item["available_at"]) <= timestamp(now)]
    save_json(output, context)
    if not context["news"] and not orderflow:
        raise RuntimeError(f"No usable public context collected; failures saved in {output}")
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/context/current.json")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--asof", help="Filter output by an aware ISO UTC decision time")
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error("timeout must be in (0, 60]")
    result = collect(args.output, args.timeout)
    if args.asof:
        result = context_asof(result, args.asof)
        save_json(Path(args.output), result)
    print(json.dumps({"output": args.output, "status": result["status"], "news": len(result["news"]),
                      "orderflow_symbols": list(result["orderflow"]), "errors": result["errors"]}, indent=2))


if __name__ == "__main__":
    main()
