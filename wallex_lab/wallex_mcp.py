"""Local read-only MCP adapter for the Iranian exchange at api.wallex.ir."""
from __future__ import annotations

import os
import re
import time
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field
import requests

from .data import BASE, PublicClient, utc_now


def validate_symbol(symbol):
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]{5,30}", symbol):
        raise ValueError("Use an uppercase Wallex symbol such as BTCTMN")
    return symbol


def read_balances():
    """Private data stays in memory; failures never include headers or response bodies."""
    key = os.environ.get("WALLEX_API_KEY", "")
    if os.environ.get("WALLEX_ENABLE_PRIVATE_READS") != "1" or not key:
        raise RuntimeError("Private balance reads require explicit opt-in and WALLEX_API_KEY")
    try:
        with requests.Session() as session:
            response = session.get(BASE + "/v1/account/balances", headers={"x-api-key": key},
                                   timeout=(8, 25), allow_redirects=False)
            if response.status_code != 200:
                raise ValueError("Balance request failed")
            payload = response.json()
            if payload.get("success") is not True or not isinstance(payload.get("result", {}).get("balances"), dict):
                raise ValueError("Invalid balance response")
            return {"balances": payload["result"]["balances"], "authenticated": True,
                    "fetched_utc": utc_now(), "persisted": False}
    except Exception:
        raise RuntimeError("Private balance read failed; credentials and response body omitted") from None


def build_server(client=None):
    public = client if client is not None else PublicClient(os.environ.get("WALLEX_MCP_OUTPUT_ROOT", "data/mcp"))
    server = FastMCP("wallex-ir-read-only", instructions="Read-only Iranian Wallex market data. No order or withdrawal tools.")
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)

    def fetch(path, params=None, label="public"):
        try:
            payload, receipt = public.get(path, params, label)
        except Exception:
            raise RuntimeError("Public Wallex request failed; inspect the public raw receipts") from None
        if not isinstance(payload, dict) or payload.get("success") is False:
            raise RuntimeError("Wallex returned an invalid or unsuccessful public response")
        return {"payload": payload, "receipt": receipt, "source": BASE, "read_only": True}

    @server.tool(annotations=annotations)
    def wallex_markets(symbol: str = "") -> dict:
        """Read public market statistics; optional symbol filters the saved full response."""
        if symbol:
            validate_symbol(symbol)
        result = fetch("/v1/markets", label="mcp_markets")
        if symbol:
            symbols = result["payload"].get("result", {}).get("symbols", {})
            if symbol not in symbols:
                raise ValueError("Symbol is absent from Wallex market metadata")
            result["payload"] = {"success": True, "result": {"symbols": {symbol: symbols[symbol]}}}
            result["receipt_scope"] = "Receipt hashes the full original markets response before symbol filtering"
        return result

    @server.tool(annotations=annotations)
    def wallex_orderbook(symbol: str) -> dict:
        """Read a public bid/ask snapshot. Prices use the market quote currency."""
        return fetch("/v1/depth", {"symbol": validate_symbol(symbol)}, "mcp_depth")

    @server.tool(annotations=annotations)
    def wallex_trades(symbol: str) -> dict:
        """Read the latest public trades. Avoid frequent polling due to venue rate limits."""
        return fetch("/v1/trades", {"symbol": validate_symbol(symbol)}, "mcp_trades")

    @server.tool(annotations=annotations)
    def wallex_history(symbol: str, start: Annotated[int, Field(strict=True)], end: Annotated[int, Field(strict=True)]) -> dict:
        """Read at most one day of native minute candles; end is exclusive UTC epoch seconds."""
        validate_symbol(symbol)
        if not 0 <= start < end <= int(time.time()) or end - start > 86400:
            raise ValueError("History requires a past UTC epoch range of at most 86400 seconds")
        return fetch("/v1/udf/history", {"symbol": symbol, "resolution": "1", "from": start, "to": end - 1}, "mcp_history")

    if os.environ.get("WALLEX_ENABLE_PRIVATE_READS") == "1":
        server.add_tool(read_balances, name="wallex_balances", annotations=annotations)
    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
