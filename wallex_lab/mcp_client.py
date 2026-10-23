"""Real MCP SDK sessions with an explicit read-only tool allowlist."""
from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
import asyncio
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from .data import utc_now
from .wallex_mcp import validate_symbol

PUBLIC_TOOLS = frozenset({"wallex_markets", "wallex_orderbook", "wallex_trades", "wallex_history"})
OFFICIAL_URL = "https://api.wallex.ir/mcp"
OFFICIAL_TOOLS = {"wallex_markets": "wallex_basic_get_markets",
                  "wallex_orderbook": "wallex_basic_get_market_depth",
                  "wallex_trades": "wallex_basic_get_latest_trades",
                  "wallex_history": "wallex_basic_candle_history",
                  "wallex_balances": "wallex_basic_get_account_balances"}


def validate_config(config):
    config = dict(config or {})
    if config.keys() - {"transport", "python", "url", "token_env", "timeout_seconds", "private_reads"}:
        raise ValueError("Unknown MCP configuration fields; keep credentials in environment variables")
    transport = config.setdefault("transport", "stdio")
    if transport not in {"stdio", "streamable_http"}:
        raise ValueError("MCP transport must be stdio or streamable_http")
    timeout = config.setdefault("timeout_seconds", 120)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 1 <= timeout <= 300:
        raise ValueError("MCP timeout_seconds must be between 1 and 300")
    if not isinstance(config.get("private_reads", False), bool):
        raise ValueError("MCP private_reads must be boolean")
    if "python" in config and (not isinstance(config["python"], str) or not config["python"].strip() or "\n" in config["python"]):
        raise ValueError("MCP python must name an installed Python executable")
    if transport == "streamable_http":
        url = urlsplit(config.get("url", ""))
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or (url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}))):
            raise ValueError("Use an HTTPS MCP URL without credentials, query or fragment; localhost HTTP is allowed")
        token_env = config.setdefault("token_env", "WALLEX_MCP_BEARER_TOKEN")
        if not isinstance(token_env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", token_env):
            raise ValueError("token_env must name an uppercase environment variable")
    return config


@asynccontextmanager
async def connect(config, outputroot):
    """Open, initialize and enumerate an MCP connection; never persist authentication."""
    config = validate_config(config)
    official = config["transport"] == "streamable_http" and config.get("url") == OFFICIAL_URL
    http_receipts = {}
    authenticated = False
    try:
        async with AsyncExitStack() as stack:
            if config["transport"] == "stdio":
                env = {"WALLEX_MCP_OUTPUT_ROOT": str(Path(outputroot).resolve()), "PYTHONUNBUFFERED": "1",
                       "WALLEX_ENABLE_PRIVATE_READS": "1" if config.get("private_reads") else "0"}
                if config.get("private_reads") and os.environ.get("WALLEX_API_KEY"):
                    env["WALLEX_API_KEY"] = os.environ["WALLEX_API_KEY"]
                params = StdioServerParameters(command=config.get("python", sys.executable),
                                              args=["-m", "wallex_lab.wallex_mcp"], env=env,
                                              cwd=str(Path(__file__).resolve().parents[1]))
                streams = await stack.enter_async_context(stdio_client(params))
            else:
                if official:
                    key = os.environ.get("WALLEX_API_KEY", "") if config.get("private_reads") else ""
                    headers = {"X-API-Key": key} if key else {}
                    authenticated = bool(key)
                else:
                    token = os.environ.get(config["token_env"], "")
                    headers = {"Authorization": "Bearer " + token} if token else {}

                async def record_response(response):
                    if response.request.method == "POST":
                        try:
                            message = json.loads(response.request.content)
                        except (ValueError, RuntimeError):
                            return
                        if message.get("method") == "tools/call":
                            http_receipts[message["params"]["name"]] = {
                                "status": response.status_code,
                                "headers": {key: response.headers[key] for key in ("date", "content-type") if key in response.headers}}

                http = await stack.enter_async_context(httpx.AsyncClient(headers=headers,
                        timeout=httpx.Timeout(config["timeout_seconds"], connect=10), follow_redirects=False,
                        event_hooks={"response": [record_response]} if official else {}))
                streams = await stack.enter_async_context(streamable_http_client(config["url"], http_client=http))
            session = await stack.enter_async_context(ClientSession(streams[0], streams[1],
                        read_timeout_seconds=timedelta(seconds=config["timeout_seconds"])))
            initialized = await session.initialize()
            session._wallex_official = official
            session._wallex_outputroot = Path(outputroot).resolve()
            session._wallex_http_receipts = http_receipts
            session._wallex_authenticated = authenticated
            # ponytail: serialize calls per session to correlate HTTP receipts; use separate sessions for parallel reads.
            session._wallex_read_lock = asyncio.Lock()
            tools, cursor = {}, None
            while True:
                page = await session.list_tools(cursor=cursor)
                tools.update({tool.name: tool for tool in page.tools})
                cursor = page.nextCursor
                if not cursor:
                    break
            if official:
                session._wallex_outputroot.mkdir(parents=True, exist_ok=True)
                discovery = {"created_utc": utc_now(), "url": OFFICIAL_URL,
                             "initialize": initialized.model_dump(mode="json"),
                             "tools": [tool.model_dump(mode="json") for tool in tools.values()]}
                (session._wallex_outputroot / "discovery.json").write_text(json.dumps(discovery, ensure_ascii=False, indent=2), encoding="utf-8")
                for canonical, remote in OFFICIAL_TOOLS.items():
                    if remote in tools and (canonical in PUBLIC_TOOLS or config.get("private_reads")):
                        tools[canonical] = tools[remote]
            yield session, initialized, tools
    except Exception:
        raise RuntimeError("Read-only MCP connection or request failed; remote error details omitted") from None


async def call_read_only(session, name, arguments, available, private_reads=False):
    allowed = PUBLIC_TOOLS | ({"wallex_balances"} if private_reads else set())
    if name not in allowed or name not in available:
        raise ValueError("MCP tool is absent or outside the explicit read-only allowlist")
    official = getattr(session, "_wallex_official", False)
    remote, remote_arguments = name, arguments
    if official:
        arguments = dict(arguments)
        if name == "wallex_balances":
            if arguments:
                raise ValueError("Balance reads accept no arguments")
        else:
            if arguments.keys() - ({"symbol", "start", "end"} if name == "wallex_history" else {"symbol"}):
                raise ValueError("Unexpected public tool arguments")
            if name != "wallex_markets" or arguments.get("symbol"):
                validate_symbol(arguments.get("symbol"))
        remote = OFFICIAL_TOOLS[name]
        if name == "wallex_markets":
            remote_arguments = {}
        elif name == "wallex_history":
            start, end = arguments.get("start"), arguments.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= int(time.time()) or end - start > 86400:
                raise ValueError("History requires integer UTC epoch range of at most 86400 past seconds")
            remote_arguments = {"symbol": arguments["symbol"], "resolution": "1", "from": start, "to": end - 1}
    try:
        async with getattr(session, "_wallex_read_lock", asyncio.Lock()):
            started = utc_now()
            if official:
                session._wallex_http_receipts.pop(remote, None)
            result = await session.call_tool(remote, arguments=remote_arguments)
            received = utc_now()
            transport_receipt = session._wallex_http_receipts.pop(remote, {}) if official else {}
        if result.isError:
            raise RuntimeError("MCP tool reported failure")
        value = result.structuredContent
        if value is None:
            blocks = [block.text for block in result.content if getattr(block, "type", None) == "text"]
            if len(blocks) != 1:
                raise ValueError("Expected one JSON MCP result")
            value = json.loads(blocks[0])
        if not isinstance(value, dict):
            raise ValueError("Expected an object MCP result")
        if official:
            if name == "wallex_history":
                if value.get("s") not in {"ok", "no_data"}:
                    raise ValueError("Unsuccessful history response")
            elif value.get("success") is not True or not isinstance(value.get("result"), dict):
                raise ValueError("Unsuccessful Wallex response")
            if name == "wallex_balances":
                return {"balances": value["result"]["balances"], "authenticated": session._wallex_authenticated,
                        "fetched_utc": received, "persisted": False}
            raw = result.model_dump_json().encode("utf-8")
            key = os.environ.get("WALLEX_API_KEY", "") if session._wallex_authenticated else ""
            if key:
                raw = raw.replace(key.encode(), b"[redacted]")
            directory = session._wallex_outputroot / "raw"
            directory.mkdir(parents=True, exist_ok=True)
            body_path = directory / f"{time.time_ns()}_{remote}.mcp.json"
            body_path.write_bytes(raw)
            receipt = {"url": OFFICIAL_URL, "method": "MCP tools/call", "tool": remote,
                       "request_started_utc": started, "response_received_utc": received,
                       "authenticated": session._wallex_authenticated, "sha256": hashlib.sha256(raw).hexdigest(),
                       "bytes": len(raw), "body_path": str(body_path),
                       "body_representation": "SDK-decoded MCP CallToolResult JSON, serialized locally; credentials redacted when present",
                       **transport_receipt}
            body_path.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            if name == "wallex_markets":
                rows = value["result"].get("markets")
                if not isinstance(rows, list):
                    raise ValueError("Official market response requires result.markets")
                symbols = {row["symbol"]: {**row, "baseAsset": row.get("base_asset"), "quoteAsset": row.get("quote_asset"),
                    "stats": {old: row.get(current) for old, current in (("lastPrice", "price"), ("24h_ch", "change_24h"),
                        ("24h_volume", "volume_24h"), ("24h_quoteVolume", "quote_volume_24h"),
                        ("24h_highPrice", "highest_24h_price"), ("24h_lowPrice", "lowest_24h_price"), ("7d_ch", "change_7D"))}}
                    for row in rows if isinstance(row, dict) and isinstance(row.get("symbol"), str)}
                symbol = arguments.get("symbol")
                if symbol and symbol not in symbols:
                    raise ValueError("Symbol absent from official market metadata")
                value = {**value, "result": {"symbols": {symbol: symbols[symbol]} if symbol else symbols}}
            return {"payload": value, "receipt": receipt, "source": OFFICIAL_URL, "read_only": True,
                    "receipt_scope": "Receipt hashes the full SDK tool result before market symbol filtering"}
        return value
    except Exception:
        # Remote errors can echo bearer headers or private response bodies.
        raise RuntimeError("Read-only MCP call failed; remote error details omitted") from None


async def probe(config, outputroot, symbol="BTCTMN"):
    """Verify initialization, tool discovery and one actual public market read."""
    validate_symbol(symbol)
    config = validate_config(config)
    async with connect(config, outputroot) as (session, initialized, tools):
        market = await call_read_only(session, "wallex_markets", {"symbol": symbol}, tools)
        return {"initialized": True, "server": initialized.serverInfo.model_dump(),
                "protocol": initialized.protocolVersion, "tools": sorted({tool.name for tool in tools.values()}),
                "allowed_tools": sorted(PUBLIC_TOOLS & tools.keys()), "transport": config["transport"],
                "read_only": True, "symbol": symbol, "market": market, "fetched_utc": utc_now()}


async def call_read_tool(config, outputroot, name, arguments):
    """Open one real SDK session and call an explicitly permitted read tool."""
    config = validate_config(config)
    if name not in PUBLIC_TOOLS and not (name == "wallex_balances" and config.get("private_reads")):
        raise ValueError("Tool is outside the explicit read-only allowlist")
    async with connect(config, outputroot) as (session, _, tools):
        return await call_read_only(session, name, arguments, tools, config.get("private_reads", False))


async def market_snapshot(config, outputroot, symbol="BTCTMN"):
    """Return receipt-backed public envelopes for graph context and diagnostics."""
    validate_symbol(symbol)
    config = validate_config(config)
    async with connect(config, outputroot) as (session, initialized, tools):
        result = {"symbol": symbol, "transport": config["transport"], "server": initialized.serverInfo.name,
                  "read_only": True, "fetched_utc": utc_now()}
        for key, tool in (("market", "wallex_markets"), ("orderbook", "wallex_orderbook"), ("trades", "wallex_trades")):
            result[key] = await call_read_only(session, tool, {"symbol": symbol}, tools)
        result["fetched_utc"] = utc_now()
        return result


class MCPPublicClient:
    """Synchronous PublicClient.get contract using only official public MCP tools."""
    def __init__(self, root="data/mcp", timeout=25):
        self.root = Path(root)
        self.config = validate_config({"transport": "streamable_http", "url": OFFICIAL_URL, "timeout_seconds": timeout})

    def get(self, path, params=None, label="public"):
        params = dict(params or {})
        names = {"/v1/markets": "wallex_markets", "/v1/depth": "wallex_orderbook",
                 "/v1/trades": "wallex_trades", "/v1/udf/history": "wallex_history"}
        if path not in names:
            raise ValueError("MCPPublicClient permits only public markets, depth, trades and native history")
        allowed = {"symbol", "resolution", "from", "to"} if path == "/v1/udf/history" else {"symbol"}
        if params.keys() - allowed:
            raise ValueError("Unexpected public MCP parameters")
        if path == "/v1/udf/history":
            if params.get("resolution", "1") != "1" or type(params.get("from")) is not int or type(params.get("to")) is not int:
                raise ValueError("Public MCP history requires resolution='1' and integer from/to")
            arguments = {"symbol": params.get("symbol"), "start": params["from"], "end": params["to"] + 1}
        else:
            arguments = params
        result = asyncio.run(call_read_tool(self.config, self.root, names[path], arguments))
        return result["payload"], result["receipt"]
