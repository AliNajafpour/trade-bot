# Iranian Wallex MCP connection

Checked 2026-10-22. The official Iranian Wallex MCP endpoint is [https://api.wallex.ir/mcp](https://api.wallex.ir/mcp), explicitly identified by Wallex's [machine-readable instructions](https://api.wallex.ir/llm.txt). The project connects to it using the official Python MCP SDK and actual initialize, tools/list and tools/call messages. The earlier search did not locate this endpoint; the user supplied it and direct protocol verification confirmed it.

The local stdio adapter over the Iranian REST API remains available as an alternative. It is a project adapter, distinct from the official hosted MCP service. The active paper trial was not restarted or changed by these connection checks.

## Primary sources

| Source | Verified information and use |
| --- | --- |
| [Wallex machine-readable instructions](https://api.wallex.ir/llm.txt) | Identifies the canonical MCP URL. Public initialization, discovery and public tool calls need no API key. Private operations use `X-API-Key`, never a key in prompts, URLs or arguments. State-changing calls require immediate explicit confirmation. The document was fetched directly and saved as `runs/portable-mcp-official/llm.txt`. |
| [Official MCP endpoint](https://api.wallex.ir/mcp) | Direct initialize reported server `api.wallex.ir`, version `1.0.0`, protocol `2025-11-25`. The verified discovery advertised 52 tools, including state-changing operations. The complete schema is saved in `runs/portable-mcp-official/verified/discovery.json`. |
| [Wallex developer documentation](https://developers.wallex.ir/) | Iranian cryptocurrency exchange API. Its examples target `api.wallex.ir`. |
| [Spot markets](https://developers.wallex.ir/spot/activemarket) | Documents public `GET /hector/web/v1/markets` with market activity and precision metadata. |
| [Official earlier API reference](https://api-docs.wallex.ir/) | Documents `/v1/markets` statistics, `/v1/depth` book, `/v1/trades` recent trades and `/v1/udf/history` OHLCV. The adapter retains these existing project endpoints. During research the full page timed out; search-indexed official content confirmed history, while saved project receipts provide separate live evidence. |
| [Latest trades](https://developers.wallex.ir/basicservices/latesttrades) | Documents `/v1/trades`, its symbol filter and result.latestTrades. Repeated polling can trigger venue rate limits; latest trades are not a complete fixed-duration tape. |
| [Main account balances](https://developers.wallex.ir/basicservices/mainaccountwallet) | Documents `GET /v1/account/balances`, `x-api-key`, and result.balances. |
| [Official Python SDK, v1 maintenance branch](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x) | Real SDK with ClientSession, FastMCP and supported stdio/Streamable HTTP transports. The SDK recommends a `<2` bound when retaining v1 APIs. This project pins v1.30.0 for reproducibility. |
| [Official v1 client guide](https://github.com/modelcontextprotocol/python-sdk/blob/v1.x/docs/client.md) | Demonstrates stdio_client, ClientSession.initialize(), list_tools(), call_tool(), and streamable_http_client with a configured httpx.AsyncClient. |
| [Wallex Asia concepts](https://docs.wallex.asia/docs/concept) | Describes a cross-border payments Partner API and its account model. It is a different service and is not an API source for this Iranian exchange adapter. |

## Running and calling

Select the official endpoint with `{"transport":"streamable_http", "url":"https://api.wallex.ir/mcp"}` in the `mcp` configuration. Public mode sends no credentials even when API key or bearer environment variables exist. No extra configuration field is required for official tool mapping.

The optional local server starts with `.venv-portable/Scripts/python.exe -m wallex_lab.wallex_mcp`. Standard output is reserved for MCP JSON-RPC. The stdio client launches this process itself, using the SDK's Windows process helper with `CREATE_NO_WINDOW`.

Python entry points in `wallex_lab.mcp_client` are async:

```python
config = {"transport": "streamable_http", "url": "https://api.wallex.ir/mcp"}
await probe(config, "runs/mcp-probe", "BTCTMN")
await market_snapshot(config, "runs/mcp-snapshot", "BTCTMN")
await call_read_tool(config, "runs/mcp-history", "wallex_history",
                     {"symbol": "BTCTMN", "start": 1792540800, "end": 1792627200})
```

The empty config still selects local stdio with the current Python executable. `python` can select another installed interpreter. `timeout_seconds` defaults to 120 and is bounded to 1 through 300. Unknown fields, including inline credentials, are rejected. Market, book, trades and history return a `payload` plus a `receipt`. Each helper opens a fresh protocol session.

The official tool names and arguments were discovered live, not guessed:

| Project read tool | Official tool | Translation |
| --- | --- | --- |
| `wallex_markets` | `wallex_basic_get_markets` | Send `{}`. Convert returned `result.markets` rows into a `result.symbols` dictionary filtered by the requested symbol. Rows retain their original fields such as `price`, `base_asset` and `quote_asset`. |
| `wallex_orderbook` | `wallex_basic_get_market_depth` | Send the symbol; retain the original `result.bid` and `result.ask`. |
| `wallex_trades` | `wallex_basic_get_latest_trades` | Send the symbol; retain `result.latestTrades`. |
| `wallex_history` | `wallex_basic_candle_history` | Send symbol, `resolution="1"`, `from=start`, `to=end-1`. Retain original UDF arrays. |
| `wallex_balances`, explicit opt-in only | `wallex_basic_get_account_balances` | Send `{}`. Return balances in memory without saving the result. |

Official public reads save the SDK-decoded `CallToolResult` JSON and SHA-256 receipts. These are explicitly labelled as locally serialized SDK results, not HTTP wire bytes. The receipt hashes the complete result before market filtering. Real HTTP response status, Date and Content-Type come from an httpx response hook. Local times bracket the tool call; no server Date or successful status is invented when unavailable. Calls within one session are serialized to match each call to its HTTP receipt. Existing quote validation checks local receipt timing and actual server HTTP Date. The SDK handles transport envelopes; the local stdio adapter continues saving original REST response entity bytes through PublicClient.

History only accepts strict integer UTC epoch seconds, a past interval no longer than one day, and an exclusive end. It always requests native resolution `1`. Callers must reuse `normalize_history` and complete-bucket aggregation before trading analysis. The existing acquisition code observed misleading `no_data` across larger ranges and minute bars under other resolution labels. No missing bars are invented.

Generic hosted MCP remains supported with `{"transport":"streamable_http", "url":"https://verified-host.example/mcp", "token_env":"WALLEX_MCP_BEARER_TOKEN"}`. The URL must have no user information, query or fragment. HTTPS is required except for loopback development. This generic template is not a real endpoint. Generic servers must expose the project's canonical tool names and envelopes. Their bearer token is read from the named environment variable. Wallex API keys are never forwarded to generic hosts, even with `private_reads=true`. Official translation and `X-API-Key` handling activate only for the exact canonical Wallex MCP URL.

## Read-only boundary and credentials

The client public allowlist is `wallex_markets`, `wallex_orderbook`, `wallex_trades` and `wallex_history`. The official service advertises orders, transfers and account operations, but the project client refuses to call them. Official discovery returns null annotations, so fixed tool names and inspected schemas determine the allowed mapping. The local adapter exposes no state-changing tools.

Private balance reads require `private_reads: true` in the client config and `WALLEX_API_KEY` in the environment. At the exact official URL the client sends it only in `X-API-Key`. For local stdio the client explicitly passes the key to its child with `WALLEX_ENABLE_PRIVATE_READS=1`; direct local server launch requires those variables. Private balance values remain in memory. No private call runs as part of probe or market_snapshot. Error messages omit remote bodies and headers; public receipt headers include only Date and Content-Type. Public saved SDK results redact the configured exchange key if an authenticated server response happens to repeat it.

The API key is different from a remote MCP bearer token. Do not put either in command arguments, repository files, configuration JSON, receipts or reports. Use a read-only exchange key for the optional balance tool.

The offline check is `.venv-portable/Scripts/python.exe checks/check_wallex_mcp.py`. It passed after adding official mappings. Its separate stdio fixture completes a real SDK handshake, lists and calls tools, then checks the allowlist, malformed symbol/history rejection, official history argument conversion, market normalization, raw receipt writing, host-specific authentication and safe failures. Fixture payloads exist only in the check script; production has no fake-data switch.

## Live verification

`runs/portable-mcp-official/verified/validation.json` records a successful real official connection on 2026-10-22 at approximately 10:46 UTC. Probe, snapshot and history calls used public tools without credentials. The checks confirmed server identity and protocol, the four allowed public mappings, five hashed public receipts with real HTTP 200 and Date headers, 1,440 closed native minute candles and 287 complete five-minute bars. Two partial edge buckets were rejected. The normalized BTCUSDT quote passed existing freshness validation with an actual HTTP Date, bid `82111.70` and ask `82138.94`. These are point-in-time observations, not current prices. Zero private or state-changing calls were made.
