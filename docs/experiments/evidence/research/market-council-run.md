# Run the market council

The council uses a persistent $100 USDT paper spot account in `runs/market-council-20261022`. It reads official Wallex MCP public books and native candles, real liquid global reference candles, and independently refreshed market and macro reports. The example configuration uses the existing external GPU Python environment read-only for technical inference. It does not launch or restart earlier studies.

Run from the project directory in PowerShell:

```powershell
.\.venv-portable\Scripts\python.exe -m wallex_lab.market_council --config config/market-council.example.json --seconds 300
```

Omit `--seconds` for continuous operation. Stop with Ctrl+C. A per-run OS lock prevents two runtime processes from managing the same ledger. Keep the same run directory when resuming a stopped council; the ledger, reports and pending reviews persist. Use a new directory for a separate account. Never point the council at an earlier study or another monitor's directory.

For a hidden background process, request a safe stop from another terminal:

```powershell
.\.venv-portable\Scripts\python.exe -m wallex_lab.market_council --config config/market-council.example.json --stop
```

The provider defaults to the current Codex login. For an explicitly configured compatible API, set `LLM_PROVIDER=openai_compatible`, `LLM_API_KEY`, `LLM_MODEL`, and optionally `LLM_BASE_URL` in the launching process environment. Keep credentials out of configuration files. A missing provider, unavailable model or failed source remains a recorded failure. There is no synthetic opinion or required trade count.

Open [local paper status](http://127.0.0.1:8784/) while running. [The JSON status](http://127.0.0.1:8784/api/state) includes account state, pending reviews, quote provenance, report expiry and coverage, worker errors, last scan and last manager cycle. It binds only to `127.0.0.1`, accepts read requests, and exposes no order or configuration action. Set `port` to `0` to disable HTTP. The same status is saved in `runtime-state.json`.

## Scheduling and edits

The runtime rereads the configuration during operation. `quote_seconds` defaults to 10, `technical_seconds` and `manager_seconds` to 60, `market_seconds` to 1800 and `macro_seconds` to 86400. A durable WAIT review becoming due also triggers manager work independently of a longer ordinary manager cadence, subject to a completed scan and the single manager worker. Failed review attempts wait for the shorter of one minute and the ordinary manager cadence before retrying. Valid cadence and source configuration edits apply without a restart. Shorter report cadence triggers earlier refresh; increasing cadence cannot extend an existing report's saved expiry. Invalid JSON or unsafe changes keep the last valid configuration and appear in status.

The run directory, initial balance, symbols, model directory, technical Python, provider selection and HTTP port require a restart to change. Initial balance cannot reset an existing ledger. Risk configuration is passed to the manager's independent admission checks. Paper execution remains cash-funded long spot; bearish opinions may avoid buys or reduce held inventory.

Quote collectors run independently for each symbol. The protective loop processes the newest completed books independently of report collection, technical inference and manager inference. There is one technical scanner worker, one manager worker and at most two report workers. The first manager cycle waits for a completed scan. A slow worker retains its running status and cannot create overlapping copies of itself.

Reference-candle collectors also run independently per asset, so an unavailable venue for one asset does not age every asset's data. Signals target one-minute refresh; slow network/model work remains visible rather than generating invented updates. Fresh report versions or restoration of technical coverage can revisit an earlier WAIT before its timer, while preserving its original lifetime. Flat assets without usable technical data are shown in coverage and do not trigger repeated empty manager calls. The last actual decision stays in status when the current cycle is waiting.

The manager receives the best six scored candidates per eligible asset and an actual three-timeframe chart with the leading candidate overlaid. Chart price axes fit observed candles, so distant targets cannot flatten the visible pattern. All 80 candidate scores remain in technical history. Named-pattern accuracy and profitable execution have not been established by this implementation.

## Price and data provenance

Wallex executable bid and ask come from `https://api.wallex.ir/mcp` through the existing read-only MCP client. Quote admission checks the original request/receipt clock and HTTP Date; the exchange event timestamp is unavailable. Receipt time is never presented as exchange event time. Stale, crossed, empty or failed books are rejected. Raw MCP responses and HTTP receipts remain under the run directory.

Technical reference data first tries Binance public spot `GET /api/v3/klines` at `https://data-api.binance.vision`, then `https://api.binance.com`. If those fail, it uses documented KuCoin public spot `GET /api/v1/market/candles` at `https://api.kucoin.com` with the same USDT pair. The actual working venue is explicit in each snapshot. A successful venue preference persists for 24 hours to avoid repeatedly waiting on failed endpoints. Caches remain separate by venue, so switching cannot silently splice different venues into one price series.

The collector downloads three days of completed one-minute bars initially, then requests incremental pages. It retains all returned raw HTTP bytes/receipts, local CSV caches, immutable scan snapshots and provenance. It removes incomplete minutes using the earlier of the requested cutoff and request-start clock, validates OHLCV and preserves gaps and zero volumes. See the [Binance market-data documentation](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market) and [KuCoin public candle documentation](https://www.kucoin.com/docs-new/rest/spot-trading/market-data/get-klines).

Reference OHLCV and Wallex native candles are separate files. Native Wallex candles are retained as observed, including sparse or zero volume. Global reference prices and volume describe a different venue. Every reference record labels this basis explicitly; no unverified conversion or basis adjustment is applied. Only Wallex books can supply executable paper prices. A model trained on Binance candles and applied to KuCoin candles also has unvalidated reference-venue transfer risk.

Live source collection uses a completion cutoff after network requests. Report generation begins with an as-of time at or after source availability. Source failures and incomplete coverage remain visible, including when there is no usable report.

## Stop policy and verification

Stopping prevents further scheduling and sets the manager's stop event before another entry or amendment can be admitted. Existing paper positions remain open with their saved stop and target. No quote protection runs after this process stops. A restart reloads those positions and resumes protection with fresh public quotes. A model forecast horizon is not a timed liquidation policy.

Workers are daemon threads so an unfinished model cannot hold up `Runtime.stop()`. Source collection uses bounded HTTP timeouts and its existing bounded thread pool; Python may wait for in-flight source requests during interpreter shutdown. The manager's subprocess/provider timeouts still bound their work. No stop method can cancel an already completed paper fill.

Run the focused offline check:

```powershell
.\.venv-portable\Scripts\python.exe checks/check_council_runtime.py
```

It uses injected offline collectors and a deliberately blocked manager to verify that quote protection continues, failed source analysis remains visible, cadence edits take effect, unsafe initial-balance edits are rejected, and stop returns without waiting for the blocked model. A bounded live smoke requires actual public source access, trained model artifacts and the selected model provider; its PnL is an operational observation, not a profitability claim.
