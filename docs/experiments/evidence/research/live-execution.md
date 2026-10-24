# Credit-account execution preparation

Status: **disabled**. This module is separate from the running paper broker. Its execution monitor has not been started or registered with the recovery supervisor. The default CLI command is read-only.

The selected account is the credit/trading subaccount associated with the local `WALLEX_API_KEY`. The key stays in `.env` and is sent only as `X-API-Key` to the official MCP endpoint. The persisted account binding is a fingerprint, not the key. Changing the key requires another read-only preflight and a separate ledger; this implementation deliberately does not guess whether two keys select the same account.

## Verified reads

Run from the repository root:

```powershell
.venv-portable/Scripts/python.exe -B -m wallex_lab.live_execution --config config/live-execution.json
```

This command only reads. It writes the local account snapshot, official market-limit receipts, and `runs/live-execution-20261023/preflight.json`. It does not enable trading. On 2026-10-23 the credit balance read returned 18.24 available USDT and zero open orders. Historical single-order detail was also checked for the fill/fee fields without placing an order.

MCP supplies credit balances, authenticated per-market fees, open orders and orderbooks. Its market listing omits minimum sizes, so the existing public client reads `https://api.wallex.ir/v1/markets`, caches limits for at most five minutes and cross-checks precision against MCP on each snapshot. No main-wallet balances or synthetic TMN conversions enter the execution budget. SOLUSDT and SUSDT currently report market-order support disabled; this gateway uses LIMIT and STOP_LIMIT only.

## Execution interface

`Gateway.prepare_entry(...)` persists an intent without sending it. The caller supplies the manager's quantity, risk multiplier and freshly scored executable levels. The gateway enforces the existing Weekly/late policy: rounded TP +2%, SL -2%, probability horizon 1440 minutes, absolute TP-first probability above 55%, and no unsupported model score. A rounded or changed entry/barrier needs a fresh score.

`await Gateway.execute_entry(id)` submits once, obtains detailed fills, cancels any unfilled entry remainder and protects the confirmed owned quantity. Creation acknowledgements alone are never treated as fills. `await Gateway.reconcile()` recovers uncertain outcomes by client ID without resending a timed-out POST. `await Gateway.close_position(parent, reason)` reconciles, settles/cancels existing children, sells only owned available inventory at the visible bid, cancels the LIMIT remainder and restores protection for remaining inventory. A partial fill below exchange minimum or unconfirmed protection raises an error and prevents normal admission of further entries.

`await Gateway.monitor_once()` recovers pending entries/protection, checks owned positions against fresh executable bids, attempts exits at TP or breached SL, and only then processes prepared manager intents. The optional `--monitor` actor repeats this every 15 seconds, stops on an uncertain outcome, and persists its status. It refuses to start with the current disabled/unverified-stop configuration. It has only been tested against the fake exchange. It does not run technical/news inference itself: the manager must receive the real account snapshot and owned positions, produce a fresh decision and call `prepare_entry`; paper decisions must not be replayed as real orders.

Dispatch, reconciliation and the complete cancellation/protection lifecycle share the existing kernel file lock. SQLite FULL/WAL persists intent identity before network submission. Prepared children superseded by a close are abandoned locally; active or uncertain sibling sells prohibit another sell. Foreign holdings are never adopted, sold, or silently canceled. Unmanaged open orders block new entries.

Exits prevalidate executable quotes and venue minimums before removing protection. If a later read or local validation fails before a possible exit submission, the gateway attempts to restore protection before raising. It never restores a second SELL across an uncertain exit POST. If recovery also fails, the monitor records a halt; it cannot promise protection during an exchange outage.

The agent may choose sizing within these initial limits: 15% allocation per position, 0.5% planned stop risk per trade, 2% aggregate planned stop risk, and four positions. These are planning limits, not guarantees against price gaps. Fees come from the authenticated account. Calculations include a 20 bps stop-limit execution reserve; this reserve is a configurable assumption, not an exchange fee or guaranteed fill price.

## Intentionally unarmed

Both `config.enabled` and the dedicated transport's `allow_writes` must be explicitly true before a mutation can occur. `stop_capabilities_verified` is also false. The preflight never changes those flags. The public/read-only MCP client's allowlist is unchanged.

Native STOP_LIMIT acceptance, returned trigger fields and actual triggering have **not** been verified with a live order. A stop-limit can remain unfilled after a gap. There is no claimed atomic OCO bracket: target exits require the execution actor to be running, while a confirmed native stop would reside at the exchange. The current paper loop must not be switched to this gateway as a drop-in broker: its simulated-fill semantics differ from exchange acknowledgements. Enabling unattended real execution still requires the real-account manager feed, startup supervision and live stop behavior to be checked after the user authorizes starting. No existing paper position is migrated into a real holding.

## Offline verification

```powershell
.venv-portable/Scripts/python.exe -B checks/check_live_execution.py
```

The fake exchange tests disabled writes, repeated and conflicting intents, pending exposure, foreign orders, concurrent protection, accepted-but-timed-out entry recovery after restart, partial entry cancellation, owned inventory, rejected stops, stale open-order observations, superseded exits and a stop filling during cancellation. They never load credentials or use the network. Passing them does not establish live STOP_LIMIT behavior or profitability.

Official references: [MCP guidance](https://api.wallex.ir/llm.txt), [API documentation](https://api-docs.wallex.ir/).
