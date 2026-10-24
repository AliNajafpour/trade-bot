# Persistent market council

The requested outcome is a spot paper-trading system with three independently refreshed evidence streams and one accountable manager. News/market context is normally analyzed every30minutes, slow macro every24hours, and technical signals every60seconds. All intervals are editable. Reports remain usable only until their explicit expiry or invalidation. The manager sees report versions, genuine technical probabilities, a chart image, current account state and its own earlier decisions; WAIT creates a durable future review instead of a sleep or an implicit buy.

This extends the existing MCP, provider, LangGraph, source-receipt and PaperBroker modules. Existing stopped studies and active paper-account monitors are not restarted or overwritten. External Trade-Bot files/environments remain read-only.

## Alternatives and choice

A single LLM doing everything every minute repeats slow news analysis and cannot supply calibrated first-touch probabilities. Three independently timed reports with a persistent manager reuse the current architecture and isolate slow sources. A fully autonomous research/training swarm at runtime adds latency and makes decisions hard to reproduce. Use the second approach, with training as a separate reproducible job.

## Components and interfaces

### Public evidence: wallex_lab/intelligence_sources.py

`collect_bundle(root, kind, asof, previous=None, config=None) -> dict`, kind=`market|macro`.

Result fields: `kind`, `available_at`, `facts`, `news`, `sources`, `coverage`. Facts have a stable `id`, `metric`, `value`, `unit`, `source`, `source_url`, `observed_at`, `available_at`, and an optional provider `measurement_at`. News retains publisher time, first observation, source URL and deduplication identity. All raw responses/receipts are retained. Provider delay, market closure and failed coverage are explicit; collection time never masquerades as quote time.

Market scope: gold/silver, major stock indices, volatility, dollar and oil references plus crypto/global news. Use documented public sources, with explicit proxy labels if a futures/ETF quote stands in for a spot commodity or index. Macro scope: inflation, policy rates, employment, yields/liquidity and official policy/release news. Revised observations retain first-observed/revision history. No paid API is silently assumed and no missing metric is filled with a neutral value.

### Periodic analysis: wallex_lab/intelligence_reports.py

`ReportStore(root)`, `refresh(kind, bundle, asof, interval_seconds, provider=None) -> dict`, `get(kind, asof) -> dict|None`, `invalidate(kind, reason, asof)`.

A real model response contains a narrative, directional/risk assessment, confidence, recommended risk multiplier, evidence IDs and invalidation conditions. The envelope includes `report_id`, `kind`, `generated_at`, `valid_until`, `status`, `assessment`, `coverage`, `source_bundle_sha256`, and the genuine provider receipt. Reports are immutable history plus an atomic current pointer. Expired/invalidated reports are never silently accepted. Invalid or failed model responses remain failures. Market and macro schedules are independent and survive restart.

### Learned technical engine: wallex_lab/council_technical.py

`TechnicalEngine(model_dir).predict(candles, symbol, asof, quote, candidates=None) -> dict`.

Use genuine fitted neural/sequence features and an observed-outcome head conditioned on the proposed stop/target distances, horizon and market features. Train and calibrate on disjoint chronological data, with label overlap purged across boundaries. Use liquid public spot reference data; Wallex candles/quotes remain the execution source and local/global basis is explicit. Never claim named-pattern accuracy without labeled validation. The visual manager can make source-bound pattern observations from actual supplied charts.

Output includes `status`, `model_id`, `model_available_at`, `signals`, diagnostics/provenance. Each signal has `signal_id`, `symbol`, `reference_price`, `stop_price`, `target_price`, `p_tp_first`, `p_sl_first`, `p_unresolved`, `horizon_minutes`, `expected_marked_net_return`, `net_reward_risk`, and a geometry identity. Probabilities refer to reaching those exact levels within the stated forecast horizon. Unresolved positions are not labeled as realized wins. A forecast horizon does not impose a timed exit. Manager changes to barriers require a new call with `candidates=[{stop_price,target_price,horizon_minutes}]`; old probabilities cannot be carried over.

Training writes immutable data/source receipts, weights, training/validation curves and held-out probability metrics. Actual training and validation limits are reported; more epochs are not treated as proof of quality. A one-minute signal update may reuse a higher-timeframe representation while incorporating new completed-minute data and current executable quotes.

### Vision/provider transport: wallex_lab/llm.py and providers.py

Extend the existing callable contract with optional `images=None`. Default text-only callers retain their interface. Actual chart pixels are passed to Codex CLI or the selected compatible vision API; paths alone are not image input. Cache identities include image hashes, and original chart bytes/receipts are retained. No key is written to prompts/receipts. A provider without vision support fails explicitly. Render real completed OHLCV charts with timestamp, timeframe, quote currency and proposed levels.

### Persistent manager: wallex_lab/market_council.py

Use LangGraph + SQLite checkpointing for prepare -> manager -> validate/rescore -> persist. Inputs include valid market/macro report IDs, technical signals, chart attachments, fresh quotes, open positions, recent realized results and prior manager decisions. Outputs include an explanation, normalized technical/market/macro weights and per-symbol actions: enter, wait, reject, hold, close, modify. Weights may change by decision and must explain the change; expired sources cannot be given an active weight.

WAIT records `review_at`, its original plan ID and expiry in durable state. At due time the manager receives the previous instruction, intervening history and fresh reports/signals/chart; it does not automatically enter. Repeated waits expire by configured plan lifetime. Restarts load pending reviews, not duplicate orders.

All proposed price changes are rescored by the technical model, then checked independently for cash, fees, market eligibility, quote age, allowed exposure and stop risk. Bullish/bearish assessments are allowed, but execution is cash-funded long spot only. Bearish means reject buys, reduce/close held inventory or keep cash. No short/borrow/leverage tools are enabled.

The protective exit loop runs independently of model inference and report refresh. Existing protections remain active when a model or source fails. Amendments are idempotent, atomic and cannot exceed the configured account risk budget. No forced trade quota or fake fallback opinion exists.

### Runtime and review

`python -m wallex_lab.market_council --config config/market-council.example.json --seconds N` uses a fresh run root, configurable provider environment, official MCP public data and paper-only broker. Configuration is reread so cadence changes apply without editing code. Every report, chart, technical signal, model response, pending review, fill and rejection is retained. A read-only local status surface exposes freshness, source coverage, next review and account state.

## Validation

Offline checks cover report TTL/change/restart, source failures/revisions/zero volume, candidate-specific probability rescoring, training chronology, real-image transport/cache invalidation, five-minute due review after restart, stale-report rejection, normalization of weights, spot-only actions, protection changes and idempotent execution. A bounded live smoke verifies real public sources, real LLM reports and a chart-bearing manager request. Its trading/PnL is reported without any profitability claim. Stopped historical studies remain stopped.
