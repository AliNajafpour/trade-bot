# Data contracts and provenance

## Execution prices

The numerical model and the execution quote answer different questions. Wallex executable quotes contain bid, ask and displayed size. The official MCP URL is `https://api.wallex.ir/mcp`; public REST is a fallback where supported. The last-trade price shown on an exchange page can differ from either executable side.

Original request/receipt timestamps and HTTP Date are retained. Exchange matching-engine event time is not available through these responses, so receipt freshness is not described as event freshness. Missing, crossed, stale or empty books block execution.

## History

Native candles use UTC **open** timestamps and become usable only when fully completed. Expected columns are `timestamp,open,high,low,close,volume`. Validation checks ordering, duplicates, finite/positive prices, nonnegative volume, OHLC bounds and interval continuity. Aggregation requires complete source buckets; gaps are not interpolated or silently replaced with flat candles.

Training and historical evaluation use frozen source manifests and SHA-256 receipts. Raw data and those large caches are deliberately excluded from this repository. The bundled final model retains its training metadata and member hashes.

The current live paper experiment can use a coherent KuCoin window, or a coherent MEXC window for SOL. The whole OHLC window is scaled to the same-decision completed Wallex anchor. Volume remains reference-venue volume. Switching venue does not stitch individual missing candles from different exchanges. This provides an operational input path, **not** validated transfer of probability calibration from native Wallex history.

## News and macro

Public RSS/news and cross-asset data pass through `intelligence_sources.py`. Reports contain source IDs, measurement and receipt information, coverage failures, generated time, validity deadline, confidence, risk multiplier and invalidation conditions. The report is advisory evidence, not order authorization.

Financial time series may be delayed, revised or represent a closed-market last observation. Publication, measurement and local availability are distinct. A missing inflation or rate series does not become a neutral observation. News text is untrusted input; it cannot supply system instructions or relax trading controls.

Chronos numeric covariates include past-only aligned TVL-derived fields and known future calendar features. Historical TVL dates alone do not prove original publication availability. Retrospectively assumed lags are explicitly labeled and are not interchangeable with prospectively captured observations.

## Fees and costs

`config/wallex-official-fees.json` and `docs/fee-evidence/public-structured-fee.json` preserve a public schedule snapshot and its body hash. Public schedule values are not proof of a particular authenticated account's current tier. The real gateway queries authenticated per-market fees.

The sample paper account uses a declared fee scenario. Historical backtests add separately declared spread/slippage friction. A modeled friction allowance is not an exchange commission or a guaranteed achievable price. Keep fee rate units explicit: one basis point is `0.0001` as a fraction, or `0.01%`.

## Research boundaries

- No future candle appears in a model input window.
- Barrier queries and entry offsets are determined before reading future outcomes.
- A same-minute TP/SL ambiguity is not assigned a known causal order.
- Missing held-position paths invalidate affected full-period account results; null is not zero loss.
- Open-position marks remain separate from realized profit.
- HTML demo data is wholly synthetic and never a source for model evaluation.
