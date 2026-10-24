# Data and label improvements for arbitrary first-hit probabilities

Research date: 2026-10-23. Read-only investigation of existing data and primary sources. No training run, numerical source, label array or live job was changed. Recommendations below are hypotheses to test, not measured model improvements.

## The first priority is the meaning of the observed price

The current task is to estimate TP-first, SL-first and neither-hit probabilities at exactly 5, 15, 60, 240 and 1440 minutes for arbitrary supplied barriers. Conditional TP probability given a hit is a separate derived output. Keep the horizon as an observation deadline. Do not force an exit or invent a hit at that deadline. Minute data can remain an input and label-resolution source without adding a one-minute prediction output.

The dataset has many rows but relatively little independent market history. More importantly, most returned minutes contain no volume. Verify what the Wallex chart feed represents before increasing model complexity or declaring its calendar coverage adequate for short-horizon inference.

Local evidence comes from `data/barrier-dense-20261022/dataset.json`, `data/barrier-history-dense-20261022/market/*_1m.csv`, their quality reports, `wallex_lab/data.py`, `wallex_lab/barrier_dataset.py` and `docs/barrier-survival-dense.md`. The manifest declares public `https://api.wallex.ir/v1/udf/history`, native resolution `1`, UTC candle-open timestamps and fully completed bars. The 365-day interval runs from 2025-10-22 16:00 UTC through the bar opening at 2026-10-22 15:59 UTC.

| Symbol | Returned M1 rows | Zero-volume minutes | Zero-volume share | Flat high=low share |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 525,239 | 395,646 | 75.3269% | 89.9600% |
| ETHUSDT | 524,996 | 434,927 | 82.8439% | 93.5678% |
| SOLUSDT | 525,107 | 450,074 | 85.7109% | 95.4994% |
| SUSDT | 524,887 | 500,289 | 95.3137% | 99.1442% |

These are fresh full-file counts, not estimates from samples. There are no zero-volume nonflat bars. Among zero-volume bars, close differs from the previous returned bar on zero BTC, zero SOL, zero S and three ETH observations. A change across a missing-minute gap is not by itself an inconsistency. This pattern is consistent with many carried prices during inactive minutes, but does not establish the exchange's construction rule. No-trade, unchanged quote, exchange halt and failed collection need different flags. A zero-volume chart bar is not evidence that an executable bid and ask stayed fixed for that minute.

The files contain 2,100,229 returned bars and 2,171 missing calendar minutes. Those measures must be reported alongside activity. The training dataset has 414,289 five-minute origins and 128 geometries per origin, or 53,028,992 requests. Mirrored barriers and overlapping origins share paths. The common week-history cohort has 358,517 origins; strict continuity leaves just five UTC days for early calibration and twelve for the late test. Adding geometries does not add independent trading regimes.

### SUSDT identity is verified, historical identity still needs versioning

The saved official MCP market response identifies `SUSDT` as base asset `S`, English name `SONIC`, Persian name Sonic, quote `USDT`, spot enabled and margin disabled. Its `created_at` is 2025-01-14 and current recorded `price_precision` is 4. This is Sonic, not Sui, a dollar stablecoin, or a guess based on ticker spelling.

Evidence is `runs/portable-mcp-official/verified/raw/1792666017577343600_wallex_basic_get_markets.mcp.json`, SHA-256 `bb414c7548ad0d278af9e5a69841f325242f6a604356c0a387d6b5f465afe6a2`. The adjacent receipt records official `https://api.wallex.ir/mcp`, tool `wallex_basic_get_markets`, HTTP 200, request/receive UTC times and server Date on 2026-10-22. This is an SDK-decoded response serialized locally, not HTTP entity bytes. The response hash was recomputed during this investigation.

Current metadata does not prove the instrument's historical tick size, corporate-action history or contract mapping on another venue. Before pooling older Sonic history, store effective-dated instrument IDs, listing dates, redenominations, migrations, tick sizes and contract multipliers. Do not silently splice a predecessor ticker or map spot and perpetual contracts as identical instruments. If four decimal places correspond to the recorded S price lattice near 0.0527, one 0.0001 increment is about 19 basis points. The dataset permits barriers near one basis point. That conditional calculation makes lattice-aware label diagnostics necessary; current precision is not proof of the historical increment.

## Ranked acquisition and labeling proposals

| Rank | Addition | Why it comes first | Expected horizon relevance and limit |
| --- | --- | --- | --- |
| 1 | Define chart-price, trade-price and executable-quote targets separately; reconstruct a small same-venue trade sample against UDF bars | The target can otherwise change without anyone noticing | All horizons; most urgent at 5/15m and narrow barriers. Executable fill claims require quote/depth evidence |
| 2 | Collect ordered trades, best bid/ask and top-of-book size at the target venue, with sequence and receive clocks | Resolves many both-hit minutes, identifies stale last trades, measures spread and available liquidity | Highest priority at 5/15m. A recent-trades endpoint cannot guarantee a complete year of tape |
| 3 | Add more independent dates and point-in-time instruments; retain a target-venue holdout | Four related assets over one year do not cover broad regimes | All horizons, especially 4/24h. More symbols in the same shock are not independent episodes |
| 4 | Add synchronized major-venue spot/perpetual prices, trades and market returns as context | Measures common shocks, venue basis and delayed local adjustment | Useful candidate at 5m through 24h; another venue's touch must never overwrite the Wallex label |
| 5 | Add realized funding history, lagged OI, mark/index premium and spot-perpetual basis | Tests positioning and financing state beyond local OHLCV | Prioritize 60m/4h/24h; no empirical guarantee, and published OI/basis history may be short |
| 6 | Add replayable multi-level order-book events where history actually exists | Supports depth, cancellations, order-flow imbalance and liquidity resilience | Candidate at 5/15m. Historical full LOB is materially harder than downloadable trades |
| 7 | Add versioned liquidation observations | Tests forced-flow regimes and stress asymmetry | Event-driven candidate; public stream selection and price semantics must be retained |

This order favors label credibility over adding another hundred technical indicators. Cheap features from existing data can help diagnose the problem now: elapsed time since last positive-volume bar, inactive-minute fraction, trade-active duration, flat-run length, price-change frequency, gap age, and barrier distance in price increments. A bar with positive volume is not itself a trade count. Do not call OHLCV-derived volume imbalance order-flow imbalance unless aggressor direction is actually observed.

### Executable and trade labels

For a trade-price first passage task, replay exchange-ordered trades after the decision and record the earliest level crossing. Use sequence/trade ID as the tie-breaker supported by the feed. Coarse equal timestamps without ordering information remain ambiguous. Aggregated trade records may group fills; do not assume they preserve every distinct event. Trade observations recover transaction ordering, not a historical executable spread.

For a small immediate long position, entry uses the ask and sale uses the bid; a short uses the bid for entry and ask for repurchase. A size-specific target needs depth consumption, latency, fees and any price/quantity rounding rules. These are different labels from a completed close with a predetermined offset. A resting TP limit also needs queue/fill treatment, so quote touch alone does not prove that a limit order filled. Stop trigger price, trigger source and resulting fill can differ. State whether each probability predicts a trigger, a quote crossing or a completed fill.

If the user supplies a hypothetical entry, keep that supplied price in the query contract; do not silently replace it with a quote. Report the quote's age and distance from that entry. If modeling a pending entry order, the process also includes whether and when entry fills. That is a separate task, not an accidental extra outcome in the three-way first-hit output.

The existing M1 labeler correctly keeps unresolved same-minute double touches as unknown cause and uses the opening price when it resolves order. Keep that representation for the existing data. Do not fill unknown causes with high-first/low-first heuristics or random coin flips and then score them as truth. On the new tape, quantify how often M1 and tick labels disagree by symbol, barrier width, spread and horizon. Fine path labels may change while their horizon output bins remain unchanged.

Missing future data remain right-censored. Never encode an unobserved remaining horizon as neither-hit. Preserve complete no-touch, partial observation, cause-ambiguous touch, no-trade period, and collection outage separately. At each deadline, the three cumulative probabilities must sum to one; TP-first and SL-first incidence cannot decrease with horizon and neither-hit cannot increase. Compute conditional TP given hit as `p_tp / (p_tp + p_sl)` only when the denominator is supported and nonzero; display hit probability alongside it. A high conditional TP estimate can accompany a very low chance of either hit.

## What the official sources actually provide

The sources below were checked on the research date. Documentation availability is not a verified per-symbol historical coverage guarantee. No large archive was downloaded.

1. [Binance public-data repository](https://github.com/binance/binance-public-data) documents daily/monthly spot and futures trade, aggregate-trade and kline archives with checksums. Spot timestamps from 2025-01-01 use microseconds. Klines include trade count, quote volume and taker-buy volume. The repository warns that archived files can later be corrected. Freeze the downloaded bytes, checksum and archive version. Its described trade/kline archive is not proof of complete historical full LOB, and its calendar coverage is not Wallex execution evidence.

2. [Binance Spot REST market data](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market) documents database-backed aggregate trades, recent/old trades, order-book snapshots and book tickers. Aggregate trades combine fills from the same taker order at the same price and time. Use archive plus API repair where permitted; validate trade-ID gaps and overlap. Snapshot endpoints are current state, not a year of reconstructed depth.

3. [Binance USD-M Futures REST market data](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data) documents funding history, OI, mark/index/premium klines and basis. OI history permits only the latest month; basis only the latest 30 days. OI timestamp is period-end while basis timestamp is period-start. Do not join on timestamp without resolving aggregation and publication time. Its [official liquidation stream documentation](https://developers.binance.info/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/ws-streams/market) describes one selected liquidation snapshot per symbol within 1000ms, not a complete count of liquidation orders. Retain the documented selection rule and its version; do not claim total market liquidations from this feed.

4. [Coinbase Exchange WebSocket channels](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels) provide live matches, level2 snapshots/updates and a full order-event feed with a documented snapshot/replay sequence procedure. Matches can be dropped; heartbeat trade IDs can identify missing trades for REST repair. Level2 updates specify replacement size, not size delta. This source supports prospective capture and reconstruction; it does not establish a free historical full-LOB download. USD and USDT quotes also require explicit currency treatment.

5. [Kraken downloadable time and sales](https://support.kraken.com/articles/360047543791-downloadable-historical-market-data-time-and-sales-) describes official downloadable trade CSVs, manifests and checksum verification. Its currently described complete archive ends on 2026-06-30, with quarterly incremental releases. It is a feasible independent-venue trade source, but cannot be assumed complete through the Wallex cutoff without checking available increments or API continuation. Trade history is not historical order-book depth.

6. Bybit's [recent-trades documentation](https://bybit-exchange.github.io/docs/v5/market/recent-trade) points to archived historical trades and reports taker side. Its [developer page](https://www.bybit.com/future-activity/developer) advertises historical orderbook, kline, trade and funding data, while the [download page](https://www.bybit.com/en/derivative-activity/history-data) requires interactive product selection. Exact book history, depth, symbol coverage and schema were not verified here. The [OI endpoint](https://bybit-exchange.github.io/docs/v5/market/open-interest) supports pagination with 5m through daily intervals and contract-dependent units; it warns of delays under volatility. Its [all-liquidation stream](https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation) advertises all liquidations at 500ms push frequency, but its price field is bankruptcy price. It is not a realized execution-price tape. Probe small requested periods before committing to annual acquisition.

7. [Wallex's official earlier API reference](https://api-docs.wallex.ir/) search-indexed content confirms a public depth response with bid/ask prices and quantities. Full-page retrieval timed out; previously recorded developer subpages returned 404 during this check. Stronger local evidence is the saved official MCP response and raw UDF HTTP receipts described above. The acquisition code records that multi-day queries can return misleading `no_data`, so it uses one-day requests. No verified free annual Wallex tick/LOB archive was found. Ask the venue about an official historical tape or capture prospectively rather than fabricate depth from chart bars.

## Empirical justification and its limits

8. Cont, Kukanov and Stoikov, [The price impact of order book events](https://arxiv.org/abs/1011.6402), study 50 US stocks using NYSE TAQ. They find a short-interval relationship between price changes and imbalance of supply/demand at best bid/ask, with depth affecting impact. This supports testing event-derived order-flow imbalance and depth. It does not demonstrate a causal feature improvement for this crypto classifier or a durable signal at 24 hours. Contemporaneous explanatory association is not automatically a forecast available before the decision.

9. Gould and Bonart, [Queue imbalance as a one-tick-ahead price predictor](https://arxiv.org/abs/1512.03492), fit probabilistic direction models on ten Nasdaq stocks. Predictive strength differs between large- and small-tick stocks. This supports preserving queue size and tick regime as candidate information. Their target is the next midprice movement, not arbitrary TP/SL first hit over 5 to 1440 minutes. Any crypto transfer claim requires our own chronological evaluation.

10. Makarov and Schoar, [Price discovery in cryptocurrency markets](https://pubs.aeaweb.org/doi/10.1257/pandp.20191020), report that price discovery differs across Bitcoin exchanges and over time, especially under segmentation. The publisher's indexed abstract was accessible; direct article opening returned 403. The authors' longer primary working paper, [Trading and arbitrage in cryptocurrency markets](https://www.fmg.ac.uk/sites/default/files/publications/DP782.pdf), provides related cross-exchange evidence. This motivates venue-specific context and holdouts, not a fixed assumption that Binance always leads Wallex or that old Bitcoin results transfer to Sonic.

## Point-in-time collection and evaluation contract

Store exchange event time, transaction/matching-engine time where provided, local receive time, sequence/trade IDs, source endpoint, source file checksum, acquisition time and schema version. Keep separate clocks rather than renaming them all `timestamp`. Check server clock offset and network delay. Normalize seconds, milliseconds and microseconds explicitly; do not infer units only from one historical date.

A predictor at origin `t` can use only information available by `t`, including the collection delay. Period-end values and settled funding observed later must not be backfilled into earlier decisions. If historical receive times are unavailable, call that an idealized event-time experiment and run conservative lag sensitivity checks. Cross-venue as-of joins need maximum-age limits and missingness flags, not a future nearest match. Historical latest-trade bars may need an availability delay after their nominal completion.

Define a point-in-time asset universe before looking at future success: listing date, minimum observed trading history and lagged liquidity rules. Include delisted, inactive and failed instruments where obtainable. Keep outages, tick-size changes, forks and stablecoin dislocations as events. Adding assets chosen because they are liquid today creates survivorship and hindsight selection. Evaluate asset holdouts and venue holdouts in addition to chronological holdouts.

Use 24-hour target purges when 1440m labels overlap a role boundary. Hyperparameter selection and calibration need independent dates. Since cross-assets share shocks and origins overlap, report results by UTC day/week and regime, and use grouped time blocks for uncertainty rather than treating 53 million requests as IID. Group calibration and bootstrap draws at the origin/day level across all geometries and assets. Retain all mandated horizons and distinguish absolute-price, percentage-distance, volatility-distance and tick-distance slices.

Do not discard every origin lacking an entire week from every future deployment model solely to make one long-context competitor eligible. Keep a common-cohort comparison for fairness, but separately report eligibility and deployment coverage by context. The current continuity requirement can remove stress periods and late dates. A future missingness-aware model is a separate protocol, not a silent imputation into the frozen annual run.

## Small, reviewable next experiment

First freeze a new data protocol and instrument metadata table. Select several contiguous weeks and stress episodes using rules decided before label inspection. Obtain target-venue trade/book evidence if available, plus one major external spot/perpetual tape. Audit trade IDs, clock units, sequence gaps, bar reconstruction and price increments. Compare trade-based and M1 labels, with unknown-cause bounds. Then run matched-cohort ablations for activity/staleness, trade flow, top-of-book, cross-venue context and derivatives state. Score proper horizon loss and calibration by asset, horizon, barrier distance and activity state. Keep external-venue labels in their own task. If same-venue tape is unavailable, retain the existing task's chart-price qualification.

The fastest useful improvement may be exposing that the input is a stale local last trade and the supplied barrier lies inside one price increment. Neither a larger neural network nor a million more mirrored queries supplies that missing observation.

### Reproduce the local activity counts

Run with the existing Python environment from the repository root. This reads only the four CSVs and prints one row per source; it does not write output or import training code.

```python
from pathlib import Path
import pandas as pd

for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "SUSDT"):
    frame = pd.read_csv(Path("data/barrier-history-dense-20261022/market") / f"{symbol}_1m.csv")
    zero = frame.volume.eq(0)
    flat = frame.high.eq(frame.low)
    assert not (zero & ~flat).any()
    print(symbol, len(frame), int(zero.sum()), float(zero.mean()), int(flat.sum()), float(flat.mean()))
```
