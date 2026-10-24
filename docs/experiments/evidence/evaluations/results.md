# Actual experiment results, 21 October 2026

All six requested strategy families were implemented and executed on actual public Wallex BTC/TMN data. Four numerical families also ran on ETH/TMN, SOL/TMN and USDT/TMN. An additional six-hour-label comparison ran the three non-pretrained numerical/rule families on all four markets. There are 30 successful strategy runs, not 30 independent discoveries. No active strategy demonstrated positive net historical performance under the stated costs.

## BTC/TMN main comparison

| Strategy | Net return | Stress return | Fills | Closed position episodes |
|---|---:|---:|---:|---:|
| rules | -8.2722% | -8.3917% | 50 | 21 |
| trees | +0.0000% | +0.0000% | 0 | 0 |
| neural | +0.0000% | +0.0000% | 0 | 0 |
| kronos | -4.3823% | -6.4623% | 22 | 9 |
| agent | +0.0000% | +0.0000% | 0 | 0 |
| fusion | -0.8143% | -1.2725% | 73 | 15 |

The cash baseline returns 0%. The same-cap constant-allocation baseline returns -1.0637%. Zero-fill tree, neural and agent runs retained cash; they do not establish a profitable trading edge or satisfy the challenge activity criterion. Fusion reduced losses relative to the active components in this sample but underperformed cash.

## Data and chronology

| Symbol | Native M1 rows | Complete M15 rows | Missing M15 intervals |
|---|---:|---:|---:|
| BTCTMN | 86304 | 5750 | 10 |
| ETHTMN | 86275 | 5746 | 14 |
| SOLTMN | 86285 | 5747 | 13 |
| USDTTMN | 85273 | 5682 | 78 |

The BTC final test begins at 2026-10-09T13:15:00+00:00; acquisition covers approximately 60 days ending 21 October 2026. All symbols use actual Wallex candles, with raw HTTP bodies and hashes retained. The API returned minute bars for resolution=15; only complete native-minute buckets became M15. No OHLC gaps were interpolated. All 352 initial acquisition raw-body receipts were verified and all four aggregate CSVs reconstructed.

Training uses the first 60%; base-model selection the next 10%; fusion selection the next 10%; final historical testing the last 20%. Each fit/selection omits future test labels, but the overall development history is now exposed. The additional horizon study is exploratory, not a second independent confirmation. Model pretraining overlap and historical LLM knowledge contamination remain possible.

## Costs and execution

Every independent account starts with 5,000,000 toman, cash only, with a 50% allocation cap. The main cost assumption is 20bp per-side fees plus per-side slippage of 15bp BTC, 25bp ETH/SOL and 10bp USDT. These are assumptions informed by current spread snapshots; challenge account fees and historical spreads are unverified. Stress adds 10bp to each fee and slippage component. Minimum rebalance notional is 50,000 toman. Independent risk stops observe 3% UTC daily loss and 8% close-sampled drawdown, then exit at the next available open. Higher costs can trigger risk stops sooner, so stress-return ordering is not guaranteed to be monotonic.

Signals use completed closes and fill at the next candle open. A gap cancels its stale crossing order and liquidates held inventory at the first observed post-gap open. The initial hindsight gap-close implementation was rejected by review; initial experiments are preserved separately and final figures use fresh processes. No authentic exchange fills are claimed. Candle fills ignore order queues, partial fills and volume impact; fractional quantities are not rounded. Forced exits can bypass venue minima, which is counted in every ledger. Final simulated cash is not guaranteed account-executable.

## Actual AI and prospective smoke test

Historical agent execution contains 10 validated actual LLM decisions, with 1 bounded transient retry. The earlier timeout and invalid-evidence response remain in immutable receipts; accepted decisions are not mocked substitutes. The decision spacing is 230 M15 bars, a sparse budget-limited comparator rather than continuous minute trading. The actual served model identity is not exposed by the CLI. All accepted event streams contain zero tool calls.

Fusion uses actual expert outputs with separate selection blocks verified as `True`. Selected weights: `{"agent": 0.75, "kronos": 0.0625, "neural": 0.0625, "rules": 0.0625, "trees": 0.0625}`. The numerical-only and agent-only validation ablations are retained in its result metadata.

The true-current LLM decision is `hold`, allocation 0, confidence 0.87. It used freshly collected public news and BTC order-flow context. First-observed timestamps prevent current context entering historical packets. This result contains a decision rather than a backtest trace and is explicitly nonbacktestable.

Live shadow collectors retain four actual depth observations per run, quote spreads, receive times and raw hashes. These use stale research targets with their ages disclosed; they are operational smoke tests, not prospective model-performance validation. Paper entry/exit prices are visible ask/bid proxies and do not guarantee fills. See the shadow summaries below.

## Verification and retained artifacts

- 12 bounded offline checks passed; 0 failed in the final check receipt.
- 30 actual strategy outputs replayed all three economic scenarios exactly, without new training or network calls.
- Tree forecasts reloaded exactly; selected neural forecasts reproduced within declared CPU/CUDA floating-point tolerance; saved Kronos latent/PCA/Ridge scores reloaded exactly.
- Actual LLM raw files, final JSON schemas and zero-tool event streams passed receipt verification.
- Source/data/output hashes are checked before experts enter fusion.
- Authenticated exchange order count: zero. Market eligibility, credit account constraints, actual fees and live order reconciliation remain unverified.

Artifacts: [complete comparison](comparison.csv), [structured results](results.json), [offline checks](checks.json), [replay verification](replay-verification.json), [research](../research/research.md), [review](../research/code-review.md), [reuse provenance](../research/reuse-manifest.json), [run instructions](../../../../README.md).

- [Shadow receipt 20261021T134958Z](../experiments/shadow/20261021T134958Z/summary.json)
- [Shadow receipt 20261021T135653Z](../experiments/shadow/20261021T135653Z/summary.json)
