# ETH/TMN frozen-forecast technical policy replay

This is an actual candle backtest of frozen HGB/GRU forecasts and deterministic technical policies. It is not an LLM/news fusion profit claim.

The original zero-slippage benchmark was replayed exactly for both models. 96 scenario/window ledgers reconcile cash, inventory and closed trade PnL.

## Full 12-day TEST at 30 bp fee plus 15 bp adverse slippage per side

| Model | Policy | Return % | PnL toman | DD % | Trades | Wins/losses | Fees toman |
|---|---|---:|---:|---:|---:|---:|---:|
| hgb | original_50pct | 3.056 | 152815 | 1.446 | 8 | 5/3 | 122181 |
| hgb | cap_only_15pct | 0.913 | 45668 | 0.437 | 8 | 5/3 | 36406 |
| hgb | technical_conditional_risk | 0.353 | 17643 | 0.159 | 1 | 1/0 | 4498 |
| mean_hgb_gru | original_50pct | 3.426 | 171301 | 1.658 | 7 | 4/3 | 106838 |
| mean_hgb_gru | cap_only_15pct | 1.018 | 50903 | 0.510 | 7 | 4/3 | 31871 |
| mean_hgb_gru | technical_conditional_risk | 0.000 | 0 | -0.000 | 0 | 0/0 | 0 |

The operational variant uses later completed price and volume confirmation, a fixed original six-hour forecast deadline, 15% exposure ceiling and a 0.25% equity risk budget after the explicit unknown-news reduction. Stop and cost geometry must support at least 1:1 net reward/risk. Each friction scenario gates and sizes against its declared costs, with no return-based scenario selection.

The cap-only control isolates the effect of reducing the original 50% exposure to 15%. Baseline/control thresholds remain frozen from the original validation period: HGB 0.011, HGB/GRU mean 0.0085. The operational gate uses the corresponding frozen margin plus current declared execution costs, matching AIMarket. Its stop and target fractions match production ATR/range/volatility formulas and anchor at entry.

## Limits

- Already exposed historical TEST; added operational policy was designed after viewing historical results
- No current news, sentiment, order book, or fabricated LLM decisions are injected
- OHLC proxy executions; fee tier official public schedule, actual account eligibility unverified
- Sizing uses candle-close allocation risk approximation, not actual broker quantity sizing or exchange precision
- Stops observe closes and fill next opens; gaps can exceed risk budgets
- Each window starts cash; partial-window results do not compound to the full-window result

Full 12-day, first-six-day, last-six-day and last-three-day rows, including zero/5/15/30 bp slippage scenarios, are in the adjacent CSV/JSON. Cash-only return is zero.

Fee evidence: [tmn_level1_taker](https://api.wallex.ir/v2/structured-fee). Raw evidence and input/source hashes are preserved in the run identity.
