# Trading test results

96 backtest scenario/window ledgers were evaluated. The table uses the full 12-day ETH/TMN M5/H6 window, with published 30 bp fees and a declared 15 bp adverse slippage scenario on each side. Starting cash is 5,000,000 toman per independent account.

| Model | Policy | Closed trades | Wins/losses | Net return | Net PnL, toman | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| hgb | original_50pct | 8 | 5/3 | 3.056% | 152,815 | 1.446% |
| hgb | cap_only_15pct | 8 | 5/3 | 0.913% | 45,668 | 0.437% |
| hgb | technical_conditional_risk | 1 | 1/0 | 0.353% | 17,643 | 0.159% |
| mean_hgb_gru | original_50pct | 7 | 4/3 | 3.426% | 171,301 | 1.658% |
| mean_hgb_gru | cap_only_15pct | 7 | 4/3 | 1.018% | 50,903 | 0.510% |
| mean_hgb_gru | technical_conditional_risk | 0 | 0/0 | 0.000% | 0 | -0.000% |

The conditional policy uses the production candle geometry and reduced risk, but its WATCH/trigger choice is deterministic. It does not replace unavailable historical LLM/news decisions. The original exposure is 50%, the cap-only control 15%, and conditional sizing is at most 15%. These returns are not equal-risk evidence that one model is superior.

Prospective shadow ran from 2026-10-21T15:48:53.460298+00:00 to 2026-10-21T15:54:54.710749+00:00, with 2 context snapshots and six independent BTC/ETH/SOL accounts.

| Shadow account | Ticks | Paper buys/sells | LLM evaluations | Realized PnL | Marked total PnL |
|---|---:|---:|---:|---:|---:|
| BTCTMN_hgb | 24 | 0/0 | 0 | 0 | 0.0 |
| BTCTMN_mean_hgb_gru | 24 | 0/0 | 0 | 0 | 0.0 |
| ETHTMN_hgb | 24 | 0/0 | 0 | 0 | 0.0 |
| ETHTMN_mean_hgb_gru | 24 | 0/0 | 0 | 0 | 0.0 |
| SOLTMN_hgb | 24 | 0/0 | 0 | 0 | 0.0 |
| SOLTMN_mean_hgb_gru | 24 | 0/0 | 0 | 0 | 0.0 |

Separate fixed-15% buy-and-hold paper controls used actual initial asks and final bids, with entry fees and hypothetical exit fees. These baseline buys are not AI trades:

| Market | Baseline paper buys | Final marked PnL, toman |
|---|---:|---:|
| BTCTMN | 1 | -253 |
| ETHTMN | 1 | -4,435 |
| SOLTMN | 1 | -4,415 |

33 software checks passed. Shadow raw receipts, immutable event inputs and account marks passed the offline verification. Live account fees, additional market impact and exchange fills remain unverified. Marks include a hypothetical exit fee; open-position marks are not realized profit.

Collection status: partial. Account/quote/candle execution failures: 0. Context source errors: [{"source": "https://api-docs.wallex.ir/", "error": "URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Hostname mismatch, certificate is not valid for 'api-docs.wallex.ir'. (_ssl.c:1016)>"}]. The failed old documentation fetch is retained; it does not supply trading data.

Zero eligible technical opportunities means the full fusion gate was not activated for those ticks. A short shadow interval cannot validate a six-hour trading horizon. Existing earlier four-role diagnostic calls remain separate from this run.

Recorded-observation replay separately checked four real historical books and sparse news/fundamental coverage. It created no quotes or historical fusion profit. Detailed cost sensitivities, reset windows, logs and independent review are linked from the adjacent JSON evidence paths.
