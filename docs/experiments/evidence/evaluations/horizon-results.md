# Multi-timeframe and multi-horizon results

Native Wallex data, cash-only spot, completed-bar decisions and next-open paper fills. TRAIN ends 2026-09-27 13:00 UTC; validation ends 2026-10-09 13:00 UTC; TEST ends 2026-10-21 13:00 UTC. Unsupported candles and missing minutes are not filled. Fixed holding horizons are 1/3/6/24 hours on M5/M15/M60 for TMN; USDT uses M15/M60 and 1/6/24 hours. Each window restarts with cash.

Completed 528 primary model/combination configurations, 5760 window/profile rows and 528 exact primary ledger replays. 258 primary configurations traded; 36 had positive fee-net returns. 4 also had positive returns in both separate six-day windows and at least five full-period trades. 2 additionally had positive earlier validation returns. These remain retrospectively screened candidates, not independent confirmation.

Public official first-tier taker fees are 30bp TMN and 13bp USDT per side. Additional TMN maker25/high-volume-taker13 and USDT maker10 scenarios are sensitivities, not verified account rates or maker fills. The source is [Wallex structured fees](https://api.wallex.ir/v2/structured-fee), captured with raw bytes and SHA256. Actual account promotions and competition/credit-package charges remain unknown.

These returns are after those official-fee scenarios and before unobserved historical spread/market impact. No historical execution cost has been guessed. Positive results are candidates for forward depth-aware testing, not established all-cost profit.

## Highest observed fee-net returns

| Market | Candle | Horizon | Model | Closed trades | Net return | P&L, quote | VAL | First6d | Last6d | Constant50 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ETHTMN | 5m | 6h | mean_hgb_gru | 7 | 4.523% | 226139.3551 TMN | 1.259% | 1.162% | 2.819% | -1.903% |
| ETHTMN | 5m | 6h | hgb | 8 | 4.306% | 215278.7976 TMN | 0.424% | 1.247% | 2.519% | -1.903% |
| ETHTMN | 5m | 6h | mean_all | 4 | 3.416% | 170816.9535 TMN | 1.308% | 0.696% | 0.892% | -1.903% |
| ETHUSDT | 60m | 6h | kronos | 7 | 2.398% | 0.4480 USDT | -0.854% | 0.133% | 2.262% | -0.113% |
| ETHTMN | 5m | 6h | mean_gru_kronos | 1 | 2.236% | 111823.2471 TMN | 0.438% | 0.000% | 2.236% | -1.903% |
| ETHTMN | 15m | 6h | hgb | 12 | 1.950% | 97487.1466 TMN | -1.579% | 0.168% | 2.224% | -1.875% |
| ETHTMN | 5m | 6h | mean_hgb_kronos | 1 | 1.793% | 89665.3068 TMN | 1.166% | 0.000% | 0.000% | -1.903% |
| ETHTMN | 5m | 6h | median_all | 4 | 1.390% | 69502.2982 TMN | 1.658% | -1.277% | 0.892% | -1.903% |

## Candidates passing the additional screen

Positive earlier VAL, positive full TEST and both restarted six-day windows, at least five full-period trades. Screening uses TEST outcomes, so these are still candidates for a new forward period, not confirmed winners.

| Market | Model | Candle | Horizon | Trades | Wins/losses | VAL | TEST |
|---|---|---:|---:|---:|---:|---:|---:|
| ETHTMN | mean_hgb_gru | 5m | 6h | 7 | 4/3 | 1.259% | 4.523% |
| ETHTMN | hgb | 5m | 6h | 8 | 5/3 | 0.424% | 4.306% |

## Model choice using validation only

For each market, choose the model/timeframe/horizon with highest earlier VAL return, without using TEST rank. If that VAL result is nonpositive, the selection baseline stays in cash. This table exposes the difference between selecting a model beforehand and retrospectively showing the highest TEST profit.

| Market | Model | Candle | Horizon | VAL | Subsequent TEST | Closed trades |
|---|---|---:|---:|---:|---:|---:|
| BTCTMN | hgb | 60m | 6h | 3.241% | -3.972% | 10 |
| BTCUSDT | mean_gru_kronos | 60m | 24h | 0.847% | -1.010% | 2 |
| ETHTMN | gru | 60m | 24h | 5.678% | -2.716% | 6 |
| ETHUSDT | mean_gru_kronos | 60m | 6h | 1.743% | -0.020% | 14 |
| SOLTMN | median_all | 5m | 3h | 2.052% | 0.000% | 0 |
| SOLUSDT | kronos | 60m | 6h | 2.714% | -1.347% | 10 |
| USDTTMN | mean_hgb_kronos | 15m | 24h | 4.316% | -1.012% | 5 |

## Matched comparison across seven markets

Each value below is an equal-weight mean of seven separate account returns with their own quote-currency fee. It is not one executable portfolio or a portfolio drawdown estimate.

| Model | Candle | Horizon | Mean return | Closed trades | Profitable markets | Worst market |
|---|---:|---:|---:|---:|---:|---:|
| gru | 60m | 6h | 0.071% | 8 | 1/7 | -0.133% |
| mean_hgb_gru | 15m | 1h | 0.057% | 2 | 1/7 | 0.000% |
| mean_hgb_kronos | 15m | 1h | 0.027% | 4 | 1/7 | 0.000% |
| hgb | 60m | 1h | 0.026% | 3 | 1/7 | 0.000% |
| gru | 15m | 1h | 0.000% | 0 | 0/7 | 0.000% |
| gru | 60m | 1h | 0.000% | 0 | 0/7 | 0.000% |

## Execution-cost headroom

The JSON report records break-even additional adverse execution per side for up to25 positive candidates. This is a counterfactual tolerance, not an estimate of historical slippage. The full CSV also includes losing and zero-trade configurations; no positive-only filtering hides the denominator.

HGB/GRU/Kronos learned only purged TRAIN labels. Kronos uses its genuine frozen hidden representation with a locally fitted Ridge head. The GRU uses a purged inner TRAIN early-stopping holdout. Threshold cost+[0,25,50]bp is selected on earlier validation. Historical TEST was already inspected in prior work; ranking its winners introduces selection risk. No independent future period was available in this run.

TMN accounts start at5,000,000 toman; USDT capital is normalized to the same initial amount using the recorded pre-TEST USDT/TMN close. Funding conversion execution is not modeled. Max exposure50%, daily close-observed stop3%, drawdown stop8%. Unrounded candle fills, venue precision, queue/depth, partial fills and forced-exit minima remain execution limitations.

Files: `reports/horizon-comparison.csv`, `horizon-primary-ranking.csv`, `horizon-matched-summary.csv`, `horizon-results.json`; full ledgers, targets and fitted-model hashes under the two run directories. Research: `docs/forecast-model-research-20261021.md`; fee evidence: `docs/wallex-fees-20261021.md`.
