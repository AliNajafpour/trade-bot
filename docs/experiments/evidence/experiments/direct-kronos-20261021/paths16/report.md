# Direct frozen Kronos price-path diagnostic

Actual local GPU/CPU inference with the pinned upstream autoregressive predictor; no Ridge return head and no ATR substitute.

Model: small; device: cuda:0; context: 128 M5 bars; horizon: 60 minutes; paths per cutoff: 16.
Every path uses sample_count=1 with a recorded distinct seed. Upstream averaging across sample_count is never used as a distribution.

Draws: 144; valid: 21; discarded: 123. Whole invalid paths are discarded, never repaired.
Load: 8.540s; inference total: 61.713s.

| Symbol | Cutoff | Valid / sampled | Direct median MAE | Last-close MAE | Inference s |
|---|---:|---:|---:|---:|---:|
| BTCUSDT | 3983 | 0 / 16 | invalid forecasts | 0.388332% | 6.959 |
| BTCUSDT | 3995 | 2 / 16 | 0.464201% | 0.409281% | 7.419 |
| BTCUSDT | 4007 | 0 / 16 | invalid forecasts | 0.062851% | 7.204 |
| ETHUSDT | 3958 | 14 / 16 | 0.228339% | 0.236929% | 7.448 |
| ETHUSDT | 3970 | 0 / 16 | invalid forecasts | 0.471829% | 6.687 |
| ETHUSDT | 3982 | 5 / 16 | 0.449568% | 0.347214% | 6.900 |
| SUSDT | 3985 | 0 / 16 | invalid forecasts | 0.386215% | 6.716 |
| SUSDT | 3997 | 0 / 16 | invalid forecasts | 1.383929% | 6.264 |
| SUSDT | 4009 | 0 / 16 | invalid forecasts | 0.349040% | 6.116 |

MAE is mean absolute close forecast error divided by the last observed close. The baseline is a flat last-close price forecast, not a trading strategy.

TP is q75 of sampled path maximum highs; SL is q25 of sampled path minimum lows. Both are actual predicted price quantiles. A long setup exists only when SL < last close < TP. A same-bar TP and SL touch counts as SL first. Neither-hit paths remain censored at the horizon. Frequencies are uncalibrated model-sample frequencies, conditioned on valid outputs, not measured fill probabilities.

No fine-tuning or test-based parameter selection occurred. Historical weight/data overlap is unknown. These few cutoffs cannot establish forecast skill or profit. No historical books or trade fills were fabricated. Official public fee scenarios are shown only for terminal-price cost-exceedance diagnostics; actual account fees, spread and extra market impact are unknown.

Plan, source/data/weight hashes, original context CSVs, raw forecast path NPZs, per-draw seeds/timing and observed outcome CSVs are preserved beside this report.
