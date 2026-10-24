# Direct frozen Kronos price-path diagnostic

Actual local GPU/CPU inference with the pinned upstream autoregressive predictor; no Ridge return head and no ATR substitute.

Model: small; device: cuda:0; context: 128 M5 bars; horizon: 60 minutes; paths per cutoff: 16.
Every path uses sample_count=1 with a recorded distinct seed. Upstream averaging across sample_count is never used as a distribution.

Price draws: 144; finite positive CLOSE valid: 144; CLOSE discarded: 0. Full OHLC/volume valid: 21; physically invalid: 123. No generated prices or wicks were repaired.
Load: 8.540s; inference total: 61.713s.


All-cutoff price MAE: direct median 0.452961%; flat last-close baseline 0.448402%. Terminal q10–q90 empirical coverage: 33.3%; terminal direction accuracy: 55.6%. Nine cutoffs do not establish generalization or calibration.

| Symbol | Cutoff | Valid / sampled | Direct median MAE | Last-close MAE | Inference s |
|---|---:|---:|---:|---:|---:|
| BTCUSDT | 3983 | 16 / 16 | 0.334949% | 0.388332% | 6.959 |
| BTCUSDT | 3995 | 16 / 16 | 0.514766% | 0.409281% | 7.419 |
| BTCUSDT | 4007 | 16 / 16 | 0.036203% | 0.062851% | 7.204 |
| ETHUSDT | 3958 | 16 / 16 | 0.227394% | 0.236929% | 7.448 |
| ETHUSDT | 3970 | 16 / 16 | 0.386727% | 0.471829% | 6.687 |
| ETHUSDT | 3982 | 16 / 16 | 0.417787% | 0.347214% | 6.900 |
| SUSDT | 3985 | 16 / 16 | 0.444608% | 0.386215% | 6.716 |
| SUSDT | 3997 | 16 / 16 | 1.347475% | 1.383929% | 6.264 |
| SUSDT | 4009 | 16 / 16 | 0.366742% | 0.349040% | 6.116 |

MAE is mean absolute close forecast error divided by the last observed close. The baseline is a flat last-close price forecast, not a trading strategy.

Price-only TP is q75 of path maximum CLOSE; SL is q25 of path minimum CLOSE. A long setup exists only when SL < last close < TP. These frequencies describe predicted CLOSE barrier crossings only, never intrabar touches. Neither-crossed paths remain censored at the horizon. The original full physical OHLC diagnostic separately uses max high/min low, rejects invalid entire OHLC/volume paths, and conservatively counts same-bar TP/SL touches as SL first. All model-sample frequencies are uncalibrated and do not measure real fill probabilities.

No fine-tuning or test-based parameter selection occurred. Historical weight/data overlap is unknown. These few cutoffs cannot establish forecast skill or profit. No historical books or trade fills were fabricated. Official public fee scenarios are shown only for terminal-price cost-exceedance diagnostics; actual account fees, spread and extra market impact are unknown.

Plan, source/data/weight hashes, original context CSVs, raw forecast path NPZs, per-draw seeds/timing and observed outcome CSVs are preserved beside this report.
