# Chronos-2 clean hourly adaptation result

Full neural adaptation completed on 2026-10-22 after 700 optimizer steps. Early stopping selected the step-300 checkpoint, with validation loss 0.5546126366; steps 400/500/600/700 did not improve it. Actual trainer runtime was 855.65 seconds. Total receipt elapsed time, 1922.88 seconds, includes waiting for the shared GPU lock. Peak allocated CUDA memory was 2,819,795,456 bytes.

The checkpoint is `checkpoints/selected-model`. Its weights SHA256 is `28731e6c29c1c9d67a378204b71530de24ca68b5b7eb0985da79c4d3566dd753`, differing from the pinned pretrained weights `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`. The numeric audit verifies changed float32 parameter values in five independent tensors, including layer normalization and attention K/O/Q/V weights. This is full model adaptation, not LoRA or a fitted trading-policy wrapper.

Both pretrained and adapted models were evaluated on the same 48 validation and 48 test windows, each with 24 future hourly closes. Thus each split contains 1,152 marginal forecast points. Full contexts and labels stay within their chronological split. The common boundaries are April 1, August 7 02:00 UTC, September 3 13:00 UTC, and October 1, 2026. Only historical OHLCV and known future calendar inputs were used. No historical macro publication times or fundamentals were invented. A legacy receipt reference to an older TVL manifest is not an input to this run; the actual input hashes list only the four verified hourly price files and facts were not supplied.

| Split/model | Mean price MAE / reference | Twice-pinball / reference | Direction accuracy | Empirical q10-q90 coverage |
|---|---:|---:|---:|---:|
| Validation pretrained | 2.0413% | 1.5354% | 48.96% | 68.58% |
| Validation adapted | 1.9599% | 1.4788% | 48.87% | 70.83% |
| Validation flat | 1.8605% | 1.8605% | 0%* | 0%* |
| Test pretrained | 1.4600% | 1.0411% | 49.44% | 72.22% |
| Test adapted | 1.3283% | 0.9665% | 45.18% | 72.05% |
| Test flat | 1.2243% | 1.2243% | 0%* | 0.09%* |

*Flat predicts zero movement and a degenerate price interval. Its directional and coverage values are mechanical comparisons, not a meaningful directional classifier or probabilistic forecast.

Adaptation reduced test MAE by about 9.0% relative to pretrained, but both models lost to flat on validation and test. Direction accuracy also declined. The preregistered validation gate therefore rejected promotion. The council adapter explicitly uses pretrained forecasts, while preserving the adapted checkpoint and complete comparison artifacts. Test outcomes did not select the checkpoint or the gate. These results do not establish profitability.

| Test endpoint | Pretrained MAE | Adapted MAE | Flat MAE |
|---|---:|---:|---:|
| 1 hour | 0.4180% | 0.4380% | 0.3789% |
| 3 hours | 0.6269% | 0.6097% | 0.5085% |
| 6 hours | 0.8429% | 0.8618% | 0.8145% |
| 12 hours | 1.5308% | 1.3338% | 1.1950% |
| 24 hours | 2.4026% | 2.1368% | 2.0372% |

`paired-comparison.json` also contains per-asset MAE in USDT, amplitude errors, all 21 quantile empirical CDF values, and endpoint metrics. `training.json` is the original completed worker receipt. The first attempt at `../hourly-full` remains preserved with its actual step-75 Windows atomic-file failure.

## Runtime input and latency checks

The portable adapter accepts `{asof, markets:[{symbol,candles_path,quote}], settings}` through `python -m wallex_lab.council_chronos --request ... --output ...`. Native M1 source calls declare `settings.interval_seconds=60`; direct hourly inputs are also supported. Minute inputs use the shared bounded repair helper only as an intermediate regular clock. Hourly HIGH/LOW/VOLUME become native NaN when any contributing minute is unobserved. OPEN and CLOSE remain observed only when their own boundary minute is observed. A real hour-end CLOSE is retained even when an interior minute is absent. Calendar values remain known. Source CSV files are never rewritten. Observed masks, missingness quality, real completion timestamps, forecast-origin clocks, and forecast expiry are disclosed.

The real four-asset M1 CPU benchmark took 8.74 seconds to prepare six months of source CSVs, 14.88 seconds to load the model, and 4.16 seconds for neural inference, totaling 28.00 seconds inside the worker excluding Python/torch startup. Every median path exactly matched its direct hourly-input counterpart. A separate actual neural inference using three missing minutes succeeded with one native missing hourly target, three masked OHLCV hours, and an observed latest hour-end close. That worker took 29.23 seconds including torch import. The final warm four-asset benchmark took 1.85 seconds for inference and 2.22 seconds including input preparation. Council checks use a 15-minute cadence with hourly completed-bar context.

The M1 filename/first-gap integration check exercises the real shared council request builder with `reference.csv` and a 120-second first observed delta. Explicit source cadence correctly routes it through M1 handling. The checked status writer uses shared `wallex_lab.locking.atomic_replace` with bounded retries.

Quantiles are marginal future CLOSE distributions. They are not joint sampled paths, TP-before-SL probabilities, calibrated fills, or trade authorization. Unknown pretrained data overlap, correlated evaluation windows, empirical undercoverage, and Binance-to-Wallex venue differences remain limitations.
