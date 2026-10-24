# Chronos-2 full fine-tuning result, 2026-10-21

Actual cached official120M Chronos-2 weights were fine-tuned on Wallex crypto data; no gold rows were used.

- Full parameter adaptation, 200 optimizer steps, gradient accumulation4, context512 completed M5 bars and36-bar/3-hour targets.
- Native21 trained quantiles; raw CLOSE target; past-only OHLCV plus log TVL and1-day TVL change; four calendar cycles also known into the future.
- Four90-day source assets BTC/USDT, ETH/USDT, SOL/USDT and S/USDT. Common70/15/15 calendar split. Every entire context+target remains within its own split and contiguous segment.
- 16 train segments contain72,282 rows;41 validation windows;40 test windows. Training randomly samples windows inside train segments.
- Historical fundamental publication times are UNKNOWN. Input files assume vendor measurement time+48hours. These are retrospective research inputs, not verified point-in-time historical facts. No current CoinGecko data was backfilled.

| Heldout metric | Pretrained | Fine-tuned |
|---|---:|---:|
| Median-price MAE | 0.563688% | 0.597842% |
| Flat last-close baseline MAE | 0.535825% | 0.535825% |
| Normalized mean quantile loss | 0.402252% | 0.418235% |
| Correct direction on changed-price points | 185/340 | 140/340 |
| Empirical10–90% interval coverage | 377/432 | 347/432 |

Metrics use12 fixed evenly spaced heldout tasks,432 forecast prices. Overlapping horizon points are not independent trades. MAE and quantile loss are divided by the last observed context CLOSE. No test metric was used to choose the checkpoint or hyperparameters.

The selected validation loss decreased slightly across checkpoints, but heldout price error and direction worsened. The fine-tuned model must not be promoted merely because training completed. This experiment establishes neither profitable trading nor the value of fundamentals. Further comparisons require fresh validation or prospective observations.

Training took 278.53 seconds, peak CUDA allocation2.858GiB. Selected checkpoint: `workspace:\runs\chronos2-finetune-fundamental-20261021\checkpoints\selected-model`.

Inspect `training.json`, `input-schema-audit.json`, the original per-task quantile NPZs and native Trainer checkpoint metadata for full receipts. The `fundamentals.status` legacy field refers only to the unused optional long-format facts argument; the aligned two fundamental covariates were actually included, as recorded by the schema audit.
