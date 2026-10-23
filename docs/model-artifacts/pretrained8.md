# Existing eight-asset probability model

`models/barrier-survival/pretrained8/` packages an already trained checkpoint. This export performed **no training**, learned no new weights and remapped no asset IDs. The existing three-member Weekly late bundle remains separate.

The source is `runs/barrier-vnext-20261023/early/pretrained/seed-42/best.pt`, SHA-256 `fdc447607006f8c72b25df36e94b2f00bcd85eb367dfd4758cbb60505d6c5042`. The exported member hash is `07dbefdb945f00294a30bea9354d8daf5450a423dff9035a52167ec0d5e76a50`. Its exact symbol order is ADAUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, LINKUSDT, SOLUSDT, XRPUSDT.

This is one seed-42 member, with 10,080-minute input, width 96, three layers, causal `relative-v2` normalization and saved calibration temperature 1.0. The original NextBarrierModel has all four optional feature flags disabled and directly calls MultiScaleSurvival on its first ten input channels. The export copies all 62 used tensors unchanged and removes only the unused `mask_token`, `reconstruction.weight` and `reconstruction.bias` tensors. Inference requires the existing MultiScaleSurvival implementation; it does not require the archived NextBarrierModel code.

`provenance/` preserves the original protocol, fit identity, selected checkpoint receipt, calibration and test results. Historical fitting selected supervised epoch 1 over 2,822,568 fit origins and 66,816 selection origins. Masked pretraining had processed 832,000 origin visits but completed no full epoch before the earlier deadline. The name “pretrained” is the archived variant name, not a claim of completed pretraining.

The early fold uses fit through 2026-05-26, selection through 2026-06-25, calibration through 2026-07-25 and test through 2026-08-24, with the existing 24-hour target purges. Existing test results cover 66,816 origins and 8,552,448 correlated origin/query visits across eight Binance assets. At 24 hours the pooled categorical log loss is 0.654765 and Brier score is 0.420470. These metrics pool 128 barrier geometries per origin; they are not per-asset or fixed ±2% policy results. Historical results do not establish prospective profitability.

## Observed-context verification

The original model and exported/reloaded model produced bit-identical CPU logits on each actual contiguous 10,080-minute context. The maximum absolute logit difference was zero for XRPUSDT, DOGEUSDT and BNBUSDT. Full hashes, timestamps, support flags and all forecast horizons are in `pretrained8-observed-equivalence.json`.

For a long entry at the last completed close, +2% TP, −2% SL, and a 24-hour horizon:

| Asset | Context venue | TP first | SL first | No touch |
|---|---|---:|---:|---:|
| XRPUSDT | MEXC | 0.265491 | 0.196260 | 0.538248 |
| DOGEUSDT | KuCoin | 0.307185 | 0.132035 | 0.560780 |
| BNBUSDT | KuCoin | 0.237942 | 0.116519 | 0.645539 |

Every row has valid probability mass and no geometry support flags. These are recorded historical snapshots, not current market quotes. All three absolute TP probabilities are below 0.55. Conditional TP probability given a touch must not replace the absolute TP-first probability. Training used Binance; probability calibration on MEXC and KuCoin has not been measured. A single member's zero member standard deviation is not an estimate of zero model uncertainty.

## Reproduce without training

From the published checkout, with an original workspace that contains the archived artifacts:

```powershell
python -m scripts.export_pretrained8 --source-root ORIGINAL_WORKSPACE --output NEW_OUTPUT_DIRECTORY
python -m checks.check_pretrained8_export --model NEW_OUTPUT_DIRECTORY --source-root ORIGINAL_WORKSPACE --output OBSERVED_EQUIVALENCE_JSON
```

The export refuses to overwrite an existing directory. `python -m checks.check_pretrained8_export` checks the bundled member and historical provenance without requiring the original workspace. Passing `--source-root` additionally loads both original model classes, verifies all used tensors and runs the actual observed-context equivalence checks. Neither command optimizes parameters or submits orders.
