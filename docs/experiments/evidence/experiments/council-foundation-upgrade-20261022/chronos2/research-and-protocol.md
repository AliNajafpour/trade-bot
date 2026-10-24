# Chronos-2 council adaptation evidence

The local pretrained model is amazon/chronos-2 revision 29ec3766d36d6f73f0696f85560a422f50e8498c. Its weights SHA256 is ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42. The source directory is read-only and is never rewritten.

## Primary documentation inspected on 2026-10-22

- [Amazon model card](https://huggingface.co/amazon/chronos-2): encoder-only 120M parameters; native multivariate/covariate inputs; marginal multi-step quantile forecasts; CPU and GPU support.
- [Official repository](https://github.com/amazon-science/chronos-forecasting): separate historical target/covariates and future covariates.
- [Official pipeline implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py): fit supports full and LoRA adaptation, validation inputs, trainer callbacks and best validation checkpoint selection. LoRA needs peft, which is absent in the installed runtime; this experiment uses genuine FULL updates and installs nothing. Cross-learning can help or hurt and depends on batch size. This bounded experiment keeps cross_learning=False for both models; no untested cross-asset improvement is claimed.

## What actually existed before this work

Prior receipt: runs/chronos2-finetune-fundamental-20261021/training.json. Complete at 200 full optimizer steps in 278.53 seconds on 2026-10-21. Training calendar began 2026-07-23 20:15+03:30 and ended 2026-09-24 20:15+03:30; validation ended 2026-10-08 08:15+03:30; test ended 2026-10-21 20:15+03:30. Four 5-minute assets BTC/ETH/SOL/S, with missing-market segments and historical TVL assuming measurement+48h availability. That assumption is not verified point-in-time provenance.

Selected checkpoint-200 had validation loss 0.2697479725. On 12 fixed test tasks / 432 forecast points, pretrained MAE fraction was 0.0056368806, adapted 0.0059784165, flat 0.0053582467. Pretrained/adapted normalized twice-pinball losses were 0.0040225220 / 0.0041823451. Direction correct counts were 185 / 140 out of 340 changed targets. Adaptation therefore did not justify promotion.

## Preregistered clean experiment

Source: verified data/council-reference-20261022/manifest.json, Binance USDT spot BTC/ETH/SOL/XRP April-September 2026. Hourly completed OHLCV is used for this experiment. No historical macro series or assumed historical release times are added. Past covariates are open/high/low/volume and calendar sine/cosine; only calendar is known in the future. Close is the raw-price target.

Common 70/15/15 chronological boundaries apply across every asset. A full 128-hour context and 24-hour future stays within its split; no context or target crosses a boundary or source-data gap. Validation/test forecast origins are fixed before predictions and evaluated identically for pretrained/adapted models. Forty-eight evenly spaced tasks per split are retained with per-task numeric quantile archives. Endpoints are reported at 1/3/6/12/24 hours. Training ceiling is 1500 optimizer steps, evaluation every 100, patience four checks, validation loss alone selects the checkpoint. Test data never selects the checkpoint or the gate.

Promotion requires validation MAE at least 2% below BOTH pretrained and flat, lower pinball than pretrained, and empirical q10-q90 coverage between 70% and 90%. Even a passing model is advisory only. A rejected adaptation remains inspectable, while the adapter explicitly uses pretrained output. This threshold is experimental and is not evidence of profitability. Unknown pretrained overlap and Binance-to-Wallex differences remain limitations.

The first clean attempt at hourly-full failed at optimizer step 75 because Windows denied an atomic status-file replacement during a read. The failed receipt is preserved. The retry uses the same modeling protocol and a bounded retry around atomic replacement. There was no outcome-driven change of hyperparameters.

Marginal quantiles do not define a joint price path or TP-before-SL probability. No independence assumption or forced-positive forecast is used.
