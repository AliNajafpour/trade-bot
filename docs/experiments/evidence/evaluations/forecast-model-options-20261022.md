# Forecast model candidates, 2026-10-22

**Run TimesFM 2.5 first as a pretrained comparator.** Its Transformers implementation is already installed in the existing GPU runtime, its public checkpoint is Apache-2.0, and there is an official LoRA recipe. This is an integration choice, not evidence that it forecasts Wallex better than Chronos-2 or produces profitable trades. [Official checkpoint](https://huggingface.co/google/timesfm-2.5-200m-transformers), [official fine-tuning example](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/examples/finetuning/README.md).

| Candidate | Forecast and additional inputs | Adaptation, license and local cost |
|---|---|---|
| **TimesFM 2.5, nominal 200M** | Point plus marginal quantile forecasts. The installed Transformers interface takes univariate history. Google's separate XReg wrapper adds numerical covariates; the present probe does not use it. Context up to 16,384 and optional quantile head up to 1,024 steps. | Apache-2.0 weights. Official Transformers/PEFT LoRA workflow; PEFT is not installed here. Existing native inference class needs no dependency upgrade. Public F32 checkpoint is **925,187,448 bytes**. [Model](https://huggingface.co/google/timesfm-2.5-200m-transformers), [capabilities and XReg](https://github.com/google-research/timesfm), [LoRA](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/examples/finetuning/README.md). |
| **TiRex-2** | Native multiple targets, past covariates and future-known covariates. Nine marginal quantiles, not jointly sampled price paths. 38.4M active parameters univariate, another 44.1M active multivariate. | Apache-2.0 checkpoint. Public release focuses on zero-shot inference; advertised fine-tuning belongs to Pro, so a ready public training pipeline is not established. Not installed. Its dependencies require NumPy >=2.1.3 versus current 1.26.4, plus specialized kernels; use a separate environment and verify Windows support first. [Model](https://huggingface.co/NX-AI/TiRex-2), [forecast API](https://github.com/NX-AI/tirex-2/blob/main/docs/how-to/forecasting.md), [dependencies](https://github.com/NX-AI/tirex-2/blob/main/pyproject.toml). |
| **Moirai 2.0-R-small, 11.4M** | Decoder-only quantile model, missing-value-aware patch embedding. Uni2TS supplies multivariate and dynamic-feature forecast interfaces. | Very small weights, but this checkpoint is **CC-BY-NC-4.0 / research only** despite Apache-2.0 framework code. Uni2TS provides training workflows; Uni2TS/GluonTS are absent locally. Extra environment integration and licensing constraints make it a secondary research comparator. [Checkpoint](https://huggingface.co/Salesforce/moirai-2.0-R-small), [framework](https://github.com/SalesforceAIResearch/uni2ts). |

The current upstream release is TimesFM 3.0, with native multivariate and past/future covariates. Its weights have separate non-commercial/non-production terms; the repository explicitly distinguishes those from Apache-2.0 weights through 2.5. Therefore 2.5 is the simpler candidate for this project, not the latest version. [Upstream release and licensing](https://github.com/google-research/timesfm).

## Actual local acquisition and probe

- Runtime inventory: PyTorch `2.13.0+cu130`, Transformers `5.16.1`, NumPy `1.26.4`; native `transformers.models.timesfm2_5` present. PEFT, TiRex-2 and Uni2TS absent. Existing model inventory had no TimesFM/TiRex/Moirai weights.
- Script: `scripts/probe_timesfm25.py`; run with the existing Trade-Bot `.venv-gpu/Scripts/python.exe`, optionally `--download`.
- Pinned public revision: `5a9806b9b291fad9233b5249d88263f1846304d3`.
- Official metadata and verified-file receipt: `models/timesfm-2.5-research-20261022/official-metadata.json` and `receipt.json`. Expected weight SHA256: `b53f6d52114e2ad786890f3c4637ce05f580b7800d6e24401f88b398b76035ef`.
- Probe: a complete 256-bar BTC/USDT M15 context, followed by 12 unseen bars. The last complete block is chosen by clock coverage only. No future-fine-tuned checkpoint, gold data, fundamental backfill or trading-order submission is involved.
- Input, observed outcomes, raw point/quantile outputs and metrics are persisted under `runs/timesfm25-probe-20261022/`. The script refuses incomplete/hash-mismatched weights and non-finite outputs. Full output channels are explicitly labeled; native `mean_predictions` selects q50, not a distributional expected mean.
- **Acquisition and CUDA inference completed**, with no API spend or package upgrade. Training has not been started alongside the other GPU jobs.

| Actual probe measurement | Result |
|---|---:|
| Decision origin, UTC | 2026-10-19 19:00 |
| Context / horizon | 256 / 12 M15 bars |
| Median-price MAE, normalized by last input close | 0.145496% |
| Flat-last-close MAE on identical future bars | 0.124008% |
| Predicted / observed terminal change | -0.035155% / +0.086388% |
| Import and load / inference time | 11.45 s / 0.393 s |
| Peak process CUDA allocation | 935,215,104 bytes, approximately 892 MiB |
| Raw output shapes | point 12; full 12 by 10, including separate point channel and q10 through q90 |

The model runs cheaply on this GPU, but **lost to flat-last-close on this window**. A second execution reproduced the same numerical forecast after correcting the timing measurement to exclude hashing and disk writes. `probe.json` records the final measurements and hashes. This is not a trading backtest or evidence that another model should replace Chronos-2.

## How to decide whether this improves the system

One probe validates execution only. Compare pretrained TimesFM on the same predeclared chronological origins and horizons as Chronos-2, including flat-last-close, quantile loss, coverage and direction. Then run the identical trading policy, fees and executable bid/ask assumptions. Select adaptation settings using training/validation only; evaluate the final choice once on untouched later test data. A past July test must never use a checkpoint fine-tuned on August–October data. Marginal quantiles do not establish TP-before-SL probabilities or tradable joint paths.
