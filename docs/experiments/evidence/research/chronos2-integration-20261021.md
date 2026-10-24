# Chronos-2 integration evidence, 2026-10-21

The official `amazon/chronos-2` is a 120M parameter time-series model supporting
multivariate targets, past numeric/categorical covariates, known future covariates,
context up to8192 and prediction up to1024 steps. Its output is a set of marginal
quantiles at each forecast step. Those are not independent sampled joint price
paths. News text needs a separate, time-stamped numeric representation before it
can become a meaningful covariate. These capabilities do not establish Wallex
trading profitability. [Official model card](https://huggingface.co/amazon/chronos-2)

The official package exposes `Chronos2Pipeline.predict_quantiles`, `predict_df` and
native full/LoRA fine-tuning. Cross-series learning is an optional inference
setting, distinct from merely supplying covariates.
[Official package](https://github.com/amazon-science/chronos-forecasting)

## Local reuse and actual implementation

The existing local checkpoint at
`Trade-Bot/models/chronos-2-29ec376` is reused read-only. No gold training data,
forecasts, models fitted on gold, or trade outcomes were imported. Original HF
revision is `29ec3766d36d6f73f0696f85560a422f50e8498c`; config and477,930,472-byte
weights are validated against the existing download receipt and fixed SHA256s.
Installed `chronos-forecasting`2.3.2 already worked; no dependency upgrades or
re-download were necessary.

`wallex_lab/chronos_market.py` exposes `ChronosMarket.forecast` and `build_task`.
The target is raw Wallex CLOSE. Past OHLCV and separately supplied fundamental
metrics are native covariates. Four cyclic calendar fields are supplied both in
history and as known future inputs. Missing Wallex candles become explicit NaNs
on the regular clock, retaining the package's native masks. Prices are never
interpolated, zero filled or compressed over gaps. All21 trained quantile levels
are retained, including q01 andq99. Raw native task arrays, input CSV, quantile
arrays, source hashes and model provenance are saved per inference.

Installed source warns that the second return named `mean` in
`predict_quantiles` actually contains the median. This integration labels it
MEDIAN and does not treat it as expected value.

The retrospective fundamental sensitivity input uses real DefiLlama historical
TVL with explicitly assumed measurement+48h availability. Historical publication
availability is unverified. The strict verified mode has no known historical TVL
before its actual first receipt; missing information is not neutral evidence.
No fundamental metric is supplied as known future. Fine-tuning and profitability
assessment remain separate from inference accuracy diagnostics.
