# Dense current-history barrier survival

414,289 indexed origins; 358,517 common origins eligible for every tested context, every five minutes, 128 queries per origin; 2,100,229 native M1 candles.
Acquisition cutoff: 2026-10-22T16:00:00+00:00. Prices remain in native USDT; no gold or named technical-pattern labels.
Price/volume unit invariance, causal windows, unknown same-minute causes, missing-minute censoring and 24h split purges are checked.
All eligible fit origins and all 128 barrier requests are used in every epoch. Origins lacking a contiguous week are excluded identically for every candidate. Shared raw arrays reside on GPU, with double-precision normalization and FP32 feature output. No random history or request subsampling.
Three independent seeds, query-only comparison, separate temperature calibration and two expanding historical folds.

|Fold|Context min|Horizon min|Query-only log loss|Raw log loss|Calibrated log loss|Calibrated Brier|
|---|---:|---:|---:|---:|---:|---:|
|early|10080|5|0.27143|0.27071|0.27086|0.14258|
|early|10080|15|0.43774|0.42610|0.42655|0.23823|
|early|10080|60|0.65262|0.62647|0.62755|0.36954|
|early|10080|240|0.77970|0.74191|0.74255|0.45498|
|early|10080|1440|0.74589|0.72385|0.72342|0.46507|
|late|10080|5|0.29915|0.29151|0.29133|0.15485|
|late|10080|15|0.47463|0.45883|0.45870|0.26006|
|late|10080|60|0.67133|0.64172|0.64206|0.38080|
|late|10080|240|0.75586|0.72494|0.72538|0.44654|
|late|10080|1440|0.70539|0.70565|0.70468|0.45839|

## Context selection on common early-fold validation origins

|Input context min|Best validation partial-label NLL|Selected epoch|
|---:|---:|---:|
|10080|1.716283|2|
|2880|1.728439|2|
|1440|1.742957|2|
|720|1.735127|3|
|360|1.722001|3|

Window length and its declared hierarchical aggregation are compared together. Selection uses seed42; two additional seeds are trained for the chosen context. All held-out metrics above are scored after choices freeze.

## Pattern learning and normalization

Convolutional patches preserve local candle shape; hierarchical aggregation retains recent M1 detail and extends positional attention through up to a week. Barriers query this history by cross-attention. No triangle, head-and-shoulders or support/resistance labels are supplied.
Capacity and sensitivity to history are not evidence that a named technical pattern was discovered. The shuffled-history diagnostic is a perturbation experiment, not a semantic pattern-recognition score.
OHLC uses log(price / last completed close) divided by causal history volatility. Volume uses its ratio to recent history. TP/SL use relative log distances; volatility is supplied separately. Multiplying all prices by a constant does not change the modeled geometry.
The new normalization is explicitly versioned relative-v2. Old bundles retain relative-v1 behavior.

## Limits

Five-minute origins, mirrored directions and adjacent queries are correlated. Resampling uses UTC-day blocks, not individual requests. Historical dates overlap prior project experiments; these are held-out folds for this pipeline, not prospective evidence.
No orders, fills, fees or trading profit are evaluated. Entry prices are predetermined historical proxies. Touch horizons never trigger liquidation. Probability coherence across time is constructed; monotonicity across arbitrary barriers is not guaranteed.
Raw full-year sources and hashes, immutable protocol and snapshots, learned weights, prediction files, reliability bins and daily-block comparisons are retained in this run.
