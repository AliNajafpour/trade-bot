# Arbitrary-barrier survival model

This is a candle-only probability model. It does not submit orders, choose position size or close a position when a forecast horizon ends.

The five horizons are 5, 15, 60, 240 and 1440 minutes. At each horizon it returns cumulative TP-first probability, cumulative SL-first probability and no-touch probability. Their sum is one. `p_tp_given_touch` divides TP-first by the sum of the two first-touch probabilities; it is not the unconditional chance of TP within the horizon.

## Data and labels

`data/barrier-survival-20261022` contains 17,236 decision origins over four verified native Wallex USDT M1 sources, April 24 through July 22 2026. Each origin has 24 barrier queries: 12 volatility-scaled geometries in each orientation. Source hashes, UTC timestamp semantics, minute counts and dataset file hashes are in `dataset.json`. Original sources and the separate Trade-Bot directory are read-only inputs.

Historical entries use the last completed close with a predetermined offset of minus/zero/plus half the trailing minute volatility. This is a historical query convention, not a measured executable bid or ask. Price moves are followed from the next candle, whose open is exactly the minute-boundary decision time. No future open is an input feature.

If opening price resolves the first touched level, its cause is known. Otherwise a minute touching both barriers has unknown cause. Its likelihood is the sum of the TP and SL event probabilities. A missing minute ends observation before that gap. Source exhaustion also censors observation. A fully observed horizon without a touch is a real no-touch outcome; an unobserved horizon is not.

Event time is represented by the completion of the touched minute, without inventing intrabar order. Within-bin censoring contributes survival only through completed output boundaries. The observed minute count and censor reason remain in the dataset. Gold was excluded because the available large source has unresolved timezone, candle-label and quote-side provenance.

## Architecture and evaluation

The encoder pools the historical sequence at 1/5/15-minute scales, uses convolutional patch embeddings, a temporal Transformer, and cross-attention from the requested barriers and time-bin embeddings. The scales belong to one short-history branch. The validation comparison is between 180 and 360 minutes; both retain a 60-minute causal normalization window. The API requires 360 completed minutes for a common input contract.

Each output bin has conditional TP, SL and survival probabilities. Cumulative incidence is constructed from these hazards, so probabilities remain coherent across horizons. The architecture does not mathematically guarantee monotonicity across different barriers. `scripts/audit_barrier_survival.py` measures this separately, including random continuous queries outside the discrete training geometries.

Two expanding temporal folds use common calendar boundaries across assets, with a full 24-hour target purge at every boundary. Early stopping uses selection data, and temperature calibration has its own later dates. The first fold's selection scores choose the context. Each evaluated ensemble has three seeds; a separately trained and calibrated query-only MLP measures how much the history encoder adds. Both folds' models and calibration parameters freeze before test scoring.

The historical calendar dates have appeared in earlier project experiments. These are held-out tests for this pipeline, not project-wide untouched or prospective evidence. Correlated origins and mirrored long/short queries are not independent observations. Paired bootstrap resamples UTC days with all assets/queries together. Few calendar days limit those intervals.

## Run

Use the existing GPU environment from the workspace root:

```powershell
$py = 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe'
& $py -B -m checks.check_barrier_survival
& $py -B -m checks.check_barrier_training
& $py -u -B -m scripts.train_barrier_survival --output runs/barrier-survival-repeat --dataset data/barrier-survival-20261022
& $py -B -m scripts.audit_barrier_survival --run runs/barrier-survival-repeat
```

Output directories refuse overwrite. Existing datasets and model weights are checked by hash. The original training sources for the first run are retained in `runs/barrier-survival-20261022/source-snapshot`; subsequent API-only support/provenance corrections do not change the trained network or likelihood.

## Predict

The request JSON contains:

```json
{
  "symbol": "BTCUSDT",
  "decision_at": "2026-07-20T12:00:00Z",
  "entry": 100000,
  "tp_fraction": 0.01,
  "sl_fraction": 0.005,
  "side": "long"
}
```

This is a schema example, not an observed market quote. Fractions are decimals, so `0.01` means one percent. Alternatively supply absolute `tp` and `sl` prices. Specify each barrier exactly once. `side` defaults to `long`; mathematical short probabilities do not imply the exchange permits opening a short position.

The history CSV must contain timezone-aware candle-open `timestamp`, `open`, `high`, `low`, `close` and `volume`, with at least 360 contiguous final M1 candles. Its last candle must finish exactly at `decision_at`. Future or stale final candles are rejected. Only the four trained symbols are supported.

```powershell
& $py -B -m wallex_lab.barrier_survival `
  --model runs/barrier-survival-20261022/models/late `
  --history runs/barrier-survival-20261022/arbitrary-barrier-audit/example-history.csv `
  --request runs/barrier-survival-20261022/arbitrary-barrier-audit/example-request.json `
  --output runs/barrier-survival-20261022/example-cli-response.json
```

The audit writes a real saved-model request/history/response example. Returned support flags describe extrapolation beyond trained distances, ratios and entry offsets; they do not certify accuracy when empty. Member standard deviations describe disagreement, not a confidence interval for the true event probability. The response includes the protocol, dataset and weight hashes.

No-touch at 24 hours supplies no probability for eventual resolution after 24 hours and no continuation value. Evaluating an actual trading policy requires separate fill, cost, risk, holding and end-of-challenge settlement rules.
