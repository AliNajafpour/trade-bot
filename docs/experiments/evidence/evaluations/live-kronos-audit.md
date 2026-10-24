# Live Kronos mature-prefix audit

Snapshot 2026-10-21T17:21:00.538288+00:00. Fully mature one-hour forecasts: 0. Scored origins: 44; scored points: 137.

Only previously unknown, actually observed completed candle closes are scored. These short prefixes do not establish one-hour accuracy or profit.

| Horizon min | Points | MAE % | Flat MAE % | Bias % | Direction correct | Direction N | q10–q90 coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 40 | 0.9162 | 0.1260 | 0.8632 | 0.6666666666666666 | 21 | 0.625 |
| 10 | 40 | 0.4519 | 0.1854 | 0.3693 | 0.6956521739130435 | 23 | 0.675 |
| 15 | 31 | 0.7273 | 0.2726 | 0.6582 | 0.6 | 20 | 0.742 |
| 20 | 19 | 0.6045 | 0.2871 | 0.5186 | 0.5833333333333334 | 12 | 0.684 |
| 25 | 7 | 1.1550 | 0.2780 | 1.0850 | 0.75 | 4 | 0.857 |

Markets with no mature observed points have NA metrics in JSON. Repeated forecast origins and horizons overlap and are not independent samples. Direction excludes actual unchanged closes. Generated CLOSE paths are assessed independently of malformed OHLC/volume output.

Immutable copied outcomes, hashes and full per-market metrics: [snapshot](../experiments/paper-kronos-hour-20261021-1657/evaluation/20261021T172100538288Z/audit.json).

Invalid/unscorable forecast records: 0.
