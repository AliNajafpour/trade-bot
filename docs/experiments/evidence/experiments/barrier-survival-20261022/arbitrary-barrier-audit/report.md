# Arbitrary-barrier audit

256 held-out origins; new continuous random barriers, both orientations.

|Horizon min|Query-only log loss|Ensemble log loss|Ensemble Brier|
|---:|---:|---:|---:|
|5|0.21878|0.21722|0.09714|
|15|0.42775|0.42377|0.22659|
|60|0.82716|0.81890|0.48800|
|240|0.88053|0.87376|0.53865|
|1440|0.74434|0.72762|0.48654|

Monotonicity results:
{
  "long": {
    "tp_violations": 0,
    "tp_comparisons": 25600,
    "sl_violations": 0,
    "sl_comparisons": 23040,
    "no_touch_tp_violations": 572,
    "no_touch_sl_violations": 37,
    "largest_tp_increase": 1.1920928955078125e-07,
    "largest_sl_increase": 1.1920928955078125e-07,
    "tolerance": 1e-05
  },
  "short": {
    "tp_violations": 0,
    "tp_comparisons": 25600,
    "sl_violations": 0,
    "sl_comparisons": 23040,
    "no_touch_tp_violations": 411,
    "no_touch_sl_violations": 87,
    "largest_tp_increase": 8.940696716308594e-08,
    "largest_sl_increase": 1.1920928955078125e-07,
    "tolerance": 1e-05
  }
}

Violations are measured comparisons, not guaranteed corrections. All probabilities still satisfy the time-coherence construction.
The example request, completed history and actual saved-model response are next to this report.

The parent evaluation.png shows the main temporal holdout scores and classwise reliability.

## Query novelty correction

Clipping caused 20.638% of sampled random requests to match training query vectors. 52 of 256 origins have no distinct random requests. These are still temporally held-out origins. The distinct-query scores below exclude matching requests; full details are in query-novelty.json.

|Horizon min|Distinct query-only log loss|Distinct ensemble log loss|Scored queries|
|---:|---:|---:|---:|
|5|0.212036|0.209782|4876|
|15|0.429516|0.420818|4876|
|60|0.790488|0.776712|4876|
|240|0.878286|0.86736|4876|
|1440|0.757141|0.73551|4876|
