# Short-history arbitrary-barrier survival results

Historical price-touch evaluation. No orders, fills, fees or realized trading return are claimed.
The source dates have appeared in earlier project experiments; this is a held-out test for this new pipeline, not project-wide untouched or prospective evidence.

Dataset: 17,236 origins, 24 barrier queries per origin, four native Wallex USDT markets.
Unresolved within-minute double touches retain unknown cause. Opening-price touches resolve the cause. Missing minutes censor observation. All split boundaries purge 24 hours.
The 180/360-minute choice uses only early-fold selection NLL. Each fold uses three seeds, separate calibration dates, and a separately calibrated query-only baseline.

|Fold|Context min|Calibration temperature|Horizon min|Baseline log loss|Raw ensemble log loss|Calibrated log loss|Calibrated Brier|Scored / total|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|early|360|1.0446|5|0.23170|0.23244|0.23442|0.10528|46080 / 46080|
|early|360|1.0446|15|0.46718|0.46386|0.46588|0.24727|46077 / 46080|
|early|360|1.0446|60|0.83825|0.83377|0.83351|0.49052|46076 / 46080|
|early|360|1.0446|240|0.94552|0.93104|0.92769|0.56691|46076 / 46080|
|early|360|1.0446|1440|0.73805|0.73878|0.73642|0.49466|46076 / 46080|
|late|360|0.9628|5|0.22714|0.22944|0.22801|0.10499|55296 / 55296|
|late|360|0.9628|15|0.45451|0.45239|0.45045|0.24132|55296 / 55296|
|late|360|0.9628|60|0.80371|0.79634|0.79235|0.46696|55296 / 55296|
|late|360|0.9628|240|0.89598|0.89010|0.88699|0.54479|55296 / 55296|
|late|360|0.9628|1440|0.75616|0.74261|0.74252|0.49564|55296 / 55296|

Negative paired log-loss differences favor the calibrated history ensemble. See results.json for daily-block intervals, classwise reliability, symbol/geometry slices, censoring and ambiguity coverage.
Model probabilities are coherent across time; monotonicity across arbitrary barrier pairs is measured separately and is not guaranteed by this architecture.
Historical entries are last completed close with small predetermined offsets, not observed executable ask/bid prices. Inference accepts an explicit supplied entry and reports extrapolation flags.
No-touch is remaining probability mass at the horizon. It never closes a position or implies zero continuation value.

## Reproduce

```powershell
$py = 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe'
& $py -B -m checks.check_barrier_survival
& $py -B -m checks.check_barrier_training
& $py -B -m scripts.train_barrier_survival --output runs/barrier-survival-repeat --dataset data/barrier-survival-20261022
```

Use a new output directory. The existing frozen dataset is hash-verified, not overwritten. Final model bundles are under models/early and models/late inside the run.

## Additional artifacts and limits

- [Arbitrary-barrier audit](arbitrary-barrier-audit/report.md) includes distinct-query scores after excluding requests that collapse to training geometries under clipping.
- Evaluation plot (`evaluation.png`; local supporting artifact) shows late temporal scores and classwise reliability.
- Actual CLI response (`example-cli-response.json`; local supporting artifact) includes five horizons, conditional-on-touch probabilities, support flags and model/data hashes.
- [Verification receipt](verification.json) records passing model checks, eight verified weight files and 96/98 whole-project checks passing across the existing GPU/portable runtimes. The two remaining failures are unrelated legacy paper-resume and unfinished backtest-artifact checks; full failure evidence is retained in full-suite-original.json.

Improvements are modest and depend on horizon. The five-minute test does not establish an advantage over the query-only baseline. Time coherence is guaranteed by construction; cross-barrier no-touch monotonicity is not guaranteed and sampled violations are recorded. These results do not establish calibrated probabilities under future distribution shift or profitable execution.
