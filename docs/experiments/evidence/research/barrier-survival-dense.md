# Dense yearly candle model

This extension follows the requested five-minute decision cadence, larger TP/SL coverage and current historical data. It supplies no named technical-pattern labels.

## Data

The public Wallex acquisition is frozen at **2026-10-22 16:00 UTC**, or **19:30 Asia/Tehran**, the latest completed minute when this task began. All four sources end with the candle opening at 15:59 UTC. The requested start is 2025-10-22 16:00 UTC. The returned 2,100,229 native M1 candles contain 2,171 missing calendar minutes across BTCUSDT, ETHUSDT, SOLUSDT and SUSDT. Gaps remain unfilled.

Sources, response entity bytes, request/receive clocks and hashes are under `data/barrier-history-dense-20261022`. The indexed dataset under `data/barrier-dense-20261022` has **414,289 origins and 53,028,992 barrier requests**. Every origin has 720 contiguous historical minutes. Decisions occur on UTC five-minute boundaries. All output horizons remain exactly 5/15/60/240/1440 minutes.

Each origin has 64 pairs per direction: 12 anchor geometries, 26 continuously sampled volatility-relative geometries and 26 absolute-percentage geometries. Distances are clipped to log-distance .0001 through .15, with ratios generally bounded by 8. All sampling and entry offsets use only the origin and its observed history. The entry proxy is a completed close with a small predetermined offset, not an observed executable fill.

First-touch causes remain unknown when an opening price cannot resolve a minute that touches both levels. Missing future minutes censor observation. Fully observed no-touch outcomes are distinct from unobserved horizons. The vectorized cumulative-extrema labeler was compared to the original scalar labeler on randomized cases covering both directions, gaps and source exhaustion.

## Learning shapes without supplying their names

The one-minute branch encodes consecutive candle patches with learned convolutions. Hierarchical pooled branches preserve broader price structure. The new comparison includes 6h, 12h, 1d, 2d and 1week inputs on identical origins with complete continuous week history. Near history retains M1 detail, middle history uses 5/15m pooling, and the far branch uses 15/60m pooling. A week uses 352 learned tokens. Positional Transformer attention connects these patches across the selected span. Barrier-conditioned cross-attention asks how that representation relates to the requested levels.

This gives the network access to repeated touches, narrowing ranges, reversals and the relative heights of past highs/lows. Training optimizes observed first-touch probability, so any useful representations must help that task. There is no supplied triangle, head-and-shoulders, pivot, support/resistance or pattern-trading rule.

The network is not a named-pattern classifier. Architectural capacity, history sensitivity and a shuffled-history score do not prove that a particular human-named pattern has been discovered. The temporal evaluation compares against a query-only model to measure whether history adds predictive information, then perturbs order while retaining exactly the selected candles.

## Normalization

The new version is `relative-v2`:

- Each OHLC price becomes log(price / last completed close), scaled by history volatility.
- Volatility is the maximum of trailing 60/360-minute log-return standard deviations and a numerical floor.
- Volume is relative to recent median/mean volume; zero-volume periods fall back to whole-window mean, then one only for an entirely zero-volume window.
- The request uses relative TP/SL distances, distances relative to volatility, their ratio, volatility itself, entry gap and direction.

For example, 100 to 101 and .01 to .0101 both represent the same one-percent movement. Tests multiply all prices by one million and volumes by ten thousand and verify that normalized features remain unchanged. Original `relative-v1` bundles keep their preprocessing.

Every request supplies at least720 completed candles, or the selected context length when longer. A 360-minute encoder shares the 720-minute normalization contract. The rare sparse-volume fallback uses the required history window. No globally fitted normalization uses validation/test rows.

## Training and verification

The current main run is `runs/barrier-survival-multiday-20261022`. The earlier GPU run was stopped before calibration/test scoring when the user requested longer windows; checkpoints and curves remain intact. Shared raw source arrays and labels reside on the existing GPU; double-precision relative calculations produce float32 features. This avoids storing every overlapping window and was checked against the CPU/inference path. Every epoch uses every eligible origin and all 128 requests; no query or origin subsampling is used in this run.

The encoder has width96, three Transformer layers and four attention heads. Context selection compares 360/720/1440/2880/10080 minutes using early-fold validation only. All candidates use the same origins eligible for a contiguous week, with no filled gaps. This compares each lookback together with its stated hierarchical resolution. Three seeds form each ensemble. Query-only baselines have their own fit and calibration. Both expanding folds retain a full24h target purge and separate selection/calibration/test blocks. All fold choices freeze before test scoring. Daily-block comparisons account for correlated origins and mirrored queries.

The common cohort has **358,517 origins**. Early fit/selection/calibration/test contain 274,355/14,976/5,380/31,504 origins; late roles contain 313,543/14,976/14,976/11,566. Strict week continuity leaves five UTC days for early calibration and twelve UTC days for the late test, so large request counts must not hide limited independent time coverage. Latest weekly contiguous input currently ends at October12; shorter <=2d inputs reach the October22 acquisition cutoff. Inference examples use the latest actually eligible history and state both dates.

|Context|Recent M1 span|Middle span / pooling|Full span pooling|Attention tokens|
|---:|---:|---:|---:|---:|
|6h|6h|6h / 5m|15m|108|
|12h|12h|12h / 5m|15m|216|
|1d|12h|1d / 5m|15m|312|
|2d|12h|2d / 15m|60m|232|
|1week|12h|2d / 15m|60m|352|

These token counts include learned patches6/3/1 after pooling. They are declared parts of the comparison, not identical resolutions across lengths. The fixed relative-v2 feature clipping was audited on512 common origins: approximately1.8% of weekly historical OHLC channels reach the +/-50 limit, versus0.35% for12h. This can compress older extreme moves; no numerical preprocessing is changed after the protocol freezes.

The earlier `runs/barrier-survival-dense-20261022` is an interrupted CPU-streaming performance pilot, stopped during its first epoch before selection/test scoring. It is not the trained main result.

```powershell
$py = 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe'
& $py -B -m checks.check_barrier_dense --gpu-batch
& $py -u -B -m scripts.train_dense_barrier_survival --dataset data/barrier-dense-20261022 --output runs/barrier-dense-repeat --batch 512 --query-chunk 128
& $py -B -m scripts.audit_dense_barrier_survival --run runs/barrier-dense-repeat
```

The full request CLI remains `python -B -m wallex_lab.barrier_survival`. Point `--model` at the dense run's `models/late`; provide aware UTC minute-open OHLCV history ending exactly at the supplied decision boundary. The post-training audit saves an actual latest-history request/example and checks probability invariance under changes of price/volume units.

Five-minute origins and millions of queries are not millions of independent market days. These historical periods overlap prior project research. The results must be interpreted through the held-out scores and reliability curves, not sample count or the visual appeal of named patterns. No trading execution, transaction costs or profitability are evaluated here.
