# Dense current-history barrier model

User explicitly requests refreshing through now, more training history, decisions every five minutes, many more arbitrary TP/SL queries and learning candle shapes without supplying technical-pattern labels. Execute the requested extension inline and preserve the prior experiment.

## Frozen design

- Request one year of native Wallex M1 for BTCUSDT/ETHUSDT/SOLUSDT/SUSDT, ending at 2026-10-22T16:00:00Z, the last full minute at task start. Public HTTP only, raw receipts/hashes, no interpolation and no private account access. Report returned coverage, no invented candles or availability.
- Decisions every 5 minutes; 128 queries per origin, 64 per orientation. Mix anchor geometries with continuous history-volatility-scaled and absolute-percentage distances. Entry offsets and all query generation use history only and deterministic sampling.
- Reuse partial-cause competing-risk labels and replace repeated per-query future scans with cumulative extrema/searchsorted, tested against literal cases and the previous scalar algorithm. Preserve gaps/censoring and unknown same-minute cause.
- Store raw minute arrays once plus indexed origins and memory-mapped labels/distances. Build normalized batches from GPU-resident raw arrays; do not duplicate every overlapping six/twelve-hour history. All origins and all 128 queries are used in every epoch of the optimized main run.
- Version normalization: retain legacy v1 inference; new v2 uses relative log price, history-only volatility, relative volume and explicit scale/volatility query information. Check invariance under multiplying every price by a constant. No named-pattern labels, pivots, technical indicator recipes or future-derived levels.
- Compare 360/720-minute contexts with a shared convolution/Transformer/cross-attention model. A learned positional sequence gives it access to shape/order, not just summary statistics. Verify history sensitivity and measure a temporally shuffled-history diagnostic; this cannot establish discovery of a particular named pattern.
- Two expanding calendar folds, 24h purges, separate early-stop/calibration/test periods. Most recent partial 24h remains censored and is not scored as a complete future. Train from scratch, 3 seeds; separately calibrated query-only baseline. Freeze all choices/calibration before test scores. Report dependence of five-minute origins using daily paired blocks.

## Tasks and ledger

- [x] Download audited current-year history in a new immutable data directory.
- [x] Test-first vectorize labels, implement versioned normalization and indexed dataset with 128 queries/5-minute origins.
- [ ] Train streaming short-history models, validate context, calibrate independently and evaluate both folds; audit arbitrary query/history sensitivity and normalization.
- [ ] Independently review, run meaningful regression checks, verify saved bundles and publish local results/examples.

Ruling: "Now" is an explicit immutable acquisition cutoff at task start, not a moving end time during training. A later wall-clock finish does not silently change training/test data.
Ruling: More dense origins are correlated, not independent evidence. Avoid random train/test splits and retain the 24h purge and daily-block uncertainty.

Acquisition complete: 2,100,229 native M1 candles, 2,171 missing calendar minutes across assets, four sources ending at 2026-10-22T15:59:00Z. Gaps retained. Dense indexed dataset complete: 414,289 origins and 53,028,992 queries, under 1GB instead of duplicating all overlapping windows.

Independent review reconstructed 20,992 labels at 164 stratified real origins exactly; all origins passed cadence/history/decision-clock/purge checks. Fixed sparse-volume unit invariance, pre-full-coverage checkpoint selection, history-shuffle membership and float32 support-bound rounding. Regression checks passed before main training.

Performance ruling: The initial CPU-streaming pilot was stopped during its first epoch before selection/test scoring. Measured CPU batching was 0.0507s at four threads; GPU-resident raw/label arrays reduce batching to 0.0061s. GPU/CPU x matched exactly and maximum q difference was 2.4e-7. GPU memory profiling established batch512 fits both contexts on the available 6GB device. Preserve pilot artifacts; main run is runs/barrier-survival-dense-gpu-20261022.

Query-schedule ruling: The optimized main run trains all 128 requests for every origin in every epoch, with no query subsampling. This reuses each encoded history across all queries. The original balanced32 schedule remains available for reproducing the pilot, and checkpoint eligibility explicitly requires complete query exposure in either mode. All actual context/model/calibration choices freeze before test scoring.

## User extension: one day, two days and one week of input

The user requested longer input windows while the 6/12h run was still in validation. That run was stopped with its weights/curves intact, before any calibration/test scoring. It is a superseded context pilot, not a finished result.

- Compare encoder contexts 360, 720, 1440, 2880 and 10080 minutes, with identical fit/selection/calibration/test origin indices for every candidate. Require a full contiguous week for the common-origin cohort; retain original indexed labels without rebuilding or altering them.
- Hierarchical convolution/Transformer layout: last <=720 minutes at M1; middle <=2880 minutes pooled into 5m bars for <=1d contexts or 15m bars for longer contexts; full context pooled into 15m bars for <=1d or 60m for longer contexts. Pool OHLC with first/max/min/last, other normalized channels by mean; learned patch sizes remain 6/3/1. Week has 352 tokens, not 10080 unaggregated attention tokens.
- Shared causal volatility remains trailing60/360 M1, and normalization requires at least720 input minutes for legacy comparability. Longer contexts use their own history in the sparse-volume fallback. No future samples enter coarse bars. This compares the window plus its stated hierarchical resolution, rather than falsely claiming resolution is identical.
- Screen all five contexts on early-fold selection loss with seed42; freeze the selected length for both temporal folds, then fit two more seeds. No test score selects a window or calibration.
- New immutable output: runs/barrier-survival-multiday-20261022. Keep all earlier protocols/snapshots and a stopped-run receipt.
- Checks must cover week-old candle influence, rejection of truncated inputs, true contiguous clock eligibility, CPU/GPU normalization parity for all five lookbacks, legacy model reload and calibrated time coherence.

Implementation check: all five forward paths, common origins, week-old influence and truncated-history rejection pass. Actual CUDA/CPU parity passed at all five lengths on two real origins. No longer-context training outcome is claimed at this stage.
