# Barrier survival implementation plan

> Execute inline using executing-plans. User explicitly requested starting the approved architecture; no additional design gate is needed.

**Goal:** Build and actually train/calibrate/evaluate the short-history arbitrary-barrier model.

**Architecture:** Audited causal M1 dataset, shared intraday multi-scale encoder, query-conditioned competing risks, three-seed ensemble and separate chronological calibration.

**Tech stack:** Existing NumPy, pandas, PyTorch and matplotlib; no installation.

**Spec:** ../specs/2026-10-22-barrier-survival-design.md

## Global constraints

- Horizons exactly 5/15/60/240/1440 minutes. No horizon liquidation.
- Preserve existing trading code and all old run artifacts. Source directories are read-only.
- No git checkout exists here; work in the authorized workspace without git initialization/commits.
- Dataset/model/run directories refuse overwrite. Keep provenance and runnable checks.

## Review focus

- Same-minute ambiguity must never become a certain SL.
- Missing minutes stop observation; they are not a no-touch label at later horizons.
- Every query from one origin remains in one calendar split, with 24h purge.
- Entry and barrier features never use future opens, volatility or extrema.
- Save/reload and calibration must preserve sums and nondecreasing incidence.

## Tasks

1. [x] Write failing checks for labels, partial likelihood, leakage, probability coherence and inference validation. Implement `wallex_lab/barrier_dataset.py` and verify the data/label checks.
2. [x] Implement `wallex_lab/barrier_survival.py`: multi-scale model, risk likelihood, probability conversion, temperature calibration and save/reload request interface. Pass the model checks, including a real gradient update.
3. [x] Implement `scripts/train_barrier_survival.py`: immutable protocol, verified dataset, temporal folds, validation-only lookback selection, 3-seed ensembles, query-only baseline, calibration and saved predictions/metrics. Run actual CUDA training.
4. [x] Independently review data/time flow and fixes; run the full existing check runner, audit the resulting artifacts, and write reproducible results and commands.

## Progress ledger

- Planning: approved in chat; explicit authorization to execute retained.
- Data: 4 verified Wallex M1 files available for 90 days; gold excluded due unresolved clock/quote provenance.
- Ruling: Historical price-touch labels use last completed close as entry proxy. No executable bid/ask history exists; costs/fills/profit are outside this model's claim.
- Ruling: Retain five output bins and censor within-bin partial exposure at the preceding completed output boundary. This sacrifices within-bin censor information without inventing survival beyond observation.
- Tasks 1/2: Observed initial missing-module failures; implemented and passed causal/ambiguous/censor/short-side/boundary/purge tests, literal likelihood and incidence tests, real gradient update, temperature improvement and model reload/request checks.
- Task 3: Actual dataset/training process launched in `runs/barrier-survival-20261022`; protocol frozen before data generation. No external downloads were needed.
- Review: One independent read-only reviewer found no time leakage or likelihood/mask defect. Corrected clipped support flags and missing data hash, with a failing regression check then passing check. Added a separately checked off-grid/monotonicity audit. Original training sources retained under the run's `source-snapshot/` before API-only corrections; numerical training code is unchanged.
- Whole-project check run in progress; `check_council_chronos.py` currently fails on the existing GPU runtime's missing `langgraph`, outside the new model pipeline.
- Tasks 3/4 complete: Actual CUDA fits and both separate calibrations finished, with frozen choices before test scoring. Validation selected 360 minutes. Late held-out calibrated log loss beats query-only at 15/60/240/1440 minutes, not 5 minutes. No test-driven tuning followed.
- Final audit: 256 held-out origins, 6,144 random queries plus monotonicity grids. Clipping makes 20.638% of random query vectors match training queries; separate scores exclude all 1,268 matches. TP/SL monotonicity had no violations on the sampled grids; no-touch monotonicity had violations, retained as an explicit model limitation.
- Final review verified all source snapshot, dataset, protocol and eight saved weight identities, and equality of CLI/audit example responses. Corrected the report's unresolved-double-touch wording and added distinct-query reporting.
- Verification: Both new check modules pass. Of 98 whole-project checks, 90 pass directly in GPU runtime, six more pass using the already-installed portable dependencies/interpreter. Remaining unrelated failures: `check_paper_resume.py` expects a missing legacy cash field; `check_suite_backtest_outputs.py` requires completion of a separate pending backtest artifact. No dependency was installed. See run/verification.json and retained detailed receipts.
- Deferred scope: No order execution, profitability backtest, long-history branch, gold, news, pretrained checkpoint or prospective market validation was added.
