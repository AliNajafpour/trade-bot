# Full-data barrier comparison implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task. Current checkout is not a Git repository; use the existing authorized workspace, separate owned files and the adjacent progress ledger.

**Goal:** Replace unequal1024-origin auxiliary pilots with models trained on exactly the original model's complete fit/selection cohorts, then compare on complete common chronological calibration/test cohorts.

**Architecture:** Preserve the original dataset, main run and numerical sources. A resumable orchestrator builds compact feature caches, trains all neural candidates with streamed batches and all CatBoost candidates with explicitly reported CPU continuation across every query, freezes all choices, calibrates and tests both folds. Existing original weekly ensembles and full query-only baselines are reused.

**Tech stack:** Existing NumPy/Pandas/PyTorch/CatBoost/scikit-learn runtimes. No new dependency. Python is `D:/ALI/Programming-AI/Personal Projects and Tests/Trade-Bot/.venv-gpu/Scripts/python.exe`.

**Spec:** User explicitly requires all competitors to receive the same training amount as the original weekly model and share chronological test evaluation. Frozen encoders plus a trained head remain supervised transfer, not native zero-shot.

## Global constraints

- Common10080-minute eligibility mask and original purged folds, exact indices without subsampling. Early fit274355/selection14976/cal5380/test31504; late fit313543/selection14976/cal14976/test11566.
- All128arbitrary ex-ante queries per origin. All five horizons5/15/60/240/1440. Correct unknown-cause/censor likelihood; no invented SL-first tie outcome.
- Three seeds42/43/44 per neural/boosted candidate. Neural max12epochs, patience3, AdamW lr0.0006/decay0.03, effective batch512, clip1, BF16 forward/FP32 likelihood on CUDA, matching the main supervised budget. TCN uses microbatches and gradient accumulation without changing effective batch/coverage.
- Features/scalers fit only original fit rows. No calibration/test-based model selection or refitting. Freeze all candidates before test scoring. Keep pretraining-overlap uncertainty explicit.
- Original main GPU process is not interrupted. GPU cache/training waits for main completion and adequate free VRAM. Stop superseded CPU pilots only after recording their status and updating the follow-up.
- Disk-bounded FP32 memmaps for pooled candles500dims, Kronos Small1536, Kronos Base2496, Chronos-22304 over the sorted union of eligible cohort indices. No full TCN history cache. Original relative-v2 TCN histories derive on demand.
- Checkpoint/progress writes must be atomic; caches flush before advancing their saved watermark. Resume must reject changed source/dataset/cohort/model identities.
- CatBoost cannot materialize all repeated candle/query/bin columns in RAM. Use original500candle features, exact touch/cause likelihood factorization and native CPU `init_model` continuation over a seeded permutation of all128query IDs per full pass, each stage visiting every fit origin. Fixed fit-only quantization borders, two added trees per query stage, max12full passes and patience3 only after complete coverage. This is explicitly **streamed continued boosting**, not an identical algorithm to one full in-memory boosting fit. No hidden sample caps.
- All original sources remain frozen; new files alone implement the full-data comparison. Pilot scores are retained as historical diagnostics and excluded from the full-data ranking.

## Tasks and interfaces

### 1. Resumable feature cache

- [ ] Own `wallex_lab/barrier_full_cache.py` and `checks/check_barrier_full_cache.py`.
- [ ] `ensure_cache(data, indices, root, kind, device='cpu', batch_size=8)` returns a compact `(len(indices), width)` memmap; indices sorted unique. Kinds `candles`, `kronos-small`, `kronos-base`, `chronos-2`. Sidecar `<kind>.json` records watermark, readiness, cohort/source/model identity and final hash; `<kind>.npy` stores values.
- [ ] `cache_rows(cache, indices, selected)` verifies exact global-index mapping.
- [ ] Vectorized rolling OHLCV/native-calendar prep reproduces old pilot helpers. Kronos native tokenizer/decode_s1; Chronos native separate5-var scale groups/CLOSE REG. No GPU calls while main active; CPU parity checks allowed.
- [ ] Real causal/unit/parity and interrupted-cache-resume checks.

### 2. Full-data neural training

- [ ] Own `wallex_lab/barrier_full_neural.py` and `checks/check_barrier_full_neural.py`.
- [ ] `train_neural(data, fit, selection, root, kind, seed, args, cache=None, cache_indices=None)` returns member metadata, saves `best.pt`, `scaler.npz`, `fit.json`, `last.pt`, progress/curve. Kinds `features` (existing KronosBarrierHead, cache required) and `tcn` (existing TemporalBarrier, on-demand sequence_windows).
- [ ] `collect_neural(data, indices, member_root, args, cache=None, cache_indices=None)` returns CPU logits `[N,128,5,3]`, loading verified config/weights/scaler.
- [ ] `args`: device, batch512, microbatch32, epochs12, patience3. Train-only streaming scaler, original likelihood and complete query coverage, resumable model/optimizer/RNG epoch checkpoints. No full cached TCN history.
- [ ] Tiny actual training, no dropped last batches, all queries, fit-only stats, checkpoint/inference roundtrip and resume checks.

### 3. Full-data streamed CatBoost

- [ ] Own `wallex_lab/barrier_full_trees.py` and `checks/check_barrier_full_trees.py`.
- [ ] `train_trees(data, fit, selection, root, seed, args, cache=None, cache_indices=None)` saves touch/cause `.cbm`, fit metadata, last state and progress; returns metadata. Cache=None means query-only control. Args epochs12, patience3, tree_iterations2, threads4.
- [ ] `collect_trees(data, indices, member_root, cache=None, cache_indices=None)` returns CPU logits `[N,128,5,3]` via existing factorization/inference helpers and bounded batches.
- [ ] Every full pass processes all128query IDs and every fit origin before a checkpoint is eligible. Fit-only frozen quantization borders. Unknown cause trains touch only; censoring contributes observed survival bins only. Exact checkpoint coverage receipts and resume.
- [ ] Tiny actual continued fits, exact label-factorization parity, full coverage and save/load/resume checks.

### 4. Orchestration and benchmark

- [ ] Own `scripts/train_full_barrier_comparison.py`, its runnable check and new run protocols/cohorts.
- [ ] Prepare exact two-fold cohorts and sorted cache union under `runs/barrier-full-compare-20261022`. Build raw candle cache on CPU while waiting for main; then GPU caches and neural members. Trees may use a separate bounded CPU worker while GPU models run.
- [ ] Resume finished members; no duplicated active pipeline through a native process-held file lock. Record source hashes and verify them before reuse.
- [ ] Freeze all full-data members, calibrate each ensemble on identical full calibration origins, score complete chronological tests in both folds, report raw/calibrated proper NLL/horizon metrics/conditional probability/coherence/day-block uncertainty/timing. Original weekly and query-only bundles are references with identical supervised data counts.
- [ ] Update the existing `tp-sl` thread heartbeat to monitor/complete this full-data run rather than requiring superseded pilots to finish. Preserve old artifacts; stop only old CPU pilot processes, not the main process.
- [ ] Independent scoped reviews and final integrated runnable checks before the queued production launch.
