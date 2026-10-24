# vNext runner implementation receipt

Task 4 implementation is ready for parent review. Production has not been launched by this worker. Existing comparison files and jobs were not changed.

Changed files are `scripts/train_barrier_vnext.py`, `checks/check_barrier_vnext_runner.py`, and this receipt.

The production CLI fixes seven independent variants, seeds 42/43/44, two dataset folds, week-contiguous common eligibility, all fit origins and all 128 queries per epoch, effective batch 64, microbatch 16, maximum 12 epochs and patience 3. Pretrained and combined variants each optimize two fit-only masked reconstruction epochs before supervised optimization. AdamW uses learning rate 0.0006, decay 0.03 and gradient clipping 1. Forward uses BF16 when supported on the leased CUDA device; likelihood stays FP32. Tiny CPU checks override budgets only through direct function calls on synthetic data.

Initial, 200-batch and epoch-boundary atomic checkpoints retain model, optimizer, best weights, order/cursor, NumPy order RNG, CPU Torch RNG, masking generator RNG and CUDA RNG when applicable. Resume checks source hashes, dataset identity, cohort hashes, versions, flags and optimizer settings. Member receipts retain counts and total compute time. Checkpoint writes preserve a 5 GiB disk reserve plus temporary-file headroom. OOM stops with an actionable error and retains the last committed checkpoint; it does not change the frozen microbatch.

The queue waits for acquisition readiness, builds and validates indexed CPU data, waits for the old comparison's `results.json`, then acquires its shared GPU lease before any CUDA inspection or allocation. Partial, failed or interrupted acquisition leaves `source_blocked` with the manifest path and resume instructions. Source changes while queued require a restart.

All 42 member choices freeze before calibration. Every temperature freezes before test access. Calibration searches a fixed grid containing 1 using the three-seed mixture likelihood. Evaluation streams proper path-mixture NLL, coherent ensemble incidences, all five horizon log losses/Brier/reliability, known-hit conditional TP metrics, and UTC-day sums/counts. Paired day-block intervals use these sufficient statistics. No full per-query prediction arrays are written.

Validation run on the provided Python 3.11 GPU environment, with CPU devices only:

```powershell
& 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe' -B -m checks.check_barrier_vnext_runner
& 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe' -B -m scripts.train_barrier_vnext --help
```

Both exit 0. The runner check was first observed RED because the runner was absent. A later source-readiness check was observed RED before `wait_source` was implemented. Final checks establish bitwise equal fresh/resumed weights for interruptions during both fit and masked pretraining, every-origin/128-query counts, source/cohort rejection, three-seed mixture NLL against direct calculation, calibration grid behavior, streamed daily metrics and no-CUDA waiting. The real tiny coordinator trains two variants with three seeds on separate fit/selection/calibration/test cohorts, asserts the freeze gates, writes complete reports, then replays to an identical results checksum.

Production command for the parent to launch after review:

```powershell
& 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe' -u -B -m scripts.train_barrier_vnext --source data/barrier-binance-history-20261023 --dataset data/barrier-vnext-20261023 --output runs/barrier-vnext-20261023 --device cuda --wait-comparison runs/barrier-full-compare-20261022 --gpu-lock 'C:\Users\Lenovo\AppData\Local\Temp\wallex-barrier-full-compare-20261022-cache\gpu.lock'
```

No GPU throughput/memory test or broad legacy GPU suite was run, as instructed. CUDA/BF16 exact-resume behavior and microbatch-16 memory use still require validation after the old comparison releases the GPU. The full experiment's runtime is not estimated from the tiny synthetic checks. These historical native-venue price-touch metrics do not establish executable fills, trading returns or prospective generalization.

Integration review repair: `prepare()` can now resume after protocol commit or a torn snapshot copy. It recovers missing snapshots only from current sources matching the frozen protocol hashes, verifies and fsyncs a temporary copy with a writable Windows descriptor, then uses atomic replacement. Existing corrupt completed snapshots and changed current sources are rejected. Root progress now includes the actual Python PID for ownership checks.

The new regression was observed RED with `FileNotFoundError` after an interruption immediately following protocol commit. Final full CPU runner check exited 0 in session 14481, Python PID 31188, with both snapshot interruption cases recovering and both corruption/source-change cases rejecting. Existing resume, mixture, stage-gating and tiny coordinator checks stayed green. Coordinator replay retained identical results hash `f98e04dd7650029b189486c09b948d783602e6ad43c5529891864cc78cd36050` within that run. No production launch was performed by this worker.

Final SHA256 identities for parent review and queue launch:

- `scripts/train_barrier_vnext.py`: `c4317f9eb813d254e6babd4d654ce377e2115e1ceaf72e50a666c05681c19f19`
- `checks/check_barrier_vnext_runner.py`: `44352a320dfb52d835e6388a0a6de92aa810b3e1ebc08b8a67894fde91c947a3`
