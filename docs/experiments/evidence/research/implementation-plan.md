# Wallex six-strategy experiment

Goal: implement and run six complete, comparable research strategies today using actual public Wallex market data. Preserve raw responses, failures, model outputs and cost assumptions. No authenticated orders are authorized by this experiment.

User approved the six-variant design and explicitly requested immediate parallel implementation. Work in this directory; it is not a Git repository. Existing Trade-Bot remains read-only. Reuse portable functions and record their origin and hashes; do not transplant gold data, trained gold signals or venue calendars.

## Shared interfaces

- Package: `wallex_lab`; Python runtime: existing `D:/ALI/Programming-AI/Personal Projects and Tests/Trade-Bot/.venv-gpu/Scripts/python.exe`. No environment edits outside this workspace.
- Canonical data frame: sorted unique UTC `timestamp` column describing candle OPEN time, numeric `open,high,low,close,volume`; `symbol` optional. Closed bars only. No interpolated OHLC. Missing intervals flagged and trades/labels crossing gaps excluded.
- `wallex_lab.features.make_features(frame)` returns a finite pandas feature frame of equal length using past/current COMPLETED bars only. `forward_returns(frame, horizon=3)` returns fractional future close returns, NaN for immature/gapped labels. `split_indices(n)` returns `(train_end, val_end)` with chronological 60/20/20 fractions. Callers purge `horizon` rows before every target boundary.
- Each strategy module exposes `generate(frame, train_end, val_end, config, run_dir) -> dict`: `target` is length-N float array in [0,1], target allocation known at that candle CLOSE; `metadata` JSON-compatible, including models, selected settings, source and validity caveats. Optional `score` length N. No training or selection on TEST labels. Config includes `horizon`, `seed`, `fee_bps`, `slippage_bps`, `max_exposure`. Failure must raise or return explicit `status=blocked` with reason, never silently substitute another strategy.
- Engine: `backtest(frame, target, config, start=0) -> dict` with `metrics`, `trades` and `equity` JSON-compatible or pandas convertible. Decisions at candle close execute at NEXT candle open, long-only spot cash/inventory, fractional quantity, capital 5,000,000 toman, fees and slippage per side, max exposure, explicit end liquidation. No leverage, shorts, fabricated order-book fills or guarantees of fills. Record gaps and execution limits.
- Strategies: `rules.py`, `trees.py`, `neural.py`, `kronos.py`, `agent.py`, `fusion.py`. Fusion may consume numeric expert outputs via config; parent handles dependencies. Each has an offline synthetic runnable check in `checks/`.
- Data module `data.py`: public HTTP only, raw bytes and receipt hashes, actual market metadata, history with pagination/dedup/closed-bar validation, current order-book snapshots when possible; executable `python -m wallex_lab.data --help`. Canonical CSVs under `data/market/` and acquisition manifest. Never treat public market availability as competition allowlist.
- Parent owns CLI, run orchestration, source-reuse manifest, README, consolidated evidence, integration tests.

## Parallel tasks and checks

1. Data: public endpoint discovery, acquisition, provenance and quality checks.
2. Shared features/execution: causal features, purged labels, fee-aware cash ledger and gap validation; test delayed execution, costs, liquidation and future perturbation.
3. Six strategy implementers each own one module and one check. Write and run a meaningful failing check before implementation, then pass it. CPU thread count at most 2; do not launch heavy training while other GPU jobs run. Tests use tiny synthetic inputs.
4. Research agent: primary sources on crypto agents, academic/industry/competition/open-source methods and Wallex API; save source-backed findings and actionable implications.
5. Parent: integrate; execute real-data training sequentially where GPU/RAM contend; compare all strategies on identical held-out timestamps and explicit baseline/stress costs. LLM calls are real with persisted receipts; historical LLM results flagged for knowledge contamination. If access fails, preserve failure and try supported alternatives, never label synthetic outputs as real.
6. Independent review and repairs; rerun applicable checks; report actual metrics and remaining gaps. Research protocol and constraints remain in final evidence.

## Rulings

- User's immediate execution request supersedes additional conversational approval stages. No worktree/commits are applicable without a Git repository.
- Public downloads and model calls are authorized; real exchange orders require account setup and are outside this test run.
- Candles support approximate execution only. Historical fee/precision/credit-account conditions are unverified unless returned from an authorized account. Assumptions and stress costs must be visible.
- Parallel authorship is authorized; heavy fitting is scheduled to avoid the prior project's memory failures.
- Independent review rejected hindsight liquidation at a pre-gap close. The corrected engine cancels stale crossing orders and liquidates held inventory only at the first observed post-gap OPEN. Prior initial runs are retained separately; final runs use `runs/wallex-20261021-causal` in fresh processes and regenerate model selection and baselines.
- Base selection uses 60-70% and fusion selection uses 70-80%, keeping the final 20% untouched by either selection stage. Expert NPZ/data/config/source identities are verified before stacking.

## Progress

- Complete: six independent strategy modules, shared features/execution/risk, public acquisition, context collector, current-only AI and short shadow collector.
- Complete: 60 days of M1/M15 data for four TMN markets, 352 original raw receipts verified, all aggregate CSVs reconstructed.
- Complete: six families on BTC, four numerical families on the other three markets (18 main runs), and three rule/numerical families at a six-hour label horizon on four markets (12 exploratory runs).
- Complete: 10 actual historical LLM decisions with one preserved transient timeout/retry, plus a separate current news/flow decision. Historical results never consume current news.
- Complete: 12 offline checks passed; all 30 real-data strategy ledgers replayed exactly under three cost scenarios and selected fitted artifacts reloaded without training.
- Complete: initial lookahead, stale baseline, expert-integrity, prospective-mode and evidence-schema findings repaired; immutable failed/initial runs retained. Practical exchange-filter approximations remain disclosed.
- Results: reports/results.md, reports/results.json and reports/comparison.csv. No active strategy demonstrated positive net results at the tested costs. All-cash is abstention, not a challenge-winning result.
