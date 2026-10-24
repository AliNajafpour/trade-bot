# Implementation of workflow audit

Authorized by user on 2026-10-22. Preserve existing accounts, positions, models and training.

1. Correct shared executable-return units; manager-only default; keep positive signals able to wake a review.
2. Persist structured reentry requirements and exit receipts. Never rewrite a rejected manager geometry into another trade. Keep proposal, effective decision and actual execution separately.
3. Score owned barriers against immediate liquidation using future proceeds only. Require an exact supported rescore for amendments; retain independent protective exits. Schedule horizon reviews without timed liquidation.
4. Add a prospective matched-entry, unchanged-barriers comparison. Mark the existing numeric control as unmatched.
5. Cache foundation inference on completed input and checkpoint/settings; preserve forecast origin and expiry. Record lifecycle start/stop and downtime.
6. Expose scan/decision/execution timestamps and position status; reduce default prose in dashboard. Run targeted regressions, restart hidden service, verify live persistence.

Ruling: no Git repository is present; edit this authorized workspace directly. No worktree or commit is possible here.
Ruling: model training already running is preserved. None of these workflow repairs establishes trading profitability or validates learned news weights.
Ruling: compare future liquidation value over an explicit model horizon, not eventual TP/SL outcomes. Unresolved outcomes remain marked, with no forced time exit.

Progress: tasks 1..6 implemented; deployment and final review pending.

Verified offline:
- check_council_audit_fixes.py: duplicate spread regression RED then GREEN; durable reentry, sunk costs, completed-hour cache, matched entries and rejection of a worse amendment pass.
- check_market_council.py: admission, explicit-geometry rejection, proposal history, no timed exit, slow-manager independent protection and partial fills pass.
- check_council_runtime.py: lifecycle reason regression RED then GREEN; official MCP failure to validated official REST fallback RED then GREEN; async workers pass.
- check_council_model_memory.py: heavy subprocesses serialized; protection independent.
- check_council_reference_repair.py: 176 observed missing minutes recovered without overwriting primary candles.
- check_council_dashboard.py: timestamp/position display regression RED then GREEN; live schema, percent units, literal source prose and model variants pass.
- python -m checks.check_council_chronos: causal 5/15/60m clocks, gap masks and model-selection rules pass.
- check_council_broker.py: fee-inclusive stop risk, partial exit basis and idempotence pass.

Ruling: the additional same-entry comparison is an independent cohort per buy. Overlapping reentries cannot be advertised as a funded $100 portfolio. Old numeric-account comparison remains explicitly unmatched.
Ruling: protective price exits and discretionary CLOSE remain available without any holding-model score. An unsupported or economically worse amendment retains the existing barriers.
Ruling: no automatic process supervisor is added. Normal/requested/fatal stop reasons and heartbeat gaps are recorded; paper protection still requires a running process.
Ruling: no hindsight rebuild of past reentry conditions or trades. New structured conditions apply prospectively.

Independent review found two Important restart gaps. Both fixed and regression-tested: completed sells repair exit guards from the authoritative fill ledger; explicitly enrolled buys repair missing same-entry cohorts from their persisted quote and plan. Conditions for a manager CLOSE are persisted before the sell. Partial sells do not count as completed exits.

Live deployment found and fixed a stale assessment-clock bug: each owned assessment now snapshots its own current clock/quote. Two-symbol slow-assessment regression passes.
Live input inspection found 243,671-byte manager packets, mostly redundant foundation paths. Preserve full receipts on disk; send bounded closing-quantile summaries and candidate fields. Current packet: 86,816 bytes. Constrain decision prose; no relaxed execution, report or risk validation.

Deployment: 2026-10-22T19:56:30Z, hidden launcher PID 16020, same run/accounts/positions. Both foundation caches reused their original availability timestamps. Public Wallex live probe succeeded. Runtime reports manager_only, fresh quotes and scan, no worker errors. Final manager-cycle confirmation pending.

Final follow-up: genuine Codex structured decoding failed even with an entirely artificial tiny input when the new schema used minimum=1e-300. Tiny boolean and old decision schemas passed; the new condition bounded at 1e-12 passed, and the old extreme-bound schema still failed even for an empty decisions array. The four supported spot assets exceed this representable bound. Preserve numeric positivity checks; do not loosen them to bypass validation.

The real packet probe was initially blocked by automatic approval review. User explicitly approved transmitting the virtual paper account/history/charts to Codex. Subsequent actual 86-KiB packet plus charts passed in 30.031 seconds, with no execution enabled. Evidence: real-llm-schema-probe.json and immutable inference subdirectory. No reasoning downgrade or model switch was needed.

Final deployment: 2026-10-22T20:06:41Z, hidden launcher PID 7712. Runtime health now exposes unsuccessful manager responses, and dashboard no longer presents an old vote as a fresh one. Eight final checks passed; see final-verification.json.

Remaining model limitation: older owned barriers can move outside the fitted head's 0.5..3% stop / 1..4 ratio domain. Their hold-value score is explicitly unavailable; protection, HOLD and discretionary CLOSE still work. No fabricated extrapolated probabilities, model promotion or profit claim.

Verified real manager cycle at 20:10:47Z: ETH close +0.01428627596512122 USDT, SOL close -0.028674144773944477 USDT, BTC/XRP HOLD. Fifteen fills, two remaining positions; original thirteen fills preserved exactly. Both genuine foundation forecasts present, four fresh Wallex MCP books and no runtime errors. See final-live-health.json.

Performance follow-up: per-position subprocess reloads made cold owned assessments too slow. Replaced them with one portfolio batch over one atomic BBO snapshot and freshest runtime candle references. Regression failed before and passed after batching; core, runtime and audit checks pass. Independent focused review found no Important issue in ordering, zero-fee units, per-market error isolation or protection concurrency. Subsequent deployment receipt identifies owned_assessment=single_batch.

Completion: final batch deployment PID 19092 passed a real manager cycle at 20:16:09Z with no runtime errors. Four fresh books, both foundation caches reused correctly. Sixteen retained fills and one remaining XRP position at that cutoff; prior fills preserved. Batch-deployment-health.json and post-fix-trades.json contain the actual execution results. Existing dense survival training process was preserved. Profitability remains unproven.
