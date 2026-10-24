# Adaptive paper council implementation plan

> For agentic workers: use executing-plans to implement and verify this plan.

**Goal:** Run an account-sized adaptive paper council against a fresh fixed control.

**Architecture:** Reuse MarketCouncil, FirstTouchBook, shared feed, PaperBroker and
existing dashboard infrastructure. Add one optional residual-calibration module
and extend the existing paired runner; default fixed policy remains unchanged.

**Tech stack:** Python standard library, SQLite, installed NumPy/Pandas; no new dependency.

**Spec:** `docs/superpowers/specs/2026-10-23-adaptive-paper-council.md`.

## Global constraints

- Paper only; Wallex-only observed source prices; no credential in model packets.
- Do not change active baseline accounts, risk caps or model weights.
- Learn only matured outcomes; no synthetic winners or timed forced exits.
- Native execution is authorized by the user's explicit implementation request.

## Review focus

- Prospective timestamps and immutable feature/prediction snapshots prevent future leakage.
- Origin/time-slot assignment prevents geometry-count inflation and train/evaluation overlap.
- Crash/restart cannot train a sample twice or lose a promotion record.
- Stale or missing candles censor results without starving fresh protective quotes.
- Learned scores remain coherent between entry/holding and do not bypass execution guards.

## Tasks

- [x] Add `checks/check_adaptive_council.py`; observe failure before implementation.
- [x] Add `wallex_lab/adaptive_council.py`: persistent examples, chronological
  maturation, grouped residual updates, prequential evaluation and version gates.
- [x] Add opt-in config and shared `_predict` integration, expose bounded learning
  status and actual prior manager weights; prove disabled behavior is unchanged.
- [x] Extend the paired fixed/adaptive runner and concise dashboard, reusing shared
  input snapshots and account-sized seed; keep existing runs untouched.
- [x] Run focused regression checks and deterministic learning/promotion checks.
- [x] Run prospective paper collection, inspect actual model calls, saved examples,
  quote provenance and ledger; record warmup limitations and review changes.

## Execution ledger

- User approved the concrete in-chat feature design and explicitly requested
  implementation. Proceed natively under that authorization.
- Workspace has no Git repository; preserve files in place, no artificial commits.
- Seven focused checks passed, including shared entry/holding calibration and costs.
  Independent review's checkpoint, cost, provenance, paired-context, barrier-bound
  and delayed-ingestion findings were corrected and rechecked; no remaining
  Critical/Important findings in the reviewed scope.
- Live paired run started at 2026-10-23T12:04:27Z. Comparison dashboard: port 8832;
  fixed/adaptive detail: ports 8830/8831. Each account starts at 18.246562963625525
  virtual USDT; USDT fees 13 bps, TMN fees 30 bps. Two-hour new-entry window;
  collection and protection continue afterward.
- At 12:06:16Z, both arms completed actual exported-model inference: 80 geometries
  each for BTC/ETH/XRP. Adaptive stored 240 prospective pending examples. Both
  accounts had zero fills and zero realized P&L; no mature outcomes or learned
  promotion yet. This is startup verification, not profitability evidence.
- Native Wallex SOL context still has excessive missingness; Kronos/Chronos native
  contexts remain insufficient. No foreign bars or fabricated fills were used.
  Recorded evidence: `runs/adaptive-wallex-20261023/live-verification.json`.
- Price display ambiguity corrected in the shared dashboard: separately label the
  executable Bid (user sells) and Ask (user buys), rather than show an unnamed Bid.
