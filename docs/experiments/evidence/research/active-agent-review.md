# Active agent review

What this change does: The new scanner finds causal long-only setups on a fixed clock. The agent asks the model to enter, watch or reject each eligible setup, then manages a bounded plan without requiring more model calls. This review covers the new scanner, agent, CLI and their three runnable checks, plus the existing packet, receipt and execution boundaries they call.

## Must fix

1. **A required exit can fail while the plan says it closed.** P1, `wallex_lab/active_agent.py:90-106`.

   The agent sends zero allocation for a stop or holding expiry, then closes its plan at the next index without checking the ledger. The shared engine treats this as an ordinary rebalance, subject to `min_notional`, at `wallex_lab/engine.py:69-74` and `129-140`.

   Reproduced with real engine execution and mocked inference only: initial cash 1,100,000, minimum notional 50,000, allocation .05, entry at index 65 with price 100, and prices falling to 89 at index 66. The default 3% daily and 8% account risk stops do not trigger on this small allocation. The plan records `stop_close`, exit index 67 and state `CLOSED`. The ledger buys at 65 and sells only at final index 119, rejecting 54 subminimum sell requests. Inventory therefore remains open for 13.5 hours on M15 despite the four-bar plan. A subsequent accepted plan can overlap that inventory.

   Fix the execution contract for mandatory plan exits so they bypass the rebalance minimum, and derive filled state from actual execution or explicitly distinguish planned state from filled state. Add this small failing case to the existing executor check. Do not change previously reviewed artifacts merely to repair their historical results.

## Should fix

2. **The six-hour limit depends on an unchecked M15 assumption.** P2, `wallex_lab/opportunities.py:10-14`, `wallex_lab/active_agent.py:43` and `scripts/run_active_agent.py:20-28`.

   Both scanner and agent accept any positive native candle interval but bound holding only by 24 bars. A flat H1 synthetic frame with a 24-bar accepted response fills at index 65 and exits at index 89, holding for 24 hours. The CLI also subtracts a hard-coded 15 minutes when choosing its slice while accepting arbitrary `--data` filenames.

   Either reject non-M15 input in this M15 experiment or derive the holding cap and slice end from the validated candle interval. The current production M15 data is unaffected.

3. **Reusing an output directory replaces its run identity before validation.** P2, `scripts/run_active_agent.py:30-34` and `49-56`.

   `identity.json` is written unconditionally, and successful runs replace the saved targets and result. Running the same output with another data file, cost setting or horizon silently replaces the prior identity while leaving earlier inference receipts in the directory. The source identity also omits the experiment script and data loader, even though both affect the experiment.

   Compare an existing identity with the requested identity before writing any run files, rejecting a mismatch. Include the CLI and loader in the recorded source hashes. A matching rerun can still use the existing receipt cache.

## Checks and limits

All three new runnable checks passed with the existing Trade-Bot Python environment: `checks.check_opportunities`, `checks.check_active_agent` and `checks.check_active_run`. Their model calls are explicitly mocked. The additional short reproductions above also mocked inference and used the real shared engine. No network, model calls, training, old result edits or source-code edits were performed for this review.

The scanner uses completed prefixes and resets its 48-bar warmup after a gap. Future perturbation and prefix checks pass. The fixed scan cadence does not depend on the inference budget. The ordinary M15 executor path exits at the next open after the last permitted hold close even when its call budget is exhausted. Watch never enters, stop tests use close observation followed by next-open execution, and schema validation rejects unknown evidence fields. Cost filtering describes possible payoff geometry and clearly disclaims expected profit. No trade quota is required, and rejection or all-cash results remain valid when the gates fail.

Verdict: fix 1 before claiming that actual positions always obey the holding limit. Fix 2 and 3 before supporting arbitrary CLI inputs or resuming into existing output directories.

Not checked: production model responses, BTC/SOL live experiment artifacts, remote model identity, unseen future performance or exchange execution. The parent owns those checks. Findings describe the source snapshot reviewed and should be updated after repairs.

## Repair re-review

The parent supplied a stable repaired source snapshot for a second scoped review. All four small checks passed: `checks.check_opportunities`, `checks.check_active_agent`, `checks.check_active_run` and `checks.check_active_execution`. This review again used no network or model calls and changed only this document.

- Finding 1 is closed for the production experiment CLI. `scripts/run_active_agent.py:49-59` derives mandatory exit indices from completed proposed plans, passes them to every primary, stress and replay backtest, and saves the full execution configuration. `wallex_lab/engine.py:134-137` performs a full `plan_exit` before ordinary rebalancing, bypassing the minimum notional. The regression check reproduces the original small-account case and verifies the exit at index 67 with an explicit forced-subminimum count. `active_agent.py:75` labels plan states as proposals, and CLI lines 60-65 reconcile their entries and full exits with the paper ledger. This remains a paper-execution assumption; venue minima still apply outside this simulator.
- Finding 2 is closed. `wallex_lab/active_agent.py:53-55` rejects non-M15 cadence. The agent check verifies H1 rejection. The CLI's 15-minute slice end now agrees with its supported cadence.
- Finding 3 is closed. `scripts/run_active_agent.py:31-37` includes the experiment script and data loader hashes and rejects a mismatching existing identity before replacing it or invoking inference. The integration check verifies that changed horizon settings fail before inference.

WATCH now means a conditional order proposal. This supersedes the earlier zero-allocation WATCH behavior. The schema requires a positive contingent allocation for WATCH and a typed future confirmation trigger. Arming freezes the four-bar high, twelve-bar mean, current close and positive-volume median. Only a later completed close can activate the plan, and execution follows at the next open. Confirmation requires positive volume at least the frozen median. The plan expires after its chosen 4, 8 or 12 bars or cancels at a gap. Its confirmation, expiry and subsequent holding timer run outside the model-call budget. Checks cover confirmed entry, no-volume expiry, next-open execution and budget exhaustion.

No unresolved P0 or P1 finding blocks this repaired CLI mechanism. A caller using `active_agent.generate` directly must also derive and pass its mandatory plan exits to the engine, as the CLI does. The returned target vector alone does not carry those exits. The saved `execution_config` is part of a complete replay, alongside the targets.

The review did not independently replay the earlier 30 artifacts or inspect the new production BTC/SOL responses. Those remain parent-owned evidence. No profitability conclusion follows from these interface and mechanism checks.

Verdict: ship the repaired paper-research CLI, with the recorded simulation limits and production-evidence checks above.
