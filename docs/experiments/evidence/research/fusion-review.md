# Final specialist fusion review

What this change does: Technical forecasts now feed separate regime, news and fundamental assessments, followed by a required final model decision. The orchestrator can create an entry plan only from that final decision and still applies deterministic cash, risk and fresh-quote checks. An optional explicit adapter reloads the earlier benchmark HGB/GRU candidate for fresh prefix inference.

## Finding repaired during review

1. **Eight older fundamental facts can hide a newly received material fact.** P2, `wallex_lab/specialists.py:125-163`.

   Fundamental selection iterates the supplied list and stops after eight valid facts. It does not order eligible facts by availability time before applying that cap. With eight valid BTC facts observed at 13:00, appending a ninth distinct valid BTC fact observed at 13:59 leaves both the selected facts and the context fingerprint unchanged at 14:00. A pending entry therefore does not require a new final assessment even if that new fact reports a chain halt or withdrawal suspension.

   Reproduced directly against `_fundamentals` and `context_fingerprint`, without inference. Validate candidates, order them newest-first, then deduplicate and cap. Add the ninth-new-fact case to the existing context check. News already orders by availability before its cap.

   Closed after repair. Fundamental candidates now pass clock/content validation before sorting by availability and observation time, then deduplicate and apply the eight-fact cap. The updated specialist check confirms that an appended newest ninth fact is selected first and changes the context fingerprint.

## Checked boundaries

The real `specialists.analyze` runs regime, eligible news and eligible source-bound fundamentals before the final merger. Missing roles remain unknown. The final schema limits risk to the technical allowance and at most half risk when context is missing. High observed event risk restricts the final stance to reject; missing context restricts it to watch or reject. Final inference or schema failure returns a blocked analysis rather than substituting a technical decision.

`orchestrator.entry_decision` independently requires an observed final decision, preserves technical rejection and both event-risk vetoes, checks stance/trigger consistency, bounds durations and rejects a larger risk multiplier. Missing context permits only conditional WATCH at half risk or less. The production path has no legacy technical-entry fallback. `size_order` then takes the lower technical/final allowance, keeps event-risk vetoes and applies cash, fee, depth and stop-budget admission.

Before entering from a pending plan, the orchestrator recomputes the selected context fingerprint and checks analysis age. A context change or stale analysis removes the plan before any broker submit. Relevant new evidence also participates in setup identity, permitting a fresh assessment of the same technical candle. Facts require observed/available clocks no later than the decision clock, reject expired validity and have a hard 24-hour observation age. Future or unrelated observations do not invalidate an otherwise unchanged plan. Finding 1 originally limited this protection when more than eight fundamental facts were supplied and is now repaired.

The benchmark adapter requires explicit case and combination configuration. It verifies fitted source hashes, original candle evidence, checkpoint/output hashes, the native timeframe, frozen validation boundary and a holding horizon no greater than six hours. It recomputes the latest HGB/GRU forecast from the supplied completed prefix, preserves gross fractional units, checks symbol/cadence/gap history and records that its candidate was chosen after exposed historical tests. It does not automatically select a historical test winner.

No current news or fundamental assessment is inserted into the saved historical benchmark profit ledgers by these modules. The fusion gate changes current/replay decisions; the benchmark return remains evidence for its original technical policy. Replay still needs its explicit historical timing and knowledge-contamination caveats.

## Checks

Five focused checks passed independently: `checks.check_specialists`, `checks.check_specialist_decision`, `checks.check_entry_context`, `checks.check_orchestrator` and `checks.check_spot`. `checks.check_ai_market` also passed. These cover the required merger, exact evidence fields, missing-context semantics, hard vetoes, risk reductions, changed-context invalidation, stale plans and actual paper entry.

`checks.check_benchmark_signals` initially ran while the adapter integration was still being edited. Its saved-model parity and admission checks reached the current-cost assertion, which failed because `aggregate_net_return` remained null. After the adapter repair, the reviewer reran the check and it passed. This verifies actual saved HGB/GRU latest-prefix parity, explicit candidate configuration, current-cost fields, artifact provenance, freeze/cadence/symbol/gap admission and six-hour scanner plans. No fit, GPU inference, network request or model CLI call was made by this review. Only this document was written.

## Final repair verification

The reviewer reread the completed specialist and orchestrator repairs, then reran `checks.check_specialists`, `checks.check_specialist_decision`, `checks.check_entry_context`, `checks.check_orchestrator` and `checks.check_spot`. All five passed independently. Together with the already repaired and passing benchmark adapter check above, no unresolved P1 or P2 finding remains in this scoped review.

Context identity now compares the semantic contents of selected evidence. Refreshing an unchanged fact's snapshot ID or receipt/observation time does not cancel a pending WATCH. A changed value, source-bound content, quality flag, publication clock or measurement clock does change the identity. Selection still enforces availability, observation age and validity on every tick, so evidence removal on expiry invalidates a pending plan. Fundamentals pass a bounded measurement clock and quality flags to the specialist; future measurement times are rejected. The tests cover unchanged observation refresh, new material ninth facts, changed measurements, future measurements and old observations.

The orchestrator also fixes forecast expiry to the original completed candle plus the actual model horizon. A pending plan cannot outlive that deadline. Once entered, its position expiry is the earlier of entry plus the allowed hold and that same original deadline. Analysis finishing after the deadline creates no plan. The entry-context check verifies a 15-minute model forecast expiring at 10:15 despite entry at 10:00:01, and rejects a still-pending plan at 10:15. Waiting or inference latency therefore cannot move a forecast's validity into the future.

The four actual final-gate model calls reported by the parent are diagnostic current-context evidence. This review did not relabel them as historical profit evidence, and it did not rerun those calls. A current specialist decision does not revise the old technical benchmark policy or its P&L.

Verdict: the repaired fusion boundaries and explicit benchmark adapter pass the scoped checks. Continue to disclose paper execution limits, unknown future performance and the separation between current-context diagnostics and exposed historical results.
