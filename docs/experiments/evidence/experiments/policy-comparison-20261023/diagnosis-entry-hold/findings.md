# Frozen ETH entry/holding diagnosis

No production file changed. Replayed the exact entry rescore request, candles, barriers, horizon, and quote from the two rapid ETH entries. All three models use the same neural weights and report the same completed-context dates; the calibrated classifier/regression heads differ.

| UTC entry fill | Primary entry net, bps | Management hold at bid, bps | Management net entry at ask, bps |
|---|---:|---:|---:|
| 04:18:59.628938 | +4.2521 | -1.4698 | -44.3383 |
| 04:23:02.456131 | +2.2723 | -4.1905 | -48.0746 |

Management entry economics here reuse its bid-based prediction and then convert expected future liquidation to the actual ask plus both 13-bps fees. No slippage is included in this diagnostic table. The saved live config's slippage, if positive, only lowers entry value further.

The disagreement is present with a common ask reference too: first primary gross +30.2970 bps versus management -8.0781 bps; second +28.3121 versus +2.0749. Even the primary head's bid-based prediction converts to negative entry value (-6.2281/-7.2770 bps), showing that reference price affects learned covariates and geometry, not just an algebraic rebase.

## Actual close/rebuy flow

04:20:09 owned value -2.0301 bps: close blocked waiting for a second completed M1 assessment. 04:21:14 owned value -3.7155 bps: close filled. Fresh entry accepted 04:23:02. 04:24:13 owned value -2.9122 bps: close blocked. 04:25:13 owned value -2.6641 bps: close filled. These rapid examples retain their initial stops (2358.6934 and 2359.6538); these closes are not stop-ratchet exits.

## Semantics and minimal coherent correction

The sunk-fee distinction is already intentional: entry uses ask and round-trip fees; owned scoring uses bid and zero entry fees. A future constant sale fee cancels when comparing hold versus immediate sale. Both branches ask the same TP-first/SL-first/unresolved-at-horizon question; neither treats horizon as a forced exit, and owned scoring keeps the original horizon length rather than subtracting elapsed minutes.

Use one predictive head and one executable bid reference for a given context and exact future bid barriers. Convert the shared expected final bid F to economic decisions: hold-versus-close = F / current_bid - 1; entry-versus-cash = F * (1-exit_fee) / (ask * (1+entry_fee)) - 1. Include the same declared slippage assumptions consistently. Since ask >= bid and fees >= 0, positive entry value then implies positive same-context holding value. An immediate buy/close contradiction becomes impossible without a changed quote, context, barrier, or horizon.

The management head already supports independently varying remaining distances; keep explicit entry geometry validation against the real ask and the existing entry grid before scoring with that head. Do not apply entry ratio bounds to owned positions. Using the management head for entries expands its recorded protocol ('no entry-policy changes'), so frozen portfolio replay must establish its practical behavior; this diagnostic establishes consistency, not profitability.

Current midpoint stop proposals have a separate structural defect: for stationary bid B, repeated acceptance maps stop distance d to d/2 each review, hence d_n=d_0/2^n regardless of new market information. Candidate validation permits down to 1e-6 relative distance, below the resolving power of M1 path labels. Remove time-driven recursive tightening proposals; derive any alternatives from new observed support/price evidence and score their exact geometry with the same head. Statistical validity of very close stops remains limited by M1 data.

## Regression scenario and commands

Run `.venv-portable/Scripts/python.exe runs/policy-comparison-20261023/diagnosis-entry-hold/replay.py` to rebuild full evidence. Run with `--check-saved` for a fast frozen contradiction check: it currently fails on the first +4.2521 bps entry versus -1.4698 bps owned signal. The production regression should drive both actual entry and owned paths at one frozen context and assert positive entry implies nonnegative owned value, then confirm both saved ETH buys are rejected under the unified prediction.

Also assert stop candidate proposals do not converge toward a stationary bid under repeated identical input, and that a genuinely changed context can still alter assessed value. No cooldown or invented minimum-edge threshold is required for the observed semantic contradiction.

Skipped training, external APIs, orders, service restarts, and production edits. Cross-venue transfer, profitability, parameter uncertainty, and unresolved mark bias remain unvalidated.

## Revised production-path verification

`revised_predict.py --all` called the actual revised `MarketCouncil._predict` for entry and holding on all 18 saved matched closed-entry contexts. All 18 were available and passed: same management model ID, exact same bid reference, identical three probabilities and gross EV, correct `entry_fee_signal` conversion, contributions summing to net EV, and positive entry implying nonnegative holding. Both rapid ETH buys become negative executable entries (-44.3383/-48.0746 bps). Of all 18 frozen candidates, 17 become negative; only the XRP entry at 2026-10-22T21:19:28 UTC remains positive (+7.3920 bps entry / +61.8291 bps holding). This is a frozen candidate-admission comparison, not a portfolio return estimate.

`policy_checks.py` passed the actual production candidate generator for 20 identical-input iterations, adversarially selecting the maximum stop each time: no recursive tightening after the first proposal. The allowance in the first ETH fixture was 42.8748 bps and the stable maximum stop was 2392.328765. The actual `_check_reentry` rejected a new scan within the same completed 15m frame and accepted the next completed frame. It does not establish whether that policy is optimal.

Read-only review found the comparison account's old ask/reference drift check treated coherent bid reference as ask, and ask-only final drift missed changed model-conditioning bid. Both findings were sent to the parent; subsequent source inspection shows `_entry_quote_drift` now checks stored entry ask and model bid and is used at both entry sites.

Evidence: `revised-evidence.json`, `revised-summary.json`, and `policy-checks.json`. No additional material defect found in the reviewed conversions, stop floor, or completed-frame guard. The 90th-percentile adverse-excursion allowance remains an explicit chosen empirical risk allowance, not a calibrated future stop survival probability.
