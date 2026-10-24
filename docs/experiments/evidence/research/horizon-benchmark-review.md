# Horizon benchmark review

What this change does: The benchmark fits fixed HGB, GRU and Kronos return heads for several candle sizes and physical holding horizons. It compares those models and their combinations using validation-selected thresholds, next-open spot execution and published Wallex fee scenarios. This review covers `horizon_benchmark.py`, `horizon_models.py`, their small checks and the engine optimization used by the running grid.

## Should fix

1. **The root run identity does not pin its market inputs.** P2, `wallex_lab/horizon_benchmark.py:110-131`.

   The identity guard records source code, fees and grid settings, but omits the native candle hashes. Editing or replacing a native CSV, then resuming the same output directory, passes that guard and overwrites the aggregated candles and per-case metadata. The model input key does detect changed candles and causes a new fit, so this is an experiment identity and preservation problem rather than a cached prediction silently using different data.

   Pin all native source hashes in the root identity before writing case outputs, then reject mismatches. For the currently frozen job, retain the saved per-case native and aggregate hashes and use fresh directories if the inputs change. The reviewer found no evidence that inputs changed during this job. No source was edited for this review.

## Causality and execution

The calendar boundaries are fixed independently of candle count. TRAIN ends at 2026-09-27 13:00 UTC, validation ends at 2026-10-09 13:00 UTC, and the exposed historical test ends at 2026-10-21 13:00 UTC. The 5, 15 and 60 minute candles come from complete native-minute buckets. Missing buckets remain gaps.

Labels are constructed only from the TRAIN prefix. The HGB training rows satisfy `origin + horizon < train_end`. GRU fitting and early stopping use separate TRAIN blocks, with the same horizon purge before their boundaries. Kronos scaler, PCA and Ridge fit only purged TRAIN hidden states. Its pretrained encoder remains frozen; unknown pretraining overlap remains disclosed. No validation or test labels reach these learned fits.

Validation selects one threshold from the fixed fee hurdle plus 0, 25 or 50 basis points. Validation execution ends within that calendar block. Test and subwindow execution reuse that chosen threshold. Means and medians propagate unsupported NaN forecasts rather than filling them from another model. Comparing or ranking the many exposed test results remains exploratory selection and does not produce independent confirmation.

Targets become known at completed closes and execute at the next open. A signal at index 64 with a four-bar holding horizon enters at 65 and exits at 69. Mandatory exits bypass the rebalance minimum through the explicit paper `plan_exit` path. A gap cancels the stale entry request, liquidates existing inventory at the first observed post-gap open and clears its later planned exit. Missing-price exposure remains unobservable and can extend the actual wall-clock holding time across the gap.

Every evaluation window starts in cash, with no preceding-window allocation. Plans near a window boundary are shortened to the last executable open inside the window. Thus a reported horizon is the intended maximum on contiguous data, not a promise that every trade has exactly that duration. Account risk stops can exit sooner. The shared engine rebalances the allocation while held, charging the configured fee on every actual traded side.

## Fees and interpretation

The fee loader verifies the saved official response body hash for the supplied verified public schedule. Its primary TMN first-tier taker scenario charges 30 basis points per side. The 25 basis point maker scenario is conditional fee sensitivity and supplies no evidence of maker fills. The 13 basis point high-volume taker scenario requires unverified account eligibility. The account's actual private fee tier is unknown.

Historical spread, slippage and impact are excluded with explicit zero additional slippage. These are results after the selected published fee and before unknown market friction. They must not be described as complete executable net profit. The no-fee diagnostic freezes the selected signal threshold and holding policy; it is a counterfactual ledger, not a newly optimized no-fee strategy.

## Independent checks

`checks.check_horizon_benchmark` passed. It verifies next-open entry, the planned holding interval, actual two-sided fee loss on flat prices, gap cancellation, unsupported combination forecasts and the official fee profiles.

`checks.check_horizon_models` passed. It uses small actual CPU HGB/GRU fits with a fixture encoder, then checks artifact reuse and corruption rejection, purged TRAIN indices, unchanged earlier predictions after TEST prices and volumes change, and NaN histories after a gap. It does not claim to test a new external or pretrained encoder.

The reviewer independently ran `scripts.verify_runs` against `runs/wallex-20261021-causal` and `runs/wallex-20261021-horizon24`, suppressing its report-file save to keep review writes limited to this document. All 30 old runs passed all three primary, stress and no-cost ledger replays exactly. Their data/source identities, archived source hashes, saved tree/neural/Kronos reloads and actual agent receipt checks passed as implemented by that verifier. The run completed in 18.34 seconds with no fitting or network access.

The engine diff against archived `engine-dff19dec80277c3723cec9f7cd4a1290edadeb83a96889c6982412bc936bb1c4.py` only precomputes price arrays, calendar days and ISO timestamps. The execution formulas and event ordering are unchanged. The exact 90-scenario replay provides direct regression evidence for the previous artifacts.

No P0 or P1 issue was confirmed. Finding 1 should be addressed before supporting same-directory resume after market input changes. The active frozen grid need not be interrupted on that basis.

Not checked: completion or ranking of the running full GPU grid, new independent forward data, private account fees, actual maker eligibility/fills, historical spread or exchange execution. No production fit, GPU inference, model CLI call or network request was launched by this review. Only this review document was created.

Verdict: the current frozen-input research grid is reviewable, subject to the input identity limitation and fee/friction disclosures above.

## Native USDT market extension

The reviewer inspected `scripts/run_usdt_horizons.py` and `scripts/report_horizon_benchmark.py`. The new market data comes directly from the public BTCUSDT, ETHUSDT and SOLUSDT history endpoints. The script does not obtain those prices by converting TMN candles. The saved acquisition manifest labels the raw quote as USDT and records no price conversion.

The initial account is 18.681113394358302 USDT, calculated as 5,000,000 toman divided by the recorded 267,650 toman per USDT close. That reference candle opens at 2026-10-09 12:59 UTC and closes at the test boundary. The reviewer recomputed the reference selection and capital exactly. It normalizes starting capital only; the traded prices, cash, P&L, fees and 1 USDT minimum notional remain in USDT. Funding conversion execution and its costs are excluded.

The saved official response hash matches the configured evidence. Its first USDT tier specifies 0.0013 taker and 0.001 maker fractions, equivalent to 13 and 10 basis points per side. The reviewer checked those raw values, the fee configuration hash, all three native input hashes and the recorded model/engine source hashes. USDT root identity pins the three native files before case output writes.

All 144 USDT first-tier-taker full-test ledgers replayed exactly from saved targets, explicit forced exits and quote-currency execution settings. Their comparison returns match the replays. The full USDT CSV contains 1,152 unique configuration/profile/window rows, including losses and abstaining configurations. No fit or GPU inference was rerun.

The observed USDT winner is ETHUSDT, M60, six-hour horizon, frozen Kronos hidden states with the local Ridge head. Its exact replay has seven closed trades, 2.3980361404% account return, and 0.4479798506 USDT P&L after the 13 basis point fee scenario and before unknown spread/impact. The saved encoder cache hash and fitted scaler/PCA/Ridge artifacts reproduce every stored sampled forecast exactly without rerunning the encoder.

Separate restarted windows give 0.1329200176% over the first six days with three trades and 2.2621093267% over the last six days with four trades. The same-fee constant-50% target baseline gives -0.1126498230%. The selected threshold's earlier validation return is -0.8536037789%; the other predeclared thresholds did worse. Thus the positive test result is an exploratory candidate selected from a large grid, with much of its gain in the final half. Positive halves do not establish stable independent validation or future profitability.

The report script includes all primary configurations in its denominator and preserves losing/zero-trade rows in the CSV. It distinguishes primary taker fees from conditional maker/VIP sensitivities, states the unknown friction and private charges, and labels the cross-market mean as separate account returns rather than one executable portfolio. Its ranking warning correctly identifies test-based TF/horizon/model selection as exploratory. Its friction headroom is a frozen-target counterfactual, not observed historical spread.

The external TMN `native-input-snapshot.json` mitigates finding 1 for the current run. All four native hashes remain unchanged, and all seven timeframe data manifests present at the time of review agree with that snapshot. The final report checks it again. The module's future same-directory resume guard still omits those input hashes, as documented above.

### Additional unit metadata finding

2. **The USDT derived-data manifests label prices as toman.** P2, `scripts/run_usdt_horizons.py:55-56`.

   The shared `quality_report` hardcodes `price_unit='toman'`, and the USDT benchmark initially saved its result unchanged. The reviewer confirmed that `runs/usdt-horizon-benchmark-20261021/ETHUSDT/60m/data.json` therefore reported toman, while the acquisition quality manifest correctly reported USDT. The ledger and result arithmetic use USDT correctly; the defect is contradictory provenance metadata.

   Closed after parent repair. The script overrides the quality field to USDT and records `quote_currency='USDT'`. All six corrected metadata files match their new hashes in `metadata-unit-correction.json`, and their numerical candle hashes still match the existing CSV files. The reviewer reconstructed every old metadata hash exactly by reversing only the unit-label correction and added quote-currency field. This verifies a label-only repair without refitting or return recalculation.

No P1 issue was confirmed in the native USDT method or report logic. This extension review changed only this document and made no network, training, GPU or model CLI calls. Final all-grid reporting remains parent-owned until the TMN run finishes.

The parent also amended the report to show validation returns beside the observed test winners. A separate per-market table chooses the highest earlier validation return across the declared grid and uses cash when that return is nonpositive. Its screened-candidate flag requires positive validation, positive full-test return, positive returns in both restarted halves and at least five full-test trades. These retrospective screens improve disclosure but remain test-dependent selection. In particular, the ETHUSDT winner above fails the positive-validation screen despite its positive test return.
