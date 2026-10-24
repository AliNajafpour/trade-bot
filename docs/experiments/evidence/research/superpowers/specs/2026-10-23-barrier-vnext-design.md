# Barrier vNext design

The user approved the improvement review and requested implementation, better history, pretraining and isolated architecture experiments while the existing comparison continues. This document makes that approved scope executable without another approval round.

## Scope and contract

Keep the existing experiment's sources, labels, weights, protocols and GPU workers unchanged. New work lives in new modules and directories. Existing `risk_nll`, `incidence`, five deadlines, 128 requests, five-minute origins, relative price normalization and 24h split purges remain the reference semantics. No trading executor or profitability claim is added.

Download Binance spot M1 history from 2023-01-01 to the latest fully available day before 2026-10-23, for BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT and LINKUSDT. Freeze cutoff before acquisition. This fixed research universe is selected today; it is not a survivorship-free historical universe. Keep venue identities separate. Binance-native labels are a separate task, not replacements for Wallex labels or executable-fill labels. Use official checksum-verified monthly archives and available daily archives; record missing periods and never invent missing future prices. Capture trade count, quote volume and taker-buy volumes alongside OHLCV. Do not claim this is historical depth or bid/ask data.

Use at least 5 GiB free-disk reserve on the data drive. Download and materialize in bounded chunks with resumable receipts and atomic completion. Keep sources sufficient for reproduction. Archives unavailable on their expected dates are errors/gaps, not successful downloads.

## Data interface

New source root `data/barrier-binance-history-20261023` contains immutable `protocol.json`, per-archive receipts, `progress.json`, and final `manifest.json`. Each completed symbol supplies `market/SYMBOL-values.npy` float64 Nx9 in order open, high, low, close, volume, quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume, plus `market/SYMBOL-times.npy` int64 UTC open nanoseconds. Arrays are sorted unique M1 and contain only completed bars before cutoff. Symbol records include rows, first/last, hashes, gaps, zero/flat fractions and observed source coverage. Status ready is written only after all expected available partitions have an explicit disposition and every included array validates.

Dataset vNext indexes raw arrays rather than duplicating windows or materializing all query geometries. Each indexed origin has symbol, bar index, time, causal sigma, reference and 128 event/cause/survived labels. Distances are generated deterministically with the existing ex-ante sampler and stored only if needed for compatibility. Exact event minute and observed duration are retained. All queries are used per supervised epoch. Historical activity inputs use completed bars only. Missing past minutes are not interpolated; initial comparisons share contiguous-week eligibility and report exclusions.

## Architecture and pretraining

Add an independent `NextBarrierModel` with forward `(x, query, symbol) -> B,K,5,3`. Inputs preserve the first ten relative-v2 channels; optional causal activity channels extend them. Configuration toggles isolate activity, explicit token age/duration, barrier-relative attention and a local last-60-minute convolution residual. Baseline retains the old model as a separately constructed comparator. The direct model uses the existing coherent hazard likelihood. No named technical-pattern labels or external indicator library.

Pretraining masks contiguous physical historical intervals in the raw multiscale input before aggregation, across all channels. Use one mask for all overlapping scales so no unmasked coarse branch reveals the target. Reconstruct normalized return/range/volume patch summaries using only training-period windows. Pretraining code must actually optimize encoder parameters and produce resumable weights; representation loss alone is not a model-selection win.

## Experimental design

Separate data expansion from architecture changes: old Wallex reference remains frozen; a new Binance baseline establishes the native-venue task. On the same new data compare baseline, activity, age, level-aware attention, local residual, pretraining alone and the combined model. Fixed seeds 42/43/44, all origins and 128 queries, common roles and cohorts, identical optimizer settings, and total compute accounting. Selection NLL selects checkpoints and combinations; calibration and test remain closed until candidate weights freeze. A future untouched evaluation period is still required for prospective evidence.

Initial two expanding folds use chronological development/calibration/test blocks relative to the frozen end. Selection and calibration each span 30 calendar days where sufficient observations exist; test spans 30 days. A full 24h purge applies at each role boundary. Pretraining uses fit origins only. Aggregate scores include censored NLL, five-horizon log loss/Brier, conditional TP among known hits, reliability and time-coherence checks; paired uncertainty groups full UTC days across assets and queries.

## Resources and continuation

Acquisition and label preparation run on CPU with bounded concurrency. The existing comparison owns GPU priority. New GPU stages wait for the current comparison to complete and acquire its shared GPU lease before allocating models. The queue reports waiting distinctly from training and can resume checkpoints after interruption. CPU checks may use tiny synthetic data, but no reduced-data pilot may be presented as the full comparison. Persist source hashes, manifest hashes, RNG/optimizer state, epoch counts and progress. A failed or unavailable download must leave an actionable error and resumable state.

## Acceptance

Runnable CPU checks must establish timestamp units, checksum rejection, gaps/censoring, deterministic 128 queries, feature causality/unit invariance, pretraining mask safety and actual parameter updates, coherent probabilities, partial-label loss and checkpoint/resume identity. A real source download must validate before a large acquisition is called running. A queued training job is not reported as training started. Do not change the frozen old sources to support this version.
