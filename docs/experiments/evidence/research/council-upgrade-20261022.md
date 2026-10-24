# Live council: usable dashboard and measured model adaptation

The user wants a Persian live dashboard, repaired market data, decisions based on the available evidence, a precise account of what was trained, and genuine Kronos/Chronos-2 adaptation with comparisons. Preserve the current $100 paper ledger and every source/decision receipt. No real exchange order or leveraged/short position is authorized.

## Changes

- Replace the raw JSON landing page with an RTL trading dashboard. Show account results, model/stream freshness, all-candidate opportunity counts, exact TP/SL probabilities, the manager's rationale, pending reviews, source quality and training progress. Missing values remain unknown.
- Requery missing native-minute windows on their original venue. Preserve successful-empty responses separately from failed requests. Remaining small gaps may enter a bounded, explicitly masked causal reconstruction inside a model trained/evaluated with matching missingness; raw candles are never rewritten as observed prices.
- Let the manager review prices, charts and reports even when one technical stream is unavailable. It must give a grounded action and review conditions, not a universal missing-data veto. Numerical probabilities and entry admission still require a valid exact-geometry forecast.
- Add a separate $100 paper comparison account for positive modeled net expectation without the manager's discretionary veto. Keep execution costs, source freshness, cash/inventory, stop risk and report risk caps. This isolates the effect of the manager's filter; it is an independent paper scenario, not additional real capital.
- Reuse actual pretrained Kronos and Chronos-2 checkpoints and existing fine-tuning code. Compare frozen pretrained and adapted forecasts on the same chronological windows. Choose checkpoints using validation; retain held-out scores and losses even when adaptation does not help. Serialize GPU work on the 6GB device. More epochs or more positive forecasts are not selection criteria.
- Feed completed foundation forecasts to the manager as labeled advisory evidence. Sampled paths and marginal quantiles do not become calibrated first-touch probabilities without a fitted outcome head. Minute technical updates and protective exits remain independent of these slower forecasts.

## Verification

Regression checks cover old-hole recovery, honest remaining-gap masks, no future interpolation, clean/corrupt holdout scoring, incomplete-stream manager decisions, positive-EV paper comparison admission and separate ledgers, stale/cost/depth/risk failures, provider/chart provenance, safe model-status reads and dashboard interaction. Run the actual service, inspect it in a browser at desktop/mobile widths, and retain screenshots plus live inference/training receipts.
