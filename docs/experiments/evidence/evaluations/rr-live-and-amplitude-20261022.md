# Current RR experiments and amplitude audit

The frozen 24h Chronos-2 close-calendar/g0 neural policy was tested at RR floors 0, 0.25, 0.5, 0.75, 1, 1.5, 2 with unchanged forecasts, costs and shared-cash accounting. Floors0.5/0.75/1 tied on baseline validation performance; stable predeclared ordering selected0.5. Historical TEST was already exposed and is a retrospective diagnostic.

See `runs/rr-filter-comparison-20261022/report-readable.md` for counts. Floors below1 each admit one unresolved native-M1-gap outcome. Their closed realized PnL is partial; no complete total return is asserted. Fresh public re-queries still omit all12 missing minutes, documented in `data/rr-gap-check-20261022/report.json`.

Live independent paper runs: selected RR0.5 at http://127.0.0.1:8783/ and all-positive specialist review at http://127.0.0.1:8782/. Both have1000 virtual USDT, original entry-window deadline02:31Tehran22Oct2026. Existing positions/plans continue monitoring after that window. All HTTP receipts, model forecasts, actual specialist outputs and fills are retained. Startup compatibility and ledger-contention fixes required documented restarts; this is not an uninterrupted-session claim.

The all-positive experiment passes valid positive learned expected-net-payoff candidates with positive after-fee TP payoff to actual specialist analysis. The ratio floor and extra pTP floor are zero. Spread and bid-breached-stop checks are deferred to fresh post-analysis execution risk; no missing/invalid quotes or forecast geometry is fabricated. Fresh entry EV is checked again at the new ask.

Amplitude quick audit: `runs/selected-24h-amplitude-quick-audit-20261022.json` uses271 complete24h asset-origin windows from the shared292-origin cohort, pooling previously exposed TRAIN/VAL/TEST only for diagnosis. Median absolute forecast terminal move0.3933%; realized absolute terminal move1.3416%; actual favorable excursion1.4700%; model max marginalq75 target1.4481%; model terminalq05-q95 width7.0927%. Actual favorable excursion>=3% in26.94% of these windows; its90th percentile6.9498%. Missing windows are excluded from this descriptive amplitude audit, not imputed.

A central median close forecast and the largest intrawindow move are different quantities. g0 uses max marginalq75 CLOSE as TP and min marginalq10 CLOSE as SL; this asymmetric geometry can produce low payoff ratios. Marginal price quantiles do not supply a joint path-extreme quantile, and their nominal coverage is not verified. There is no artificial percentage cap in this model_geometry path.
