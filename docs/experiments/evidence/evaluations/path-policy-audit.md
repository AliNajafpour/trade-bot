# Sampled CLOSE-path policy audit

Snapshot 2026-10-21T17:59:50.402219+00:00; official recorded fee scenario 13 bp per side.

Excluding PEPE: 103 scored origins across 11 markets.
Terminal-mean positive: 3; old terminal plus geometry eligible: 0; split-sample path-policy eligible: 2; newly eligible: 2.
Negative terminal mean but positive path payoff: 0.
Path-eligible markets: SUSDT.

These are model-derived candidate counts, not observed profitable opportunities. No trades are required or fabricated; eligibility does not establish edge.

## Limits

- Diagnostic eligibility only, no actual new orders or realized profit
- Only future model CLOSE steps after the first fresh observed ask following output completion are evaluated
- First half derives TP/SL; other half scores their ex-ante sample payoff; nine-pair selection remains sampling-sensitive
- Original unscaled samples are unchanged; PEPE excluded from comparative counts due suspect price units
- Repeated forecast origins overlap and are not independent; model sample probabilities are uncalibrated
- Candidate levels and evaluation never use future actual market outcomes; executable exit spreads/depth remain unknown

Full frozen-input hashes and per-origin diagnostics: workspace:/runs/path-policy-audit-20261021/20261021T175950402219Z/audit.json
Errors: 0.
