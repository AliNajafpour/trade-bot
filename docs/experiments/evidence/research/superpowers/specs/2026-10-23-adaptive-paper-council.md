# Adaptive paper council

The user approved the in-chat design and explicitly requested implementation.
Implement inside this session, keeping the existing fixed paper runs intact.

## Outcome

Add a persistent, lightweight online residual calibrator over the existing
candidate-conditioned technical distributions. Its inputs include the technical
forecast/geometry, observed trend/volatility, and contemporaneously valid market
and macro assessments. The learned coefficients must affect numerical TP/SL
probabilities and candidate expected value, not just narrative weights.

The same calibration must serve entry and holding scores before entry fees are
applied. Existing risk, positive-edge, data-freshness, reentry and stop rules remain
independent. No exchange order or private credential enters the learner.

## Data and causality

Record candidate forecasts before outcomes, including rejected candidates and
owned-position alternatives. Sample once per symbol, role and 15-minute slot.
Each record includes exact barriers, horizon, base/champion/challenger predictions,
bounded numeric features, report versions, forecast and observation timestamps,
model version and origin quote. Source prices must be Wallex-only.

Label from subsequently completed, continuous native M1 candles, reusing
FirstTouchBook. Never treat an open position as a win or force a timed exit.
An untouched barrier at a completed horizon is a third forecast class, not a
closed trade. Wait for observed tail coverage to reach the horizon before finalizing
any geometry in its group; an ingestion delay stays pending. Missing interior
price coverage censors a label; ambiguous same-minute
touches conservatively count as SL first. Counterfactual price-path returns are
explicit proxies, separate from actual paper ledger P&L.

## Learning and evaluation

Use NumPy softmax residual updates over frozen base log probabilities, with
regularization, bounded features and conservative learning rate. A separate
bounded linear residual may adjust unresolved-horizon returns only from matured
unresolved examples. Persist weights as JSON and examples in SQLite.

Whole origin time slots are assigned prospectively to training or evaluation,
never both. Evaluate the exact predictions logged before outcomes. Aggregate
correlated geometry examples by symbol/slot/role/horizon instead of counting
80 geometries as 80 independent observations. Default gates are 32 matured
training groups, 16 evaluation groups, lower Brier loss by 0.005 and higher
cost-adjusted counterfactual selection return. These are configurable engineering
gates, not statistical significance or profitability guarantees.

Evaluate promotion hourly. Keep an immutable version and evidence for each
promotion/rollback. Freeze a separately identified candidate checkpoint for each
prospective evaluation cohort; never promote later, unevaluated training weights.
A challenger first trains in shadow; until promotion,
execution retains baseline probabilities. Roll back a promoted version after
16 new evaluation groups if both forecast loss and selection return deteriorate
against baseline. Coefficient influence shares are descriptive sensitivity,
not causal attribution and not the LLM's own declared weights.

## Operation

Keep state and queries bounded: process at most 2,000 pending examples per tick;
cap outstanding examples at 50,000 and expose saturation without dropping or
fabricating outcomes. No new dependency, GPU training or foundation fine-tune.
Show mode, pending/mature/censored counts, actual paper P&L, calibration metrics,
input influence and promotion/rollback history on a separate local dashboard.

Run adaptive and fresh fixed controls on the same frozen candle/report input per round and
the same account-sized paper capital, fees, time window and risk settings.
Preserve live execution quotes and audit their per-arm consumption separately;
independent manager calls and execution timing remain comparison limitations.
Also preserve current older runs. Offline deterministic checks must demonstrate
real coefficient learning, temporal exclusion, persistence/idempotency,
censoring, promotion/rollback and entry/holding cost coherence. Live results
must distinguish collecting data from having established an improvement.
