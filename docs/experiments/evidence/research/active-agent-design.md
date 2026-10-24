# Opportunity-driven agent repair

User requests an agent that actively evaluates and trades opportunities rather than indefinitely choosing cash. Immediate implementation is authorized; retain risk/cost gates and do not manufacture profitable outcomes or require a trade quota.

Observed reproduction: the original ten-call budget spread 230-bar decisions over the whole dataset. At M15 this commits for 57.5h while the packet covers only12h. Actual repeated model reasons cite this mismatch. A bounded call budget must cap new evaluations, not stretch trade holding or disappear exit management.

## Interface and ownership

- Add `wallex_lab/opportunities.py`: `scan(frame, config) -> list[dict]`, candidates ordered causally by candle index. Calls use only completed history; reset warmup after gaps. Each candidate contains index, unique id, template (trend_breakout/trend_pullback/reversion), feature evidence, stop_fraction, target_fraction, max_hold_bars<=24 default, roundtrip_cost_fraction, potential_net_reward_fraction and reward_risk_after_cost. This describes payoff geometry, not expected profit/probability. A per-side cost is charged only once per traded side; hard filter inadequate reward/cost geometry, not unseen returns. Hourly scan (4 M15 bars), configurable fixed clock independent of max_calls. Generate causal evidence + candidate rejection diagnostics. No future labels. At most one candidate per index; config default max allocation=.15; no minimum trade quota.
- Add `wallex_lab/active_agent.py`: `generate(frame, train_end, val_end, config, run_dir)` same target[N]/metadata contract. Use opportunities.scan and existing llm.run_one unchanged. Sequential chronological flat/open/cooldown state. AI chooses enter/watch/reject plus allowed holding and allocation from finite options capped by risk; stop/target geometry fixed by candidate. Enter at next OPEN through shared engine. Stop/TP observed at completed CLOSE => target0 and execute nextOPEN, no intrabar fabricated fill. Hard time expiry<=24 bars independent of call budget and later decisions. Max calls stops new entries only; never exits. Reject all if no opportunity passes gates, preserve reasons and counts. No forced trade. Store every actual response/attempt receipt, rule diagnostics, closed position plan states and targets. Historical knowledge contamination remains disclosed. Transient retry once under separate namespace. Synthetic tests mock inference only; production requires real outputs.
- Parent owns new CLI or runner integration and real test/reports. Do not change old agent.py or runner.py: preserve previously reviewed code/results, give the new module its own runnable CLI for use now. Existing low-level llm.py, engine.py and features.py are reused unchanged.
- New experiment BTC main historical TEST only starts at old test_start=80% so no new selection uses that period. Fixed new settings are a post-diagnostic exploratory comparison; not independent validation. First compare on a fixed contiguous three-day slice of TEST, scanning hourly, max12-24 actual evaluations. Horizon and call budget are separate; gaps/overlap/cooldown enforced. Start cash.
- Tests: hourly cadence invariant to call budget; future perturbation/prefix invariance; reward/cost rejects bad candidates; maxhold expiry even after last call; no overlap; stop at close fills nextOPEN; watch never enters; low-budget still exits; real-data activity diagnostic, not a forced positivity assertion.

## Plan

1. Scout worker implements opportunities.py + check_opportunities.py test-first.
2. Active-agent worker implements active_agent.py + check_active_agent.py test-first against the contract; coordinate dependency with scout. No real calls until parent schedules.
3. Parent adds experiment CLI/test; runs real LLM decisions and baseline comparison; validates existing checks, new tests, exact ledgers and raw receipts.
4. Independent scoped review, fixes, actual results report. No exchange orders.

## Actual feedback and conditional-watch correction

The first real short-horizon run evaluated4BTC and12SOL candidates. BTC returned4watch; SOL returned11watch and1reject. The stop/expiry clock worked, but watch still meant zero allocation with no executable confirmation condition. This reproduced a second inactivity mechanism.

In the corrected schema, enter is immediate intent; watch arms a conditional plan with a positive contingent allocation, fixed confirmation rule and4/8/12bar expiry; reject reserves no allocation. Watch itself never creates a fill. Breakout confirmation needs a later completed close above the frozen four-bar high plus5bp and volume at least the frozen positive-volume median. Reclaim confirmation needs a later close above the frozen12bar mean plus5bp and above the arming close plus10bp, with the same volume check. On confirmation, enter at the following open; otherwise expire. Monitoring and exits continue after the inference budget is exhausted.

M15 input is enforced so24bars really means6hours. Required plan exits become explicit forced paper-exit requests rather than ordinary minimum-notional rebalances. Subminimum paper exits remain counted and are not exchange-executable promises. The previous engine source is preserved under docs/source-archive with its SHA256; without the new optional exit requests, all30oldresearch ledgers reproduce unchanged.

Fresh output directories preserve the zero-trade first attempt. Existing identities must match data/config/source/CLI/loader hashes before a matching cached rerun. Metadata separates proposed plan lifecycle from actual shared-engine fills.

Completed conditional runs: BTC evaluated3, armed3, expired3, filled0. SOL evaluated9, armed8, rejected1, activated3, expired5 and completed3roundtrips/6fills. SOL account return after assumed costs was -0.2587998293% (-12,939.99toman on5million); stress return -0.3183908897%. All three trades lost. The inactivity mechanism is repaired; profitability is not established. This is a post-diagnostic historical experiment, not fresh independent validation. Full receipts, timings and replays are in reports/active-agent-results.md/json.
