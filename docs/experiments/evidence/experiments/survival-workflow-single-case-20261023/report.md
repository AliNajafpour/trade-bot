# Single strategy: model versus news/macro/manager workflow

TP2%, SL2%, forecast24h, strict absolute P(TP first by24h)>0.55; original BTC/ETH/SOL/S four-asset cohort. No other strategy tested.

Period: 2026-09-28T09:02:00+00:00 to 2026-10-12T22:58:00+00:00. Starting capital18.246562963625525USDT; fees13bp and assumed execution friction5bp per side. TP/SL exits only, no expiry liquidation.

Reproduced baseline: +1.343137872%;17closed trades (12TP,5SL),1open. All simulator summary fields exactly match the original result.

116 qualified historical signals. The1250-snapshot local audit finds zero news/fact rows recorded as available by the end of this test. GDELT and ALFRED public archive probes both timed out without HTTP responses. External reconstruction is unproven, not ruled out.

**Workflow return: unavailable. Improvement/deterioration: unknown.** No manager was invoked on future or fabricated context. No rejected trades or zero return are attributed to missing archives. See results.json and the hashed coverage/probe evidence.

This is a previously inspected historical test winner. Historical book/depth was absent; the baseline uses assumed friction and price-touch execution.
