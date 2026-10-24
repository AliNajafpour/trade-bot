# Conditional opportunity agent: actual results

Fixed three-day historical slice: 2026-10-18 13:30 through 2026-10-21 13:30 UTC. Real Wallex M15 data and actual tool-free Codex responses; simulated spot fills only.

| Market | AI evaluations | Armed | Activated | Expired | Closed trades | Net return | Stress return |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTCTMN | 3 | 3 | 0 | 3 | 0 | 0.0000% | 0.0000% |
| SOLTMN | 9 | 8 | 3 | 5 | 3 | -0.2588% | -0.3184% |

The prior opportunity-agent attempt produced zero closed trades on both markets. Conditional WATCH now creates a positive contingent allocation, a frozen price/volume trigger and a short expiry. Confirmation enters at the next open without another model call. No confirmation means expiry, not a forced trade. Hourly scanning, confirmation monitoring and exits have separate clocks; exhausting the inference budget never cancels exits.

| Market | Entry UTC | Exit UTC | Minutes held | Allocation | Exit signal | P&L, toman |
|---|---|---|---:|---:|---|---:|
| SOLTMN | 2026-10-19T03:00:00+00:00 | 2026-10-19T04:00:00+00:00 | 60 | 5% | stop_close | -5033.58 |
| SOLTMN | 2026-10-20T14:00:00+00:00 | 2026-10-20T15:15:00+00:00 | 75 | 5% | stop_close | -4008.80 |
| SOLTMN | 2026-10-20T21:15:00+00:00 | 2026-10-21T03:15:00+00:00 | 360 | 5% | stop_close | -3897.60 |

Each market starts separately with 5,000,000 toman. Fees are assumed 20bp per side; slippage is 15bp BTC / 25bp SOL per side. Stress adds 10bp fee and 10bp slippage per side to the frozen targets. The three SOL trades lost 12,939.99 toman together; all were losing trades. This repairs inactivity, not profitability.

Verification: source/data/output hashes, all 12 raw response receipts and their file hashes, exact primary and stressed ledger replay, later-bar confirmation, next-open entry/exit, no overlapping filled plans, maximum six-hour holding and final cash inventory passed. The separate offline suite passed all 16 checks; the scoped review closed its findings. Existing 30 research ledgers still replay unchanged with the optional forced-exit feature.

Limits: exploratory follow-up on already inspected historical data; no independent validation or profit claim. Historical LLM knowledge contamination remains possible, and actual served model identity is unverified. Candle fills omit quantity precision, queue/partial fills and historical depth. Forced paper exits may bypass venue minima (zero such exits in these two runs). No exchange order was submitted.

Evidence: `runs/conditional-agent-v2/{BTCTMN,SOLTMN}/result.json`, `decisions.csv`, `outputs.npz`, and hashed `llm/*/receipt.json`. Recheck with `python -B scripts/report_active_agent.py`.
