# Isolated net-RR comparison

Frozen model: close_calendar, 24h, g0; common origins 292.
VAL-selected minimum net RR: **0.5** (validation_qualified). RR0 means EV-only with positive net target payoff.

Previously exposed historical TEST: retrospective diagnostic, not a fresh untouched holdout.

|RR floor|VAL trades|VAL net USDT|TEST trades|Wins|Losses|TEST net USDT|Fees|Marked DD %|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|13|-0.9773|21|11|10|INVALID|7.1077|4.7599|
|0.25|10|6.0705|20|10|10|INVALID|6.7146|4.8306|
|0.5|6|10.3976|14|7|7|INVALID|4.8627|4.5517|
|0.75|6|10.3976|7|5|2|INVALID|2.5426|3.1653|
|1|6|10.3976|3|3|0|7.5896|0.9752|0.3500|
|1.5|4|9.4753|1|1|0|0.0853|0.3906|0.0651|
|2|4|9.4753|1|1|0|0.0853|0.3906|0.0651|

Official public USDT level1 fee: 13.0bp per side. Additional0/5/15bp slippage scenarios are assumptions. Shared initial cash1000USDT; no leverage.

Daily forecast origins; live intraday frequency not validated.
Slippage0/5/15bp per side is an assumption; historical spread unknown.
Historical full news/LLM decisions were not backtested.
Candidate profitability counts do not imply independently allocatable profits.

RR1 blocking diagnostics:
- validation: 7 blocked candidates; 3 profitable, 4 losing, 0 censored; exits {'tp_first': 3, 'sl_first': 4}.
- test: 24 blocked candidates; 11 profitable, 12 losing, 1 censored; exits {'tp_first': 8, 'timeout': 10, 'sl_first': 5}.

No sum of isolated candidate PnL is presented as allocatable portfolio profit. Inspect attribution.json and exact shared-cash ledgers.