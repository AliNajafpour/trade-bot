# Wallex margin: descriptive long/short grid

Completed 3840/3840 independent1000USDT accounts. Same fixed native test interval: 2026-09-28 09:02:00+00:00 to 2026-10-12 22:58:00+00:00.

Fresh native Wallex API candles equal frozen history exactly. BTC/ETH/SOL eligible; Sonic excluded by current official margin markets. All fees/rules are current public previews replayed historically, not historical borrowing/fill verification.

Strict absolute P(TPfirst by H)>threshold. Direct short model query. H never exits a position. Shared collateral and0.5% stop-risk held fixed;2x leverage does not imply twice the return. Costs include25bp trading and0.015% loan activation/each4h with asset rounding, plus assumed5bp/side friction.

Positive 64; negative 506; zero 3270. Unsupported rows 320. Repeated related accounts are not independent statistical evidence and their profits cannot be added.

|Mode|L|TP%|SL%|H min|p>|Closed|Open|Return%|DD%|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|long_only|1|0.5|1|240|0.65|10|0|-0.456421|0.716452|
|long_only|2|0.5|1|240|0.65|10|0|-0.887398|1.410608|
|short_only|1|2|4|1440|0.50|11|2|-0.302291|1.627468|
|short_only|2|2|4|1440|0.50|11|2|-0.303401|1.630890|
|long_short|1|2|2|1440|0.50|11|1|-0.324970|1.393610|
|long_short|2|2|2|1440|0.50|11|1|-0.539309|1.956927|

Rows above are historical maxima among supported geometries with at least10 closed trades, not selected validated policies. Full matrix includes zero-trade and unsupported rows. Open P&L is a hypothetical liquidation mark; no end-of-data sale.
Original18.24656USDT capital cannot open a position: minimum collateral100USDT atL1 and50 atL2. Original1.343% spot result uses different universe/costs/capital; do not compare it as a matched margin test.
See stress.json for identical fixed parameters at0/5/15bp friction, and ledgers/ for actual trade lists. No LLM news/manager replay was included in this technical-model Cartesian request.

## Best observed rows, with small samples disclosed

The minimum10-trade table above is a separate descriptive summary. There are52 positive supported rows and12 positive extrapolated rows; no supported row with10 or more closed trades is positive. None of these post-hoc maxima selects a validated policy.

|Mode|L|TP%|SL%|H min|p>|Closed|Wins|Open|Realized USDT|Unrealized USDT|Net return%|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|long_only|1|2|4|1440|0.60|2|2|1|3.203356|-0.742570|0.246079|
|long_only|2|2|4|1440|0.60|2|2|1|3.203144|-0.742523|0.246062|
|short_only|1|0.5|2|1440|0.85|2|2|0|0.755012|0.000000|0.075501|
|short_only|2|0.5|2|1440|0.85|2|2|0|1.131305|0.000000|0.113131|
|long_short|1|2|4|1440|0.55|8|8|1|11.907528|-1.868381|1.003915|
|long_short|2|2|4|1440|0.55|8|8|1|11.924917|-1.868497|1.005642|

For the combined best row (TP2%,SL4%,24h,p>.55), fixed-parameter friction stress0/5/15bp returns are:
- L1: 0bp: 1.132085%, 5bp: 1.003915%, 15bp: 0.764577%
- L2: 0bp: 1.132108%, 5bp: 1.005642%, 15bp: 0.765979%

## Original TP2%/SL2%,24h,p>.55, under matched margin rules

|Mode|L1 return%|L2 return%|
|---|---:|---:|
|long_only|0.099670|0.128467|
|short_only|-0.823795|-1.062772|
|long_short|-0.328107|-0.431468|

Data dates in Tehran:2026-09-28 12:32 to2026-10-13 02:28,14.580556 days.3270 rows have no entries. No liquidation occurred in any row. Independent full-path conservative residual collateral bound62.2116% exceeds15%maintenance, including worst price envelope and maximum financing; see independent-margin-maintenance.json.
Independent checks: independent-matrix-verification.json, independent-export-verification.json, raw-source-verification.json. Completion repaired metadata serialization only, preserving inference, simulated trades and their result hashes; see completion-repair.json.
