# Multi-horizon first-touch portfolios

Policy selected exclusively on second-half validation: **validation_qualified**.
Common admitted origins: 292. Fee 13.0bp per side; slippage0/5/15bp are assumptions. Initial shared balance1000USDT.

|Horizon|VAL-selected view|Geometry|pTP min|TEST trades|Wins|Losses|Net USDT|Fees|M1 marked DD %|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|1h|close_calendar|g0|0.0|0|0|0|0.0000|0.0000|0.0000|
|3h|close_calendar|g2|0.0|1|0|1|-1.0663|0.3891|0.1066|
|6h|close_calendar|g0|0.0|1|0|1|-2.0574|0.3878|0.2057|
|12h|close_calendar|g0|0.0|3|2|1|0.0638|1.1715|0.0829|
|24h|close_calendar|g0|0.0|3|3|0|7.5896|0.9752|0.3500|

The table evaluates a separately VAL-selected policy for each horizon. The global frozen winner is recorded in selected-policy.json. TEST diagnostic rows never select a policy.

Daily forecast origins, not intraday scanner frequency.
Historical spread unavailable; 0/5/15bp adverse price haircuts are assumptions.
Historical numerical fundamentals use assumed48h delay; historical full-news/LLM decisions were not backtested.
Many validation configurations; TEST is exposed by diagnostics and must not be reused for tuning.

Inspect comparison.csv and ledgers/ for every configuration, including zero trades and accepted unresolved outcomes.