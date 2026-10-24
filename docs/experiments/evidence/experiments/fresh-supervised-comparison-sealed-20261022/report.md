# Fresh supervised price-path comparison

Frozen VAL winner: **tcn32**, 24h, g1, pTP≥0, netRR≥0. Status: validation_qualified.

All models share116 VAL and116 TEST origins; model/strategy selected on VAL before TEST calculation. Shared initial balance1000USDT; official public fee13bp per side. Additional0/5/15bp adverse slippage are assumptions.

|Model|VAL-selected horizon|RR|pTP|TEST trades|Wins|Losses|Net USDT|Fees|M1 DD %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|hgb15|12h|0|0|21|10|11|-5.6276|7.9916|1.9981|
|hgb31|24h|0.5|0.5|14|9|5|4.1178|5.4422|0.8661|
|tcn32|24h|0|0|12|8|4|17.4339|4.6453|0.8725|
|tcn64|12h|1|0|17|7|10|-4.3026|6.6036|1.0107|
|close_calendar|24h|0|0|1|1|0|3.2518|0.3947|0.3181|

|Forecast model|VAL mean pinball|TEST mean pinball|
|---|---:|---:|
|hgb15|0.00461906|0.00270549|
|hgb31|0.00476040|0.00273968|
|tcn32|0.00441733|0.00286808|
|tcn64|0.00446056|0.00278560|
|TRAIN_unconditional_quantiles|0.00468017|0.00270381|

Pinball covers actual future HIGH, LOW drawdown, and terminal close across all five horizons; lower is better. It does not measure profit.
Pretrained Chronos economic control is evaluated on identical origins. Marginal Chronos CLOSE quantiles are not represented as future HIGH/LOW paths.
No historical news/fundamental backfill; no old July–October fine-tuned weights; daily regimes and the short TEST period limit generalization.