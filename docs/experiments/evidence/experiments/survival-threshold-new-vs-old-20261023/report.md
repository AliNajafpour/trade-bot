# Old versus new paired strategy backtest



Complete640 paired scenarios,1280 independent account simulations, on8942 identical BTC/ETH/SOL Wallex origins. Each model is replayed on three assets; the earlier four-asset account results are not the comparator.



Primary interval: 2026-09-28T09:02:00+00:00 to 2026-10-12T22:58:00+00:00. Initial18.246562963625525USDT,13bp fee plus5bp assumed execution friction per side. Strict absolute TP probability thresholds, no expiry or data-end sale.



Old three-seed versus new one-seed; different training venues/data/pretraining. Same baseline-compatible ten channels, all activity/age/local/level flags false. Does not isolate architecture. Previously inspected historical TEST, statistically dependent rows.

Historical bid/ask and depth unavailable. Assumed per-side spread/slippage combined friction; price-touch signal replay, not verified executable fills.



Per-model positive marked/negative/no-trade rows: {"old": {"positive": 14, "negative": 84, "no_trade": 542}, "new": {"positive": 57, "negative": 53, "no_trade": 530}}.

Rows are descriptive post-test comparisons and share market paths. No test winner is adopted. Marked P&L includes hypothetical net liquidation of open positions; realized and unrealized fields remain separate.

old-matrix.csv, new-matrix.csv and paired-matrix.csv contain every scenario, support flag, entry count, trade count, fee, drawdown and old/new difference. Both sources training support is independently pinned in protocol.json.

## Sample size and the frozen comparison point

The primary forecast comparison worsened: old mean log loss 0.549701, new 0.616403, a 12.13% increase. More positive strategy rows do not establish better calibration or validated alpha.

|Model|Positive marked rows|Positive with at least5 closed|Positive with at least10 closed|Median closed trades in positive rows|
|---|---:|---:|---:|---:|
|old|14|2|0|2|
|new|57|1|0|1|

Of640 paired scenarios, new marked return was higher in111, lower in16, and equal in513. These outcomes are dependent and frequently reflect different trade counts.

The prior frozen TP0.5%/SL1%, H60, p>0.55 comparison uses the same three-asset account in both models:

|Model|Closed|Wins|Losses|Open|Realized USDT|Unrealized USDT|Return %|
|---|---:|---:|---:|---:|---:|---:|---:|
|old|11|3|8|0|-0.20920227|0.00000000|-1.146530|
|new|3|1|2|0|-0.06347425|0.00000000|-0.347870|

Largest NEW observed marked return: TP1%/SL2%, H1440, p>0.65; 0.287348% or 0.05243108USDT, from only3 closed trades, 3 wins and0 losses, with0 open. Realized0.05243108USDT; unrealized0.00000000USDT. This is an observed post-test row, not an adopted policy.

Neither model has a positive marked-return row with at least10 closed trades. Both have40 explicitly flagged unsupported geometry rows. No forecast horizon or test-end mark forces liquidation.
