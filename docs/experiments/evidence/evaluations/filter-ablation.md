# Entry-filter ablation

All policies use fixed15% exposure, official30bp TMN taker fees, and frozen validation margins. Filter bit order is cost, geometry, price, volume. Outcomes never select a filter setting.

| Model | Filters | Slip bp | Return % | Trades | Wins/losses | DD % | Armed/expired |
|---|---|---:|---:|---:|---:|---:|---:|
| hgb | 0000 | 15 | -0.032 | 7 | 4/3 | 0.848 | 7/0 |
| hgb | 0001 | 15 | -0.013 | 7 | 4/3 | 0.722 | 7/0 |
| hgb | 0010 | 15 | 0.145 | 3 | 1/2 | 0.286 | 9/6 |
| hgb | 0100 | 15 | -0.032 | 7 | 4/3 | 0.848 | 7/0 |
| hgb | 1000 | 15 | -0.117 | 3 | 1/2 | 0.573 | 3/0 |
| hgb | 1111 | 15 | 0.359 | 1 | 1/0 | 0.161 | 3/2 |
| hgb | 1111_ttl30 | 15 | 0.359 | 1 | 1/0 | 0.161 | 3/2 |
| mean_hgb_gru | 0000 | 15 | -0.045 | 6 | 3/3 | 0.848 | 6/0 |
| mean_hgb_gru | 0001 | 15 | 0.095 | 6 | 3/3 | 0.722 | 6/0 |
| mean_hgb_gru | 0010 | 15 | 0.155 | 2 | 1/1 | 0.286 | 7/5 |
| mean_hgb_gru | 0100 | 15 | -0.304 | 5 | 2/3 | 0.848 | 5/0 |
| mean_hgb_gru | 1000 | 15 | -0.233 | 1 | 0/1 | 0.234 | 1/0 |
| mean_hgb_gru | 1111 | 15 | 0.000 | 0 | 0/0 | -0.000 | 1/1 |
| mean_hgb_gru | 1111_ttl30 | 15 | 0.000 | 0 | 0/0 | -0.000 | 1/1 |

## Ex-post blocked-candidate audit

A profitable blocked candidate means its separate next-open to original six-hour-deadline counterfactual was positive after the declared fees/slippage. This diagnostic uses future prices only after decisions. These dense candidates overlap; their PnL sums are not portfolio profit or missed attainable profit.

| Model | Filter | Slip bp | Blocked profitable | Blocked lossmaking |
|---|---|---:|---:|---:|
| hgb | cost | 15 | 10 | 6 |
| hgb | geometry | 15 | 0 | 0 |
| hgb | price_60m | 15 | 6 | 4 |
| hgb | volume_60m | 15 | 0 | 3 |
| hgb | ttl30_vs60 | 15 | 1 | 0 |
| mean_hgb_gru | cost | 15 | 5 | 5 |
| mean_hgb_gru | geometry | 15 | 1 | 0 |
| mean_hgb_gru | price_60m | 15 | 5 | 3 |
| mean_hgb_gru | volume_60m | 15 | 0 | 2 |
| mean_hgb_gru | ttl30_vs60 | 15 | 0 | 0 |

## Limits

- Already exposed TEST and after-the-fact filter analysis, no new independent validation
- Technical deterministic WATCH policies only, not actual LLM/news fusion performance
- Fixed15% exposure deliberately removes production risk-sizing differences
- Only complete six-hour plans enter; no truncated final-window entries
- Each counterfactual uses next-open/deadline-open prices without stop/target; it differs from the actual strategy exit policy
- Dense overlapping candidate PnL is nonadditive and cannot be traded simultaneously with one account
- Portfolio candidate counters count only while flat; the diagnostic audit includes all overlapping forecast candidates
- Candle proxy fills, no historical books; stops observe closes then exit next open
