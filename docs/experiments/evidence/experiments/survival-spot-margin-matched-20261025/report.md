# Matched spot / margin correction



Same frozen model, native Wallex M1,7622 identical primary origins,BTC/ETH/SOL,1000USDT,14.58days,position sizing/marked equity,decision and entry clocks,TP/SL exits,0.5%stop risk and15%collateral cap for every arm. Same640 barrier/horizon/threshold configurations in every arm.

Original4asset18.24656USDT anchor reproduced EXACTLY: 1.3431378718012699%. This preserved historical anchor is not the matched comparator.

Equal-cost controls use13bp/side,zero activation/renewal and$1 minimum for every arm. Only long-onlyL1 is a cash-funded spot baseline. Short andL2 equal-cost arms are deliberately hypothetical; their costs/minimums are not executable Wallex margin rules.

Native margin results use saved official25bp/side,financing andminimum100USDTloan. These product terms legitimately differ; paired report separates these effects. Prior3840 native results are reused unchanged.

For a claim about adding short/leverage, compare equal-cost arms at the SAME geometry/horizon/threshold against long-onlyL1, not independently selected historical maxima. All observations are posthoc and candle-price proxies, not verified fills.

## TP2.0% / SL2.0%,H24h,p>.55

|Mode|L|Matched spot%|Equal-cost arm%|Equal-cost delta pp|Native margin%|Native delta pp|
|---|---:|---:|---:|---:|---:|---:|
|long_only|1|0.178913|0.178913|0.000000|0.099670|-0.079243|
|long_only|2|0.178913|0.250994|0.072081|0.128467|-0.050446|
|short_only|1|0.178913|-0.583858|-0.762771|-0.823795|-1.002708|
|short_only|2|0.178913|-0.842919|-1.021832|-1.062772|-1.241685|
|long_short|1|0.178913|-0.046963|-0.225877|-0.328107|-0.507021|
|long_short|2|0.178913|-0.090094|-0.269007|-0.431468|-0.610382|

## TP2.0% / SL4.0%,H24h,p>.55

|Mode|L|Matched spot%|Equal-cost arm%|Equal-cost delta pp|Native margin%|Native delta pp|
|---|---:|---:|---:|---:|---:|---:|
|long_only|1|0.063845|0.063845|0.000000|-0.176360|-0.240205|
|long_only|2|0.063845|0.064617|0.000772|-0.175494|-0.239339|
|short_only|1|0.063845|0.326141|0.262297|-0.014815|-0.078660|
|short_only|2|0.063845|0.324450|0.260605|-0.013378|-0.077223|
|long_short|1|0.063845|1.421294|1.357449|1.003915|0.940070|
|long_short|2|0.063845|1.423482|1.359637|1.005642|0.941797|

## Changes from the original anchor

[
  {
    "symbols": [
      "BTCUSDT",
      "ETHUSDT",
      "SOLUSDT"
    ],
    "initial_cash": 18.246562963625525,
    "entries": 2,
    "closed_trades": 1,
    "wins": 1,
    "losses": 0,
    "win_rate": 1.0,
    "realized_pnl": 0.04426944833444679,
    "open_positions": 1,
    "unrealized_net_pnl": -0.011412618956004206,
    "cash": 15.542267242058587,
    "end_net_liquidation_equity": 18.279419793003967,
    "total_net_pnl": 0.03285682937844214,
    "return_pct": 0.1800713342230198,
    "max_drawdown_pct": 0.2504449962547617,
    "observed_prefix_drawdown_pct": 0.2504449962547617,
    "fees_paid": 0.010755106188584721,
    "unknown_paths": 0,
    "missing_mark_minutes": 0,
    "unknown_equity_minutes": 0,
    "valid_for_comparison": true,
    "exit_counts": {
      "tp": 1
    },
    "skipped": {
      "same_symbol_position_open": 9
    },
    "fee_bps_per_side": 13.0,
    "slippage_bps_per_side": 5.0,
    "stop_spec": "candidate",
    "reward_multiple": 1,
    "ratio_basis": "gross",
    "timeout_exits": 0,
    "open_marks": "Hypothetical net liquidation mark only; no data-end sale or win/loss recorded."
  },
  {
    "symbols": [
      "BTCUSDT",
      "ETHUSDT",
      "SOLUSDT"
    ],
    "initial_cash": 1000.0,
    "entries": 2,
    "closed_trades": 1,
    "wins": 1,
    "losses": 0,
    "win_rate": 1.0,
    "realized_pnl": 2.4261801207546796,
    "open_positions": 1,
    "unrealized_net_pnl": -0.625466778524526,
    "cash": 851.7915002974563,
    "end_net_liquidation_equity": 1001.8007133422302,
    "total_net_pnl": 1.8007133422302104,
    "return_pct": 0.1800713342230198,
    "max_drawdown_pct": 0.2504449962547728,
    "observed_prefix_drawdown_pct": 0.2504449962547728,
    "fees_paid": 0.5894318952026745,
    "unknown_paths": 0,
    "missing_mark_minutes": 0,
    "unknown_equity_minutes": 0,
    "valid_for_comparison": true,
    "exit_counts": {
      "tp": 1
    },
    "skipped": {
      "same_symbol_position_open": 9
    },
    "fee_bps_per_side": 13.0,
    "slippage_bps_per_side": 5.0,
    "stop_spec": "candidate",
    "reward_multiple": 1,
    "ratio_basis": "gross",
    "timeout_exits": 0,
    "open_marks": "Hypothetical net liquidation mark only; no data-end sale or win/loss recorded."
  }
]
Original legacy sizing used entry-reference inventory value andfractional quote-fee accounting. Matched spot andmargin now BOTH use the same marked-equity sizing andlot-aware engine. Thus changes in equity marking/rounding are also disclosed rather than attributing the entire anchor difference to shorting.
Files: controlled-matrix.csv (3840 controls including640 spot baselines),paired-comparison.csv (3840 direct paired comparisons),original-anchor-and-bridges.json,selected-cases.json and24 selected ledgers.
