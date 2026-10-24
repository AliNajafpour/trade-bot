# Survival probability threshold backtest

Five minute decisions lost 0.30902794 USDT, or 1.69362274%, under the policy selected before test outcomes. Initial 18.24656296 USDT ended at 17.93753502 USDT. There were 17 closed trades, five TP exits and 12 SL exits, with no open positions. This threshold strategy failed this historical test.

The frozen model is the native late-fold three-seed ensemble with 10080 completed M1 history minutes and its saved temperature calibration. All 26542 eligible five minute origins were evaluated. Selection used five closed trades, all winners, for only +0.02402971 USDT. Selection qualification means at least five closed trades; it does not establish profitable trading.

Selection status: selection_qualified. Frozen TP 0.50%, SL 1.00%, horizon 60 minutes, p>0.55.

Initial 18.24656296 USDT. Fee13bp and assumed spread/slippage5bp each side. Primary 2026-09-28T09:02:00+00:00 to 2026-10-12T22:58:00+00:00.

|p strict >|Entries|Closed|Wins|Losses|Open|Realized USDT|Unrealized USDT|Marked net USDT|DD %|Fees USDT|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0.50|93|93|50|43|0|-1.023197|0.000000|-1.023197|5.963795|0.643435|
|0.55|17|17|5|12|0|-0.309028|0.000000|-0.309028|1.714130|0.119795|
|0.60|2|2|1|1|0|-0.027680|0.000000|-0.027680|0.264602|0.014207|
|0.65|0|0|0|0|0|0.000000|0.000000|0.000000|0.000000|0.000000|
|0.70|0|0|0|0|0|0.000000|0.000000|0.000000|0.000000|0.000000|
|0.75|0|0|0|0|0|0.000000|0.000000|0.000000|0.000000|0.000000|
|0.80|0|0|0|0|0|0.000000|0.000000|0.000000|0.000000|0.000000|
|0.85|0|0|0|0|0|0.000000|0.000000|0.000000|0.000000|0.000000|

Strict absolute P(TP before SL by horizon)>p. No-touch is not a win.
Forecast horizon does not liquidate. TP/SL exits only; end marks include hypothetical liquidation costs and are separate from realized P&L.
Historical bid/ask and depth unavailable. Assumed per-side spread/slippage combined friction; price-touch signal replay, not verified executable fills.
Repeated historical project dates. Temporal test for this strategy protocol, not project-wide untouched or prospective evidence.
Cash-funded long spot only, one position per asset, max15% allocation, .5% equity risk, $1 min notional, shared cash. Existing barrier_only simulator. Sizing equity is cash plus open quantity times entry reference price, not current marked equity. Four assets make native cap5 equivalent to cap4 here. Fractional quantities are paper proxies.
One minute from completed-candle decision to next native M1 OPEN proxy. Probability horizon begins at decision; execution starts one minute later. No future entry price enters model features or queries.
Historical threshold-only sweep has no no-touch cap, conditional net-margin or minimum netRR gate. Live operational paper policy initially also requires no-touch<=.25, positive conditional net margin, netRR>=1, supported entry gap and real fresh bid/ask/depth. Capital risk limits agree; signal eligibility differs.
Threshold curve is descriptive and predeclared. Test profit never selects p, geometry, or horizon. Full-period gap-censor diagnostics and 0/15bp friction scenarios are in report.json and threshold-curve.csv.
Simultaneous signals rank by absolute TP probability, then symbol. Net target reward must cover actual assumed entry/exit costs. A high TP probability can still have poor economics when the stop is wider.
All model probabilities are inferred once for every fixed geometry/horizon, shared across every threshold. No independent-label profit summation.

The selected p>0.55 policy lost 1.43102845% with zero assumed spread/slippage, 1.69362274% with 5bp, and 1.87647437% with 15bp per side; each scenario also charged the actual 13bp fee per side. The full September 21 to October 21 diagnostic had the same selected-policy results and zero unresolved paths. Higher thresholds of 0.65 through 0.85 produced no entries for the frozen geometry and horizon; these are abstentions, not successful trades.

Native M1 zero-volume fractions in the primary interval were BTC 69.30%, ETH 85.08%, SOL 84.28%, and S 92.15%. Continuous candle coverage does not establish executable liquidity. Source hashes and exact counts are in execution-coverage.json.

In native trade ledgers, the legacy original_expected_net field stores the absolute TP probability used to rank signals. It is not an expected-return estimate. Independent verification passed the exact cohort, strict threshold, frozen policy/source hashes, normalized probabilities, public-predictor parity, cash/accounting, censoring and no-timeout checks. Maximum public-predictor probability difference was 1.78814e-7. The smaller fifteen minute run is a superseded pilot.
