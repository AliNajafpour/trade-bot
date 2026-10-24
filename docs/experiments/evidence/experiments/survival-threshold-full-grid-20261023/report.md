# Full TP/SL probability-threshold TEST matrix

Independent verification passed all640 unique Cartesian rows on11566 five minute TEST origins. Nine geometry pairs reuse the original probabilities bitwise; only seven new pairs were inferred. All seven match the public predictor, maximum absolute error1.19209e-7. The prior selected policy and all frozen source/model hashes remain unchanged.

There are13 positive marked-return rows,101 negative rows, and526 no-trade rows. These are separate account simulations sharing the same market paths, so their outcomes are statistically dependent and their profits must not be added. This expanded matrix was requested after the TEST was observed and does not select a validated winner.

The model is the existing native late-fold three-seed ensemble with10080 completed M1 history minutes and its saved temperature calibration. All returns below use the same18.246562963625525USDT account, native13bp fee and assumed5bp execution friction per side.

Largest observed marked returns among rows with at least10 closed trades, all within saved geometry support:

|TP %|SL %|H min|Strict p>|Closed|Wins|Losses|Open|Realized USDT|Unrealized USDT|Marked net USDT|Return %|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|2|2|1440|0.55|17|12|5|1|0.256622|-0.011545|0.245076|1.343138|
|2|4|1440|0.55|55|42|13|2|0.318014|-0.106513|0.211501|1.159129|
|2|4|1440|0.60|39|29|10|1|0.177016|-0.008770|0.168246|0.922070|
|2|4|1440|0.50|59|44|15|2|0.187579|-0.116304|0.071274|0.390618|
|2|4|1440|0.65|16|11|5|0|0.036760|0.000000|0.036760|0.201461|

Open positions are hypothetical net liquidation marks, not data-end sales or wins. Forecast horizons never liquidate positions. The earlier frozen TP0.5%/SL1%,H60,p>0.55 case exactly reproduces its original17 trades,5 wins,12 losses,-0.30902794USDT and-1.69362274% result.

In matrix-baseline.csv, all_test_cohort_threshold_signals counts probability-qualified signals across all11566 TEST origins. primary_threshold_signals applies the native entry-time filter, decision plus one minute inside the reported primary interval, before execution gates. Entries, trades, fees and returns refer to that primary interval.



Coverage: 640/640 independent baseline accounts. Status: complete.



Every account starts with18.246562963625525USDT, native13bp fees and assumed5bp spread/slippage per side. Entries every five minutes, fixed completed-close barriers, one-minute entry latency, TP/SL exits only.

This is a descriptive expansion of the already-observed TEST. No combination is selected as a validated policy from these profits. The previously frozen selected policy is preserved.

The TP0.5%/SL4% pair has a physical log-distance ratio below1/8, outside the saved training support. All40 rows remain visible and flagged; these probabilities are extrapolations.

Primary source interval: 2026-09-28T09:02:00+00:00 to 2026-10-12T22:58:00+00:00. Model/source/account/causality and execution limitations are inherited from the pinned prior protocol.



Results and exact no-trade rows: matrix-baseline.csv. Positive rows, if any, remain historical observations. Historical books/depth are absent; price touches and assumed friction do not verify fills.



Rows: 640. Supported geometries:15/16. Profitable baseline rows: 13; losing: 101; zero-return: 526. Supported profitable rows: 13.

Support flags describe forecast geometry only. The legacy threshold-only execution gates remain fixed for comparability; unsupported rows are not silently removed.
