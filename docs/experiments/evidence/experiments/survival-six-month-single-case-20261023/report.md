# Six-month single-strategy replay

TP2%, SL2%, absolute P(TPfirst by24h)>0.55; unchanged old three-seed weekly model.
Interval: 2026-04-22T16:00:00+00:00 to 2026-10-22T16:00:00+00:00, six calendar months, 183 days.
Six months includes training and calibration. This does not measure six-month out-of-sample generalization.
Missing candle paths censor account results; null is not zero profit. No interpolation or forced horizon/data-end sale.
Original scenario reproduced exactly: +1.3431378718%,17closed12TP5SL,1open.

|Scope|Closed|Wins|Losses|Open|Unknown paths|Valid full return %|
|---|---:|---:|---:|---:|---:|---:|
|six-months|105|66|39|1|1|UNAVAILABLE|
|after-calibration|18|13|5|1|1|UNAVAILABLE|

Historical OHLC touches and assumed friction are execution proxies; no historical depth/fills.
Accounts are independent across the six-month and after-calibration diagnostic scopes; no strategy parameters selected from these results.

## Observed prefixes before a held-position data gap

six-months: 2026-04-22T16:00:00+00:00 to 2026-07-28T16:02:00+00:00, 97.0014 days, marked net return 0.576718%, 105closed, 1open, maximum drawdown 5.062019%.

after-calibration: 2026-09-21T00:00:00+00:00 to 2026-10-12T22:58:00+00:00, 21.9569 days, marked net return 1.591895%, 18closed, 1open, maximum drawdown 0.843775%.

These are explicitly shorter observed prefixes, not six-month returns. The six-month prefix includes training data; the after-calibration prefix does not, but this strategy was selected after inspecting the historical test.

Cross-device FP32 public predictor error exceeded the initial2e-5 check. Maximum across the boundary audit: 3.76701355e-05. All135 origins within0.001of the0.55 entry threshold were rechecked through the canonical CPU public predictor; zero entry decisions changed. Original frozen caches were preserved bitwise.
