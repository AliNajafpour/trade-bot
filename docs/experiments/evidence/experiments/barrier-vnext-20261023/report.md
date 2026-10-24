# Barrier vNext results

1-seed models; all128queries and full common cohorts. Forward precision bf16; FP32 likelihood.
Temperature uses separate calibration dates. Every model freezes before calibration and test.

|Fold/model|Horizon min|Raw log loss|Calibrated log loss|Calibrated Brier|Scored|
|---|---:|---:|---:|---:|---:|
|early/pretrained|5|0.30241007160623823|0.30241007160623823|0.17733688914286413|8525620|
|early/pretrained|15|0.42230654905722087|0.42230654905722087|0.2512269208536174|8525239|
|early/pretrained|60|0.5682943641089552|0.5682943641089552|0.34676582077482737|8525143|
|early/pretrained|240|0.6521297582967915|0.6521297582967915|0.40826568578066635|8525143|
|early/pretrained|1440|0.6547646380323644|0.6547646380323644|0.4204698346264528|8525143|

Native Binance price touches, not executable fills or trading returns.
The fixed current asset universe is not survivorship-free. Held-out historical periods are not prospective evidence.
No full per-query predictions stored; reproduce from frozen source/dataset/weights.
Day bootstrap intervals are descriptive under temporal dependence.
