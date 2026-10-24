# Twenty-method recent-day backtest

Retrospective comparison of the same frozen twenty methods. Full-day figures precede some recorded model/policy availability clocks; model_available and suite_available views explicitly restrict entries when requested. No full-day untouched/prospective claim.

Decision at completed candle close, next native minute OPEN at least one minute later; real historical bid/ask and depth are unavailable. 0/5/15bps per-side execution haircut scenarios. No invented historical order-book fill.

Complete: 18/20. Independent initial100USDT per method. Public level1 taker13bp each side; account tier unverified.

|Method|Entries|Closed|Wins|Losses|Open|Realized USDT|Marked total PnL USDT|DD %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|TCN · SL۲٪ / TP۸٪ · ورود عددی|8|5|0|5|3|-1.7728|-1.8777|2.5742|
|TCN · SL۲٪ / TP۸٪ · خبر و بنیاد|pending||||||||
|TCN · SL آموخته‌شده / R۴ · ورود عددی|6|3|0|3|3|-1.1585|-1.7101|2.1975|
|TCN · SL آموخته‌شده / R۴ · خبر و بنیاد|pending||||||||
|TCN · SL آموخته‌شده / R۳|6|3|0|3|3|-1.1585|-1.7101|2.1975|
|TCN · SL۱٪ / TP۳٪|11|9|1|8|2|-1.3306|-1.4589|2.0880|
|TCN · SL۳٪ / TP۳٪|6|3|1|2|3|-0.5855|-1.6287|2.3125|
|HGB · SL۲٪ / TP۸٪|6|3|0|3|3|-1.0958|-1.4320|2.1233|
|Chronos-2 · SL۲٪ / TP۸٪|0|0|0|0|0|0.0000|0.0000|0.0000|
|Kronos · SL۲٪ / TP۸٪|7|3|0|3|4|-1.0572|-1.5118|1.8406|
|ترکیب واقعی HGB و GRU · افق۶ ساعت|16|16|1|15|0|-1.4594|-1.4594|1.5179|
|Kronos · پیش‌بینی مستقیم قیمت|3|3|0|3|0|-0.1821|-0.1821|0.1821|
|Kronos · سود و زیان مسیرهای نمونه|11|11|1|10|0|-0.5546|-0.5546|0.6261|
|Chronos-2 · قیمت و تکنیکال|0|0|0|0|0|0.0000|0.0000|0.0000|
|Chronos-2 · تکنیکال، بنیاد عددی و خبر|0|0|0|0|0|0.0000|0.0000|0.0000|
|Chronos-2 · TP پیش از SL · حداقل نسبت۱|0|0|0|0|0|0.0000|0.0000|0.0000|
|TCN · برخورد اول · افق۱ ساعت|17|16|3|13|1|-0.9007|-0.8830|1.0544|
|TCN · برخورد اول · افق۳ ساعت|16|13|0|13|3|-1.5017|-1.4384|1.6570|
|TCN · برخورد اول · افق۶ ساعت|17|14|1|13|3|-1.9141|-1.9210|2.5754|
|TCN · برخورد اول · افق۱۲ ساعت|10|6|0|6|4|-1.5330|-2.2005|2.9375|

Archived causal news/fundamental evidence only; missing evidence is REJECT under the original model_context contract. Real Codex reviews use original role prompts/schemas; retrospective inference latency is not reproduced.

Original timed expiry remains for horizon methods; barrier methods have no timeout. Open positions are marked net of hypothetical exit fee, never forced sold or counted as wins.