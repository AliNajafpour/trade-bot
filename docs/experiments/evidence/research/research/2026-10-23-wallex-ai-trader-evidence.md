# Wallex AI Trader: public evidence review

Reviewed 2026-10-23. Repository `ssepehrnoush/wallex-ai-trader`, pinned commit `60fa57ff7e8253e9ef16baf9e5361df736376809`.

The repository documents a plausible fair-value reversion hypothesis and substantial execution/risk engineering. Its public results do **not establish a profitable executable edge**. The authors themselves say they have not found a reliable edge after fees on real order books. [README, line 104](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/README.md#L104)

## Scope and verification

This review read first-party documentation and public JSON/CSV snapshots through GitHub. No repository code was executed, no trading endpoints were called, and no private files or account state were inspected. "Recomputed" below means arithmetic over the published rows, not an independent rerun of market data, signal construction, execution, or the risk manager.

The public snapshot has several dates. The zero-trade live results in `RESULTS.md` apply to **2026-10-21 21:20 UTC**, not the pinned commit's latest live state. The latest committed `state.json` is **2026-10-23 15:51:40 UTC**, with three closed trades. [Older cutoff](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L249-L277), [latest state](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/live_main/state.json#L1-L10)

## What is being tested

The hypothesis is that the local executable ask temporarily falls below its normal relationship to global value. Global value is Binance price multiplied by Wallex USDT/TMN; a trailing 180-minute median removes the market's normal premium. The reference rule buys at a deviation of at least -0.7%, with a 0.3% taker fee per side, a -0.9% price stop, and 25-minute maximum hold. A target exit requires both premium convergence and enough bid appreciation to cover fees plus a margin. The historical input is one-minute last-trade closes with a fixed measured spread added/subtracted, rather than historical executable books. [Method and costs](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L33-L42)

## Main comparison

The 30-day window is 2026-09-22 07:00 to 2026-10-22 07:00 UTC, over eight TMN markets. "Next minute" means the **next traded minute**, which can be more than 60 seconds later. These scenarios generate different trade counts and are not paired identical fills. [Summary and convention](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_live_rule_30d/summary.json#L2-L25)

| Published scenario | Raw trades | Mean net return per trade | Account trades | Return on 5m TMN account |
|---|---:|---:|---:|---:|
| Reference rule, signal-minute fill, size factor 0.5 | 361 | +0.136% | 276 | +360,109 TMN, +7.20% |
| Same rule, next traded-minute fill, factor 0.5 | 353 | -0.487% | 33 | -166,993 TMN, -3.34%; kill switch on Sep 25 |
| Next traded-minute, factor 0.5, kill switch disabled | 353 input trades | -0.487% input mean | 231 | -1,077,989 TMN, -21.56% |
| Reference signal-minute, factor 0.3 | 361 input trades | +0.136% input mean | 276 | +229,629 TMN, +4.59% |
| Reference next traded-minute, factor 0.3 | 353 input trades | -0.487% input mean | 45 | -152,721 TMN, -3.05% |
| Static "AI policy" parameters, signal-minute, factor 0.5 | 262 | +0.205% | 195 | +334,463 TMN, +6.69% |
| Static "AI policy" parameters, next traded-minute, factor 0.5 | 256 | -0.480% | 29 | -155,364 TMN, -3.11% |

Sources: [reference raw summary](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_live_rule_30d/summary.json#L47-L61), [reference account scenarios](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_live_rule_30d/toman.json#L35-L110), [smaller sizing](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_live_rule_30d/toman.json#L134-L178), [AI policy raw summary](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_ai_policy_30d/summary.json#L47-L61), [AI policy account scenarios](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_ai_policy_30d/toman.json#L35-L80).

The account simulation changes equity only when positions close, assumes full fills, and disables contest-end phases. Published maximum drawdown is 2.34% for the +7.20% case, but open-position drawdowns and available depth are not reconstructed. Thus this is a conditional simulation result, not a verified achievable monthly return. [Account assumptions](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_live_rule_30d/toman.json#L2), [discussion](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L176-L189)

### Arithmetic independently checked

Summing the public `ret_pct` CSV column reproduces the rounded summary:

| Input | Rows | Sum of trade percentage points | Arithmetic mean | Winners |
|---|---:|---:|---:|---:|
| Signal-minute CSV | 361 | +48.9158987597 | +0.1355011046% | 222, 61.50% |
| Next traded-minute CSV | 353 | -171.7928152993 | -0.4866651991% | 110, 31.16% |

The five largest signal-minute returns sum to +25.3964575079 percentage points, about 51.9% of the total. Removing them leaves +0.0660658462% mean across 356 trades. **Summed percentage points are not account return.** Source rows: [signal-minute CSV](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_live_rule_30d/trades.csv#L1-L362), [next traded-minute CSV](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/backtests/fv_live_rule_30d/trades_next_minute.csv#L1-L354).

## Executable-book and live evidence

The recorded-book replay covers 16.24 hours from Oct 21 16:45:30 to Oct 22 08:59:58 UTC. It switches every market except POL on, independently of the historical AI policy. At the 0.7% threshold it has **three trades, zero wins, -1.083% average**, all time exits after 25 minutes: ARB -1.423%, BNB -0.991%, BNB -0.836%. This is a counterfactual replay, not three executed live losses. It omits the 60-second drop filter, loss cooldown, and partial fills. The authors report those markets were off under AI at the time, so the policy avoided these particular losses. Three avoided entries cannot establish general AI value. [Replay definition and omissions](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L207-L220), [recorded replay results](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/docs/live_tick_replay.json#L351-L382)

During the 14.25-hour overlap with the 30-day candle test, one of three candle entries had no corresponding executable-book signal; the other two were already losing trades in the candle test. [Overlap results](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L195-L200)

The pinned live state reports equity **4,996,224 TMN**, **-0.076%**, **three closed trades**, and no open positions. The public app snapshot lists these PEPETMN exits:

| Exit UTC | Reason | Trade return | Account P&L |
|---|---|---:|---:|
| 2026-10-22 22:15:40 | Time | -0.139% | -668 TMN |
| 2026-10-23 08:17:24 | Target | +0.051% | +245 TMN |
| 2026-10-23 08:36:31 | Stop | -0.699% | -3,353 TMN |

The three P&Ls add to -3,776 TMN, agreeing with 5m minus the latest equity. All three recorded trade-agent decisions were "buy", with response latency 5.01, 4.20, and 5.37 seconds. The app snapshot was generated at 15:44:08 UTC, before the latest state. These are first-party recorded outcomes, not independently verified exchange statements. [Latest live state](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/runs/live_main/state.json#L1-L10), [public snapshot trades and decisions, single-line JSON](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/docs/app/data.json#L1)

## Holdouts, selection, and AI added value

- The seven-day reference result is 67 trades, +0.291% mean. PEPE, BNB, and ARB were selected on those same seven days, account for 59/67 trades, and the authors explicitly say the four-day/three-day split does not remove that selection bias. Extending backward to 30 days is useful robustness evidence, but does not create a clean prospective holdout. [Selection](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L59-L65), [limitations](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L388-L395)
- The reference week's mean more than halves over 30 days. Five weekly next-minute means are all negative; the first two same-minute weeks are near zero. [Weekly results](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L159-L174)
- Four other families were tested with parameters selected on the first 20 days and reports on the last ten days. They were rejected as unviable. This does not mean every holdout variant lost: the trend report has three of 32 positive out-of-sample variants, driven by one PEPE gain, while none was profitable in-sample. These controls concern alternative strategies, not whether AI improves this strategy. [Experiment design and outcomes](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L476-L489)
- The "AI policy" backtest is a static alternative parameter set approximating one policy review, not a historical replay of time-varying LLM decisions. It preserves the same latency fragility. [Parameter description](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L191)
- The per-trade LLM was enabled on Oct 22 at 15:17 UTC. Documentation acknowledges that it costs seconds and says improved win rate is unproven. One manual response took about three seconds; the latest public real decisions took 4.2 to 5.37 seconds. Responses expire after 20 seconds and the opportunity must still exist. [Agent design and evidence](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/RESULTS.md#L452-L472)

There is no controlled, sufficiently sized out-of-sample AI-versus-fixed-policy comparison in the inspected evidence. The three avoided replay losses are favorable examples; the three actual buys, two losses and one win, are far too few to settle the question.

## Additional source-level risks

These are static code findings, not reproduced live incidents.

- **The TMN conversion reference can outlive its book.** `step()` updates `self.usdttmn` only when `USDTTMN` is present. If that book disappears, its prior rate remains usable in `fair_price()` and the entry checks have no age check for it. A changed FX rate during this gap can create a false discount despite fresh coin quotes. Require a fresh timestamped FX reference for new TMN entries. [Reference update](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/wallexbot/engine.py#L425-L434), [fair-price use and entry checks](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/wallexbot/engine.py#L456-L525).
- **An open position uses the latest policy's stop and holding time.** `Position` stores neither its original stop nor holding limit; policy reviews replace `self.policies`, and exits read that current policy. An AI review can therefore loosen a position's stop after sizing it against a tighter stop. Freeze initial limits per position, or permit subsequent tightening only. [Position fields](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/wallexbot/engine.py#L35-L44), [policy replacement](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/wallexbot/engine.py#L386-L387), [exit checks](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/wallexbot/engine.py#L456-L489).
- **Contest constraints override some model choices.** Healthy regular markets can be forced on, regular-market entry thresholds are capped, and size is clamped to a minimum as well as maximum. This intentionally targets contest participation; it should not be mistaken for unconstrained risk optimization. [Contest bounds](https://github.com/ssepehrnoush/wallex-ai-trader/blob/60fa57ff7e8253e9ef16baf9e5361df736376809/wallexbot/supervisor.py#L103-L111).

## Overall assessment

This is a transparent prototype with useful evidence against its own optimistic backtest. The central unresolved issue is whether the apparent discount is available at executable prices long enough to overcome both fees, spread, model latency, and fill risk. The published delay sensitivity and book replay currently argue against treating its headline +7.20% as a validated edge.

