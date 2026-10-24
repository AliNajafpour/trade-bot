# Improving the arbitrary-barrier first-touch model

Research date: 2026-10-23. This review combines inspected local code and artifacts with primary research and official data documentation. Research files are the only additions. Existing numerical sources, labels, model weights, comparison protocols and running jobs were not changed.

The recommendation is to improve the observation and target contract, add independent market history, and test a level-aware multiscale encoder with self-supervised pretraining. Increasing parameter count is a separate experiment after these changes. Published forecasting results motivate experiments; they do not establish a better TP/SL classifier or profitable trading.

## Contract that every candidate must retain

At a decision after the latest completed candle, accept symbol, executable or explicitly hypothetical entry, arbitrary TP/SL prices or percentages, and available historical observations. Output TP-first, SL-first and neither-hit probabilities at 5, 15, 60, 240 and 1440 minutes. Each triple sums to one. Neither-hit is not a win, and the deadline does not close a position. Report `p_tp / (p_tp + p_sl)` separately, alongside the chance of any hit and an unsupported/undefined flag where appropriate. There is no one-minute forecast product.

Minute or tick data can improve inputs and labels without changing those five output deadlines. Price-trigger probability, executable-quote crossing and completed-order fill are different tasks. Keep the supplied entry in the request; do not silently substitute a historical close or a quote.

## Findings from the present implementation

Evidence: [completed report](../../experiments/barrier-survival-multiday-20261022/report.md), dataset manifest (`../../data/barrier-dense-20261022/dataset.json`; local supporting artifact), [cohort audit](../../experiments/barrier-survival-multiday-20261022/cohort-audit.json), [clipping audit](../../experiments/barrier-survival-multiday-20261022/input-clipping-audit.json), [model](../../../../../wallex_lab/barrier_survival.py), [dense features](../../../../../wallex_lab/barrier_dense.py), and [earlier diagnostic research](barrier-loss-primary-sources.md). These are completed historical experiment results, not a fresh training run.

* The dataset contains 2,100,229 M1 bars from four Wallex USDT markets over one year, with 414,289 five-minute origins and 128 barrier queries per origin. The resulting 53,028,992 requests reuse the same paths. Their count is not an independent-sample count.
* The selected weekly encoder has approximately 382,179 parameters. Its recent, middle and distant branches produce 120, 64 and 168 tokens, respectively. There are 352 history tokens and 640 barrier/time queries. Convolutional patch projections can retain ordering within a patch; they are not orderless OHLC aggregation. Older aggregated bars nevertheless lose their intrabar chronology.
* Relative log prices, causal volatility scaling and relative volume already remove raw price-unit differences. Bitcoin's price being numerically larger than Sonic's is not an untreated problem. Activity, spread, tick size and volatility regimes remain real differences worth encoding. The sampled weekly audit puts roughly 1.80% of each price channel at the clipping limit. That merits a normalization ablation, not a claim that all inputs collapsed.
* The likelihood handles observed event bins, unknown event cause and right censoring. Hazard-to-incidence construction provides coherence across time. It does not ensure consistency across different requested barrier geometries.
* All six weekly seed/fold fits selected epoch 1 or 2, with subsequent fit improvement and selection deterioration. This is repeatable deterioration, not proof of one particular cause. A previous controlled CPU continuation from epoch 2 compared learning rates 0.0006 and 0.0002 on 1,024 fit/512 selection origins. Neither retained an improvement after 96 updates. An optimizer restart and the small cohort limit the causal conclusion, but lowering LR alone is already an unsuccessful diagnostic, not an untested cure.
* Strict contiguous-week eligibility retains 358,517 of 414,289 origins, excluding 13.46%. Early calibration contains only five distinct UTC dates; late test contains twelve. A large number of queries cannot compensate for this small number of independent market episodes.
* The history-shuffle diagnostic damages predictions, but also makes physically incoherent inputs. It demonstrates sensitivity to the perturbation, not discovery of named technical patterns or independent directional alpha.

The completed calibrated three-class log losses below are lower-is-better. Query-only sees geometry, asset and volatility, but no candle sequence.

| Test fold | Horizon | Query-only | Our history model | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Early | 5m | 0.27143 | 0.27086 | Small descriptive difference |
| Early | 1h | 0.65262 | 0.62755 | Clearer descriptive gain |
| Early | 4h | 0.77970 | 0.74255 | Clearer descriptive gain |
| Early | 24h | 0.74589 | 0.72342 | Gain in this fold |
| Late | 5m | 0.29915 | 0.29133 | Gain in this fold |
| Late | 1h | 0.67133 | 0.64206 | Clearer descriptive gain |
| Late | 4h | 0.75586 | 0.72538 | Clearer descriptive gain |
| Late | 24h | 0.70539 | 0.70468 | Nearly tied; do not claim robust 24h superiority |

The window search changed historical length and aggregation layout together. It selected a useful configuration, but does not isolate an effect of length. Current comparison work with external encoders and TCN must finish under its frozen protocol before using it for a model ranking.

## The most urgent data finding

Fresh full-file activity counts from the four source CSVs:

| Symbol | M1 rows | Zero-volume rows | Zero-volume share | High equals low |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 525,239 | 395,646 | 75.3269% | 89.9600% |
| ETHUSDT | 524,996 | 434,927 | 82.8439% | 93.5678% |
| SOLUSDT | 525,107 | 450,074 | 85.7109% | 95.4994% |
| SUSDT | 524,887 | 500,289 | 95.3137% | 99.1442% |

No zero-volume row has a nonflat range. This is consistent with many carried last prices during inactive minutes. It does not prove an erroneous feed, and a zero-volume bar does not prove unchanged executable quotes. Trade inactivity, stale quotes, failed collection and exchange halts require separate evidence and flags. The [data research appendix](2026-10-23-barrier-data-primary.md) includes a runnable full-file counting check, verified Sonic identity and source receipt hash. SUSDT is Sonic/USDT in the saved official venue metadata.

The current labels concern local chart prices and synthetic offsets from the completed close. They do not establish the result of entering at the live ask and exiting at the bid. For 5/15m and narrow levels, resolving that distinction is more consequential than another generic forecasting backbone.

Do not delete flat minutes to make the series appear active. The target uses wall-clock deadlines. Instead expose time since last trade, inactive-minute fraction, flat-run duration, price-change frequency, observation age and missingness. Reconstruct a small same-venue trade sample against its bars before asserting which construction rule produced them.

## Data acquisition, in order

| Priority | Addition | What it can contribute | Main limitation |
| --- | --- | --- | --- |
| 1 | Target-venue ordered trades and best bid/ask, with timestamps and sequence IDs | Credible first-touch labels; spread, staleness, activity and aggressor flow | Current endpoints are not proof of annual archives |
| 2 | Several independent years and a point-in-time universe of assets | More regimes and less dependence on one year/four markets | Listing/delisting and venue coverage must be retained |
| 3 | Synchronized BTC/ETH and major-venue spot/perpetual history | Common market movement, cross-venue basis and local lag | Available-by-decision joins and venue identity are essential |
| 4 | Funding, OI, premium/mark/index and spot-perpetual basis | Positioning/financing context, especially at 1/4/24h | History limits, units, aggregation and publication delay vary |
| 5 | Replayable depth and order events | Queue/depth imbalance, cancellations and order-flow imbalance at 5/15m | Harder historical acquisition and clock/sequence validation |
| 6 | Versioned liquidation observations | Forced-flow stress context | Public streams may be selected snapshots, not complete tape |

A practical history expansion could test three to five years and 20 to 50 assets where available. These are proposed experiment ranges, not literature-derived optimal sizes. Sonic cannot be given history from before its actual listing by silently stitching unrelated symbols. Include inactive and delisted instruments where possible and decide eligibility using information known at each historical date.

[Binance's official archive](https://github.com/binance/binance-public-data) documents downloadable spot/futures trades, aggregate trades, klines and checksums. It also documents microsecond timestamps for spot data from 2025. This is a realistic route to richer history, not evidence that identical Wallex execution prices are available. Freeze source bytes/checksums because archives can be corrected.

The [Binance derivatives reference](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data) limits OI history to the latest month and basis to the latest 30 days. Do not promise years of OI from those REST endpoints. [Coinbase's channels](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels) document live trade/depth reconstruction, not a free multi-year LOB archive. [Bybit advertises historical data](https://www.bybit.com/future-activity/developer); exact book coverage still requires a small retrieval and schema audit. More acquisition detail and qualifications are in the data appendix.

[Cont, Kukanov and Stoikov](https://arxiv.org/abs/1011.6402) relate short-interval US-stock price changes to order-flow imbalance and depth. [Kolm, Turiel and Westray](https://onlinelibrary.wiley.com/doi/abs/10.1111/mafi.12413) study stationary order-flow inputs on 115 Nasdaq stocks, with effective forecasts on very short price-change scales. These motivate an experiment; neither establishes 24h crypto predictability. OHLCV-derived signed volume is not actual observed aggressor flow.

Gold is a later domain-transfer experiment. Preserve sessions, weekends, currency, instrument IDs and futures contract rolls; broker spot volume and exchange futures volume are different observations. [CME's depth documentation](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457091894/Market%2BDepth) describes historical exchange depth, with product/date coverage and licensing to verify. Pooling gold bars into crypto without these distinctions is not a justified improvement.

## Labels and query distribution

Keep existing partial/ambiguous observations honest. An unresolved M1 double touch contributes the sum of TP and SL event mass, not a fabricated order. Only 27,451 of the 53,028,992 current requests have unknown cause, approximately 0.052%. Thus tie resolution alone is unlikely to explain a large whole-dataset score gain. Quote-versus-last-trade semantics and narrow-barrier realism remain important even with few ties.

On ordered trade tape, label the earliest crossing after the decision. Equal timestamps without a supported sequence remain ambiguous. A small immediate long enters at ask and sells at bid; a short reverses those sides if that trade is supported by the venue. Size-specific fills need depth and latency. A touched resting limit does not guarantee queue execution. Preserve hypothetical-entry requests as hypothetical and expose their distance from available quotes. Pending-entry execution is a different stochastic process.

The current entry offsets are predetermined close-based proxies, not observed executable prices. Keep the old task frozen; define a separately versioned quote-trigger or fill task before retraining on different semantics. No future gap may turn an incompletely observed request into a confirmed neither-hit label.

Audit barrier distance in percentage, volatility units, spread multiples and effective tick increments. The current sampler covers 128 geometries but permits distances that may be below an instrument's price lattice. Round only according to the chosen request/venue contract and effective-dated metadata. Evaluate both broad research coverage and the intended production query distribution. Adding yet more correlated geometries is secondary to this coverage audit.

Choose training requests without looking at their future outcome. Sampling barriers from future extrema changes the question distribution and can produce favorable selection. A hard-query proposal based only on training data can be tested with explicit weighting; test requests must stay fixed.

## Architecture proposal

All adaptations in this section are hypotheses for our task. The target design is:

`causal normalized candles + activity/missingness -> local M1 residual + multiscale history encoder -> optional synchronized market-context gate -> continuous barrier-relative attention -> competing-risk time head -> coherent calibration`

The five public horizons remain unchanged.

1. **Expose physical time.** Add token age and duration, not just learned position and scale IDs. Overlapping M1/M15/H1 tokens refer to different spans. A 30-minute-old token should expose that age regardless of scale. This is a cheap isolated ablation.
2. **Make retrieval level-aware.** Supply signed distances from each requested upper/lower level to each historical token's high, low and close as an attention bias in the same normalized coordinates. Querying a level near a repeatedly visited region can then retrieve that region directly. These are continuous price relationships, not hand-labeled support/resistance patterns. Preserve upper/lower geometry separately from long/short TP/SL names.
3. **Preserve recent detail.** Add a residual temporal convolution over the last 30 or 60 M1 bars, summarized into a few tokens or a query residual. Test whether current six-minute patches compress relevant approach speed and wick structure. [ModernTCN](https://github.com/luodhhh/ModernTCN) motivates a convolutional alternative; its benchmark results are not first-touch evidence.
4. **Learn scale selection.** Test explicit cross-scale mixing or a dense gate before a full mixture-of-experts system. [TimeMixer](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html) provides a multiscale forecasting precedent. Its decomposition assumptions still require testing on our data. A week, month and several months can be tested at controlled resolution; an experiment should vary length separately from aggregation.
5. **Add simultaneous market context.** Start with synchronized BTC/ETH summaries or a fixed basket, gated into the query state. The current symbol embedding is identity, not cross-asset information. Use lagged as-of joins and explicit observation age; no nearest future match.

Attention is already capable of representing shape relationships, but that does not show it has learned named technical patterns. Examine paired out-of-sample performance with local detail, long context and market context removed. Attention pictures and chosen chart examples are supporting diagnostics, not proof.

The [architecture appendix](2026-10-23-barrier-architecture-primary.md) examines PatchTST, TimeMixer, ModernTCN, Mamba, TS2Vec, MOMENT, Chronos-2, TFT, DeepHit and sparse experts. At 352 history tokens, changing only the encoder to Mamba does not establish a speed advantage: the 640-by-352 decoder attention remains. Sparse MoE is not an evidence-backed first step at the current scale. Bigger dense capacity remains a legitimate control, rather than being ruled out by the current early stopping.

## Pretraining to learn morphology without named patterns

[PatchTST](https://arxiv.org/abs/2211.14730), [TS2Vec](https://arxiv.org/abs/2106.10466) and [MOMENT](https://arxiv.org/html/2402.03885v3) support representation learning from historical time series. Their results concern forecasting/representation tasks, not this financial event target.

Pretrain the compact encoder on diverse training-period histories using masked returns/ranges/volume reconstruction, then fine-tune it on our event likelihood. Compare reconstruction with a controlled contrastive objective. Mask the same physical historical interval at every overlapping scale, otherwise M15/H1 prices reveal hidden M1 targets. Do not use validation/test history for a strict prospective pretraining experiment merely because it is unlabeled.

Compare supervised-only, pretrained-and-frozen, and pretrained-then-fine-tuned at the same encoder size. Count pretraining compute in the budget. A better reconstruction loss is not a downstream win. Preserve signed movement and chronology; arbitrary temporal shuffles, OHLC jitter, MixUp or time warps can invalidate first-touch labels. Historical masking/cropping for a representation objective is different from changing a supervised path while retaining its original event label.

## Event head, constraints and a stronger research branch

Retain proper censored NLL as the anchor. [Rindt et al.](https://proceedings.mlr.press/v151/rindt22a.html) establish propriety of right-censored log likelihood and identify problems with certain censored-survival scoring formulations. This does not mean ordinary fully observed multiclass Brier or log loss is improper. Censoring/observation assumptions still matter when outages depend on market stress.

Diagnose chance of any hit separately from which side hits first. A factorization into touch intensity and cause conditional on a hit can expose separate training/metric failures. A three-class softmax can already be factorized this way, so changing algebra alone adds no information or expressive power. Conditional directional log loss among known hits must accompany the full three-class scores; compare on the same geometries and activity regimes.

Test finer internal time resolution, for example 24 or 48 bins with all five requested endpoints retained, or a piecewise-constant continuous-time hazard head. Current widths are 5, 10, 45, 180 and 1200 minutes. They are valid but coarse. [Kvamme and Borgan](https://arxiv.org/abs/1910.06724) investigate discretization/interpolation and hazard parameterizations in survival analysis. Their results do not imply that more bins always help. Store event minute and observed exposure before this experiment; preserve unknown cause and censoring. Encode history and barrier context once, then apply a small time-conditioned head instead of multiplying expensive cross-attention by every new bin.

Check these geometric properties on cumulative incidence, holding entry and the other physical barrier fixed:

* Moving an upper TP farther upward cannot increase upper-first probability by a fixed deadline.
* Moving the lower SL farther downward cannot decrease upper-first probability.
* Expanding both sides cannot decrease neither-hit probability.
* Swapping TP/SL labels for the same physical upper/lower interval must swap the cause probabilities. Reusing unequal unsigned long/short distances does not describe the same interval.

These follow from inclusion of path events. The present head ensures time coherence, not these cross-request properties. Paired consistency penalties are a research option. Naively constraining individual hazards does not automatically enforce all cumulative-incidence inequalities.

A more ambitious branch is a **shared conditional distribution over complete price paths**, evaluated against all requested barriers. Each sampled ordered path supplies one coherent outcome for every deadline and geometry. Using the same paths for all requests gives empirical cross-barrier consistency and TP/SL relabeling symmetry, subject to defined jump/tie conventions. A calibrated path generator can capture a joint future structure that marginal endpoint quantiles cannot identify.

[TimeGrad](https://proceedings.mlr.press/v139/rasul21a.html) demonstrates autoregressive diffusion for multivariate probabilistic forecasting, not trading or calibrated first passage. A finance adaptation must handle inactivity, jumps, volatility, tick lattice and correct price/quote semantics. Autoregressive diffusion over a full day can be expensive, and small probabilities require enough Monte Carlo samples. This is a second research branch, not an assumed upgrade. Begin with volatility-conditioned simulated or block-bootstrapped return paths as a transparent comparator. A driftless continuous log-price Brownian baseline has eventual upper-first probability `lower_distance / (upper_distance + lower_distance)` for a start inside fixed barriers; finite-deadline survival also depends on time and volatility. These assumptions are intentionally simpler than this market and should be stated.

## Training, calibration and uncertainty

Use calendar/asset-aware minibatch ordering to reduce nearly duplicate adjacent origins in one batch while still visiting every mandated origin and all 128 requests per full epoch. Test warmup/decay, stronger regularization, independent Transformer-layer initialization and EMA/SWA as isolated controls. The existing small LR continuation did not solve selection deterioration. More epochs do not create independent history.

If activity/regime balancing changes sampling, restore the intended request distribution with appropriate weighting or report the changed target. Uncorrected class weights or focal losses are not an automatic fix for a probability-estimation task. At five minutes neither-hit is roughly 87% to 88% in the completed test folds; raw accuracy can be high for an unhelpful always-neither classifier.

Temperature scaling is already present. It slightly worsened early-fold path NLL and improved late-fold NLL slightly. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) motivate this simple method on classification benchmarks, but do not guarantee improvement here. A regularized time/cause-aware calibrator is a candidate after obtaining more independent calibration dates. [Dirichlet calibration](https://arxiv.org/abs/1910.12656) supplies a richer multiclass alternative. Apply an appropriate map to interval hazards and rebuild cumulative incidence so time coherence survives; fitting five independent cumulative probability maps can violate it.

The present three-seed ensemble already exists. A later heterogeneous ensemble of genuinely complementary encoders, a simple query/control model and a path model is a different hypothesis. Learn any combination on past out-of-fold predictions, never test outcomes. [Ovadia et al.](https://arxiv.org/abs/1906.02530) find limits to post-hoc calibration under dataset shift and support investigating model averaging. Disagreement is a diagnostic, not a guaranteed confidence interval or a fourth first-touch outcome.

## Foundation models after the frozen comparison

Finish the matched-data Kronos/Chronos/TCN benchmark first. Frozen representations plus an event head, native model fine-tuning and native zero-shot forecasting are different experiments.

The [Chronos-2 report](https://arxiv.org/abs/2510.15821), [model card](https://huggingface.co/amazon/chronos-2) and [official pipeline source](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py) support a future adapter experiment. The model has 120M parameters; its stated maximum context is 8192, shorter than 10080 M1 points. Current source implements full and LoRA fine-tuning, but the native fit objective is forecasting. Training an event head with our partial-label NLL requires explicit integration. Missing optional adapter dependencies can change the actual fine-tuning mode; verify trainable parameters and pin source/checkpoint identities.

Compare frozen encoder/event head with LoRA/event head and budget-matched training of our encoder. Preserve sequence information rather than assuming one pooled embedding is sufficient. Quantile forecasts at future endpoints do not supply a joint path or barrier order. Published general forecasting superiority cannot decide our ranking.

## Experiments and acceptance rules

These are proposals, not new completed improvements. Each row changes one mechanism before combining winners.

| Order | Experiment | Primary question | Controls |
| --- | --- | --- | --- |
| 0 | Trade/quote/bar reconciliation and activity/tick audit | What price process do the labels actually measure? | Same venue, immutable sample receipts |
| 1 | Activity, missingness, quote-age and tick-aware features | Is the model missing whether the observed market is active? | Same old labels/cohort for an initial feature ablation |
| 2 | Token age/duration, then barrier-relative attention | Can level-specific history retrieval help? | Same loss, context, queries, seeds |
| 3 | Short M1 residual; length/resolution controlled sweep | Is relevant short morphology being compressed? | Separate length from tokenization and capacity |
| 4 | Diverse years/assets and synchronized BTC/ETH | Does independent context improve temporal transfer? | Fixed as-of joins; per-venue tasks; same final holdout |
| 5 | Compact encoder pretraining | Do unlabeled historical patterns transfer to first passage? | Train-period only; include pretraining budget |
| 6 | Finer hazard times and geometric consistency | Does event resolution/structure improve coherent probabilities? | Five public deadlines, exact censoring and event labels |
| 7 | Native foundation adaptation and larger dense controls | Is pretrained capacity useful beyond a frozen head? | Same supervised data, objective, seeds; disclose base pretraining |
| 8 | Shared path-model comparator and complementary ensemble | Can structural path consistency add predictive value? | Same paths for all geometries; Monte Carlo uncertainty |

Run six to twelve nonoverlapping walk-forward test windows where the expanded data permit, with selection and calibration on preceding dates. These counts are proposed ranges, not evidence that twelve windows are sufficient. Use a full 24h target purge at split boundaries. A week of overlapping historical context alone does not require discarding a week between roles; future label overlap is the critical leakage mechanism.

Use common eligible origins for direct candidate comparisons. A missingness-aware model may separately report improved deployment coverage, but must not get credit on an easier or different test subset. Preserve wall-clock timing and do not blindly interpolate future label gaps. Report performance by asset, horizon, activity, volatility, direction, spread/tick distance and regime, including both balanced research queries and production-weighted queries.

Measure raw/calibrated proper partial-label NLL, horizon multiclass log loss/Brier, reliability with support counts, conditional cause loss, time coherence and cross-barrier violations. Resample UTC-day or week blocks containing all associated assets/origins/queries; 53 million requests are not 53 million IID observations. Few days mean uncertain comparisons. Score missingness/outage sensitivity and hold out assets/venues where sufficient data exist.

The old test results have now informed development proposals. Keep their artifacts frozen and describe future reuse as historical research. Reserve a newly untouched future period for final confirmation. [Bailey et al.](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) discuss multiple-testing/backtest overfitting. This motivates trial accounting and fresh confirmation; it is not a reason to replace causal walk-forward evaluation with training on future folds.

An improvement must transfer across multiple held-out windows, preserve probability and target semantics, and avoid a material degradation hidden by the global average. Record total extraction/pretraining/training/inference cost and eligible coverage. No expected percentage improvement is claimed before these experiments.

## Companion evidence

* [Architecture primary-source investigation](2026-10-23-barrier-architecture-primary.md)
* [Data, labeling and official acquisition investigation](2026-10-23-barrier-data-primary.md)
* [Earlier loss/overfitting investigation and executed LR probe](barrier-loss-primary-sources.md)

The data appendix's runnable pandas check reproduced the activity counts. Source-manifest rows sum to 2,100,229, query count is 414289 times 128, and cohort exclusion is computed from 358517 of 414289. No additional GPU run or archive acquisition was needed for these research findings.
