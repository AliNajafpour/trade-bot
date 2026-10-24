# Orchestrator review

What this change does: The orchestrator scans completed candles and fresh quotes, requests specialist analysis once per setup, and persists conditional plans. The paper broker handles spot entries and exits. The review covers the repaired scanner, specialists, sizing, paper ledger, orchestrator and CLI.

## Findings repaired during review

1. **A crash leaves the orchestrator permanently locked.** P1, initial `wallex_lab/orchestrator.py:76-84`.

   The process creates `orchestrator.lock` with `O_EXCL` and removes it only in `finally`. Killing the process or losing power during analysis or an account write skips `finally`. The next process fails with `FileExistsError` before it can monitor or exit existing positions. Use an operating-system lock released on process death, or verify and recover a stale owner safely. A normal Python exception already removes the file; the missing case is process death.

   Closed. `wallex_lab/locking.py` now uses kernel-managed locks, shared by orchestrator construction, ticks and broker writes. `checks.check_locking` passed contention and abrupt-owner-exit cases.

2. **The specialist rejected every orchestrator symbol.** P1, original specialist input validation and orchestrator analyzer call.

   The orchestrator supplied `BTC/TMN`, while the initial specialist accepted only compact identifiers such as `BTCTMN`. Its news filter also parsed the compact form. This caused analysis failure before inference.

   Closed. The specialist normalizes supported compact or slash pairs and parses the base and quote from that representation. An independent provider-mocked run through the real scanner, specialist, risk and broker boundaries accepted `BTC/TMN` and created the paper entry.

3. **The risk check rejected the execution backend as a nonspot market.** P1, initial `spot.size_order` call from the orchestrator.

   The sizing helper checks `mode='spot'`, while the orchestrator supplied `mode='paper'` or `mode='replay'`. Every proposed entry therefore failed the spot-only admission check.

   Closed. The orchestrator now passes a separate sizing configuration with `mode='spot'` and the current `asof`. Its persisted execution backend remains paper or replay. The full conditional-entry integration and independent real-boundary check pass.

## Verification

`checks.check_fast_market` passed. The scanner removes unfinished bars before reading their OHLCV values, bounds its scan to 100 completed candles, checks quote availability against the decision time and resets after gaps. Its setup fingerprint does not change when the same quote changes slightly. Its check includes an invalid unfinished candle and future quote rejection.

The exclusion file contains 345 unique exact pairs. Every row matches `BASE/TMN` or `BASE/USDT`. The reviewer has not independently matched those rows against the original human-supplied list; that list is not in this delegated task context.

The real policy loader accepted all 345 rows, and every loaded pair failed `market_allowed`. Broker tests verify exact pair matching, reject malformed and leveraged identifiers, and distinguish TMN from USDT balances.

All four architecture checks passed independently on the repaired source: `checks.check_fast_market`, `checks.check_specialists`, `checks.check_spot` and `checks.check_orchestrator`. They cover causal numeric packets, bounded parallel specialist responses, missing-news uncertainty, typed schema failures, cash/fee/depth admission, persistent idempotent fills, restart, WATCH confirmation and expiry, and exits after the inference budget is exhausted.

The reviewer also ran a short end-to-end check using the real scanner and real specialist packet/schema construction, with only its inference provider replaced. The summary contained native finite numeric values, the slash market identifier crossed every boundary, and the real risk and broker modules admitted the entry. This check addresses integration mistakes hidden by a replacement analyzer or scanner.

An injected interruption after the broker committed the buy and before the orchestrator saved plan removal left one fill and a persisted stale plan. Restart recovered the account, discarded that stale plan, then exited at a fresh target quote. The ledger contained exactly two fills and the next tick honored cooldown. Broker properties reload persisted account state, so an existing instance observes another instance's writes. Kernel locks prevent concurrent ledger mutation, and atomic replacement preserves either the old or committed account state.

Entry requires a quote observed after analysis. WATCH confirmation requires a later completed close, positive volume, a quote available at or after that close and recent valid candle data. A post-arming candle gap cancels the plan. Expiry precedes confirmation. Open-position monitoring and stop, target or time exits precede the new-analysis budget gate.

The position size uses same-quote cash and equity, a bounded position fraction and a stop/cost risk budget. Unknown news reduces risk, high event risk blocks entry, buys include fees, and sells cannot exceed owned inventory. Depth is enforced when the snapshot supplies it. The broker applies the observed ask to buys and bid to sells, plus fees. The extra configured slippage enters conservative sizing geometry; it is not a separately charged broker fill adjustment.

No unresolved P0 or P1 finding blocks this paper mechanism. No additional concrete P2 finding was confirmed in the scoped review.

Not checked: real specialist responses, independent transcription against the original user market list, exchange order routing, venue precision/minima, actual queue fills or profitability. The implementation contains no exchange order backend. A missing fresh quote defers exits, and the process must keep polling to observe triggers; a protection expiry is a planned maximum rather than a guaranteed fill time. Replay explicitly omits real inference latency. No network or model calls were made, and this review changed only this document.

Verdict: ship the repaired paper orchestrator with those execution limits and parent-owned production evidence.

## Learned opportunity discovery re-review

The default scanner is now `AIMarket`, constructed by `Orchestrator` when `technical_models=True`. The CLI uses this default and exposes the old scanner only through the explicit `--rules-baseline` comparison option. A model failure cannot silently select that baseline.

`AIMarket` reuses `fast_market` for completed-candle features and quote/freshness admission. An old indicator template is not required. Both an existing numeric setup and the old scanner's `latest completed candle has no eligible setup` outcome proceed to learned inference. Missing history, stale/future quotes, wide spread and other independent admission failures remain blocked. The model check constructs an opportunity from the old no-template outcome and verifies that negative forecasts or missing models remove it.

The discovery vote requires all three aligned forecasts, with at least two exceeding the current configured roundtrip fee/slippage hurdle. The median must also exceed that hurdle. Stops, targets and payoff geometry remain separate deterministic risk controls, and do not create a positive forecast. The specialist receives the learned forecasts in its numeric summary. There is no forced-trade quota.

`ForecastBank` performs current inference from actual saved HGB and GRU checkpoints. Kronos uses its frozen pretrained encoder, followed by the locally fitted scaler/PCA/Ridge return head. This is learned hidden-state inference with a local return head, rather than a native autoregressive Kronos price forecast. The bank checks original data identity, checkpoint hashes, feature/source hashes, common three-bar horizon, native cadence and the frozen selection boundary. It requires 64 contiguous completed candles and rejects unavailable required models. It caches by actual OHLCV, timestamps, cadence and checkpoint identities rather than replaying saved forecast scores.

The forecast unit conversions match the fitting code. HGB was fitted to net fractional returns, so the bank adds its original training cost to recover gross return. GRU produces gross basis points and divides by 10,000. The Kronos Ridge head predicts gross fractional returns directly. `AIMarket` then subtracts the current configured cost from each gross forecast, avoiding a second charge of the old training hurdle.

Both new checks passed independently: `checks.check_ai_market` and `checks.check_model_signals`. The first uses an explicit fake bank for the discovery contract. The second fits tiny CPU HGB/GRU fixtures, reloads their real saved artifacts and verifies prediction equality, gross units, hashes, pre-freeze rejection, cadence/symbol admission, gap warmup and input-sensitive caching. No production encoder inference, GPU work, network access or model CLI calls were launched by this review.

The reviewer read the parent's saved `runs/orchestrator-model-bank/fresh-inference.json`. It contains successful three-model forecasts for four markets. Those records explicitly flag `retrospective_before_training_wallclock=true`; they must be described as retrospective inference rather than forecasts available at the recorded historical market time. Their negative net estimates honestly support no current opportunity at the configured costs. The reviewer did not independently rerun that production inference or its GPU benchmark.

No new concrete P0, P1 or P2 finding was confirmed in this addition. Learned discovery is now distinct from independent cash, quote and risk admission. The estimates remain uncalibrated close-return forecasts; they do not establish profitable fills or future profitability.
