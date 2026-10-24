# Financial-model spot orchestrator: verified results

Default discovery now uses fresh HGB, GRU and genuine Kronos financial-encoder forecasts. A model majority must clear current trading costs, then independent risk geometry, before waking LLM specialists. Indicator templates remain an explicit comparison baseline.

| Market | HGB gross forecast | GRU gross forecast | Kronos gross forecast | Inference seconds |
|---|---:|---:|---:|---:|
| BTCTMN | 0.01702% | 0.03259% | 0.00357% | 11.465 |
| ETHTMN | -0.01053% | 0.00431% | 0.14383% | 0.388 |
| SOLTMN | -0.05256% | 0.12450% | -0.05667% | 0.401 |
| USDTTMN | 0.07130% | 0.01385% | 0.08796% | 0.363 |

All forecasts cover three completed M15 bars. First inference includes cold imports, weight verification and GPU loading; later markets reuse the encoder. Changed-input warm inference took 0.176s. Identical-candle caching was also verified.

In the recorded ETH/SOL feed, current assumed costs were 0.7% roundtrip. None of the three forecasts cleared costs. The default model pipeline therefore performed zero LLM evaluations and zero paper fills. This is an observed rejection, not a profitability result.

Before the model-discovery correction, the indicator baseline woke both specialists on ETH and SOL. Four actual tool-free Codex calls succeeded. Numeric regime packets were 996/999 characters; news packets were 1455/828 characters. Calls ran in parallel within each opportunity. ETH received a 30-minute conditional plan, SOL a 60-minute conditional plan; both awaited a post-analysis quote. The recorded feed supplied no such new book, so no fill was invented.

Verification passed all 23 offline checks, all four raw response/file-hash receipts, all twelve fresh financial forecasts, current source/checkpoint hashes, separate quote balances and all 345 exclusion pairs. The integration checks include conditional activation, expiry, budget-independent exits, restart and an interruption between account commit and orchestrator state save. Synthetic check fills are separate from recorded-market evidence.

The three additional Kronos heads were fitted on the causal TRAIN prefix with later validation selection. HGB/GRU checkpoints were reused. Historical model and LLM evaluation is retrospective; model training completed after the replay observation times. Independent forward testing is still needed.

No Wallex order API/MCP is connected. Paper polling accepts a supplied snapshot every five seconds; it does not manufacture fresh API observations. Missing news is unknown and reduces risk. Missing books block entry and defer exits. Actual model identity served by Codex is unverified. Broker fills use observed bid/ask plus fee; configured extra slippage only enters conservative risk sizing.

Code: `wallex_lab/{ai_market,model_signals,fast_market,specialists,spot,orchestrator}.py`. Run: `python -B scripts/run_orchestrator.py --events runs/orchestrator-input/events.jsonl --output runs/new-ai-experiment --model-device cuda`. Reverify this report without inference: `python -B scripts/report_orchestrator.py`.

Primary model source: [Kronos](https://github.com/shiyu-coder/Kronos). Full evidence and remaining limits are recorded in `reports/orchestrator-results.json` and `docs/orchestrator-review.md`. The failed restricted-network smoke remains preserved under `runs/orchestrator-smoke-20261021`; successful specialist evidence uses `runs/orchestrator-smoke-network`.
