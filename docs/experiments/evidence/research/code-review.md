# Independent implementation review

Reviewed the implementation plan, all `wallex_lab` adapters and offline checks, the relevant vendored Kronos inference paths, and the acquisition/context validation receipts. This is an implementation review, not a profitability endorsement. No network requests, GPU work or model fitting were run for this review.

No P0 issue found. The original P1 gap lookahead has been repaired during review. Remaining findings below describe the source and artifacts observed before final reruns; the parent owns repairs and final evidence.

At handoff, the parent confirmed final results will use `runs/wallex-20261021-causal` in fresh Python processes, preserving earlier outputs as pre-fix evidence. Baseline recomputation, expert integrity checks and the prospective-context guard are being repaired. The findings retain their original evidence below; those repairs and regenerated results were not independently verified in this pass.

## P1: rebuild persisted comparisons after changing execution

`wallex_lab/runner.py:82-85` creates baselines only when `baselines.json` does not exist. Therefore rerunning every strategy after an engine repair still compares them with the old execution model. This is present in the actual run: `runs/wallex-20261021/BTCTMN/baselines.json` still records "Gap-boundary close liquidation is offline censoring using the future missing interval" after the causal engine repair.

Smallest fix: recompute the two cheap baselines on every run, or reject their reuse when the engine hash changes. Rebuild all final primary/stress/no-cost metrics in a fresh Python process. Re-run tree and Kronos validation selection as well, because `trees.py:139` and `kronos.py:199` select settings using the execution engine. Restarting matters: an already-running process retains the previous imported engine, while `runner.py:128` hashes the current file at completion and can attach a new source hash to old execution behavior.

## P1: prospective context can be attached to historical allocations

`wallex_lab/agent.py:96-103` filters context using the current inference time whenever the caller sets `prospective=True`. It never checks the decision candle against that time. A caller can therefore pass January candles, October news received before inference, and a start index with historical outcomes remaining; the returned allocation then executes in January through the ordinary runner. `historical_knowledge_contamination_possible` also becomes false merely because of this flag.

The default historical path safely omits context, and `context_asof` correctly filters individual items. The missing guard is at the caller boundary. Smallest fix: allow prospective inference only at the terminal observation and prevent its allocations from entering historical-return scoring. Alternatively, filter context at each historical candle close and retain the historical-contamination flag. Add one mock-only check where context arrives after a historical decision but before current inference; it must be rejected or omitted. Checking only that a headline is later than wall-clock now does not cover this failure.

## P2: resumed expert artifacts are not verified before fusion

`wallex_lab/runner.py:47-54` checks only an expert's status, then loads its NPZ and trusts the saved metadata. Stored `outputs_sha256` and `data_sha256` are never checked. `runner.py:77` also ignores changes to `extra`, although that contains model, call-budget and training settings. Same-length stale or modified allocations can enter fusion while retaining another run's provider and inference receipts. This is an integrity/reproducibility problem, not evidence that the current files were tampered with.

Smallest fix: pass the current data hash to `load_expert`, verify both hashes and relevant split boundaries, and reject changed inference settings when resuming. Treat integrity mismatches as explicit failures rather than silently omitting that expert. A small NPZ-tampering check is sufficient.

## P2: minimum-size enforcement excludes all forced exits

`wallex_lab/engine.py:72` enforces minimum notional only for `rebalance`. End liquidation, gap exits and risk-stop exits can sell below the configured minimum. Quantities also remain continuous and ignore exchange quantity steps. The actual public metadata in `data/acquisition_validation.json` records 50,000 toman minimum notional for these markets and different quantity steps. Thus a final all-cash value can depend on fills the venue would reject, especially after a losing position shrinks below the minimum.

For the current candle experiment, explicitly describe forced subminimum liquidation as a valuation assumption in every result and count affected exits. Do not call the resulting cash value account-executable. For an order-capable implementation, validate the account's actual market filters, round quantities downward to valid steps, retain unsellable residual inventory, and value it separately. Public market filters do not verify hackathon credit-account rules or the allowed-market list.

## Resolved P1: future gap liquidation

The reviewed initial engine used `gaps[i + 1]` to sell at candle `i` close, avoiding the subsequent unobserved price jump with knowledge unavailable at that close. This cannot be repaired by a caveat alone.

The current repair in `wallex_lab/engine.py:121-128` exits held inventory at the first observed post-gap open, cancels the stale request and preserves the gap price change and fees. The previous-close boundary lookup is gone. Independently ran:

```text
python -m checks.check_engine
engine: next-open, each-side costs, exposure cap, cash test reset, gap liquidation, roundtrips and validation pass
```

The updated check covers both a gap gain and a gap loss, costs, a pending stop at a gap, and a test account that starts flat at the gap. This resolves the execution-code defect. Previously saved metrics remain invalid until regenerated as described above.

## Findings that did not warrant changes

- Ordinary allocations use a completed close and execute at the next contiguous open. The cash ledger debits each-side fees and slippage, caps exposure at execution, proportionally allocates entry cost on partial exits and starts every test account with fresh cash. The lightweight engine check passed these paths.
- The runner separates base selection at 60-70%, fusion selection at 70-80%, and test at 80-100%. Tree labels and preprocessing are fitted on purged TRAIN; neural early stopping and architecture selection remain inside TRAIN. Kronos normalizes completed windows, fits its scaler/PCA/head on purged TRAIN, verifies pinned local assets, and exposes unsampled scores as missing. No direct test-label fit was found.
- Kronos metadata omits `train_end`/`val_end`, so fusion's `selection_blocks_disjoint` can be false despite the runner using disjoint blocks. Add those two metadata fields when touching that module; this is a reporting defect, not observed test leakage.
- Fusion consumes expert allocations, includes a real-agent provenance requirement and holds selected weights fixed through TEST. Its arithmetic validation proxy is disclosed. The caller still needs artifact verification as above.
- Historical agent mode does not include current market context. News availability uses local receipt time; individual future observations are filtered. Current book depth is not presented as historical fills, and unsigned trade volume is not mislabeled as aggressor flow.
- The acquisition validation records 352 raw receipts and reconstructs complete local 15-minute buckets without filling missing OHLC. The four markets retain explicit gaps. The context validation records 11 verified body hashes and zero observations available at its historical cutoff. The Wallex documentation TLS failure remains recorded rather than bypassed.
- Actual completed tree and neural outputs observed during review had zero fills and zero return. That is a legitimate abstaining strategy outcome. It must be reported as no trading evidence and compared with the cash baseline, not rewritten into a synthetic active strategy or described as proof of profitability. Offline mocked LLM and tiny random Kronos checks are interface checks, not production inference evidence.

This review did not rerun expensive training, verify remote model identity, audit external model pretraining data, or establish live fill quality. Parent-run inference receipts and final regenerated artifacts must supply the corresponding evidence.

## Final corrective verification

The findings above remain as the original audit trail. This section records a second, independent inspection of the repaired source and final artifacts. No source code was changed during either review.

The final roots are `runs/wallex-20261021-causal` and `runs/wallex-20261021-horizon24`. Independently loaded all 30 successful results, their canonical-data identities and NPZ outputs. Verified every recorded top-level source hash against the current files, the data hashes against the canonical CSVs, execution configuration against each market identity, and the expert output hash through the repaired `load_expert` boundary. No final result uses the obsolete engine source.

Independently recomputed every primary, stress and no-cost ledger from the stored targets in a fresh Python process, without training, model inference or network calls. All 90 complete results matched exactly, including fills, equity, risk events and metrics. Independently recomputed the cash and constant-allocation baselines for all eight market/run identities. All 16 baseline accounts matched exactly and every baseline records the current engine hash. These checks resolve the stale-engine and stale-baseline P1 findings for the delivered artifacts. The separate parent-run `reports/replay-verification.json` also records 30 successful three-scenario replays and fitted-model reload checks. This review inspected that verifier's implementation but did not repeat model inference.

The current engine uses only the observed post-gap open for gap liquidation. Final forced exits expose `forced_subminimum_exit_count`, and results explicitly disclose both unrounded fractional quantities and forced exits below venue minima. The primary ETH rules run has two such exits; the other 29 primary strategy results have zero. The execution-realism P2 therefore remains a measured, disclosed limitation of the candle experiment. The implementation does not claim these final cash values are account-executable.

The prospective-context P1 is resolved for the supported interfaces. `agent.generate` restricts prospective mode to one terminal decision, marks it `nonbacktestable`, and the runner rejects such output before historical scoring. `collect_current` returns only a decision and metadata. The actual `runs/prospective/BTCTMN/current.json` contains no target, has `nonbacktestable=true` and `market_context_used=true`, and its context hash matches the saved input. Historical agent packets inspected from the actual inference receipts contain no market context.

The historical BTC agent archive contains 10 successful receipts and one preserved error receipt. Independently verified each receipt's saved file hashes, the current LLM adapter hash, successful response schemas and successful event streams with zero tools. The bounded retry preserves the initial failure and successful retry separately. This verifies actual CLI decision artifacts, not remote model identity or freedom from pretrained historical knowledge.

BTC fusion includes all five expected experts. `selection_blocks_disjoint=true`, with every numeric expert selection ending at index 4025 and the fusion freeze at index 4600. Its independently replayed primary return is **-0.8142766454%**. Kronos now records both split boundaries, resolving the earlier reporting defect. Abstaining tree/neural outcomes remain valid zero-trade results and provide no evidence of active trading profitability.

`reports/checks.json` records 12 passing checks and zero failures. Independently matched all 12 stored check hashes to the current check files. The runner regression now rejects an appended-byte NPZ modification, and the agent checks cover prospective historical-index rejection, no-target current collection, strict schema failures and bounded transient retries.

No unresolved P0 or P1 finding blocks these final research artifacts. Resume integrity is still narrower than a complete run specification: `identity.json` does not reject changed `extra` settings, `load_expert` checks selected direct source files rather than every transitive dependency, and the fusion caller may omit an invalid optional expert after a caught validation error. All final expert artifacts inspected here were valid, and the delivered BTC mixture includes every expected expert. Future experiments should use fresh run directories when inference settings or supporting modules change and inspect the recorded expert membership.

Remaining limits are historical single-holdout exposure, possible pretrained contamination, horizon-24 exploration on the same exposed test period, assumed fees/slippage, missing historical order books, unobserved intrabar/gap risk, and venue/account rules not established by public metadata. The prospective decision and short paper-book smoke test do not establish a profitable live strategy or three-day performance. No real exchange orders were part of this review.
