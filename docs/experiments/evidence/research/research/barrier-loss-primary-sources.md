# Barrier survival loss evidence

Reviewed 2026-10-22. Read-only code and run inspection; no training, model, data, protocol, checkpoint, or GPU changes.

## What 1.7163 means

`wallex_lab/barrier_survival.py:19` computes mean negative natural log likelihood per origin/query observation. Each of five time bins has conditional probabilities `q[j] = softmax(logits[j])` for TP first, SL first, or surviving that bin. Define `S[j] = product(q[i,2], i < j)`.

- Known first-touch cause `c` in bin `j`: likelihood `S[j] * q[j,c]`.
- Unknown cause in bin `j`: likelihood `S[j] * (q[j,0] + q[j,1])`.
- Censored after `m` completely observed bins: likelihood `S[m]`.

This matches the discrete survival likelihood principle: an event contributes survival through previous intervals times event probability; a censored observation contributes survival over observed intervals. The project extends this to two competing causes and summed ambiguous-cause mass. [Gensheimer and Narasimhan, equations 4-6](https://arxiv.org/html/1805.00917v3).

A completely resolved query has eleven mutually exclusive terminal outcomes: TP in one of five bins, SL in one of five bins, or no touch through 1440 minutes. A censored observation sums the terminal outcomes compatible with the observed survival. Unknown cause sums the two causes in its event bin. Censoring inside a bin retains only completely observed bins in this implementation.

For losses `L[i] = -log(p[i])`, `exp(-mean(L))` is exactly the geometric mean of the assigned observed-outcome likelihoods. Thus NLL `1.7162832021713257` gives `0.179732938977413`. That is not 17.97% accuracy, TP success probability, average TP prediction, or trading win rate. It is not an arithmetic mean probability either. The logarithmic score measures the probability assigned to what occurred, and lower is better. [Gneiting, Balabdaoui and Raftery, section 3.4](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf).

`log(3)` is not the correct generic baseline here: this objective scores a whole event-time path, not one three-class horizon outcome. Even `log(11)` describes only a hypothetical uniform distribution over eleven fully resolved outcomes. Censoring, unknown causes, unequal outcome frequencies, and query difficulty change useful reference scores. Compare models on the identical selection observations, especially the planned query-only baseline. The per-horizon three-class log losses in `probability_metrics` are different metrics.

## What the recorded deterioration establishes

The exact curve in `runs/barrier-survival-multiday-20261022/early-context10080-seed42/curve.json` is:

|Epoch|Online fit NLL|Selection NLL|
|---:|---:|---:|
|1|1.821449|1.743634|
|2|1.715956|1.716283|
|3|1.627450|1.755731|
|4|1.534339|1.851295|
|5|1.447464|2.016971|

Epoch 2 is the best observed selection checkpoint. Subsequent training improves the online fit objective while degrading held-out likelihood. This supports worsening generalization. It does not isolate memorization, overconfidence, optimizer behavior, or temporal distribution mismatch as the cause. Epoch 5 assigns a geometric mean observed likelihood of 0.13306 versus 0.17973 at epoch 2, about 26% lower. Best-checkpoint retention is appropriate; the official PyTorch training example also saves the state with lowest validation loss. [PyTorch training tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html).

The run protocol and all recorded rows use 128 queries each epoch. The trainer's 32-query rotation is irrelevant to this run. Checkpoint eligibility starts immediately for 128-query training. Patience is three, so epochs 3-5 exhaust patience after the epoch-2 best. The run preserves cloned CPU tensors for the best state, avoiding later optimizer updates overwriting it.

## Plausible causes and evidence still needed

- The model can fit correlated training origins and adjacent barrier queries without learning features that transfer to the next date block. Measure endpoint eval-mode training NLL and per-day, asset, geometry, event-bin, and cause validation losses on the same fixed sets. One aggregate curve cannot identify which observations deteriorate.
- AdamW uses fixed `lr=0.0006`, `weight_decay=0.03`; there is no scheduler or warmup. AdamW's adaptive moments and decoupled decay do not guarantee monotone validation improvement. A later, separately versioned lower-learning-rate or decay experiment could test optimizer sensitivity. This is a hypothesis, not evidence that the chosen rate is wrong. [Official AdamW definition and algorithm](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW).
- Temporal mismatch is plausible because selection follows fit chronologically. `fold_roles` purges a full 1440-minute target window before each boundary. Preserve temporal validation; random row splitting can train on future observations and score past ones. Compare feature and label distributions by date, and compare independent seeds and the second temporal fold before attributing the result to one context length. [Official TimeSeriesSplit guidance](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- The encoder is constructed by cloning one `TransformerEncoderLayer` without per-layer reinitialization. PyTorch warns that cloned encoder layers begin with identical parameter values and recommends manual initialization. This is a concrete initialization caveat for a future controlled run; it does not prove the cause of this curve. [Official TransformerEncoder warning](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html).

## Scoring and train/eval inspection

`fit_stream` calls `model.train()` at every epoch. `collect_stream` calls `.eval()` and uses `torch.inference_mode()`. Validation therefore disables dropout correctly; training resumes in training mode next epoch. PyTorch documents evaluation as `train(False)`. [Official Module.eval documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval).

The fit number is an online average across successive model states in shuffled batches, with dropout and BF16 forward computation when supported. Selection uses the epoch-end model with dropout disabled and no autocast. Consequently the logged fit-selection gap is not a clean same-checkpoint generalization-gap measurement. Dropout independently masks values during training and acts as identity at evaluation. [Official Dropout documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html).

Inspection found no obvious event-bin off-by-one error in `risk_nll`: `cumulative[j]` excludes event bin `j`, and censoring uses exactly `survived_bins`. Unknown cause correctly sums event mass in log space, and censoring is not relabeled SL. The model's unmasked self-attention only sees history ending at the decision boundary, so lack of an internal causal attention mask does not itself expose future candles for this endpoint forecasting task.

This was inspection, not an exhaustive executable correctness audit. CPU/FP32 toy likelihood enumeration, label-boundary checks, CPU/resident feature parity, and same-checkpoint fit/validation scoring remain useful checks. Do not infer genuine out-of-sample skill or calibration from 1.7163 alone. This review left the active GPU run untouched.

The inspected multiday source snapshots match their protocol SHA-256 hashes. The protocol records PyTorch `2.13.0+cu130`; current official web documentation resolved to 2.14. The cited train/eval, dropout, AdamW and encoder construction semantics should be checked against the installed version before any implementation change.

## Comparing the weekly model with Kronos

The closest executable comparison is **frozen Kronos transfer with a supervised barrier head**. It answers whether pretrained financial representations improve the same barrier task. It is a separate experiment from native zero-shot price-path forecasting.

The local `kronos.py` already verifies pinned small/base and tokenizer weights, normalizes each completed input window, extracts the final normalized `decode_s1` hidden state, and supports CPU inference. Reuse those functions. Its current PCA/Ridge head predicts returns and selects trading thresholds, so those existing return/backtest results cannot establish superiority on the current barrier likelihood. Official Kronos small/base have 512-bar context limits. The upstream `decode_s1` returns hidden representations explicitly. [Official Kronos model zoo and context limit](https://github.com/shiyu-coder/Kronos#-model-zoo), [pinned upstream representation API](https://raw.githubusercontent.com/shiyu-coder/Kronos/67b630e67f6a18c9e9be918d9b4337c960db1e9a/model/kronos.py).

Recommended controlled protocol:

1. Use the exact native M1 source hashes, origins, 128 arbitrary TP/SL queries, side and entry definitions, `BarrierLabelBook` labels, 5/15/60/240/1440-minute horizons, chronological folds, purges, and common weekly-history eligibility mask from the ongoing experiment. The dataset has 414,289 total indexed origins before the weekly mask. Never substitute legacy return labels or model-chosen barriers.
2. For every origin, derive three completed OHLCV views: last 512 M1 bars, last 512 M5 bars, and 336 M30 bars spanning the preceding week. Aggregate only past native minutes with first open, maximum high, minimum low, last close, and summed volume. End-align the groups at each decision to preserve the exact week mask. If reusing epoch-aligned `data.aggregate_complete`, adjust common eligibility for the extra alignment lookback and document the older final coarse bar. Never include an incomplete future-facing bucket.
3. Extract and cache one frozen Kronos-small representation per view/origin. Preserve per-window upstream normalization, real timestamps and approximate amount semantics. Concatenate representations and project to the existing width. Use the existing arbitrary query features, asset/bin embeddings, five conditional three-way outputs, `risk_nll`, and `incidence`. Train only the compact head on fit labels. Keep the pretrained encoder frozen. This is an adapted multiscale model, not an upstream architecture or a zero-shot result.
4. Use the same early-fold selection criterion, independent head seeds, separate temperature-calibration block, and test freeze. Report raw and calibrated partial-label NLL, per-horizon log loss/Brier/reliability, coverage, paired daily loss differences, extraction/training time, and trainable parameter counts. The current model and Kronos are identical in targets and scoring but differ in representation architecture and pretraining. Report that limitation instead of attributing all differences to pretraining alone.
5. Reserve the active GPU for its existing run. A small CPU timing pilot must precede runtime promises: roughly three encoder windows per eligible origin is substantial work even with cached frozen features. Small is the first candidate; base is a later explicitly budgeted comparison. Pin source and weight hashes and write a separate comparison protocol. Never silently thin the annual origin set. If a reduced experiment is needed, label it and rerun both models on its exact common evaluation set.

Native zero-shot Kronos belongs in a separate diagnostic. `DirectKronos` and `BatchedKronos` preserve real autoregressive samples, but their generated OHLC paths can be invalid or touch both barriers in one candle. The old `summarize_paths` chooses its own barriers and treats same-bar ambiguity as SL first, which differs from the current task. Retain user queries and unknown-cause handling. Report invalid mass and ambiguity; discarding invalid paths conditions the forecast on validity. Unknown first cause in a generated candle generally gives probability bounds until a cause-allocation assumption is declared.

The pinned upstream sampler averages `sample_count` paths before returning them and decodes only its final context window. Therefore independent paths require separate rows/calls with `sample_count=1`, as the current adapter does. The local wrappers cap predictions at 512 bars. Requesting 1440 M1 steps alone cannot return a full 1440-step path through this pinned routine. Coarse M5 paths cover 24 hours with 288 steps but lose minute first-touch ordering, so they are not identical-target evidence. A chunked full-minute sampler would need its own explicit implementation and validation. [Pinned upstream sampling and final decode, lines 350-419](https://raw.githubusercontent.com/shiyu-coder/Kronos/67b630e67f6a18c9e9be918d9b4337c960db1e9a/model/kronos.py).

For a similar-model comparison, frozen Chronos-2 embeddings plus the same supervised risk head are defensible after checking the installed API. The current official `Chronos2Pipeline.embed` supports multivariate history and returns patch embeddings; its forecast API returns quantiles. A fixed, declared embedding pooling rule can feed the same barrier head. The existing `chronos_market.py` returns marginal future CLOSE quantiles, which do not specify a joint path distribution or TP-before-SL probability. Connecting equal quantile levels across time or drawing independent quantiles would add an unvalidated dependence assumption. [Official Chronos-2 pipeline APIs](https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/src/chronos/chronos2/pipeline.py).

Keep original pinned pretrained checkpoints distinct from the project's previously fine-tuned checkpoints. Neither original model's full pretraining overlap with the annual Wallex history has been independently excluded; label historical comparison evidence accordingly. No foundation-model execution, training, download, or GPU access was performed for this methodological review.

## Executed concurrent CPU investigation

The separate `scripts/diagnose_barrier_training.py --lr-probe` run completed while the original GPU training continued. Artifacts are in `runs/barrier-diagnostics-20261022`; the original source protocol and numerical settings were unchanged. Dataset and common-origin-mask hashes were checked against the main run. Only early fit and selection roles were used. No calibration or test observations were scored here.

Fixed, stratified cohorts contain 1,024 fit origins across the training calendar and 512 selection origins across13 UTC dates. Every origin retains all128 queries. CPU FP32/eval-mode scoring uses frozen epoch2 checkpoints, so the fit/selection differences below do not mix dropout or changing model states.

|Model|Fixed fit NLL|Fixed selection NLL|
|---|---:|---:|
|Constant hazards, fitted to all fit labels|2.344113|2.338620|
|Query histogram hazards, fitted to all fit labels|1.835127|1.819479|
|Two-day encoder, epoch2|1.671595|1.724587|
|Weekly encoder, epoch2|1.676649|1.715410|

The histogram uses asset/direction, fit-only volatility tertiles, relative geometric barrier distance and barrier ratio bins, with low-exposure cells falling back to globally fitted hazards. It sees no candle sequence. It is a simple diagnostic baseline, not the stronger neural query-only comparison still queued in the main protocol. Beating this histogram does not isolate the value of candle history from smoother query modeling.

On the diagnostic cohort, fit median sigma is0.0010850 versus0.0007720 for selection, about29% lower. The fraction surviving24h without touching is9.58% versus12.33%. Fit contains1.27% origins with incomplete future observation, selection0%; neither sampled cohort contains completely uninformative censoring. These are descriptive differences, not proof of conditional distribution shift or the mechanism of deterioration.

The weekly-minus-two-day selection NLL difference is-0.009177. Descriptive paired UTC-day resampling gives95% bounds[-0.031083,+0.010265]. This is selection data used in checkpoint choice and is not independent test evidence or a model-selection-adjusted confidence statement.

### Controlled learning-rate continuation

Both branches start from exactly the same weekly epoch2 weights, seed904, optimizer type/decay,1024fit/512selection cohorts, batches32 and96additional updates. AdamW restarts in both branches; training uses CPU FP32. These conditions differ from continuing the original full-cohort optimizer, so this probe cannot establish the original causal mechanism.

|Additional updates|LR0.0006 selection NLL|LR0.0002 selection NLL|
|---:|---:|---:|
|0|1.715410|1.715410|
|32|1.755353|1.739326|
|64|1.770864|1.762849|
|96|1.741969|1.745497|

Final fit NLL decreased to1.588079 and1.598112 respectively. All recorded selection scores were worse than the starting checkpoint. Lower LR did not provide a sustained improvement in this bounded test; the main run was not modified on this evidence. The exploratory global-temperature grid0.5..8 chose1.0 for both frozen checkpoints and all probe endpoints. This is not independent calibration and does not exclude local/class-specific miscalibration.

The installed PyTorch2.13 TransformerEncoder docstring contains the same initialization warning. Direct inspection confirmed equal initial layer values but distinct storage; layers are not accidentally tied. A changed initialization would require its own future controlled experiment, not an attribution based on inspection.

The observed deterioration is real, and preserving epoch2 is justified by selection. Rapid fitting of correlated history, temporal mismatch and optimizer/regularization behavior remain plausible explanations. No single cause was established, and no semantic technical-pattern recognition or trading profitability is inferred. The final neural query-only baseline, independent seeds, later fold and calibration/test metrics remain pending in the active experiment.

Plots: `runs/barrier-diagnostics-20261022/learning-curves.png` and `lr-probe.png`. Rebuild with the adjacent `plot.py`. The diagnostic hazard-count and likelihood self-check covers exact events, ambiguous causes and censored observations with literal expected probabilities.

## Active Kronos comparison implementation

`scripts/compare_kronos_barriers.py` is now running on CPU with original pinned small/base weights, using existing encoder verification/loading/hidden-state helpers. Full raw labels and arbitrary query inputs are unchanged. No existing main training file or protocol was modified. The comparator is explicitly supervised transfer with a frozen encoder, not a native zero-shot score.

The1024fit/512selection origins are exactly the earlier diagnostic cohorts. Every origin retains128queries and every feature ends at that same five-minute decision. Derived M30 candles are rolling decision-aligned bars; a12:05 decision ends its last bar at12:05, not12:30. Head normalization fits only sampled fit features. Independent calibration/test evaluation is deliberately absent from this pilot.

On a real small-model CPU timing origin, extraction at512M1/512M5/336M30 took0.345/0.216/0.165seconds respectively, producing finite512-dimensional representations per branch. Receipts are in `runs/barrier-kronos-timing-20261022`. Base runtime has not yet been measured.

The active comparison directory is `runs/barrier-kronos-compare-20261022`; protocol and source snapshots are frozen, progress and provisional scores are written incrementally to`*-feature-progress.json` and`validation-pilot.json`. The same-head query-only control was trained to selected epoch24 with validation NLL1.811296; the full-fit weekly seed42 scores1.715410 on the same512selection origins. Kronos transfer scores are pending feature extraction and head training.

The pilot's weekly model used the full original fit cohort, whereas transfer heads use1024origins. This is a declared budget difference, so a final architectural ranking requires expanded and matched evaluation. Paired-day intervals are explicitly labeled as per-horizon three-class logloss differences, not the headline whole-path NLL. `checks.check_barrier_kronos` passes causal/gap/aggregation checks and an actual tiny head-fit, fit-only-scaler and saved-head inference roundtrip.

## Additional baseline comparison

The user requested further competitors. `scripts/compare_barrier_baselines.py` adds a bounded CPU pilot using the exact frozen Kronos fit/selection origins and all128 arbitrary queries. It uses the same source dataset, complete-week mask, chronological roles, censoring and ambiguous-cause labels. It never scores calibration or test periods and never changes the active GPU/Kronos sources.

Competitors are CatBoost without a candle sequence, CatBoost with normalized pooled candles, and the existing small risk-head MLP with exactly the same pooled candles. Histories contain24rolling M5 bars,48rolling H1 bars and28rolling H6 bars. Each block retains normalized open/high/low/close and mean normalized log-volume. This is a compressed raw-candle representation without named pattern labels or technical indicators, not the original transformer's full token stream.

CatBoost is already installed. Its binary Logloss fits probabilities. Two independently fitted binary models exactly factor the shared observed-path likelihood: `qTP=t*s`, `qSL=t*(1-s)`, `qNone=1-t`. Every completed at-risk bin trains the touch model; only known-cause events train the TP-given-touch model. Ambiguous events train touch only, and partially unobserved bins are absent. Thus no same-minute ambiguity is invented as a loss, and censoring is not labeled as no-touch through24h. Composing hazards through the existing incidence function guarantees probability sums and monotonic cumulative incidence. [CatBoost binary Logloss documentation](https://catboost.ai/docs/en/concepts/loss-functions-classification).

The two tree heads use CPU2threads, seed42, depth6, learning rate0.05, L2leaf5, up to400iterations and40iterations of validation patience, without class weights. The raw-candle MLP reuses the Kronos head fitter with80epochs and patience8. Tree and MLP fit/selection budgets match the transfer pilot; the weekly reference still has a much larger supervised training cohort. These remain uncalibrated selection comparisons rather than final test rankings.

`checks.check_barrier_baselines` first failed for the missing comparator, then passed exact factorized likelihood reconstruction including unknown cause and partial/zero observation, literal rolling OHLC pooling, cumulative probability checks and a real tiny CatBoost training/saved-model roundtrip. Run artifacts are written to`runs/barrier-baselines-20261022` with source and cohort hashes, incremental validation scores and predictions.

Chronos-2 remains a separately identified candidate. Its local installed `Chronos2Pipeline.embed` API and original pinned weights are present; no additional Chronos-2 barrier fit or score has been run by this baseline script. Marginal price quantiles remain insufficient for a first-touch comparison without a declared probability adapter.

## Chronos-2 and TCN extension

The user explicitly requested adding both candidates under the same comparison conditions. `scripts/compare_barrier_chronos_tcn.py` uses the exact frozen1024fit/512selection origins,128queries andfive horizons from the Kronos pilot. It checks dataset/cohort/mask hashes and chronological roles. No independent calibration/test score or modification to active experiment sources is performed.

Chronos-2 uses the original pinned checkpoint through `ChronosMarket(device='cpu')`, with frozen weights and three rolling histories identical in clock coverage to Kronos:512M1,512M5 and336M30. Price units are last-close100 and volume units use a past-only trailing reference before native normalization. Native `embed` treats each five-variable OHLCV window as a separate group. Four origins are batched as12distinct scale tasks, not one60-variable group. The installed encoder appends context patches, REG, then one masked output patch. The CLOSE REG vector at`[3,-2]` is retained per scale, producing2304features. Short scales may have23or34native embeddings because of batch left padding. [Official embedding API](https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/src/chronos/chronos2/pipeline.py).

An actual original-model CPU check produced finite2304features in2.43seconds for one origin. A four-origin batch took5.52seconds, with the first origin differing from its isolated result by at most7.15e-7. Thus grouping did not mix different origins. Timing receipts and original/native source hashes are stored in`runs/barrier-chronos-tcn-timing-20261022`.

TCN trains from scratch on the same three scales of relative-v2 history. Each scale has an independent64-channel encoder with8residual levels, two kernel3convolutions per level and dilations1..128. Left-only padding and per-time channel LayerNorm preserve prefix causality. The final state has a1021-bar receptive field, covering every supplied512/512/336-bar window. The three summaries feed the existing query-conditioned five-bin risk head. Fit-only scale/channel statistics are saved with the weights. [TCN sequence-model reference](https://arxiv.org/abs/1803.01271).

Both supervised fits use seed42, CPUFP32, batch64, AdamW lr0.0006/decay0.03, gradient clip1, maximum80epochs and validation patience8, matching the existing neural transfer-head budgets. Native feature dimensions, trainable encoder capacity and pretraining differ; this is matched supervised data/budget, not matched total compute. The original full-fit weekly reference still used274355training origins. A real4-origin128-query TCN forward/backward took0.525seconds. These timings are individual probes, not guaranteed whole-run completion times.

`checks.check_barrier_chronos_tcn` covers literal multiscale pooling, native REG selection under padding, price/volume-unit and future-tail exclusion, prefix-causal convolution, week-old input sensitivity, exact saved-model inference, fit-only normalization and incomplete sibling-report reads. A reviewer independently checked native group handling, token order, receptive field, label/scoring alignment and fit-only scaling. Its one reporting-race finding was fixed before launching the extension.

The user also asked whether the original ready-made foundation models had been tested directly. The answer for this barrier benchmark is **no native zero-shot first-touch experiment yet**. Frozen original weights plus a newly trained barrier head are supervised transfer, not direct forecasting without added training. The first Small transfer score is1.954779 versus1.811296 for its same-head query-only control on the same pilot selection origins. This extraction/head configuration did not improve the control; it does not establish that native Kronos forecasting is inferior. Chronos-native marginal quantiles do not define joint first-touch probabilities, and a future native-path experiment must handle the previously documented path length and ambiguous-cause issues explicitly.

## Post-training benchmark and architecture proposals

The CPU baseline pilot completed: CatBoost query-only selection NLL1.762734, CatBoost pooled candles1.750324, and pooled-candle MLP1.820405 on the common512selection origins. The full-fit weekly seed42 scores1.715410 there; its larger supervised fit budget remains explicit. These are raw selection results, not calibrated test rankings.

The user authorized a benchmark when current training/comparisons finish. `runs/barrier-benchmark-plan-20261022/protocol.json` and`cohorts.npz` now freeze **all5380early calibration origins and31504early test origins**, each with128queries. This supersedes a512-origin draft before any scoring; the draft is retained as`*-pilot-superseded.*`. Cohorts use only original chronological roles and complete-week eligibility, never outcome-based sampling. Candidate weights/configs/normalizers must freeze before calibration/test scoring. Full-fit weekly candidates and1024-fit auxiliary candidates are separate budget groups. The requested benchmark includes raw/calibrated proper path NLL, horizon logloss/Brier/reliability, conditional TP given a hit, coherence checks, descriptive paired UTC-day intervals, source/weight receipts and inference timing. Main-run full two-fold tests remain a separate result from the common early-fold comparison. A thread heartbeat is configured to check completion every30minutes, remain quiet while runs are healthy, and execute the benchmark after all current runs complete. It must not alter active numerical sources or tune on test outcomes.

Architecture changes below are **proposals, not active modifications or demonstrated fixes**:

1. First fit and freeze a query/asset/volatility baseline, then learn a regularized history-dependent correction to its hazard logits. Initialize the correction output at zero and preserve the existing incidence construction. This tests whether anchoring history to a strong baseline reduces negative transfer and memorization; no benefit is assumed. The current full model jointly fits query/history transformations and does not impose this frozen-baseline constraint.
2. Pretrain the candle encoder using only the fold's fit history, for example masked candle-patch reconstruction or a declared time-series representation objective. Then train the barrier head and unfreeze the encoder gradually with a lower encoder learning rate. Do not use calibration/test history even as unlabeled pretraining data. PatchTST and TS2Vec provide precedents for patch/representation pretraining, not evidence of improved barrier prediction on this dataset. [PatchTST](https://arxiv.org/abs/2211.14730), [TS2Vec](https://arxiv.org/abs/2106.10466).
3. Compare backbone, dropout and optimization changes through controlled fit/selection ablations and multiple training seeds. TCN is already the independently trained alternative underway. Additional TP/SL queries from one origin share its future path, adjacent five-minute origins have overlapping24h outcomes, and four crypto markets share regimes. Treating these query counts as independent sample size would exaggerate the available information.

Both completed weekly seeds selected epoch2 and then deteriorated. This supports a repeatable training/validation gap, but does not uniquely identify its cause. The earlier restarted-optimizer LR probe did not rescue selection performance by reducing the learning rate; its small-cohort conditions remain different from the full run. Architecture may improve inductive bias and regularization, but cannot guarantee removal of overfitting, temporal shift or weak predictability. Current architecture and training sources remain unchanged while their frozen experiment finishes.
