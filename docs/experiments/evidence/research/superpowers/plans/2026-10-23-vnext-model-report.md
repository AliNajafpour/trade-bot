# vNext model implementation receipt

Task 3 is implemented in `wallex_lab/barrier_vnext_model.py` with its runnable synthetic CPU check in `checks/check_barrier_vnext_model.py`. This worker edited only those files and this receipt. No GPU was allocated, source acquisition was untouched, and frozen model/data modules were read without editing.

## Contract and design

`NextBarrierModel(symbols, context=10080, width=96, layers=3, activity=False, age=False, level_attention=False, local=False)` accepts `x: B,T,14`, `query: B,K,8`, and symbol indices, and returns hazard logits `B,K,5,3`. Without activity it uses only the first ten history channels. `.context`, `.query_only=False`, and constructor-compatible `.config` are available. Save both `.config` and `.state_dict()`; reload with `NextBarrierModel(**config)` and strict `load_state_dict`.

The model subclasses the unchanged `MultiScaleSurvival` with hierarchical aggregation. All shared baseline layers are initialized before optional modules. The check proves exact seeded shared-parameter parity for every variant and bitwise baseline forecast parity. Activity extends the input convolution by four initially zero weights, retaining the first ten weights and the bias. Those additional weights receive training gradients. Age adds learned physical token age and duration. Level attention reconstructs normalized physical TP/SL coordinates from entry offset, side, log distance and sigma, then adds continuous distance-to-token-range bias to cross-attention. Local adds a learned convolution residual from the final 60 raw M1 inputs.

`historical_mask(batch, context, generator, device='cpu')` uses a CPU `torch.Generator` to choose up to three physical contiguous intervals, then transfers the boolean B,T mask to the requested device. Mask RNG state belongs in the runner checkpoint. Pass a mask whose time dimension equals the configured context.

`pretrain_loss(x, symbol, mask)` masks every channel before all pooling and the local branch. It reuses the downstream patch convolutions, transformer, symbol embeddings and applicable optional encoder layers. A saved reconstruction head predicts normalized return/range/volume means from channels 7/6/4. Every token with any physical hidden overlap has a target, and its target averages hidden minutes only. Unmasked later minutes within partially hidden coarse patches cannot alter the target. Reconstruction targets never enter downstream `forward`. Empty masks are rejected.

## Verification

Interpreter: `D:/ALI/Programming-AI/Personal Projects and Tests/Trade-Bot/.venv-gpu/Scripts/python.exe`.

Commands run from the workspace root:

```powershell
& 'D:/ALI/Programming-AI/Personal Projects and Tests/Trade-Bot/.venv-gpu/Scripts/python.exe' -B -m checks.check_barrier_vnext_model
& 'D:/ALI/Programming-AI/Personal Projects and Tests/Trade-Bot/.venv-gpu/Scripts/python.exe' -B -m checks.check_barrier_survival
& 'D:/ALI/Programming-AI/Personal Projects and Tests/Trade-Bot/.venv-gpu/Scripts/python.exe' -B -m checks.check_barrier_vnext_data
```

The model check was written first and failed with `AssertionError: Missing vNext model` before production implementation. The completed check passes on CPU with context720/width16/layers1. It covers baseline parity, independent flags and shared initialization, physical long/short geometry with nonzero entry offset, coherent cumulative incidence, deterministic interval masks, raw-channel hiding across all scales and the local window, hidden-only reconstruction summaries with hand-computed values, gradients through each optional encoder branch, actual transformer parameter updates, downstream reuse of the same encoder, trainable level bias, and safe `torch.load(weights_only=True)` reload parity.

Neighbor checks also returned exit code 0: existing survival labels/ambiguity/censoring/purge/gradients/calibration/reload and vNext data causality/128 requests/ambiguity/censoring/unit invariance/source identity. No umbrella project test command is defined in the inspected workspace; these were the relevant bounded CPU checks. An interrupted verification invocation was rerun rather than counted as passing.

## File identity and limits

SHA256 at handoff:

- `wallex_lab/barrier_vnext_model.py`: `44cab8463ec0801a9ee732ca89a66136e45ef5adaec24b9e7bb76308daced9d3`
- `checks/check_barrier_vnext_model.py`: `e56a6059999bb4759f721e9fecd8e602b1d76229be7596520506e45e4ef9917a`

Unmodified imported reference files observed at handoff:

- `wallex_lab/barrier_survival.py`: `80b67fce8bf56567c396aeef386e5d57637617ed88bce9f2ef279e8ca73fc612`
- `wallex_lab/barrier_dataset.py`: `14bb34b3cf6f16f892b4330146641741975db5a844f49bb34a2ba97fb741e9d1`

No real-data pretraining, GPU fit, model selection, or performance claim was performed by this worker. The runner must restrict reconstruction origins to the fit role and preserve config/data/RNG/optimizer identity on resume. CPU checks establish implementation behavior, not downstream predictive improvement. Production-width/week-context GPU memory and throughput remain unmeasured. The target is a summary of already supplied relative-v2 features; raw-price normalization remains the dataset's causal normalization contract.
