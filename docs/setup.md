# Setup and operation

## Supported paths

1. **Offline UI demo:** open `video-demo/dashboard.html` directly. Everything on this page is synthetic. No Python, model, key, server or internet connection is involved.
2. **Bundled numerical model:** install `requirements.txt`, then run `python run.py verify`. The three final Weekly/late members are included; inference runs on CPU.
3. **Live-data paper experiment:** configure a real LLM provider and run `python run.py paper`. Quotes, history and news require network access. Orders remain simulated.
4. **Optional research paths:** Kronos, Chronos-2 and legacy HGB pipelines have their own artifacts and dependencies. They are not prerequisites for the bundled Weekly/late route.

Use Python 3.11 and a virtual environment. `requirements-portable.txt` is a lighter MCP/LangGraph/HGB environment; it intentionally does not install Torch and cannot run the bundled survival model. `requirements.txt` includes Torch. Install a CUDA-compatible Torch build separately if training on a GPU; CPU inference does not require CUDA.

```bash
python -m venv .venv
# Linux/macOS:
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py verify
```

Windows commands are in the root README. Run scripts from the repository root. Prefer `python -m scripts.<module>` for research scripts so repository imports resolve.

## Language model configuration

Copy `.env.example` to `.env`. Choose one provider:

```dotenv
LLM_PROVIDER=codex
LANGSMITH_TRACING=false
```

This requires an installed, authenticated Codex CLI on the machine running the program. The ChatGPT/Codex conversation itself is not a persistent background provider.

Or use a compatible endpoint supporting the requested JSON Schema and chart/image inputs:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_API_KEY=replace_locally
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=replace_with_a_supported_model
LANGSMITH_TRACING=false
```

The API provider reads these environment variables; requests use Authorization headers. Do not embed keys in URLs, JSON configurations or Git commits. The chart-aware manager needs a provider/model with image support. Missing inference or failed schema validation is recorded as failure, never a generated fallback opinion.

## Paper launch

```bash
python run.py paper
```

The launcher verifies model hashes, creates `runs/weekly-paper/launch.json`, starts a public quote/report collector, and starts `wallex_lab.survival_news`. The source collector calls existing Wallex quote and intelligence-report components; it does not run a second technical strategy or manage another trading account. The dashboard binds to `127.0.0.1:8850`.

Configuration files:

| File | Purpose | Path semantics |
|---|---|---|
| `weekly-paper.example.json` | Model, run directory, port, history mode, entry window | Relative to repository working directory |
| `weekly-source.example.json` | Quote and report schedules, sources | Relative to repository working directory |
| `paper-capital.example.json` | Virtual starting cash and fee assumptions | Read and frozen into the paper run |
| `portable-paper.example.json` | Optional older HGB/MCP route | Loader resolves paths relative to config file |
| `live-execution.json` | Separate, disabled real-order boundary | Run from repository root |

The default entry window is two hours. Its original deadline persists across restarts; restarting does not grant another two hours. Protection of open paper positions continues while the runner is active. Stop with Ctrl+C. Resume with the same configuration and run directory to retain the account. For a genuinely new experiment, choose a new run directory instead of deleting/resetting an account file.

Changing the capital seed, fee assumptions or model identity of an existing run may be rejected. Preserve the original ledger and create a separate experiment when changing those assumptions. The launcher computes the model metadata digest from the bundled model for the runtime's identity check.

## First-start data and failures

Weekly/late needs 10,080 contiguous completed one-minute bars. The default live experiment collects coherent KuCoin/MEXC reference history and anchors it to a completed Wallex close. This can take time and can fail due to source availability; the model must not be fed invented candles. Quote fallback reads the official public Wallex REST endpoint if MCP is unavailable. Quote age is preserved through failures.

Look at `runs/weekly-paper/status.json`, `runs/weekly-source/runtime-state.json`, and `runs/weekly-paper/source.log`. A missing LLM login, unavailable API, incomplete candle history or stale book is an operational failure with a recorded reason. A running dashboard does not prove all workers are healthy.

The standalone `wallex_lab.paper_services` module is retained because runtime code shares its resource helpers. Its old local study registry is not the launch path for this distribution. Use `run.py paper`; do not start multiple launchers on the same account directory or port.

## Public MCP probe

```bash
python -m wallex_lab.portable probe --config config/portable-paper.example.json
```

The probe performs actual initialize/list/call operations against `https://api.wallex.ir/mcp`. It does not require HGB weights or private account access. Running the optional portable HGB **trader**, however, requires its separately trained model artifacts. Do not confuse those with the bundled survival ensemble.

## Containers

The existing Dockerfile is a lightweight **public MCP probe** image using `requirements-portable.txt`. It does not copy Torch weights or launch Weekly/late. Use the Python setup above for the main bundled experiment. This distinction prevents a probe container from silently becoming a trading service.
