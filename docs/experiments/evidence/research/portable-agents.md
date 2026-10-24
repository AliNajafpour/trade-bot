# Run the agents locally or on another machine

The language model defaults to the installed, authenticated Codex CLI. You can keep using this assistant workflow without a separate LLM API key. This is real CLI inference; the desktop conversation does not remain an unattended inference server by itself.

The new path is:

`Wallex MCP → stored candles/quotes → existing trained scanner → LangGraph specialists → existing independent risk checks → paper broker`

LangGraph prepares bounded packets, runs regime/news/fundamental roles in parallel, and joins their validated results for the decision. SQLite checkpoints and receipt hashes persist each analysis. An identical completed request does not call the model again. A provider error blocks that decision; it never becomes a simulated model response.

## Local setup

Python 3.11 is the tested runtime. From the project directory:

```powershell
python -m venv .venv-portable
.\.venv-portable\Scripts\python.exe -m pip install -r requirements-portable.txt
Copy-Item .env.example .env
.\.venv-portable\Scripts\python.exe -m wallex_lab.portable probe
.\.venv-portable\Scripts\python.exe -m wallex_lab.portable run --seconds 3600
```

`probe` performs real MCP initialize/list_tools/call_tool operations against the official `https://api.wallex.ir/mcp` server and saves its public response in `runs/portable-probe/probe.json`. The example run uses the existing CPU HGB artifacts in `runs/paper-models-20261021`, with a separate $100 paper portfolio at `runs/portable-paper`. It does not change or restart the 20-account trial. Model files must exist on the destination machine; only load trusted joblib artifacts.

Run the same command and config again to retain the paper ledger and monitor positions. Stopping this bounded collector does not liquidate positions. Keep it running to monitor TP/SL; it cannot execute while the machine sleeps. One process per run directory is the supported collector mode. The existing portfolio runner remains the fuller dashboard/multimodel path.

The HGB route imports no GPU framework. `scanner=learned_path` or `barrier-live` uses the existing policy/model loaders and requires their numerical model dependencies and policy artifacts; those artifacts may still contain source-machine paths. Copying the CPU HGB model directories is the currently portable example. Numerical forecasters are separate from the language model's API key.

The existing multi-model paper runner also accepts `--market-transport mcp`. It uses the same public market, candle, depth and trade payload contract through the official server, retaining source receipts and original server clocks. REST remains the default for previously recorded runs. Start a fresh run directory when changing transport; an existing run refuses a different transport.

```powershell
python -m wallex_lab.paper_session --help
python -m wallex_lab.paper_session --market-transport mcp --run-dir runs/paper-mcp-new --initial-usdt 100 --seconds 3600
```

That Python environment must contain both the chosen model dependencies and `requirements-portable.txt`. The separate portable environment contains MCP; the older external GPU environment has not been modified. Historical model replay can consume saved MCP candle CSVs without installing MCP into that GPU environment.

The recent-day study saves native candles, source receipts, model draws, review packets and independent ledgers under `runs/suite-backtest-24h-20261022`. Its collector and replay commands are reproducible:

```powershell
.\.venv-portable\Scripts\python.exe scripts/collect_suite_mcp_data.py --cutoff 2026-10-22T10:50:00Z
.\.venv-portable\Scripts\python.exe scripts/run_suite_backtest_24h.py --watch
```

The replay waits for complete forecast exports from the frozen model generators. Full-day figures are retrospective; artifact availability and the shorter interval after the exact suite definition are audited separately. Missing historical bid/ask/depth is represented by next-minute-open proxy fills and explicit slippage scenarios.

## Keep Codex now; connect an API later

The default `.env` contains `LLM_PROVIDER=codex`; the CLI must be installed and logged in on that host. It reuses the existing strict JSON schema and inference receipts, with hidden child processes on Windows.

For an API supporting OpenAI-compatible chat completions and JSON-schema structured output, edit local `.env`:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-id
LLM_API_KEY=your-key
```

Existing environment variables take precedence over `.env`. The CLI loads `.env` at startup; restart with the same run directory after editing it to retain the ledger and use the new provider. A newly requested analysis reads current process environment settings. Keys are never put in JSON configs, prompts or receipts. Some compatible providers do not implement strict JSON-schema output; the adapter fails explicitly in that case. An external provider's real credentials were not available for live verification; its HTTP contract is checked with a local test server.

To analyze an explicit stored packet:

```powershell
.\.venv-portable\Scripts\python.exe -m wallex_lab.portable analyze --packet path/to/packet.json --output runs/my-analysis
```

The packet contains `summary` (bounded finite numeric features), `context`, `symbol` and timezone-aware `asof`. Causal evidence selection and the existing independent risk rules remain in force.

## Existing portfolio runner

For a **new** session, add `--agent-engine langgraph` to the existing `python -m wallex_lab.paper_session ...` command. Its other model, policy, universe and dashboard options stay available. `review-mode=model_only` intentionally bypasses language inference. The GPU environment used for Kronos/Chronos also needs the LangGraph packages installed before selecting the new engine; this task installs new packages only into `.venv-portable`.

The original active trial keeps its loaded engine/config and was not restarted. Resuming a stored session requires the same engine used to create it.

## MCP connection

The example config connects directly to the official Streamable HTTP server at `https://api.wallex.ir/mcp`. Public tools need no API key. The client maps four verified public tools into the existing market/history contract. The project-owned stdio server remains available by setting `{"transport":"stdio"}`. See [primary sources and tool details](wallex-mcp-sources.md).

When a remote MCP URL is supplied, change the JSON connection:

```json
{"transport":"streamable_http", "url":"https://your-wallex-mcp.example/mcp", "token_env":"WALLEX_MCP_BEARER_TOKEN"}
```

For generic remote servers, put the bearer token in `.env`; their tool names/envelopes must match the local contract. For the verified official Wallex URL, optional private balance reads instead use `X-API-Key` from `WALLEX_API_KEY` and require `private_reads: true`. They do not use a bearer token. Keys are not forwarded to other hosts. Order/withdrawal tools advertised by the official server are outside this client's allowlist; the paper broker remains the only execution backend.

The official server has also been registered in this machine's Codex config with only the four public market tools enabled. Existing configuration was preserved. Reload the Codex client to load the native tools into a new session; this conversation can already use the SDK connection. No private key has been configured.

## Container

```sh
docker build -t wallex-agents .
docker run --rm wallex-agents probe
docker run --rm --env-file .env -v "$PWD/runs:/app/runs" wallex-agents run --seconds 3600
```

The image selects the API provider; supply its `.env` settings. To use Codex inside a container, install/authenticate that CLI there and select `LLM_PROVIDER=codex`; the host app login is not automatically available in a container. Paths in the example config are relative to the config file. Docker deployment requires mounted model artifacts and persistent run storage. Docker commands are provided but not claimed tested when Docker is unavailable on the host.

## Checks

```powershell
.\.venv-portable\Scripts\python.exe -m checks.check_wallex_mcp
.\.venv-portable\Scripts\python.exe -m checks.check_providers
.\.venv-portable\Scripts\python.exe -m checks.check_graph_agents
.\.venv-portable\Scripts\python.exe -m checks.check_portable
.\.venv-portable\Scripts\python.exe -m checks.check_specialists
.\.venv-portable\Scripts\python.exe -m checks.check_orchestrator
```
