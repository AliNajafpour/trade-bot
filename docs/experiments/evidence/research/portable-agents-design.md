# Portable Wallex agents

The existing paper run and its ledgers continue unchanged. New sessions can select a real LangGraph specialist workflow while retaining the current scanners, causal evidence filters, independent spot risk checks and idempotent paper broker.

The graph prepares bounded numeric/evidence packets, runs regime/news/fundamental nodes in parallel, then joins them for a validated decision. SQLite checkpoints and model/provider identities make completed analyses reusable after restart. Failed inference never becomes a heuristic approval.

`LLM_PROVIDER=codex` remains the default: the installed, authenticated Codex CLI supplies actual language-model inference without a separate API key. `openai_compatible` uses configurable base URL, model and environment key. The desktop conversation itself is not an unattended server; CLI authentication/runtime must remain available.

The client now defaults in the example configuration to the verified official Streamable HTTP endpoint `https://api.wallex.ir/mcp`, supplied by the user after the initial local implementation. Four public tool schemas are explicitly mapped to the existing data contract. A project-owned stdio server remains a fallback. Public tools work without credentials; optional private balances require explicit opt-in and `X-API-Key` from the environment. No live order tool is allowed through the client.

A portable entry point connects MCP market/history tools → existing trained scanner → graph-backed Orchestrator → independent paper execution. Models are explicit local artifacts; an API key does not replace numerical weights. JSON configuration uses paths relative to its file and environment variables hold credentials. Docker runs the CPU HGB route with mounted artifacts; native Kronos/Chronos sessions retain their separate dependencies.

Checks cover actual MCP initialization/tool dispatch, strict API schemas and secret redaction, graph persistence/parallel roles, unchanged legacy specialist behavior and paper execution without duplicate fills. Network probes are recorded separately from deterministic checks.
