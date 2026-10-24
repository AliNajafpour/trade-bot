# Portable agents implementation plan

1. Add local read-only Wallex MCP bridge and stdio/HTTP clients; prove protocol initialization and public call.
2. Add interchangeable Codex and API inference providers with receipts, strict schema validation and secret-safe failures.
3. Extract existing specialist stages and execute them as real parallel LangGraph nodes with SQLite persistence. Preserve legacy callers.
4. Connect graph selection to the existing paper runner; add portable MCP-fed CPU runner, environment/config examples and container instructions.
5. Run deterministic boundary/integration checks, a live public MCP probe and real Codex graph smoke. Keep current running suite untouched.
