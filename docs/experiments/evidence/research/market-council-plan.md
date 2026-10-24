# Market council implementation plan

> For agentic workers: execute the independent source, report, technical and vision tasks in parallel, then integrate and review the manager in this session.

Goal: implement the user-specified periodic evidence streams and persistent, image-aware spot manager.

Architecture: reuse existing public-source receipts, MCP, LLM providers, LangGraph and PaperBroker. Independently timed immutable reports feed a candidate-conditioned technical engine and a durable manager, with a separate protective exit loop.

Tech stack: Python3.11, installed NumPy/pandas/sklearn/Torch runtimes, LangGraph/SQLite, existing public MCP and provider adapters. New dependencies only where an existing runtime cannot supply required chart functionality.

Spec: docs/market-council-design.md

## Global constraints

- Paper spot only; no leverage, short orders or withdrawals.
- Market report1800seconds, macro86400seconds, technical60seconds by default; all editable.
- WAIT300seconds must survive restart and trigger reanalysis with history.
- Data, probabilities, model responses and source dates must be genuine and versioned.
- Preserve stopped studies and external gold project files/environments.

## Review focus

- A delayed/closed-market quote must not appear as a live quote.
- A macro observation date must not become a historical first-availability date.
- Changing TP/SL must invalidate old probability/economic scores.
- Model inference must not hold up protective exits.
- A process restart must not duplicate fills or forget pending reviews.

## Task1: public evidence

Files: intelligence_sources.py, check_intelligence_sources.py. Consume existing context.fetch_public/receipt/parser contracts; produce collect_bundle from the spec. Write failing checks for clocks, absent facts and revisions, then implement and run. Verify actual public source responses with raw evidence, including explicit failed coverage.

## Task2: reusable reports

Files: intelligence_reports.py, check_intelligence_reports.py. Consume collect_bundle dictionaries and existing provider callable. Produce ReportStore refresh/get/invalidate. Test independent intervals, expiry at the exact boundary, invalidation, corruption, restart and genuine provider failures before implementation. No neutral fake model response.

## Task3: learned technical engine

Files: council_technical.py, train_council_technical.py, check_council_technical.py. Produce TechnicalEngine.predict and genuine trained artifacts. Test candidate geometry features, chronological/censoring labels and different probabilities for rescored levels before implementation. Train on real spot data with disjoint chronology; save curves, source hashes and validation metrics. Report named-pattern limitations honestly.

## Task4: charts and vision

Files: llm.py, providers.py, council_charts.py, check_council_vision.py. Optional images argument; exact image bytes enter requests and cache hashes. Test default compatibility, changed image, bad input and actual compatible API content before implementation. Verify actual Codex image-bearing inference and retain receipt. Existing provider checks remain passing.

## Task5: manager/runtime/integration

Files: market_council.py, config/market-council.example.json, focused check, docs/runtime instructions, minimal status surface. Implement the LangGraph manager, normalized weights, candidate rescoring and independent exit monitor. Persist pending review/history and idempotent amendments using existing broker/locks. Test WAIT300/restart/history, stale reports, malicious/invalid actions, changed barriers and protective exits during slow inference. Integrate tasks1–4, run appropriate existing checks and bounded real public-data/LLM smoke. Obtain independent read-only review and repair substantive findings. No automatic old-test restart or live orders.
