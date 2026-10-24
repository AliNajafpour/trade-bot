# Recorded-observation fusion replay

Window: 2026-10-18T15:36:41.555651+00:00 to 2026-10-21T15:36:41.555651+00:00.

Verified real ETH/SOL books: 4. Rejected failed/mismatched receipts: 6.
Positive technical candidates: 0; new fusion evaluations: 0; actual new CLI calls: 0.

| Market | Five-minute coverage ticks | With observed fresh news | With observed facts | With both |
|---|---:|---:|---:|---:|
| ETH/TMN | 865 | 25 | 1 | 1 |
| SOL/TMN | 865 | 25 | 0 | 0 |

Admission counts: {"technical_gate_failed": 4}.

No historical fusion PnL is claimed. No historical order-book interpolation or candle-derived fill prices were used. Admission-risk diagnostics are not executed trades. Existing four-role probe receipts were checked separately and are not counted as new inference.

Coverage uses original receipt availability, not publication dates. Three days of candles do not imply three days of news/fundamental observations. Missing evidence remains unknown. Recorded books cover isolated moments; no full holding-window executable book path is available.

Historical replay using current pretrained LLM and retrospectively fitted/exposed model research candidate; not prospective validation.

Full raw receipt references, model identities, numeric scans and evidence hashes are in `observed-fusion-replay.json`.
