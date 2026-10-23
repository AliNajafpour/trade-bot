# Paper and real execution

## Default behavior

`run.py paper`, `survival_news.py` and `spot.PaperBroker` manage virtual balances and simulated fills. They do not call a private create-order tool. The read-only MCP client has an explicit read allowlist. A bearish opinion in a spot account means cash, reduced exposure or selling owned inventory.

The separate `live_execution.py` gateway is shipped **disabled**, with no bound account and `stop_capabilities_verified=false`. The paper launcher never starts that gateway. No current private balance, API key or live account ledger is included in Git.

## Read-only preflight

Set `WALLEX_API_KEY` locally in the environment or `.env`, then explicitly run:

```bash
python -m wallex_lab.live_execution --config config/live-execution.json
```

The default command reads the selected credit subaccount, authenticated fees, venue constraints and open orders. It persists local preflight status and an account fingerprint. It does not activate trading or submit orders. Public market data and the bundled model need no private key.

## Gateway safeguards

- Credit-account binding and a separate SQLite FULL/WAL intent ledger.
- Stable client IDs, economic-parameter consistency and persistence before a possible POST.
- No blind repeat POST after an uncertain result; reconcile by client ID first.
- Detailed executed quantity and fill/fee accounting; acknowledgement alone is not a fill.
- Sell only confirmed bot-owned inventory, never unrelated account holdings.
- Account-level lifecycle locking and active/uncertain sibling SELL checks.
- Pending BUYs, unprotected inventory and unmanaged open orders prevent new entry admission.
- Final reconciliation plus fresh account/open-order observations before admission.
- Native STOP_LIMIT protection of filled quantity; reject/unknown protection is an error.
- Quote/size validation before removing protection for exit. Known pre-submission failure attempts protection recovery; an uncertain exit does not cause a second SELL reservation.

Initial planned limits are 15% allocation per position, 0.5% account stop risk per trade, 2% aggregate planned stop risk and four positions. These are configured planning limits, not guarantees against gaps. A 20 bps stop-limit reserve is an assumption separate from actual fees.

## Not yet verified for unattended live use

Native stop acceptance, exact returned trigger fields and actual triggering have not been tested with a real order. Stop-limits can remain unfilled through a price gap. There is no asserted atomic OCO bracket; target exits need a running actor. A failed API or local process can interrupt target monitoring and protective recovery.

Both enabled configuration and an explicitly write-enabled transport are required for mutation. The default configs remain disabled. The separate `live_runner.py` connects the Weekly/late predictor and manager to the real gateway; `cloud.py` supervises it alongside public reporting and the authenticated control dashboard. It receives fresh real-account decisions and never replays a virtual account's trades. The protection loop retires never-submitted BUY intents instead of dispatching them after restart; uncertain submissions and protective orders retain their reconciliation state. See [deployment and user activation](deployment.md).

The focused offline boundary check uses a fake exchange:

```bash
python checks/check_live_execution.py
```

It covers duplicate and conflicting intents, pending exposure, concurrent dispatch, partial fills, accepted-but-timed-out submissions, cancellation races, owned inventory, stale orders, rejected protection and pre-submission recovery. It makes no network requests and does not validate live stop behavior or profitability.

## Secret handling

Keep keys in `.env` or the launching process environment. No key should have withdrawal permission for this research workflow. Do not commit private MCP responses, account snapshots, logs or live SQLite ledgers. The `.gitignore` excludes those artifacts and all model checkpoints except the three intentional final inference members. Original third-party licenses and attribution are retained.
