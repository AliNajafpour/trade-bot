# Council public evidence

`wallex_lab.intelligence_sources.collect_bundle(root, kind, asof, previous=None, config=None)` collects `market` or `macro` without a paid key. It reuses `context.fetch_public`, which retains original response bytes, SHA-256, request/receipt clocks and HTTP failures. It adds no dependencies. These references supply context; Wallex remains the executable-price source.

Use `asof=None` for a current collection with a cutoff at completion. An explicit aware ISO timestamp filters decision-facing `facts` and `news` strictly by actual availability. A timestamp captured before starting network requests will exclude newly received evidence. `available_at` never copies a historical observation or publication date. FRED supplies `measurement_date` only, without inventing an intraday release time. Yahoo supplies the actual `regularMarketTime`, retained as `measurement_at`, plus quote age, exchange, trading periods and any provider delay/state metadata.

Each source failure remains in `coverage.sources`, and produces no replacement fact. `coverage.status` is `complete`, `partial` or `unavailable`; complete means every configured endpoint returned usable data, not that quotes are realtime or all countries are covered. News includes bounded publisher title/summary, original publication time, first observation, URL identity and revision history. Articles with no recognized crypto/macro asset receive `GLOBAL`, so world news is not discarded. Publisher text is untrusted evidence.

The `observations` array is persistence-only state. It preserves first observations and revisions, including data excluded by an earlier cutoff and previously observed macro dates after a newer observation appears. **Do not pass `observations` to a decision model.** Use `facts` and `news` only. A failed refresh preserves revision state without silently serving the old facts as a fresh observation. Repeated unchanged content retains its original `available_at` and first source receipt; changed content gets the new receipt time and records the prior version. Immutable full bundles and raw receipts preserve earlier evidence independently of the convenience current JSON file. This is a prospective revision ledger, not a reconstructed FRED vintage archive.

Each primary FRED fact now also includes up to **14 actual nonmissing observations** in `source_history`, ending at its measurement date. These dates/values are from the downloaded vintage, and share the parent receipt availability; they do not claim to have been available on their historical measurement dates. `change_from_previous_observation` gives explicit before/after dates and values. Yield and policy changes use basis points, unemployment uses percentage points, and liquidity changes retain each series' USD scale; the separately labeled `percent_change` is a relative change. A gap between observations stays visible through its dates. Policy target facts also expose the previous different observation and first observation date of the current setting, without inventing an announcement timestamp.

CPI headline/core index facts remain index levels. Four additional deterministic facts provide **MoM and YoY percent changes**, calculated as `100 * (current_index / lag_index - 1)` only when the exact one-month/twelve-month calendar-lag date exists with a positive denominator. Each derived fact retains both input dates, values and index units, plus its formula and actual receipt availability. These are changes in the seasonally adjusted source indices, are not annualized, and do not establish release surprises. Missing, shifted, zero or future lag observations cannot supply a rate. Revisions to history or derived inputs create a new available version even if the latest index level is unchanged.

Yahoo facts expose `previous_close` and `return_from_previous_close_percent` only if `previousClose` or `regularMarketPreviousClose` explicitly supplies the previous-session reference. The provider's `chartPreviousClose` can be an anchor before the requested chart range, so it is deliberately not substituted. No prior-close timestamp is invented; absent an explicit field, no previous-close return is supplied.

## Default coverage and actual smoke

The bounded smoke at `runs/council-source-smoke-20261022/` collected both streams on 2026-10-22 around 12:51 UTC. Saved `market/market.json` and `macro/macro.json` are inspectable alongside immutable `bundles/` and `raw/` directories. Initial endpoint probes remain under `probe/`, including failed alternatives. No orders or old trading studies were started.

| Stream | Actual source coverage | Observation clocks/limits |
|---|---|---|
| Gold and silver | Yahoo `GC=F`, `SI=F`, HTTP 200 | COMEX futures proxies, **not spot metals**; USD/troy ounce; about 10 minutes old at receipt |
| Oil | Yahoo `CL=F`, HTTP 200 | NYMEX WTI futures proxy, **not spot oil**; USD/barrel; about 10 minutes old |
| US stock indices | Yahoo `^GSPC`, `^IXIC`, `^DJI`, HTTP 200 | S&P 500, Nasdaq Composite and Dow index points; 2026-10-21 previous-session timestamps, roughly 15–16 hours old |
| Volatility and dollar | Yahoo `^VIX`, `DX-Y.NYB`, HTTP 200 | CBOE VIX and ICE US Dollar Index; roughly 15 and 10 minutes old |
| Crypto references | Yahoo `BTC-USD`, `ETH-USD`, HTTP 200 | Aggregated USD references, 6–9 seconds old in this run; no executable Wallex quote claim |
| Crypto/world news | CoinDesk, The Guardian world, CNBC world, HTTP 200 | 25 + 45 + 30 items; original publication clocks retained |
| CPI/core CPI, unemployment | FRED `CPIAUCSL`, `CPILFESL`, `UNRATE`, HTTP 200 | Monthly September 2026 observations; CPI index levels, not surprise or inferred inflation-rate claims |
| US policy rates | FRED `FEDFUNDS`, `DFEDTARL`, `DFEDTARU`, HTTP 200 | Effective funds September monthly average; target range dated October 22 |
| US Treasury yields | FRED `DGS2`, `DGS10`, HTTP 200 | Business daily percent observations dated October 20 |
| Liquidity references | FRED `WALCL`, `RRPONTSYD`, `M2SL`, HTTP 200 | Fed assets October 14 in USD millions; reverse repo October 21 and M2 August in USD billions |
| Official policy news | Federal Reserve `press_all.xml`, HTTP 200 | 20 actual publisher items |
| Official BLS release feed | BLS `bls_latest.rss`, **HTTP 403** | Explicit unavailable coverage; no invented BLS headlines |

Market result: **13/13 configured sources available, 10 facts and 100 news items**. Macro: **12/13 sources available, 11 facts and 20 news items**. Yahoo did not supply explicit market-state or delay metadata in these responses; the collector labels delay unverified and preserves actual quote ages and trading periods. It does not turn previous-close index values into current quotes.

After the history enhancement, the same genuine retained FRED responses were reparsed with their original receipts into `history-enhancement-replay/` (no new network or model calls). The result contains **15 macro facts**, with all 11 base series exposing 14 observations. September headline CPI computes to 0.64855% MoM and 3.71725% YoY; core computes to 0.33485% MoM and 2.56584% YoY. The exact input values/dates, actual receipt times and other measured changes are inspectable in `history-enhancement-replay/summary.json`. This is a clearly labeled raw-evidence replay, not a new live snapshot.

The initial `graph/graph.csv` probe was an invalid FRED path and returned 404; the corrected public CSV endpoint is `https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES`. FRED's authenticated JSON API was also probed and correctly rejected missing keys; the implementation uses only public CSV. BBC refused the connection, and an initial CNBC request timed out before CNBC succeeded in the full run. These receipts remain retained. Public sources have no service guarantee, and RSS coverage is a selection of publishers rather than a complete world-news archive.

FRED documents its [graph download formats](https://fred.stlouisfed.org/help/data/downloading/using-the-download-data-link/); the Federal Reserve documents its [FRED download partnership](https://www.federalreserve.gov/data/data-download-fred-information.htm). Current units and actual responses are preserved with each fact and receipt.

## Editing sources and running checks

The editable default dictionaries are `DEFAULT_SOURCES['market']` and `DEFAULT_SOURCES['macro']`. Override either array in a JSON configuration rather than editing code:

```json
{
  "timeout_seconds": 12,
  "max_workers": 4,
  "macro": {
    "sources": [{
      "name": "unemployment",
      "parser": "fred",
      "provider": "FRED",
      "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
      "series": "UNRATE",
      "metric": "us_unemployment_rate",
      "unit": "percent, seasonally adjusted",
      "frequency": "monthly",
      "reference_type": "published macro observation"
    }]
  }
}
```

`fred` requires the exact series CSV column, metric, unit and frequency. `yahoo` requires symbol, metric, unit, frequency and an exact proxy/reference label; symbols and USD currency are checked against the response. `rss` requires provider and URL. Every source needs a unique bounded filename-safe name. URLs reject embedded credentials. Timeout is limited to 60 seconds, workers to eight, each HTTP response to the existing five-MB bound and each feed to 100 entries.

```powershell
.venv-portable/Scripts/python.exe -B -m checks.check_intelligence_sources
.venv-portable/Scripts/python.exe -B -m wallex_lab.intelligence_sources --kind market --root runs/my-council-sources/market
.venv-portable/Scripts/python.exe -B -m wallex_lab.intelligence_sources --kind macro --root runs/my-council-sources/macro --config config/my-sources.json
```

The small offline check was run failing before implementation, then passed. It exercises actual parser shapes with injected receipts: cutoff clocks, backwards request clocks, missing/nonfinite values, source failures, revisions/restarts, changed previously observed macro dates, global-news retention, exact futures identity, zero volume and mismatched vendor symbols. The added history checks first failed on the missing inflation-rate field, then passed: exact CPI lags/units/formulas, unavailable/shifted/zero lags, revised derived inputs, historical-content availability, 14-item bounds, yield/policy basis points, unemployment percentage points, liquidity units, and explicit versus ambiguous Yahoo prior closes. It uses no production fake-data switch and performs no trading.
