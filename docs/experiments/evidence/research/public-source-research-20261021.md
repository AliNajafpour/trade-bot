# Public source integration review, 2026-10-21

This review uses official product documentation and source repositories. Documentation examples are separate from the live observations recorded below.

## OpenBB news

Current OpenBB V5 exposes `obb.news.rss`, not a guessed `news.world` RSS provider. A documented call is:

```python
from openbb import obb
result = obb.news.rss(source="yahoo_finance", limit=10, fetch_body=False)
```

`source` is a registered feed key. Custom URLs belong in `[news.rss_feeds]` in `openbb.toml`; `merge_defaults=true` preserves bundled feeds when adding custom entries. RSS needs no provider credentials. `fetch_body=False` avoids fetching arbitrary article pages. Installation is `pip install openbb-news` followed by `openbb-build`; the documented dependencies are `openbb-core[pandas]`, `feedparser`, and `lxml`. The existing runtime inspection found `openbb`, `openbb_news`, and `feedparser` absent. Since this repository already parses RSS/Atom with the standard library, an optional adapter plus the existing parser meets the requested ingestion task without changing the training environment. [Official extension documentation](https://docs.openbb.co/odp/python/extensions/openbb-news), [command reference](https://docs.openbb.co/odp/python/reference/news/rss).

## RSSHub

RSSHub is the DIYgod project that produces feeds from many source-specific routes. Its official repository links deployment documentation and includes Docker and Node package deployment assets. A configured instance and a verified route URL can be consumed by the same RSS parser. The integration should stay explicitly unconfigured until such a URL is supplied; setting up Docker is unnecessary for sources already publishing RSS. The documentation host returned HTTP 403 to this research browser, so no particular route's availability was inferred. [Official repository and deployment links](https://github.com/DIYgod/RSSHub), [deployment documentation](https://docs.rsshub.app/deploy/).

## DefiLlama

The official API index separates free and Pro operations. The free API documents `GET https://api.llama.fi/v2/chains` for current chain TVL, and `/v2/historicalChainTvl/{chain}` for dated chain TVL. The latter excludes liquid staking and double-counted TVL. Chain identity must be checked before attaching a USD metric to an asset. Receipt time is this system's availability time; a vendor measurement date is separate. A USD TVL level establishes neither a trend nor an intraday price effect. [Official API documentation](https://api-docs.defillama.com/), [official free-endpoint index](https://api-docs.defillama.com/llms-free.txt).

## CoinGecko

The Demo REST API documents `https://api.coingecko.com/api/v3/` with the `x-cg-demo-api-key` header. `/coins/markets` accepts comma-separated explicit coin IDs, with up to 250 IDs per request, and returns fields including market capitalization, fully diluted valuation, circulating supply, and `last_updated`. Supply fields may be null; null is unknown, never zero. Bulk IDs avoid ambiguous ticker matches. A keyless attempt may be recorded, but its observed result must not be represented as a credentialed Demo guarantee. [Demo authentication](https://docs.coingecko.com/demo/reference/authentication), [Demo markets reference](https://docs.coingecko.com/demo/reference/coins-markets).

CoinGecko supplies official Python and TypeScript SDKs compatible with Demo and Pro APIs. Its official MCP documentation also advertises a keyless remote endpoint, `https://mcp.api.coingecko.com/mcp`, with shared rate limits, plus an API-key endpoint and a local package. These are alternatives, not prerequisites for one bounded bulk HTTP request. No SDK installation or MCP connection is implied by this review. [Official SDK documentation](https://docs.coingecko.com/docs/sdk), [official MCP documentation](https://docs.coingecko.com/ai-integration/mcp-server).

## Receipt and safety contract

All provider text remains untrusted data. Keep bounded feed titles and summaries rather than arbitrary article HTML. Persist URLs, HTTP status, request/receipt clocks, raw response bytes and SHA256 hashes. Attach only finite validated numeric values to an explicitly identified asset. Keep vendor update/publication clocks distinct from local observation and availability. Report unavailable, unconfigured, stale, missing and rate-limited sources honestly; none supplies a verified directional trading prediction by itself.

## Actual integration run

`wallex_lab.public_sources.collect_sources(root, assets, previous=None, asof=None)` ran against the twelve selected paper-universe assets on 2026-10-21. Evidence is in `data/public-sources-20261021/current.json`, `raw/`, and `validation.json`.

- CoinGecko's single bulk unauthenticated REST request returned HTTP 200 at 16:29:32 UTC and supplied 50 finite cap, FDV and supply facts for eleven exact ID-and-symbol matches. USOON remained explicitly unmapped. This observed keyless success does not guarantee future access, quota or Demo authentication policy.
- DefiLlama returned HTTP 200 at 16:29:31 UTC and supplied eight explicitly matched chain TVL facts. USD vendor valuation levels remain descriptive.
- CoinDesk and Cointelegraph returned HTTP 200 and produced 25 relevant feed items. Kraken timed out after the bounded request and remains unavailable in this packet. No substitute headline was invented.
- Four nonempty original HTTP bodies passed SHA256 verification. Every numeric fact retained receipt-time availability and separate optional vendor measurement time. Source, numeric and future-measurement synthetic checks passed.
- OpenBB was not installed, so its bounded V5 adapter was not exercised. No environment or package installation was performed. RSSHub had no configured feed URL and no server was deployed. These statuses are explicit in `integrations`; neither is described as a working installed service.
