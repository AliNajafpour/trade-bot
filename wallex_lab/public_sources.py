"""Small public news/fundamental adapters; optional packages never auto-install."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlencode, urlsplit

from .context import FEEDS, ASSETS, fetch_public, parse_feed, save_json, text_content, timestamp
from .data import utc_now

# Explicit vendor IDs are checked again against response IDs AND symbols. USOON is deliberately unmapped.
COINGECKO_IDS = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'XRP': 'ripple',
    'DOGE': 'dogecoin', 'ADA': 'cardano', 'BNB': 'binancecoin', 'PEPE': 'pepe', 'SUI': 'sui',
    'XAUT': 'tether-gold', 'S': 'sonic-3', 'USDT': 'tether', 'LTC': 'litecoin', 'BCH': 'bitcoin-cash',
    'LINK': 'chainlink', 'AVAX': 'avalanche-2', 'DOT': 'polkadot', 'TRX': 'tron', 'ETC': 'ethereum-classic',
    'SHIB': 'shiba-inu', 'UNI': 'uniswap'}
CHAINS = {'ETH': 'Ethereum', 'SOL': 'Solana', 'BTC': 'Bitcoin', 'BNB': 'BSC', 'SUI': 'Sui',
    'AVAX': 'Avalanche', 'ADA': 'Cardano', 'S': 'Sonic', 'TRX': 'Tron', 'XRP': 'XRPL'}
FUNDAMENTAL_CACHE_TTL_SECONDS = 900


def fact(provider, asset, metric, value, unit, receipt, measurement=None):
    if isinstance(value, bool):
        raise ValueError('Boolean vendor metric')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Vendor metric must be finite and nonnegative')
    received = receipt['response_received_utc']
    if receipt.get('error') or receipt.get('status') != 200 or timestamp(received) < timestamp(receipt['request_started_utc']):
        raise ValueError('Failed source response or backwards receipt clocks')
    result = dict(id=hashlib.sha256(f"{provider}|{asset}|{metric}|{receipt['sha256']}".encode()).hexdigest(),
        source=provider, source_url=receipt['url'], url=receipt['url'], assets=[asset], metric=metric,
        value=value, unit=unit, observed_at=received, available_at=received, source_receipt=receipt,
        text=f'{provider} reports {asset} {metric}: {value:g} {unit}. This level alone establishes no directional price effect.',
        quality=['Vendor aggregate, not independently verified; current levels do not prove trend, surprise or intraday predictive skill.'])
    if measurement:
        if timestamp(measurement) > timestamp(received):
            raise ValueError('Future vendor measurement timestamp')
        result['measurement_at'] = measurement
    else:
        result['quality'].append('Metric update timestamp unavailable; only receipt-time observation asserted.')
    return result


def coingecko_facts(payload, assets, receipt):
    if not isinstance(payload, list):
        raise ValueError('CoinGecko market response must be a list')
    result = []
    for row in payload:
        asset = str(row.get('symbol', '')).upper()
        if asset not in assets or row.get('id') != COINGECKO_IDS.get(asset):
            continue
        for metric in ('market_cap', 'fully_diluted_valuation', 'circulating_supply', 'total_supply', 'max_supply'):
            value = row.get(metric)
            if value is None:
                continue  # Documented nullable values remain unavailable, never zero.
            unit = 'USD' if metric in ('market_cap', 'fully_diluted_valuation') else asset + ' tokens'
            result.append(fact('CoinGecko', asset, metric, value, unit, receipt, row.get('last_updated')))
    return result


def defillama_facts(payload, assets, receipt):
    if not isinstance(payload, list):
        raise ValueError('DefiLlama chains response must be a list')
    result = []
    for row in payload:
        asset = str(row.get('tokenSymbol', '')).upper()
        if asset in assets and row.get('name') == CHAINS.get(asset):
            result.append(fact('DefiLlama', asset, 'chain_defi_tvl', row['tvl'], 'USD', receipt))
    return result


def asset_name(value):
    name = str(value).upper()
    if '/' in name:
        return name.split('/', 1)[0]
    if name in COINGECKO_IDS or name in ASSETS:
        return name
    return re.sub(r'(?:USDT|TMN)$', '', name)


def _cached_fundamentals(previous, provider, assets, now):
    rows = [row for row in (previous or {}).get('fundamentals', [])
        if row.get('source') == provider and set(row.get('assets', [])) & assets]
    if not rows:
        return []
    try:
        for row in rows:
            receipt = row['source_receipt']
            original = row['observed_at']
            if (row['available_at'] != original or receipt['response_received_utc'] != original
                    or receipt.get('error') or receipt.get('status') != 200
                    or not 0 <= (now - timestamp(original)).total_seconds() < FUNDAMENTAL_CACHE_TTL_SECONDS):
                return []
    except (KeyError, TypeError, ValueError):
        return []
    return deepcopy(rows)


def _openbb(root, assets, previous):
    """Use documented V5 RSS in an isolated bounded process when already installed."""
    installed = importlib.util.find_spec('openbb') is not None and importlib.util.find_spec('openbb_news') is not None
    if not installed:
        return [], None, dict(status='not_installed', fallback='receipt-preserving stdlib RSS active',
            dependencies_reviewed=['openbb-core[pandas]', 'feedparser', 'lxml'])
    source = os.environ.get('WALLEX_OPENBB_RSS_SOURCE', 'yahoo_finance')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', source):
        raise ValueError('OpenBB source must be a registry key')
    started = utc_now()
    command = "import sys,json; from openbb import obb; r=obb.news.rss(source=sys.argv[1],limit=20,fetch_body=False); print(json.dumps([x.model_dump(mode='json') for x in r.results]))"
    completed = subprocess.run([sys.executable, '-c', command, source], capture_output=True, timeout=25,shell=False,
        **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))
    received = utc_now()
    raw = completed.stdout
    path = Path(root) / 'raw' / f'{time.time_ns()}_openbb_normalized.body'
    path.write_bytes(raw)
    receipt = dict(method='OpenBB SDK news.rss', url='https://docs.openbb.co/odp/python/reference/news/rss',
        request_started_utc=started, response_received_utc=received, authenticated=False,
        status=200 if completed.returncode == 0 else 500, sha256=hashlib.sha256(raw).hexdigest(), body_path=str(path),
        representation='OpenBB normalized output, not original HTTP feed bytes', source_key=source)
    save_json(path.with_suffix('.receipt.json'), receipt)
    if completed.returncode:
        raise ValueError('OpenBB RSS subprocess failed; normalized output receipt preserved')
    # Convert only the bounded SDK fields to RSS, then reuse the existing clock/sanitization/parser contract.
    import xml.etree.ElementTree as ET
    channel = ET.Element('channel')
    for row in json.loads(raw):
        item = ET.SubElement(channel, 'item')
        for key, name in (('title', 'title'), ('url', 'link'), ('excerpt', 'description'), ('date', 'pubDate')):
            ET.SubElement(item, name).text = str(row.get(key) or '')
    rss = ET.Element('rss'); rss.append(channel)
    news = parse_feed(ET.tostring(rss), receipt, previous)
    return news, receipt, dict(status='active', source_key=source, rows=len(news), fetch_body=False,
        receipt_scope='Normalized SDK output; direct publisher feeds below retain original HTTP bytes')


def collect_sources(root, assets, previous=None, asof=None):
    """Collect public facts/news; caller merges these arrays into its context packet."""
    root = Path(root)
    (root / 'raw').mkdir(parents=True, exist_ok=True)
    assets = {asset_name(x) for x in ([assets] if isinstance(assets, str) else assets)}
    if not all(re.fullmatch(r'[A-Z0-9]{1,20}', x) for x in assets):
        raise ValueError('Invalid requested asset')
    old = previous.get('news', previous) if isinstance(previous, dict) else (previous or [])
    seen = {x['id']: x for x in old} if isinstance(old, list) else old
    result = dict(news=[], fundamentals=[], sources=[], errors=[], integrations={})
    jobs = {name: ('rss', url) for name, url in FEEDS.items()}
    rsshub = os.environ.get('WALLEX_RSSHUB_FEED_URL')
    result['integrations']['rsshub'] = dict(status='unconfigured', reason='Set WALLEX_RSSHUB_FEED_URL to an existing trusted RSSHub route; no server deployed')
    if rsshub:
        parsed = urlsplit(rsshub)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('RSSHub URL must be an HTTP(S) feed without embedded credentials')
        jobs['rsshub'] = ('rss', rsshub)
    jobs['defillama'] = ('defillama', 'https://api.llama.fi/v2/chains')
    ids = [COINGECKO_IDS[x] for x in sorted(assets) if x in COINGECKO_IDS]
    result['integrations']['coingecko'] = dict(status='unconfigured' if not ids else 'pending', unmapped_assets=sorted(assets - COINGECKO_IDS.keys()),
        method='one bulk unauthenticated REST attempt', sdk='official Python/TypeScript SDK documented, not required for GET',
        mcp='Official keyless MCP documented at https://mcp.api.coingecko.com/mcp; not connected by this adapter')
    if ids:
        jobs['coingecko'] = ('coingecko', 'https://api.coingecko.com/api/v3/coins/markets?' + urlencode(dict(vs_currency='usd', ids=','.join(ids), per_page=250, page=1, sparkline='false')))
    for name, provider in (('coingecko', 'CoinGecko'), ('defillama', 'DefiLlama')):
        cached = _cached_fundamentals(previous if isinstance(previous, dict) else None, provider, assets, timestamp(utc_now()))
        if cached:
            jobs.pop(name, None)
            result['fundamentals'].extend(cached)
            receipts = {(row['source_receipt']['url'], row['source_receipt']['sha256']): row['source_receipt'] for row in cached}
            result['sources'].extend(receipts.values())
            result['integrations'][name] = {**result['integrations'].get(name, {}), 'status': 'active',
                'cache_hit': True, 'cache_ttl_seconds': FUNDAMENTAL_CACHE_TTL_SECONDS, 'rows': len(cached),
                'original_observed_at': min(row['observed_at'] for row in cached), 'clocks_preserved': True}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch_public, url, root, name, 8): (name, kind) for name, (kind, url) in jobs.items()}
        for future in as_completed(pending):
            name, kind = pending[future]
            raw, receipt = future.result()
            result['sources'].append(receipt)
            try:
                if receipt.get('error'):
                    raise ValueError(receipt['error'])
                if kind == 'rss':
                    rows = parse_feed(raw, receipt, seen)
                    for row in rows:
                        row['observed_at'] = receipt['response_received_utc']
                    result['news'].extend(rows)
                else:
                    rows = (coingecko_facts if kind == 'coingecko' else defillama_facts)(json.loads(raw), assets, receipt)
                    result['fundamentals'].extend(rows)
                result['integrations'][name] = {**result['integrations'].get(name, {}), 'status': 'active', 'http_status': receipt['status'], 'rows': len(rows), 'cache_hit': False}
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                result['errors'].append(dict(source=name, error=str(exc), receipt=receipt))
                result['integrations'][name] = {**result['integrations'].get(name, {}), 'status': 'unavailable', 'http_status': receipt.get('status'), 'error': str(exc)}
    try:
        news, receipt, status = _openbb(root, assets, seen)
        result['news'].extend(news)
        result['integrations']['openbb'] = status
        if receipt:
            result['sources'].append(receipt)
    except Exception as exc:
        result['integrations']['openbb'] = dict(status='unavailable', error=f'{type(exc).__name__}: {exc}')
        result['errors'].append(dict(source='openbb', error=str(exc)))
    cutoff = timestamp(asof or utc_now())
    result['news'] = list({x['id']: x for x in result['news'] if timestamp(x['available_at']) <= cutoff and
        (set(x['assets']) & assets or 'MACRO' in x['assets'] or 'USDT' in x['assets'])}.values())
    result['fundamentals'] = [x for x in result['fundamentals'] if timestamp(x['available_at']) <= cutoff]
    if isinstance(previous, dict):
        order = {row['id']: index for index, row in enumerate(previous.get('fundamentals', []))}
        result['fundamentals'].sort(key=lambda row: order.get(row['id'], len(order)))
    result['sources'] = [x for x in result['sources'] if timestamp(x['response_received_utc']) <= cutoff]
    result.update(observed_at=utc_now(), asof=cutoff.isoformat(), available_at=max((x['available_at'] for x in result['news'] + result['fundamentals']), default=None),
        status='partial' if result['errors'] else 'ok', quality=['Publisher text is untrusted bounded text; no article-body HTML fetched.',
            'Publication and metric update clocks are separate from actual receipt availability.',
            'Missing/null fundamental fields remain unavailable; no synthetic facts or directional claims.'])
    save_json(root / 'current.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/public-sources-20261021')
    parser.add_argument('--assets', nargs='+', default=['BTC', 'ETH', 'PEPE', 'XRP', 'USOON', 'SOL', 'XAUT', 'SUI', 'DOGE', 'S', 'BNB', 'ADA'])
    args = parser.parse_args()
    result = collect_sources(args.root, args.assets)
    print(json.dumps(dict(news=len(result['news']), fundamentals=len(result['fundamentals']), integrations=result['integrations'], errors=[dict(source=x['source'], error=x['error']) for x in result['errors']]), indent=2))
