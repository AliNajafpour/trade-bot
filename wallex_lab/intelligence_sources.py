"""Public council evidence with receipt-time availability and retained revisions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
from urllib.parse import quote, urlsplit
import xml.etree.ElementTree as ET

from .context import ASSETS, fetch_public, save_json, text_content, timestamp
from .data import utc_now


def _market(name, symbol, metric, unit, reference_type):
    return dict(name=name, parser='yahoo', provider='Yahoo Finance', symbol=symbol, metric=metric,
                unit=unit, reference_type=reference_type, frequency='provider snapshot; delay unverified',
                url=f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe="")}?interval=1d&range=5d')


def _fred(series, metric, unit, frequency):
    return dict(name=series.lower(), parser='fred', provider='FRED', series=series, metric=metric,
                unit=unit, frequency=frequency, reference_type='published macro observation',
                url=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}')


DEFAULT_SOURCES = {
    'market': [
        _market('gold', 'GC=F', 'gold_reference', 'USD/troy ounce', 'COMEX gold futures proxy; not spot gold'),
        _market('silver', 'SI=F', 'silver_reference', 'USD/troy ounce', 'COMEX silver futures proxy; not spot silver'),
        _market('sp500', '^GSPC', 'sp500_index', 'index points', 'S&P 500 index'),
        _market('nasdaq', '^IXIC', 'nasdaq_composite_index', 'index points', 'Nasdaq Composite index'),
        _market('dow', '^DJI', 'dow_jones_index', 'index points', 'Dow Jones Industrial Average index'),
        _market('volatility', '^VIX', 'vix_index', 'index points', 'CBOE volatility index'),
        _market('dollar', 'DX-Y.NYB', 'dollar_index', 'index points', 'ICE US Dollar Index'),
        _market('oil', 'CL=F', 'wti_reference', 'USD/barrel', 'NYMEX WTI crude oil futures proxy; not spot oil'),
        _market('bitcoin', 'BTC-USD', 'btc_reference', 'USD/BTC', 'Yahoo aggregated cryptocurrency reference'),
        _market('ethereum', 'ETH-USD', 'eth_reference', 'USD/ETH', 'Yahoo aggregated cryptocurrency reference'),
        dict(name='coindesk', parser='rss', provider='CoinDesk publisher', url='https://www.coindesk.com/arc/outboundfeeds/rss/'),
        dict(name='world_guardian', parser='rss', provider='The Guardian world publisher', url='https://www.theguardian.com/world/rss'),
        dict(name='world_cnbc', parser='rss', provider='CNBC world publisher', url='https://www.cnbc.com/id/100727362/device/rss/rss.html')],
    'macro': [
        _fred('CPIAUCSL', 'us_cpi_index', 'index 1982-1984=100, seasonally adjusted', 'monthly'),
        _fred('CPILFESL', 'us_core_cpi_index', 'index 1982-1984=100, seasonally adjusted', 'monthly'),
        _fred('FEDFUNDS', 'effective_federal_funds_rate', 'percent', 'monthly'),
        _fred('DFEDTARL', 'fed_target_lower', 'percent', 'daily'),
        _fred('DFEDTARU', 'fed_target_upper', 'percent', 'daily'),
        _fred('UNRATE', 'us_unemployment_rate', 'percent, seasonally adjusted', 'monthly'),
        _fred('DGS2', 'us_treasury_2y_yield', 'percent', 'business daily'),
        _fred('DGS10', 'us_treasury_10y_yield', 'percent', 'business daily'),
        _fred('WALCL', 'fed_total_assets', 'USD millions', 'weekly'),
        _fred('RRPONTSYD', 'overnight_reverse_repo', 'USD billions', 'business daily'),
        _fred('M2SL', 'us_m2_money_stock', 'USD billions, seasonally adjusted', 'monthly'),
        dict(name='fed_releases', parser='rss', provider='Federal Reserve official', url='https://www.federalreserve.gov/feeds/press_all.xml'),
        dict(name='bls_releases', parser='rss', provider='BLS official', url='https://www.bls.gov/feed/bls_latest.rss')]
}


def _identity(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


def _number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean observation')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite observation')
    return value


def _version(row, old, content):
    row['content_sha256'] = _identity(json.dumps(content, sort_keys=True, ensure_ascii=False, allow_nan=False))
    row['first_observed_at'] = old.get('first_observed_at', row['observed_at'])
    row['revisions'] = deepcopy(old.get('revisions', []))
    if old.get('content_sha256') == row['content_sha256']:
        row['available_at'] = old['available_at']
        row['first_source_receipt'] = old.get('first_source_receipt', old.get('source_receipt'))
    elif old:
        row['revisions'].append({key: old[key] for key in ('value', 'title', 'summary', 'published_at', 'available_at', 'observed_at', 'content_sha256', 'source_receipt', 'source_history', 'derivation', 'change_from_previous_observation') if key in old})
    return row


def _fact(spec, receipt, value, measurement):
    identity = _identity(spec['provider'], spec.get('series', spec.get('symbol')), spec['metric'], measurement)
    observed = receipt['response_received_utc']
    row = dict(id=identity, metric=spec['metric'], value=_number(value), unit=spec['unit'],
               source=spec['provider'], source_url=spec['url'], observed_at=observed, available_at=observed,
               frequency=spec['frequency'], reference_type=spec.get('reference_type'), source_receipt=receipt,
               quality=['Receipt time proves observation here, not original public release time.',
                        'Reference level alone establishes neither surprise nor directional price effect.'])
    if len(measurement) == 10:
        row['measurement_date'] = measurement
        row['quality'].append('Provider supplies an observation date only; no intraday quote/release timestamp asserted.')
    else:
        row['measurement_at'] = measurement
    return row


def _fred_rows(raw, spec, receipt, old):
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if not rows or spec['series'] not in rows[0] or not ({'observation_date', 'DATE'} & rows[0].keys()):
        raise ValueError('FRED CSV missing expected date/series columns')
    values = {}
    for row in rows:
        date = row.get('observation_date') or row.get('DATE')
        datetime.strptime(date, '%Y-%m-%d')
        if date > timestamp(receipt['response_received_utc']).date().isoformat():
            raise ValueError('Future FRED observation date')
        value = row[spec['series']]
        if value in ('', '.', 'NA'):
            continue
        value = _number(value)
        values[date] = value
    if not values:
        raise ValueError('FRED has no nonmissing observations')
    dates, result = sorted(values), []
    latest = dates[-1]
    for index, date in enumerate(dates):
        value = values[date]
        identity = _identity(spec['provider'], spec['series'], spec['metric'], date)
        if date != latest and identity not in old:
            continue
        row = _fact(spec, receipt, value, date)
        row['is_latest_observation'] = date == latest
        row['quality'].append('Published series level, potentially revised; calendar/frequency lag remains explicit.')
        row['source_history'] = [dict(measurement_date=d, value=values[d], unit=spec['unit']) for d in dates[max(0, index - 13):index + 1]]
        row['quality'].append('History is the vintage returned at this receipt; measurement dates do not establish historical availability.')
        if index:
            previous = dates[index - 1]
            change = value - values[previous]
            change_unit = spec['unit']
            if spec['series'] in ('DGS2', 'DGS10', 'DFEDTARL', 'DFEDTARU', 'FEDFUNDS'):
                change, change_unit = change * 100, 'basis points'
            elif spec['series'] == 'UNRATE':
                change_unit = 'percentage points'
            row['change_from_previous_observation'] = dict(value=change, unit=change_unit,
                from_date=previous, from_value=values[previous], to_date=date, to_value=value,
                comparison='previous nonmissing observation; dates are explicit')
            if values[previous] != 0:
                row['change_from_previous_observation']['percent_change'] = 100 * (value / values[previous] - 1)
        if spec['series'] in ('DFEDTARL', 'DFEDTARU'):
            different = next((d for d in reversed(dates[:index]) if values[d] != value), None)
            row['current_setting_first_measurement_date'] = dates[dates.index(different) + 1] if different else dates[0]
            if different:
                row['previous_different_observation'] = dict(measurement_date=different, value=values[different], unit=spec['unit'])
                row['change_from_previous_setting_basis_points'] = 100 * (value - values[different])
            row['quality'].append('Policy setting dates are FRED observation dates, not inferred announcement timestamps.')
        _version(row, old.get(identity, {}), [value, date, spec['unit'], row['source_history'],
            row.get('previous_different_observation'), row.get('current_setting_first_measurement_date')])
        result.append(row)
        if spec['series'] in ('CPIAUCSL', 'CPILFESL'):
            measured = datetime.strptime(date, '%Y-%m-%d')
            for lag, label in ((1, 'mom'), (12, 'yoy')):
                year, month = divmod(measured.year * 12 + measured.month - 1 - lag, 12)
                try:
                    lag_date = measured.replace(year=year, month=month + 1).date().isoformat()
                except ValueError:
                    continue
                if lag_date not in values or values[lag_date] <= 0:
                    continue
                rate_spec = {**spec, 'metric': spec['metric'] + f'_{label}_percent_change',
                    'unit': 'percent change, ' + ('month over month' if lag == 1 else 'year over year')}
                rate = _fact(rate_spec, receipt, 100 * (value / values[lag_date] - 1), date)
                rate['is_latest_observation'] = date == latest
                rate['reference_type'] = 'derived change in the cited CPI index; not an index level or release surprise'
                rate['derivation'] = dict(formula='100 * (current_index / lag_index - 1)', lag_months=lag,
                    inputs=[dict(measurement_date=lag_date, value=values[lag_date], unit=spec['unit']),
                            dict(measurement_date=date, value=value, unit=spec['unit'])])
                rate['quality'].append('Exact calendar lags from this source vintage; not annualized; seasonal adjustment matches the input index.')
                _version(rate, old.get(rate['id'], {}), [rate['value'], date, rate['unit'], rate['derivation']])
                result.append(rate)
    return result


def _yahoo_rows(raw, spec, receipt, old):
    chart = json.loads(raw)['chart']
    if chart.get('error') or not chart.get('result'):
        raise ValueError('Yahoo chart returned no result')
    result = chart['result'][0]
    meta = result['meta']
    if meta['symbol'] != spec['symbol'] or meta.get('currency') != 'USD':
        raise ValueError('Yahoo reference symbol/currency mismatch')
    measured = datetime.fromtimestamp(_number(meta['regularMarketTime']), timezone.utc).isoformat()
    if timestamp(measured) > timestamp(receipt['response_received_utc']):
        raise ValueError('Future Yahoo quote timestamp')
    price = _number(meta['regularMarketPrice'])
    if price <= 0:
        raise ValueError('Nonpositive reference price')
    row = _fact(spec, receipt, price, measured)
    row.update(symbol=spec['symbol'], exchange_timezone=meta.get('exchangeTimezoneName'),
               instrument_type=meta.get('instrumentType'), exchange=meta.get('fullExchangeName') or meta.get('exchangeName'),
               provider_delay_seconds=meta.get('exchangeDataDelayedBy'),
               quote_age_seconds=(timestamp(receipt['response_received_utc']) - timestamp(measured)).total_seconds(),
               market_state=meta.get('marketState', 'not supplied; may be closed or delayed'),
               trading_period=meta.get('currentTradingPeriod'))
    quote_rows = result.get('indicators', {}).get('quote', [{}])[0]
    volumes = quote_rows.get('volume', [])
    if volumes and volumes[-1] is not None:
        row['volume'] = _number(volumes[-1])
        if row['volume'] < 0:
            raise ValueError('Negative quote volume')
        row['volume_scope'] = 'last returned daily bar; not a fixed-time live flow'
    row['quality'].append('Provider snapshot is potentially delayed or a closed-market last quote; never asserted realtime.')
    # chartPreviousClose is the chart-range anchor, which need not be the previous session close.
    previous_key = next((key for key in ('regularMarketPreviousClose', 'previousClose') if meta.get(key) is not None), None)
    if previous_key:
        previous = _number(meta[previous_key])
        if previous <= 0:
            raise ValueError('Nonpositive provider previous close')
        row.update(previous_close=previous, previous_close_provider_field=previous_key,
            previous_close_measurement_at=None, return_from_previous_close_percent=100 * (price / previous - 1))
        row['quality'].append('Previous close is an explicit provider field; its observation timestamp is not supplied.')
    _version(row, old.get(row['id'], {}), [price, measured, row['unit'], row.get('previous_close'), row.get('previous_close_provider_field')])
    return [row]


def _news_rows(raw, spec, receipt, old):
    root = ET.fromstring(raw)
    items = root.findall('./channel/item') or root.findall('{http://www.w3.org/2005/Atom}entry')
    if not items and root.tag not in ('rss', '{http://www.w3.org/2005/Atom}feed'):
        raise ValueError('Source is not RSS/Atom')
    result = []
    for node in items[:100]:
        fields = {child.tag.split('}')[-1]: child.text or '' for child in node}
        url = fields.get('link', '')
        for child in node:
            if child.tag.endswith('}link') and child.get('rel', 'alternate') == 'alternate':
                url = child.get('href', url)
                break
        if not url.startswith(('http://', 'https://')):
            continue
        url = url.split('#', 1)[0]
        title = text_content(fields.get('title'))[:500]
        summary = text_content(fields.get('description') or fields.get('summary') or fields.get('content'))[:2000]
        if not title:
            continue
        published, quality = None, []
        date = fields.get('pubDate') or fields.get('published') or fields.get('updated')
        if date:
            try:
                try:
                    parsed = timestamp(date)
                except ValueError:
                    parsed = parsedate_to_datetime(date)
                    if parsed.tzinfo is None:
                        raise ValueError('Missing publisher timezone')
                published = parsed.astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError, OverflowError):
                quality.append('invalid_published_time')
        else:
            quality.append('missing_published_time')
        identity = _identity(url)
        observed = receipt['response_received_utc']
        assets = [asset for asset, pattern in ASSETS.items() if re.search(pattern, title + ' ' + summary, re.I)] or ['GLOBAL']
        row = dict(id=identity, title=title, summary=summary, url=url, source_url=url, source=spec['provider'],
                   assets=assets, published_at=published, observed_at=observed, available_at=observed,
                   source_receipt=receipt, quality=quality)
        _version(row, old.get(identity, {}), [title, summary, published])
        if published and timestamp(published) > timestamp(row['available_at']):
            row['available_at'] = published
            row['quality'].append('future_publication_time')
        result.append(row)
    return result


def collect_bundle(root, kind, asof, previous=None, config=None):
    """Collect current public observations; asof=None uses the actual completion cutoff.

    Config can override ``market.sources``/``macro.sources`` with the documented
    source dictionaries, and timeout_seconds/max_workers. Failed coverage is data.
    """
    if kind not in DEFAULT_SOURCES:
        raise ValueError('Council evidence kind must be market or macro')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    current_path = root / f'{kind}.json'
    if previous is None and current_path.exists():
        previous = json.loads(current_path.read_text(encoding='utf-8'))
    previous = previous or {}
    old = {row['id']: row for row in previous.get('observations', previous.get('facts', []) + previous.get('news', []))}
    config = config or {}
    settings = config.get(kind, {})
    specs = deepcopy(settings.get('sources', DEFAULT_SOURCES[kind]))
    timeout = float(config.get('timeout_seconds', 12))
    workers = int(config.get('max_workers', 4))
    if not 0 < timeout <= 60 or not 1 <= workers <= 8:
        raise ValueError('Bounded timeout/workers required')
    names = set()
    for spec in specs:
        parsed = urlsplit(spec['url'])
        if parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Source URL must be public HTTP(S) without embedded credentials')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', spec['name']) or spec['name'] in names:
            raise ValueError('Source names must be unique bounded filename-safe keys')
        if spec['parser'] not in ('fred', 'yahoo', 'rss'):
            raise ValueError('Unsupported public source parser')
        names.add(spec['name'])
    facts, news, sources, coverage, observations = [], [], [], {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch_public, spec['url'], root, spec['name'], timeout): spec for spec in specs}
        for future in as_completed(jobs):
            spec = jobs[future]
            entry = dict(status='unavailable', rows=0, url=spec['url'], provider=spec['provider'])
            coverage[spec['name']] = entry
            try:
                raw, receipt = future.result()
                sources.append({**receipt, 'source_name': spec['name']})
                entry['http_status'] = receipt.get('status')
                if receipt.get('error') or receipt.get('status') != 200:
                    raise ValueError(receipt.get('error', 'Non-200 response'))
                if timestamp(receipt['response_received_utc']) < timestamp(receipt['request_started_utc']):
                    raise ValueError('Backwards source receipt clock')
                rows = {'fred': _fred_rows, 'yahoo': _yahoo_rows, 'rss': _news_rows}[spec['parser']](raw, spec, receipt, old)
                if not rows:
                    raise ValueError('No usable observations')
                observations.extend(rows)
                target = news if spec['parser'] == 'rss' else facts
                current = [r for r in rows if r.get('is_latest_observation', True)]
                target.extend(current)
                entry.update(status='available', rows=len(current), frequency=spec.get('frequency'), reference_type=spec.get('reference_type'))
                dates = [r.get('measurement_at', r.get('measurement_date')) for r in rows if r.get('measurement_at', r.get('measurement_date'))]
                if dates:
                    entry['latest_measurement_at'] = max(dates)
            except Exception as exc:
                entry['error'] = f'{type(exc).__name__}: {exc}'
    cutoff = timestamp(asof or utc_now())
    known = lambda row: timestamp(row['available_at']) <= cutoff
    facts, news = [r for r in facts if known(r)], [r for r in news if known(r)]
    for entry in coverage.values():
        if entry['status'] == 'available':
            entry['rows'] = sum(r['source_receipt']['url'] == entry['url'] for r in facts + news)
            if not entry['rows']:
                entry['status'] = 'unavailable_asof'
    successful = sum(entry['status'] == 'available' for entry in coverage.values())
    status = 'complete' if successful == len(specs) and specs else ('partial' if successful else 'unavailable')
    bundle = dict(schema_version=1, kind=kind, asof=cutoff.isoformat(), observed_at=utc_now(),
                  available_at=max((r['available_at'] for r in facts + news), default=None),
                  facts=sorted(facts, key=lambda r: r['metric']), news=sorted({r['id']: r for r in news}.values(), key=lambda r: r['available_at'], reverse=True),
                  sources=sources, coverage=dict(status=status, requested_sources=len(specs), successful_sources=successful,
                  sources=coverage, limitations=['Public endpoints have no availability SLA; failures are explicit.',
                  'Commodity futures are labeled proxies; market snapshots may be delayed or closed.',
                  'Observation dates and publisher times never establish historical system availability.']),
                  observations=list({**old, **{r['id']: r for r in observations}}.values()))
    # Retain immutable bundles even when every source failed; current is only a pointer-like convenience.
    identity = _identity(json.dumps(bundle, sort_keys=True, ensure_ascii=False, allow_nan=False))
    history = root / 'bundles'
    history.mkdir(exist_ok=True)
    save_json(history / f'{kind}-{identity}.json', bundle)
    save_json(current_path, bundle)
    return bundle


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/council-sources')
    parser.add_argument('--kind', choices=['market', 'macro'], required=True)
    parser.add_argument('--asof')
    parser.add_argument('--config')
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else None
    bundle = collect_bundle(args.root, args.kind, args.asof, config=settings)
    print(json.dumps(dict(kind=args.kind, facts=len(bundle['facts']), news=len(bundle['news']), coverage=bundle['coverage']), indent=2))
