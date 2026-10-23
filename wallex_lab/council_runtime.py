"""Independent public-data, report, scanner, manager and protective paper loops."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import sys
import math
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from .context import fetch_public, save_json, timestamp
from .data import load_market, normalize_history, utc_now
from .spot import canonical_symbol


DEFAULTS = dict(mode='paper', run_dir='../runs/market-council-20261022', initial_usdt=100,
                symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT'],
                quote_seconds=10, technical_seconds=60, manager_seconds=60,
                market_seconds=1800, macro_seconds=86400, foundation_seconds=900, reference_history_days=7, reference_venue='auto', port=8784,
                technical_model_dir='../runs/council-technical-20261022',
                technical_gap_fallback_model_dir=None,
                management_model_dir=None,
                management_prescore_options=True, model_exit_confirmation=True, management_comparison=True,
                coherent_management_policy=False,
                adaptive_learning=False, adaptive_settings={}, strategy_bank=False,
                manager_reasoning_effort=None, manager_chart_limit=2,
                manager_opportunity_gate=True,
                automatic_model_entries=False, model_entry_risk_multiplier=.5,
                technical_python=str(Path(sys.executable).resolve()),
                source_config={}, risk={}, provider='environment', require_reports=False,
                plan_lifetime_seconds=21600, max_signal_age_seconds=180,
                manager_timeout_seconds=120, technical_timeout_seconds=120,
                max_rescore_price_drift=.0005, minimum_expected_net_return=0, model_comparison=True)


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8'))
    json.dumps(raw, allow_nan=False)
    if not isinstance(raw, dict) or raw.keys() - DEFAULTS.keys():
        raise ValueError('Unknown council configuration fields; credentials belong in environment variables')
    config = deepcopy(DEFAULTS)
    config.update(raw)
    if config['mode'] != 'paper':
        raise ValueError('Council permits paper spot mode only')
    if config['reference_venue'] not in ('auto', 'wallex', 'kucoin', 'mexc'):
        raise ValueError('reference_venue must be auto, wallex, kucoin or mexc')
    symbols = config['symbols']
    if not isinstance(symbols, list) or not 1 <= len(symbols) <= 4:
        raise ValueError('Use 1..4 liquid USDT spot reference symbols')
    config['symbols'] = [canonical_symbol(symbol) for symbol in symbols]
    allowed = {'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT'}
    if len(set(config['symbols'])) != len(symbols) or set(config['symbols']) - allowed:
        raise ValueError('Distinct BTC/ETH/SOL/XRP USDT spot symbols required')
    for key, low, high in [('initial_usdt', .01, 1e9), ('quote_seconds', 5, 30),
                           ('technical_seconds', 10, 86400), ('manager_seconds', 10, 86400),
                           ('market_seconds', 60, 604800), ('macro_seconds', 60, 2592000),
                           ('foundation_seconds', 1, 86400),
                           ('plan_lifetime_seconds', 60, 604800), ('max_signal_age_seconds', 1, 3600),
                           ('manager_timeout_seconds', 1, 120), ('technical_timeout_seconds', 1, 300),
                           ('max_rescore_price_drift', 0, .1), ('minimum_expected_net_return', 0, 1)]:
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid ' + key)
    if type(config['port']) is not int or not 0 <= config['port'] <= 65535:
        raise ValueError('port must be 0..65535; 0 disables local HTTP')
    if type(config['reference_history_days']) is not int or not 3 <= config['reference_history_days'] <= 30:
        raise ValueError('reference_history_days must be an integer from 3 to 30')
    if not isinstance(config['source_config'], dict) or not isinstance(config['risk'], dict):
        raise ValueError('source_config and risk must be objects')
    if type(config['require_reports']) is not bool:
        raise ValueError('require_reports must be a boolean')
    if type(config['model_comparison']) is not bool:
        raise ValueError('model_comparison must be a boolean')
    if type(config['automatic_model_entries']) is not bool:
        raise ValueError('automatic_model_entries must be a boolean')
    for key in ('management_prescore_options','model_exit_confirmation','management_comparison','manager_opportunity_gate','coherent_management_policy','adaptive_learning','strategy_bank'):
        if type(config[key]) is not bool: raise ValueError(key+' must be a boolean')
    if not isinstance(config['adaptive_settings'],dict):raise ValueError('adaptive_settings must be an object')
    if config['adaptive_learning'] and not config['coherent_management_policy']:
        raise ValueError('Adaptive learning requires the shared coherent entry/holding model')
    if config['strategy_bank'] and not config['coherent_management_policy']:
        raise ValueError('Strategy bank requires the shared coherent candidate model')
    if type(config['manager_chart_limit']) is not int or not 1 <= config['manager_chart_limit'] <= 4:
        raise ValueError('manager_chart_limit must be1..4')
    from .llm import validate_reasoning_effort
    validate_reasoning_effort(config['manager_reasoning_effort'])
    if type(config['model_entry_risk_multiplier']) not in (int, float) or not math.isfinite(config['model_entry_risk_multiplier']) or not 0 < config['model_entry_risk_multiplier'] <= 1:
        raise ValueError('model_entry_risk_multiplier must be in (0, 1]')
    from .spot import _number, _position_limit, _spot_only
    risk = config['risk']
    _spot_only(risk)
    _position_limit(risk.get('max_open_positions'))
    for key in ('max_position_fraction', 'risk_per_trade_fraction', 'max_account_stop_risk_fraction'):
        if key in risk:
            _number(risk[key], key, 0, 1)
    for key in ('min_notional', 'minimum_net_rr', 'max_spread_bps', 'slippage_bps'):
        if key in risk:
            _number(risk[key], key, 0, 9999 if key == 'slippage_bps' else None)
    if config['provider'] != 'environment':
        raise ValueError('provider must be environment; configure LLM_PROVIDER in the process environment')
    for key in ('run_dir', 'technical_model_dir', 'technical_python'):
        if not isinstance(config[key], str) or not config[key].strip() or '\n' in config[key]:
            raise ValueError('Explicit ' + key + ' path required')
        config[key] = str((path.parent / config[key]).resolve())
    for key in ('technical_gap_fallback_model_dir', 'management_model_dir'):
        if config[key] is not None:
            value = config[key]
            if not isinstance(value, str) or not value.strip() or '\n' in value:
                raise ValueError(key + ' must be a valid model path or null')
            config[key] = str((path.parent / value).resolve())
    if config['coherent_management_policy'] and not config['management_model_dir']:
        raise ValueError('Coherent entry/hold policy requires management_model_dir')
    return config


def fetch_quote(config, symbol):
    from .shadow import validate_quote
    primary_error = None
    source = 'official Wallex public MCP'
    try:
        from .mcp_client import MCPPublicClient
        client = MCPPublicClient(Path(config['run_dir']) / 'quotes' / symbol.replace('/', ''), timeout=8)
        payload, receipt = client.get('/v1/depth', {'symbol': symbol.replace('/', '')}, 'council_book')
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        primary_error = type(error).__name__
        directory = Path(config['run_dir'])/'quotes'/symbol.replace('/', '')
        raw, receipt = fetch_public('https://api.wallex.ir/v1/depth?'+urlencode({'symbol':symbol.replace('/','')}),
                                    directory, 'council_book_rest_fallback', 8)
        payload = json.loads(raw)
        source = 'official Wallex public REST fallback'
    quote = validate_quote(payload, receipt, utc_now())
    quote.update(available_at=quote['observed_utc'], source_receipt=receipt,
                 source=source, primary_error=primary_error, symbol=symbol, quote_currency='USDT')
    return quote


def _merge(old, new):
    import pandas as pd
    if old is None:
        return new
    # The last completed minute may be revised by the source; immutable raw receipts retain both versions.
    return pd.concat([old, new]).drop_duplicates('timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)


def _write_frame(path, frame):
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _merge_native_gap_bars(reference, native):
    """Fill only absent global-reference minutes with observed Wallex OHLCV and report the mix."""
    if native is None or native.empty or reference.empty:
        return reference, dict(source='Wallex native M1', recovered_minutes=0, timestamps=[])
    import pandas as pd
    start, end = reference.timestamp.iloc[0], reference.timestamp.iloc[-1]
    expected = set(pd.date_range(start, end, freq='min'))
    observed = set(reference.timestamp)
    missing = expected - observed
    recovered = native[native.timestamp.isin(missing)].copy()
    merged = _merge(reference, recovered) if not recovered.empty else reference
    return merged, dict(source='Wallex native M1', recovered_minutes=len(recovered),
                        timestamps=[value.isoformat() for value in recovered.timestamp])


_VENUE_LOCK = threading.Lock()


def _reference_quality(frame, cutoff, history_days):
    start = cutoff - history_days * 86400
    observed = {int(value.timestamp()) for value in frame.timestamp if start <= value.timestamp() < cutoff}
    missing = sorted(set(range(start, cutoff, 60)) - observed)
    windows = []
    for minute in missing:
        if windows and minute == windows[-1]['to_exclusive']:
            windows[-1]['to_exclusive'] += 60
            windows[-1]['missing_minutes'] += 1
        else:
            windows.append(dict(start=minute, to_exclusive=minute + 60, missing_minutes=1))
    expected = history_days * 1440
    return dict(rows=len(observed), expected_minutes=expected, missing_minutes=len(missing),
                history_days=history_days, coverage_ratio=len(observed) / expected, interpolated=False,
                missing_timestamps=[datetime.fromtimestamp(value, timezone.utc).isoformat() for value in missing],
                gaps=windows)


def _reference(config, directory, symbol, cutoff, tail_only=False):
    """Refresh observed candles, then revisit bounded missing windows on that same venue."""
    (directory / 'reference').mkdir(parents=True, exist_ok=True)
    preferred_path = Path(config['run_dir']) / 'reference-venue.json'
    with _VENUE_LOCK:
        preferred = json.loads(preferred_path.read_text(encoding='utf-8')) if preferred_path.exists() else {}
    # A local execution venue is a fallback, not the trained global candle basis.
    requested=config.get('reference_venue')
    venues = [requested] if requested in ('wallex','kucoin','mexc') else ['kucoin', 'binance-data', 'binance', 'wallex']
    preference_current = preferred.get('venue') in venues[:-1] and 0 <= time.time() - preferred.get('tested_at', 0) < 86400
    if preference_current:
        venues.remove(preferred['venue'])
        venues.insert(0, preferred['venue'])
    failures, receipts = [], []
    history_days = config.get('reference_history_days', DEFAULTS['reference_history_days'])
    history_start = cutoff - history_days * 86400
    for venue in venues:
        path = directory / (venue + '-cache.csv')
        frame = load_market(path, 60) if path.exists() else None
        start = int(frame.timestamp.iloc[-1].timestamp()) - 60 if frame is not None and len(frame) else history_start
        if tail_only: start = max(start, cutoff-180)

        def page(begin, end, label='1m'):
            if venue == 'wallex':
                query = urlencode(dict(symbol=symbol.replace('/', ''), resolution='1', **{'from': begin, 'to': end - 1}))
                url = 'https://api.wallex.ir/v1/udf/history?' + query
            elif venue == 'kucoin':
                query = urlencode(dict(symbol=symbol.replace('/', '-'), type='1min', startAt=begin, endAt=end - 1))
                url = 'https://api.kucoin.com/api/v1/market/candles?' + query
            else:
                query = urlencode(dict(symbol=symbol.replace('/', ''), interval='1m', startTime=begin * 1000,
                                       endTime=end * 1000 - 1, limit=500 if venue=='mexc' else 1000))
                host = 'https://api.mexc.com' if venue=='mexc' else 'https://data-api.binance.vision' if venue == 'binance-data' else 'https://api.binance.com'
                url = host + '/api/v3/klines?' + query
            raw, receipt = fetch_public(url, directory / 'reference', venue + '_' + label, 12)
            receipts.append(receipt)
            if receipt.get('error') or receipt.get('status') != 200:
                raise ValueError(receipt.get('error', 'HTTP ' + str(receipt.get('status'))))
            response = json.loads(raw)
            if venue == 'wallex':
                if not isinstance(response, dict) or response.get('s') != 'ok':
                    raise ValueError('Wallex public candles failed')
                payload = response
            elif venue == 'kucoin':
                if not isinstance(response, dict) or response.get('code') != '200000':
                    raise ValueError('KuCoin public candles failed')
                rows = response.get('data')
            else:
                rows = response
            if venue != 'wallex':
                if not isinstance(rows, list):
                    raise ValueError('Malformed reference candles')
                payload = dict(s='ok', t=[int(row[0]) if venue == 'kucoin' else row[0] / 1000 for row in rows],
                    o=[row[1] for row in rows], h=[row[3] if venue == 'kucoin' else row[2] for row in rows],
                    l=[row[4] if venue == 'kucoin' else row[3] for row in rows],
                    c=[row[2] if venue == 'kucoin' else row[4] for row in rows], v=[row[5] for row in rows])
            new, _ = normalize_history(payload, 60, min(cutoff, timestamp(receipt['request_started_utc']).timestamp()))
            return new.loc[new.timestamp.map(lambda value: begin <= value.timestamp() < end).astype(bool)]

        try:
            width = 86400 if venue == 'wallex' else (1500 if venue == 'kucoin' else 500 if venue=='mexc' else 1000) * 60
            while start < cutoff:
                end = min(cutoff, start + width)
                frame = _merge(frame, page(start, end))
                # Time pagination also advances across valid empty (no published candles) windows.
                start = end
            if frame is None or frame.empty or cutoff - frame.timestamp.iloc[-1].timestamp() - 60 > 120:
                raise ValueError('Reference completed-minute data is stale')
            # A longer configured lookback must also extend already-warm caches.
            older_end = int(frame.timestamp.iloc[0].timestamp())
            older_start = history_start
            history_attempts = []
            while older_start < older_end and not tail_only:
                end = min(older_end, older_start + width)
                attempt = dict(start=older_start, to_exclusive=end)
                receipt_count = len(receipts)
                try:
                    new = page(older_start, end, 'history_backfill')
                    frame = _merge(frame, new)
                    attempt.update(status='ok', returned_rows=len(new))
                except Exception as error:
                    # Older-history failure must not discard otherwise fresh observed market data.
                    attempt.update(status='request_failed', error=f'{type(error).__name__}: {error}'[:1500])
                attempt['receipts'] = receipts[receipt_count:]
                history_attempts.append(attempt)
                older_start = end
            before = _reference_quality(frame, cutoff, history_days)
            quality_path = directory / 'reference-quality.json'
            previous = json.loads(quality_path.read_text(encoding='utf-8')) if quality_path.exists() else {}
            windows = [dict(start=begin, to_exclusive=min(gap['to_exclusive'], begin + width - 120),
                            missing_minutes=(min(gap['to_exclusive'], begin + width - 120) - begin) // 60)
                       for gap in before['gaps'] for begin in range(gap['start'], gap['to_exclusive'], width - 120)]
            cursor = previous.get('next_gap_window', 0) % max(1, len(windows))
            ordered = windows[cursor:] + windows[:cursor]
            attempted = []
            # ponytail: two historical pages per refresh keep live tails responsive; rotate the remainder.
            for window in ordered[:0 if tail_only else 2]:
                begin = max(history_start, window['start'] - 60)
                end = min(cutoff, window['to_exclusive'] + 60, begin + width)
                attempt = dict(start=begin, to_exclusive=end, missing_before=window['missing_minutes'])
                receipt_count = len(receipts)
                try:
                    new = page(begin, end, 'gap_retry')
                    frame = _merge(frame, new)
                    observed = {int(value.timestamp()) for value in frame.timestamp}
                    remaining = sum(value not in observed for value in range(window['start'], min(end, window['to_exclusive']), 60))
                    attempt.update(status='observed_missing' if remaining else 'recovered', returned_rows=len(new), missing_after=remaining)
                except Exception as error:
                    attempt.update(status='request_failed', error=f'{type(error).__name__}: {error}'[:1500])
                attempt['receipts'] = receipts[receipt_count:]
                attempted.append(attempt)
            after = _reference_quality(frame, cutoff, history_days)
            quality = dict(symbol=symbol, venue=venue, available_at=utc_now(), before=before, after=after,
                attempted_windows=attempted, history_backfill=history_attempts, next_gap_window=cursor + len(attempted),
                missing_semantics='Unpublished or unavailable buckets; absent bars and request failures do not establish no trades.',
                venue_documentation=('https://api-docs.wallex.ir/' if venue == 'wallex' else
                    'https://www.kucoin.com/docs-new/rest/spot-trading/market-data/get-klines' if venue == 'kucoin'
                    else 'https://www.mexc.com/api-docs/spot-v3/market-data-endpoints' if venue=='mexc'
                    else 'https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints'))
            save_json(quality_path, quality)
            # Retain every retry audit independently of mutable quality summaries.
            save_json(directory / 'reference' / (str(time.time_ns()) + '_gap_quality.json'), quality)
            _write_frame(path, frame)
            tested_at = preferred['tested_at'] if preference_current and venue == preferred['venue'] else time.time()
            with _VENUE_LOCK:
                save_json(preferred_path, dict(venue=venue, tested_at=tested_at))
            source = ('Wallex official public UDF M1' if venue == 'wallex' else
                      'KuCoin public spot REST /api/v1/market/candles' if venue == 'kucoin'
                      else 'MEXC public spot REST /api/v3/klines' if venue=='mexc'
                      else 'Binance public spot REST /api/v3/klines')
            return frame, receipts, source
        except Exception as error:
            failures.append(f'{venue}: {type(error).__name__}: {error}'[:1500])
    raise ValueError('; '.join(failures))


def collect_markets(config):
    """Cache real completed global M1 references and retain distinct Wallex native candles."""
    from .mcp_client import MCPPublicClient
    result, errors = {}, {}
    for symbol in config['symbols']:
        compact = symbol.replace('/', '')
        directory = Path(config['run_dir']) / 'market-data' / compact
        directory.mkdir(parents=True, exist_ok=True)
        cutoff = int(time.time()) // 60 * 60
        receipts, native_error = [], None
        try:
            reference, receipts, reference_source = _reference(config, directory, symbol, cutoff)
            reference = reference[reference.timestamp.map(lambda value: value.timestamp() + 60 <= cutoff)]
            if reference.empty or cutoff - reference.timestamp.iloc[-1].timestamp() - 60 > 120:
                raise ValueError('Reference completed-minute data is stale')
            snapshot = directory / 'snapshots' / str(time.time_ns())
            snapshot.mkdir(parents=True)
            # Keep native history separate, then use it only for timestamps absent from the reference.
            native_path = directory / 'wallex-native.csv'
            native = load_market(native_path, 60) if native_path.exists() else None
            native_start = (int(native.timestamp.iloc[-1].timestamp()) - 60
                            if native is not None and len(native) else cutoff - 86400)
            try:
                client = MCPPublicClient(directory / 'wallex', timeout=12)
                payload, receipt = client.get('/v1/udf/history',
                    {'symbol': compact, 'resolution': '1', 'from': native_start, 'to': cutoff - 1}, compact + '_native')
                frame, quality = normalize_history(payload, 60, min(cutoff, timestamp(receipt['request_started_utc']).timestamp()))
                native = _merge(native, frame)
                _write_frame(native_path, native)
                save_json(snapshot / 'wallex-quality.json', dict(quality=quality, receipt=receipt,
                          rows=len(native), zero_volume_rows=int((native.volume == 0).sum())))
            except Exception as error:
                native_error = f'{type(error).__name__}: {error}'[:1500]
            reference, gap_repair = _merge_native_gap_bars(reference, native)
            quality = json.loads((directory / 'reference-quality.json').read_text(encoding='utf-8'))
            cached_repairs = []
            repair_venues = ('wallex',) if config.get('reference_venue') == 'wallex' else ('binance-data', 'binance', 'kucoin', 'wallex')
            for venue in repair_venues:
                cache = directory / (venue + '-cache.csv')
                if venue == quality.get('venue') or not cache.exists():
                    continue
                try:
                    reference, repair = _merge_native_gap_bars(reference, load_market(cache, 60))
                    if repair['recovered_minutes']:
                        with cache.open('rb') as handle:
                            checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
                        cached_repairs.append(dict(repair, source=venue+' cached observed M1',
                            cache_path=str(cache.resolve()), cache_sha256=checksum))
                except (OSError, ValueError) as error:
                    cached_repairs.append(dict(source=venue, error=f'{type(error).__name__}: {error}'[:1500]))
            # Historical repair/native fetch can cross a minute boundary: refresh the tail last.
            current_cutoff = int(time.time())//60*60
            tail_refresh = None
            if current_cutoff > cutoff or reference.timestamp.iloc[-1].timestamp()+60 < current_cutoff:
                tail, tail_receipts, tail_source = _reference(config,directory,symbol,current_cutoff,tail_only=True)
                tail = tail.loc[tail.timestamp.map(lambda value:value.timestamp() >= cutoff-60)]
                if tail_source != reference_source:
                    tail = tail.loc[~tail.timestamp.isin(reference.timestamp)]
                reference = _merge(reference,tail)
                receipts.extend(tail_receipts)
                tail_refresh = dict(source=tail_source,completed_before=datetime.fromtimestamp(current_cutoff,timezone.utc).isoformat(),
                                    timestamps=[value.isoformat() for value in tail.timestamp])
                cutoff = current_cutoff
            if time.time()-reference.timestamp.iloc[-1].timestamp()-60 > 120:
                raise ValueError('Reference completed-minute data still stale after final tail refresh')
            quality['after_cache_repair'] = _reference_quality(reference, cutoff, config.get('reference_history_days', DEFAULTS['reference_history_days']))
            _write_frame(snapshot / 'reference.csv', reference)
            gaps = int((reference.timestamp.diff().dt.total_seconds().dropna() != 60).sum())
            metadata = dict(source=reference_source, symbol=symbol, quality=quality,
                            quote_currency='USDT', interval_seconds=60, timestamp_semantics='UTC candle open',
                            completed_before=datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
                            available_at=utc_now(), rows=len(reference), gap_count=gaps,
                            receipts=receipts, native_candles_path=str(native_path.resolve()),
                            native_error=native_error, gap_repair=gap_repair, cached_gap_repairs=cached_repairs,tail_refresh=tail_refresh,
                            basis=('Wallex-only observed OHLCV; no foreign venue repair or invented missing candles.' if config.get('reference_venue') == 'wallex' else
                                   'Primary observed candles preserved; absent minutes filled from observed native/cached venue bars. Mixed venue price/volume basis is recorded, not corrected.'))
            save_json(snapshot / 'provenance.json', metadata)
            result[symbol] = dict(candles_path=str((snapshot / 'reference.csv').resolve()), quote=None,
                                  reference_source=metadata)
        except Exception as error:
            errors[symbol] = f'{type(error).__name__}: {error}'[:1500]
    if not result:
        raise ValueError('All reference markets failed: ' + json.dumps(errors))
    return dict(markets=result, errors=errors)


def _bounded(value, depth=0, tail=False):
    """Keep local status responses bounded while retaining actual source fields."""
    if depth > 8:
        return None
    if isinstance(value, dict):
        return {key: _bounded(item, depth + 1, tail) for key, item in list(value.items())[:64]}
    if isinstance(value, list):
        return [_bounded(item, depth + 1, tail) for item in (value[-20:] if tail else value[:24])]
    return value[:5000] if isinstance(value, str) else value


class Runtime:
    """Daemon workers do slow work; only the main loop schedules and consumes results."""
    def __init__(self, config_path, council, store=None, quote_fetcher=None,
                 market_fetcher=None, bundle_collector=None):
        from .intelligence_reports import ReportStore
        from .intelligence_sources import collect_bundle
        self.path = Path(config_path).resolve()
        self.config = load_config(self.path)
        self.root = Path(self.config['run_dir'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.council = council
        self.council.config = self.config
        self.store = store or ReportStore(self.root / 'reports')
        self.council.report_reader = lambda kind, asof: self.store.get(kind, asof)
        self.quote_fetcher = quote_fetcher or fetch_quote
        self.market_fetcher = market_fetcher or collect_markets
        self.bundle_collector = bundle_collector or collect_bundle
        self.jobs, self.last, self.quotes, self.markets, self.errors = {}, {}, {}, {}, {}
        self.council.market_reader = self.market
        self.latest_scan = self.latest_cycle = None
        self.foundation_forecasts = {}
        self.stopped = threading.Event()
        self.council.stop_event = self.stopped
        self.status = 'running'
        self._state = {}
        self.server = None
        previous = self.root/'runtime-state.json'
        prior = json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else {}
        started_at = utc_now()
        self.lifecycle = dict(started_at=started_at, previous_status=prior.get('status'),
            previous_updated_at=prior.get('updated_at'), stop_reason=None, stopped_at=None,
            observed_protection_gap_seconds=max(0., (timestamp(started_at)-timestamp(prior['updated_at'])).total_seconds())
                if prior.get('updated_at') else None,
            gap_interpretation='Time since previous persisted heartbeat; paper barriers cannot execute while offline')
        (self.root/'lifecycle').mkdir(exist_ok=True)
        save_json(self.root/'lifecycle'/('start-'+started_at.replace(':','-')+'.json'), self.lifecycle)

    def market(self, symbol):
        value = deepcopy(self.markets.get(symbol))
        if value is not None:
            value['quote'] = deepcopy(self.quotes.get(symbol))
        return value

    def _error(self, name, error):
        reason = f'{type(error).__name__}: {error}'[:2000]
        if self.errors.get(name, {}).get('error') == reason:
            return
        entry = dict(at=utc_now(), error=reason)
        self.errors[name] = entry
        with (self.root / 'runtime-errors.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(worker=name, **entry), ensure_ascii=False) + '\n')

    def reload(self):
        try:
            config = load_config(self.path)
            for key in ('run_dir', 'initial_usdt', 'symbols', 'port', 'technical_model_dir',
                        'technical_gap_fallback_model_dir', 'management_model_dir', 'technical_python', 'provider', 'model_comparison','management_comparison','adaptive_learning','adaptive_settings','strategy_bank'):
                if config[key] != self.config[key]:
                    raise ValueError('Restart required to change ' + key)
            self.config = config
            self.council.config = config
            self.errors.pop('config', None)
        except Exception as error:
            # Keep the last valid settings and the existing protective policy.
            if str(error) != self.errors.get('config', {}).get('error', '').partition(': ')[2]:
                self._error('config', error)

    def _launch(self, name, callback, now):
        if name in self.jobs or self.stopped.is_set():
            return
        job = dict(done=threading.Event(), started_at=utc_now())
        def invoke():
            try:
                job['result'] = callback()
            except Exception as error:
                job['error'] = error
            finally:
                job['done'].set()
        job['thread'] = threading.Thread(target=invoke, name='council-' + name, daemon=True)
        self.jobs[name] = job
        self.last[name] = now
        job['thread'].start()

    def _reports(self, asof):
        reports = {}
        for kind in ('market', 'macro'):
            try:
                reports[kind] = self.store.get(kind, asof)
                self.errors.pop('report_read_' + kind, None)
            except Exception as error:
                self._error('report_read_' + kind, error)
                reports[kind] = None
        return reports

    def _report(self, kind, config):
        bundle = self.bundle_collector(self.root / 'sources', kind, asof=None, config=config['source_config'])
        asof = max(timestamp(utc_now()), timestamp(bundle.get('available_at') or utc_now())).isoformat()
        report = self.store.refresh(kind, bundle, asof, config[kind + '_seconds'])
        if report.get('status') not in ('ok', 'partial'):
            raise ValueError(report.get('error') or ('Report unavailable: ' + report.get('status', 'unknown')))
        return report

    def tick(self, monotonic=None):
        if self.stopped.is_set():
            return
        stop_request = self.root / 'stop-request.json'
        if stop_request.exists():
            stop_request.unlink()
            self.stop('requested')
            return
        now = time.monotonic() if monotonic is None else monotonic
        self.reload()
        for name, job in list(self.jobs.items()):
            if not job['done'].is_set():
                continue
            del self.jobs[name]
            if 'error' in job:
                self._error(name, job['error'])
                continue
            self.errors.pop(name, None)
            result = job['result']
            if name.startswith('quote_'):
                self.quotes[name[6:]] = result
            elif name == 'candles' or name.startswith('candles_'):
                received = result.get('markets', result)
                self.markets.update(received)
                for symbol in received:
                    self.errors.pop('candles_' + symbol, None)
                if 'markets' in result:
                    for symbol, error in result.get('errors', {}).items():
                        self._error('candles_' + symbol, ValueError(error))
            elif name == 'scan':
                self.latest_scan = dict(completed_at=utc_now(), result=result)
            elif name == 'manager':
                self.latest_cycle = dict(completed_at=utc_now(), result=result)
                if result.get('status') == 'error':
                    self._error('manager_decision', ValueError(result.get('error','Manager returned no valid decision')))
                elif result.get('status') == 'ok':
                    self.errors.pop('manager_decision', None)
            elif name.startswith('foundation_'):
                self.foundation_forecasts[name[len('foundation_'):]] = result
        config = deepcopy(self.config)
        due = lambda name, seconds: now - self.last.get(name, -float('inf')) >= seconds
        for symbol in config['symbols']:
            name = 'quote_' + symbol
            if due(name, config['quote_seconds']):
                self._launch(name, lambda symbol=symbol: self.quote_fetcher(config, symbol), now)
        # Every tick protects using the newest completed quotes. Slow HTTP and model jobs never run here.
        if due('protect', min(1, config['quote_seconds'])):
            try:
                self.council.protect(deepcopy(self.quotes), utc_now())
                self.errors.pop('protect', None)
            except Exception as error:
                self._error('protect', error)
            self.last['protect'] = now
        for symbol in config['symbols']:
            name = 'candles_' + symbol
            if due(name, config['technical_seconds']):
                single_market = dict(config, symbols=[symbol])
                self._launch(name, lambda single_market=single_market: self.market_fetcher(single_market), now)
        markets = deepcopy(self.markets)
        for symbol in markets:
            markets[symbol]['quote'] = deepcopy(self.quotes.get(symbol))
        markets = {symbol: item for symbol, item in markets.items() if item.get('quote')}
        if markets and due('scan', config['technical_seconds']):
            self._launch('scan', lambda: self.council.scan(markets, utc_now()), now)
        foundation_inputs_complete = set(markets) == set(config['symbols'])
        if foundation_inputs_complete and callable(getattr(self.council, 'forecast_foundation', None)):
            for model in ('kronos', 'chronos2'):
                if self._state.get('model_training', {}).get(model, {}).get('status') == 'complete':
                    name = 'foundation_' + model
                    interval = min(60,config['foundation_seconds']) if name in self.errors else config['foundation_seconds']
                    if due(name, interval):
                        self._launch(name, lambda model=model: self.council.forecast_foundation(model, markets, utc_now()), now)
        reports = self._reports(utc_now())
        review_due = False
        if callable(getattr(self.council, 'status', None)) and due('manager', min(60, config['manager_seconds'])):
            try:
                pending = self.council.status(utc_now()).get('pending_reviews', {})
                reviews = pending.values() if isinstance(pending, dict) else pending
                review_due = any(timestamp(review['review_at']) <= timestamp(utc_now()) for review in reviews)
                self.errors.pop('pending_reviews', None)
            except Exception as error:
                self._error('pending_reviews', error)
        if markets and self.latest_scan and (due('manager', config['manager_seconds']) or review_due):
            self._launch('manager', lambda: self.council.cycle(markets, reports, utc_now()), now)
        for kind in ('market', 'macro'):
            name = 'report_' + kind
            try:
                if due(name, min(60, config[kind + '_seconds'])) and self.store.due(kind, utc_now(), config[kind + '_seconds']):
                    self._launch(name, lambda kind=kind: self._report(kind, config), now)
            except Exception as error:
                self._error(name, error)
        if due('publish', 1):
            self._publish(reports)
            self.last['publish'] = now

    def _publish(self, reports=None):
        reports = self._reports(utc_now()) if reports is None else reports
        manager = {}
        try:
            if callable(getattr(self.council, 'status', None)):
                manager = self.council.status(utc_now())
            elif (self.root / 'account.json').exists():
                manager['account'] = json.loads((self.root / 'account.json').read_text(encoding='utf-8'))
            self.errors.pop('status', None)
        except Exception as error:
            self._error('status', error)
        source_coverage = {}
        for kind in ('market', 'macro'):
            path = self.root / 'sources' / (kind + '.json')
            try:
                if path.exists():
                    bundle = json.loads(path.read_text(encoding='utf-8'))
                    source_coverage[kind] = dict(available_at=bundle.get('available_at'),
                                                 asof=bundle.get('asof'), coverage=bundle.get('coverage'))
                    self.errors.pop('coverage_' + kind, None)
            except (OSError, ValueError) as error:
                self._error('coverage_' + kind, error)
        model_training = {}
        for name in ('kronos', 'chronos2'):
            path = self.root / 'model-status' / (name + '.json')
            try:
                if path.exists() and path.stat().st_size <= 2_000_000:
                    model_training[name] = _bounded(json.loads(path.read_text(encoding='utf-8')), tail=True)
                    self.errors.pop('model_status_' + name, None)
            except (OSError, ValueError) as error:
                self._error('model_status_' + name, error)
        active_model = {}
        model_path = Path(self.config['management_model_dir'] if self.config.get('coherent_management_policy')
                          else self.config['technical_model_dir']) / 'model.json'
        try:
            if model_path.exists() and model_path.stat().st_size <= 2_000_000:
                model = json.loads(model_path.read_text(encoding='utf-8'))
                active_model = _bounded({key: model[key] for key in
                    ('model_id', 'available_at', 'format_version', 'head_sha256', 'evaluation', 'limitations', 'training_source') if key in model})
                self.errors.pop('active_model', None)
        except (OSError, ValueError) as error:
            self._error('active_model', error)
        scan_view = deepcopy(self.latest_scan)
        if scan_view and isinstance(scan_view.get('result'), dict):
            for item in scan_view['result'].get('results', []):
                if isinstance(item.get('signals'), list):
                    item['candidate_count'] = len(item['signals'])
                    quote = self.quotes.get(item.get('symbol'))
                    if quote and callable(getattr(self.council, '_net_expectation', None)):
                        for signal in item['signals']:
                            try:
                                signal['executable_expected_net_return'] = self.council._net_expectation(
                                    signal, quote['ask'], quote['bid'])
                                signal['execution_spread_bps'] = (quote['ask']-quote['bid'])/quote['ask']*10000
                            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                                signal['executable_expected_net_return'] = None
        self._state = dict(updated_at=utc_now(), status=self.status, paper_only=True,
                           lifecycle=deepcopy(self.lifecycle), entry_authority=manager.get('entry_authority'),
                           report_policy=dict(require_reports=self.config['require_reports']),
                           initial_usdt=self.config['initial_usdt'], symbols=self.config['symbols'],
                           account=manager.get('account'), pending_reviews=manager.get('pending_reviews', []),
                           last_decision=manager.get('last_decision'), decision_count=manager.get('decision_count'),
                           admission_diagnostics=_bounded(manager.get('admission_diagnostics')),
                           comparison_accounts=deepcopy(manager.get('comparison_accounts', {})),
                           matched_barriers=deepcopy(manager.get('matched_barriers', {})),
                           model_training=model_training, active_model=active_model,
                           foundation_forecasts=_bounded(self.foundation_forecasts),
                           adaptive_learning=manager.get('adaptive_learning'),
                           data_quality={symbol: _bounded(item.get('reference_source', {})) for symbol, item in self.markets.items()},
                           source_coverage=source_coverage,
                           reports={kind: dict(usable=report is not None,
                               report_id=report.get('report_id') if report else None,
                               generated_at=report.get('generated_at') if report else None,
                               valid_until=report.get('valid_until') if report else None,
                               assessment=_bounded(report.get('assessment')) if report else None,
                               coverage=report.get('coverage') if report else None) for kind, report in reports.items()},
                           quotes=deepcopy(self.quotes), errors=deepcopy(self.errors),
                           workers={name: dict(started_at=job['started_at'], running=not job['done'].is_set()) for name, job in self.jobs.items()},
                           last_scan=scan_view, last_manager_cycle=_bounded(self.latest_cycle),
                           intervals={key: self.config[key] for key in self.config if key.endswith('_seconds')},
                           stop_policy='Positions remain open in the paper ledger; quote protection runs only while this process is running.')
        try:
            save_json(self.root / 'runtime-state.json', self._state)
            self.errors.pop('snapshot', None)
        except PermissionError as error:
            # A dashboard snapshot is replaceable; account/decision writes retain their failure behavior.
            self._error('snapshot', error)

    def state(self):
        state = deepcopy(self._state)
        state.update(status=self.status, errors=deepcopy(self.errors))
        return state

    def serve(self):
        if not self.config['port']:
            return
        runtime = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/api/state':
                    body, kind = json.dumps(runtime.state(), ensure_ascii=False).encode(), 'application/json'
                elif self.path == '/':
                    from .council_dashboard import dashboard_html
                    body, kind = dashboard_html().encode('utf-8'), 'text/html'
                elif urlsplit(self.path).path == '/api/chart':
                    query = parse_qs(urlsplit(self.path).query)
                    values = query.get('symbol', [])
                    if set(query) != {'symbol'} or len(values) != 1 or values[0] not in runtime.config['symbols']:
                        self.send_error(404)
                        return
                    charts = (runtime.root / 'charts').resolve()
                    body = None
                    for receipt_path in sorted(charts.glob('*.receipt.json'), key=lambda path: path.stat().st_mtime, reverse=True)[:64]:
                        try:
                            if receipt_path.stat().st_size > 64_000:
                                continue
                            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                            if receipt.get('symbol') != values[0]:
                                continue
                            image = Path(receipt['path']).resolve()
                            if not image.is_relative_to(charts) or image.suffix != '.png' or image.stat().st_size > 20_000_000:
                                continue
                            import hashlib
                            candidate = image.read_bytes()
                            if hashlib.sha256(candidate).hexdigest() != receipt.get('sha256'):
                                continue
                            body = candidate
                            break
                        except (OSError, ValueError, KeyError):
                            continue
                    if body is None:
                        self.send_error(404)
                        return
                    kind = 'image/png'

                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind + ('; charset=utf-8' if kind != 'image/png' else ''))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', self.config['port']), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, name='council-status', daemon=True).start()

    def stop(self, reason='shutdown'):
        if self.lifecycle['stopped_at'] is None:
            self.lifecycle.update(stopped_at=utc_now(), stop_reason=reason)
            save_json(self.root/'lifecycle'/('stop-'+self.lifecycle['stopped_at'].replace(':','-')+'.json'), self.lifecycle)
        self.stopped.set()
        self.status = 'stopped'
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self._publish()


def run(config_path, seconds=None):
    from .market_council import MarketCouncil
    from .locking import exclusive
    if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('--seconds must be finite positive seconds')
    config = load_config(config_path)
    root = Path(config['run_dir'])
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / 'runtime.lock'):
        council = MarketCouncil(root, config, technical=None, provider=None)
        runtime = Runtime(config_path, council)
        started = time.monotonic()
        try:
            runtime.serve()
            while not runtime.stopped.is_set() and (seconds is None or time.monotonic() - started < seconds):
                runtime.tick()
                runtime.stopped.wait(.25)
        except KeyboardInterrupt:
            runtime.stop('keyboard_interrupt')
        except BaseException as error:
            runtime._error('fatal', error)
            runtime.stop('fatal_error: '+type(error).__name__)
            raise
        finally:
            runtime.stop('duration_complete' if seconds is not None else 'shutdown')
        return runtime.state()
