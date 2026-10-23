"""Bounded, checksum-verified official Binance spot M1 acquisition."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

import numpy as np

BASE_URL = 'https://data.binance.vision/data/spot'
MINUTE_NS = 60_000_000_000
RESERVE_BYTES = 5 * 1024 ** 3
SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT')
COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_count',
           'taker_buy_base_volume', 'taker_buy_quote_volume']
PRODUCER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PRODUCER = dict(module_sha256=PRODUCER_SHA256, revision='atomic-resume-reserve-v2')
WRITE_LOCK = threading.RLock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def utc_ns(value):
    stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp()) * 1_000_000_000 + stamp.microsecond * 1000


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def reserve(path, additional=0):
    if shutil.disk_usage(path).free < RESERVE_BYTES + additional + 4 * 1024 ** 2:
        raise OSError('Acquisition stopped to preserve at least 5 GiB free disk; free space then resume')


@contextmanager
def bulk_write(path, additional):
    # ponytail: serialize bulk writes; use byte reservations if measured throughput needs concurrent writes.
    with WRITE_LOCK:
        reserve(path, additional)
        yield


def verify_checksum(path, checksum):
    fields = checksum.strip().split()
    if len(fields) != 2 or not re.fullmatch('[0-9a-fA-F]{64}', fields[0]) or fields[1].lstrip('*') != Path(path).name:
        raise ValueError('Malformed official checksum or unexpected archive filename')
    actual = sha256(path)
    if actual != fields[0].lower():
        raise ValueError('Archive SHA256 differs from official checksum')
    return actual


def parse_klines(stream, start_ns, end_ns, audit=None):
    raw = np.loadtxt(stream, delimiter=',', dtype=np.float64, ndmin=2)
    if raw.ndim != 2 or raw.shape[1] != 12 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Expected finite official twelve-column klines')
    opened, closed = raw[:, 0], raw[:, 6]
    if np.all((opened >= 1e12) & (opened < 1e14)):
        factor = 1_000_000
    elif np.all((opened >= 1e15) & (opened < 1e16)):
        factor = 1000
    else:
        raise ValueError('Expected consistent millisecond or microsecond epoch timestamps')
    if (np.any(opened != np.floor(opened)) or np.any(closed != np.floor(closed)) or
            np.any(opened > np.iinfo(np.int64).max // factor)):
        raise ValueError('Invalid integer timestamps')
    times = opened.astype(np.int64) * factor
    if np.any(times % MINUTE_NS) or np.any(np.diff(times) <= 0):
        raise ValueError('M1 opens must be sorted, unique and minute aligned')
    duration = closed - opened
    if np.any(duration < 0) or np.any(duration >= MINUTE_NS // factor):
        raise ValueError('Invalid inclusive candle close timestamp')
    if audit is not None:
        audit.update(timestamp_unit='milliseconds' if factor == 1_000_000 else 'microseconds',
                     source_rows=len(raw), shortened_close_rows=int(np.count_nonzero(duration != MINUTE_NS // factor - 1)))
    values = np.ascontiguousarray(raw[:, [1, 2, 3, 4, 5, 7, 8, 9, 10]])
    op, high, low, close, volume, quote, count, taker, taker_quote = values.T
    if (np.any(values[:, :4] <= 0) or np.any(low > np.minimum(op, close)) or
            np.any(high < np.maximum(op, close)) or np.any(low > high) or
            np.any(values[:, 4:] < 0) or np.any(count != np.floor(count)) or np.any(count > 2 ** 53) or
            np.any(taker > volume) or np.any(taker_quote > quote) or
            np.any((volume == 0) != (count == 0)) or np.any((volume == 0) != (quote == 0)) or
            np.any((taker == 0) != (taker_quote == 0))):
        raise ValueError('Malformed OHLC, nonnegative flow or trade count constraints')
    tolerance = np.maximum(1e-8, quote * 1e-7)
    if np.any(quote < low * volume - tolerance) or np.any(quote > high * volume + tolerance):
        raise ValueError('Quote volume outside candle price bounds')
    keep = (times >= start_ns) & (times + MINUTE_NS <= end_ns)
    return values[keep], times[keep]


def quality_report(values, times, start_ns, end_ns):
    expected = max(0, (end_ns - start_ns) // MINUTE_NS)
    gaps = np.diff(times) // MINUTE_NS - 1
    if len(times) and (np.any(np.diff(times) <= 0) or times[0] < start_ns or times[-1] + MINUTE_NS > end_ns):
        raise ValueError('Canonical arrays violate source time range/order')
    missing = int(expected - len(times))
    return dict(expected_minutes=int(expected), observed_minutes=len(times), missing_minutes=missing,
                gap_count=int(np.count_nonzero(gaps > 0)), internal_missing_minutes=int(gaps[gaps > 0].sum()),
                leading_missing_minutes=int((times[0] - start_ns) // MINUTE_NS) if len(times) else int(expected),
                trailing_missing_minutes=int((end_ns - times[-1] - MINUTE_NS) // MINUTE_NS) if len(times) else 0,
                coverage_fraction_1m=len(times) / expected if expected else 0,
                coverage_complete=missing == 0,
                zero_volume_fraction=float(np.mean(values[:, 4] == 0)) if len(values) else None,
                zero_trade_fraction=float(np.mean(values[:, 6] == 0)) if len(values) else None,
                flat_fraction=float(np.mean(values[:, 1] == values[:, 2])) if len(values) else None)


def download(url, destination):
    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + '.part')
    last = None
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'barrier-research/1.0'}), timeout=45) as response:
                headers = dict(response.headers)
                expected = int(headers.get('Content-Length', 0))
                with bulk_write(destination.parent, expected):
                    with partial.open('wb') as target:
                        for block in iter(lambda: response.read(1024 * 1024), b''):
                            reserve(destination.parent, len(block))
                            target.write(block)
                        if expected and target.tell() != expected:
                            raise ConnectionError('Incomplete HTTP response body')
                    os.replace(partial, destination)
            return dict(http_status=200, headers=headers, received_utc=utc_now())
        except HTTPError as error:
            if error.code == 404 or error.code < 500:
                raise
            last = error
        except (URLError, TimeoutError, ConnectionError) as error:
            last = error
        if attempt < 2:
            time.sleep(attempt + 1)
    raise last


def partitions(start_ns, end_ns):
    current = datetime.fromtimestamp(start_ns // 1_000_000_000, timezone.utc)
    end = datetime.fromtimestamp(end_ns // 1_000_000_000, timezone.utc)
    while current < end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        if current.day == 1 and next_month <= end:
            yield 'monthly', current.strftime('%Y-%m'), int(current.timestamp()) * 1_000_000_000, int(next_month.timestamp()) * 1_000_000_000
            current = next_month
        else:
            next_day = current + timedelta(days=1)
            yield 'daily', current.strftime('%Y-%m-%d'), int(current.timestamp()) * 1_000_000_000, int(next_day.timestamp()) * 1_000_000_000
            current = next_day


def save_array(path, array):
    with bulk_write(path.parent, array.nbytes + 128):
        temporary = path.with_suffix('.tmp.npy')
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)


def delete_redundant(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Refusing to delete outside new acquisition source')
    size = path.stat().st_size if path.exists() else 0
    path.unlink(missing_ok=True)
    return size


def cached_arrays(record):
    for kind in ('values', 'times'):
        path = record[kind + '_path']
        if sha256(path) != record[kind + '_sha256']:
            raise ValueError(f'Cached {kind} array SHA256 mismatch: {path}')
    values = np.load(record['values_path'], mmap_mode='r', allow_pickle=False)
    times = np.load(record['times_path'], mmap_mode='r', allow_pickle=False)
    if values.dtype != np.float64 or values.shape != (record['rows'], 9) or times.dtype != np.int64 or times.shape != (record['rows'],):
        raise ValueError('Cached array schema mismatch')
    return values, times


def finish_partition_cleanup(root, result):
    cached_arrays(result)
    for kind in ('values', 'times'):
        delete_redundant(Path(result[kind + '_path']).with_suffix('.previous.npy'), root / 'market')
    for receipt_path in result['source_receipts']:
        record = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
        if record['status'] == 'ready':
            for key in ('values_path', 'times_path'):
                record['reclaimed_bytes'] = record.get('reclaimed_bytes', 0) + delete_redundant(record[key], root / 'partitions')
            record['retention'] = 'consolidated_symbol'
            write_json(receipt_path, record)
    result['reclaimed_bytes'] = sum(json.loads(Path(path).read_text(encoding='utf-8')).get('reclaimed_bytes', 0) for path in result['source_receipts'])
    write_json(root / 'market' / f"{Path(result['values_path']).name.removesuffix('-values.npy')}.json", result)
    return result


def recover_symbol_commit(root, symbol):
    receipt = root / 'market' / f'{symbol}.json'
    if not receipt.exists():
        return
    record = json.loads(receipt.read_text(encoding='utf-8'))
    for kind in ('values', 'times'):
        path = Path(record[kind + '_path'])
        backup = path.with_suffix('.previous.npy')
        if path.exists() and sha256(path) == record[kind + '_sha256']:
            continue
        if backup.exists() and sha256(backup) == record[kind + '_sha256']:
            os.replace(backup, path)
        else:
            raise ValueError('No valid committed symbol array or recovery backup')


def acquire_partition(root, symbol, partition, progress, previous=None):
    kind, period, begin, end = partition
    name = f'{symbol}-1m-{period}'
    receipt_path = root / 'receipts' / symbol / (name + '.json')
    old = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
    if old:
        if old['status'] == 'ready':
            if previous and not all(Path(old[key + '_path']).exists() for key in ('values', 'times')):
                values = np.load(previous['values_path'], mmap_mode='r', allow_pickle=False)
                times = np.load(previous['times_path'], mmap_mode='r', allow_pickle=False)
                first, last = np.searchsorted(times, [begin, end])
                save_array(Path(old['values_path']), values[first:last])
                save_array(Path(old['times_path']), times[first:last])
                del values, times
                old['retention'] = 'canonical_partition'
                write_json(receipt_path, old)
            cached_arrays(old)
            progress(symbol, name, 'cached')
            return old
    url = f'{BASE_URL}/{kind}/klines/{symbol}/1m/{name}.zip'
    archive = root / 'archives' / symbol / (name + '.zip')
    checksum_path = archive.with_suffix('.zip.CHECKSUM')
    archive.parent.mkdir(parents=True, exist_ok=True)
    part_root = root / 'partitions' / symbol
    part_root.mkdir(parents=True, exist_ok=True)
    record = dict(symbol=symbol, period=period, kind=kind, url=url, checksum_url=url + '.CHECKSUM',
                  start_ns=begin, end_ns=end, started_utc=utc_now(), status='downloading', producer=PRODUCER,
                  attempt_errors=list(old.get('attempt_errors', [])))
    if old.get('status') in ('missing', 'error'):
        record['attempt_errors'].append(dict(status=old['status'], error=old.get('error'), utc=old.get('completed_utc')))
    progress(symbol, name, 'downloading')
    try:
        valid_cache = False
        if checksum_path.exists() and archive.exists():
            try:
                verify_checksum(archive, checksum_path.read_text(encoding='utf-8'))
                valid_cache = True
            except ValueError as error:
                record['attempt_errors'].append(dict(status='invalid_cache', error=str(error), utc=utc_now(),
                                                     cached_zip_sha256=sha256(archive)))
                write_json(receipt_path, dict(record, status='error', error=str(error)))
        if valid_cache:
            checksum_response = response = dict(cached_verified_download=True)
        else:
            checksum_response = download(url + '.CHECKSUM', checksum_path)
            response = download(url, archive)
        checksum = checksum_path.read_text(encoding='utf-8')
        archive_hash = verify_checksum(archive, checksum)
        record.update(source_zip_sha256=archive_hash, official_checksum=checksum,
                      checksum_response=checksum_response, archive_response=response)
        write_json(receipt_path, record)
        audit = {}
        with zipfile.ZipFile(archive) as zipped:
            entries = zipped.infolist()
            if len(entries) != 1 or entries[0].filename != name + '.csv' or entries[0].file_size > 64 * 1024 ** 2:
                raise ValueError('Expected one bounded flat CSV member in official kline ZIP')
            with zipped.open(entries[0]) as member:
                values, times = parse_klines(io.TextIOWrapper(member, encoding='utf-8'), begin, end, audit)
        values_path, times_path = part_root / (name + '-values.npy'), part_root / (name + '-times.npy')
        save_array(values_path, values)
        save_array(times_path, times)
        record.update(status='ready', rows=len(times), values_path=str(values_path), times_path=str(times_path),
                      values_sha256=sha256(values_path), times_sha256=sha256(times_path),
                      source_zip_sha256=archive_hash, official_checksum=checksum,
                      checksum_response=checksum_response, archive_response=response,
                      parser_audit=audit,
                      quality=quality_report(values, times, begin, end), retention='canonical_partition', completed_utc=utc_now())
        cached_arrays(record)
        write_json(receipt_path, record)
        reclaimed = delete_redundant(archive, root / 'archives') + delete_redundant(checksum_path, root / 'archives')
        record['reclaimed_bytes'] = reclaimed
        write_json(receipt_path, record)
        progress(symbol, name, 'ready')
        return record
    except HTTPError as error:
        record.update(status='missing' if error.code == 404 else 'error', http_status=error.code,
                      error=str(error), completed_utc=utc_now())
        write_json(receipt_path, record)
        progress(symbol, name, record['status'])
        if error.code != 404:
            raise
        return record
    except BaseException as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}', completed_utc=utc_now())
        write_json(receipt_path, record)
        raise


def acquire_symbol(root, symbol, start_ns, end_ns, progress, previous=None):
    records = []
    for partition in partitions(start_ns, end_ns):
        record = acquire_partition(root, symbol, partition, progress, previous)
        records.append(record)
        if record['status'] == 'missing' and partition[0] == 'monthly':
            # A missing monthly archive may have available official daily partitions.
            _, _, begin, end = partition
            for day in range(begin, end, 1440 * MINUTE_NS):
                period = datetime.fromtimestamp(day // 1_000_000_000, timezone.utc).strftime('%Y-%m-%d')
                records.append(acquire_partition(root, symbol, ('daily', period, day, day + 1440 * MINUTE_NS), progress, previous))
    ready = [record for record in records if record['status'] == 'ready']
    rows = sum(record['rows'] for record in ready)
    market = root / 'market'
    market.mkdir(exist_ok=True)
    values_path, times_path = market / f'{symbol}-values.npy', market / f'{symbol}-times.npy'
    values_tmp, times_tmp = market / f'{symbol}-values.tmp.npy', market / f'{symbol}-times.tmp.npy'
    with bulk_write(root, rows * 80 + 256):
        values = np.lib.format.open_memmap(values_tmp, mode='w+', dtype=np.float64, shape=(rows, 9))
        times = np.lib.format.open_memmap(times_tmp, mode='w+', dtype=np.int64, shape=(rows,))
        offset = 0
        for record in ready:
            chunk_values, chunk_times = cached_arrays(record)
            length = len(chunk_times)
            if offset and length and times[offset - 1] >= chunk_times[0]:
                raise ValueError('Overlapping/unsorted source partitions')
            values[offset:offset + length], times[offset:offset + length] = chunk_values, chunk_times
            offset += length
            del chunk_values, chunk_times
        values.flush()
        times.flush()
        quality = quality_report(values, times, start_ns, end_ns)
        first, last = (int(times[0]), int(times[-1])) if rows else (None, None)
        del values, times
        for path in (values_path, times_path):
            if previous and path.exists():
                os.replace(path, path.with_suffix('.previous.npy'))
        os.replace(values_tmp, values_path)
        os.replace(times_tmp, times_path)
    result = dict(status='ready', rows=rows, values_path=str(values_path), times_path=str(times_path),
                  values_sha256=sha256(values_path), times_sha256=sha256(times_path),
                  first_open_ns=first, last_open_ns=last, quality=quality,
                  archive_coverage_complete=not any(r['status'] == 'missing' and r['kind'] == 'daily' for r in records),
                  observed_source_coverage=[dict(period=r['period'], kind=r['kind'], rows=r['rows'], source_zip_sha256=r['source_zip_sha256']) for r in ready],
                  missing_partitions=[dict(period=r['period'], kind=r['kind'], http_status=r.get('http_status')) for r in records if r['status'] == 'missing'],
                  source_receipts=[str(root / 'receipts' / symbol / f"{symbol}-1m-{r['period']}.json") for r in records],
                  completed_utc=utc_now(), producer=PRODUCER)
    cached_arrays(result)
    # Commit the final symbol receipt before removing its redundant partition arrays.
    write_json(market / f'{symbol}.json', result)
    return finish_partition_cleanup(root, result)


@contextmanager
def source_lock(root):
    with open(root / '.acquisition.lock', 'a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            if not stream.read(1):
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def acquire_history(output, start, end, symbols, workers=2):
    root = Path(output).resolve()
    start_ns, end_ns = utc_ns(start), utc_ns(end)
    symbols = list(symbols)
    if (not 1 <= workers <= 2 or not symbols or len(set(symbols)) != len(symbols) or
            any(not re.fullmatch('[A-Z0-9]{3,24}', symbol) for symbol in symbols) or
            start_ns >= end_ns or start_ns % (1440 * MINUTE_NS) or end_ns % (1440 * MINUTE_NS)):
        raise ValueError('Need unique safe symbols, 1-2 workers and increasing UTC midnight dates')
    protocol = dict(schema='barrier-binance-history-v1', start_ns=start_ns, end_ns=end_ns, symbols=symbols,
                    requested_from_utc=datetime.fromtimestamp(start_ns // 1_000_000_000, timezone.utc).isoformat(),
                    requested_to_exclusive_utc=datetime.fromtimestamp(end_ns // 1_000_000_000, timezone.utc).isoformat(),
                    columns=COLUMNS, values_dtype='float64', times_dtype='int64', times_unit='UTC open nanoseconds',
                    source=BASE_URL, venue='Binance spot', resolution='1m', max_workers=2, reserve_bytes=RESERVE_BYTES,
                    missing_policy='No interpolation; explicit missing archive dispositions; monthly 404 falls back to daily archives.',
                    retention='Retain canonical symbol arrays and official URL/checksum/hash receipts. Delete ZIP after verified partition commit; delete partition arrays after verified final symbol commit.',
                    universe_bias='Fixed present-day research universe; not survivorship-free; OHLCV is not depth or executable quotes.')
    root.mkdir(parents=True, exist_ok=True)
    with source_lock(root):
        reserve(root)
        protocol_path = root / 'protocol.json'
        if protocol_path.exists():
            existing = json.loads(protocol_path.read_text(encoding='utf-8'))
            if any(existing.get(key) != value for key, value in protocol.items()):
                raise ValueError('Resume arguments differ from immutable acquisition protocol')
        else:
            protocol['created_utc'] = utc_now()
            protocol['frozen_before_download'] = True
            write_json(protocol_path, protocol)
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else dict(symbols={})
        if manifest.get('protocol_sha256', sha256(protocol_path)) != sha256(protocol_path):
            raise ValueError('Source protocol SHA256 mismatch')
        manifest.update(status='downloading', protocol_sha256=sha256(protocol_path), schema=protocol['schema'], producer=PRODUCER)
        manifest.pop('error', None)
        state = dict(stage='acquisition', status='running', pid=os.getpid(), workers=workers,
                     updated_utc=utc_now(), symbols={}, dispositions_this_session=0)
        guard = threading.Lock()

        def progress(symbol, archive, status):
            with guard:
                state.update(updated_utc=utc_now(), free_bytes=shutil.disk_usage(root).free)
                state['symbols'][symbol] = dict(archive=archive, status=status)
                if status in ('ready', 'missing', 'cached'):
                    state['dispositions_this_session'] += 1
                write_json(root / 'progress.json', state)
                print(json.dumps(dict(symbol=symbol, archive=archive, status=status, utc=state['updated_utc'])), flush=True)

        write_json(manifest_path, manifest)
        write_json(root / 'progress.json', state)
        todo = {}
        try:
            for symbol in symbols:
                final_receipt = root / 'market' / f'{symbol}.json'
                if final_receipt.exists():
                    recover_symbol_commit(root, symbol)
                    record = json.loads(final_receipt.read_text(encoding='utf-8'))
                    cached_arrays(record)
                    if record['archive_coverage_complete']:
                        manifest['symbols'][symbol] = finish_partition_cleanup(root, record)
                    else:
                        todo[symbol] = record
                else:
                    todo[symbol] = None
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = {pool.submit(acquire_symbol, root, symbol, start_ns, end_ns, progress, previous): symbol for symbol, previous in todo.items()}
                for job in as_completed(jobs):
                    symbol = jobs[job]
                    manifest['symbols'][symbol] = job.result()
                    write_json(manifest_path, manifest)
            complete = all(manifest['symbols'][symbol]['quality']['coverage_complete'] for symbol in symbols)
            archives_complete = all(manifest['symbols'][symbol]['archive_coverage_complete'] for symbol in symbols)
            manifest.update(status='ready' if archives_complete else 'partial', completed_utc=utc_now(),
                            coverage_complete=complete, archive_coverage_complete=archives_complete,
                            reclaimed_bytes=sum(r.get('reclaimed_bytes', 0) for r in manifest['symbols'].values()))
            state.update(status=manifest['status'], stage='complete', updated_utc=utc_now())
        except BaseException as error:
            manifest.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                            error=f'{type(error).__name__}: {error}', updated_utc=utc_now())
            state.update(status=manifest['status'], error=manifest['error'], updated_utc=utc_now())
            raise
        finally:
            write_json(manifest_path, manifest)
            write_json(root / 'progress.json', state)
        return manifest
