# Scoped review fixes
```diff
--- before/wallex_lab/barrier_binance_history.py
+++ after/wallex_lab/barrier_binance_history.py
@@ -23,6 +23,9 @@
 SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT')
 COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_count',
            'taker_buy_base_volume', 'taker_buy_quote_volume']
+PRODUCER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
+PRODUCER = dict(module_sha256=PRODUCER_SHA256, revision='atomic-resume-reserve-v2')
+WRITE_LOCK = threading.RLock()
 
 
 def utc_now():
@@ -55,6 +58,14 @@
 def reserve(path, additional=0):
     if shutil.disk_usage(path).free < RESERVE_BYTES + additional + 4 * 1024 ** 2:
         raise OSError('Acquisition stopped to preserve at least 5 GiB free disk; free space then resume')
+
+
+@contextmanager
+def bulk_write(path, additional):
+    # ponytail: serialize bulk writes; use byte reservations if measured throughput needs concurrent writes.
+    with WRITE_LOCK:
+        reserve(path, additional)
+        yield
 
 
 def verify_checksum(path, checksum):
@@ -125,16 +136,21 @@
 
 def download(url, destination):
     destination = Path(destination)
+    partial = destination.with_suffix(destination.suffix + '.part')
     last = None
     for attempt in range(3):
         try:
             with urlopen(Request(url, headers={'User-Agent': 'barrier-research/1.0'}), timeout=45) as response:
                 headers = dict(response.headers)
-                reserve(destination.parent, int(headers.get('Content-Length', 0)))
-                with destination.open('wb') as target:
-                    for block in iter(lambda: response.read(1024 * 1024), b''):
-                        reserve(destination.parent, len(block))
-                        target.write(block)
+                expected = int(headers.get('Content-Length', 0))
+                with bulk_write(destination.parent, expected):
+                    with partial.open('wb') as target:
+                        for block in iter(lambda: response.read(1024 * 1024), b''):
+                            reserve(destination.parent, len(block))
+                            target.write(block)
+                        if expected and target.tell() != expected:
+                            raise ConnectionError('Incomplete HTTP response body')
+                    os.replace(partial, destination)
             return dict(http_status=200, headers=headers, received_utc=utc_now())
         except HTTPError as error:
             if error.code == 404 or error.code < 500:
@@ -162,10 +178,10 @@
 
 
 def save_array(path, array):
-    reserve(path.parent, array.nbytes + 128)
-    temporary = path.with_suffix('.tmp.npy')
-    np.save(temporary, array, allow_pickle=False)
-    os.replace(temporary, path)
+    with bulk_write(path.parent, array.nbytes + 128):
+        temporary = path.with_suffix('.tmp.npy')
+        np.save(temporary, array, allow_pickle=False)
+        os.replace(temporary, path)
 
 
 def delete_redundant(path, root):
@@ -189,18 +205,57 @@
     return values, times
 
 
-def acquire_partition(root, symbol, partition, progress):
+def finish_partition_cleanup(root, result):
+    cached_arrays(result)
+    for kind in ('values', 'times'):
+        delete_redundant(Path(result[kind + '_path']).with_suffix('.previous.npy'), root / 'market')
+    for receipt_path in result['source_receipts']:
+        record = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
+        if record['status'] == 'ready':
+            for key in ('values_path', 'times_path'):
+                record['reclaimed_bytes'] = record.get('reclaimed_bytes', 0) + delete_redundant(record[key], root / 'partitions')
+            record['retention'] = 'consolidated_symbol'
+            write_json(receipt_path, record)
+    result['reclaimed_bytes'] = sum(json.loads(Path(path).read_text(encoding='utf-8')).get('reclaimed_bytes', 0) for path in result['source_receipts'])
+    write_json(root / 'market' / f"{Path(result['values_path']).name.removesuffix('-values.npy')}.json", result)
+    return result
+
+
+def recover_symbol_commit(root, symbol):
+    receipt = root / 'market' / f'{symbol}.json'
+    if not receipt.exists():
+        return
+    record = json.loads(receipt.read_text(encoding='utf-8'))
+    for kind in ('values', 'times'):
+        path = Path(record[kind + '_path'])
+        backup = path.with_suffix('.previous.npy')
+        if path.exists() and sha256(path) == record[kind + '_sha256']:
+            continue
+        if backup.exists() and sha256(backup) == record[kind + '_sha256']:
+            os.replace(backup, path)
+        else:
+            raise ValueError('No valid committed symbol array or recovery backup')
+
+
+def acquire_partition(root, symbol, partition, progress, previous=None):
     kind, period, begin, end = partition
     name = f'{symbol}-1m-{period}'
     receipt_path = root / 'receipts' / symbol / (name + '.json')
-    if receipt_path.exists():
-        record = json.loads(receipt_path.read_text(encoding='utf-8'))
-        if record['status'] == 'ready':
-            cached_arrays(record)
+    old = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
+    if old:
+        if old['status'] == 'ready':
+            if previous and not all(Path(old[key + '_path']).exists() for key in ('values', 'times')):
+                values = np.load(previous['values_path'], mmap_mode='r', allow_pickle=False)
+                times = np.load(previous['times_path'], mmap_mode='r', allow_pickle=False)
+                first, last = np.searchsorted(times, [begin, end])
+                save_array(Path(old['values_path']), values[first:last])
+                save_array(Path(old['times_path']), times[first:last])
+                del values, times
+                old['retention'] = 'canonical_partition'
+                write_json(receipt_path, old)
+            cached_arrays(old)
             progress(symbol, name, 'cached')
-            return record
-        if record['status'] == 'missing':
-            return record
+            return old
     url = f'{BASE_URL}/{kind}/klines/{symbol}/1m/{name}.zip'
     archive = root / 'archives' / symbol / (name + '.zip')
     checksum_path = archive.with_suffix('.zip.CHECKSUM')
@@ -208,10 +263,22 @@
     part_root = root / 'partitions' / symbol
     part_root.mkdir(parents=True, exist_ok=True)
     record = dict(symbol=symbol, period=period, kind=kind, url=url, checksum_url=url + '.CHECKSUM',
-                  start_ns=begin, end_ns=end, started_utc=utc_now(), status='downloading')
+                  start_ns=begin, end_ns=end, started_utc=utc_now(), status='downloading', producer=PRODUCER,
+                  attempt_errors=list(old.get('attempt_errors', [])))
+    if old.get('status') in ('missing', 'error'):
+        record['attempt_errors'].append(dict(status=old['status'], error=old.get('error'), utc=old.get('completed_utc')))
     progress(symbol, name, 'downloading')
     try:
+        valid_cache = False
         if checksum_path.exists() and archive.exists():
+            try:
+                verify_checksum(archive, checksum_path.read_text(encoding='utf-8'))
+                valid_cache = True
+            except ValueError as error:
+                record['attempt_errors'].append(dict(status='invalid_cache', error=str(error), utc=utc_now(),
+                                                     cached_zip_sha256=sha256(archive)))
+                write_json(receipt_path, dict(record, status='error', error=str(error)))
+        if valid_cache:
             checksum_response = response = dict(cached_verified_download=True)
         else:
             checksum_response = download(url + '.CHECKSUM', checksum_path)
@@ -258,42 +325,45 @@
         raise
 
 
-def acquire_symbol(root, symbol, start_ns, end_ns, progress):
+def acquire_symbol(root, symbol, start_ns, end_ns, progress, previous=None):
     records = []
     for partition in partitions(start_ns, end_ns):
-        record = acquire_partition(root, symbol, partition, progress)
+        record = acquire_partition(root, symbol, partition, progress, previous)
         records.append(record)
         if record['status'] == 'missing' and partition[0] == 'monthly':
             # A missing monthly archive may have available official daily partitions.
             _, _, begin, end = partition
             for day in range(begin, end, 1440 * MINUTE_NS):
                 period = datetime.fromtimestamp(day // 1_000_000_000, timezone.utc).strftime('%Y-%m-%d')
-                records.append(acquire_partition(root, symbol, ('daily', period, day, day + 1440 * MINUTE_NS), progress))
+                records.append(acquire_partition(root, symbol, ('daily', period, day, day + 1440 * MINUTE_NS), progress, previous))
     ready = [record for record in records if record['status'] == 'ready']
     rows = sum(record['rows'] for record in ready)
     market = root / 'market'
     market.mkdir(exist_ok=True)
-    reserve(root, rows * 80 + 256)
     values_path, times_path = market / f'{symbol}-values.npy', market / f'{symbol}-times.npy'
     values_tmp, times_tmp = market / f'{symbol}-values.tmp.npy', market / f'{symbol}-times.tmp.npy'
-    values = np.lib.format.open_memmap(values_tmp, mode='w+', dtype=np.float64, shape=(rows, 9))
-    times = np.lib.format.open_memmap(times_tmp, mode='w+', dtype=np.int64, shape=(rows,))
-    offset = 0
-    for record in ready:
-        chunk_values, chunk_times = cached_arrays(record)
-        length = len(chunk_times)
-        if offset and length and times[offset - 1] >= chunk_times[0]:
-            raise ValueError('Overlapping/unsorted source partitions')
-        values[offset:offset + length], times[offset:offset + length] = chunk_values, chunk_times
-        offset += length
-        del chunk_values, chunk_times
-    values.flush()
-    times.flush()
-    quality = quality_report(values, times, start_ns, end_ns)
-    first, last = (int(times[0]), int(times[-1])) if rows else (None, None)
-    del values, times
-    os.replace(values_tmp, values_path)
-    os.replace(times_tmp, times_path)
+    with bulk_write(root, rows * 80 + 256):
+        values = np.lib.format.open_memmap(values_tmp, mode='w+', dtype=np.float64, shape=(rows, 9))
+        times = np.lib.format.open_memmap(times_tmp, mode='w+', dtype=np.int64, shape=(rows,))
+        offset = 0
+        for record in ready:
+            chunk_values, chunk_times = cached_arrays(record)
+            length = len(chunk_times)
+            if offset and length and times[offset - 1] >= chunk_times[0]:
+                raise ValueError('Overlapping/unsorted source partitions')
+            values[offset:offset + length], times[offset:offset + length] = chunk_values, chunk_times
+            offset += length
+            del chunk_values, chunk_times
+        values.flush()
+        times.flush()
+        quality = quality_report(values, times, start_ns, end_ns)
+        first, last = (int(times[0]), int(times[-1])) if rows else (None, None)
+        del values, times
+        for path in (values_path, times_path):
+            if previous and path.exists():
+                os.replace(path, path.with_suffix('.previous.npy'))
+        os.replace(values_tmp, values_path)
+        os.replace(times_tmp, times_path)
     result = dict(status='ready', rows=rows, values_path=str(values_path), times_path=str(times_path),
                   values_sha256=sha256(values_path), times_sha256=sha256(times_path),
                   first_open_ns=first, last_open_ns=last, quality=quality,
@@ -301,22 +371,11 @@
                   observed_source_coverage=[dict(period=r['period'], kind=r['kind'], rows=r['rows'], source_zip_sha256=r['source_zip_sha256']) for r in ready],
                   missing_partitions=[dict(period=r['period'], kind=r['kind'], http_status=r.get('http_status')) for r in records if r['status'] == 'missing'],
                   source_receipts=[str(root / 'receipts' / symbol / f"{symbol}-1m-{r['period']}.json") for r in records],
-                  completed_utc=utc_now())
+                  completed_utc=utc_now(), producer=PRODUCER)
     cached_arrays(result)
     # Commit the final symbol receipt before removing its redundant partition arrays.
     write_json(market / f'{symbol}.json', result)
-    reclaimed = 0
-    for record in ready:
-        reclaimed += record.get('reclaimed_bytes', 0)
-        record['retention'] = 'consolidated_symbol'
-        for key in ('values_path', 'times_path'):
-            removed = delete_redundant(record[key], root / 'partitions')
-            reclaimed += removed
-            record['reclaimed_bytes'] = record.get('reclaimed_bytes', 0) + removed
-        write_json(root / 'receipts' / symbol / f"{symbol}-1m-{record['period']}.json", record)
-    result['reclaimed_bytes'] = reclaimed
-    write_json(market / f'{symbol}.json', result)
-    return result
+    return finish_partition_cleanup(root, result)
 
 
 @contextmanager
@@ -373,7 +432,7 @@
         manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else dict(symbols={})
         if manifest.get('protocol_sha256', sha256(protocol_path)) != sha256(protocol_path):
             raise ValueError('Source protocol SHA256 mismatch')
-        manifest.update(status='downloading', protocol_sha256=sha256(protocol_path), schema=protocol['schema'])
+        manifest.update(status='downloading', protocol_sha256=sha256(protocol_path), schema=protocol['schema'], producer=PRODUCER)
         manifest.pop('error', None)
         state = dict(stage='acquisition', status='running', pid=os.getpid(), workers=workers,
                      updated_utc=utc_now(), symbols={}, dispositions_this_session=0)
@@ -390,18 +449,22 @@
 
         write_json(manifest_path, manifest)
         write_json(root / 'progress.json', state)
-        todo = []
+        todo = {}
         try:
             for symbol in symbols:
                 final_receipt = root / 'market' / f'{symbol}.json'
                 if final_receipt.exists():
+                    recover_symbol_commit(root, symbol)
                     record = json.loads(final_receipt.read_text(encoding='utf-8'))
                     cached_arrays(record)
-                    manifest['symbols'][symbol] = record
+                    if record['archive_coverage_complete']:
+                        manifest['symbols'][symbol] = finish_partition_cleanup(root, record)
+                    else:
+                        todo[symbol] = record
                 else:
-                    todo.append(symbol)
+                    todo[symbol] = None
             with ThreadPoolExecutor(max_workers=workers) as pool:
-                jobs = {pool.submit(acquire_symbol, root, symbol, start_ns, end_ns, progress): symbol for symbol in todo}
+                jobs = {pool.submit(acquire_symbol, root, symbol, start_ns, end_ns, progress, previous): symbol for symbol, previous in todo.items()}
                 for job in as_completed(jobs):
                     symbol = jobs[job]
                     manifest['symbols'][symbol] = job.result()
--- before/checks/check_barrier_binance_history.py
+++ after/checks/check_barrier_binance_history.py
@@ -5,6 +5,10 @@
 from pathlib import Path
 import sys
 import tempfile
+import threading
+import time
+from concurrent.futures import ThreadPoolExecutor
+from types import SimpleNamespace
 import zipfile
 from urllib.error import HTTPError
 
@@ -104,6 +108,13 @@
         assert not list((output / 'archives').rglob('*.zip')), 'Delete redundant ZIP after partition commit'
         assert not list((output / 'partitions').rglob('*.npy')), 'Delete redundant partitions after final commit'
         assert record['reclaimed_bytes'] > 0
+        receipt_path = Path(record['source_receipts'][0])
+        receipt = json.loads(receipt_path.read_text())
+        np.save(receipt['values_path'], np.load(record['values_path'])[:1])
+        np.save(receipt['times_path'], np.load(record['times_path'])[:1])
+        history.acquire_history(output, '2023-01-31', '2023-02-02', ['BTCUSDT'], 1)
+        assert not Path(receipt['values_path']).exists(), 'Resume must finish interrupted final cleanup'
+        assert not Path(receipt['times_path']).exists()
         Path(record['values_path']).write_bytes(b'corrupted cached array')
         raises(ValueError, history.acquire_history, output, '2023-01-31', '2023-02-02', ['BTCUSDT'], 1)
 
@@ -119,4 +130,131 @@
     finally:
         history.download = original
 
-print('PASS: ms/us parity, validation, gaps, cutoff, checksum, interruption/resume and missing archives')
+def check_atomic_download():
+    original_urlopen = history.urlopen
+    class Response:
+        headers = {'Content-Length': '8'}
+        def __init__(self, interrupted):
+            self.step, self.interrupted = 0, interrupted
+        def __enter__(self):
+            return self
+        def __exit__(self, *args):
+            return False
+        def read(self, size):
+            self.step += 1
+            if self.step == 1:
+                return b'partial' if self.interrupted else b'complete'
+            if self.interrupted:
+                raise KeyboardInterrupt('connection interrupted')
+            return b''
+    try:
+        with tempfile.TemporaryDirectory() as temp:
+            destination = Path(temp) / 'archive.zip'
+            history.urlopen = lambda *args, **kwargs: Response(True)
+            raises(KeyboardInterrupt, history.download, 'https://fixture/archive.zip', destination)
+            assert not destination.exists(), 'Interrupted download must not publish a completed ZIP'
+            assert destination.with_suffix('.zip.part').exists()
+            history.urlopen = lambda *args, **kwargs: Response(False)
+            history.download('https://fixture/archive.zip', destination)
+            assert destination.read_bytes() == b'complete'
+            assert not destination.with_suffix('.zip.part').exists()
+    finally:
+        history.urlopen = original_urlopen
+
+
+def check_legacy_truncated_zip():
+    original_download = history.download
+    attempts = [0]
+    def truncated(url, destination):
+        if url.endswith('.zip') and attempts[0] == 0:
+            attempts[0] += 1
+            Path(destination).write_bytes(b'PK-truncated')
+            raise KeyboardInterrupt('legacy partial ZIP')
+        return fake_download(url, destination)
+    try:
+        with tempfile.TemporaryDirectory() as temp:
+            history.download = truncated
+            raises(KeyboardInterrupt, history.acquire_history, temp, '2023-02-01', '2023-02-02', ['BTCUSDT'], 1)
+            completed = history.acquire_history(temp, '2023-02-01', '2023-02-02', ['BTCUSDT'], 1)
+            assert completed['status'] == 'ready' and completed['symbols']['BTCUSDT']['rows'] == 1
+            receipt = json.loads(next((Path(temp) / 'receipts').rglob('*.json')).read_text())
+            assert any('checksum' in failure['error'].lower() for failure in receipt['attempt_errors'])
+            assert receipt['producer']['module_sha256'] == history.PRODUCER_SHA256
+    finally:
+        history.download = original_download
+
+
+def check_missing_published_on_resume():
+    original_download = history.download
+    def missing_day(url, destination):
+        if '2023-02-01' in url:
+            raise HTTPError(url, 404, 'not published yet', {}, None)
+        return fake_download(url, destination)
+    try:
+        with tempfile.TemporaryDirectory() as temp:
+            history.download = missing_day
+            partial = history.acquire_history(temp, '2023-01-31', '2023-02-02', ['BTCUSDT'], 1)
+            old = partial['symbols']['BTCUSDT']
+            assert partial['status'] == 'partial' and old['rows'] == 1
+            history.download = fake_download
+            original_replace = history.os.replace
+            interrupted_commit = [False]
+            def interrupt_commit(source, destination):
+                if str(source).endswith('BTCUSDT-times.tmp.npy') and not interrupted_commit[0]:
+                    interrupted_commit[0] = True
+                    raise KeyboardInterrupt('between final values/times replacement')
+                return original_replace(source, destination)
+            history.os.replace = interrupt_commit
+            try:
+                raises(KeyboardInterrupt, history.acquire_history, temp, '2023-01-31', '2023-02-02', ['BTCUSDT'], 1)
+                for kind in ('values', 'times'):
+                    path = Path(old[kind + '_path'])
+                    candidates = [path, path.with_suffix('.previous.npy')]
+                    assert any(p.exists() and history.sha256(p) == old[kind + '_sha256'] for p in candidates), 'Interrupted refresh must preserve committed data'
+            finally:
+                history.os.replace = original_replace
+            completed = history.acquire_history(temp, '2023-01-31', '2023-02-02', ['BTCUSDT'], 1)
+            new = completed['symbols']['BTCUSDT']
+            assert completed['status'] == 'ready' and new['rows'] == 2, 'Resume must retry published missing days'
+            assert np.load(new['times_path']).tolist() == [history.utc_ns('2023-01-31'), history.utc_ns('2023-02-01')]
+    finally:
+        history.download = original_download
+
+
+def check_parallel_reserve():
+    original_usage, original_save = history.shutil.disk_usage, history.np.save
+    budget = [history.RESERVE_BYTES + 16 * 1024 ** 2]
+    result = []
+    def slow_save(path, array, **kwargs):
+        time.sleep(0.15)
+        budget[0] -= array.nbytes + 128
+        Path(path).write_bytes(b'fixture')
+    def writer(path, array):
+        try:
+            history.save_array(path, array)
+            result.append('saved')
+        except OSError:
+            result.append('reserve')
+    try:
+        history.shutil.disk_usage = lambda path: SimpleNamespace(free=budget[0])
+        history.np.save = slow_save
+        with tempfile.TemporaryDirectory() as temp:
+            array = np.zeros(1024 ** 2, dtype=np.float64)
+            with ThreadPoolExecutor(max_workers=2) as pool:
+                futures = [pool.submit(writer, Path(temp) / f'{index}.npy', array) for index in range(2)]
+                for future in futures:
+                    future.result()
+            assert sorted(result) == ['reserve', 'saved'], 'Concurrent writes must share reserve protection'
+            assert budget[0] >= history.RESERVE_BYTES
+    finally:
+        history.shutil.disk_usage, history.np.save = original_usage, original_save
+
+
+failures = []
+for regression in (check_atomic_download, check_legacy_truncated_zip, check_missing_published_on_resume, check_parallel_reserve):
+    try:
+        regression()
+    except Exception as error:
+        failures.append(f'{regression.__name__}: {type(error).__name__}: {error}')
+assert not failures, '\n'.join(failures)
+print('PASS: parsing, checksums, gaps, atomic download/resume, published missing days and shared disk reserve')

```
