# Binance acquisition task report

Task 1 implementation and runnable checks are complete. The full real acquisition is running, not complete. No old experiment sources, processes, caches or GPU state were changed by this task.

## Reproduction and current owner

From `workspace:`:

```powershell
& 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe' -u scripts/refresh_barrier_binance_history.py --start 2023-01-01 --end 2026-10-23 --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT --output data/barrier-binance-history-20261023 --workers 2 --resume
```

The hidden background launch used this exact argument list. Current launcher PID is **2184**; actual acquisition worker PID is **18748**, also recorded in `data/barrier-binance-history-20261023/progress.json`. `acquisition.pid` contains the launcher PID. Logs are `acquisition-resume1.stdout.log` and `acquisition-resume1.stderr.log` inside the source root. The prior failed attempt's logs are preserved as `acquisition.stdout.log` and `acquisition.stderr.log`.

At 2026-10-23 09:02:48 UTC, stage was `acquisition`, status `running`, with BTCUSDT downloading 2023-11 and ETHUSDT downloading 2023-12. A read immediately afterward counted 33 verified partition receipts and 1,428,319 canonical rows. Other symbols retained verified January/February partitions from the initial attempt and await their turn. Six March error receipts from that attempt remain explicitly errors until their symbol is resumed. The new worker has passed March for BTC and ETH.

Full eight-symbol coverage, 2026 archive availability, final arrays and a final ready manifest have **not** been claimed. The immutable protocol ends exclusively at 2026-10-23 00:00 UTC. Missing monthly archives trigger official daily probes; daily HTTP 404s remain explicit missing periods and produce a partial source. Other HTTP/parser/checksum errors fail the acquisition. There is no REST fallback in this version.

## Verified sources and checks

The assert check was written first and failed with `AssertionError: Binance acquisition implementation is missing`. It now passes:

```powershell
& 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe' checks/check_barrier_binance_history.py
```

It checks millisecond/microsecond parity, the nine-column mapping, malformed prices/flows/counts, missing minutes, incomplete cutoff, checksum/hash rejection, interrupted download reuse, corrupt final arrays, missing archives, redundant-file removal and cleanup after an interrupted final commit. The official shortened-close regression was also observed red before its fix. All three new Python files passed compilation. No unrelated training suite was executed.

Before full launch, actual official HTTPS ZIPs and their `.CHECKSUM` files were downloaded, hashed and parsed:

| Official archive | Rows | ZIP SHA256 |
| --- | ---: | --- |
| `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-01.zip` | 44,640 | `664116c2f33465d191684887f0d4834c02eb0a2635ada43eca855a99e6b6b0c5` |
| `https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-01-01.zip` | 1,440 | `10a12909f1b0e3fcc6b7f502e5ea9be5d1ba3455dd8ab16cc61c8650640ba7c0` |

Both had complete observed minute coverage and expected dates. Probe files were removed by their temporary-directory context. Normal sandbox sockets were denied, so public downloads and the background launch used approved shell escalation with ordinary TLS validation.

The first full attempt found an actual March 2023 maintenance bar with open `1679661540000` ms and close `1679661581646` ms, a 41.646-second zero-trade bar. The parser now accepts official closes inside their nominal minute and records `parser_audit.shortened_close_rows`; overlong/backward closes remain invalid. Nominal whole-minute completion is still required before cutoff. Before restarting, both old PIDs were confirmed absent, the manifest/progress were failed, and the exclusive source lock was acquired successfully. A stale-state automatic approval rejection was resolved by these fresh checks before the approved restart.

## Interface and retention

`acquire_history(output, start, end, symbols, workers=2)` writes immutable `protocol.json`, per-archive receipts, atomic progress/manifest updates and final `market/SYMBOL-values.npy` and `market/SYMBOL-times.npy`. Symbol records map directly under `manifest.symbols` and include absolute `values_path`, `times_path`, `values_sha256`, `times_sha256`, `rows`, first/last open nanoseconds, observed source coverage and quality.

Values are float64 `N x 9`, ordered open, high, low, close, volume, quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume. Times are sorted unique int64 UTC candle-open nanoseconds. No CSV/window duplication or interpolation is introduced. Quality includes expected/observed/missing minutes, internal/edge gaps, coverage fraction, zero-volume/trade fractions and flat fraction.

`archive_coverage_complete` distinguishes complete archive acquisition from `coverage_complete`, which reports actual calendar-minute coverage. A valid complete archive set may be ready despite genuine exchange gaps, with those gaps retained explicitly. Uncovered daily archive periods produce a partial manifest.

Concurrency is bounded to two symbol workers. Every download/materialization checks the data drive's free space with a minimum **5 GiB** reserve and a small additional margin. An OS file lock prevents two acquisitions from owning the same source.

After an official ZIP verifies and its canonical partition arrays validate, the ZIP and local checksum file are deleted; URL, official checksum text, ZIP SHA256, response headers, parser audit and canonical hashes remain in the receipt. After final symbol arrays commit and verify, redundant partition arrays are deleted. Resume completes interrupted cleanup. It does not delete anything outside this new source root.

At the 09:02 snapshot, receipt-accounted exact redundant bytes reclaimed were **61,071,715 bytes**, with **124,297,169 bytes** remaining under the new source root. Counts continue changing while acquisition runs. Drive free space at that read was **10,136,043,520 bytes**. No duplicate probe outputs remain. Source checksums and archive URLs permit reproduction; original compressed archives are deliberately not retained.

The running worker imported the module before the final additional cleanup-resume check/helper was added. That addition affects recovery after interruption and does not change the already-running normal download path. A future resume imports the checked current module.

## Review fixes, 2026-10-23 09:17 UTC

Three operational defects from review are fixed for future launches/resumes. The healthy existing acquisition was not stopped or duplicated. The data schema, parsed numeric values and exclusive cutoff are unchanged.

- Downloads write `*.part` and publish with `os.replace` only after the whole HTTP body arrives. A pre-existing completed ZIP must pass its checksum before reuse. Legacy truncated ZIPs now record an `invalid_cache` attempt with the failed hash and reacquire the official source instead of failing forever. Prior missing/error attempts remain in `attempt_errors`.
- Resume retries missing archive dispositions. A previously committed partial symbol is rebuilt from its verified existing arrays plus newly available archives. Existing source partitions can be reconstructed and hash-checked from the committed final arrays without fetching unchanged history again. The previous final pair is retained as hash-checked recovery backups until the replacement pair and symbol receipt commit. A simulated interruption between replacement of values and times recovers successfully on the next resume.
- One shared lock covers the reserve check and bulk write for HTTP bodies, partition saves and final consolidation. Two workers can no longer both spend the same free bytes. This deliberately serializes bulk writes; public requests and parsing can still overlap. The small metadata allowance remains separate. Other programs can consume disk independently, so checks continue during HTTP writes.

Four regressions were observed failing before these changes: interrupted download publication, legacy truncated-cache recovery, newly published missing-day retry, and concurrent reserve spending. All now pass in `checks/check_barrier_binance_history.py`; the interrupted final-pair replacement check also passes. Compilation passed again. No extra framework, dependency or GPU work was introduced.

New partition/final/manifest records include producer revision `atomic-resume-reserve-v2` and the SHA256 of the module bytes loaded at import. Checked module SHA256 is `2176476e44efe8966b83c4950826a8d06b5075f3135d874abc0aacf01364bdee`. The already-running worker continues with its prior imported implementation; its existing receipts are not relabeled with this new producer hash. The healthy run currently has ample reserve; the operational fixes take effect on its next resume.

At **2026-10-23 09:17:49 UTC**, both launcher **2184** and worker **18748** existed, and the exclusive source lock was held. Progress advanced to SOLUSDT **2025-02** and BNBUSDT **2024-08**, after BTC/ETH had reached their **2026-10-22** final daily archives. `dispositions_this_session` was **178**, stderr remained empty, and D had **13,020,114,944 bytes** free. This remains an active acquisition, not a claim of eight-symbol completion.
