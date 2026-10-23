"""Acquire frozen official Binance spot M1 archives without touching old runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wallex_lab.barrier_binance_history import SYMBOLS, acquire_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2026-10-23', help='Exclusive UTC midnight cutoff')
    parser.add_argument('--symbols', nargs='+', default=list(SYMBOLS))
    parser.add_argument('--output', default='data/barrier-binance-history-20261023')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (Path(args.output) / 'protocol.json').exists() and not args.resume:
        raise FileExistsError('Source preserved. Use --resume to verify and continue the same frozen protocol.')
    manifest = acquire_history(args.output, args.start, args.end, args.symbols, args.workers)
    print(json.dumps(dict(status=manifest['status'], coverage_complete=manifest['coverage_complete'],
                          rows={s: r['rows'] for s, r in manifest['symbols'].items()})), flush=True)
    return 0 if manifest['status'] == 'ready' else 2


if __name__ == '__main__':
    raise SystemExit(main())
