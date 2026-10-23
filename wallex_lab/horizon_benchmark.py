"""Calendar-held-out multi-horizon forecasts and fee-sourced spot comparisons."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from .data import aggregate_complete, load_market, quality_report
from .engine import backtest
from .features import _validated_frame
from .runner import save, sha

SYMBOLS = ('BTCTMN', 'ETHTMN', 'SOLTMN', 'USDTTMN')
TIMEFRAMES = (5, 15, 60)
HORIZONS = (60, 180, 360, 1440)


def make_targets(frame, predictions, start, end, horizon, threshold, exposure=.5):
    """Close-known, nonoverlapping fixed-horizon plans; gaps cancel stale entries."""
    _, _, gaps, _ = _validated_frame(frame)
    predictions = np.asarray(predictions, dtype=float)
    if (predictions.shape != (len(frame),) or not 0 <= start < end <= len(frame)
            or type(horizon) is not int or horizon < 1 or not np.isfinite(threshold)
            or not 0 <= exposure <= 1):
        raise ValueError('Invalid prediction trace, range, horizon or threshold')
    target = np.zeros(len(frame))
    exits = []
    entry, exit_index = None, None
    for index in range(start, end):
        if gaps[index]:
            if exit_index in exits:
                exits.remove(exit_index)
            entry, exit_index = None, None
        if exit_index is not None and index >= exit_index:
            entry, exit_index = None, None
        if exit_index is not None:
            if index < exit_index - 1:
                target[index] = exposure
            continue
        if index + 1 >= end or not np.isfinite(predictions[index]) or predictions[index] <= threshold:
            continue
        entry = index + 1
        exit_index = min(entry + horizon, end - 1)
        if entry >= exit_index:
            entry, exit_index = None, None
            continue
        target[index] = exposure
        if exit_index < end:
            exits.append(exit_index)
    return target, exits


def combinations(predictions):
    result = dict(predictions)
    for names in itertools.combinations(('hgb', 'gru', 'kronos'), 2):
        values = np.vstack([predictions[name] for name in names])
        result['mean_' + '_'.join(names)] = values.mean(axis=0)
    values = np.vstack([predictions[name] for name in ('hgb', 'gru', 'kronos')])
    result['mean_all'] = values.mean(axis=0)
    result['median_all'] = np.median(values, axis=0)
    return result


def evaluate(frame, score, start, end, horizon, threshold, config):
    target, exits = make_targets(frame, score, start, end, horizon, threshold, config['max_exposure'])
    # Start with cash at each window; no pre-window allocation leaks into the account.
    target[:start] = 0
    simulated = backtest(frame.iloc[:end].copy(), target[:end],
        {**config, 'forced_exit_indices': [index for index in exits if start <= index < end]}, start=start)
    return simulated, target, exits


def load_fee_profiles(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    profiles = data.get('benchmark_profiles')
    if profiles is None and data.get('status') == 'official_public_schedule_verified_account_unknown':
        source = data['source']
        if sha(source['raw_body_path']) != source['sha256']:
            raise ValueError('Official fee evidence hash mismatch')
        tiers = data['spot_public_schedule']['TMN']['tiers']
        profiles = [dict(name='tmn_level1_taker', fee_bps_per_side=tiers[0]['taker_bps'],
            source_url=source['url'], account_verified=False, condition='Published TMN first-tier taker scenario'),
            dict(name='tmn_level1_maker', fee_bps_per_side=tiers[0]['maker_bps'], source_url=source['url'],
                account_verified=False, condition='Fee-only maker sensitivity; candle proxy does not establish maker fills'),
            dict(name='tmn_level6_taker', fee_bps_per_side=tiers[-1]['taker_bps'], source_url=source['url'],
                account_verified=False, condition='Published high-volume tier sensitivity; account eligibility is unverified')]
    if not profiles or not all(isinstance(p.get('fee_bps_per_side'), (int, float))
            and not isinstance(p['fee_bps_per_side'], bool) and 0 <= p['fee_bps_per_side'] < 10000
            and p.get('source_url', '').startswith('https://') for p in profiles):
        raise ValueError('Verified official benchmark_profiles with source_url are required; no assumed fee default')
    return profiles


def run(root, fee_path, device='cuda', symbols=SYMBOLS, timeframes=TIMEFRAMES, horizons=HORIZONS):
    from .horizon_models import fit_forecasters
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    profiles = load_fee_profiles(fee_path)
    end = pd.Timestamp('2026-10-21T13:00:00Z')
    begin, train_cut, test_cut = end - pd.Timedelta(days=60), end - pd.Timedelta(days=24), end - pd.Timedelta(days=12)
    windows = [('test_12d', test_cut, end), ('first_6d', test_cut, end - pd.Timedelta(days=6)),
        ('last_6d', end - pd.Timedelta(days=6), end), ('last_3d', end - pd.Timedelta(days=3), end)]
    source_files = [Path(__file__), Path(__file__).with_name('horizon_models.py'),
        Path(__file__).with_name('engine.py'), Path(__file__).with_name('features.py')]
    sources = {p.name: sha(p) for p in source_files}
    identity = dict(timeframes=list(timeframes), horizons_minutes=list(horizons), symbols=list(symbols),
        end=end.isoformat(), train_cut=train_cut.isoformat(), test_cut=test_cut.isoformat(),
        sources=sources, fees_sha256=sha(fee_path), fees=profiles,
        protocol='Fixed TRAIN fits, purged VAL threshold selection, exposed retrospective TEST; no independent forward confirmation',
        execution='next-open spot; exact published fee scenario; historical spread/impact unknown and excluded, not guessed')
    identity_path = root / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Experiment identity changed; use a fresh output directory')
    save(identity_path, identity)
    rows = []
    for symbol in symbols:
        native_path = Path('data/market') / f'{symbol}_1m.csv'
        native = load_market(native_path)
        native = native.loc[(native.timestamp >= begin) & (native.timestamp + pd.Timedelta(minutes=1) <= end)].copy()
        for minutes in timeframes:
            frame, aggregation = aggregate_complete(native, minutes * 60, 60)
            directory = root / symbol / f'{minutes}m'
            directory.mkdir(parents=True, exist_ok=True)
            csv_path = directory / 'candles.csv'
            frame.to_csv(csv_path, index=False)
            save(directory / 'data.json', dict(native_sha256=sha(native_path), data_sha256=sha(csv_path),
                aggregation=aggregation, quality=quality_report(frame, minutes * 60)))
            train_end = int(frame.timestamp.searchsorted(train_cut))
            val_end = int(frame.timestamp.searchsorted(test_cut))
            for horizon_minutes in horizons:
                if horizon_minutes % minutes:
                    raise ValueError('Physical horizons must be exact multiples of candle duration')
                horizon = horizon_minutes // minutes
                case_dir = directory / f'h{horizon_minutes}'
                began = time.perf_counter()
                print(f'FIT {symbol} M{minutes} H{horizon_minutes}m rows={len(frame)}', flush=True)
                generated = fit_forecasters(frame, horizon, train_end, val_end, case_dir / 'models',
                    device=device, encoder_cache_dir=directory / 'encoder-cache')
                predictions = combinations(generated['predictions'])
                for profile in profiles:
                    profile_name, fee_bps = profile['name'], float(profile['fee_bps_per_side'])
                    config = dict(initial_cash=5_000_000., max_exposure=.5, min_notional=50_000.,
                        fee_bps=fee_bps, slippage_bps=0., max_drawdown_stop=.08, daily_loss_stop=.03)
                    cost = 2 * fee_bps / 10000
                    for name, score in predictions.items():
                        selection = []
                        # Fixed predeclared margins. VAL, not TEST, selects the threshold.
                        for margin in (0., .0025, .005):
                            threshold = cost + margin
                            result, _, _ = evaluate(frame, score, train_end, val_end, horizon, threshold, config)
                            m = result['metrics']
                            selection.append(dict(threshold=threshold, return_=m['total_return'], trades=m['roundtrip_count']))
                        chosen = max(selection, key=lambda trial: (trial['return_'], -trial['trades'], trial['threshold']))
                        threshold = chosen['threshold']
                        plan_dir = case_dir / profile_name / name
                        for window, lower, upper in windows:
                            start, stop = int(frame.timestamp.searchsorted(lower)), int(frame.timestamp.searchsorted(upper))
                            if stop <= start + 1:
                                continue
                            result, target, exits = evaluate(frame, score, start, stop, horizon, threshold, config)
                            metrics = result['metrics']
                            row = dict(symbol=symbol, timeframe_minutes=minutes, horizon_minutes=horizon_minutes,
                                model=name, fee_profile=profile_name, fee_bps_per_side=fee_bps, window=window,
                                trades=metrics['roundtrip_count'], fills=metrics['fill_count'],
                                return_pct=metrics['total_return'] * 100, pnl_toman=metrics['net_profit'],
                                win_rate=metrics['win_rate'], drawdown_pct=metrics['max_drawdown'] * 100,
                                fees_toman=metrics['fees_paid'], threshold=threshold,
                                validation_return_pct=chosen['return_'] * 100,
                                forced_subminimum_exits=metrics['forced_subminimum_exit_count'])
                            rows.append(row)
                            if window == 'test_12d':
                                save(plan_dir / 'test.json', dict(row=row, selection=selection, execution_config=config,
                                    start=start, end=stop, forced_exits=exits, result=result,
                                    data_sha256=sha(csv_path), model_metadata=generated['metadata']))
                                np.savez_compressed(plan_dir / 'targets.npz', target=target)
                        # A no-friction diagnostic uses the chosen frozen target policy, no retraining.
                        zero, _, _ = evaluate(frame, score, val_end, len(frame), horizon, threshold,
                            {**config, 'fee_bps':0.})
                        save(plan_dir / 'no_fee_diagnostic.json', zero['metrics'])
                print(f'DONE {symbol} M{minutes} H{horizon_minutes}m seconds={time.perf_counter()-began:.1f}', flush=True)
                pd.DataFrame(rows).to_csv(root / 'comparison.csv', index=False)
                save(root / 'progress.json', dict(completed_rows=len(rows), last_case=str(case_dir),
                    updated_utc=datetime.now(timezone.utc).isoformat()))
    assert all(sha(p) == sources[p.name] for p in source_files), 'Source changed during benchmark'
    save(root / 'completed.json', dict(rows=len(rows), completed_utc=datetime.now(timezone.utc).isoformat(), sources=sources))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/horizon-benchmark-20261021')
    parser.add_argument('--fees', default='config/wallex-official-fees.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--symbols', nargs='+', default=list(SYMBOLS), choices=SYMBOLS)
    parser.add_argument('--timeframes', nargs='+', type=int, default=list(TIMEFRAMES))
    parser.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    args = parser.parse_args()
    run(args.output, args.fees, args.device, args.symbols, args.timeframes, args.horizons)


if __name__ == '__main__':
    main()
