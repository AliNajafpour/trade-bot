"""Run independent strategy accounts on identical untouched Wallex candles."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .data import load_market
from .engine import backtest

STRATEGIES = ('rules', 'trees', 'neural', 'kronos', 'agent', 'fusion')


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_expert(directory, expected_data_sha256=None, expected_config=None):
    saved = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if saved['status'] != 'ok':
        raise ValueError(f"Expert is blocked: {directory}")
    if sha(directory/'outputs.npz') != saved.get('outputs_sha256'):
        raise ValueError(f'Expert output hash mismatch: {directory}')
    if expected_data_sha256 is not None and saved.get('data_sha256') != expected_data_sha256:
        raise ValueError(f'Expert data identity mismatch: {directory}')
    if expected_config is not None and saved.get('execution_config') != expected_config:
        raise ValueError(f'Expert execution assumptions differ: {directory}')
    for name in (saved['strategy']+'.py','features.py','engine.py'):
        if saved.get('source_sha256',{}).get(name) != sha(Path(__file__).with_name(name)):
            raise ValueError(f'Expert code changed since generation: {name}')
    with np.load(directory/'outputs.npz', allow_pickle=False) as arrays:
        result = {key: arrays[key].copy() for key in arrays.files}
    result['metadata'] = saved['metadata']
    return result


def run_market(path, root, names, config, extra):
    path, root = Path(path), Path(root)
    frame = load_market(path)
    symbol = path.name.split('_')[0]
    n = len(frame); train_end, test_start = int(n*.6), int(n*.8)
    base_val_end = (train_end+test_start)//2
    if n < 300:
        raise ValueError('At least 300 closed candles required')
    identity = dict(data_path=str(path.resolve()), data_sha256=sha(path), rows=n,
        train_end=train_end, base_validation_end=base_val_end, fusion_validation_end=test_start,
        test_start=test_start, test_first_open=frame.timestamp.iloc[test_start].isoformat(),
        config=config, extra=extra, currency='TMN', source='https://api.wallex.ir',
        clock='timestamp is candle OPEN; information becomes available at candle CLOSE',
        selection='TRAIN0-60%; base selection60-70%; fusion selection70-80%; final TEST80-100%',
        costs='Assumed per-side account fees; slippage includes snapshot half-spread plus buffer, not historical quotes',
        validity='One exposed historical holdout, not a prospective profitability guarantee or actual exchange fills')
    market_dir=root/symbol; market_dir.mkdir(parents=True,exist_ok=True)
    identity_path=market_dir/'identity.json'
    if identity_path.exists():
        old=json.loads(identity_path.read_text(encoding='utf-8'))
        if old['data_sha256']!=identity['data_sha256'] or old['config']!=config:
            raise ValueError('Run identity changed; choose a new run directory')
    else:
        save(identity_path,identity)
    # Cash and same-cap passive position baselines share every execution assumption.
    passive=np.full(n,float(config.get('max_exposure',.5)))
    save(market_dir/'baselines.json', {'engine_sha256':sha(Path(__file__).with_name('engine.py')),
        'cash':backtest(frame,np.zeros(n),config,start=test_start),
        'constant_allocation':backtest(frame,passive,config,start=test_start)})
    for name in names:
        if name not in STRATEGIES:
            raise ValueError(f'Unknown strategy {name}')
        directory=market_dir/name; directory.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        print(f'START {symbol} {name} rows={n} test={n-test_start}',flush=True)
        try:
            settings={**config,**extra}
            if name=='agent':
                settings.setdefault('agent_start_index',train_end)
            if name=='fusion':
                experts={}
                for candidate in STRATEGIES[:-1]:
                    candidate_dir=market_dir/candidate
                    if (candidate_dir/'result.json').exists():
                        try:
                            experts[candidate]=load_expert(candidate_dir,identity['data_sha256'],config)
                        except ValueError:
                            pass
                settings['experts']=experts
            module=importlib.import_module('.'+name,__package__)
            source_paths=(Path(module.__file__),Path(__file__),Path(__file__).with_name('features.py'),Path(__file__).with_name('engine.py'))
            source_hashes={p.name:sha(p) for p in source_paths}
            generated=module.generate(frame,base_val_end if name=='fusion' else train_end,
                test_start if name=='fusion' else base_val_end,settings,directory)
            if generated.get('status','ok')!='ok' or generated.get('target') is None:
                raise RuntimeError(json.dumps(clean(generated),ensure_ascii=False))
            if generated.get('metadata',{}).get('nonbacktestable'):
                raise ValueError('Prospective context decisions cannot be backtested on historical candles')
            target=np.asarray(generated['target'],dtype=float)
            primary=backtest(frame,target,config,start=test_start)
            stress_config={**config,'fee_bps':config.get('fee_bps',20)+10,
                'slippage_bps':config.get('slippage_bps',15)+10}
            stress=backtest(frame,target,stress_config,start=test_start)
            no_cost=backtest(frame,target,{**config,'fee_bps':0,'slippage_bps':0},start=test_start)
            arrays={k:np.asarray(v) for k,v in generated.items() if k in ('target','score','validation_target','forecast_bp')}
            np.savez_compressed(directory/'outputs.npz',**arrays)
            metadata=generated.get('metadata',{})
            decision=pd.DataFrame({'timestamp':frame.timestamp,'target_allocation':target})
            if 'score' in arrays:
                decision['score']=arrays['score']
            decision['phase']=np.where(np.arange(n)<train_end,'train',np.where(np.arange(n)<test_start,'validation','test'))
            decision.to_csv(directory/'decisions.csv',index=False)
            result=dict(status='ok',strategy=name,symbol=symbol,metadata=metadata,
                primary=primary,stress=stress,no_cost=no_cost,elapsed_seconds=time.perf_counter()-started,
                train_end=train_end,base_validation_end=base_val_end,test_start=test_start,
                source_sha256=source_hashes,
                outputs_sha256=sha(directory/'outputs.npz'),data_sha256=identity['data_sha256'],
                execution_config=config,
                completed_utc=datetime.now(timezone.utc).isoformat())
            if any(source_hashes[p.name]!=sha(p) for p in source_paths):
                raise RuntimeError('Source changed during execution; rerun in a fresh process to bind actual code')
            save(directory/'result.json',result)
            print(f"DONE {symbol} {name} return={primary['metrics']['total_return']:.6%} fills={primary['metrics']['fill_count']} seconds={result['elapsed_seconds']:.1f}",flush=True)
        except Exception as exc:
            result=dict(status='blocked',strategy=name,symbol=symbol,error=f'{type(exc).__name__}: {exc}',
                traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started,
                completed_utc=datetime.now(timezone.utc).isoformat(),data_sha256=identity['data_sha256'])
            save(directory/'result.json',result)
            print(f"BLOCKED {symbol} {name}: {result['error'][:500]}",flush=True)
    return summarize(root)


def summarize(root):
    root=Path(root); rows=[]
    for path in sorted(root.glob('*/*/result.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        row={k:value.get(k) for k in ('symbol','strategy','status','elapsed_seconds','error')}
        if value['status']=='ok':
            m=value['primary']['metrics']; s=value['stress']['metrics']; z=value['no_cost']['metrics']
            row.update(return_pct=m['total_return']*100,stress_return_pct=s['total_return']*100,
                no_cost_return_pct=z['total_return']*100,final_toman=m['final_equity'],
                drawdown_pct=m['max_drawdown']*100,roundtrips=m['roundtrip_count'],fills=m['fill_count'],
                fees_toman=m['fees_paid'],slippage_toman=m['slippage_paid'],win_rate=m['win_rate'])
        rows.append(row)
    save(root/'summary.json',rows)
    if rows:
        pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/market/BTCTMN_15m.csv')
    p.add_argument('--run-dir',default='runs/wallex-20261021')
    p.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=list(STRATEGIES))
    p.add_argument('--fee-bps',type=float,default=20)
    p.add_argument('--slippage-bps',type=float,default=15)
    p.add_argument('--min-notional',type=float,default=50_000)
    p.add_argument('--max-calls',type=int,default=10)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--device',default='cpu')
    p.add_argument('--kronos-device',default='cpu')
    p.add_argument('--kronos-cache-dir')
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args()
    if args.summarize_only:
        print(json.dumps(summarize(args.run_dir),indent=2));return
    config=dict(fee_bps=args.fee_bps,slippage_bps=args.slippage_bps,min_notional=args.min_notional,
        initial_cash=5_000_000.,max_exposure=.5,horizon=3,seed=17,
        max_drawdown_stop=.08,daily_loss_stop=.03)
    extra=dict(max_calls=args.max_calls,epochs=args.epochs,device=args.device,
        kronos_device=args.kronos_device,kronos_context=64,kronos_stride=8,
        kronos_max_samples=2048,kronos_batch_size=16,llm_timeout=120)
    if args.kronos_cache_dir:
        extra['kronos_cache_dir']=args.kronos_cache_dir
    run_market(args.data,args.run_dir,args.strategies,config,extra)


if __name__=='__main__':
    main()
