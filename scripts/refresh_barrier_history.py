"""Acquire a frozen, current native-M1 cohort from public Wallex endpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import pandas as pd
from wallex_lab.data import PublicClient,download_history,utc_now
from wallex_lab.barrier_dataset import sha256,write_json


def protocol_for(until,days,symbols):
    end=pd.Timestamp(until)
    if end.tzinfo is None or days<1 or not symbols or len(set(symbols))!=len(symbols):raise ValueError('Aware cutoff, positive days and unique symbols required')
    end=end.tz_convert('UTC').floor('min');start=end-pd.Timedelta(days=days)
    return dict(created_utc=utc_now(),frozen_before_download=True,requested_from_utc=start.isoformat(),
        requested_to_exclusive_utc=end.isoformat(),requested_days=days,symbols=symbols,quote_currency='USDT',
        source='https://api.wallex.ir/v1/udf/history',resolution='1',authentication='none',
        missing_policy='Retain gaps, never interpolate; record actual returned coverage.',
        timestamp_semantics='UTC candle-open, only candles fully closed by frozen cutoff.')


def acquire_one(root,symbol,protocol):
    start=int(pd.Timestamp(protocol['requested_from_utc']).timestamp());end=int(pd.Timestamp(protocol['requested_to_exclusive_utc']).timestamp())
    report=download_history(PublicClient(root,timeout=25),symbol,start,end,interval_seconds=60)
    report['native_quality']['price_unit']=report['quality']['price_unit']='USDT'
    report.update(native_path=str(Path(report['native_path']).resolve()),native_sha256=sha256(report['native_path']),
        price_conversion='none',coverage_fraction_1m=report['native_quality']['rows']/((end-start)//60))
    write_json(root/'market'/f'{symbol}_1m.quality.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until',required=True);parser.add_argument('--days',type=int,default=365)
    parser.add_argument('--root',default='data/barrier-history-dense-20261022')
    parser.add_argument('--symbols',nargs='+',default=['BTCUSDT','ETHUSDT','SOLUSDT','SUSDT'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    expected=protocol_for(args.until,args.days,args.symbols)
    if root.exists():
        if not args.resume:raise FileExistsError('Existing acquisition is preserved; use --resume for an interrupted acquisition')
        protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
        for key in ('requested_from_utc','requested_to_exclusive_utc','symbols'):
            if expected[key]!=protocol[key]:raise ValueError('Resume acquisition arguments differ')
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if sha256(root/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('Acquisition protocol mismatch')
    else:
        root.mkdir(parents=True);protocol=expected;write_json(root/'protocol.json',protocol)
        manifest=dict(status='downloading',protocol_sha256=sha256(root/'protocol.json'),symbols={},errors=[])
        write_json(root/'manifest.json',manifest)
    todo=[]
    for symbol in args.symbols:
        record=manifest['symbols'].get(symbol,{})
        if record.get('status')=='ready':
            if sha256(record['data']['native_path'])!=record['data']['native_sha256']:raise ValueError('Cached native source changed')
        else:todo.append(symbol)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(acquire_one,root,symbol,protocol):symbol for symbol in todo}
        for job in as_completed(jobs):
            symbol=jobs[job]
            try:
                report=job.result();manifest['symbols'][symbol]=dict(status='ready',data=report)
                print('READY_ASSET '+symbol+' '+str(report['native_quality']),flush=True)
            except Exception as error:
                manifest['symbols'][symbol]=dict(status='failed',error=f'{type(error).__name__}: {error}')
                print('FAILED_ASSET '+symbol+' '+str(error),flush=True)
            write_json(root/'manifest.json',manifest)
    manifest.update(status='ready' if all(manifest['symbols'][s].get('status')=='ready' for s in args.symbols) else 'partial',completed_utc=utc_now())
    write_json(root/'manifest.json',manifest)
    if manifest['status']!='ready':raise RuntimeError('Partial acquisition retained; rerun with --resume')
    print('ACQUISITION_COMPLETE '+str(root.resolve()),flush=True)


if __name__=='__main__':main()
