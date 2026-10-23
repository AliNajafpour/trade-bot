"""Verify the bundled model or launch the Weekly/late paper system. Never places real orders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models/barrier-survival/late'


def verify_model():
    import numpy as np
    import pandas as pd
    import torch
    from wallex_lab.barrier_survival import SurvivalPredictor
    torch.set_num_threads(1)
    predictor=SurvivalPredictor.load(MODEL)
    count=predictor.metadata['minimum_history_minutes']
    decision=pd.Timestamp('2026-10-22T16:00:00Z')
    close=80000*np.exp(.0001*np.sin(np.arange(count)/11))
    history=pd.DataFrame(dict(timestamp=pd.date_range(end=decision-pd.Timedelta(minutes=1),periods=count,freq='min'),
        open=close,high=close*1.0002,low=close*.9998,close=close,volume=np.ones(count)))
    result=predictor.predict(history,'BTCUSDT',decision,float(close[-1]),tp_fraction=.02,sl_fraction=.02)
    for row in result['forecasts']:
        p=[row[k] for k in ('p_tp_first','p_sl_first','p_no_touch')]
        assert all(np.isfinite(p)) and all(0<=x<=1 for x in p) and abs(sum(p)-1)<1e-5
    return dict(status='passed',members=len(predictor.models),context_minutes=count,device='cpu',
        weight_hashes_verified=True,input='synthetic diagnostic candles, not a trading result',
        forecast=next(r for r in result['forecasts'] if r['horizon_minutes']==1440))


def source(config):
    from wallex_lab.context import save_json, timestamp
    from wallex_lab.council_runtime import fetch_quote
    from wallex_lab.intelligence_sources import collect_bundle
    from wallex_lab.intelligence_reports import ReportStore
    root=ROOT/config['run_dir'];root.mkdir(parents=True,exist_ok=True)
    store=ReportStore(root/'reports');jobs={};last={};quotes={};errors={}
    def report(kind):
        bundle=collect_bundle(root/'sources',kind,asof=None,config=config['source_config'])
        asof=max(timestamp(datetime.now(timezone.utc).isoformat()),timestamp(bundle.get('available_at') or datetime.now(timezone.utc).isoformat())).isoformat()
        result=store.refresh(kind,bundle,asof,config[kind+'_seconds'])
        if result.get('status') not in ('ok','partial'):raise RuntimeError('Report unavailable')
        return result
    with ThreadPoolExecutor(max_workers=4) as books,ThreadPoolExecutor(max_workers=2) as reports:
        try:
            while True:
                clock=time.monotonic()
                for key,job in list(jobs.items()):
                    if not job.done():continue
                    del jobs[key]
                    try:
                        value=job.result()
                        if key.startswith('quote:'):quotes[key[6:]]=value
                        errors.pop(key,None)
                    except Exception as error:errors[key]=dict(error_type=type(error).__name__,at=datetime.now(timezone.utc).isoformat())
                for symbol in config['symbols']:
                    key='quote:'+symbol
                    if key not in jobs and clock-last.get(key,-1e9)>=config['quote_seconds']:
                        last[key]=clock;jobs[key]=books.submit(fetch_quote,dict(run_dir=str(root)),symbol)
                for kind in ('market','macro'):
                    if kind not in jobs and clock-last.get(kind,-1e9)>=config[kind+'_seconds']:
                        last[kind]=clock;jobs[kind]=reports.submit(report,kind)
                save_json(root/'runtime-state.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),
                    status='running',paper_only=True,quotes=dict(quotes),errors=dict(errors),
                    workers=list(jobs),role='Shared public quotes and report collection; no technical model or order execution'))
                time.sleep(1)
        except KeyboardInterrupt:pass


def paper():
    verify_model()
    template=json.loads((ROOT/'config/weekly-paper.example.json').read_text(encoding='utf-8'))
    template['model_metadata_sha256']=hashlib.sha256((ROOT/template['model']/'model.json').read_bytes()).hexdigest()
    root=ROOT/template['run_dir'];root.mkdir(parents=True,exist_ok=True)
    path=root/'launch.json';path.write_text(json.dumps(template,indent=2),encoding='utf-8')
    children=[]
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    try:
        with (root/'source.log').open('a',encoding='utf-8') as log:
            children.append(subprocess.Popen([sys.executable,'-B',str(ROOT/'run.py'),'source'],cwd=ROOT,stdout=log,stderr=log,creationflags=flags))
            children.append(subprocess.Popen([sys.executable,'-B','-m','wallex_lab.survival_news','--config',str(path)],cwd=ROOT,creationflags=flags))
            print('Paper dashboard: http://127.0.0.1:'+str(template['port'])+' ; Ctrl+C stops paper monitoring.',flush=True)
            while all(p.poll() is None for p in children):time.sleep(1)
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('A paper worker stopped; inspect run status and logs')
    except KeyboardInterrupt:pass
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('verify','paper','source'))
    args=parser.parse_args();os.chdir(ROOT)
    if args.command=='verify':print(json.dumps(verify_model()));return
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    if args.command=='source':source(json.loads((ROOT/'config/weekly-source.example.json').read_text(encoding='utf-8')))
    else:paper()


if __name__=='__main__':main()
