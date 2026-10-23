"""Cloud supervisor. Bootstraps disabled execution; only the user's control can arm it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def bootstrap(data_dir):
    data=Path(data_dir).resolve();data.mkdir(parents=True,exist_ok=True)
    execution=data/'execution.json'
    if not execution.exists():
        config=json.loads((ROOT/'config/live-cloud.example.json').read_text('utf-8'))
        config.update(run_dir=str(data/'live'),source=str(data/'source'),
            model=str(ROOT/'models/barrier-survival/late'),
            model_metadata_sha256=hashlib.sha256((ROOT/'models/barrier-survival/late/model.json').read_bytes()).hexdigest())
        # Exclusive creation preserves the user's state across redeploys, without a first-boot race.
        try:
            with execution.open('x',encoding='utf-8') as f:json.dump(config,f,indent=2)
        except FileExistsError:pass
    config=json.loads(execution.read_text('utf-8'))
    if os.environ.get('WALLEX_FORCE_DISABLE','false').lower()=='true':
        from .context import save_json
        config['enabled']=False;save_json(execution,config)
    source=json.loads((ROOT/'config/weekly-source.example.json').read_text('utf-8'))
    source['run_dir']=str(data/'source')
    source['symbols']=[s[:-4]+'/USDT' for s in config['symbols']]
    source['source_config'].update(timeout_seconds=12,max_workers=3)
    return execution,source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component',choices=('all','source'),nargs='?',default='all')
    args=parser.parse_args();os.chdir(ROOT)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key,'1')
    data=Path(os.environ.get('WALLEX_DATA_DIR','/data'))
    config,source_config=bootstrap(data)
    if args.component=='source':
        from run import source
        source(source_config);return
    password=os.environ.get('WALLEX_DASHBOARD_PASSWORD','')
    if len(password)<20:raise ValueError('A private dashboard password of at least20characters is required')
    if not os.environ.get('WALLEX_API_KEY'):raise ValueError('WALLEX_API_KEY is required for authenticated read-only preflight')
    stop=False;children=[]
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    commands=[
        [sys.executable,'-B','-m','wallex_lab.cloud','source'],
        [sys.executable,'-B','-m','wallex_lab.live_runner','--config',str(config)],
        [sys.executable,'-B','-m','wallex_lab.deployment_web','--config',str(config),'--host','0.0.0.0','--port',os.environ.get('PORT','8080')],
    ]
    try:
        for command in commands:children.append(subprocess.Popen(command,cwd=ROOT))
        print('Cloud service started; first-boot execution is disabled. Control: /control',flush=True)
        while not stop:
            failed=[p.returncode for p in children if p.poll() is not None]
            if failed:raise RuntimeError('A cloud worker stopped; supervisor exiting for platform restart')
            time.sleep(1)
    finally:
        for child in reversed(children):
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
