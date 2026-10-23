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
    template=json.loads((ROOT/'config/live-cloud.example.json').read_text('utf-8'))
    model=ROOT/template['model']
    identity=dict(model=str(model),model_metadata_sha256=hashlib.sha256((model/'model.json').read_bytes()).hexdigest())
    if not execution.exists():
        config=dict(template,run_dir=str(data/'live'),source=str(data/'source'),**identity)
        # Exclusive creation preserves the user's state across redeploys, without a first-boot race.
        try:
            with execution.open('x',encoding='utf-8') as f:json.dump(config,f,indent=2)
        except FileExistsError:pass
    config=json.loads(execution.read_text('utf-8'))
    if os.environ.get('WALLEX_FORCE_DISABLE','false').lower()=='true':
        from .context import save_json
        config['enabled']=False;save_json(execution,config)
    old_model=ROOT/'models/barrier-survival/late'
    old_hash=hashlib.sha256((old_model/'model.json').read_bytes()).hexdigest()
    if (config.get('enabled') is False and not config.get('universe_upgrade_from_sha256')
            and config.get('symbols')==['BTCUSDT','ETHUSDT','SOLUSDT','SUSDT']
            and len(template['symbols'])==10 and config.get('model_metadata_sha256')==old_hash):
        # Upgrade only the recognized, disabled four-asset bundle; keep the owner's account and risk settings.
        from .context import save_json
        config.update(symbols=template['symbols'],**identity,universe_upgrade_from_sha256=old_hash,
            proxy_venues={**template.get('proxy_venues',{}),**config.get('proxy_venues',{})},
            llm_chart_symbols=template.get('llm_chart_symbols',[]),
            manager_scan_seconds=template.get('manager_scan_seconds',600))
        config.update({key:template[key] for key in ('probability_symbols','probability_model',
            'probability_model_metadata_sha256','legacy_technical_symbols','legacy_model_dirs','legacy_timeout_seconds')})
        save_json(execution,config)
    if (config.get('enabled') is False and not config.get('technical_split_version')
            and config.get('symbols')==template['symbols']
            and config.get('model_metadata_sha256')==old_hash
            and config.get('llm_chart_symbols')==template['symbols'][4:]):
        from .context import save_json
        config.update({key:template[key] for key in ('llm_chart_symbols','probability_symbols',
            'probability_model','probability_model_metadata_sha256','legacy_technical_symbols',
            'legacy_model_dirs','legacy_timeout_seconds')})
        config['technical_split_version']=1
        save_json(execution,config)
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
