"""Offline release checks. No market requests, model-provider calls or order submission."""
import ast
from concurrent.futures import Future
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for file in [*ROOT.glob('*.py'),*(ROOT/'wallex_lab').rglob('*.py'),*(ROOT/'scripts').glob('*.py')]:
    ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
metadata=json.loads((ROOT/'models/barrier-survival/late/model.json').read_text(encoding='utf-8'))
assert len(metadata['members'])==3
for member in metadata['members']:
    assert hashlib.sha256((ROOT/'models/barrier-survival/late'/member['file']).read_bytes()).hexdigest()==member['sha256']
cfg=json.loads((ROOT/'config/live-execution.json').read_text(encoding='utf-8'))
assert cfg['enabled'] is False and cfg['account_binding'] is None and cfg['stop_capabilities_verified'] is False
assert 'fetch(' not in (ROOT/'video-demo/dashboard.html').read_text(encoding='utf-8')
assert (ROOT/'third_party/kronos-67b630e/LICENSE').exists()
spec=importlib.util.spec_from_file_location('release_run',ROOT/'run.py');launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
class Pool:
    def __init__(self,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def submit(self,fn,*args):
        future=Future()
        try:future.set_result(fn(*args))
        except Exception as error:future.set_exception(error)
        return future
with TemporaryDirectory() as tmp:
    ticks=[]
    def tick(_):
        ticks.append(1)
        if len(ticks)==3:raise KeyboardInterrupt
    config=json.loads((ROOT/'config/weekly-source.example.json').read_text(encoding='utf-8'))
    with patch.object(launcher,'ROOT',Path(tmp)),patch.object(launcher,'ThreadPoolExecutor',Pool),\
         patch.object(launcher,'time',SimpleNamespace(sleep=tick,monotonic=lambda:float(len(ticks)))),\
         patch('wallex_lab.council_runtime.fetch_quote',return_value={'bid':100,'ask':101,'available_at':'2026-10-23T00:00:00Z'}) as quote,\
         patch('wallex_lab.intelligence_sources.collect_bundle',return_value={}) as bundle,\
         patch('wallex_lab.intelligence_reports.ReportStore') as store:
        store.return_value.refresh.return_value={'status':'ok'}
        launcher.source(config)
    state=json.loads((Path(tmp)/config['run_dir']/'runtime-state.json').read_text(encoding='utf-8'))
    assert len(state['quotes'])==4 and not state['errors']
    assert quote.call_count==4 and bundle.call_count==2,'Collector duplicated a worker before its scheduled refresh'
print('release: Python syntax, three final weight hashes, disabled real execution, offline demo and source scheduler OK')
