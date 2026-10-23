"""Offline persistent startup check. Never imports Torch or contacts exchanges."""
import json
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.cloud import bootstrap
from wallex_lab.cloud import ROOT

template=json.loads((ROOT/'config/live-cloud.example.json').read_text('utf-8'))
original=['BTCUSDT','ETHUSDT','SOLUSDT','SUSDT']

with TemporaryDirectory() as d,patch.dict('os.environ',{'WALLEX_FORCE_DISABLE':'false'}):
    path,source=bootstrap(d)
    config=json.loads(path.read_text())
    assert config['enabled'] is False and config['stop_capabilities_verified'] is False
    assert config['account_binding'] is None and config['max_capital_usdt'] is None
    assert Path(config['run_dir']).is_relative_to(Path(d).resolve())
    assert source['symbols']==[s[:-4]+'/USDT' for s in template['symbols']]
    assert config['model']==str(ROOT/template['model'])
    config['enabled']=True;config['max_capital_usdt']=7;path.write_text(json.dumps(config))
    again,_=bootstrap(d);assert json.loads(again.read_text())['enabled'] is True
    with patch.dict('os.environ',{'WALLEX_FORCE_DISABLE':'true'}):bootstrap(d)
    assert json.loads(path.read_text())['enabled'] is False

if len(template['symbols'])==10:
    with TemporaryDirectory() as d,patch.dict('os.environ',{'WALLEX_FORCE_DISABLE':'false'}):
        path,_=bootstrap(d)
        old_hash=hashlib.sha256((ROOT/'models/barrier-survival/late/model.json').read_bytes()).hexdigest()
        config=json.loads(path.read_text())
        config.update(symbols=original,model=str(ROOT/'models/barrier-survival/late'),
            model_metadata_sha256=old_hash,enabled=True,account_binding='owner-binding',
            max_capital_usdt=17,stop_capabilities_verified=True,max_open_positions=2,proxy_venues={'SOLUSDT':'mexc'})
        path.write_text(json.dumps(config))
        bootstrap(d)
        assert json.loads(path.read_text())==config,'Active configuration must not be migrated'
        config['enabled']=False;path.write_text(json.dumps(config))
        _,source=bootstrap(d);upgraded=json.loads(path.read_text())
        assert upgraded['symbols']==template['symbols'] and len(source['symbols'])==10
        assert upgraded['universe_upgrade_from_sha256']==old_hash
        assert upgraded['proxy_venues']==dict(template.get('proxy_venues',{}),SOLUSDT='mexc')
        assert upgraded['llm_chart_symbols']==template['llm_chart_symbols']
        assert upgraded['manager_scan_seconds']==600
        for key in ('enabled','account_binding','max_capital_usdt','stop_capabilities_verified','max_open_positions'):
            assert upgraded[key]==config[key],key
        bootstrap(d);assert json.loads(path.read_text())==upgraded
        upgraded['symbols']=original;path.write_text(json.dumps(upgraded))
        bootstrap(d)
        assert json.loads(path.read_text())==upgraded,'An operator override must not be undone on restart'
print('PASS disabled bootstrap, state persistence, owner-preserving universe upgrade and emergency-off override')
