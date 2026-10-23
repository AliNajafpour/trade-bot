"""Offline persistent startup check. Never imports Torch or contacts exchanges."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.cloud import bootstrap

with TemporaryDirectory() as d,patch.dict('os.environ',{'WALLEX_FORCE_DISABLE':'false'}):
    path,source=bootstrap(d)
    config=json.loads(path.read_text())
    assert config['enabled'] is False and config['stop_capabilities_verified'] is False
    assert config['account_binding'] is None and config['max_capital_usdt'] is None
    assert Path(config['run_dir']).is_relative_to(Path(d).resolve())
    assert source['symbols']==['BTC/USDT','ETH/USDT','SOL/USDT','S/USDT']
    config['enabled']=True;config['max_capital_usdt']=7;path.write_text(json.dumps(config))
    again,_=bootstrap(d);assert json.loads(again.read_text())['enabled'] is True
    with patch.dict('os.environ',{'WALLEX_FORCE_DISABLE':'true'}):bootstrap(d)
    assert json.loads(path.read_text())['enabled'] is False
print('PASS disabled bootstrap, state persistence and emergency-off override')
