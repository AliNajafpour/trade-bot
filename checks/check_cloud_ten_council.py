"""Ten-asset Weekly council restores an OFF model-only disk without changing its budget."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.cloud import ROOT,bootstrap

template=json.loads((ROOT/'config/live-cloud.example.json').read_text())
late=json.loads((ROOT/'models/barrier-survival/late/model.json').read_text())
with TemporaryDirectory() as temporary:
    root=Path(temporary);(root/'config').mkdir()
    for name in ('live-cloud.example.json','weekly-source.example.json'):
        (root/'config'/name).write_bytes((ROOT/'config'/name).read_bytes())
    for name in ('late','ten-assets'):(root/'models/barrier-survival'/name).mkdir(parents=True)
    (root/'models/barrier-survival/late/model.json').write_bytes((ROOT/'models/barrier-survival/late/model.json').read_bytes())
    extended=deepcopy(late);extended['symbols']=template['symbols']
    extended['base_metadata_sha256']=hashlib.sha256((ROOT/'models/barrier-survival/late/model.json').read_bytes()).hexdigest()
    (root/'models/barrier-survival/ten-assets/model.json').write_text(json.dumps(extended))
    with patch('wallex_lab.cloud.ROOT',root),patch.dict('os.environ',{'WALLEX_EXECUTION_PRESET':'weekly_model_only','WALLEX_FORCE_DISABLE':'false'}):
        path,_=bootstrap(root/'data');config=json.loads(path.read_text())
        config.update(account_binding='owner',max_capital_usdt=17,stop_capabilities_verified=True)
        path.write_text(json.dumps(config))
        with patch.dict('os.environ',{'WALLEX_EXECUTION_PRESET':'weekly_council_ten'}):
            _,source=bootstrap(root/'data');changed=json.loads(path.read_text())
            assert changed['symbols']==template['symbols'] and source['symbols']==[]
            assert changed['decision_mode']=='council' and changed['adaptive_learning'] is True
            assert changed['model']==str(root/'models/barrier-survival/ten-assets')
            assert all(not changed[k] for k in ('llm_chart_symbols','probability_symbols','legacy_technical_symbols'))
            for k in ('enabled','account_binding','max_capital_usdt','stop_capabilities_verified'):
                assert changed[k]==config[k]
            bootstrap(root/'data');assert json.loads(path.read_text())==changed
            active=deepcopy(changed);active['enabled']=True;path.write_text(json.dumps(active))
            bootstrap(root/'data');assert json.loads(path.read_text())==active,'Same-preset redeploy disabled an armed account'
            changed.update(enabled=True,decision_mode='weekly_model_only');path.write_text(json.dumps(changed))
            try:bootstrap(root/'data')
            except ValueError:pass
            else:raise AssertionError('A live preset was changed without stopping execution')
print('PASS ten-asset Weekly council preset, old account/capital preserved, source workers selected, no auto activation')
