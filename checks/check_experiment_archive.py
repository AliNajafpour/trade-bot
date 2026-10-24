"""Verify published evidence and headline numbers without network/model execution."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'docs/experiments'

def load(path):
    return json.loads(path.read_text('utf-8-sig'))

manifest=load(ARCHIVE/'manifest.json')
seen=set();json_count=csv_count=0
for entry in manifest['files']:
    path=(ARCHIVE/entry['published']).resolve()
    assert path.is_relative_to(ARCHIVE.resolve()),entry['published']
    assert path not in seen,entry['published']
    seen.add(path)
    assert path.is_file(),entry['published']
    data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==entry['published_sha256'],entry['published']
    assert len(data)==entry['bytes'],entry['published']
    assert re.fullmatch('[a-f0-9]{64}',entry['source_sha256'])
    if path.suffix=='.json':load(path);json_count+=1
    if path.suffix=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as handle:
            rows=csv.reader(handle);header=next(rows,None)
            assert header is not None,path
            assert all(len(row)==len(header) for row in rows if row),path
        csv_count+=1

for path in [ARCHIVE/'README.md',ARCHIVE/'catalog.md']:
    for target in re.findall(r'\]\(([^)\n]+)\)',path.read_text('utf-8')):
        if target.startswith(('https://','http://','#')):continue
        assert (path.parent/unquote(target.partition('#')[0])).exists(),(path.name,target)

evidence=ARCHIVE/'evidence/experiments'
with (evidence/'survival-threshold-full-grid-20261023/matrix-baseline.csv').open(newline='',encoding='utf-8') as handle:
    rows=list(csv.DictReader(handle))
assert len(rows)==640
assert sum(float(r['return_pct'])>0 for r in rows)==13
assert sum(float(r['return_pct'])<0 for r in rows)==101
assert sum(float(r['return_pct'])==0 for r in rows)==526
selected=[r for r in rows if float(r['tp_fraction'])==.02 and float(r['sl_fraction'])==.02
          and int(r['forecast_horizon_minutes'])==1440 and float(r['probability_threshold'])==.55]
assert len(selected)==1
chosen=selected[0]
for key,value in {'closed_trades':17,'wins':12,'losses':5,'open_positions':1}.items():assert int(chosen[key])==value
assert math.isclose(float(chosen['return_pct']),1.3431378718012699,abs_tol=1e-9)
assert math.isclose(float(chosen['max_drawdown_pct']),.8437750928647292,abs_tol=1e-9)
workflow=load(evidence/'survival-workflow-reconstructed-20261023/results.json')
for arm in workflow['arms'].values():assert arm['summary']['closed_trades']==0 and arm['summary']['return_pct']==0
fixed=load(evidence/'ten-asset-extension-20261025/fixed-policy-heldout.json')
assert len(fixed['results'])==6
for result in fixed['results']:
    assert result['test_origins']==124
    long=next(x for x in result['by_side'] if x['side']=='long')
    assert long['passes_probability_threshold']==0
observation=load(ARCHIVE/'deployment-observation-20261025.json')
assert observation['trading_enabled'] is False and observation['real_orders']==0
assert len(observation['configured_markets'])==10
print(json.dumps(dict(status='passed',evidence_files=len(seen),json_files=json_count,csv_files=csv_count,
    matrix_rows=640,headline_numbers_verified=True,ten_asset_zero_long_passes_verified=True,
    scope='Saved artifact consistency; no historical experiments rerun or live status refreshed')))
