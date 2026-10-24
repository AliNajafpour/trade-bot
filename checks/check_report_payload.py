"""Full evidence stays stored; provider inputs deduplicate HTTP receipt metadata."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.intelligence_reports import ReportStore

receipt=dict(url='https://example.org/feed',status=200,sha256='a'*64,
    request_started_utc='2026-10-25T00:00:00Z',response_received_utc='2026-10-25T00:00:01Z',
    headers={'set-cookie':'unnecessary_http_metadata '*2000},body_path='/private/raw/feed.body')
items=[dict(id=str(i),title='Actual headline '+str(i),summary='Full supplied summary',
    available_at=receipt['response_received_utc'],published_at='2026-10-24T23:00:00Z',
    source_receipt=receipt,first_source_receipt=receipt,
    revisions=[dict(value=3,source_receipt=receipt)],quality=['delayed']) for i in range(10)]
bundle=dict(kind='market',available_at=receipt['response_received_utc'],facts=[dict(
    id='fact',value=2,unit='percent',measurement_date='2026-09-01',source_history=[dict(value=1)],
    available_at=receipt['response_received_utc'],
    source_receipt=receipt)],news=items,coverage=dict(status='partial',limitations=['one source missing']))
before=deepcopy(bundle);captured=[]
def provider(packet,prompt,schema,*_):
    assert set(schema['properties']['evidence']['items']['enum'])=={r['id'] for r in packet['bundle']['facts']+packet['bundle']['news']}
    captured.append(packet)
    return dict(status='ok',ended_utc='2026-10-25T00:00:02Z',response=dict(
        market_bias='uncertain',risk_regime='uncertain',confidence=.5,risk_multiplier=.5,
        summary='Source-backed fixture',key_points=['partial coverage'],evidence=['fact'],invalidation_conditions=[]))
with TemporaryDirectory() as temporary:
    result=ReportStore(temporary).refresh('market',bundle,'2026-10-25T00:00:02Z',1800,provider)
    assert result['status']=='partial'
packed=captured[0]['bundle']
assert bundle==before and len(packed['news'])==10 and len(packed['facts'])==1
assert all(packed['coverage'][k]==v for k,v in bundle['coverage'].items()) and len(packed.get('source_receipts',{}))==1,'Provider inputs still duplicate HTTP receipts'
for old,new in zip(bundle['facts']+bundle['news'],packed['facts']+packed['news']):
    for k in old.keys()-{'source_receipt','first_source_receipt','revisions'}:assert new[k]==old[k],k
    assert packed['source_receipts'][new['source_receipt_ref']]['sha256']==receipt['sha256']
assert packed['news'][0]['revisions'][0]['value']==3
serialized=json.dumps(packed)
assert 'set-cookie' not in serialized and '/private/raw/' not in serialized
assert len(serialized)<len(json.dumps(bundle))*.1,'Duplicated transport metadata still reaches the model'
large=deepcopy(bundle);large['coverage']['status']='complete'
large['news']=[dict(items[i%10],id='large-'+str(i),source='A' if i%2 else 'B') for i in range(100)]
captured.clear()
with TemporaryDirectory() as temporary:
    result=ReportStore(temporary).refresh('market',large,'2026-10-25T00:00:02Z',1800,provider)
assert result['status']=='partial','A bounded news analysis must disclose omitted items'
packed=captured[0]['bundle']
assert len(packed['news'])==40 and {r['source'] for r in packed['news']}=={'A','B'}
assert result['coverage']['analysis_news_selection']['omitted_news']==60 and len(large['news'])==100
print('PASS raw facts/news retained, provider receipt deduplication, timestamps/revisions preserved and bounded news coverage disclosed')
