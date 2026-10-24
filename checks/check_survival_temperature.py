"""Per-asset calibration preserves unchanged assets and rejects invalid metadata."""
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.barrier_survival import SurvivalPredictor

torch.set_num_threads(1)
root=Path(__file__).resolve().parents[1]
base=SurvivalPredictor.load(root/'models/barrier-survival/late')
count=base.metadata['minimum_history_minutes'];decision=pd.Timestamp('2026-10-22T16:00:00Z')
close=100*np.exp(.0001*np.sin(np.arange(count)/11))
frame=pd.DataFrame(dict(timestamp=pd.date_range(end=decision-pd.Timedelta(minutes=1),periods=count,freq='min'),
    open=close,high=close*1.0002,low=close*.9998,close=close,volume=np.ones(count)))
def predict(p,symbol):
    return p.predict(frame,symbol,decision,float(close[-1]),tp_fraction=.02,sl_fraction=.02)['forecasts']
metadata=deepcopy(base.metadata);metadata['temperatures_by_symbol']={'BTCUSDT':2.}
mapped=SurvivalPredictor(base.models,metadata)
scalar=deepcopy(base.metadata);scalar['temperature']=2.
assert predict(mapped,'BTCUSDT')==predict(SurvivalPredictor(base.models,scalar),'BTCUSDT'),'Per-asset calibration was ignored'
assert predict(mapped,'ETHUSDT')==predict(base,'ETHUSDT'),'Calibration changed another asset'
for mapping in ({'BTCUSDT':0},{'BTCUSDT':float('nan')},{'UNKNOWNUSDT':1},None):
    try:SurvivalPredictor(base.models,dict(base.metadata,temperatures_by_symbol=mapping))
    except ValueError:pass
    else:raise AssertionError('Invalid calibration metadata accepted')
from scripts.backtest_survival_threshold import predict_grid
values=frame[['open','high','low','close','volume']].to_numpy();times=np.array([t.value for t in frame.timestamp],dtype=np.int64)
data=SimpleNamespace(metadata=dict(symbols=['ETHUSDT','BTCUSDT']),series=[(values,times),(values,times)],
    arrays=dict(symbol=np.array([0,1]),bar_index=np.array([count-1,count-1]),time_ns=np.array([decision.value]*2)))
with TemporaryDirectory() as temporary,patch('scripts.backtest_survival_threshold.GEOMETRY',[(.02,.02)]):
    batch=predict_grid(data,mapped,np.array([0,1]),2,Path(temporary))
for i,symbol in enumerate(data.metadata['symbols']):
    expected=np.array([[row[k] for k in ('p_tp_first','p_sl_first','p_no_touch')] for row in predict(mapped,symbol)])
    assert np.allclose(batch[i,0],expected,atol=1e-6),('Batch calibration or symbol mapping differs from live prediction',symbol)
print('PASS per-asset temperature, unchanged-asset equality, invalid calibration rejection')
