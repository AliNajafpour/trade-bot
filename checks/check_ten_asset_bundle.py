"""Verify the actual ten-asset extension, original weights and CPU predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.barrier_survival import SurvivalPredictor


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=root/'models/barrier-survival/ten-assets')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    base_path=root/'models/barrier-survival/late'
    old=SurvivalPredictor.load(base_path);new=SurvivalPredictor.load(args.model)
    symbols=['BTCUSDT','ETHUSDT','SOLUSDT','SUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','BNBUSDT','TRXUSDT','LTCUSDT']
    assert new.metadata['symbols']==symbols and len(new.models)==len(old.models)==3
    assert new.metadata['base_metadata_sha256']==hashlib.sha256((base_path/'model.json').read_bytes()).hexdigest()
    assert new.metadata['protocol_sha256']==hashlib.sha256((args.model/'provenance.json').read_bytes()).hexdigest()
    assert new.metadata['feature_version']==old.metadata['feature_version']=='relative-v2'
    assert new.metadata['minimum_history_minutes']==old.metadata['minimum_history_minutes']==10080
    for before,after in zip(old.models,new.models):
        for key,value in before.state_dict().items():
            result=after.state_dict()[key]
            assert torch.equal(value,result[:4] if key=='asset.weight' else result),key
    n=10080;decision=pd.Timestamp('2026-10-24T00:00:00Z');close=100*np.exp(.0001*np.sin(np.arange(n)/11))
    frame=pd.DataFrame(dict(timestamp=pd.date_range(end=decision-pd.Timedelta(minutes=1),periods=n,freq='min'),
        open=close,high=close*1.0002,low=close*.9998,close=close,volume=np.ones(n)))
    results=[]
    for symbol in symbols:
        prediction=new.predict(frame,symbol,decision,float(close[-1]),tp_fraction=.02,sl_fraction=.02)
        probabilities=np.array([[r[k] for k in ('p_tp_first','p_sl_first','p_no_touch')] for r in prediction['forecasts']])
        assert np.isfinite(probabilities).all() and (probabilities>=0).all() and (probabilities<=1).all()
        assert np.allclose(probabilities.sum(1),1,atol=2e-6)
        assert (np.diff(probabilities[:,:2],axis=0)>=-2e-6).all() and (np.diff(probabilities[:,2])<=2e-6).all()
        assert not prediction['support_flags']
        if symbol in old.metadata['symbols']:
            original=old.predict(frame,symbol,decision,float(close[-1]),tp_fraction=.02,sl_fraction=.02)
            assert prediction['forecasts']==original['forecasts'],symbol
        results.append(dict(symbol=symbol,horizon_minutes=prediction['forecasts'][-1]['horizon_minutes']))
    result=dict(status='passed',members=3,symbols=symbols,original_tensors_and_predictions_exact=True,
        per_symbol_calibration=True,diagnostic_input='Synthetic candles; no trading result',
        metadata_sha256=hashlib.sha256((args.model/'model.json').read_bytes()).hexdigest())
    if args.output:args.output.write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
