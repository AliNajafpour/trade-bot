"""Frozen Kronos + Chronos-2 + retained rules. No fitting or execution capability."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SYMBOLS=('ADAUSDT','TRXUSDT','LTCUSDT')
CONTEXT=128
HORIZONS=(1,4,24)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def inputs(request):
    import pandas as pd
    from .data import load_market, aggregate_complete
    from .features import _validated_frame
    if request.get('symbol') not in SYMBOLS: raise ValueError('Only reviewed legacy technical markets are supported')
    decision=pd.Timestamp(request.get('decision_at',request['asof']))
    if decision.tzinfo is None or decision.second or decision.microsecond: raise ValueError('Aware minute-aligned decision required')
    path=Path(request['history_path']).resolve()
    frame=load_market(path,60)
    frame=frame.loc[frame.timestamp+pd.Timedelta(minutes=1)<=decision].tail(10080).copy()
    frame.attrs['interval_seconds']=60
    _,times,gaps,interval=_validated_frame(frame)
    if len(frame)!=10080 or gaps.any() or interval!=60 or times[-1]+pd.Timedelta(minutes=1)!=decision:
        raise ValueError('Exactly 10080 contiguous completed M1 bars ending at decision required')
    hourly,_=aggregate_complete(frame,3600)
    if len(hourly)<CONTEXT or hourly.timestamp.iloc[-1]+pd.Timedelta(hours=1)!=decision.floor('h'):
        raise ValueError('Insufficient current completed hourly context')
    hourly=hourly.tail(CONTEXT).copy();hourly.attrs['interval_seconds']=3600
    provenance=dict(history_path=str(path),history_sha256=sha256(path),used_m1_rows=len(frame),
        first_open_at=times[0].isoformat(),last_completed_at=decision.isoformat(),
        forecast_origin_at=decision.floor('h').isoformat(),input_interval_seconds=60,model_interval_seconds=3600,
        hourly_context=CONTEXT,interpolated=False)
    return frame,hourly,provenance


def model_stage(request,stage):
    began=time.perf_counter()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    _,hourly,history=inputs(request)
    root=Path(request['run_dir']).resolve();root.mkdir(parents=True,exist_ok=True)
    directory=root/(stage+'-'+str(time.time_ns()))
    source=request['model_dirs']
    if stage=='kronos':
        from .kronos import _verify_assets, _load_encoder
        from .direct_kronos import DirectKronos, _predictor_class, UPSTREAM, UPSTREAM_SHA256
        from .council_kronos import cached_decoder
        model_dir,tokenizer_dir,provenance=_verify_assets(dict(kronos_model='small',
            kronos_model_dir=source['kronos'],kronos_tokenizer_dir=source['tokenizer']))
        model,tokenizer=_load_encoder(model_dir,tokenizer_dir,'cpu')
        torch.set_num_threads(1)
        predictor=_predictor_class()(model,tokenizer,device='cpu',max_context=512)
        provenance.update(predictor_source=str(UPSTREAM),predictor_source_sha256=UPSTREAM_SHA256)
        adapter=DirectKronos(device='cpu',predictor=predictor,provenance=provenance)
        with cached_decoder(model):
            result=adapter.forecast(hourly,history['last_completed_at'],directory,context=CONTEXT,pred_len=24,paths=4)
        summary=result['price_summary']
        if summary['status']!='ok': raise ValueError('Kronos produced no valid positive finite close paths')
        levels=summary['close_quantiles_by_step']
        native_file=directory/'forecast.json'
        evidence=dict(valid_close_paths=summary['valid_paths'],requested_paths=4,
            physically_valid_paths=result['valid_paths'],raw_paths_sha256=result['raw_paths_sha256'],
            distribution_kind='Four autoregressive CLOSE paths; empirical quantiles are noisy and uncalibrated')
    elif stage=='chronos2':
        from .chronos_market import ChronosMarket
        adapter=ChronosMarket(model_dir=source['chronos2'],device='cpu')
        torch.set_num_threads(1)
        result=adapter.forecast(hourly,request['symbol'],history['last_completed_at'],directory,context=CONTEXT,pred_len=24)
        if result['status']!='ok': raise ValueError('Chronos-2 produced invalid or crossed quantiles')
        levels=result['price_summary']['close_quantiles_by_step'];provenance=result['provenance']
        native_file=directory/'forecast.json'
        evidence=dict(distribution_kind='Marginal future CLOSE quantiles; not joint paths or first-touch probabilities',
            native_task_sha256=result['native_task_sha256'],quantiles_sha256=result['quantiles_sha256'])
    else: raise ValueError('Unknown foundation stage')
    horizons={str(h):{q:float(values[h-1]) for q,values in levels.items() if q in ('q10','q25','q50','q75','q90')} for h in HORIZONS}
    if not all(np.isfinite(v) and v>0 for row in horizons.values() for v in row.values()): raise ValueError('Invalid closing forecast')
    closes={q:values for q,values in levels.items() if q in ('q10','q25','q50','q75','q90')}
    ask=float(request['quote']['ask'])
    if not np.isfinite(ask) or ask<=0: raise ValueError('Positive finite quote ask required')
    return dict(status='ok',model=stage,symbol=request['symbol'],variant='pristine_frozen_pretrained',
        history=history,context=CONTEXT,prediction_hours=24,cpu_threads=torch.get_num_threads(),
        close_quantiles_by_step=closes,horizons_hours=horizons,provenance=provenance,evidence=evidence,
        last_close=float(hourly.close.iloc[-1]),current_ask_reference=ask,
        max_median_close=float(max(closes['q50'])),central_upside_fraction=float(max(closes['q50'])/ask-1),
        native_forecast_path=str(native_file),native_forecast_sha256=sha256(native_file),
        runtime_versions={name:importlib.metadata.version(name) for name in ('torch','chronos-forecasting','transformers','einops','safetensors')},
        completed_at=stamp(),total_seconds=time.perf_counter()-began,
        probability_semantics='Advisory closing-price distributions only; no calibrated TP-before-SL probability')


def run_stage(command,env):
    with subprocess.Popen(command,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True) as child:
        try: _,error=child.communicate(timeout=180)
        except BaseException:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
            else: child.kill()
            child.communicate()
            raise
        if child.returncode: raise subprocess.CalledProcessError(child.returncode,command,stderr=error)


def forecast(request,request_path):
    began=time.perf_counter()
    root=Path(request['run_dir']).resolve();root.mkdir(parents=True,exist_ok=True)
    models={}
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
    for stage in ('kronos','chronos2'):
        output=root/(stage+'-result-'+str(time.time_ns())+'.json')
        run_stage([sys.executable,'-m','wallex_lab.legacy_technical','--request',str(request_path),
            '--output',str(output),'--stage',stage],env)
        result=json.loads(output.read_text(encoding='utf-8'))
        if result.get('status')!='ok' or result.get('symbol')!=request['symbol']: raise ValueError('Foundation worker did not return this market')
        models[stage]=result
    # Import price libraries only after model subprocesses exit, keeping the supervisor small.
    from .strategy_bank import evaluate
    frame,_,history=inputs(request)
    if any(result['history']!=history for result in models.values()): raise ValueError('History changed between foundation stages')
    rules=evaluate(frame,request['symbol'],request['asof'],request['quote'])
    if rules['status']!='ok': raise ValueError('Retained technical rules unavailable')
    return dict(status='ok',technical_mode='legacy_ensemble',symbol=request['symbol'],asof=request['asof'],
        decision_at=history['last_completed_at'],models=models,rules=rules,history=history,completed_at=stamp(),total_seconds=time.perf_counter()-began,
        adapter_sha256=sha256(__file__),rule_source_sha256=sha256(Path(__file__).with_name('strategy_bank.py')),
        limitations=['Frozen foundation forecasts are uncalibrated on these Wallex markets.',
            'The ten rules share price inputs and are not independent votes or probabilities.',
            'Hourly model forecasts originate at the latest completed hour; rules use completed M1 bars.',
            'Four Kronos samples are a CPU budget, not a reliable tail-risk estimate.',
            'No fitting, learned adaptation, account reads, orders, fills or profitability claims.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=('kronos','chronos2'))
    args=parser.parse_args();request_path=Path(args.request).resolve()
    request=json.loads(request_path.read_text(encoding='utf-8'))
    result=model_stage(request,args.stage) if args.stage else forecast(request,request_path)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__': main()
