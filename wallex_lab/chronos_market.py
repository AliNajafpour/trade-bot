"""Genuine local Chronos-2 covariate-aware price forecasts, with causal receipts."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .features import _validated_frame
from .kronos import _sha256

LOCAL_MODEL=Path(__file__).resolve().parents[1]/'models/chronos-2-29ec376'
REVISION='29ec3766d36d6f73f0696f85560a422f50e8498c'
MODEL_HASHES={'config.json':'ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031',
    'model.safetensors':'ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42'}
QUANTILES=[.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99]
CALENDAR=('hour_sin','hour_cos','dow_sin','dow_cos')


def prepare_fundamentals(enriched,availability_mode,source_sha256=None):
    """Validate source-aligned fundamental values before exposing numeric past inputs."""
    if availability_mode not in ('prospectively_verified','retrospective_assumed_48h'):
        raise ValueError('Explicit verified or assumed48h availability mode required')
    rows=enriched.copy();rows['timestamp']=pd.to_datetime(rows.timestamp,utc=True)
    available=pd.to_datetime(rows['fund_available_at'],utc=True,errors='coerce')
    known=~rows['fund_missing'].astype(bool)
    if ((available.isna() | (available>rows.timestamp+pd.Timedelta(minutes=5))) & known).any():
        raise ValueError('Fundamental value unavailable at its past candle close')
    metrics=['fund_tvl_usd','fund_log_tvl','fund_tvl_change_1d','fund_missing','fund_availability_verified']
    result=rows[['timestamp']+metrics].copy()
    result.attrs.update(availability_mode=availability_mode,source_sha256=source_sha256)
    return result


def calendar_values(closes):
    clocks=pd.DatetimeIndex(closes)
    hour=(clocks.hour+clocks.minute/60.)*2*np.pi/24
    day=clocks.dayofweek*2*np.pi/7
    return {name:np.asarray(value,dtype=np.float32) for name,value in
        zip(CALENDAR,(np.sin(hour),np.cos(hour),np.sin(day),np.cos(day)))}


def build_task(frame,asof,context=512,pred_len=36,technical=True,fundamentals=None):
    """Raw CLOSE target; OHLCV and released numeric facts are past-only covariates.

    fundamentals is an aligned DataFrame with timestamp and numeric columns. Its
    caller must provide an availability audit; no current fact is repeated into
    historical bars by this function. All-missing columns remain explicitly absent.
    """
    if type(context) is not int or not 32<=context<=8192 or type(pred_len) is not int or not 1<=pred_len<=1024:
        raise ValueError('Require context32..8192 and prediction1..1024')
    cutoff=pd.Timestamp(asof)
    if cutoff.tzinfo is None:raise ValueError('Aware asof required')
    times=pd.DatetimeIndex(pd.to_datetime(frame.timestamp,errors='raise'))
    if times.tz is None:raise ValueError('Aware candle timestamps required')
    interval=float(frame.attrs.get('interval_seconds',300))
    observed=frame.loc[times+pd.Timedelta(seconds=interval)<=cutoff].copy().reset_index(drop=True)
    observed.attrs['interval_seconds']=interval
    if not len(observed):raise ValueError('No completed context')
    _,observed_times,_,_=_validated_frame(observed)
    first=observed_times[-1]-pd.Timedelta(seconds=(context-1)*interval)
    if observed_times[0]>first:raise ValueError('Insufficient completed context clock range')
    opens=pd.date_range(first,observed_times[-1],periods=context)
    # Native Chronos masks NaNs: represent missing observations explicitly, never compress time or interpolate prices.
    observed['timestamp']=observed_times
    window=observed.set_index('timestamp').reindex(opens).rename_axis('timestamp').reset_index()
    window.attrs['interval_seconds']=interval
    values=window[['open','high','low','close','volume']]
    closes=opens+pd.Timedelta(seconds=interval)
    future_closes=pd.date_range(closes[-1]+pd.Timedelta(seconds=interval),periods=pred_len,freq=pd.Timedelta(seconds=interval))
    task={'target':values.close.to_numpy(dtype=np.float32)}
    covariates={name:values[name].to_numpy(dtype=np.float32) for name in ('open','high','low','volume')} if technical else {}
    covariates.update(calendar_values(closes))
    fundamental_names=[];missing=[]
    if fundamentals is not None:
        if 'timestamp' not in fundamentals:raise ValueError('Aligned fundamental frame needs timestamp')
        supplied=fundamentals.copy();supplied['timestamp']=pd.to_datetime(supplied.timestamp,utc=True)
        if supplied.timestamp.duplicated().any():raise ValueError('Fundamental aligned timestamps must be unique')
        joined=pd.DataFrame({'timestamp':opens}).merge(supplied,on='timestamp',how='left',validate='one_to_one')
        for name in supplied.columns:
            if name=='timestamp':continue
            vector=pd.to_numeric(joined[name],errors='raise').to_numpy(dtype=np.float32)
            if np.isinf(vector).any():raise ValueError('Infinite fundamental covariate')
            if np.isnan(vector).all():missing.append(name);continue
            covariates[name]=vector;fundamental_names.append(name)
    task['past_covariates']=covariates
    task['future_covariates']=calendar_values(future_closes)
    clock=dict(asof=cutoff.isoformat(),context=context,prediction_length=pred_len,interval_seconds=interval,
        first_open_at=opens[0].isoformat(),last_completed_at=closes[-1].isoformat(),
        native_missing_steps=int(values.close.isna().sum()),missing_policy='Native NaN masks on regular clock; no interpolation or zero-filled prices',
        forecast_open_at=(future_closes-pd.Timedelta(seconds=interval)).astype(str).tolist(),
        forecast_close_at=future_closes.astype(str).tolist(),past_covariates=list(covariates),
        known_future_covariates=list(CALENDAR),fundamental_covariates=fundamental_names,
        all_missing_fundamental_covariates=missing,
        fundamental_scope=fundamentals.attrs.get('availability_mode','aligned past-only; availability audit required') if fundamentals is not None else 'unavailable; not imputed as neutral',
        fundamental_source_sha256=fundamentals.attrs.get('source_sha256') if fundamentals is not None else None)
    return task,clock,window


class ChronosMarket:
    def __init__(self,model_dir=LOCAL_MODEL,device='cuda',pipeline=None,checkpoint=None):
        self.device=device
        original=Path(model_dir).resolve()
        verified={name:_sha256(original/name) for name in MODEL_HASHES}
        if verified!=MODEL_HASHES:raise ValueError('Pinned original Chronos-2 assets differ')
        self.provenance=dict(model='amazon/chronos-2',revision=REVISION,original_model_dir=str(original),
            original_asset_sha256=verified,chronos_version=importlib.metadata.version('chronos-forecasting'),
            pretrained_overlap='Unknown; retrospective validation cannot rule out contamination')
        if pipeline is None:
            import torch
            from chronos import Chronos2Pipeline
            torch.set_num_threads(2)
            if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable')
            source=original if checkpoint is None else Path(checkpoint).resolve()
            if checkpoint is not None:
                self.provenance['finetuned_checkpoint']={'path':str(source),'files':{p.name:_sha256(p) for p in source.glob('*') if p.is_file()}}
            pipeline=Chronos2Pipeline.from_pretrained(str(source),device_map=device,torch_dtype=torch.float32,
                local_files_only=True,trust_remote_code=False)
            pipeline.model.eval().requires_grad_(False)
        self.pipeline=pipeline

    def forecast(self,frame,symbol,asof,run_dir,context=512,pred_len=36,technical=True,fundamentals=None):
        import torch
        task,clock,window=build_task(frame,asof,context,pred_len,technical,fundamentals)
        folder=Path(run_dir).resolve();folder.mkdir(parents=True,exist_ok=False)
        window.to_csv(folder/'input.csv',index=False)
        arrays={'target':task['target'],**{'past_'+name:value for name,value in task['past_covariates'].items()},
            **{'future_'+name:value for name,value in task['future_covariates'].items()}}
        np.savez_compressed(folder/'native-task.npz',**arrays)
        started=datetime.now(timezone.utc).isoformat();before=time.perf_counter()
        with torch.inference_mode():
            quantiles,point=self.pipeline.predict_quantiles([task],prediction_length=pred_len,quantile_levels=QUANTILES,
                batch_size=16,context_length=context,cross_learning=False,limit_prediction_length=True)
        latency=time.perf_counter()-before
        raw=quantiles[0].detach().float().cpu().numpy();median=point[0].detach().float().cpu().numpy()
        if raw.shape!=(1,pred_len,len(QUANTILES)) or median.shape!=(1,pred_len):raise ValueError('Unexpected native Chronos output shape')
        np.savez_compressed(folder/'quantiles.npz',quantiles=raw,point_median=median,levels=np.asarray(QUANTILES),
            forecast_open_ns=pd.to_datetime(clock['forecast_open_at'],utc=True).as_unit('ns').asi8)
        finite=bool(np.isfinite(raw).all() and (raw>0).all());crossings=int((np.diff(raw[0],axis=1)<0).any(axis=1).sum())
        result=dict(status='ok' if finite and not crossings else 'invalid_quantiles',symbol=symbol,clock=clock,
            model_family='chronos2',horizon_minutes=pred_len*clock['interval_seconds']/60,context=context,
            started_at=started,completed_at=datetime.now(timezone.utc).isoformat(),inference_seconds=latency,
            quantile_crossed_steps=crossings,positive_finite_quantiles=finite,quantile_levels=QUANTILES,
            provenance=self.provenance,adapter_sha256=_sha256(__file__),input_sha256=_sha256(folder/'input.csv'),
            native_task_sha256=_sha256(folder/'native-task.npz'),quantiles_sha256=_sha256(folder/'quantiles.npz'),
            input_file=str(folder/'input.csv'),quantiles_file=str(folder/'quantiles.npz'),
            point_forecast_kind='MEDIAN; native API second return is named mean but implemented as median',
            distribution_kind='Marginal quantiles per future step; not joint sampled paths or TP-before-SL probabilities',
            caveats=['Quantile coverage is uncalibrated on Wallex.','No historical order book, fills or profitability inferred.',
                'Fundamentals must be joined by actual or explicitly assumed availability; calendar alone is future-known.'])
        if finite:
            q={f'q{round(level*100):02d}':raw[0,:,j].tolist() for j,level in enumerate(QUANTILES)}
            last=float(window.close.iloc[-1]);tp=max(q['q75']);sl=min(q['q25'])
            result['price_summary']=dict(close_quantiles_by_step=q,terminal_close_quantiles={name:value[-1] for name,value in q.items()},
                terminal_close_median=float(median[0,-1]),terminal_return_median=float(median[0,-1]/last-1),
                tp_sl=dict(take_profit_price=tp,stop_loss_price=sl,long_brackets_last_close=sl<last<tp,
                    derivation='TP=max marginal q75 CLOSE; SL=min marginal q25 CLOSE',
                    calibration='Uncalibrated marginal levels; no joint path or first-touch probability claim'))
        (folder/'forecast.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        return result
