"""CPU-compatible council adapter for genuine marginal Chronos-2 forecasts."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .chronos_market import ChronosMarket,LOCAL_MODEL,QUANTILES,build_task
from .chronos_training import save
from .data import load_market,aggregate_complete
from .features import _validated_frame
from .kronos import _sha256

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'runs/market-council-20261022/model-status/chronos2.json'


def prepare_request(request):
    """Use only completed candles and future calendar, preserving source gaps."""
    if not isinstance(request,dict) or not isinstance(request.get('markets'),list) or not request['markets']:
        raise ValueError('Request requires asof and nonempty markets list')
    asof=pd.Timestamp(request['asof'])
    if asof.tzinfo is None:raise ValueError('Aware asof required')
    settings=request.get('settings',{});context=settings.get('context',128)
    prediction=settings.get('prediction',24);tasks=[];clocks=[]
    for market in request['markets']:
        frame=load_market(market['candles_path'],settings.get('interval_seconds'))
        interval=_validated_frame(frame)[3];quality=None;hour_masks=None
        if interval==60:
            from .council_technical import repair_m1_context
            if 'observed_mask' in frame:
                if not frame.observed_mask.isin([True,False]).all():raise ValueError('Boolean observed_mask required')
                frame=frame.loc[frame.observed_mask.astype(bool)].copy()
            minutes,quality=repair_m1_context(frame,asof.floor('h'),(context+2)*60)
            buckets=minutes.set_index('timestamp').observed_mask.resample('h',origin='epoch')
            hour_masks=pd.DataFrame(dict(all_observed=buckets.min(),open_observed=buckets.first(),close_observed=buckets.last()))
            frame,_=aggregate_complete(minutes,3600)
        item,clock,window=build_task(frame,asof,context,prediction)
        if clock['interval_seconds']!=3600:
            raise ValueError('Council checkpoint requires hourly candles; resample completed bars before requesting')
        if asof-pd.Timestamp(clock['last_completed_at'])>=pd.Timedelta(hours=1):
            raise ValueError('Latest completed hourly candle is stale by a full bar')
        if hour_masks is not None:
            masks=hour_masks.reindex(pd.DatetimeIndex(window.timestamp))
            item['target'][~masks.close_observed.to_numpy(dtype=bool)]=np.nan
            item['past_covariates']['open'][~masks.open_observed.to_numpy(dtype=bool)]=np.nan
            unknown=~masks.all_observed.to_numpy(dtype=bool)
            for name in ('high','low','volume'):item['past_covariates'][name][unknown]=np.nan
            clock.update(input_quality=quality,source_interval_seconds=60,
                native_missing_steps=int(np.isnan(item['target']).sum()),native_unknown_ohlv_hours=int(unknown.sum()),
                native_missing_policy='Observed hour-end CLOSE retained; uncertain OPEN/HIGH/LOW/VOLUME and missing hour-end CLOSE are native NaN masks.',
                observed_hour_close_mask=masks.close_observed.astype(bool).tolist(),observed_full_hour_mask=masks.all_observed.astype(bool).tolist())
        if not np.isfinite(item['target'][-1]):raise ValueError('Latest completed close is missing')
        tasks.append(item);clocks.append((market,clock,float(window.close.iloc[-1])))
    if len({c[1]['last_completed_at'] for c in clocks})!=1:raise ValueError('Markets must share the latest completed candle clock')
    return tasks,clocks,context,prediction


def forecast(request,pipeline=None):
    began=time.perf_counter()
    import torch
    imported=time.perf_counter()
    tasks,clocks,context,prediction=prepare_request(request);settings=request.get('settings',{})
    status_path=Path(settings.get('status_path',STATUS));status=json.loads(status_path.read_text()) if status_path.exists() else {}
    approved=bool(status.get('promotion_allowed',False));checkpoint=settings.get('checkpoint',status.get('selected_checkpoint')) if approved else None
    if settings.get('variant')=='pretrained':checkpoint=None
    variant='finetuned' if checkpoint else 'pretrained';started=time.perf_counter()
    torch.set_num_threads(2)
    if pipeline is None:
        model=ChronosMarket(device=settings.get('device','cpu'),checkpoint=checkpoint);pipeline=model.pipeline
        provenance=model.provenance
    else:provenance=dict(model='amazon/chronos-2',benchmark_pipeline_injected=True,selected_checkpoint=checkpoint)
    loaded=time.perf_counter()
    with torch.inference_mode():
        predictions,_=pipeline.predict_quantiles(tasks,prediction_length=prediction,quantile_levels=QUANTILES,
            context_length=context,batch_size=9*len(tasks),cross_learning=False,limit_prediction_length=True)
    inference_seconds=time.perf_counter()-loaded;outputs=[]
    if len(predictions)!=len(tasks):raise ValueError('Missing per-market forecasts')
    for (market,clock,last),tensor in zip(clocks,predictions):
        values=tensor[0].detach().float().cpu().numpy()
        if values.shape!=(prediction,len(QUANTILES)) or not np.isfinite(values).all() or (values<=0).any() or (np.diff(values,axis=1)<0).any():
            raise ValueError('Invalid/crossed Chronos quantiles')
        levels={f'q{round(level*100):02d}':values[:,j].tolist() for j,level in enumerate(QUANTILES)}
        horizons={str(hour):{name:vector[hour-1] for name,vector in levels.items()} for hour in (1,3,6,12,24) if hour<=prediction}
        outputs.append(dict(symbol=market['symbol'],quote=market.get('quote'),last_close=last,clock=clock,
            candles_path=str(Path(market['candles_path']).resolve()),input_sha256=_sha256(market['candles_path']),
            close_quantiles_by_step=levels,horizons_hours=horizons,model_id='amazon/chronos-2',variant=variant,
            forecast_origin_at=clock['last_completed_at'],horizon_close_at={hour:clock['forecast_close_at'][int(hour)-1] for hour in horizons},
            model_available_at=status.get('model_available_at'),available_at=datetime.now(timezone.utc).isoformat(),
            expires_at=(pd.Timestamp(clock['last_completed_at'])+pd.Timedelta(seconds=prediction*clock['interval_seconds'])).isoformat(),
            distribution_kind='Marginal future CLOSE quantiles; no joint path or TP-before-SL probability',
            forecast_role='advisory',past_covariates=list(tasks[len(outputs)]['past_covariates']),
            known_future_covariates=list(tasks[len(outputs)]['future_covariates'])))
    return dict(status='ok',model='chronos2',model_id='amazon/chronos-2',variant=variant,
        selected_checkpoint=checkpoint,pretrained_path=str(LOCAL_MODEL),promotion_allowed=approved,
        provenance=provenance,
        asof=request['asof'],available_at=datetime.now(timezone.utc).isoformat(),markets=outputs,
        expires_at=min(row['expires_at'] for row in outputs),
        inference_seconds=inference_seconds,torch_import_seconds=imported-began,prepare_seconds=started-imported,
        load_seconds=loaded-started,total_seconds=time.perf_counter()-began,
        update_cadence_seconds=900,source_bar_seconds=3600,
        limitations=['Hourly completed bars update hourly; council may reuse forecast on 15-minute checks.',
            'No macro history without verified release availability.','Binance reference prices require separate Wallex venue/cost validation.',
            'Quantile coverage is empirical research evidence, not a guarantee or fill probability.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=forecast(json.loads(Path(args.request).read_text(encoding='utf-8')));save(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','variant','inference_seconds','load_seconds')},indent=2))


if __name__=='__main__':main()
