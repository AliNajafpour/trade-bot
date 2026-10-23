"""Full Chronos-2 adaptation on purged crypto history; no automatic paper promotion."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .chronos_market import LOCAL_MODEL,QUANTILES,CALENDAR,calendar_values
from .data import load_market
from .features import _validated_frame
from .kronos import _sha256
from .locking import exclusive,atomic_replace


def calendar_splits(frames):
    """Common calendar boundaries on the actual, shared candle cadence."""
    clocks=[_validated_frame(frame) for frame in frames.values()]
    if not clocks or len({row[3] for row in clocks})!=1:raise ValueError('One shared candle interval required')
    delta=pd.Timedelta(seconds=clocks[0][3]);begin=max(row[1][0] for row in clocks);end=min(row[1][-1]+delta for row in clocks)
    bars=int((end-begin)/delta)
    if bars<10:raise ValueError('Insufficient common history')
    return [begin,begin+int(bars*.7)*delta,begin+int(bars*.85)*delta,end]


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    atomic_replace(temporary,path,timeout=5.)


def align_facts(frame,symbol,facts):
    """Long numeric facts join only after availability; unknowns stay NaN."""
    if facts is None:return {}
    required={'asset','metric','value','available_at'}
    if not required<=set(facts):raise ValueError('Facts require asset, metric, value, available_at')
    source=facts.copy();source['available_at']=pd.to_datetime(source.available_at,utc=True)
    source['value']=pd.to_numeric(source.value,errors='raise')
    if not np.isfinite(source.value).all():raise ValueError('Fundamental values must be finite')
    if 'measurement_at' in source:
        measurement=pd.to_datetime(source.measurement_at,utc=True)
        if (measurement>source.available_at).any():raise ValueError('Measurement after claimed availability')
    asset=symbol.split('_')[0].removesuffix('USDT')
    clocks=pd.DataFrame({'clock':pd.to_datetime(frame.timestamp,utc=True)+pd.Timedelta(seconds=_validated_frame(frame)[3])})
    output={}
    for metric in sorted(source.metric.unique()):
        selected=source.loc[(source.asset==asset)&(source.metric==metric),['available_at','value']].sort_values('available_at')
        if selected.available_at.duplicated().any():raise ValueError('Conflicting duplicate fundamental availability')
        output['fundamental_'+metric]=pd.merge_asof(clocks,selected,left_on='clock',right_on='available_at',direction='backward').value.to_numpy(dtype=np.float32)
    return output


def task(frame,fundamentals=None):
    values,opens,gaps,interval=_validated_frame(frame)
    if gaps.any():raise ValueError('Training task crosses a market-data gap')
    covariates={name:values[name].to_numpy(dtype=np.float32) for name in ('open','high','low','volume')}
    covariates.update(calendar_values(opens+pd.Timedelta(seconds=interval)))
    if fundamentals:covariates.update(fundamentals)
    return {'target':values.close.to_numpy(dtype=np.float32),'past_covariates':covariates,
        'future_covariates':{name:None for name in CALENDAR}}


def prepare(frames,context=512,prediction=36,facts=None):
    """A shared calendar split keeps every context and label inside its own block."""
    bounds=calendar_splits(frames);result={name:[] for name in ('train','validation','test')};receipts={name:[] for name in result}
    for symbol,frame in frames.items():
        _,times,gaps,interval=_validated_frame(frame);fundamentals=align_facts(frame,symbol,facts)
        delta=pd.Timedelta(seconds=interval)
        for column in ('fund_log_tvl','fund_tvl_change_1d'):
            if column in frame:
                if 'fund_available_at' not in frame:raise ValueError('Aligned fundamentals need availability audit')
                available=pd.to_datetime(frame.fund_available_at,utc=True)
                values=pd.to_numeric(frame[column],errors='raise').to_numpy(dtype=np.float32)
                if np.isinf(values).any() or (available>times+delta).any():
                    raise ValueError('Invalid or future-dated aligned fundamentals')
                fundamentals[column]=values
        groups=np.cumsum(gaps)
        for k,name in enumerate(result):
            eligible=(times>=bounds[k])&(times+delta<=bounds[k+1])
            for group in np.unique(groups[eligible]):
                indices=np.flatnonzero(eligible&(groups==group))
                if len(indices)<context+prediction:continue
                starts=[0] if name=='train' else range(0,len(indices)-context-prediction+1,max(prediction,24))
                for offset in starts:
                    ix=indices if name=='train' else indices[offset:offset+context+prediction]
                    segment=frame.iloc[ix].copy().reset_index(drop=True);segment.attrs['interval_seconds']=interval
                    result[name].append(task(segment,{key:value[ix] for key,value in fundamentals.items()}))
                    receipts[name].append(dict(symbol=symbol,start_at=times[ix[0]].isoformat(),
                        end_exclusive=(times[ix[-1]]+delta).isoformat(),rows=len(ix)))
    if any(not v for v in result.values()):raise ValueError('A purged split has no complete context+target task')
    return result,dict(boundaries=[x.isoformat() for x in bounds],interval_seconds=interval,tasks=receipts,
        rule='Common70/15/15 calendar; each full context and target stays inside one contiguous split segment')


def metrics(quantiles,actual,last,levels=QUANTILES):
    q=np.asarray(quantiles,dtype=float);actual=np.asarray(actual,dtype=float)
    if q.shape!=(len(actual),len(levels)) or not np.isfinite(q).all() or not np.isfinite(actual).all() or last<=0:
        raise ValueError('Finite [horizon,quantiles] predictions and aligned actuals required')
    median=q[:,levels.index(.5)];error=actual[:,None]-q;pinball=np.maximum(np.asarray(levels)*error,(np.asarray(levels)-1)*error)
    changed=actual!=last
    return dict(mae_fraction=float(np.mean(np.abs(median-actual))/last),flat_mae_fraction=float(np.mean(np.abs(actual-last))/last),
        flat_quantile_loss_fraction=float(np.mean(np.abs(actual-last))/last),
        amplitude_error_fraction=float(np.mean(np.abs(np.abs(median-last)-np.abs(actual-last)))/last),
        q10_q90_width_fraction=float(np.mean(q[:,levels.index(.9)]-q[:,levels.index(.1)])/last),
        quantile_loss_fraction=float(2*np.mean(pinball)/last),direction_correct=int(((np.sign(median-last)==np.sign(actual-last))&changed).sum()),
        direction_count=int(changed.sum()),q10_q90_covered=int(((q[:,levels.index(.1)]<=actual)&(actual<=q[:,levels.index(.9)])).sum()),points=len(actual))


def evaluate(pipeline,tasks,context,prediction,directory,limit=12,interval=300):
    """Fixed evenly spaced split tasks; no outcome-based cutoff selection."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    chosen=np.unique(np.linspace(0,len(tasks)-1,min(limit,len(tasks)),dtype=int));rows=[]
    for index in chosen:
        source=tasks[index];target=source['target'];past={k:v[:-prediction] for k,v in source['past_covariates'].items()}
        item=dict(target=target[:-prediction],past_covariates=past,
            future_covariates={k:source['past_covariates'][k][-prediction:] for k in CALENDAR})
        forecasts,medians=pipeline.predict_quantiles([item],prediction_length=prediction,quantile_levels=QUANTILES,
            context_length=context,batch_size=len(past)+1,cross_learning=False,limit_prediction_length=True)
        q=forecasts[0][0].detach().float().cpu().numpy();actual=target[-prediction:];last=float(target[-prediction-1])
        np.savez_compressed(directory/f'{index}.npz',quantiles=q,actual=actual,levels=np.asarray(QUANTILES),last=np.asarray(last))
        endpoints={str(hour):metrics(q[[int(hour*3600/interval)-1]],actual[[int(hour*3600/interval)-1]],last)
            for hour in (1,3,6,12,24) if hour*3600%interval==0 and hour*3600/interval<=prediction}
        rows.append(dict(task_index=int(index),**metrics(q,actual,last),horizons_hours=endpoints))
    return dict(tasks=len(rows),mae_fraction=float(np.mean([r['mae_fraction'] for r in rows])),
        flat_mae_fraction=float(np.mean([r['flat_mae_fraction'] for r in rows])),
        quantile_loss_fraction=float(np.mean([r['quantile_loss_fraction'] for r in rows])),
        direction_correct=sum(r['direction_correct'] for r in rows),direction_count=sum(r['direction_count'] for r in rows),
        amplitude_error_fraction=float(np.mean([r['amplitude_error_fraction'] for r in rows])),
        interval_covered=sum(r['q10_q90_covered'] for r in rows),points=sum(r['points'] for r in rows),
        interval_coverage=float(sum(r['q10_q90_covered'] for r in rows)/sum(r['points'] for r in rows)),
        horizons_hours={hour:{key:float(np.mean([r['horizons_hours'][hour][key] for r in rows]))
            for key in rows[0]['horizons_hours'][hour]} for hour in rows[0]['horizons_hours']},rows=rows)


def train(paths,output,facts_path=None,steps=1500,context=128,prediction=24,device='cuda:0',eval_limit=48,
          interval_seconds=None,eval_steps=100,patience=4,status_path=None):
    import torch
    from chronos import Chronos2Pipeline
    from transformers import TrainerCallback,EarlyStoppingCallback
    root=Path(output)
    if root.exists():raise ValueError('New training output directory required')
    frames={Path(p).stem:load_market(p,interval_seconds) for p in paths};facts=pd.read_csv(facts_path) if facts_path else None
    tasks,splits=prepare(frames,context,prediction,facts)
    root.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(20261021);np.random.seed(20261021)
    source_root=Path(__file__).resolve().parents[1]
    receipt=dict(status='loading',created_at=datetime.now(timezone.utc).isoformat(),inputs={str(Path(p).resolve()):_sha256(p) for p in paths},
        fundamentals=dict(path=str(facts_path),sha256=_sha256(facts_path)) if facts_path else {'status':'not_supplied_no_fabricated_history'},
        model=dict(path=str(LOCAL_MODEL),sha256=_sha256(LOCAL_MODEL/'model.safetensors')),
        source_sha256={str(p):_sha256(p) for p in (Path(__file__),source_root/'wallex_lab/chronos_market.py',source_root/'scripts/finetune_chronos2.py')},
        settings=dict(steps=steps,context=context,prediction=prediction,device=device,mode='full',seed=20261021,
            interval_seconds=splits['interval_seconds'],eval_steps=eval_steps,early_stopping_patience=patience,
            learning_rate=1e-6,gradient_accumulation_steps=4,trained_quantiles=QUANTILES,cross_learning=False),
        splits=splits,logs=[],selection='Lowest VALIDATION loss; TEST not used to choose checkpoints or hyperparameters',
        limitations=['Unknown pretrained financial data overlap.','Quantiles are marginal, not joint sampled price paths or calibrated fill probabilities.',
            'Training window sampling is random inside contiguous TRAIN segments; no TEST context or labels enter training.',
            'No historical macro or fundamental inputs are fabricated. Any supplied facts require explicit available_at; source PIT provenance still requires an independent audit.',
            'Reference crypto candles are Binance USDT spot; target Wallex venue and execution costs require separate validation.',
            'Validation/test full contexts are confined to each split, purging shared label and context rows.'])
    covariates_manifest=source_root/'data/chronos-covariates-20261021/manifest.json'
    if facts_path and covariates_manifest.exists():receipt['fundamental_source_manifest']={'path':str(covariates_manifest),'sha256':_sha256(covariates_manifest)}
    metadata=root/'training.json';save(metadata,receipt);started=time.perf_counter()
    def publish(status,**fields):
        receipt.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),**fields);save(metadata,receipt)
        if status_path:save(status_path,dict(model='chronos2',status=status,updated_at=receipt['updated_at'],
            started_at=receipt['created_at'],actual_steps=receipt.get('global_step',0),maximum_steps=steps,
            checkpoint=receipt.get('selected_checkpoint'),selected_checkpoint=receipt.get('selected_checkpoint'),
            adapter_module='wallex_lab.council_chronos',model_available_at=receipt.get('completed_at'),
            promotion_allowed=receipt.get('promotion',{}).get('approved',False),
            recommended_variant='finetuned' if receipt.get('promotion',{}).get('approved',False) else 'pretrained',
            benchmark_inference_seconds=receipt.get('cpu_benchmark',{}).get('inference_seconds'),training_receipt=str(metadata.resolve()),
            logs=receipt['logs'][-32:],best_validation_loss=receipt.get('best_validation_loss'),
            variant='finetuned' if receipt.get('promotion',{}).get('approved',False) else 'pretrained',
            metrics={k:{name:value for name,value in receipt[k].items() if name!='rows'} for k in ('validation_baseline','validation_final','baseline','final','promotion') if k in receipt},
            **({'error':receipt['error']} if 'error' in receipt else {})))
    class Audit(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            receipt['logs'].append(dict(step=state.global_step,**(logs or {})));publish('training',global_step=state.global_step)
        def on_train_end(self,args,state,control,**kwargs):
            receipt.update(global_step=state.global_step,best_validation_loss=state.best_metric,best_model_checkpoint=state.best_model_checkpoint)
    try:
        publish('queued');lock=source_root/'runs/foundation-training.gpu.lock'
        with exclusive(lock,timeout=3600):
            print('Chronos-2 GPU lock acquired',flush=True);publish('evaluating')
            pipeline=Chronos2Pipeline.from_pretrained(str(LOCAL_MODEL),device_map=device,torch_dtype=torch.float32,local_files_only=True,trust_remote_code=False)
            receipt['validation_baseline']=evaluate(pipeline,tasks['validation'],context,prediction,root/'baseline-validation',eval_limit,splits['interval_seconds'])
            receipt['baseline']=evaluate(pipeline,tasks['test'],context,prediction,root/'baseline-test',eval_limit,splits['interval_seconds'])
            width=1+len(tasks['train'][0]['past_covariates']);receipt['settings']['batch_size_variates']=width
            publish('training')
            fine=pipeline.fit(inputs=tasks['train'],validation_inputs=tasks['validation'],prediction_length=prediction,
                context_length=context,min_past=context,finetune_mode='full',learning_rate=1e-6,num_steps=steps,batch_size=width,
                output_dir=str(root/'checkpoints'),finetuned_ckpt_name='selected-model',callbacks=[Audit(),EarlyStoppingCallback(early_stopping_patience=patience)],
                gradient_accumulation_steps=4,bf16=device.startswith('cuda') and torch.cuda.is_bf16_supported(),tf32=False,
                save_steps=eval_steps,eval_steps=eval_steps,logging_steps=25,seed=20261021,data_seed=20261021,save_total_limit=2,
                dataloader_pin_memory=False,disable_tqdm=True)
            publish('evaluating',selected_checkpoint=str((root/'checkpoints/selected-model').resolve()))
            receipt['validation_final']=evaluate(fine,tasks['validation'],context,prediction,root/'finetuned-validation',eval_limit,splits['interval_seconds'])
            receipt['final']=evaluate(fine,tasks['test'],context,prediction,root/'finetuned-test',eval_limit,splits['interval_seconds'])
            base=receipt['validation_baseline'];adapted=receipt['validation_final']
            receipt['promotion']={'approved':adapted['mae_fraction']<min(base['mae_fraction'],adapted['flat_mae_fraction'])*.98
                and adapted['quantile_loss_fraction']<base['quantile_loss_fraction'] and .7<=adapted['interval_coverage']<=.9,
                'rule':'VALIDATION only: >=2% MAE improvement over pretrained AND flat, improved pinball, q10-q90 coverage 70%-90%; advisory only',
                'live_trading_approved':False}
            receipt['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0
            benchmark_pipeline=fine if receipt['promotion']['approved'] else pipeline
            benchmark_pipeline.model.to('cpu').eval().requires_grad_(False)
            del pipeline,fine
            if device.startswith('cuda'):torch.cuda.empty_cache()
        print('Chronos-2 GPU lock released',flush=True)
        from .council_chronos import forecast
        receipt['completed_at']=datetime.now(timezone.utc).isoformat();publish('evaluating')
        benchmark_request=dict(asof=min(pd.Timestamp(frame.timestamp.iloc[-1])+pd.Timedelta(seconds=splits['interval_seconds']) for frame in frames.values()).isoformat(),
            markets=[dict(symbol=Path(path).stem.split('_')[0],candles_path=str(Path(path).resolve())) for path in paths],
            settings=dict(device='cpu',context=context,prediction=prediction,status_path=str(status_path or source_root/'runs/market-council-20261022/model-status/chronos2.json')))
        forecast(benchmark_request,pipeline=benchmark_pipeline)
        receipt['cpu_benchmark']=forecast(benchmark_request,pipeline=benchmark_pipeline)
        save(root/'cpu-four-asset-benchmark.json',receipt['cpu_benchmark'])
        receipt.update(elapsed_seconds=time.perf_counter()-started,completed_at=datetime.now(timezone.utc).isoformat(),
            selected_checkpoint=str((root/'checkpoints/selected-model').resolve()),
            selected_files={p.name:_sha256(p) for p in (root/'checkpoints/selected-model').glob('*') if p.is_file()},
            )
        publish('complete')
    except BaseException as error:
        publish('failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-started);raise
    save(metadata,receipt);return receipt
