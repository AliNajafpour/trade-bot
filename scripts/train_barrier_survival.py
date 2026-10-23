"""Reproducible short-history survival experiment; never submits orders.

python -B -m scripts.train_barrier_survival --output runs/barrier-survival-20261022
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from wallex_lab.barrier_dataset import HORIZONS,FOLDS,MAX_CONTEXT,build_dataset,fold_roles,sha256,write_json
from wallex_lab.barrier_survival import MultiScaleSurvival,SurvivalPredictor,risk_nll,incidence,fit_temperature,mixture_nll


def horizon_targets(event,cause,survive):
    horizons=np.arange(5);event=event[...,None];cause=cause[...,None];survive=survive[...,None]
    occurred=(event>=0)&(event<=horizons)
    known_survival=((event>horizons)|((event<0)&(survive>horizons)))
    y=np.where(occurred&(cause<2),cause,np.where(known_survival,2,-1)).astype(np.int8)
    return y,y>=0


def probability_metrics(p,event,cause,survive,score_mask=None):
    y,mask=horizon_targets(event,cause,survive);rows=[];unavailable=int((~mask).sum())
    if score_mask is not None:
        score_mask=np.asarray(score_mask,bool)
        if score_mask.shape!=event.shape:raise ValueError('Query scoring mask shape mismatch')
        mask=mask&score_mask[...,None]
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1,atol=2e-6):raise ValueError('Invalid distribution')
    if (np.diff(p[...,:2],axis=-2)<-2e-6).any() or (np.diff(p[...,2],axis=-1)>2e-6).any():raise ValueError('Incoherent time probabilities')
    for k,h in enumerate(HORIZONS):
        actual=y[...,k][mask[...,k]];forecast=p[...,k,:][mask[...,k]];reliability={}
        for cls,name in enumerate(('tp','sl','no_touch')):
            bins=[]
            for lower in np.arange(10)/10:
                selected=(forecast[:,cls]>=lower)&(forecast[:,cls]<(lower+.1 if lower<.9 else 1.000001))
                bins.append(dict(low=float(lower),count=int(selected.sum()),
                    predicted=float(forecast[selected,cls].mean()) if selected.any() else None,
                    observed=float((actual[selected]==cls).mean()) if selected.any() else None))
            reliability[name]=bins
        rows.append(dict(horizon_minutes=int(h),scored=len(actual),total=int(y[...,k].size),
            log_loss=float(-np.log(np.maximum(forecast[np.arange(len(actual)),actual],1e-12)).mean()) if len(actual) else None,
            brier=float(((forecast-np.eye(3)[actual])**2).sum(-1).mean()) if len(actual) else None,
            class_frequencies=[float((actual==i).mean()) for i in range(3)] if len(actual) else None,reliability=reliability))
    return dict(by_horizon=rows,excluded_ambiguous_or_unobserved=unavailable,
        excluded_by_query_mask=int((~score_mask).sum()) if score_mask is not None else 0,
        exact_cause_ambiguities=int((cause==2).sum()))


def origin_log_losses(p,event,cause,survive):
    y,mask=horizon_targets(event,cause,survive)
    values=-np.log(np.maximum(np.take_along_axis(p,np.maximum(y,0)[...,None],axis=-1)[...,0],1e-12))
    count=mask.sum(1)
    return np.divide((values*mask).sum(1),count,out=np.full(count.shape,np.nan),where=count>0)


def daily_bootstrap(a,b,time_ns,repetitions=500):
    difference=a-b;days=np.asarray(time_ns)//86_400_000_000_000;unique=np.unique(days)
    sums=[];counts=[]
    for day in unique:
        part=difference[days==day];sums.append(np.nansum(part,axis=0));counts.append(np.isfinite(part).sum(0))
    sums=np.asarray(sums);counts=np.asarray(counts);rng=np.random.default_rng(20261022)
    means=[]
    for _ in range(repetitions):
        chosen=rng.integers(len(unique),size=len(unique));denom=counts[chosen].sum(0)
        means.append(np.divide(sums[chosen].sum(0),denom,out=np.zeros(sums.shape[1]),where=denom>0))
    return dict(days=len(unique),repetitions=repetitions,unit='UTC calendar day, all markets and queries together',
        mean_difference=np.nanmean(difference,axis=0).tolist(),lower_95=np.quantile(means,.025,axis=0).tolist(),
        upper_95=np.quantile(means,.975,axis=0).tolist(),
        limitation='Few dependent calendar days; intervals are descriptive, not proof of generalization.')


def collect(model,tensors,indices,batch=128):
    model.eval();device=tensors['x'].device;model.to(device);result=[]
    with torch.inference_mode():
        for start in range(0,len(indices),batch):
            i=torch.as_tensor(indices[start:start+batch],device=device)
            result.append(model(tensors['x'][i],tensors['q'][i],tensors['symbol'][i]).cpu())
    model.cpu()
    return torch.cat(result)


def fit_model(tensors,train,val,config,seed,path,args):
    if path.exists():raise FileExistsError(path)
    path.mkdir(parents=True);torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device=tensors['x'].device;model=MultiScaleSurvival(**config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed);best=float('inf');state=None;stale=0;curve=[];started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();order=rng.permutation(train);total=0.;count=0
        for start in range(0,len(order),args.batch):
            i=torch.as_tensor(order[start:start+args.batch],device=device)
            optimizer.zero_grad(set_to_none=True)
            z=model(tensors['x'][i],tensors['q'][i],tensors['symbol'][i])
            loss=risk_nll(z,tensors['event'][i],tensors['cause'][i],tensors['survive'][i])
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(i);count+=len(i)
        model.eval();validation_total=0.;validation_count=0
        with torch.inference_mode():
            for start in range(0,len(val),128):
                i=torch.as_tensor(val[start:start+128],device=device)
                score=risk_nll(model(tensors['x'][i],tensors['q'][i],tensors['symbol'][i]),tensors['event'][i],tensors['cause'][i],tensors['survive'][i])
                validation_total+=float(score)*len(i);validation_count+=len(i)
        validation=validation_total/validation_count
        row=dict(epoch=epoch+1,train_nll=total/count,selection_nll=validation,elapsed_seconds=time.perf_counter()-started)
        curve.append(row);write_json(path/'curve.json',curve)
        print(f'{path.name} epoch {epoch+1}: fit={total/count:.5f} select={validation:.5f}',flush=True)
        if validation<best-1e-5:
            best=validation;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=args.patience:break
    model.cpu();model.load_state_dict(state)
    torch.save(state,path/'best.pt')
    summary=dict(seed=seed,config=config,best_selection_nll=best,epochs=len(curve),fit_origins=len(train),
        selection_origins=len(val),seconds=time.perf_counter()-started,checkpoint_sha256=sha256(path/'best.pt'))
    write_json(path/'fit.json',summary)
    return model,summary


def label_tensors(arrays,indices):
    return [torch.as_tensor(arrays[key][indices],dtype=torch.long) for key in ('event_bin','cause','survived_bins')]


def probabilities(logits,temperature):
    return torch.stack([incidence(z/temperature) for z in logits]).mean(0).numpy()


def make_report(output,result):
    lines=['# Short-history arbitrary-barrier survival results','',
        'Historical price-touch evaluation. No orders, fills, fees or realized trading return are claimed.',
        'The source dates have appeared in earlier project experiments; this is a held-out test for this new pipeline, not project-wide untouched or prospective evidence.','',
        f"Dataset: {result['dataset']['origins']:,} origins, {result['dataset']['queries_per_origin']} barrier queries per origin, four native Wallex USDT markets.",
        'Unresolved within-minute double touches retain unknown cause. Opening-price touches resolve the cause. Missing minutes censor observation. All split boundaries purge 24 hours.',
        'The 180/360-minute choice uses only early-fold selection NLL. Each fold uses three seeds, separate calibration dates, and a separately calibrated query-only baseline.','',
        '|Fold|Context min|Calibration temperature|Horizon min|Baseline log loss|Raw ensemble log loss|Calibrated log loss|Calibrated Brier|Scored / total|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fold in result['folds']:
        for base,raw,cal in zip(fold['metrics']['query_only']['by_horizon'],fold['metrics']['raw']['by_horizon'],fold['metrics']['calibrated']['by_horizon']):
            lines.append(f"|{fold['name']}|{fold['context']}|{fold['temperature']:.4f}|{cal['horizon_minutes']}|{base['log_loss']:.5f}|{raw['log_loss']:.5f}|{cal['log_loss']:.5f}|{cal['brier']:.5f}|{cal['scored']} / {cal['total']}|")
    lines+=['','Negative paired log-loss differences favor the calibrated history ensemble. See results.json for daily-block intervals, classwise reliability, symbol/geometry slices, censoring and ambiguity coverage.',
        'Model probabilities are coherent across time; monotonicity across arbitrary barrier pairs is measured separately and is not guaranteed by this architecture.',
        'Historical entries are last completed close with small predetermined offsets, not observed executable ask/bid prices. Inference accepts an explicit supplied entry and reports extrapolation flags.',
        'No-touch is remaining probability mass at the horizon. It never closes a position or implies zero continuation value.','',
        '## Reproduce','',
        '```powershell',
        "$py = 'D:\\ALI\\Programming-AI\\Personal Projects and Tests\\Trade-Bot\\.venv-gpu\\Scripts\\python.exe'",
        '& $py -B -m checks.check_barrier_survival','& $py -B -m checks.check_barrier_training',
        '& $py -B -m scripts.train_barrier_survival --output runs/barrier-survival-repeat --dataset data/barrier-survival-20261022',
        '```','',
        'Use a new output directory. The existing frozen dataset is hash-verified, not overwritten. Final model bundles are under models/early and models/late inside the run.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='data/fresh-history-20261022')
    parser.add_argument('--dataset',default='data/barrier-survival-20261022')
    parser.add_argument('--output',default='runs/barrier-survival-20261022')
    parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--batch',type=int,default=64);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.epochs<1 or args.batch<1 or args.patience<1:raise ValueError('Positive training settings required')
    output=Path(args.output);dataset=Path(args.dataset)
    if output.exists():raise FileExistsError('Preserve previous experiment outputs')
    output.mkdir(parents=True)
    sources=[Path(__file__),Path('wallex_lab/barrier_dataset.py'),Path('wallex_lab/barrier_survival.py')]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),horizons=HORIZONS.tolist(),folds=FOLDS,
        lookback_candidates=[180,360],architecture_selection='Early-fold selection only; freeze before any test scores.',
        ensemble_seeds=[42,43,44],baseline='Query-only MLP, same partial likelihood and separate calibration.',
        entry_proxy='Last completed close times exp(offset); offset cycles -0.5/0/+0.5 trailing minute sigma, fixed before labels.',
        semantics='Price touch on future native M1; no automatic horizon exit, no exchange orders.',
        test_status='Historical held-out for this pipeline; market periods exposed to earlier project research.',
        source_hashes={str(p):sha256(p) for p in sources},arguments=vars(args),torch=torch.__version__,numpy=np.__version__,
        calibration='Single shared temperature 0.25..8 fitted to mixture event/censor likelihood; no independent horizon remapping.',
        bootstrap='500 paired UTC-day blocks, retaining all markets and queries per day.')
    write_json(output/'protocol.json',protocol);protocol_hash=sha256(output/'protocol.json')
    if not dataset.exists():
        print('Building audited causal dataset',flush=True);build_dataset(args.source,dataset)
    meta=json.loads((dataset/'dataset.json').read_text(encoding='utf-8'))
    for name,digest in meta['files'].items():
        if sha256(dataset/name)!=digest:raise ValueError('Dataset hash mismatch')
    write_json(output/'dataset-receipt.json',dict(path=str(dataset.resolve()),manifest_sha256=sha256(dataset/'dataset.json'),files=meta['files']))
    arrays=dict(np.load(dataset/'labels.npz',allow_pickle=False));history=np.load(dataset/'history.npy',mmap_mode='r')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    tensors=dict(x=torch.tensor(np.array(history),device=device),q=torch.tensor(arrays['query'],device=device),
        symbol=torch.tensor(arrays['symbol'],device=device,dtype=torch.long),event=torch.tensor(arrays['event_bin'],device=device,dtype=torch.long),
        cause=torch.tensor(arrays['cause'],device=device,dtype=torch.long),survive=torch.tensor(arrays['survived_bins'],device=device,dtype=torch.long))
    times=pd.to_datetime(arrays['time_ns'],utc=True);frozen=[];selected_context=None
    for fold in FOLDS:
        roles=fold_roles(times,fold['boundaries']);indices=[np.flatnonzero(roles==role) for role in range(4)]
        if min(map(len,indices))<100:raise ValueError('Insufficient chronological origins')
        fit,selection,calibration,test=indices
        identity=dict(name=fold['name'],boundaries=fold['boundaries'],origins_by_role=[len(i) for i in indices],
            query_counts_by_role=[len(i)*meta['queries_per_origin'] for i in indices],purged_origins=int((roles<0).sum()))
        write_json(output/(fold['name']+'-split.json'),identity)
        candidates=[]
        for context in ([180,360] if selected_context is None else [selected_context]):
            config=dict(symbols=len(meta['symbols']),context=context,width=64,layers=2)
            model,summary=fit_model(tensors,fit,selection,config,42,output/f"{fold['name']}-context{context}-seed42",args)
            candidates.append((model,summary))
        chosen=min(candidates,key=lambda pair:pair[1]['best_selection_nll']);selected_context=chosen[1]['config']['context']
        write_json(output/(fold['name']+'-selection.json'),dict(selected_context=selected_context,
            candidates=[summary for _,summary in candidates],selected_without_test=True))
        models=[chosen[0]]
        for seed in (43,44):
            model,_=fit_model(tensors,fit,selection,chosen[1]['config'],seed,output/f"{fold['name']}-context{selected_context}-seed{seed}",args)
            models.append(model)
        baseline,_=fit_model(tensors,fit,selection,dict(symbols=len(meta['symbols']),context=selected_context,width=64,layers=2,query_only=True),77,output/(fold['name']+'-query-only'),args)
        cal_labels=label_tensors(arrays,calibration)
        cal_logits=[collect(model,tensors,calibration) for model in models]
        baseline_cal=[collect(baseline,tensors,calibration)]
        temperature=fit_temperature(cal_logits,*cal_labels);baseline_temperature=fit_temperature(baseline_cal,*cal_labels)
        model_meta=dict(symbols=meta['symbols'],context=selected_context,width=64,layers=2,temperature=temperature,
            geometry_support=meta['geometry_support'],semantics='price_touch',protocol_sha256=protocol_hash,
            dataset_manifest_sha256=sha256(dataset/'dataset.json'),fold=identity,
            historical_entry_proxy=protocol['entry_proxy'],test_status=protocol['test_status'])
        SurvivalPredictor.save(output/'models'/fold['name'],models,model_meta)
        SurvivalPredictor.save(output/'models'/(fold['name']+'-baseline'),[baseline],dict(model_meta,temperature=baseline_temperature))
        calibration_summary=dict(temperature=temperature,baseline_temperature=baseline_temperature,
            raw_nll=float(mixture_nll(cal_logits,*cal_labels)),calibrated_nll=float(mixture_nll(cal_logits,*cal_labels,temperature)),
            selection_frozen_before_tests=True)
        write_json(output/(fold['name']+'-calibration.json'),calibration_summary)
        frozen.append(dict(fold=fold,models=models,baseline=baseline,test=test,calibration=calibration_summary,context=selected_context))
        print(f"Frozen {fold['name']}: context={selected_context}, temperature={temperature:.4f}; tests still unopened",flush=True)
    write_json(output/'choices-frozen.json',dict(protocol_sha256=protocol_hash,context=selected_context,
        models={r['fold']['name']:{'temperature':r['calibration']['temperature'],'test_origins':len(r['test'])} for r in frozen}))
    results=[]
    for item in frozen:
        name=item['fold']['name'];test=item['test'];event=arrays['event_bin'][test];cause=arrays['cause'][test];survive=arrays['survived_bins'][test]
        logits=[collect(model,tensors,test) for model in item['models']]
        baseline_logits=[collect(item['baseline'],tensors,test)]
        forecasts=dict(raw=probabilities(logits,1.),calibrated=probabilities(logits,item['calibration']['temperature']),
            query_only=probabilities(baseline_logits,item['calibration']['baseline_temperature']))
        metrics={key:probability_metrics(p,event,cause,survive) for key,p in forecasts.items()}
        slices={}
        for asset,symbol in enumerate(meta['symbols']):
            mask=arrays['symbol'][test]==asset
            slices[symbol]={key:probability_metrics(p[mask],event[mask],cause[mask],survive[mask]) for key,p in forecasts.items()}
        geometries={str(j):probability_metrics(forecasts['calibrated'][:,j:j+1],event[:,j:j+1],cause[:,j:j+1],survive[:,j:j+1]) for j in range(meta['queries_per_origin'])}
        diff=daily_bootstrap(origin_log_losses(forecasts['calibrated'],event,cause,survive),
            origin_log_losses(forecasts['query_only'],event,cause,survive),arrays['time_ns'][test])
        np.savez_compressed(output/(name+'-test-predictions.npz'),**forecasts,origin_index=test,event_bin=event,cause=cause,
            survived_bins=survive,time_ns=arrays['time_ns'][test],symbol=arrays['symbol'][test])
        row=dict(name=name,context=item['context'],temperature=item['calibration']['temperature'],metrics=metrics,
            paired_log_loss_vs_baseline=diff,symbol_slices=slices,geometry_slices=geometries,
            partial_label_test_nll=dict(raw=float(mixture_nll(logits,*label_tensors(arrays,test))),
                calibrated=float(mixture_nll(logits,*label_tensors(arrays,test),item['calibration']['temperature'])),
                query_only=float(mixture_nll(baseline_logits,*label_tensors(arrays,test),item['calibration']['baseline_temperature']))))
        results.append(row);print(name+' test log losses: '+str([r['log_loss'] for r in metrics['calibrated']['by_horizon']]),flush=True)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=protocol_hash,dataset=meta,folds=results,
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        limitations=['Price touch, not fill or trading profit.','Historical test dates exposed to prior project experiments.',
            'Arbitrary-barrier monotonicity not guaranteed.','Within-minute cause ambiguity cannot be resolved from M1 OHLCV.',
            'Only 90 calendar days and four correlated markets; daily bootstrap intervals are descriptive.'])
    write_json(output/'results.json',result);make_report(output,result)
    print('COMPLETE '+str((output/'report.md').resolve()),flush=True)


if __name__=='__main__':main()
