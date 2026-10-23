"""Streaming yearly M1 survival training: all five-minute origins, rotating dense queries."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from wallex_lab.barrier_dataset import HORIZONS,fold_roles,sha256,write_json
from wallex_lab.barrier_dense import IndexedBarrierDataset,QUERIES,CONTEXT
from wallex_lab.barrier_survival import MultiScaleSurvival,SurvivalPredictor,risk_nll,incidence,fit_temperature,mixture_nll
from scripts.train_barrier_survival import probability_metrics,daily_bootstrap,origin_log_losses


def epoch_queries(epoch,chunk_size=32):
    if chunk_size==128:return np.arange(128)
    if chunk_size!=32:raise ValueError('Query chunk must be 32 or 128')
    start=(epoch%4)*16
    return np.r_[start:start+16,start+64:start+80]


def collect_stream(model,data,indices,device,batch=128,shuffle_history=False):
    data.context=max(CONTEXT,model.context)
    model=model.to(device).eval();out=[];rng=np.random.default_rng(8021)
    with torch.inference_mode():
        for start in range(0,len(indices),batch):
            values=data.batch(indices[start:start+batch],device=device,with_history=not model.query_only)
            if shuffle_history:
                values['x']=values['x'][:,-model.context:]
                order=torch.as_tensor(rng.permutation(values['x'].shape[1]),device=device)
                values['x']=values['x'][:,order]
            out.append(model(values['x'],values['q'],values['symbol']).float().cpu())
    model.cpu()
    return torch.cat(out)


def labels_for(data,indices):
    return [torch.as_tensor(np.array(data.arrays[key][indices]),dtype=torch.long) for key in ('event_bin','cause','survived_bins')]


def fit_stream(data,fit,selection,config,seed,path,args):
    if path.exists():raise FileExistsError(path)
    path.mkdir(parents=True);torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
    device=torch.device(args.device);model=MultiScaleSurvival(**config).to(device)
    data.context=max(CONTEXT,model.context)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.03)
    rng=np.random.default_rng(seed);best=float('inf');best_state=None;best_epoch=None;stale=0;curve=[];started=time.perf_counter()
    amp=device.type=='cuda' and torch.cuda.is_bf16_supported()
    all_labels=labels_for(data,selection)
    chunk_size=getattr(args,'query_chunk',32)
    for epoch in range(args.epochs):
        model.train();order=rng.permutation(fit);queries=epoch_queries(epoch,chunk_size);total=0.;count=0
        batch_size=args.batch if not model.query_only else args.batch*4
        for start in range(0,len(order),batch_size):
            values=data.batch(order[start:start+batch_size],queries,device=device,with_history=not model.query_only)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext():
                logits=model(values['x'],values['q'],values['symbol'])
                loss=risk_nll(logits.float(),values['event'],values['cause'],values['survive'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training likelihood')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            number=len(values['symbol']);total+=float(loss.detach())*number;count+=number
            if count%max(batch_size*500,1)<batch_size:
                write_json(path/'progress.json',dict(epoch=epoch+1,origins_seen=count,fit_origins=len(fit),query_indices=queries.tolist(),seconds=time.perf_counter()-started))
                print(f'{path.name} epoch={epoch+1} origins={count}/{len(fit)}',flush=True)
        val_logits=collect_stream(model,data,selection,device,args.batch)
        score=float(risk_nll(val_logits,*all_labels));model.to(device)
        row=dict(epoch=epoch+1,fit_nll=total/count,selection_nll=score,query_indices=queries.tolist(),seconds=time.perf_counter()-started)
        curve.append(row);write_json(path/'curve.json',curve)
        print(f'{path.name} epoch {epoch+1}: fit={total/count:.5f}, selection={score:.5f}, seconds={row["seconds"]:.1f}',flush=True)
        if (chunk_size==128 or epoch>=3) and score<best-1e-5:
            best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            torch.save(best_state,path/'best.pt')
        else:stale+=1
        if (chunk_size==128 or epoch>=3) and stale>=args.patience:break
    model.cpu();model.load_state_dict(best_state)
    summary=dict(seed=seed,config=config,best_selection_nll=best,epochs=len(curve),fit_origins=len(fit),selection_origins=len(selection),
        query_coverage=sorted(set(q for r in curve for q in r['query_indices'])),selected_epoch=best_epoch,
        selected_query_coverage=sorted(set(q for r in curve[:best_epoch] for q in r['query_indices'])),checkpoint_sha256=sha256(path/'best.pt'),seconds=time.perf_counter()-started,
        precision='BF16 matmul + FP32 likelihood' if amp else 'FP32',stopping='Validation only, checkpoints eligible only after all 128 queries have been presented')
    write_json(path/'fit.json',summary)
    return model,summary


def predict_from_logits(logits,temperature):
    total=None
    for z in logits:
        p=incidence(z/temperature).numpy()
        total=p if total is None else total+p
    return total/len(logits)


def save_report(root,result):
    meta=result['dataset'];lines=['# Dense current-history barrier survival','',
        f"{meta['origins']:,} indexed origins; {result['common_origins']:,} common origins eligible for every tested context, every five minutes, {meta['queries_per_origin']} queries per origin; {sum(s['rows'] for s in meta['sources']):,} native M1 candles.",
        f"Acquisition cutoff: {meta['acquisition_protocol']['requested_to_exclusive_utc']}. Prices remain in native USDT; no gold or named technical-pattern labels.",
        'Price/volume unit invariance, causal windows, unknown same-minute causes, missing-minute censoring and 24h split purges are checked.',
        'All eligible fit origins and all 128 barrier requests are used in every epoch. Origins lacking a contiguous week are excluded identically for every candidate. Shared raw arrays reside on GPU, with double-precision normalization and FP32 feature output. No random history or request subsampling.',
        'Three independent seeds, query-only comparison, separate temperature calibration and two expanding historical folds.','',
        '|Fold|Context min|Horizon min|Query-only log loss|Raw log loss|Calibrated log loss|Calibrated Brier|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for fold in result['folds']:
        for k,h in enumerate(HORIZONS):
            m=fold['metrics'];lines.append(f"|{fold['name']}|{fold['context']}|{h}|{m['query_only']['by_horizon'][k]['log_loss']:.5f}|{m['raw']['by_horizon'][k]['log_loss']:.5f}|{m['calibrated']['by_horizon'][k]['log_loss']:.5f}|{m['calibrated']['by_horizon'][k]['brier']:.5f}|")
    lines+=['','## Context selection on common early-fold validation origins','',
        '|Input context min|Best validation partial-label NLL|Selected epoch|','|---:|---:|---:|']
    for candidate in result['context_comparison']:
        lines.append(f"|{candidate['config']['context']}|{candidate['best_selection_nll']:.6f}|{candidate['selected_epoch']}|")
    lines+=['','Window length and its declared hierarchical aggregation are compared together. Selection uses seed42; two additional seeds are trained for the chosen context. All held-out metrics above are scored after choices freeze.','',
        '## Pattern learning and normalization','',
        'Convolutional patches preserve local candle shape; hierarchical aggregation retains recent M1 detail and extends positional attention through up to a week. Barriers query this history by cross-attention. No triangle, head-and-shoulders or support/resistance labels are supplied.',
        'Capacity and sensitivity to history are not evidence that a named technical pattern was discovered. The shuffled-history diagnostic is a perturbation experiment, not a semantic pattern-recognition score.',
        'OHLC uses log(price / last completed close) divided by causal history volatility. Volume uses its ratio to recent history. TP/SL use relative log distances; volatility is supplied separately. Multiplying all prices by a constant does not change the modeled geometry.',
        'The new normalization is explicitly versioned relative-v2. Old bundles retain relative-v1 behavior.','',
        '## Limits','',
        'Five-minute origins, mirrored directions and adjacent queries are correlated. Resampling uses UTC-day blocks, not individual requests. Historical dates overlap prior project experiments; these are held-out folds for this pipeline, not prospective evidence.',
        'No orders, fills, fees or trading profit are evaluated. Entry prices are predetermined historical proxies. Touch horizons never trigger liquidation. Probability coherence across time is constructed; monotonicity across arbitrary barriers is not guaranteed.',
        'Raw full-year sources and hashes, immutable protocol and snapshots, learned weights, prediction files, reliability bins and daily-block comparisons are retained in this run.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='data/barrier-dense-20261022');parser.add_argument('--output',default='runs/barrier-survival-dense-20261022')
    parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--patience',type=int,default=4)
    parser.add_argument('--batch',type=int,default=512);parser.add_argument('--device',default='cuda')
    parser.add_argument('--query-chunk',type=int,choices=[32,128],default=128)
    parser.add_argument('--contexts',type=int,nargs='+',default=[360,720,1440,2880,10080])
    args=parser.parse_args()
    if args.epochs<(4 if args.query_chunk==32 else 1) or args.batch<1 or args.patience<1:raise ValueError('Positive settings and a complete query pass required')
    root=Path(args.output)
    if root.exists():raise FileExistsError('Preserve previous run')
    root.mkdir(parents=True);data=IndexedBarrierDataset(args.dataset);meta=data.metadata
    if not args.contexts or any(c not in (360,720,1440,2880,10080) for c in args.contexts):raise ValueError('Unsupported context candidates')
    eligible=data.eligible_history(max(args.contexts));np.save(root/'common-origin-mask.npy',eligible)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device(args.device)
    if device.type=='cuda':data.preload(device)
    files=[Path(__file__),Path('wallex_lab/barrier_dense.py'),Path('wallex_lab/barrier_dataset.py'),Path('wallex_lab/barrier_survival.py'),Path('scripts/train_barrier_survival.py')]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(args),folds=meta['folds'],
        dataset_manifest_sha256=sha256(Path(args.dataset)/'dataset.json'),contexts=args.contexts,width=96,layers=3,seeds=[42,43,44],
        common_origins=int(eligible.sum()),common_origin_mask_sha256=sha256(root/'common-origin-mask.npy'),
        history_layout='Near: last <=720 M1; middle: <=2880 minutes pooled 5/15m; far: full context pooled 15/60m; learned patches 6/3/1.',
        query_rotation='All 128 queries per origin per epoch' if args.query_chunk==128 else 'Four balanced 32-query chunks',
        batching='GPU-resident shared raw/label arrays; FP64 relative transforms produce FP32 features, no lossy feature cache.',
        source_hashes={str(p):sha256(p) for p in files},selection='Early-fold selection only; all model/calibration choices freeze before tests.',
        torch=torch.__version__,numpy=np.__version__,test_status='Frozen historical holdouts for this pipeline; not project-wide untouched dates.')
    write_json(root/'protocol.json',protocol);protocol_hash=sha256(root/'protocol.json')
    (root/'source-snapshot').mkdir()
    for path in files:shutil.copy2(path,root/'source-snapshot'/path.name)
    frozen=[];context=None;times=pd.to_datetime(data.arrays['time_ns'],utc=True)
    for fold in meta['folds']:
        roles=fold_roles(times,fold['boundaries']);fit,selection,calibration,test=[np.flatnonzero((roles==i)&eligible) for i in range(4)]
        if min(map(len,(fit,selection,calibration,test)))<100:raise ValueError('Insufficient fold coverage')
        split=dict(boundaries=fold['boundaries'],fit=len(fit),selection=len(selection),calibration=len(calibration),test=len(test))
        write_json(root/(fold['name']+'-split.json'),split);candidates=[]
        for value in (args.contexts if context is None else [context]):
            model,summary=fit_stream(data,fit,selection,dict(symbols=len(meta['symbols']),context=value,width=96,layers=3,hierarchical=True),42,
                root/f"{fold['name']}-context{value}-seed42",args)
            candidates.append((model,summary))
            write_json(root/(fold['name']+'-context-screen.json'),[record for _,record in candidates])
        chosen=min(candidates,key=lambda pair:pair[1]['best_selection_nll']);context=chosen[1]['config']['context'];models=[chosen[0]]
        write_json(root/(fold['name']+'-selection.json'),dict(context=context,candidates=[r for _,r in candidates]))
        for seed in (43,44):
            model,_=fit_stream(data,fit,selection,chosen[1]['config'],seed,root/f"{fold['name']}-context{context}-seed{seed}",args);models.append(model)
        baseline,_=fit_stream(data,fit,selection,dict(symbols=len(meta['symbols']),context=context,width=96,layers=3,query_only=True),77,root/(fold['name']+'-query-only'),args)
        cal_logits=[collect_stream(model,data,calibration,device,args.batch) for model in models]
        base_cal=[collect_stream(baseline,data,calibration,device,args.batch)];cal_labels=labels_for(data,calibration)
        temperature=fit_temperature(cal_logits,*cal_labels);base_temperature=fit_temperature(base_cal,*cal_labels)
        record=dict(temperature=temperature,baseline_temperature=base_temperature,raw_nll=float(mixture_nll(cal_logits,*cal_labels)),
            calibrated_nll=float(mixture_nll(cal_logits,*cal_labels,temperature)))
        write_json(root/(fold['name']+'-calibration.json'),record)
        bundle_meta=dict(symbols=meta['symbols'],context=context,width=96,layers=3,temperature=temperature,
            feature_version='relative-v2',minimum_history_minutes=max(CONTEXT,context),geometry_support=meta['geometry_support'],semantics='price_touch',
            protocol_sha256=protocol_hash,dataset_manifest_sha256=protocol['dataset_manifest_sha256'],fold=split,acquisition_cutoff=meta['acquisition_protocol']['requested_to_exclusive_utc'])
        SurvivalPredictor.save(root/'models'/fold['name'],models,bundle_meta)
        SurvivalPredictor.save(root/'models'/(fold['name']+'-baseline'),[baseline],dict(bundle_meta,temperature=base_temperature))
        frozen.append(dict(name=fold['name'],models=models,baseline=baseline,test=test,context=context,calibration=record))
        del cal_logits,base_cal
        print('FROZEN '+fold['name']+' '+str(record),flush=True)
    write_json(root/'choices-frozen.json',dict(context=context,protocol_sha256=protocol_hash,test_scoring_not_started=True))
    results=[]
    for row in frozen:
        test=row['test'];labels=labels_for(data,test);event,cause,survive=[value.numpy() for value in labels]
        logits=[collect_stream(model,data,test,device,args.batch) for model in row['models']]
        baseline=[collect_stream(row['baseline'],data,test,device,args.batch)]
        forecasts=dict(raw=predict_from_logits(logits,1.),calibrated=predict_from_logits(logits,row['calibration']['temperature']),
            query_only=predict_from_logits(baseline,row['calibration']['baseline_temperature']))
        metrics={key:probability_metrics(value,event,cause,survive) for key,value in forecasts.items()}
        daily=daily_bootstrap(origin_log_losses(forecasts['calibrated'],event,cause,survive),origin_log_losses(forecasts['query_only'],event,cause,survive),data.arrays['time_ns'][test])
        sample=test[np.linspace(0,len(test)-1,min(512,len(test)),dtype=int)]
        shuffled=[collect_stream(model,data,sample,device,args.batch,shuffle_history=True) for model in row['models']]
        shuffled_probability=predict_from_logits(shuffled,row['calibration']['temperature']);a,b,c=[v.numpy() for v in labels_for(data,sample)]
        diagnostics=dict(shuffled_history=probability_metrics(shuffled_probability,a,b,c),
            intact_history=probability_metrics(forecasts['calibrated'][np.linspace(0,len(test)-1,min(512,len(test)),dtype=int)],a,b,c),
            note='Order perturbation; not proof of any named chart pattern.')
        np.savez_compressed(root/(row['name']+'-test-predictions.npz'),**forecasts,event_bin=event,cause=cause,survived_bins=survive,
            origin_index=test,time_ns=data.arrays['time_ns'][test],symbol=data.arrays['symbol'][test])
        result=dict(name=row['name'],context=row['context'],calibration=row['calibration'],metrics=metrics,
            daily_log_loss_difference_vs_query_only=daily,history_diagnostic=diagnostics,
            partial_label_nll={key:float(mixture_nll(value,*labels,t)) for key,value,t in
                [('raw',logits,1.),('calibrated',logits,row['calibration']['temperature']),('query_only',baseline,row['calibration']['baseline_temperature'])]})
        results.append(result);write_json(root/(row['name']+'-results.json'),result)
        print('TEST '+row['name']+' '+str([m['log_loss'] for m in metrics['calibrated']['by_horizon']]),flush=True)
        del logits,baseline,forecasts,shuffled
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),dataset=meta,common_origins=int(eligible.sum()),
        context_comparison=json.loads((root/'early-selection.json').read_text(encoding='utf-8'))['candidates'],protocol_sha256=protocol_hash,folds=results,
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
    write_json(root/'results.json',result);save_report(root,result);print('COMPLETE '+str((root/'report.md').resolve()),flush=True)


if __name__=='__main__':main()
