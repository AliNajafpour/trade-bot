"""Descriptive TEST-only16x5x8 matrix; exact old-cache reuse and resumable missing queries."""
import argparse
from datetime import datetime,timezone
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.backtest_survival_threshold import MINUTE,THRESHOLDS,GEOMETRY,causal_batch,make_origins,threshold_candidates
from wallex_lab.barrier_dataset import HORIZONS,barrier_query,sha256,write_json
from wallex_lab.barrier_dense import IndexedBarrierDataset
from wallex_lab.barrier_survival import SurvivalPredictor,incidence
from wallex_lab.barrier_only import simulate
from wallex_lab.first_touch import FirstTouchBook

GRID=list(itertools.product([.005,.01,.02,.04],repeat=2))


def missing_pairs(existing):return [g for g in GRID if g not in [tuple(v) for v in existing]]


def check_coverage(rows):
    keys=[(float(r['tp_fraction']),float(r['sl_fraction']),int(r['forecast_horizon_minutes']),float(r['probability_threshold'])) for r in rows]
    expected=set(itertools.product([.005,.01,.02,.04],[.005,.01,.02,.04],map(int,HORIZONS),THRESHOLDS))
    if len(keys)!=640 or len(set(keys))!=640 or set(keys)!=expected:raise ValueError('Exactly640 unique Cartesian scenarios required')
    return dict(unique_scenarios=640,geometries=16,horizons=5,thresholds=8,complete=True)


def support_flags(pair,support):
    tp,sl=pair;up=np.log1p(tp);down=-np.log1p(-sl);bounds=np.log1p([support['min_fraction'],support['max_fraction']]);flags=[]
    if min(up,down)<bounds[0]-1e-8 or max(up,down)>bounds[1]+1e-8:flags.append('barrier_distance_outside_training_range')
    if up/down<1/support['max_ratio'] or up/down>support['max_ratio']:flags.append('barrier_ratio_outside_training_range')
    return flags


def save_rows(root,rows,protocol):
    path=root/'results-baseline.json';temp=root/'results-baseline.tmp.json';write_json(temp,rows);temp.replace(path)
    flat=[{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows]
    pd.DataFrame(flat).to_csv(root/'matrix-baseline.csv',index=False)
    complete=len(rows)==640
    lines=['# Full TP/SL probability-threshold TEST matrix','',f"Coverage: {len(rows)}/640 independent baseline accounts. Status: {'complete' if complete else 'PARTIAL; missing rows are not results'}.",'',
        'Every account starts with18.246562963625525USDT, native13bp fees and assumed5bp spread/slippage per side. Entries every five minutes, fixed completed-close barriers, one-minute entry latency, TP/SL exits only.',
        'This is a descriptive expansion of the already-observed TEST. No combination is selected as a validated policy from these profits. The previously frozen selected policy is preserved.',
        'The TP0.5%/SL4% pair has a physical log-distance ratio below1/8, outside the saved training support. All40 rows remain visible and flagged; these probabilities are extrapolations.',
        f"Primary source interval: {' to '.join(protocol['primary_intervals']['test'])}. Model/source/account/causality and execution limitations are inherited from the pinned prior protocol.",'',
        'Results and exact no-trade rows: matrix-baseline.csv. Positive rows, if any, remain historical observations. Historical books/depth are absent; price touches and assumed friction do not verify fills.']
    if complete:
        returns=np.array([r['return_pct'] for r in rows]);supported=[r for r in rows if r['geometry_supported']]
        lines+=['',f"Rows: {len(rows)}. Supported geometries:15/16. Profitable baseline rows: {int((returns>0).sum())}; losing: {int((returns<0).sum())}; zero-return: {int((returns==0).sum())}. Supported profitable rows: {sum(r['return_pct']>0 for r in supported)}.",
            'Support flags describe forecast geometry only. The legacy threshold-only execution gates remain fixed for comparability; unsupported rows are not silently removed.']
    (root/'report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


def evaluate_pairs(data,indices,probability,pairs,books,root,protocol,rows):
    start,end=map(pd.Timestamp,protocol['primary_intervals']['test']);done={(r['tp_fraction'],r['sl_fraction']) for r in rows}
    for g,pair in enumerate(pairs):
        if pair in done:continue
        origins=make_origins(data,indices,pair);flags=support_flags(pair,protocol['geometry_support'])
        for h,horizon in enumerate(HORIZONS):
            for p in THRESHOLDS:
                candidates=threshold_candidates(origins,probability[:,g,h],p)
                s=simulate(candidates,books,start,end,stop_spec='candidate',reward_multiple=pair[0]/pair[1],initial_cash=protocol['initial_usdt'],fee_bps=13.,slippage_bps=5.)['summary']
                rows.append(dict(**s,tp_fraction=pair[0],sl_fraction=pair[1],forecast_horizon_minutes=int(horizon),probability_threshold=p,
                    threshold_signals=len(candidates),geometry_supported=not flags,support_flags=flags,physical_log_tp_sl_ratio=float(np.log1p(pair[0])/-np.log1p(-pair[1])),
                    prediction_source='bitwise_reused' if pair in GEOMETRY else 'missing_pair_inference',account_independent=True))
        save_rows(root,rows,protocol);write_json(root/'progress.json',dict(stage='baseline_replay',completed_scenarios=len(rows),required=640,last_pair=pair,pid=os.getpid()))
        print(json.dumps(dict(stage='baseline_replay',completed_scenarios=len(rows),pair=pair)),flush=True)


def infer_missing(data,model,indices,pairs,root,batch_size,device):
    folder=root/'inference-chunks';folder.mkdir(exist_ok=True);answer=np.empty((len(indices),len(pairs),5,3),np.float32)
    started=time.perf_counter();chunk_size=256
    for m in model.models:m.to(device)
    for begin in range(0,len(indices),chunk_size):
        end=min(begin+chunk_size,len(indices));path=folder/f'{begin:05d}-{end:05d}.npz'
        if path.exists():
            cache=np.load(path,allow_pickle=False)
            if not np.array_equal(cache['origin_index'],indices[begin:end]) or not np.array_equal(cache['geometry'],np.array(pairs)):raise ValueError('Inference chunk cohort/geometry changed')
            answer[begin:end]=cache['probability'];continue
        for offset in range(begin,end,batch_size):
            positions=np.arange(offset,min(offset+batch_size,end));chosen=indices[positions];assets=np.asarray(data.arrays['symbol'][chosen]);ends=np.asarray(data.arrays['bar_index'][chosen])
            x=np.empty((len(chosen),model.metadata['context'],10),np.float32);sigma=np.empty(len(chosen));reference=np.empty(len(chosen))
            for asset in np.unique(assets):
                selected=np.flatnonzero(assets==asset);values,times=data.series[asset]
                x[selected],sigma[selected],reference[selected]=causal_batch(values,times,ends[selected],model.metadata['context'])
                assert np.array_equal(times[ends[selected]]+MINUTE,np.asarray(data.arrays['time_ns'][chosen[selected]]))
            q=np.stack([barrier_query(ref,ref*(1+np.array(pairs)[:,0]),ref*(1-np.array(pairs)[:,1]),s,ref) for s,ref in zip(sigma,reference)])
            with torch.inference_mode():
                xt=torch.from_numpy(x).to(device);qt=torch.from_numpy(q).to(device);at=torch.as_tensor(assets.astype(np.int64),device=device)
                answer[positions]=torch.stack([incidence(m(xt,qt,at)/model.metadata['temperature']) for m in model.models]).mean(0).cpu().numpy()
        temp=folder/(path.stem+'.tmp.npz');np.savez_compressed(temp,origin_index=indices[begin:end],geometry=np.array(pairs),probability=answer[begin:end]);temp.replace(path)
        progress=dict(stage='missing_inference',completed=end,total=len(indices),pairs=len(pairs),seconds=time.perf_counter()-started,pid=os.getpid(),device=device)
        write_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
    return answer


def render_heatmaps(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for p in THRESHOLDS:
        fig,axes=plt.subplots(1,5,figsize=(17,3.7),sharey=True)
        selected=[r for r in rows if r['probability_threshold']==p];scale=max(abs(r['return_pct']) for r in selected) or 1
        for ax,h in zip(axes,HORIZONS):
            values=np.array([[next(r['return_pct'] for r in selected if r['forecast_horizon_minutes']==h and (r['tp_fraction'],r['sl_fraction'])==(tp,sl)) for sl in [.005,.01,.02,.04]] for tp in [.005,.01,.02,.04]])
            im=ax.imshow(values,cmap='RdYlGn',vmin=-scale,vmax=scale)
            for i in range(4):
                for j in range(4):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8)
            ax.set(xticks=range(4),xticklabels=['.5','1','2','4'],yticks=range(4),yticklabels=['.5','1','2','4'],xlabel='SL %',title=f'{h} minute forecast')
            ax.text(3,0,'*',ha='right',va='top',fontsize=14)
        axes[0].set_ylabel('TP %');fig.suptitle(f'Strict pTP>{p:.2f}; net return %; independent18.24656USDT accounts; * outside geometry support')
        fig.colorbar(im,ax=list(axes),label='Net return %',shrink=.8);fig.subplots_adjust(top=.78,bottom=.2,right=.86,wspace=.15)
        fig.savefig(root/f'heatmap-p{p:.2f}.png',dpi=130);plt.close(fig)


def run(args):
    torch.set_num_threads(1);root=Path(args.output);source=Path(args.source);root.mkdir(parents=True,exist_ok=True)
    old=json.loads((source/'protocol.json').read_text());verified=json.loads((source/'verification.json').read_text())
    if verified['status']!='passed' or sha256(source/'compact-grid-predictions.npz')!=verified['predictions_sha256']:raise ValueError('Verified original prediction cache required')
    if sha256(Path(old['model_path'])/'model.json')!=old['model_sha256']:raise ValueError('Frozen model changed')
    data=IndexedBarrierDataset('data/barrier-dense-20261022');model=SurvivalPredictor.load(old['model_path'])
    if sha256(data.root/'dataset.json')!=old['dataset_sha256']:raise ValueError('Dataset changed')
    cache=np.load(source/'compact-grid-predictions.npz',allow_pickle=False);all_indices=cache['origin_index'];clocks=data.arrays['time_ns'][all_indices]
    left,right=map(pd.Timestamp,old['ranges']['test']);mask=(clocks>=left.value)&(clocks<right.value);indices=all_indices[mask]
    reused=cache['probability'][mask];existing=[tuple(v) for v in old['geometry_tp_sl_fractions']];missing=missing_pairs(existing)
    if existing!=GEOMETRY or len(indices)!=11566 or len(missing)!=7:raise ValueError('Original native test cohort/grid changed')
    expected=np.flatnonzero(data.eligible_history(10080)&(np.asarray(data.arrays['time_ns'])>=left.value)&(np.asarray(data.arrays['time_ns'])<right.value-1440*MINUTE))
    if not np.array_equal(np.sort(indices),expected):raise ValueError('Exact five minute test cohort required')
    protocol=dict(old,created_at=datetime.now(timezone.utc).isoformat(),purpose='Post-test descriptive full Cartesian TEST matrix. No selection or validated winner.',
        previous_selected_policy_path=str((source/'selected-policy.json').resolve()),previous_selected_policy_sha256=sha256(source/'selected-policy.json'),
        geometry_tp_sl_fractions=GRID,geometry_support=model.metadata['geometry_support'],required_baseline_rows=640,test_origins=len(indices),
        source_protocol_sha256=sha256(source/'protocol.json'),source_cache_sha256=verified['predictions_sha256'],reused_geometries=existing,missing_geometries=missing,
        device_policy='CPU1 until parent explicitly releases GPU lease; chunks resume deterministically.',source_script_sha256=sha256(__file__))
    protocol_path=root/'protocol.json'
    if protocol_path.exists():
        prior=json.loads(protocol_path.read_text())
        for key in ['model_sha256','dataset_sha256','source_cache_sha256','geometry_tp_sl_fractions','source_script_sha256']:
            if prior[key]!=protocol[key]:raise ValueError('Full-grid resume identity mismatch: '+key)
        protocol=prior
    else:
        write_json(protocol_path,protocol);(root/'source-snapshot').mkdir(exist_ok=True);shutil.copy2(__file__,root/'source-snapshot'/Path(__file__).name)
    books={}
    for asset,symbol in enumerate(data.metadata['symbols']):
        values,times=data.series[asset];frame=pd.DataFrame(np.asarray(values),columns=['open','high','low','close','volume']);frame.insert(0,'timestamp',pd.to_datetime(times,utc=True));frame.attrs['interval_seconds']=60
        books[symbol]=FirstTouchBook(frame)
    rows=json.loads((root/'results-baseline.json').read_text()) if (root/'results-baseline.json').exists() else []
    print(json.dumps(dict(stage='started',pid=os.getpid(),test_origins=len(indices),reused_pairs=9,missing_pairs=7)),flush=True)
    evaluate_pairs(data,indices,reused,existing,books,root,protocol,rows)
    new=infer_missing(data,model,indices,missing,root,args.batch,args.device)
    if not np.isfinite(new).all() or (new<0).any() or not np.allclose(new.sum(-1),1,atol=2e-5):raise ValueError('Invalid missing-pair probabilities')
    evaluate_pairs(data,indices,new,missing,books,root,protocol,rows)
    coverage=check_coverage(rows)
    if any(not r['valid_for_comparison'] for r in rows):raise ValueError('Primary grid contains unresolved accounting')
    if any(r['timeout_exits'] for r in rows):raise ValueError('Unexpected time liquidation')
    if sha256(source/'selected-policy.json')!=protocol['previous_selected_policy_sha256']:raise ValueError('Prior selection mutated')
    for m in model.models:m.cpu()
    parity=0.
    for j in [0,len(indices)-1]:
        i=indices[j];asset=int(data.arrays['symbol'][i]);end=int(data.arrays['bar_index'][i]);values,times=data.series[asset]
        history=pd.DataFrame(values[end-10079:end+1],columns=['open','high','low','close','volume']);history.insert(0,'timestamp',pd.to_datetime(times[end-10079:end+1],utc=True));history.attrs['interval_seconds']=60
        decision=pd.Timestamp(int(data.arrays['time_ns'][i]),tz='UTC');reference=float(history.close.iloc[-1])
        for g,pair in enumerate(missing):
            public=model.predict(history,data.metadata['symbols'][asset],decision,reference,tp_fraction=pair[0],sl_fraction=pair[1])
            expected=np.array([[f[k] for k in ['p_tp_first','p_sl_first','p_no_touch']] for f in public['forecasts']]);error=float(abs(expected-new[j,g]).max());parity=max(parity,error)
            if error>2e-5 or public['support_flags']!=support_flags(pair,protocol['geometry_support']):raise ValueError('Public predictor parity/support mismatch')
    if any(abs(r['total_net_pnl']-r['realized_pnl']-r['unrealized_net_pnl'])>1e-8 or r['cash']< -1e-8 for r in rows):raise ValueError('Account identity mismatch')
    full=np.empty((len(indices),16,5,3),np.float32)
    for pair in GRID:full[:,GRID.index(pair)]=reused[:,existing.index(pair)] if pair in existing else new[:,missing.index(pair)]
    for pair in existing:
        if not np.array_equal(full[:,GRID.index(pair)],reused[:,existing.index(pair)]):raise ValueError('Prior predictions not bitwise preserved')
    np.savez_compressed(root/'test-predictions-16pairs.npz',origin_index=indices,geometry=np.array(GRID),probability=full)
    write_json(root/'coverage.json',dict(**coverage,test_origins=len(indices),reused_pairs_bitwise=9,new_pairs=7,unsupported_rows=sum(not r['geometry_supported'] for r in rows),
        model_sha256=protocol['model_sha256'],dataset_sha256=protocol['dataset_sha256'],baseline_results_sha256=sha256(root/'results-baseline.json'),
        previous_selected_policy_sha256=protocol['previous_selected_policy_sha256'],all_seven_new_pairs_public_predictor_max_error=parity))
    print(json.dumps(dict(stage='baseline_complete',coverage=coverage)),flush=True)
    render_heatmaps(root,rows)
    write_json(root/'progress.json',dict(stage='complete',completed_scenarios=640,pid=os.getpid(),device=args.device))
    print(json.dumps(dict(stage='complete',rows=640,report=str((root/'report.md').resolve()))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',default='runs/survival-threshold-backtest-5m-20261023');parser.add_argument('--output',default='runs/survival-threshold-full-grid-20261023')
    parser.add_argument('--batch',type=int,default=16);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');run(parser.parse_args())
