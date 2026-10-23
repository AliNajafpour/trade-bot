"""Frozen probability-threshold portfolio replay of the native survival model, CPU only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.barrier_dense import IndexedBarrierDataset, batch_history
from wallex_lab.barrier_dataset import barrier_query, sha256, write_json, HORIZONS
from wallex_lab.barrier_survival import SurvivalPredictor, incidence
from wallex_lab.barrier_only import simulate
from wallex_lab.first_touch import FirstTouchBook

MINUTE=60_000_000_000
THRESHOLDS=[.50,.55,.60,.65,.70,.75,.80,.85]
GEOMETRY=[(.005,.005),(.005,.01),(.01,.005),(.01,.01),(.01,.02),(.02,.01),(.02,.02),(.02,.04),(.04,.02)]


def causal_batch(values,times,ends,context):
    """The feature window ends at the last completed candle, never at entry."""
    return batch_history(values,times,ends,context)


def threshold_candidates(origins,probability,threshold):
    if probability.shape!=(len(origins),3) or not np.isfinite(probability).all() or (probability<0).any() or not np.allclose(probability.sum(-1),1,atol=2e-5):
        raise ValueError('Finite normalized TP/SL/no-touch probabilities required')
    # Simulator's positive score field orders simultaneous signals; it is not an EV claim.
    return [dict(row,original_expected_net=float(p[0]),p_tp_first=float(p[0]),p_sl_first=float(p[1]),p_no_touch=float(p[2]))
        for row,p in zip(origins,probability) if p[0]>threshold]


def common_interval(books,start,end):
    """Longest common gap-free interval, chosen using candle timestamps only."""
    left=pd.Timestamp(start).value;right=pd.Timestamp(end).value
    clock=np.arange(left,right,MINUTE,dtype=np.int64);present=np.ones(len(clock),bool)
    for b in books.values():present&=np.isin(clock,b.ns)
    edge=np.diff(np.r_[False,present,False].astype(np.int8));starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
    if not len(starts):raise ValueError('No common observed minute interval')
    longest=int(np.argmax(ends-starts))
    return pd.Timestamp(clock[starts[longest]],tz='UTC'),pd.Timestamp(left+int(ends[longest])*MINUTE,tz='UTC')


def make_origins(data,indices,geometry):
    rows=[];tp_fraction,sl_fraction=geometry
    for i in indices:
        decision=int(data.arrays['time_ns'][i]);ref=float(data.arrays['reference'][i]);asset=int(data.arrays['symbol'][i])
        rows.append(dict(origin_id=f'{asset}-{decision}',symbol=data.metadata['symbols'][asset],decision_at=pd.Timestamp(decision,tz='UTC').isoformat(),
            entry_at=pd.Timestamp(decision+MINUTE,tz='UTC').isoformat(),target_price=ref*(1+tp_fraction),stop_price=ref*(1-sl_fraction),
            stop_fraction=sl_fraction,target_fraction=tp_fraction,minimum_net_rr=0.,exit_mode='survival_fixed_levels',
            forecast_entry_price=ref,history_last_open_ns=decision-MINUTE))
    return rows


def predict_grid(data,model,indices,batch_size,output):
    prediction=np.empty((len(indices),len(GEOMETRY),len(HORIZONS),3),np.float32);began=time.perf_counter()
    for start in range(0,len(indices),batch_size):
        chosen=indices[start:start+batch_size];assets=np.asarray(data.arrays['symbol'][chosen]);ends=np.asarray(data.arrays['bar_index'][chosen])
        x=np.empty((len(chosen),model.metadata['context'],10),np.float32);sigma=np.empty(len(chosen));reference=np.empty(len(chosen))
        for asset in np.unique(assets):
            mask=np.flatnonzero(assets==asset);values,times=data.series[asset]
            x[mask],sigma[mask],reference[mask]=causal_batch(values,times,ends[mask],model.metadata['context'])
            assert np.array_equal(times[ends[mask]]+MINUTE,np.asarray(data.arrays['time_ns'][chosen[mask]]))
        q=np.stack([barrier_query(ref,ref*(1+np.array(GEOMETRY)[:,0]),ref*(1-np.array(GEOMETRY)[:,1]),s,ref)
            for s,ref in zip(sigma,reference)])
        with torch.inference_mode():
            x_tensor=torch.from_numpy(x);q_tensor=torch.from_numpy(q);symbols=torch.as_tensor(assets.astype(np.int64))
            probabilities=torch.stack([incidence(m(x_tensor,q_tensor,symbols)/model.metadata['temperature']) for m in model.models]).mean(0).numpy()
        prediction[start:start+len(chosen)]=probabilities
        if start==0 or (start//batch_size)%10==0:
            progress=dict(stage='predict',done=start+len(chosen),total=len(indices),seconds=time.perf_counter()-began)
            write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.savez_compressed(output/'compact-grid-predictions.npz',origin_index=indices,probability=prediction)
    return prediction


def replay(data,indices,probability,books,start,end,geometry,horizon,threshold,capital,slip):
    rows=threshold_candidates(make_origins(data,indices,GEOMETRY[geometry]),probability[:,geometry,horizon],threshold)
    result=simulate(rows,books,start,end,stop_spec='candidate',reward_multiple=1,initial_cash=capital,fee_bps=13.,slippage_bps=slip)
    result['summary'].update(probability_threshold=threshold,geometry_index=geometry,tp_fraction=GEOMETRY[geometry][0],sl_fraction=GEOMETRY[geometry][1],
        forecast_horizon_minutes=int(HORIZONS[horizon]),threshold_signals=len(rows))
    return result


def run(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve prior threshold backtest')
    output.mkdir(parents=True);torch.set_num_threads(1)
    data=IndexedBarrierDataset(args.data);model=SurvivalPredictor.load(args.model)
    if model.metadata['feature_version']!='relative-v2' or sha256(Path(args.data)/'dataset.json')!=model.metadata['dataset_manifest_sha256']:
        raise ValueError('Frozen model and dataset mismatch')
    seed=json.loads(Path(args.capital_seed).read_text());capital=float(seed['initial_usdt'])
    mask_path=Path('runs/barrier-survival-multiday-20261022/common-origin-mask.npy');eligible=np.load(mask_path)
    if not np.array_equal(eligible,data.eligible_history(model.metadata['context'])):raise ValueError('Common history eligibility changed')
    books={}
    for asset,symbol in enumerate(data.metadata['symbols']):
        values,times=data.series[asset]
        frame=pd.DataFrame(np.asarray(values),columns=['open','high','low','close','volume']);frame.insert(0,'timestamp',pd.to_datetime(times,utc=True));frame.attrs['interval_seconds']=60
        books[symbol]=FirstTouchBook(frame)
    bounds=model.metadata['fold']['boundaries']
    ranges=dict(selection=[pd.Timestamp(bounds[1]),pd.Timestamp(bounds[2])],test=[pd.Timestamp(bounds[3]),pd.Timestamp(bounds[4])])
    intervals={name:common_interval(books,*pair) for name,pair in ranges.items()}
    clocks=np.asarray(data.arrays['time_ns']);use=np.zeros(len(clocks),bool)
    for start,end in ranges.values():use|=(clocks>=start.value)&(clocks<end.value-1440*MINUTE)
    # Fixed cadence is independent of labels and never backfills missing history.
    indices=np.flatnonzero(use&eligible&(clocks%(args.stride_minutes*MINUTE)==0))
    indices=indices[np.lexsort((np.asarray(data.arrays['symbol'][indices]),clocks[indices]))]
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),model_path=str(Path(args.model).resolve()),model_sha256=sha256(Path(args.model)/'model.json'),
        model_hashes=[m['sha256'] for m in model.metadata['members']],dataset_sha256=sha256(Path(args.data)/'dataset.json'),
        capital_seed_sha256=sha256(args.capital_seed),initial_usdt=capital,fee_bps_per_side=13.,baseline_execution_friction_bps_per_side=5.,stress_friction_bps_per_side=[0.,15.],
        geometry_tp_sl_fractions=GEOMETRY,horizons=list(map(int,HORIZONS)),thresholds=THRESHOLDS,threshold_semantics='Strict absolute P(TP before SL by horizon)>p. No-touch is not a win.',
        forecast_anchor='Last completed close, fixed absolute TP/SL. M1 OPEN one minute after decision is execution proxy; execution repricing checks entry is inside fixed levels and target reward covers costs.',
        cadence_minutes=args.stride_minutes,eligibility='10080 contiguous completed M1 history; last24h of each role purged from entry decisions. No future outcome eligibility filter.',
        ranges={k:[v.isoformat() for v in pair] for k,pair in ranges.items()},primary_intervals={k:[v.isoformat() for v in pair] for k,pair in intervals.items()},
        interval_selection='Longest common gap-free candle interval in each role, timestamps only before inference or outcomes. Full test separately censored when path unknown.',
        account='Cash-funded long spot only, one position per asset, max15% allocation, .5% equity risk, $1 min notional, shared cash. Existing barrier_only simulator. Sizing equity is cash plus open quantity times entry reference price, not current marked equity. Four assets make native cap5 equivalent to cap4 here. Fractional quantities are paper proxies.',
        live_account_config_sha256=sha256('runs/account-sized-wallex-20261023/source-config.json'),
        entry_latency='One minute from completed-candle decision to next native M1 OPEN proxy. Probability horizon begins at decision; execution starts one minute later. No future entry price enters model features or queries.',
        live_policy_differences='Historical threshold-only sweep has no no-touch cap, conditional net-margin or minimum netRR gate. Live operational paper policy initially also requires no-touch<=.25, positive conditional net margin, netRR>=1, supported entry gap and real fresh bid/ask/depth. Capital risk limits agree; signal eligibility differs.',
        selection='On selection primary interval only, maximize marked return%-0.5*minute drawdown%; require >=5 closed trades and valid accounting, else retain declared first geometry/horizon/threshold and mark insufficient. Stable grid tie.',
        no_expiry='Forecast horizon does not liquidate. TP/SL exits only; end marks include hypothetical liquidation costs and are separate from realized P&L.',
        execution_limit='Historical bid/ask and depth unavailable. Assumed per-side spread/slippage combined friction; price-touch signal replay, not verified executable fills.',
        calibration_limit='Native temperature fitted on separate calibration dates before test, later than selection dates; selection results are development diagnostics, not out-of-time calibration validation.',
        historical_status='Repeated historical project dates. Temporal test for this strategy protocol, not project-wide untouched or prospective evidence.',
        sources=data.metadata['sources'],source_code_sha256={p:sha256(p) for p in [__file__,'wallex_lab/barrier_only.py','wallex_lab/barrier_dense.py','wallex_lab/barrier_survival.py']})
    write_json(output/'protocol.json',protocol);print(json.dumps(dict(stage='protocol_frozen',origins=len(indices),intervals=protocol['primary_intervals'])),flush=True)
    probability=predict_grid(data,model,indices,args.batch,output)
    select=(clocks[indices]>=ranges['selection'][0].value)&(clocks[indices]<ranges['selection'][1].value)
    selection_rows=[]
    for g in range(len(GEOMETRY)):
        for h in range(len(HORIZONS)):
            for p in THRESHOLDS:
                s=replay(data,indices[select],probability[select],books,*intervals['selection'],g,h,p,capital,5.)['summary']
                selection_rows.append(s)
        print(json.dumps(dict(stage='selection',geometry=g)),flush=True)
    pool=[s for s in selection_rows if s['valid_for_comparison'] and s['closed_trades']>=5]
    chosen=max(pool,key=lambda s:s['return_pct']-.5*s['max_drawdown_pct']) if pool else selection_rows[0]
    selected=dict(status='selection_qualified' if pool else 'insufficient_selection_trades',geometry_index=chosen['geometry_index'],
        horizon_index=list(HORIZONS).index(chosen['forecast_horizon_minutes']),probability_threshold=chosen['probability_threshold'],selection_summary=chosen,test_used_for_selection=False)
    write_json(output/'selection-grid.json',selection_rows);write_json(output/'selected-policy.json',selected)
    write_json(output/'pre-test-selection-receipt.json',dict(created_at=datetime.now(timezone.utc).isoformat(),test_opened=False,
        selected_policy_sha256=sha256(output/'selected-policy.json'),protocol_sha256=sha256(output/'protocol.json')))
    print(json.dumps(dict(stage='policy_frozen',selected=selected)),flush=True)
    test=(clocks[indices]>=ranges['test'][0].value)&(clocks[indices]<ranges['test'][1].value);results=[]
    (output/'ledgers').mkdir();curves={}
    for scope,interval in [('primary',intervals['test']),('full_test_censor_diagnostic',ranges['test'])]:
        for p in THRESHOLDS:
            for slip in [0.,5.,15.]:
                detail=replay(data,indices[test],probability[test],books,*interval,selected['geometry_index'],selected['horizon_index'],p,capital,slip)
                s=dict(scope=scope,**detail['summary']);results.append(s);key=f'{scope}-p{p:.2f}-friction{slip:g}'
                write_json(output/'ledgers'/(key+'.json'),{k:v for k,v in detail.items() if k not in ['equity_clock_ns','equity','equity_known']})
                if scope=='primary' and slip==5.:
                    curves[p]=(detail['equity_clock_ns'],detail['equity'],detail['equity_known'])
                    np.savez_compressed(output/(key+'-equity.npz'),clock_ns=detail['equity_clock_ns'],equity=detail['equity'],known=detail['equity_known'])
    flat=pd.DataFrame([{k:v for k,v in s.items() if not isinstance(v,dict)} for s in results]);flat.to_csv(output/'threshold-curve.csv',index=False)
    primary=[s for s in results if s['scope']=='primary' and s['slippage_bps_per_side']==5.]
    report=dict(protocol=protocol,selection=selected,test_curve=results,selected_primary=next(s for s in primary if s['probability_threshold']==selected['probability_threshold']))
    write_json(output/'report.json',report)
    lines=['# Survival probability threshold backtest','',f"Selection status: {selected['status']}. Frozen TP {GEOMETRY[selected['geometry_index']][0]:.2%}, SL {GEOMETRY[selected['geometry_index']][1]:.2%}, horizon {HORIZONS[selected['horizon_index']]} minutes, p>{selected['probability_threshold']:.2f}.",
        '',f"Initial {capital:.8f} USDT. Fee13bp and assumed spread/slippage5bp each side. Primary {intervals['test'][0].isoformat()} to {intervals['test'][1].isoformat()}.",
        '', '|p strict >|Entries|Closed|Wins|Losses|Open|Realized USDT|Unrealized USDT|Marked net USDT|DD %|Fees USDT|','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'unknown' if v is None else f'{v:.6f}'
    for s in primary:lines.append('|'+ '|'.join([f"{s['probability_threshold']:.2f}",*[str(s[k]) for k in ['entries','closed_trades','wins','losses','open_positions']],*[fmt(s[k]) for k in ['realized_pnl','unrealized_net_pnl','total_net_pnl','max_drawdown_pct','fees_paid']]])+'|')
    lines+=['',protocol['threshold_semantics'],protocol['no_expiry'],protocol['execution_limit'],protocol['historical_status'],protocol['account'],protocol['entry_latency'],protocol['live_policy_differences'],
        'Threshold curve is descriptive and predeclared. Test profit never selects p, geometry, or horizon. Full-period gap-censor diagnostics and 0/15bp friction scenarios are in report.json and threshold-curve.csv.',
        'Simultaneous signals rank by absolute TP probability, then symbol. Net target reward must cover actual assumed entry/exit costs. A high TP probability can still have poor economics when the stop is wider.',
        'All model probabilities are inferred once for every fixed geometry/horizon, shared across every threshold. No independent-label profit summation.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].bar([str(s['probability_threshold']) for s in primary],[s['total_net_pnl'] or 0 for s in primary],color='#276b78')
        axes[0].axhline(0,color='black',linewidth=.7);axes[0].set(xlabel='Strict absolute TP probability threshold',ylabel='Marked net P&L, USDT',title='Frozen geometry and horizon, temporal test')
        for p,(clock,equity,known) in curves.items():
            if p==selected['probability_threshold'] or p in [.5,.7,.85]:axes[1].plot(pd.to_datetime(clock[known][::30],utc=True),equity[known][::30],label=f'p>{p:.2f}')
        axes[1].set(ylabel='Net liquidation equity, USDT',title='Shared cash, TP/SL exits only');axes[1].legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(output/'threshold-backtest.png',dpi=160);plt.close(fig)
    except ImportError:pass
    print(json.dumps(dict(stage='complete',report=str((output/'report.json').resolve()),selected_primary=report['selected_primary'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',default='data/barrier-dense-20261022')
    parser.add_argument('--model',default='runs/barrier-survival-multiday-20261022/models/late');parser.add_argument('--capital-seed',default='runs/account-sized-wallex-20261023/capital-seed.json')
    parser.add_argument('--output',default='runs/survival-threshold-backtest-5m-20261023');parser.add_argument('--stride-minutes',type=int,default=5);parser.add_argument('--batch',type=int,default=16)
    args=parser.parse_args()
    if args.stride_minutes<5 or args.stride_minutes%5 or args.batch<1:parser.error('Positive batch and cadence divisible by five required')
    run(args)
