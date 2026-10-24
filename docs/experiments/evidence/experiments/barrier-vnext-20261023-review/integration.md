# Whole vNext integration added-file package, no Git repository

## wallex_lab/barrier_vnext_data.py
```python
"""Expanded-venue indexed data; no duplicated windows or stored geometry tensor."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from .barrier_dataset import BarrierLabelBook, HORIZONS, sha256
from .barrier_dense import IndexedBarrierDataset, batch_history, query_distances, query_features, SIDE, OFFSET
from .barrier_full_cache import _atomic_json

MINUTE=60_000_000_000
FIELDS=('time_ns','symbol','bar_index','sigma','reference','event_bin','cause','survived_bins','event_minute','observed_minutes')


def next_folds(start,end):
    start=pd.Timestamp(start);end=pd.Timestamp(end).floor('D')
    if end-start<pd.Timedelta(days=200):raise ValueError('At least200days required for production temporal folds')
    return [dict(name=name,boundaries=[str(t) for t in (start,stop-pd.Timedelta(days=90),stop-pd.Timedelta(days=60),stop-pd.Timedelta(days=30),stop)])
        for name,stop in (('early',end-pd.Timedelta(days=60)),('late',end))]


def _indices(values,maximum):
    values=np.asarray(values)
    if values.ndim!=1 or not len(values) or values.dtype.kind not in 'iu' or (values<0).any() or (values>=maximum).any():
        raise ValueError('Nonempty in-range integer indices required')
    return values.astype(np.int64,copy=False)


def _source_series(source,verify=True):
    source=Path(source);manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='ready':raise ValueError('Acquisition is not ready')
    series=[];records=[]
    for symbol,record in sorted(manifest['symbols'].items()):
        paths=[Path(record[name+'_path']) for name in ('values','times')]
        paths=[p if p.is_absolute() else source/p for p in paths]
        if verify:
            for name,p in zip(('values','times'),paths):
                if sha256(p)!=record[name+'_sha256']:raise ValueError('Changed source: '+str(p))
        values,times=[np.load(p,mmap_mode='r',allow_pickle=False) for p in paths]
        if values.shape!=(len(times),9) or len(times)<720 or times.dtype!=np.int64 or (np.diff(times)<=0).any() or (times%MINUTE).any():
            raise ValueError('Invalid source array shapes or clocks')
        series.append((values,times));records.append(dict(symbol=symbol,values_path=str(paths[0].resolve()),times_path=str(paths[1].resolve()),
            values_sha256=record['values_sha256'],times_sha256=record['times_sha256'],rows=len(times)))
    if not series:raise ValueError('No completed source markets')
    return manifest,series,records


def build_dataset(source,output,folds=None):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source_hash=sha256(source/'manifest.json')
    identity=dict(source_manifest_sha256=source_hash,folds=folds,version='barrier-vnext-indexed-1',producer_sha256=sha256(__file__))
    identity_path=output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Dataset source/config changed')
    if (output/'dataset.json').exists():
        metadata=json.loads((output/'dataset.json').read_text())
        for name,digest in metadata['files'].items():
            if sha256(output/name)!=digest:raise ValueError('Dataset file changed: '+name)
        return metadata
    _atomic_json(identity_path,identity)
    manifest,series,records=_source_series(source)
    assets=[];ends=[];decisions=[]
    for asset,(_,times) in enumerate(series):
        index=np.arange(719,len(times));gaps=np.r_[0,np.cumsum(np.diff(times)!=MINUTE)]
        keep=((times[index]+MINUTE)%(5*MINUTE)==0)&(gaps[index]==gaps[index-719])
        index=index[keep];assets.append(np.full(len(index),asset,np.int16));ends.append(index);decisions.append(times[index]+MINUTE)
    assets=np.concatenate(assets);ends=np.concatenate(ends);decisions=np.concatenate(decisions);n=len(ends)
    if not n:raise ValueError('No eligible completed origins')
    shapes={'time_ns':((n,),np.int64),'symbol':((n,),np.int16),'bar_index':((n,),np.int64),'sigma':((n,),np.float64),
        'reference':((n,),np.float64),'observed_minutes':((n,),np.int16)}
    shapes.update({k:((n,128),np.int16 if k=='event_minute' else np.int8) for k in ('event_bin','cause','survived_bins','event_minute')})
    required=sum(int(np.prod(shape))*np.dtype(dtype).itemsize for k,(shape,dtype) in shapes.items() if not (output/(k+'.npy')).exists())
    if shutil.disk_usage(output).free-required<5*2**30:raise OSError('Insufficient disk reserve for indexed labels')
    arrays={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='r+' if (output/(k+'.npy')).exists() else 'w+',dtype=dtype,shape=shape)
        for k,(shape,dtype) in shapes.items()}
    progress_path=output/'progress.json';cursor=json.loads(progress_path.read_text()).get('cursor',0) if progress_path.exists() else 0
    arrays['symbol'][:]=assets;arrays['bar_index'][:]=ends;arrays['time_ns'][:]=decisions
    active=None;book=None;vol=None
    for begin in range(cursor,n,256):
        stop=min(n,begin+256)
        for row in range(begin,stop):
            asset=int(assets[row]);at=int(ends[row]);values,times=series[asset]
            if active!=asset:
                frame=pd.DataFrame(np.asarray(values[:,:5]),columns=['open','high','low','close','volume']);frame['timestamp']=pd.to_datetime(times,utc=True)
                frame.attrs['interval_seconds']=60;book=BarrierLabelBook(frame)
                returns=pd.Series(np.r_[0.,np.diff(np.log(values[:,3]))])
                vol=np.maximum(np.maximum(returns.rolling(60).std(ddof=0),returns.rolling(360).std(ddof=0)),1e-5).to_numpy();active=asset
                del frame
            sigma=float(vol[at]);reference=float(values[at,3]);distance=query_distances(sigma,int(decisions[row]//MINUTE)+asset*10_000_000)
            entries=reference*np.exp(sigma*OFFSET)
            for orientation,side in enumerate(('long','short')):
                part=slice(orientation*64,(orientation+1)*64);sign=1 if side=='long' else -1
                labels=book.label(at,entries[part],entries[part]*np.exp(sign*distance[:,0].astype(float)),entries[part]*np.exp(-sign*distance[:,1].astype(float)),side)
                for field in ('event_bin','cause','survived_bins','event_minute'):arrays[field][row,part]=labels[field]
            arrays['sigma'][row]=sigma;arrays['reference'][row]=reference;arrays['observed_minutes'][row]=labels['observed_minutes']
        for value in arrays.values():value.flush()
        _atomic_json(progress_path,dict(stage='labeling',cursor=stop,origins=n))
        if begin%10240==0:print(f'VNEXT_LABELS {stop}/{n}',flush=True)
    counts=np.zeros(4,np.int64)
    for begin in range(0,n,8192):counts+=np.bincount(arrays['cause'][begin:begin+8192].ravel(),minlength=4)
    metadata=dict(status='ready',storage='vnext-indexed-1',feature_version='relative-v2-activity',venue=manifest.get('venue','binance-spot'),
        source_root=str(source),source_manifest_sha256=source_hash,symbols=[r['symbol'] for r in records],sources=records,origins=n,
        queries_per_origin=128,stride_minutes=5,context_minutes=720,horizons=HORIZONS.tolist(),channels=14,
        folds=folds or next_folds(pd.Timestamp(int(decisions.min()),tz='UTC'),pd.Timestamp(int(max(t[-1] for _,t in series)+MINUTE),tz='UTC')),
        cause_counts={name:int(counts[j]) for j,name in enumerate(('tp','sl','unknown','censored_or_no_touch'))},
        geometry_support=dict(min_fraction=float(np.expm1(.0001)),max_fraction=float(np.expm1(.15)),max_ratio=8.),
        query_policy='Existing ex-ante query_distances:64 per direction,128 per origin; generated deterministically, not duplicated on disk.',
        entry_policy='Completed close times predetermined sigma offset; native trade-price candles, not fills.',
        activity_channels=['log1p_trade_count','signed_taker_base_fraction','log1p_observed_inactive_run','flat_range'],
        files={p.name:sha256(p) for p in output.glob('*.npy')})
    _atomic_json(output/'dataset.json',metadata);_atomic_json(progress_path,dict(stage='ready',cursor=n,origins=n))
    return metadata


class NextBarrierDataset(IndexedBarrierDataset):
    def __init__(self,root,verify=True):
        self.root=Path(root);self.metadata=json.loads((self.root/'dataset.json').read_text())
        if self.metadata['status']!='ready':raise ValueError('Incomplete dataset')
        if verify:
            for name,digest in self.metadata['files'].items():
                if sha256(self.root/name)!=digest:raise ValueError('Changed dataset: '+name)
        self.arrays={k:np.load(self.root/(k+'.npy'),mmap_mode='r') for k in FIELDS}
        self.series=[(np.load(r['values_path'],mmap_mode='r'),np.load(r['times_path'],mmap_mode='r')) for r in self.metadata['sources']]
        if verify:
            for r in self.metadata['sources']:
                for name in ('values','times'):
                    if sha256(r[name+'_path'])!=r[name+'_sha256']:raise ValueError('Changed native source')
        self.context=720;self.resident=None

    def close(self):
        for value in list(self.arrays.values())+[a for pair in self.series for a in pair]:
            if isinstance(value,np.memmap):value._mmap.close()

    def batch(self,indices,queries=None,device='cpu',with_history=True):
        indices=_indices(indices,len(self.arrays['time_ns']));queries=_indices(np.arange(128) if queries is None else queries,128)
        assets=self.arrays['symbol'][indices];sigmas=np.asarray(self.arrays['sigma'][indices]);times=self.arrays['time_ns'][indices]
        distances=np.stack([np.tile(query_distances(float(s),int(t//MINUTE)+int(a)*10_000_000),(2,1))[queries] for s,t,a in zip(sigmas,times,assets)])
        q=query_features(distances,sigmas,SIDE[queries],OFFSET[queries])
        x=np.zeros((len(indices),self.context if with_history else 1,14),np.float32)
        if with_history:
            for asset in np.unique(assets):
                rows=np.flatnonzero(assets==asset);values,clocks=self.series[asset];ends=self.arrays['bar_index'][indices[rows]]
                x[rows,:,:10]=batch_history(values[:,:5],clocks,ends,self.context)[0]
                positions=ends[:,None]+np.arange(-self.context+1,1);raw=np.asarray(values[positions]);volume=raw[:,:,4]
                active=np.where(volume>0,np.arange(self.context)[None,:],-1)
                age=np.arange(self.context)[None,:]-np.maximum.accumulate(active,axis=1)
                signed=np.divide(2*raw[:,:,7],volume,out=np.ones_like(volume),where=volume>0)-1
                activity=np.stack([np.log1p(raw[:,:,6]),signed,np.log1p(age),(raw[:,:,1]==raw[:,:,2]).astype(float)],axis=-1)
                x[rows,:,10:]=np.clip(activity,-50,50).astype(np.float32)
        result=dict(x=torch.from_numpy(x),q=torch.from_numpy(q),symbol=torch.as_tensor(np.array(assets),dtype=torch.long))
        for field,key in (('event_bin','event'),('cause','cause'),('survived_bins','survive')):
            result[key]=torch.as_tensor(np.array(self.arrays[field][indices][:,queries]),dtype=torch.long)
        return {key:value.to(device) for key,value in result.items()}

```

## checks/check_barrier_vnext_data.py
```python
"""Synthetic checks for causal multi-year data and honest partial first-touch labels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from wallex_lab.barrier_dataset import sha256, write_json
from wallex_lab.barrier_dense import batch_history


def rejects(call):
    try: call()
    except ValueError: return
    raise AssertionError('Invalid index or changed source accepted')


def main():
    assert importlib.util.find_spec('wallex_lab.barrier_vnext_data'), 'vNext data implementation missing'
    from wallex_lab import barrier_vnext_data as module
    from wallex_lab.barrier_vnext_data import build_dataset, NextBarrierDataset
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'source';source.mkdir()
        n=11520;times=pd.date_range('2026-01-01',periods=n,freq='min',tz='UTC').as_unit('ns').asi8.copy()
        close=100*np.exp(np.sin(np.arange(n)/150)*.01)
        values=np.column_stack([close,close*1.0002,close*.9998,close,np.ones(n),close,np.full(n,4),np.full(n,.6),close*.6])
        # Preserve a real calendar gap and a broad same-minute double touch.
        values[10500,1:3]=[110,90]
        keep=np.arange(n)!=10800;values=values[keep];times=times[keep]
        np.save(source/'v.npy',values);np.save(source/'t.npy',times)
        manifest=dict(status='ready',venue='binance-spot',symbols={'BTCUSDT':dict(values_path=str(source/'v.npy'),times_path=str(source/'t.npy'),
            values_sha256=sha256(source/'v.npy'),times_sha256=sha256(source/'t.npy'),rows=len(times))})
        write_json(source/'manifest.json',manifest)
        folds=[dict(name='toy',boundaries=['2026-01-01','2026-01-03','2026-01-05','2026-01-07','2026-01-09'])]
        atomic=module._atomic_json;interrupted=[False]
        def commit_then_interrupt(path,value):
            atomic(path,value)
            if Path(path).name=='progress.json' and value.get('cursor')==256 and not interrupted[0]:
                interrupted[0]=True;raise InterruptedError('Committed label chunk interruption')
        module._atomic_json=commit_then_interrupt
        try:
            try:build_dataset(source,root/'dataset',folds=folds)
            except InterruptedError:pass
            else:raise AssertionError('Label build did not interrupt')
        finally:module._atomic_json=atomic
        meta=build_dataset(source,root/'dataset',folds=folds)
        assert meta==build_dataset(source,root/'fresh-dataset',folds=folds), 'Resumed labels differ from fresh full build'
        data=NextBarrierDataset(root/'dataset');data.context=720
        i=int(np.flatnonzero(data.arrays['bar_index']==10499)[0]);out=data.batch([i])
        assert out['x'].shape==(1,720,14) and out['q'].shape==(1,128,8)
        expected=batch_history(values[:,:5],times,[10499],720)[0]
        np.testing.assert_allclose(out['x'].numpy()[:,:,:10],expected,rtol=1e-6,atol=1e-6)
        assert (out['cause']==2).any(), 'Both-hit order was invented'
        assert (np.asarray(data.arrays['observed_minutes'])<1440).any(), 'Gap/source-end censoring lost'
        assert data.eligible_history(10080).sum()<len(data.arrays['time_ns'])
        assert meta['queries_per_origin']==128 and meta['stride_minutes']==5
        assert 'event_minute' in data.arrays
        again=data.batch([i]);torch.testing.assert_close(out['q'],again['q'],rtol=0,atol=0)
        # Alter only future observations; historical features and ex-ante queries stay fixed.
        changed=values.copy();changed[10500:,:4]*=1.05
        data.series=[(changed,times)]
        after=data.batch([i]);torch.testing.assert_close(out['x'],after['x'],rtol=0,atol=0)
        torch.testing.assert_close(out['q'],after['q'],rtol=0,atol=0)
        scaled=values.copy();scaled[:,:4]*=1000;scaled[:,5]*=1000;scaled[:,8]*=1000
        data.series=[(scaled,times)]
        torch.testing.assert_close(out['x'],data.batch([i])['x'],rtol=1e-4,atol=1e-4)
        rejects(lambda:data.batch(np.array([2**64-1],dtype=np.uint64)))
        rejects(lambda:data.batch([i],queries=[128]))
        assert build_dataset(source,root/'dataset',folds=folds)==meta
        manifest['venue']='changed';write_json(source/'manifest.json',manifest)
        rejects(lambda:build_dataset(source,root/'dataset',folds=folds))
        data.close()
    print('PASS vNext causal features, deterministic128queries, ambiguity, censoring, scale invariance and source identity')


if __name__=='__main__':main()

```

## scripts/train_barrier_vnext.py
```python
"""Isolated, resumable native-venue ablations. No prediction arrays are persisted."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from wallex_lab.barrier_dataset import HORIZONS, fold_roles, sha256
from wallex_lab.barrier_full_cache import _atomic_json as atomic_json, _array_hash
from wallex_lab.barrier_survival import risk_nll, incidence, mixture_nll
from scripts.train_barrier_survival import horizon_targets
from scripts.train_full_barrier_comparison import run_lock, wait_lock

VARIANTS=('baseline','activity','age','level','local','pretrained','combined')
SEEDS=(42,43,44)
ROLES=('fit','selection','calibration','test')
TEMPERATURES=(.25,.5,.75,1.,1.25,1.5,2.,3.,4.,6.,8.)
BASE=Path(__file__).resolve().parents[1]
SOURCES=('scripts/train_barrier_vnext.py','wallex_lab/barrier_vnext_model.py','wallex_lab/barrier_vnext_data.py',
    'wallex_lab/barrier_survival.py','wallex_lab/barrier_dense.py','wallex_lab/barrier_dataset.py',
    'wallex_lab/barrier_full_cache.py','scripts/train_barrier_survival.py','scripts/train_full_barrier_comparison.py')


def save_checkpoint(path,state):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _data_identity(data):
    if hasattr(data,'root'):
        return dict(manifest_sha256=sha256(data.root/'dataset.json'),metadata=data.metadata)
    arrays={**data.arrays,**{k:v for k,v in vars(data).items() if isinstance(v,np.ndarray)}}
    return dict(synthetic=True,metadata=data.metadata,arrays={k:_array_hash(v) for k,v in arrays.items()})


def _indices(indices,data):
    value=np.asarray(indices)
    if value.ndim!=1 or not len(value) or value.dtype.kind not in 'iu' or (value<0).any() or (value>=len(data.arrays['time_ns'])).any() or (np.diff(value)<=0).any():
        raise ValueError('Nonempty sorted unique in-range cohort required')
    return value.astype(np.int64,copy=False)


def _config(data,variant,args):
    if variant not in VARIANTS:raise ValueError('Unknown ablation')
    return dict(symbols=len(data.metadata['symbols']),context=args.context,width=args.width,layers=args.layers,
        activity=variant in ('activity','combined'),age=variant in ('age','combined'),
        level_attention=variant in ('level','combined'),local=variant in ('local','combined'))


def _new_model(config):
    from wallex_lab.barrier_vnext_model import NextBarrierModel
    return NextBarrierModel(**config)


def _batch(data,rows,device):
    batch=data.batch(rows,device=device)
    if batch['q'].shape[1]!=128 or any(batch[k].shape!=(len(rows),128) for k in ('event','cause','survive')):
        raise ValueError('Every origin must carry all128queries')
    return batch


def _autocast(args):
    return torch.autocast('cuda',dtype=torch.bfloat16) if getattr(args,'precision','fp32')=='bf16' else nullcontext()


def _selection(model,data,indices,args):
    model.eval();total=0.;count=0
    with torch.inference_mode():
        for start in range(0,len(indices),getattr(args,'microbatch',args.batch)):
            rows=indices[start:start+getattr(args,'microbatch',args.batch)];b=_batch(data,rows,args.device)
            with _autocast(args):z=model(b['x'],b['q'],b['symbol'])
            loss=risk_nll(z.float(),*[b[k] for k in ('event','cause','survive')])
            total+=float(loss)*len(rows);count+=len(rows)
    if not np.isfinite(total):raise RuntimeError('Nonfinite selection loss')
    return total/count


def _verify_member(root,summary):
    if json.loads((root/'identity.json').read_text())!=summary['identity'] or summary['config']!=summary['identity']['config']:
        raise ValueError('Member identity/config metadata changed')
    for name,key in (('best.pt','checkpoint_sha256'),('last.pt','last_sha256')):
        if sha256(root/name)!=summary[key]:raise ValueError('Member artifact changed: '+str(root/name))


def train_member(data,fit,selection,path,variant,seed,args):
    """Checkpoint committed effective batches, including partial-epoch order and RNG."""
    from wallex_lab.barrier_vnext_model import historical_mask
    fit=_indices(fit,data);selection=_indices(selection,data)
    if np.intersect1d(fit,selection).size:raise ValueError('Fit/selection overlap')
    if data.context!=args.context:raise ValueError('Dataset/model context differs')
    root=Path(path);root.mkdir(parents=True,exist_ok=True);config=_config(data,variant,args)
    settings={key:getattr(args,key,default) for key,default in (('batch',64),('microbatch',args.batch),('epochs',12),
        ('patience',3),('pretrain_epochs',2),('checkpoint_batches',200),('device','cpu'),('precision','fp32'))}
    if any(settings[k]<1 for k in ('batch','microbatch','epochs','patience','checkpoint_batches')) or settings['pretrain_epochs']<0:
        raise ValueError('Invalid training budget')
    identity=dict(data=_data_identity(data),fit_sha256=_array_hash(fit),selection_sha256=_array_hash(selection),
        config=config,variant=variant,seed=int(seed),settings=settings,sources={p:sha256(BASE/p) for p in SOURCES},
        versions=dict(torch=torch.__version__,numpy=np.__version__),optimizer=dict(lr=.0006,weight_decay=.03,clip=1.))
    identity_path=root/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Member source/config/cohorts changed')
    atomic_json(identity_path,identity)
    if (root/'fit.json').exists():
        result=json.loads((root/'fit.json').read_text());_verify_member(root,result);return result
    torch.manual_seed(seed)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=_new_model(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.03)
    order_rng=np.random.default_rng(seed);mask_rng=torch.Generator(device='cpu').manual_seed(seed+1000)
    pretrain_epochs=args.pretrain_epochs if variant in ('pretrained','combined') else 0
    state=dict(identity=identity,stage='pretrain' if pretrain_epochs else 'fit',epoch=0,cursor=0,order=None,
        total=0.,seen=0,steps=0,curve=[],pretrain_curve=[],best=None,best_state=None,selected_epoch=None,stale=0,elapsed=0.)
    if (root/'last.pt').exists():
        state=torch.load(root/'last.pt',map_location=device,weights_only=False)
        if state['identity']!=identity:raise ValueError('Checkpoint identity changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        order_rng.bit_generator.state=state['order_rng'];mask_rng.set_state(state['mask_rng'].cpu())
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    started=time.perf_counter();prior=state['elapsed']

    def commit():
        # Atomic replacement needs room for the new checkpoint before removing its predecessor.
        reserve=5*2**30+max(128*2**20,(root/'last.pt').stat().st_size if (root/'last.pt').exists() else 0)
        if hasattr(data,'root') and shutil.disk_usage(root).free<reserve:raise OSError('Insufficient checkpoint space above5GiB reserve; free disk and resume')
        state.update(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            order_rng=order_rng.bit_generator.state,mask_rng=mask_rng.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,elapsed=prior+time.perf_counter()-started)
        save_checkpoint(root/'last.pt',state)

    if not (root/'last.pt').exists():commit()
    while state['stage']!='complete':
        pretraining=state['stage']=='pretrain';limit=pretrain_epochs if pretraining else args.epochs
        if state['epoch']>=limit or (not pretraining and state['stale']>=args.patience):
            state.update(stage='fit' if pretraining else 'complete',epoch=0,cursor=0,order=None,total=0.,seen=0)
            # Use a fresh downstream optimizer, preserving the pretrained encoder weights.
            if pretraining:optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.03)
            commit();continue
        if state['order'] is None:state['order']=order_rng.permutation(fit)
        model.train()
        for start in range(state['cursor'],len(fit),args.batch):
            rows=state['order'][start:start+args.batch];optimizer.zero_grad(set_to_none=True)
            mask=historical_mask(len(rows),args.context,mask_rng) if pretraining else None
            for offset in range(0,len(rows),settings['microbatch']):
                part=rows[offset:offset+settings['microbatch']];b=_batch(data,part,args.device)
                with _autocast(args):
                    if pretraining:loss=model.pretrain_loss(b['x'],b['symbol'],mask[offset:offset+len(part)].to(device))
                    else:z=model(b['x'],b['q'],b['symbol'])
                if not pretraining:loss=risk_nll(z.float(),*[b[k] for k in ('event','cause','survive')])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                (loss*(len(part)/len(rows))).backward();state['total']+=float(loss.detach())*len(part);state['seen']+=len(part)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            state['cursor']=start+len(rows);state['steps']+=1
            if state['steps']%args.checkpoint_batches==0:commit()
            atomic_json(root/'progress.json',dict(stage=state['stage'],epoch=state['epoch']+1,cursor=state['cursor'],
                origins=len(fit),optimizer_steps=state['steps'],fit_origin_query_visits=0 if pretraining else state['seen']*128))
        if state['seen']!=len(fit):raise RuntimeError('Incomplete epoch coverage')
        if pretraining:state['pretrain_curve'].append(dict(epoch=state['epoch']+1,loss=state['total']/state['seen'],origins_seen=state['seen'],optimizer_steps=state['steps']))
        else:
            score=_selection(model,data,selection,args)
            row=dict(epoch=state['epoch']+1,fit_nll=state['total']/state['seen'],selection_nll=score,
                fit_origins_seen=state['seen'],fit_origin_query_visits=state['seen']*128,optimizer_steps=state['steps'])
            state['curve'].append(row)
            if state['best'] is None or score<state['best']-1e-5:
                state.update(best=score,selected_epoch=state['epoch']+1,stale=0,
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            else:state['stale']+=1
            print('VNEXT_EPOCH '+json.dumps(dict(variant=variant,seed=seed,**row)),flush=True)
        state.update(epoch=state['epoch']+1,cursor=0,order=None,total=0.,seen=0);commit()
    if state['best_state'] is None:raise RuntimeError('No selected model')
    save_checkpoint(root/'best.pt',state['best_state'])
    result=dict(variant=variant,seed=int(seed),config=config,identity=identity,selected_epoch=state['selected_epoch'],
        selection_nll=state['best'],curve=state['curve'],fit_origins=len(fit),selection_origins=len(selection),queries_per_origin=128,
        pretraining=dict(epochs=pretrain_epochs,origins_per_epoch=len(fit),fit_only=True,curve=state['pretrain_curve']),
        checkpoint_sha256=sha256(root/'best.pt'),last_sha256=sha256(root/'last.pt'),optimizer_steps=state['steps'],
        seconds=state['elapsed'],parameters=sum(p.numel() for p in model.parameters()))
    atomic_json(root/'fit.json',result);return result


def _load_members(data,paths,args):
    models=[];data_identity=_data_identity(data)
    for path in map(Path,paths):
        summary=json.loads((path/'fit.json').read_text());_verify_member(path,summary)
        if summary['identity']['data']!=data_identity:raise ValueError('Evaluation dataset differs')
        if summary['identity']['sources']!={p:sha256(BASE/p) for p in SOURCES}:raise ValueError('Evaluation source differs')
        model=_new_model(summary['config']);model.load_state_dict(torch.load(path/'best.pt',map_location='cpu',weights_only=True))
        models.append(model.to(args.device).eval())
    if not models:raise ValueError('Empty ensemble')
    return models


def _logits(models,b,args):
    with _autocast(args):return [model(b['x'],b['q'],b['symbol']).float() for model in models]


def calibrate_members(data,indices,paths,args):
    indices=_indices(indices,data);models=_load_members(data,paths,args);scores=np.zeros(len(TEMPERATURES),np.float64)
    with torch.inference_mode():
        for start in range(0,len(indices),getattr(args,'microbatch',args.batch)):
            rows=indices[start:start+getattr(args,'microbatch',args.batch)];b=_batch(data,rows,args.device);z=_logits(models,b,args)
            for j,t in enumerate(TEMPERATURES):scores[j]+=float(mixture_nll(z,*[b[k] for k in ('event','cause','survive')],t))*len(rows)
    if not np.isfinite(scores).all():raise RuntimeError('Nonfinite calibration scores')
    choice=int(np.argmin(scores));return dict(temperature=TEMPERATURES[choice],grid=list(TEMPERATURES),nll=(scores/len(indices)).tolist(),origins=len(indices))


def _bins(probability,actual):
    index=np.minimum((probability*10).astype(int),9)
    return np.stack([np.bincount(index,weights=weight,minlength=10) for weight in (np.ones(len(index)),probability,actual)],-1)


def _reliability(values):
    return [dict(low=j/10,count=int(v[0]),predicted=float(v[1]/v[0]) if v[0] else None,
        observed=float(v[2]/v[0]) if v[0] else None) for j,v in enumerate(values)]


def evaluate_members(data,indices,paths,args,temperature=1.):
    """Bounded streaming ensemble metrics and UTC-day sufficient statistics."""
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Positive temperature required')
    indices=_indices(indices,data);models=_load_members(data,paths,args)
    count=np.zeros(5,np.int64);loss=np.zeros(5);brier=np.zeros(5);freq=np.zeros((5,3),np.int64)
    reliability=np.zeros((5,3,10,3));touch_count=np.zeros(5,np.int64);touch_loss=np.zeros(5);touch_brier=np.zeros(5)
    touch_reliability=np.zeros((5,10,3));daily={};path_loss=0.;ambiguities=0
    with torch.inference_mode():
        for start in range(0,len(indices),getattr(args,'microbatch',args.batch)):
            rows=indices[start:start+getattr(args,'microbatch',args.batch)];b=_batch(data,rows,args.device);z=_logits(models,b,args)
            nll=float(mixture_nll(z,*[b[k] for k in ('event','cause','survive')],temperature));path_loss+=nll*len(rows)
            p=torch.stack([incidence(v/temperature) for v in z]).mean(0).cpu().numpy().astype(np.float64)
            if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1,atol=2e-6) or (np.diff(p[...,:2],axis=-2)<-2e-6).any() or (np.diff(p[...,2],axis=-1)>2e-6).any():
                raise ValueError('Incoherent ensemble incidences')
            event,cause,survive=[b[k].cpu().numpy() for k in ('event','cause','survive')];y,mask=horizon_targets(event,cause,survive)
            ambiguities+=int((cause==2).sum());days=data.arrays['time_ns'][rows]//86_400_000_000_000
            per_loss=-np.log(np.maximum(np.take_along_axis(p,np.maximum(y,0)[...,None],-1)[...,0],1e-12))
            per_brier=((p-np.eye(3)[np.maximum(y,0)])**2).sum(-1)
            for day in np.unique(days):
                key=str(pd.Timestamp(int(day)*86_400_000_000_000,tz='UTC').date())
                sums=daily.setdefault(key,dict(log_loss_sum=np.zeros(5),brier_sum=np.zeros(5),scored=np.zeros(5,np.int64)))
                valid=mask[days==day];sums['log_loss_sum']+=(per_loss[days==day]*valid).sum((0,1))
                sums['brier_sum']+=(per_brier[days==day]*valid).sum((0,1));sums['scored']+=valid.sum((0,1))
            for h in range(5):
                valid=mask[...,h];actual=y[...,h][valid];forecast=p[...,h,:][valid]
                count[h]+=len(actual);loss[h]+=per_loss[...,h][valid].sum();brier[h]+=per_brier[...,h][valid].sum()
                freq[h]+=np.bincount(actual,minlength=3)
                for cls in range(3):reliability[h,cls]+=_bins(forecast[:,cls],(actual==cls).astype(float))
                touched=(actual==0)|(actual==1);q=forecast[touched,0]/np.maximum(forecast[touched,:2].sum(-1),1e-12)
                target=(actual[touched]==0).astype(float);touch_count[h]+=len(target)
                touch_loss[h]+=(-(target*np.log(np.maximum(q,1e-12))+(1-target)*np.log(np.maximum(1-q,1e-12)))).sum()
                touch_brier[h]+=((q-target)**2).sum();touch_reliability[h]+=_bins(q,target)
    def mean(total,n):return float(total/n) if n else None
    return dict(partial_label_nll=path_loss/len(indices),origins=len(indices),origin_query_visits=len(indices)*128,
        temperature=float(temperature),coherence_checked=True,exact_cause_ambiguities=ambiguities,
        by_horizon=[dict(horizon_minutes=int(h),scored=int(count[j]),total=len(indices)*128,log_loss=mean(loss[j],count[j]),
            brier=mean(brier[j],count[j]),class_frequencies=(freq[j]/count[j]).tolist() if count[j] else None,
            reliability={name:_reliability(reliability[j,c]) for c,name in enumerate(('tp','sl','no_touch'))}) for j,h in enumerate(HORIZONS)],
        conditional_tp_given_touch=[dict(horizon_minutes=int(h),scored_touches=int(touch_count[j]),log_loss=mean(touch_loss[j],touch_count[j]),
            brier=mean(touch_brier[j],touch_count[j]),reliability=_reliability(touch_reliability[j])) for j,h in enumerate(HORIZONS)],
        daily={day:{key:value.tolist() for key,value in sums.items()} for day,sums in sorted(daily.items())})


def paired_daily_bootstrap(result,baseline,repetitions=500):
    days=sorted(result['daily'])
    if days!=sorted(baseline['daily']):raise ValueError('Paired days differ')
    a=np.asarray([result['daily'][d]['log_loss_sum'] for d in days]);b=np.asarray([baseline['daily'][d]['log_loss_sum'] for d in days])
    counts=np.asarray([result['daily'][d]['scored'] for d in days])
    if not np.array_equal(counts,[baseline['daily'][d]['scored'] for d in days]):raise ValueError('Paired counts differ')
    rng=np.random.default_rng(20261023);draws=[]
    for _ in range(repetitions):
        chosen=rng.integers(len(days),size=len(days));denom=counts[chosen].sum(0)
        draws.append(np.divide((a[chosen]-b[chosen]).sum(0),denom,out=np.full(5,np.nan),where=denom>0))
    available=counts.sum(0)>0;lower=[None]*5;upper=[None]*5;means=[None]*5
    for j in np.flatnonzero(available):
        lower[j]=float(np.nanquantile(np.asarray(draws)[:,j],.025));upper[j]=float(np.nanquantile(np.asarray(draws)[:,j],.975))
        means[j]=float((a[:,j]-b[:,j]).sum()/counts[:,j].sum())
    return dict(days=len(days),repetitions=repetitions,unit='Full UTC calendar days across assets and128queries',
        mean_difference=means,lower_95=lower,upper_95=upper,limitation='Descriptive intervals; adjacent days may remain dependent.')


def progress(root,stage,**kwargs):
    value=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
    atomic_json(root/'progress.json',value);print('VNEXT '+json.dumps(value),flush=True)


def wait_comparison(root,comparison):
    # Do not inspect CUDA free memory here: even that initializes a CUDA context.
    while not (Path(comparison)/'results.json').exists():
        progress(root,'waiting_for_comparison',comparison=str(comparison),gpu_not_claimed=True);time.sleep(30)


def wait_source(root,source):
    source=Path(source)
    while True:
        manifest=source/'manifest.json'
        value=json.loads(manifest.read_text()) if manifest.exists() else {}
        if value.get('status')=='ready':return value
        if value.get('status') in ('partial','failed','interrupted'):
            message=f"Source status {value['status']}; inspect {source/'manifest.json'} and acquisition receipts, then resume acquisition before restarting this queue."
            progress(root,'source_blocked',source=str(source),reason=message,gpu_not_claimed=True)
            raise ValueError(message)
        progress(root,'acquisition_wait',source=str(source),source_status=value.get('status','not_started'),gpu_not_claimed=True)
        time.sleep(30)


def prepare(root,data,args):
    eligible=data.eligible_history(args.context);times=pd.to_datetime(data.arrays['time_ns'],utc=True);cohorts={}
    for fold in data.metadata['folds']:
        roles=fold_roles(times,fold['boundaries'])
        for i,role in enumerate(ROLES):cohorts[fold['name']+'_'+role]=_indices(np.flatnonzero(eligible&(roles==i)),data)
    settings=vars(args).copy();settings={k:str(v) if isinstance(v,Path) else v for k,v in settings.items()}
    protocol=dict(data=_data_identity(data),arguments=settings,sources={p:sha256(BASE/p) for p in SOURCES},
        cohorts={k:dict(count=len(v),sha256=_array_hash(v)) for k,v in cohorts.items()},seeds=list(SEEDS),variants=list(VARIANTS),
        all_queries=128,purge_minutes=1440,excluded_history_origins=int((~eligible).sum()),
        precision=getattr(args,'precision','fp32'),optimizer=dict(lr=.0006,weight_decay=.03,clip=1.),temperature_grid=list(TEMPERATURES))
    if (root/'protocol.json').exists():
        if json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('Frozen protocol changed')
    else:
        atomic_json(root/'protocol.json',protocol);snapshot=root/'source-snapshot';snapshot.mkdir(exist_ok=True)
        for source in SOURCES:shutil.copy2(BASE/source,snapshot/Path(source).name)
    for source,digest in protocol['sources'].items():
        if sha256(root/'source-snapshot'/Path(source).name)!=digest:raise ValueError('Source snapshot changed')
    return protocol,cohorts


def run_experiment(root,data,args,protocol,cohorts):
    groups=[]
    for fold in data.metadata['folds']:
        name=fold['name']
        for variant in VARIANTS:
            paths=[]
            for seed in SEEDS:
                path=root/name/variant/f'seed-{seed}'
                progress(root,'pretrain' if variant in ('pretrained','combined') else 'fit',fold=name,variant=variant,seed=seed)
                train_member(data,cohorts[name+'_fit'],cohorts[name+'_selection'],path,variant,seed,args);paths.append(path)
            groups.append(dict(fold=name,variant=variant,paths=paths))
    files={str(p/name):sha256(p/name) for group in groups for p in group['paths'] for name in ('fit.json','best.pt','last.pt')}
    frozen=dict(protocol_sha256=sha256(root/'protocol.json'),files=files)
    marker=root/'choices-frozen.json'
    if marker.exists() and json.loads(marker.read_text())!=frozen:raise ValueError('Frozen model choices changed')
    atomic_json(marker,frozen);progress(root,'freeze',models=len(groups)*len(SEEDS))
    calibrations={}
    for group in groups:
        key=group['fold']+'/'+group['variant'];path=root/group['fold']/group['variant']/'calibration.json'
        progress(root,'calibrate',group=key)
        if path.exists():
            calibration=json.loads(path.read_text())
            if calibration['freeze_sha256']!=sha256(marker):raise ValueError('Calibration freeze changed')
        else:
            calibration=calibrate_members(data,cohorts[group['fold']+'_calibration'],group['paths'],args)
            calibration['freeze_sha256']=sha256(marker);atomic_json(path,calibration)
        calibrations[key]=calibration
    # Test labels are first read here, after every model and temperature is frozen.
    results={}
    for group in groups:
        key=group['fold']+'/'+group['variant'];path=root/group['fold']/group['variant']/'test.json'
        progress(root,'test',group=key)
        receipt=dict(freeze_sha256=sha256(marker),calibration_sha256=sha256(root/group['fold']/group['variant']/'calibration.json'))
        if path.exists():
            result=json.loads(path.read_text())
            if result['receipt']!=receipt:raise ValueError('Test receipt changed')
        else:
            rows=cohorts[group['fold']+'_test'];result=dict(receipt=receipt,
                raw=evaluate_members(data,rows,group['paths'],args),
                calibrated=evaluate_members(data,rows,group['paths'],args,calibrations[key]['temperature']))
            atomic_json(path,result)
        results[key]=result;gc.collect()
    for group in groups:
        key=group['fold']+'/'+group['variant'];baseline=results[group['fold']+'/baseline']['calibrated']
        results[key]['paired_vs_baseline']=paired_daily_bootstrap(results[key]['calibrated'],baseline)
    report=dict(protocol_sha256=sha256(root/'protocol.json'),freeze_sha256=sha256(marker),calibrations=calibrations,groups=results,
        limitations=['Native Binance price touches, not executable fills or trading returns.',
            'The fixed current asset universe is not survivorship-free. Held-out historical periods are not prospective evidence.',
            'No full per-query predictions stored; reproduce from frozen source/dataset/weights.',
            'Day bootstrap intervals are descriptive under temporal dependence.'])
    atomic_json(root/'results.json',report)
    lines=['# Barrier vNext results','',f"Three-seed ensembles; all128queries and full common cohorts. Forward precision {getattr(args,'precision','fp32')}; FP32 likelihood.",
        'Temperature uses separate calibration dates. Every model freezes before calibration and test.','',
        '|Fold/model|Horizon min|Raw log loss|Calibrated log loss|Calibrated Brier|Scored|','|---|---:|---:|---:|---:|---:|']
    for key,result in results.items():
        for raw,cal in zip(result['raw']['by_horizon'],result['calibrated']['by_horizon']):
            lines.append(f"|{key}|{cal['horizon_minutes']}|{raw['log_loss']}|{cal['log_loss']}|{cal['brier']}|{cal['scored']}|")
    lines+=['']+report['limitations'];(root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    progress(root,'complete',results_sha256=sha256(root/'results.json'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='data/barrier-binance-history-20261023')
    parser.add_argument('--dataset',default='data/barrier-vnext-20261023');parser.add_argument('--output',default='runs/barrier-vnext-20261023')
    parser.add_argument('--device',default='cuda');parser.add_argument('--wait-comparison',default='runs/barrier-full-compare-20261022')
    parser.add_argument('--gpu-lock',required=True);parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args(argv)
    # Production budgets are fixed. Tiny checks invoke the functions with synthetic data.
    args.context=10080;args.width=96;args.layers=3;args.batch=64;args.microbatch=16
    args.epochs=12;args.patience=3;args.pretrain_epochs=2;args.checkpoint_batches=200
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True);source=Path(args.source).resolve()
    with run_lock(root/'run.lock'):
        torch.set_num_threads(args.threads)
        wait_source(root,source)
        progress(root,'build',gpu_not_claimed=True)
        from wallex_lab.barrier_vnext_data import build_dataset,NextBarrierDataset
        build_dataset(source,args.dataset);data=NextBarrierDataset(args.dataset);data.context=args.context
        try:
            wait_comparison(root,args.wait_comparison)
            progress(root,'waiting_for_gpu_lease',gpu_not_claimed=True,lease=args.gpu_lock)
            with wait_lock(args.gpu_lock) if args.device.startswith('cuda') else nullcontext():
                if args.device.startswith('cuda'):
                    torch.use_deterministic_algorithms(True);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
                args.precision='bf16' if args.device.startswith('cuda') and torch.cuda.is_bf16_supported() else 'fp32'
                protocol,cohorts=prepare(root,data,args)
                if shutil.disk_usage(root).free<5*2**30:raise OSError('Less than5GiB free disk reserve')
                try:run_experiment(root,data,args,protocol,cohorts)
                except torch.cuda.OutOfMemoryError as error:
                    progress(root,'error',reason='CUDA out of memory at frozen microbatch16; free GPU memory and resume. Last committed batch is retained. No hidden batch reduction.')
                    raise RuntimeError('Frozen microbatch16 exceeded GPU memory; last committed checkpoint can resume after resources are freed') from error
        finally:data.close()


if __name__=='__main__':main()

```

## checks/check_barrier_vnext_runner.py
```python
"""Real tiny CPU optimization and deterministic interrupted resume for vNext."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from wallex_lab.barrier_dense import query_features


class TinyData:
    def __init__(self):
        self.context=720;self.metadata=dict(symbols=['tiny'],queries_per_origin=128)
        self.arrays={'time_ns':np.arange(10,dtype=np.int64)*300_000_000_000,'symbol':np.zeros(10,np.int16)}
        rng=np.random.default_rng(4);self.x=rng.normal(size=(10,720,14)).astype(np.float32)
        sigma=np.full(10,.001);dist=np.full((10,128,2),.003)
        self.q=query_features(dist,sigma,np.repeat([1.,-1.],64),np.zeros(128))
        self.event=np.tile(np.array([0,2,-1,1,-1,4,0,3,-1,2])[:,None],(1,128))
        self.cause=np.where(self.event<0,3,np.arange(10)[:,None]%3)
        self.survive=np.where(self.event<0,3,self.event)
    def batch(self,indices,device='cpu',**kwargs):
        return {k:torch.as_tensor(v[indices],device=device,dtype=torch.float32 if k in ('x','q') else torch.long)
            for k,v in dict(x=self.x,q=self.q,symbol=self.arrays['symbol'],event=self.event,cause=self.cause,survive=self.survive).items()}


def main():
    assert importlib.util.find_spec('scripts.train_barrier_vnext'), 'vNext runner missing'
    from scripts import train_barrier_vnext as runner
    torch.set_num_threads(2);data=TinyData();fit=np.arange(0,10,2);selection=np.arange(1,10,2)
    args=SimpleNamespace(device='cpu',context=720,width=16,layers=1,batch=2,epochs=2,patience=3,pretrain_epochs=1,checkpoint_batches=1)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);original=runner.save_checkpoint;interrupted=[False]
        def interrupt(path,state):
            original(path,state)
            if state['stage']=='fit' and state['cursor']>0 and not interrupted[0]:
                interrupted[0]=True;raise InterruptedError('Committed training interruption')
        runner.save_checkpoint=interrupt
        try:
            try:runner.train_member(data,fit,selection,root/'resume','pretrained',42,args)
            except InterruptedError:pass
            else:raise AssertionError('Resume test did not interrupt')
        finally:runner.save_checkpoint=original
        resumed=runner.train_member(data,fit,selection,root/'resume','pretrained',42,args)
        fresh=runner.train_member(data,fit,selection,root/'fresh','pretrained',42,args)
        a=torch.load(root/'resume'/'best.pt',weights_only=False,map_location='cpu')
        b=torch.load(root/'fresh'/'best.pt',weights_only=False,map_location='cpu')
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        assert resumed['selected_epoch']==fresh['selected_epoch']
        assert all(row['fit_origins_seen']==5 and row['fit_origin_query_visits']==640 for row in resumed['curve'])
        assert resumed['pretraining']['origins_per_epoch']==5 and resumed['pretraining']['fit_only']
        assert runner.train_member(data,fit,selection,root/'resume','pretrained',42,args)==resumed
        try:runner.train_member(data,selection,fit,root/'resume','pretrained',42,args)
        except ValueError:pass
        else:raise AssertionError('Changed cohorts accepted on resume')
        result=runner.evaluate_members(data,selection,[root/'resume'],args,temperature=1.)
        assert len(result['by_horizon'])==5 and all(row['scored']>0 for row in result['by_horizon'])
        assert len(result['conditional_tp_given_touch'])==5 and np.isfinite(result['partial_label_nll'])
        assert result['coherence_checked'] and result['daily']
        calibration=runner.calibrate_members(data,selection,[root/'resume'],args)
        assert 1. in calibration['grid'] and calibration['temperature'] in calibration['grid']
        assert min(calibration['nll'])<=calibration['nll'][calibration['grid'].index(1.)]
        paired=runner.paired_daily_bootstrap(result,result)
        assert paired['mean_difference']==[0.]*5
        # The three-seed likelihood is the mixture of complete paths, not mean logits.
        paths=[]
        for seed in (42,43,44):
            path=root/('seed-'+str(seed));runner.train_member(data,fit,selection,path,'baseline',seed,args);paths.append(path)
        mixture=runner.evaluate_members(data,selection,paths,args)
        assert mixture['origin_query_visits']==640
        from wallex_lab.barrier_survival import mixture_nll
        models=runner._load_members(data,paths,args);b=data.batch(selection)
        with torch.inference_mode():
            direct=float(mixture_nll([m(b['x'],b['q'],b['symbol']) for m in models],b['event'],b['cause'],b['survive']))
        assert abs(mixture['partial_label_nll']-direct)<1e-6
        # An interruption during masked pretraining must replay identical mask/dropout RNG.
        interrupted[0]=False
        def interrupt_pretrain(path,state):
            original(path,state)
            if state['stage']=='pretrain' and state['cursor']>0 and not interrupted[0]:
                interrupted[0]=True;raise InterruptedError('Committed pretraining interruption')
        runner.save_checkpoint=interrupt_pretrain
        try:
            try:runner.train_member(data,fit,selection,root/'pretrain-resume','pretrained',42,args)
            except InterruptedError:pass
            else:raise AssertionError('Pretraining did not interrupt')
        finally:runner.save_checkpoint=original
        pre=runner.train_member(data,fit,selection,root/'pretrain-resume','pretrained',42,args)
        c=torch.load(root/'pretrain-resume'/'best.pt',weights_only=False,map_location='cpu')
        for k in a:torch.testing.assert_close(a[k],c[k],rtol=0,atol=0)
        assert pre['pretraining']['curve'][0]['origins_seen']==5
        # Source changes are rejected even when completed fit metadata exists.
        changed=TinyData();changed.x[0,0,0]+=1
        try:runner.train_member(changed,fit,selection,root/'resume','pretrained',42,args)
        except ValueError:pass
        else:raise AssertionError('Changed synthetic source accepted')
        # Waiting must not touch the CUDA runtime, including memory-info helpers.
        comparison=root/'comparison';comparison.mkdir();wait_root=root/'wait';wait_root.mkdir()
        def finish_wait(_): (comparison/'results.json').write_text('{}')
        def cuda_forbidden(*_args,**_kwargs):raise AssertionError('CUDA touched before lease')
        with patch.object(runner.time,'sleep',finish_wait),patch.object(torch.cuda,'is_available',cuda_forbidden),patch.object(torch.cuda,'mem_get_info',cuda_forbidden),patch.object(torch.cuda,'init',cuda_forbidden):
            runner.wait_comparison(wait_root,comparison)
        source=root/'source';source.mkdir();(source/'manifest.json').write_text(json.dumps(dict(status='partial',missing=['day.zip'])))
        try:runner.wait_source(wait_root,source)
        except ValueError as error:assert 'resume' in str(error).lower()
        else:raise AssertionError('Unavailable source partitions accepted')
        assert json.loads((wait_root/'progress.json').read_text())['stage']=='source_blocked'
        # Exercise the real stage coordinator with separate synthetic role cohorts.
        tiny=TinyData();tiny.metadata['folds']=[dict(name='tiny')];end_to_end=root/'end-to-end';end_to_end.mkdir()
        (end_to_end/'protocol.json').write_text('{}');original_batch=tiny.batch
        def guarded_batch(indices,**kwargs):
            if (np.asarray(indices)>=6).any():
                assert (end_to_end/'choices-frozen.json').exists(), 'Calibration opened before freeze'
                assert len(list(end_to_end.glob('tiny/*/seed-*/fit.json')))==6
            if (np.asarray(indices)>=8).any():
                assert len(list(end_to_end.glob('tiny/*/calibration.json')))==2, 'Test opened before all calibration choices froze'
            return original_batch(indices,**kwargs)
        tiny.batch=guarded_batch
        cohorts={'tiny_fit':np.arange(4),'tiny_selection':np.arange(4,6),'tiny_calibration':np.arange(6,8),'tiny_test':np.arange(8,10)}
        tiny_args=SimpleNamespace(**{**vars(args),'epochs':1})
        with patch.object(runner,'VARIANTS',('baseline','pretrained')):
            runner.run_experiment(end_to_end,tiny,tiny_args,{},cohorts)
            runner.run_experiment(end_to_end,tiny,tiny_args,{},cohorts)
        completed=json.loads((end_to_end/'results.json').read_text())
        assert set(completed['groups'])=={'tiny/baseline','tiny/pretrained'}
        assert json.loads((end_to_end/'progress.json').read_text())['stage']=='complete'
        assert not list(root.rglob('*predictions*.npz')), 'Do not duplicate huge per-query test arrays'
    print('PASS vNext actual pretraining/fit/calibration,128queries,exact fit+pretrain resume,source/cohort identity,3seed mixture,streaming metrics and preCUDA wait')


if __name__=='__main__':main()

```
