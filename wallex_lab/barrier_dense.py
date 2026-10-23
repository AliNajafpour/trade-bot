"""Indexed dense origins: raw candles once, continuous barrier labels in memmaps."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .barrier_dataset import HORIZONS,GEOMETRIES,BarrierLabelBook,sha256,write_json,barrier_query
from .data import load_market

CONTEXT=720
QUERIES=128
OFFSET=np.tile((np.arange(64)%3-1)*.5,2)
SIDE=np.repeat([1.,-1.],64)


def dense_folds(start,end):
    start=pd.Timestamp(start);end=pd.Timestamp(end).floor('D')-pd.Timedelta(days=1)
    result=[]
    for name,stop in [('early',end-pd.Timedelta(days=45)),('late',end)]:
        bounds=[start,stop-pd.Timedelta(days=58),stop-pd.Timedelta(days=44),stop-pd.Timedelta(days=30),stop]
        if bounds[1]<=start+pd.Timedelta(days=30):raise ValueError('Insufficient history for expanding folds')
        result.append(dict(name=name,boundaries=[value.isoformat() for value in bounds]))
    return result


def query_distances(sigma,seed):
    rng=np.random.default_rng(seed);parts=[GEOMETRIES*sigma*np.sqrt(60.)]
    for low,high,scale in [(.15,6.,sigma*np.sqrt(60.)),(.0002,.05,1.)]:
        center=np.exp(rng.uniform(np.log(low),np.log(high),26))*scale
        ratio=np.exp(rng.uniform(-np.log(8.),np.log(8.),26))
        parts.append(np.column_stack([center*np.sqrt(ratio),center/np.sqrt(ratio)]))
    return np.clip(np.concatenate(parts),.0001,.15).astype(np.float32)


def batch_history(values,times,indices,context=CONTEXT):
    indices=np.asarray(indices,dtype=np.int64)
    if (indices<context-1).any() or (indices>=len(values)).any():raise ValueError('Invalid history index')
    positions=indices[:,None]+np.arange(-context+1,1)
    window=np.asarray(values[positions],dtype=np.float64);close=window[:,:,3];ref=close[:,-1]
    returns=np.concatenate([np.zeros((len(indices),1)),np.diff(np.log(close),axis=1)],axis=1)
    sigma=np.maximum(np.maximum(returns[:,-60:].std(1),returns[:,-360:].std(1)),1e-5)
    relative=np.log(window[:,:,:4]/ref[:,None,None])/(sigma[:,None,None]*np.sqrt(60.))
    volume=window[:,:,4];volume_scale=np.median(volume[:,-60:],axis=1)
    volume_scale=np.where(volume_scale>0,volume_scale,volume[:,-60:].mean(1))
    volume_scale=np.where(volume_scale>0,volume_scale,volume.mean(1));volume_scale=np.where(volume_scale>0,volume_scale,1.)
    relative_volume=np.log1p(volume/volume_scale[:,None])
    hour=(times[positions]//60_000_000_000%1440)*(2*np.pi/1440)
    x=np.concatenate([relative,relative_volume[:,:,None],
        (np.log(close/window[:,:,0])/sigma[:,None])[:,:,None],
        (np.log(window[:,:,1]/window[:,:,2])/sigma[:,None])[:,:,None],
        (returns/sigma[:,None])[:,:,None],np.sin(hour)[:,:,None],np.cos(hour)[:,:,None]],axis=-1)
    return np.clip(x,-50,50).astype(np.float32),sigma,ref


def query_features(distances,sigma,side,offset):
    up=distances[...,0];down=distances[...,1]
    return np.stack([100*up,100*down,np.log1p(up/sigma[:,None]),np.log1p(down/sigma[:,None]),np.log(up/down),
        np.broadcast_to(np.log(sigma*1e4)[:,None],up.shape),100*sigma[:,None]*offset[None,:],
        np.broadcast_to(side[None,:],up.shape)],axis=-1).astype(np.float32)


class IndexedBarrierDataset:
    def __init__(self,root,verify=True):
        self.root=Path(root);self.metadata=json.loads((self.root/'dataset.json').read_text(encoding='utf-8'))
        if self.metadata['status']!='ready':raise ValueError('Incomplete dense dataset')
        if verify:
            for name,digest in self.metadata['files'].items():
                if sha256(self.root/name)!=digest:raise ValueError('Dense dataset changed: '+name)
        self.arrays={name:np.load(self.root/(name+'.npy'),mmap_mode='r') for name in
            ('time_ns','symbol','bar_index','sigma','reference','distances','event_bin','cause','survived_bins','observed_minutes')}
        self.series=[(np.load(self.root/f'series-{asset}.npy',mmap_mode='r'),np.load(self.root/f'times-{asset}.npy',mmap_mode='r'))
            for asset in range(len(self.metadata['symbols']))]
        self.resident=None
        self.context=CONTEXT

    def eligible_history(self,context):
        if context<CONTEXT:context=CONTEXT
        valid=np.zeros(len(self.arrays['time_ns']),dtype=bool)
        for asset,(_,times) in enumerate(self.series):
            rows=np.flatnonzero(self.arrays['symbol']==asset);ends=np.asarray(self.arrays['bar_index'][rows])
            gaps=np.r_[0,np.cumsum(np.diff(times)!=60_000_000_000)]
            starts=ends-context+1
            valid[rows]=(starts>=0)&(gaps[ends]==gaps[np.maximum(starts,0)])
        return valid

    def preload(self,device='cuda'):
        """Keep shared raw sources/labels on GPU; derive exact relative features in batches."""
        import torch
        device=torch.device(device)
        if device.type!='cuda':raise ValueError('Resident acceleration requires CUDA')
        tables=[];offsets=[];offset=0
        for values,times in self.series:
            offsets.append(offset);offset+=len(values);log_prices=np.log(values[:,:4])
            angle=(times//60_000_000_000%1440)*(2*np.pi/1440)
            table=np.column_stack([log_prices,values[:,4],np.log(values[:,3]/values[:,0]),np.log(values[:,1]/values[:,2]),
                np.r_[0.,np.diff(log_prices[:,3])],np.sin(angle),np.cos(angle)])
            tables.append(torch.as_tensor(table,device=device))
        self.resident=dict(device=device,raw=torch.cat(tables),
            origin=torch.as_tensor(np.array(self.arrays['bar_index'])+np.asarray(offsets)[self.arrays['symbol']],device=device),
            sigma=torch.as_tensor(np.array(self.arrays['sigma']),device=device),
            distances=torch.as_tensor(np.array(self.arrays['distances']),device=device),
            symbol=torch.as_tensor(np.array(self.arrays['symbol']),device=device,dtype=torch.long),
            offset=torch.as_tensor(OFFSET,device=device),side=torch.as_tensor(SIDE,device=device))
        for field,key in [('event_bin','event'),('cause','cause'),('survived_bins','survive')]:
            self.resident[key]=torch.as_tensor(np.array(self.arrays[field]),device=device)

    def resident_batch(self,indices,queries,with_history):
        import torch
        state=self.resident;device=state['device'];i=torch.as_tensor(indices,device=device);j=torch.as_tensor(queries,device=device)
        sigma=state['sigma'][i];d=state['distances'][i[:,None],j[None,:]].double();up=d[...,0];down=d[...,1]
        q=torch.stack([100*up,100*down,torch.log1p(up/sigma[:,None]),torch.log1p(down/sigma[:,None]),torch.log(up/down),
            torch.log(sigma*1e4)[:,None].expand_as(up),100*sigma[:,None]*state['offset'][j][None,:],state['side'][j][None,:].expand_as(up)],dim=-1).float()
        if with_history:
            positions=state['origin'][i][:,None]+torch.arange(-self.context+1,1,device=device)
            window=state['raw'][positions];volume=window[:,:,4];sorted_volume=volume[:,-60:].sort(-1).values
            scale=sorted_volume[:,29:31].mean(-1);scale=torch.where(scale>0,scale,volume[:,-60:].mean(-1))
            scale=torch.where(scale>0,scale,volume.mean(-1));scale=torch.where(scale>0,scale,torch.ones_like(scale))
            relative=(window[:,:,:4]-window[:,-1:,3:4])/(sigma[:,None,None]*np.sqrt(60.))
            window[:,0,7]=0
            x=torch.cat([relative,torch.log1p(volume/scale[:,None])[:,:,None],window[:,:,5:8]/sigma[:,None,None],window[:,:,8:10]],dim=-1).clamp(-50,50).float()
        else:x=torch.empty((len(indices),1,10),device=device)
        result=dict(x=x,q=q,symbol=state['symbol'][i])
        for key in ('event','cause','survive'):result[key]=state[key][i[:,None],j[None,:]].long()
        return result

    def batch(self,indices,queries=None,device='cpu',with_history=True):
        import torch
        indices=np.asarray(indices,dtype=np.int64);queries=np.arange(QUERIES) if queries is None else np.asarray(queries,dtype=np.int64)
        if self.resident is not None and torch.device(device).type=='cuda':return self.resident_batch(indices,queries,with_history)
        assets=self.arrays['symbol'][indices];x=np.empty((len(indices),self.context if with_history else 1,10),np.float32)
        if with_history:
            for asset in np.unique(assets):
                selected=np.flatnonzero(assets==asset);values,times=self.series[asset]
                x[selected]=batch_history(values,times,self.arrays['bar_index'][indices[selected]],self.context)[0]
        distances=np.asarray(self.arrays['distances'][indices][:,queries]);sigma=np.asarray(self.arrays['sigma'][indices])
        q=query_features(distances,sigma,SIDE[queries],OFFSET[queries])
        result=dict(x=torch.from_numpy(x),q=torch.from_numpy(q),symbol=torch.as_tensor(np.array(assets),dtype=torch.long))
        for field,key in [('event_bin','event'),('cause','cause'),('survived_bins','survive')]:
            result[key]=torch.as_tensor(np.array(self.arrays[field][indices][:,queries]),dtype=torch.long)
        return {key:value.to(device) for key,value in result.items()}


def build_dense(source,output):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('Dense dataset output already exists')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if manifest['status']!='ready' or sha256(source/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('Source acquisition not verified')
    output.mkdir(parents=True);origins=[];frames=[];sources=[]
    for asset,(symbol,record) in enumerate(sorted(manifest['symbols'].items())):
        info=record['data']
        if sha256(info['native_path'])!=info['native_sha256']:raise ValueError('Source hash mismatch')
        frame=load_market(info['native_path'],60);book=BarrierLabelBook(frame)
        values=frame[['open','high','low','close','volume']].to_numpy(np.float64)
        np.save(output/f'series-{asset}.npy',values);np.save(output/f'times-{asset}.npy',book.ns)
        returns=pd.Series(np.r_[0.,np.diff(np.log(values[:,3]))])
        vol=np.maximum(np.maximum(returns.rolling(60).std(ddof=0),returns.rolling(360).std(ddof=0)),1e-5).to_numpy()
        frames.append((values,book,vol));sources.append(dict(symbol=symbol,path=info['native_path'],sha256=info['native_sha256'],rows=len(frame),
            first=str(book.times[0]),last=str(book.times[-1]),coverage_fraction=info['coverage_fraction_1m']))
        gap=np.r_[0,np.cumsum(np.diff(book.ns)!=60_000_000_000)]
        index=np.arange(CONTEXT-1,len(frame));decision=book.ns[index]+60_000_000_000
        good=(decision%300_000_000_000==0)&(gap[index]==gap[index-CONTEXT+1])
        origins.extend(zip(np.full(good.sum(),asset),index[good],decision[good]))
    n=len(origins);shapes=dict(time_ns=((n,),'int64'),symbol=((n,),'int16'),bar_index=((n,),'int64'),sigma=((n,),'float64'),
        reference=((n,),'float64'),distances=((n,QUERIES,2),'float32'),event_bin=((n,QUERIES),'int8'),
        cause=((n,QUERIES),'int8'),survived_bins=((n,QUERIES),'int8'),observed_minutes=((n,),'int16'))
    arrays={key:np.lib.format.open_memmap(output/(key+'.npy'),mode='w+',dtype=dtype,shape=shape) for key,(shape,dtype) in shapes.items()}
    cause_counts=np.zeros(4,dtype=np.int64);reasons={};clipped_values=0;total_values=0
    for begin in range(0,n,256):
        cohort=origins[begin:begin+256]
        for local,(asset,index,decision) in enumerate(cohort):
            row=begin+local;values,book,vol=frames[asset];sigma=float(vol[index]);ref=float(values[index,3])
            distance=query_distances(sigma,int(decision//60_000_000_000)+int(asset)*10_000_000)
            all_distances=np.tile(distance,(2,1));entries=ref*np.exp(sigma*OFFSET)
            for orientation,side in enumerate(('long','short')):
                part=slice(orientation*64,(orientation+1)*64);sign=1 if side=='long' else -1
                tp=entries[part]*np.exp(sign*distance[:,0].astype(float));sl=entries[part]*np.exp(-sign*distance[:,1].astype(float))
                labels=book.label(int(index),entries[part],tp,sl,side)
                for key in ('event_bin','cause','survived_bins'):arrays[key][row,part]=labels[key]
                cause_counts+=np.bincount(labels['cause'],minlength=4)
            arrays['time_ns'][row]=decision;arrays['symbol'][row]=asset;arrays['bar_index'][row]=index
            arrays['sigma'][row]=sigma;arrays['reference'][row]=ref;arrays['distances'][row]=all_distances
            arrays['observed_minutes'][row]=labels['observed_minutes'];reason=labels['censor_reason'];reasons[reason]=reasons.get(reason,0)+1
        if begin%10240==0:print(f'DENSE_DATASET {begin+len(cohort)}/{n}',flush=True)
    for value in arrays.values():value.flush()
    metadata=dict(status='ready',storage='indexed-v2',feature_version='relative-v2',origins=n,queries_per_origin=QUERIES,
        stride_minutes=5,context_minutes=CONTEXT,horizons=HORIZONS.tolist(),symbols=[r['symbol'] for r in sources],sources=sources,
        source_manifest_sha256=sha256(source/'manifest.json'),acquisition_protocol=protocol,
        folds=dense_folds(protocol['requested_from_utc'],protocol['requested_to_exclusive_utc']),
        cause_counts={name:int(cause_counts[j]) for j,name in enumerate(('tp','sl','unknown','censored_or_no_touch'))},censor_reasons=reasons,
        geometry_support={'min_fraction':float(np.expm1(.0001)),'max_fraction':float(np.expm1(.15)),'max_ratio':8.},
        query_policy='64 per direction: 12 anchor + 26 continuous volatility-scaled + 26 absolute-percent; all ex ante.',
        entry_policy='Completed close * exp(-0.5/0/+0.5 trailing history sigma), predetermined; not executable fills.',
        normalization='relative-v2: log OHLC / last close; divide by max trailing 60/360 minute return sigma; relative volume, no global scaler.',
        known_limits=['Five-minute origins and mirrored queries are dependent.','No named-pattern labels or future-derived technical levels.',
            'Clip extreme input channels to +/-50; model is not invariant to changes in market volatility.'],
        files={p.name:sha256(p) for p in output.glob('*.npy')})
    write_json(output/'dataset.json',metadata);print('DENSE_DATASET_READY '+str(output.resolve()),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();build_dense(args.source,args.output)


if __name__=='__main__':main()
