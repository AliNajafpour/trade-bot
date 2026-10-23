"""Causal native-minute data and partial first-touch labels, independent of trading exits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_market
from .features import _validated_frame

HORIZONS=np.array([5,15,60,240,1440],dtype=np.int64)
MAX_CONTEXT=360
GEOMETRIES=np.array([(.4,.4),(.7,.7),(1.2,1.2),(2.,2.),(.7,1.4),(1.4,.7),
    (1.2,2.4),(2.4,1.2),(.4,1.2),(1.2,.4),(3.,1.),(1.,3.)])
FOLDS=[dict(name='early',boundaries=['2026-04-24','2026-05-31','2026-06-08','2026-06-16','2026-06-27']),
       dict(name='late',boundaries=['2026-04-24','2026-06-27','2026-07-04','2026-07-10','2026-07-23'])]


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def fold_roles(times,boundaries):
    """A full 24h target purge; historical context may precede a role boundary."""
    times=pd.DatetimeIndex(times);bounds=pd.to_datetime(boundaries,utc=True)
    roles=np.full(len(times),-1,dtype=np.int8)
    for role in range(4):
        roles[(times>=bounds[role])&(times+pd.Timedelta(minutes=int(HORIZONS[-1]))<bounds[role+1])]=role
    return roles


class BarrierLabelBook:
    def __init__(self,frame):
        data,self.times,_,interval=_validated_frame(frame)
        if interval!=60:raise ValueError('Native M1 candles required')
        self.ns=self.times.as_unit('ns').asi8
        if (self.ns%60_000_000_000).any():raise ValueError('Minute-aligned UTC opens required')
        self.prices=data[['open','high','low','close']].to_numpy()

    def label(self,index,entry,tp,sl,side='long'):
        """Event times use touched-minute completion, not invented intrabar ordering."""
        if not isinstance(index,(int,np.integer)) or index<0 or index>=len(self.ns):raise ValueError('Invalid origin')
        tp=np.atleast_1d(np.asarray(tp,float));sl=np.atleast_1d(np.asarray(sl,float))
        sign=1 if side=='long' else -1 if side=='short' else 0
        if not sign or tp.shape!=sl.shape or tp.ndim!=1:raise ValueError('Invalid barriers')
        entry=np.broadcast_to(np.asarray(entry,float),tp.shape)
        if not np.isfinite(entry).all() or (entry<=0).any():raise ValueError('Invalid entries')
        if not np.isfinite(tp).all() or not np.isfinite(sl).all() or (tp<=0).any() or (sl<=0).any():raise ValueError('Invalid prices')
        if (sign*(tp-entry)<=0).any() or (sign*(entry-sl)<=0).any():raise ValueError('Entry must be between TP and SL')
        start=index+1;end=min(start+int(HORIZONS[-1]),len(self.ns));expected=self.ns[index]+60_000_000_000
        clocks=self.ns[start:end];bad=np.flatnonzero(clocks!=expected+np.arange(len(clocks))*60_000_000_000)
        reason='gap' if len(bad) else 'source_end' if len(clocks)<HORIZONS[-1] else 'complete'
        if len(bad):end=start+int(bad[0])
        future=self.prices[start:end];duration=len(future);k=len(tp)
        event_bin=np.full(k,-1,np.int8);cause=np.full(k,3,np.int8);minute=np.full(k,-1,np.int16)
        survived=np.full(k,np.searchsorted(HORIZONS,duration,side='right'),np.int8)
        if duration:
            upper=np.maximum.accumulate(future[:,1]);lower=np.maximum.accumulate(-future[:,2])
            tp_time=np.searchsorted(upper,tp) if sign==1 else np.searchsorted(lower,-tp)
            sl_time=np.searchsorted(lower,-sl) if sign==1 else np.searchsorted(upper,sl)
            first=np.minimum(tp_time,sl_time);chosen=np.flatnonzero(first<duration);rows=first[chosen]
            open_tp=sign*(future[rows,0]-tp[chosen])>=0;open_sl=sign*(sl[chosen]-future[rows,0])>=0
            kinds=np.where(open_tp,0,np.where(open_sl,1,np.where(tp_time[chosen]==sl_time[chosen],2,np.where(tp_time[chosen]<sl_time[chosen],0,1))))
            minute[chosen]=rows+1;event_bin[chosen]=np.searchsorted(HORIZONS,rows+1)
            cause[chosen]=kinds;survived[chosen]=event_bin[chosen]
        return dict(event_bin=event_bin,cause=cause,survived_bins=survived,event_minute=minute,
            observed_minutes=duration,censor_reason=reason)


def history_inputs(frame,version='relative-v1'):
    """Only completed history supplied here; no fitted/global normalization."""
    data,times,gaps,interval=_validated_frame(frame)
    if interval!=60 or len(data)<180 or gaps.any():raise ValueError('At least 180 contiguous M1 history rows required')
    close=data.close.to_numpy();ref=float(close[-1]);ret=np.r_[0.,np.diff(np.log(close))]
    if version not in ('relative-v1','relative-v2'):raise ValueError('Unknown normalization version')
    sigma=max(float(np.std(ret[-60:])),1e-5)
    if version=='relative-v2':sigma=max(sigma,float(np.std(ret[-360:])))
    relative=np.log(data[['open','high','low','close']].to_numpy()/ref)/(sigma*np.sqrt(60.))
    volumes=data.volume.to_numpy()
    if version=='relative-v1':v=np.log1p(volumes)-np.log1p(np.median(volumes[-60:]))
    else:
        volume_scale=float(np.median(volumes[-60:]))
        if volume_scale<=0:volume_scale=float(np.mean(volumes[-60:]))
        if volume_scale<=0:volume_scale=float(np.mean(volumes))
        v=np.log1p(volumes/(volume_scale if volume_scale>0 else 1.))
    hour=(times.hour.to_numpy()+times.minute.to_numpy()/60)*2*np.pi/24
    seq=np.column_stack([relative,v,np.log(close/data.open.to_numpy())/sigma,
        np.log(data.high.to_numpy()/data.low.to_numpy())/sigma,ret/sigma,np.sin(hour),np.cos(hour)])
    return np.clip(seq,-50,50).astype(np.float32),sigma,ref


def barrier_query(entry,tp,sl,sigma,reference,side='long'):
    sign=1 if side=='long' else -1 if side=='short' else 0
    values=np.r_[entry,np.atleast_1d(tp),np.atleast_1d(sl),sigma,reference]
    if not sign or not np.isfinite(values).all() or (values<=0).any():raise ValueError('Positive finite prices and volatility required')
    tp=np.atleast_1d(np.asarray(tp,float));sl=np.atleast_1d(np.asarray(sl,float))
    if tp.shape!=sl.shape or tp.ndim!=1:raise ValueError('Equal barrier vectors required')
    up=sign*np.log(tp/entry);down=sign*np.log(entry/sl)
    if (up<=0).any() or (down<=0).any():raise ValueError('Invalid TP/SL orientation')
    return np.column_stack([100*up,100*down,np.log1p(up/sigma),np.log1p(down/sigma),np.log(up/down),
        np.full(len(tp),np.log(sigma*1e4)),np.full(len(tp),100*np.log(entry/reference)),np.full(len(tp),sign)]).astype(np.float32)


def build_dataset(source,output,stride=30):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('Dataset output already exists')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='ready' or sha256(source/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('Source protocol mismatch')
    output.mkdir(parents=True)
    sources=[];frames=[];candidates=[];audit=Counter()
    for symbol,record in sorted(manifest['symbols'].items()):
        info=record['data'];path=Path(info['native_path']);expected=info['native_sha256']
        if sha256(path)!=expected:raise ValueError('Native source hash mismatch: '+symbol)
        frame=load_market(path,60);book=BarrierLabelBook(frame);frames.append((symbol,frame,book))
        sources.append(dict(symbol=symbol,path=str(path.resolve()),sha256=expected,rows=len(frame),
            first=str(book.times[0]),last=str(book.times[-1]),quote='USDT',source='Wallex public native M1'))
        gap=np.r_[0,np.cumsum(np.diff(book.ns)!=60_000_000_000)]
        for index in range(MAX_CONTEXT-1,len(frame)):
            decision=book.ns[index]+60_000_000_000
            if decision%(stride*60_000_000_000):continue
            if gap[index]!=gap[index-MAX_CONTEXT+1]:audit['history_gap_origins']+=1;continue
            candidates.append((len(frames)-1,index,decision))
    n=len(candidates);pairs=len(GEOMETRIES)*2
    x=np.lib.format.open_memmap(output/'history.npy',mode='w+',dtype='float32',shape=(n,MAX_CONTEXT,10))
    query=np.zeros((n,pairs,8),np.float32);event=np.full((n,pairs),-1,np.int8);cause=np.full_like(event,3)
    survive=np.zeros_like(event);minutes=np.full((n,pairs),-1,np.int16);duration=np.zeros(n,np.int16)
    symbols=np.zeros(n,np.int16);clocks=np.zeros(n,np.int64);sigma_array=np.zeros(n,np.float32);reference=np.zeros(n,np.float64)
    fractions=np.zeros((n,pairs,2),np.float32);reasons=np.empty(n,dtype='U10')
    for row,(asset,index,decision) in enumerate(candidates):
        symbol,frame,book=frames[asset]
        seq,sigma,ref=history_inputs(frame.iloc[index-MAX_CONTEXT+1:index+1]);x[row]=seq
        distances=np.clip(GEOMETRIES*sigma*np.sqrt(60.),.0005,.12)
        entries=ref*np.exp((np.arange(len(GEOMETRIES))%3-1)*.5*sigma)
        for orientation,side in enumerate(('long','short')):
            sign=1 if side=='long' else -1;part=slice(orientation*len(GEOMETRIES),(orientation+1)*len(GEOMETRIES))
            tp=entries*np.exp(sign*distances[:,0]);sl=entries*np.exp(-sign*distances[:,1])
            # Each historical entry offset is determined before looking at future candles.
            for j in range(len(GEOMETRIES)):
                column=orientation*len(GEOMETRIES)+j
                query[row,column]=barrier_query(entries[j],tp[j],sl[j],sigma,ref,side)[0]
                label=book.label(index,entries[j],tp[j],sl[j],side)
                event[row,column]=label['event_bin'][0];cause[row,column]=label['cause'][0]
                survive[row,column]=label['survived_bins'][0];minutes[row,column]=label['event_minute'][0]
            fractions[row,part]=distances
        duration[row]=label['observed_minutes'];reasons[row]=label['censor_reason'];symbols[row]=asset;clocks[row]=decision
        sigma_array[row]=sigma;reference[row]=ref
        if (row+1)%2000==0:print(f'dataset {row+1}/{n}',flush=True)
    x.flush()
    np.savez_compressed(output/'labels.npz',query=query,event_bin=event,cause=cause,survived_bins=survive,
        event_minute=minutes,observed_minutes=duration,symbol=symbols,time_ns=clocks,sigma=sigma_array,
        reference=reference,distances=fractions,censor_reason=reasons)
    audit.update({f'cause_{key}':int((cause==value).sum()) for key,value in [('tp',0),('sl',1),('ambiguous',2),('censored',3)]})
    meta=dict(status='ready',origins=n,queries_per_origin=pairs,symbols=[r['symbol'] for r in sources],
        source_manifest_sha256=sha256(source/'manifest.json'),sources=sources,horizons=HORIZONS.tolist(),
        context_minutes=MAX_CONTEXT,stride_minutes=stride,audit=dict(audit),folds=FOLDS,
        censor_reasons={str(k):int(v) for k,v in zip(*np.unique(reasons,return_counts=True))},
        geometry_support={'min_fraction':float(np.expm1(fractions.min())), 'max_fraction':float(np.expm1(fractions.max()))},
        entry='Last completed close, a historical proxy; not executable bid/ask or observed fill.',
        time_semantics='UTC minute-open timestamps; decision is last context open + 1min; touch time recorded at minute completion.',
        ambiguity='Unknown cause has likelihood TP mass + SL mass; not a conservative SL label.',
        censoring='Likelihood uses only fully observed output bins; within-bin censored exposure is not invented.',
        protocol_sha256=None,files={name:sha256(output/name) for name in ('history.npy','labels.npz')})
    write_json(output/'dataset.json',meta)
    return meta
