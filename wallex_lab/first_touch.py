"""Actual M1 first-touch labels and a small neural calibration head for foundation forecasts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import _validated_frame

CLASS_NAMES=('tp_first','sl_first','timeout')
GEOMETRIES={'g0':(.75,.10),'g1':(.90,.25),'g2':(.50,.25)}


class FirstTouchBook:
    """Validate a minute book once, then label many immutable forecast origins."""
    def __init__(self,frame):
        self.data,self.times,_,interval=_validated_frame(frame)
        if interval!=60:raise ValueError('First-touch execution requires native M1 bars')
        self.opens=self.data.open.to_numpy();self.high=self.data.high.to_numpy();self.low=self.data.low.to_numpy();self.close=self.data.close.to_numpy()
        self.ns=self.times.as_unit('ns').asi8

    def label(self,decision_at,target_price,stop_price,horizon_minutes,fee_bps=13.,slippage_bps=0.,*,origin_entry_price=None):
        result={'known_outcome':False,'class_index':None,'class_name':None,'censor_reason':None,'ambiguous_same_bar':False}
        at=pd.Timestamp(decision_at)
        if at.tzinfo is None:raise ValueError('Aware decision time required')
        if isinstance(horizon_minutes,bool) or int(horizon_minutes)!=horizon_minutes or horizon_minutes<1:raise ValueError('Positive integer horizon required')
        values=np.array([target_price,stop_price,fee_bps,slippage_bps],float)
        if not np.isfinite(values).all() or min(target_price,stop_price)<=0 or not 0<=fee_bps<10000 or not 0<=slippage_bps<10000:raise ValueError('Invalid prices/costs')
        at=at.tz_convert('UTC');end=at+pd.Timedelta(minutes=int(horizon_minutes))
        start_ns=(at.floor('min')+pd.Timedelta(minutes=1)).value;end_ns=end.value
        i=int(np.searchsorted(self.ns,start_ns))
        result.update(decision_at=at.isoformat(),horizon_end_at=end.isoformat(),horizon_minutes=int(horizon_minutes),
            target_price=float(target_price),stop_price=float(stop_price),fee_bps=float(fee_bps),slippage_bps=float(slippage_bps))
        if i>=len(self.ns) or self.ns[i]!=start_ns:result['censor_reason']='missing_entry_minute';return result
        slip=slippage_bps/10000;fee=fee_bps/10000
        entry=float(self.opens[i])*(1+slip) if origin_entry_price is None else float(origin_entry_price)
        if not np.isfinite(entry) or entry<=0:raise ValueError('Positive finite origin entry required')
        result.update(entry_at=self.times[i].isoformat(),entry_reference_price=float(self.opens[i]),entry_price=entry)
        if origin_entry_price is not None:
            result.update(entry_at=at.isoformat(),entry_reference_price=entry,
                price_coverage='Subsequent complete M1 bars only; partial origin minute is unobserved; candle touches are not executable bid fills')
        if not stop_price<entry<target_price:result['censor_reason']='entry_outside_barriers';return result
        expected=start_ns;last=None
        while i<len(self.ns) and self.ns[i]+60_000_000_000<=end_ns:
            if self.ns[i]!=expected:result['censor_reason']='missing_minute';return result
            opening=float(self.opens[i]);hit_sl=self.low[i]<=stop_price;hit_tp=self.high[i]>=target_price
            if opening<=stop_price:price=opening*(1-slip);kind=1;ambiguous=False;stamp=self.times[i]
            elif opening>=target_price:price=float(target_price)*(1-slip);kind=0;ambiguous=False;stamp=self.times[i]
            elif hit_sl:price=float(stop_price)*(1-slip);kind=1;ambiguous=bool(hit_tp);stamp=self.times[i]+pd.Timedelta(minutes=1)
            elif hit_tp:price=float(target_price)*(1-slip);kind=0;ambiguous=False;stamp=self.times[i]+pd.Timedelta(minutes=1)
            else:last=i;i+=1;expected+=60_000_000_000;continue
            return self._finish(result,kind,price,stamp,entry,fee,ambiguous)
        if last is None or self.ns[last]+60_000_000_000!=end_ns:
            result['censor_reason']='missing_minute' if i<len(self.ns) and self.ns[i]<end_ns else 'immature_horizon'
            return result
        return self._finish(result,2,float(self.close[last])*(1-slip),end,entry,fee,False)

    @staticmethod
    def _finish(result,kind,exit_price,stamp,entry,fee,ambiguous):
        result.update(known_outcome=True,class_index=kind,class_name=CLASS_NAMES[kind],exit_at=stamp.isoformat(),
            exit_price=float(exit_price),exit_reason=CLASS_NAMES[kind],ambiguous_same_bar=ambiguous,
            net_return=float(exit_price*(1-fee)/(entry*(1+fee))-1))
        return result


def label_trade(frame_1m,decision_at,target_price,stop_price,horizon_minutes,fee_bps=13.,slippage_bps=0.):
    return FirstTouchBook(frame_1m).label(decision_at,target_price,stop_price,horizon_minutes,fee_bps,slippage_bps)


def model_geometry(quantiles,levels,geometry='g0'):
    q=np.asarray(quantiles,float);levels=np.asarray(levels,float)
    if geometry not in GEOMETRIES or q.ndim!=2 or q.shape[1]!=len(levels) or not len(q):raise ValueError('Invalid quantile geometry')
    target_q,stop_q=GEOMETRIES[geometry]
    def column(value):
        match=np.flatnonzero(np.isclose(levels,value))
        if len(match)!=1:raise ValueError('Required native quantile absent')
        return q[:,match[0]]
    target,stop=column(target_q),column(stop_q)
    if not np.isfinite(target).all() or not np.isfinite(stop).all() or (stop<=0).any() or (target<stop).any():raise ValueError('Invalid/crossed decision quantiles')
    return dict(geometry=geometry,target_price=float(target.max()),stop_price=float(stop.min()),target_quantile=target_q,stop_quantile=stop_q)


def forecast_features(quantiles,levels,reference_close,horizon_minutes,geometry,decision_at,view,history=None):
    """Features use ONLY forecast and observations already known at the origin."""
    q=np.asarray(quantiles,float);levels=np.asarray(levels,float);reference_close=float(reference_close)
    if not np.isfinite(reference_close) or reference_close<=0:raise ValueError('Positive reference close required')
    def column(level):return q[:,np.flatnonzero(np.isclose(levels,level))[0]]/reference_close-1
    x={}
    for level in (.10,.25,.50,.75,.90):
        c=column(level);tag=str(round(level*100))
        x['terminal_q'+tag]=float(c[-1]);x['max_q'+tag]=float(c.max());x['min_q'+tag]=float(c.min())
    x.update(mean_width_10_90=float(np.mean(column(.9)-column(.1))),log_horizon=float(np.log1p(horizon_minutes)),
        target_fraction=geometry['target_price']/reference_close-1,stop_fraction=1-geometry['stop_price']/reference_close,
        geometry_target_quantile=geometry['target_quantile'],geometry_stop_quantile=geometry['stop_quantile'])
    at=pd.Timestamp(decision_at)
    if at.tzinfo is None:raise ValueError('Aware decision time required')
    hour=(at.hour+at.minute/60)*2*np.pi/24;day=at.dayofweek*2*np.pi/7
    x.update(hour_sin=float(np.sin(hour)),hour_cos=float(np.cos(hour)),dow_sin=float(np.sin(day)),dow_cos=float(np.cos(day)))
    for name in ('close','technical','fundamental'):x['view_'+name]=float(name in view or name=='close' and not any(v in view for v in ('technical','fundamental')))
    if history is None:history=np.array([reference_close])
    values=np.asarray(history,float);values=values[np.isfinite(values)&(values>0)]
    for lag in (4,24,96):x['observed_return_'+str(lag)]=float(values[-1]/values[-min(lag+1,len(values))]-1)
    x['observed_volatility']=float(np.std(np.diff(np.log(values[-97:])))) if len(values)>1 else 0.
    result=np.array(list(x.values()),dtype=np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite forecast features')
    return result,list(x)


def _network(size):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(size,32),torch.nn.ReLU(),torch.nn.Linear(32,16),
        torch.nn.ReLU(),torch.nn.Linear(16,3))


class FirstTouchHead:
    """Small supervised neural head; foundation quantiles are never treated as paths."""
    def __init__(self,network,metadata):
        self.network=network.eval();self.metadata=metadata;self.feature_names=metadata['feature_names']
        self.mean=np.asarray(metadata['train_mean']);self.scale=np.asarray(metadata['train_scale'])
        self.active=np.asarray(metadata.get('active_feature_mask',[True]*len(self.feature_names)),dtype=bool)
        if self.active.shape!=self.mean.shape or self.scale.shape!=self.mean.shape or not np.isfinite(self.scale).all() or (self.scale<=0).any():
            raise ValueError('Invalid first-touch preprocessing metadata')

    @classmethod
    def load(cls,path):
        import torch
        path=Path(path)
        if path.is_file():path=path.parent
        metadata=json.loads((path/'head.json').read_text(encoding='utf-8'))
        from .kronos import _sha256
        if _sha256(path/'head.pt')!=metadata['weights_sha256']:raise ValueError('First-touch weight hash differs')
        network=_network(len(metadata['feature_names']))
        network.load_state_dict(torch.load(path/'head.pt',map_location='cpu',weights_only=True))
        return cls(network,metadata)

    def predict_proba(self,x):
        import torch
        x=np.asarray(x,dtype=np.float32)
        if x.ndim==1:x=x[None,:]
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names) or not np.isfinite(x).all():raise ValueError('Invalid first-touch features')
        # Unobserved variation in TRAIN-constant columns must never reach network weights.
        normalized=np.zeros_like(x)
        normalized[:,self.active]=(x[:,self.active]-self.mean[self.active])/self.scale[self.active]
        with torch.inference_mode():
            return torch.softmax(self.network(torch.tensor(normalized,dtype=torch.float32)),dim=1).numpy()

    def timeout_return(self,horizon_minutes,view):
        means=self.metadata['train_timeout_means']
        return float(means.get(str(int(horizon_minutes))+'|'+view,means['pooled']))

    @classmethod
    def fit(cls,x,y,train_mask,validation_mask,feature_names,timeout_means,seed=42,epochs=400):
        import torch
        torch.manual_seed(seed);torch.set_num_threads(2)
        x=np.asarray(x,np.float32);y=np.asarray(y,np.int64);train_mask=np.asarray(train_mask,bool);validation_mask=np.asarray(validation_mask,bool)
        if train_mask.sum()<30 or validation_mask.sum()<10:raise ValueError('Insufficient known TRAIN or early-stop validation labels')
        if np.any(train_mask&validation_mask) or not np.isfinite(x).all():raise ValueError('Invalid training separation/features')
        training=x[train_mask].astype(np.float64);mean=training.mean(axis=0);active=np.ptp(training,axis=0)>0
        scale=np.where(active,np.maximum(training.std(axis=0),1e-6),1.)
        normalized=np.zeros_like(x);normalized[:,active]=(x[:,active]-mean[active])/scale[active]
        inputs=torch.tensor(normalized);labels=torch.tensor(y);model=_network(x.shape[1])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        train=torch.tensor(train_mask);val=torch.tensor(validation_mask);best=float('inf');best_state=None;stale=0;curve=[]
        for epoch in range(int(epochs)):
            model.train();optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(inputs[train]),labels[train]);loss.backward();optimizer.step()
            model.eval()
            with torch.inference_mode():score=float(torch.nn.functional.cross_entropy(model(inputs[val]),labels[val]))
            curve.append({'epoch':epoch,'train_cross_entropy':float(loss.detach()),'early_stop_cross_entropy':score})
            if score<best-1e-5:best=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
            else:stale+=1
            if stale>=35:break
        model.load_state_dict(best_state)
        metadata=dict(feature_names=feature_names,train_mean=mean.tolist(),train_scale=scale.tolist(),
            active_feature_mask=active.tolist(),preprocessing='TRAIN-only scaling; constant columns ignored at training and inference',
            train_timeout_means=timeout_means,class_names=list(CLASS_NAMES),architecture='Linear32/ReLU/Linear16/ReLU/Linear3',
            seed=seed,train_count=int(train_mask.sum()),early_stop_count=int(validation_mask.sum()),
            best_early_stop_cross_entropy=best,epochs_run=len(curve),curve=curve,
            probabilities='Uncalibrated neural softmax; inspect held-out reliability before interpreting as confidence')
        return cls(model,metadata)

    def save(self,path):
        import torch
        from .kronos import _sha256
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        torch.save(self.network.state_dict(),path/'head.pt');self.metadata['weights_sha256']=_sha256(path/'head.pt')
        (path/'head.json').write_text(json.dumps(self.metadata,indent=2,allow_nan=False),encoding='utf-8')


def probability_metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if not len(y):return {'rows':0}
    onehot=np.eye(3)[y];confusion=np.zeros((3,3),int)
    for actual,predicted in zip(y,p.argmax(1)):confusion[actual,predicted]+=1
    bins=[]
    for low,high in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
        selected=(p[:,0]>=low)&(p[:,0]<(high if high<1 else 1.000001))
        bins.append(dict(low=float(low),high=float(high),rows=int(selected.sum()),
            mean_p_tp=float(p[selected,0].mean()) if selected.any() else None,
            actual_tp_frequency=float((y[selected]==0).mean()) if selected.any() else None))
    return dict(rows=len(y),multiclass_brier=float(np.square(p-onehot).sum(1).mean()),
        log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
        accuracy=float((p.argmax(1)==y).mean()),confusion_actual_by_predicted=confusion.tolist(),tp_reliability=bins)
