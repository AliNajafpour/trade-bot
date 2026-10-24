"""Multi-scale candle history + arbitrary barrier queries, with competing risks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .barrier_dataset import HORIZONS, MAX_CONTEXT, history_inputs, barrier_query, sha256, write_json


def incidence(logits):
    """Return cumulative TP, cumulative SL, and remaining survival for every bin."""
    q=logits.softmax(-1);survival=q[...,2].cumprod(-1)
    previous=torch.cat([torch.ones_like(survival[...,:1]),survival[...,:-1]],dim=-1)
    events=(q[...,:2]*previous.unsqueeze(-1)).cumsum(-2)
    return torch.cat([events,survival.unsqueeze(-1)],dim=-1)


def risk_nll(logits,event_bin,cause,survived_bins,reduction='mean'):
    """Unknown cause contributes summed event mass; censoring contributes survival."""
    logq=logits.log_softmax(-1);log_s=logq[...,2]
    cumulative=torch.cat([torch.zeros_like(log_s[...,:1]),log_s.cumsum(-1)],dim=-1)
    j=event_bin.clamp(min=0).long()
    at_event=logq.gather(-2,j[...,None,None].expand(*j.shape,1,3)).squeeze(-2)
    known=at_event.gather(-1,cause.clamp(max=1).long().unsqueeze(-1)).squeeze(-1)
    event_term=torch.where(cause==2,torch.logsumexp(at_event[...,:2],dim=-1),known)
    event_ll=cumulative.gather(-1,j.unsqueeze(-1)).squeeze(-1)+event_term
    censor_ll=cumulative.gather(-1,survived_bins.long().unsqueeze(-1)).squeeze(-1)
    loss=-torch.where(event_bin>=0,event_ll,censor_ll)
    return loss.mean() if reduction=='mean' else loss


class MultiScaleSurvival(nn.Module):
    def __init__(self,symbols,context=360,width=64,layers=2,query_only=False,hierarchical=False):
        super().__init__()
        if context not in (180,360,720,1440,2880,10080) or width%4:raise ValueError('Unsupported architecture')
        if context>720 and not hierarchical and not query_only:raise ValueError('Long contexts require hierarchical aggregation')
        self.config=dict(symbols=symbols,context=context,width=width,layers=layers,query_only=query_only)
        if hierarchical:self.config['hierarchical']=True
        self.context=context;self.query_only=query_only
        self.layout=[(context,1),(context,5),(context,15)]
        if hierarchical:
            self.layout=[(min(context,720),1),(min(context,2880),5 if context<=1440 else 15),(context,15 if context<=1440 else 60)]
        self.asset=nn.Embedding(symbols,width)
        self.query=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width))
        self.bin=nn.Embedding(5,width)
        if not query_only:
            self.patch=nn.ModuleList([nn.Conv1d(10,width,kernel_size=k,stride=k) for k in (6,3,1)])
            self.scale=nn.Parameter(torch.randn(3,width)*.02)
            self.position=nn.Parameter(torch.randn(max(180,sum(span//scale//k for (span,scale),k in zip(self.layout,(6,3,1)))),width)*.02)
            layer=nn.TransformerEncoderLayer(width,4,width*3,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
            self.attention=nn.MultiheadAttention(width,4,dropout=.1,batch_first=True)
            self.norm=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,3))

    def forward(self,x,query,symbol):
        b,k,_=query.shape;asset=self.asset(symbol)
        queries=self.query(query)[:,:,None,:]+self.bin.weight[None,None,:,:]+asset[:,None,None,:]
        if not self.query_only:
            if x.shape[1]<self.context:raise ValueError('Insufficient candles for configured lookback')
            x=x[:,-self.context:];parts=[]
            for index,(span,scale) in enumerate(self.layout):
                grouped=x[:,-span:].reshape(b,span//scale,scale,10)
                pooled=torch.cat([grouped[:,:,0,0:1],grouped[:,:,:,1:2].amax(2),grouped[:,:,:,2:3].amin(2),
                    grouped[:,:,-1,3:4],grouped[:,:,:,4:].mean(2)],dim=-1)
                tokens=self.patch[index](pooled.transpose(1,2)).transpose(1,2)
                parts.append(tokens+self.scale[index])
            memory=torch.cat(parts,dim=1);memory=memory+self.position[None,:memory.shape[1]]+asset[:,None,:]
            memory=self.encoder(memory);q=queries.reshape(b,k*5,-1)
            attended=self.attention(q,memory,memory,need_weights=False)[0]
            queries=self.norm(q+attended).reshape(b,k,5,-1)
        return self.head(queries)


def mixture_nll(logits,event,cause,survive,temperature=1.):
    likelihood=torch.stack([-risk_nll(z/temperature,event,cause,survive,reduction='none') for z in logits])
    return -(torch.logsumexp(likelihood,dim=0)-np.log(len(logits))).mean()


def fit_temperature(logits,event,cause,survive):
    logits=[z.detach().float().cpu() for z in logits]
    event=event.cpu();cause=cause.cpu();survive=survive.cpu()
    log_t=torch.zeros((),requires_grad=True)
    optimizer=torch.optim.LBFGS([log_t],lr=.4,max_iter=40,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=mixture_nll(logits,event,cause,survive,log_t.exp().clamp(.25,8.))
        loss.backward();return loss
    optimizer.step(closure)
    result=float(log_t.detach().exp().clamp(.25,8.))
    with torch.no_grad():
        if mixture_nll(logits,event,cause,survive,result)>mixture_nll(logits,event,cause,survive):return 1.
    return result


class SurvivalPredictor:
    def __init__(self,models,metadata):
        temperatures=metadata.get('temperatures_by_symbol',{})
        if (not isinstance(temperatures,dict) or set(temperatures)-set(metadata['symbols'])
                or any(not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0
                       for value in [metadata['temperature'],*temperatures.values()])):
            raise ValueError('Invalid calibration temperature metadata')
        self.models=[model.eval().cpu() for model in models];self.metadata=metadata

    def calibrated_probabilities(self,logits,assets):
        if logits.ndim!=4 or assets.shape!=(len(logits),):raise ValueError('Invalid calibrated batch shape')
        names=[self.metadata['symbols'][index] for index in assets.detach().cpu().tolist()]
        temperature=torch.tensor([self.metadata.get('temperatures_by_symbol',{}).get(name,self.metadata['temperature'])
            for name in names],device=logits.device,dtype=logits.dtype).reshape(-1,1,1,1)
        return incidence(logits/temperature)

    @staticmethod
    def save(path,models,metadata):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        if (path/'model.json').exists():raise FileExistsError('Model already saved')
        metadata=dict(metadata);metadata['members']=[]
        for index,model in enumerate(models):
            name=f'member-{index}.pt'
            if (path/name).exists():raise FileExistsError(name)
            torch.save(model.state_dict(),path/name)
            metadata['members'].append(dict(file=name,sha256=sha256(path/name),config=model.config))
        metadata['minimum_history_minutes']=metadata.get('minimum_history_minutes',MAX_CONTEXT)
        write_json(path/'model.json',metadata)

    @classmethod
    def load(cls,path):
        path=Path(path);metadata=json.loads((path/'model.json').read_text(encoding='utf-8'));models=[]
        for member in metadata['members']:
            if sha256(path/member['file'])!=member['sha256']:raise ValueError('Model weight hash mismatch')
            model=MultiScaleSurvival(**member['config'])
            model.load_state_dict(torch.load(path/member['file'],map_location='cpu',weights_only=True));models.append(model)
        return cls(models,metadata)

    def predict(self,history,symbol,decision_at,entry,*,tp=None,sl=None,tp_fraction=None,sl_fraction=None,side='long'):
        if symbol not in self.metadata['symbols']:raise ValueError('Unknown symbol')
        decision=pd.Timestamp(decision_at)
        if decision.tzinfo is None or decision.value%60_000_000_000:raise ValueError('Aware minute-boundary decision required')
        needed=self.metadata['minimum_history_minutes']
        if len(history)<needed:raise ValueError(f'{needed} completed history minutes required')
        # Validate all supplied rows; a future last candle must not be silently trimmed away.
        from .features import _validated_frame
        _,times,_,interval=_validated_frame(history)
        if interval!=60 or times[-1]+pd.Timedelta(minutes=1)!=decision:raise ValueError('History must end exactly at the decision boundary')
        if side not in ('long','short'):raise ValueError('Invalid side')
        if not np.isfinite(entry) or entry<=0:raise ValueError('Invalid entry')
        sign=1 if side=='long' else -1
        if (tp is None)==(tp_fraction is None) or (sl is None)==(sl_fraction is None):raise ValueError('Specify each barrier once as price or fraction')
        if tp_fraction is not None:
            if not np.isfinite(tp_fraction) or tp_fraction<=0:raise ValueError('Invalid TP fraction')
            tp=entry*(1+sign*tp_fraction)
        if sl_fraction is not None:
            if not np.isfinite(sl_fraction) or sl_fraction<=0:raise ValueError('Invalid SL fraction')
            sl=entry*(1-sign*sl_fraction)
        version=self.metadata.get('feature_version','relative-v1')
        seq,sigma,ref=history_inputs(history.iloc[-needed:],version=version)
        q=barrier_query(entry,tp,sl,sigma,ref,side)
        if q.shape!=(1,8):raise ValueError('One barrier pair per request required')
        x=torch.from_numpy(seq[None]);query=torch.from_numpy(q[None]);asset=torch.tensor([self.metadata['symbols'].index(symbol)])
        with torch.inference_mode():
            probabilities=torch.stack([self.calibrated_probabilities(model(x,query,asset),asset)[0,0] for model in self.models]).numpy()
        mean=probabilities.mean(0);std=probabilities.std(0);forecasts=[]
        for index,horizon in enumerate(HORIZONS):
            tp_p,sl_p,survival=[float(v) for v in mean[index]];touch=tp_p+sl_p
            forecasts.append(dict(horizon_minutes=int(horizon),p_tp_first=tp_p,p_sl_first=sl_p,p_no_touch=survival,
                p_tp_given_touch=tp_p/touch if touch>1e-6 else None,
                member_std_tp=float(std[index,0]),member_std_sl=float(std[index,1])))
        distances=np.abs(np.log(np.array([tp,sl])/entry));support=self.metadata['geometry_support'];flags=[]
        distance_bounds=np.log1p([support['min_fraction'],support['max_fraction']])
        if (distances<distance_bounds[0]-1e-8).any() or (distances>distance_bounds[1]+1e-8).any():flags.append('barrier_distance_outside_training_range')
        ratio=distances[0]/distances[1]
        max_ratio=support.get('max_ratio',3.)
        if ratio<1/max_ratio or ratio>max_ratio:flags.append('barrier_ratio_outside_training_range')
        local_bounds=np.clip(np.array([.4,3.])*sigma*np.sqrt(60.),.0005,.12)
        if version=='relative-v1' and ((distances<local_bounds[0]-1e-8).any() or (distances>local_bounds[1]+1e-8).any()):flags.append('volatility_normalized_distance_outside_training_grid')
        if abs(np.log(entry/ref))>.5*sigma+1e-8:flags.append('entry_gap_outside_training_range')
        return dict(symbol=symbol,side=side,decision_at=decision.tz_convert('UTC').isoformat(),
            history_last_open=times[-1].isoformat(),entry=float(entry),tp=float(tp),sl=float(sl),
            semantics='Price-touch probabilities, not fill probabilities. Horizons never trigger liquidation.',
            support_flags=flags,forecasts=forecasts,protocol_sha256=self.metadata['protocol_sha256'],
            dataset_manifest_sha256=self.metadata['dataset_manifest_sha256'],
            model_hashes=[m['sha256'] for m in self.metadata['members']])


def main():
    import argparse
    from .data import load_market
    parser=argparse.ArgumentParser(description='Predict arbitrary barrier price-touch probabilities from completed M1 candles.')
    parser.add_argument('--model',required=True);parser.add_argument('--history',required=True)
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();target=Path(args.output)
    if target.exists():raise FileExistsError('Preserve previous prediction')
    request=json.loads(Path(args.request).read_text(encoding='utf-8'))
    torch.set_num_threads(2)
    result=SurvivalPredictor.load(args.model).predict(load_market(args.history,60),**request)
    target.parent.mkdir(parents=True,exist_ok=True);write_json(target,result)
    print(str(target.resolve()))


if __name__=='__main__':main()
