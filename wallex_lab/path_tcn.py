"""Small causal TCN trained on future path extrema with ordered quantile pinball loss."""
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .neural import _array,_fit_scaler,_transform
from .kronos import _sha256

QUANTILES=(.5,.75,.9)
HORIZONS=(1,3,6,12,24)
TARGETS=('log_mfe','log_mae','log_terminal')


class _PathNet(nn.Module):
    def __init__(self,channels,width):
        super().__init__();self.projection=nn.Conv1d(channels,width,1)
        self.layers=nn.ModuleList(nn.Conv1d(width,width,3,dilation=2**i) for i in range(7))
        self.head=nn.Linear(width,45)

    def forward(self,x):
        x=self.projection(x.transpose(1,2))
        for layer in self.layers:
            x=x+F.dropout(F.gelu(layer(F.pad(x,(2*layer.dilation[0],0)))),p=.05,training=self.training)
        raw=self.head(x[:,:,-1]).reshape(-1,5,3,3)
        return torch.cat((raw[...,:1],raw[...,:1]+torch.cumsum(F.softplus(raw[...,1:]),dim=-1)),dim=-1)


def quantile_metrics(y,prediction,target_mask=None):
    y=np.asarray(y,np.float32);prediction=_array(prediction,4,'Quantile predictions')
    if prediction.shape!=y.shape+(3,):raise ValueError('Quantile target shapes differ')
    known=np.isfinite(y).all(axis=-1) if target_mask is None else np.asarray(target_mask,bool)
    if known.shape!=y.shape[:2] or not np.isfinite(y[known]).all():raise ValueError('Invalid target availability mask')
    error=y[...,None]-prediction;q=np.asarray(QUANTILES)
    loss=np.maximum(q*error,(q-1)*error);pinball=[];coverage=[];mae=[]
    for i in range(5):
        selected=known[:,i]
        pinball.append(loss[selected,i].mean(axis=0).tolist() if selected.any() else None)
        coverage.append((y[selected,i,:,None]<=prediction[selected,i]).mean(axis=0).tolist() if selected.any() else None)
        mae.append(np.abs(y[selected,i]-prediction[selected,i,:,0]).mean(axis=0).tolist() if selected.any() else None)
    return dict(rows=len(y),known_by_horizon=known.sum(axis=0).tolist(),mean_pinball=float(loss[known].mean()) if known.any() else None,
        pinball_by_horizon_target_quantile=pinball,coverage_by_horizon_target_quantile=coverage,
        median_absolute_error_by_horizon_target=mae,
        quantile_crossings=int((np.diff(prediction,axis=-1)<0).sum()))


class PathTCN:
    def __init__(self,network,metadata,device):
        self.network=network.eval();self.metadata=metadata;self.device=torch.device(device)

    @staticmethod
    def _normalized(x,scaler):
        transformed=_transform(x,scaler)
        transformed[...,~np.asarray(scaler['active'],bool)]=0
        return transformed

    def predict_quantiles(self,x,batch_size=256):
        x=_array(x,3,'Causal sequences')
        if tuple(x.shape[1:])!=tuple(self.metadata['sequence_shape']):raise ValueError('Causal sequence shape differs from TRAIN')
        if type(batch_size) is not int or not 1<=batch_size<=1024:raise ValueError('Batch size must be1..1024')
        values=self._normalized(x,self.metadata['sequence_scaler']);parts=[]
        self.network.eval()
        with torch.inference_mode():
            for start in range(0,len(values),batch_size):
                parts.append(self.network(torch.from_numpy(values[start:start+batch_size]).to(self.device)).cpu().numpy())
        raw=np.concatenate(parts) if parts else np.empty((0,5,3,3),np.float32)
        return (raw*np.asarray(self.metadata['target_scale'])[None,...,None]+np.asarray(self.metadata['target_mean'])[None,...,None]).astype(np.float32)

    @classmethod
    def fit(cls,x,y,fit_mask,early_stop_mask,width=32,epochs=30,patience=5,device='cuda',seed=17,target_mask=None):
        began=time.perf_counter();x=_array(x,3,'Causal sequences');y=np.asarray(y,np.float32)
        fit=np.asarray(fit_mask,bool);stop=np.asarray(early_stop_mask,bool)
        if y.shape!=(len(x),5,3) or fit.shape!=(len(x),) or stop.shape!=fit.shape or np.any(fit&stop):raise ValueError('Invalid causal fitting masks/targets')
        known=np.isfinite(y).all(axis=2) if target_mask is None else np.asarray(target_mask,bool)
        if known.shape!=(len(x),5) or not np.isfinite(y[known]).all():raise ValueError('Invalid target availability mask')
        fit=fit&known.any(axis=1);stop=stop&known.any(axis=1)
        if fit.sum()<30 or stop.sum()<10:raise ValueError('Insufficient base TRAIN or inner TRAIN early-stop rows')
        if width not in (32,64) or type(epochs) is not int or not 1<=epochs<=40 or type(patience) is not int or not 1<=patience<=5:raise ValueError('Predeclared TCN bounds exceeded')
        device=torch.device(device)
        if device.type not in ('cpu','cuda'):raise ValueError('CPU or CUDA required')
        torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(2)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        scaler=_fit_scaler(x[fit]);flat=x[fit].reshape(-1,x.shape[-1])
        scaler['active']=(np.ptp(flat,axis=0)>0).tolist()
        normalized=cls._normalized(x,scaler)
        if not np.all(known[fit].any(axis=0)) or not np.all(known[stop].any(axis=0)):raise ValueError('Every horizon needs known fit/inner-stop outcomes')
        mean=np.stack([y[fit&known[:,i],i].mean(axis=0,dtype=np.float64) for i in range(5)])
        scale=np.maximum(np.stack([y[fit&known[:,i],i].std(axis=0,dtype=np.float64) for i in range(5)]),1e-6)
        target=np.where(known[...,None],(y-mean)/scale,0).astype(np.float32)
        network=_PathNet(x.shape[-1],width).to(device);optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.0001)
        train_indices=np.flatnonzero(fit);stop_indices=np.flatnonzero(stop);generator=np.random.default_rng(seed)
        levels=torch.tensor(QUANTILES,device=device);best=float('inf');best_state=None;stale=0;history=[];batch_size=256
        def loss_for(indices,training=False):
            total=0.
            for offset in range(0,len(indices),batch_size):
                rows=indices[offset:offset+batch_size];batch=torch.from_numpy(normalized[rows]).to(device);truth=torch.from_numpy(target[rows]).to(device)
                if training:optimizer.zero_grad(set_to_none=True)
                difference=truth[...,None]-network(batch);availability=torch.from_numpy(known[rows]).to(device)
                loss=(torch.maximum(levels*difference,(levels-1)*difference)*availability[...,None,None]).sum()/(availability.sum()*9)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TCN pinball loss')
                if training:loss.backward();nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                total+=float(loss.detach())*len(rows)
            return total/len(indices)
        for epoch in range(1,epochs+1):
            network.train();training_loss=loss_for(generator.permutation(train_indices),True);network.eval()
            with torch.inference_mode():stopping_loss=loss_for(stop_indices)
            history.append(dict(epoch=epoch,train_pinball_scaled=training_loss,inner_train_pinball_scaled=stopping_loss))
            print(json.dumps(dict(model='tcn'+str(width),**history[-1])),flush=True)
            if stopping_loss<best-1e-5:
                best=stopping_loss;best_state={k:v.detach().cpu().clone() for k,v in network.state_dict().items()};stale=0;best_epoch=epoch
            else:stale+=1
            if stale>=patience:break
        network.load_state_dict(best_state)
        metadata=dict(format_version=1,model='causal_tcn_quantiles',width=width,sequence_shape=list(x.shape[1:]),
            quantile_levels=list(QUANTILES),horizons_hours=list(HORIZONS),targets=list(TARGETS),sequence_scaler=scaler,
            target_mean=mean.tolist(),target_scale=scale.tolist(),fit_rows=int(fit.sum()),early_stop_rows=int(stop.sum()),
            known_fit_by_horizon=known[fit].sum(axis=0).tolist(),known_inner_stop_by_horizon=known[stop].sum(axis=0).tolist(),
            seed=seed,history=history,best_epoch=best_epoch,best_inner_train_pinball_scaled=best,parameters=sum(p.numel() for p in network.parameters()),
            device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            runtime_seconds=time.perf_counter()-began,torch_version=str(torch.__version__),numpy_version=np.__version__,
            output_semantics='Marginal ordered quantiles of actual future M1 path extrema/terminal; not joint simulated paths or first-touch probabilities')
        return cls(network,metadata,device)

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError('Preserve trained artifacts; choose a new directory')
        path.mkdir(parents=True)
        torch.save({k:v.detach().cpu() for k,v in self.network.state_dict().items()},path/'weights.pt')
        self.metadata['weights_sha256']=_sha256(path/'weights.pt')
        (path/'model.json').write_text(json.dumps(self.metadata,indent=2,allow_nan=False),encoding='utf-8')

    @classmethod
    def load(cls,path,device='cpu'):
        path=Path(path);metadata=json.loads((path/'model.json').read_text(encoding='utf-8'))
        if metadata['format_version']!=1 or _sha256(path/'weights.pt')!=metadata['weights_sha256']:raise ValueError('TCN artifact hash/version differs')
        device=torch.device(device);network=_PathNet(metadata['sequence_shape'][1],metadata['width']).to(device)
        network.load_state_dict(torch.load(path/'weights.pt',map_location=device,weights_only=True))
        return cls(network,metadata,device)
