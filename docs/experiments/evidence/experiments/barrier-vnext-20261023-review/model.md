# Task3 added-file review package, no Git repository

## wallex_lab/barrier_vnext_model.py
```python
"""Isolated multiscale barrier variants and shared-encoder historical reconstruction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .barrier_survival import MultiScaleSurvival


def historical_mask(batch,context,generator,device='cpu'):
    """Hide up to three contiguous physical intervals, reproducibly on the CPU."""
    if batch<1 or context<2:raise ValueError('Positive batch and at least two minutes required')
    mask=torch.zeros(batch,context,dtype=torch.bool)
    for row in range(batch):
        for _ in range(3 if context>=12 else 1):
            length=int(torch.randint(max(1,context//48),max(2,context//12+1),(1,),generator=generator))
            start=int(torch.randint(context-length+1,(1,),generator=generator))
            mask[row,start:start+length]=True
    return mask.to(device)


class NextBarrierModel(MultiScaleSurvival):
    def __init__(self,symbols,context=10080,width=96,layers=3,activity=False,age=False,level_attention=False,local=False):
        # Construct every shared baseline layer before optional layers consume RNG.
        super().__init__(symbols,context,width,layers,hierarchical=True)
        self.activity=activity;self.age=age;self.level_attention=level_attention;self.local=local
        self.channels=14 if activity else 10
        self.config=dict(symbols=symbols,context=context,width=width,layers=layers,
                         activity=activity,age=age,level_attention=level_attention,local=local)
        if activity:
            for index,original in enumerate(self.patch):
                extended=nn.Conv1d(14,width,original.kernel_size,stride=original.stride)
                with torch.no_grad():
                    extended.weight[:,:10].copy_(original.weight);extended.weight[:,10:].zero_()
                    extended.bias.copy_(original.bias)
                self.patch[index]=extended
        if age:self.age_projection=nn.Linear(2,width,bias=False)
        if level_attention:self.level_strength=nn.Parameter(torch.zeros(2))
        if local:self.local_encoder=nn.Sequential(nn.Conv1d(self.channels,width,3,padding=1),nn.GELU(),nn.Conv1d(width,width,3,padding=1))
        self.mask_token=nn.Parameter(torch.zeros(width))
        self.reconstruction=nn.Linear(width,3)

    def _history(self,x):
        if x.ndim!=3 or x.shape[1]<self.context or x.shape[2]<self.channels:
            raise ValueError('Insufficient history or channels for configured architecture')
        return x[:,-self.context:,:self.channels]

    def _mask(self,mask,x):
        if mask.dtype!=torch.bool or mask.shape!=x.shape[:2] or mask.device!=x.device:
            raise ValueError('Mask must be boolean B,context on the history device')
        return mask

    @staticmethod
    def physical_levels(query):
        """TP and SL log prices relative to reference in history coordinate units."""
        sigma=query[...,5].exp()/1e4
        entry=query[...,6]/100
        sign=query[...,7]
        return torch.stack([entry+sign*query[...,0]/100,entry-sign*query[...,1]/100],-1)/(sigma[...,None]*math.sqrt(60))

    def level_bias(self,query,geometry):
        """Continuous bias by TP/SL distance to each token's physical low/high range."""
        levels=self.physical_levels(query)[:,:,:,None]
        low=geometry[:,None,None,:,0];high=geometry[:,None,None,:,1]
        distance=F.relu(low-levels)+F.relu(levels-high)
        return -(torch.log1p(distance)*F.softplus(self.level_strength)[None,None,:,None]).sum(2)

    def encode_history(self,x,symbol,mask=None):
        """The same encoder feeds supervised attention and masked reconstruction."""
        x=self._history(x);b=x.shape[0]
        if mask is not None:x=x.masked_fill(self._mask(mask,x)[...,None],0)
        parts=[];geometry=[]
        for index,((span,scale),patch) in enumerate(zip(self.layout,self.patch)):
            grouped=x[:,-span:].reshape(b,span//scale,scale,self.channels)
            pooled=torch.cat([grouped[:,:,0,0:1],grouped[:,:,:,1:2].amax(2),grouped[:,:,:,2:3].amin(2),
                              grouped[:,:,-1,3:4],grouped[:,:,:,4:].mean(2)],dim=-1)
            tokens=patch(pooled.transpose(1,2)).transpose(1,2)+self.scale[index]
            duration=scale*patch.kernel_size[0]
            physical=x[:,-span:].reshape(b,span//duration,duration,self.channels)
            geometry.append(torch.stack([physical[:,:,:,2].amin(2),physical[:,:,:,1].amax(2)],-1))
            if self.age:
                age=span-(torch.arange(tokens.shape[1],device=x.device,dtype=x.dtype)+.5)*duration
                clock=torch.stack([age.log1p(),torch.full_like(age,math.log1p(duration))],-1)/math.log1p(self.context)
                tokens=tokens+self.age_projection(clock)[None]
            if mask is not None:
                hidden=mask[:,-span:].reshape(b,span//duration,duration).any(-1)
                tokens=tokens+hidden[...,None]*self.mask_token
            parts.append(tokens)
        memory=torch.cat(parts,1)
        memory=memory+self.position[None,:memory.shape[1]]+self.asset(symbol)[:,None,:]
        memory=self.encoder(memory)
        if self.local:memory=memory+self.local_encoder(x[:,-60:].transpose(1,2)).mean(-1)[:,None,:]
        return memory,torch.cat(geometry,1)

    def forward(self,x,query,symbol):
        x=self._history(x)
        if query.ndim!=3 or query.shape[0]!=x.shape[0] or query.shape[-1]!=8:
            raise ValueError('Expected B,K,8 barrier queries')
        if not (self.activity or self.age or self.level_attention or self.local):
            return super().forward(x,query,symbol)
        b,k,_=query.shape
        queries=self.query(query)[:,:,None,:]+self.bin.weight[None,None,:,:]+self.asset(symbol)[:,None,None,:]
        memory,geometry=self.encode_history(x,symbol)
        q=queries.reshape(b,k*5,-1);bias=None
        if self.level_attention:
            bias=self.level_bias(query,geometry).repeat_interleave(5,1)
            bias=bias[:,None].expand(-1,self.attention.num_heads,-1,-1).reshape(b*self.attention.num_heads,k*5,-1)
        attended=self.attention(q,memory,memory,attn_mask=bias,need_weights=False)[0]
        return self.head(self.norm(q+attended).reshape(b,k,5,-1))

    def reconstruction_targets(self,x,mask):
        """Only hidden-minute return/range/volume means contribute to each target."""
        x=self._history(x);mask=self._mask(mask,x);b=x.shape[0];targets=[];selected=[]
        for (span,scale),patch in zip(self.layout,self.patch):
            duration=scale*patch.kernel_size[0]
            hidden=mask[:,-span:].reshape(b,span//duration,duration)
            values=x[:,-span:,[7,6,4]].reshape(b,span//duration,duration,3)
            summary=values.masked_fill(~hidden[...,None],0).sum(2)/hidden.sum(2).clamp_min(1)[...,None]
            targets.append(summary);selected.append(hidden.any(-1))
        return torch.cat(targets,1).detach(),torch.cat(selected,1)

    def pretrain_loss(self,x,symbol,mask):
        target,selected=self.reconstruction_targets(x,mask)
        if not selected.any():raise ValueError('Pretraining requires a hidden historical interval')
        memory,_=self.encode_history(x,symbol,mask)
        return F.mse_loss(self.reconstruction(memory)[selected],target[selected])

```

## checks/check_barrier_vnext_model.py
```python
"""Run python -B -m checks.check_barrier_vnext_model; synthetic CPU inputs only."""
import importlib.util
import io
import numpy as np
import torch
from wallex_lab.barrier_dataset import barrier_query
from wallex_lab.barrier_survival import MultiScaleSurvival, incidence, risk_nll


def main():
    assert importlib.util.find_spec('wallex_lab.barrier_vnext_model'), 'Missing vNext model'
    from wallex_lab.barrier_vnext_model import NextBarrierModel, historical_mask
    torch.set_num_threads(2)
    torch.manual_seed(41)
    x=torch.randn(2,720,14);x[:,:,1]=x[:,:,:4].amax(-1);x[:,:,2]=x[:,:,:4].amin(-1)
    symbol=torch.tensor([0,1])
    q=torch.from_numpy(np.stack([np.concatenate([
        barrier_query(100.2,[101.2],[99.2],.002,100.),
        barrier_query(100.2,[99.2],[101.2],.002,100.,'short')])]*2))
    options=dict(symbols=2,context=720,width=16,layers=1)
    torch.manual_seed(11);old=MultiScaleSurvival(**options,hierarchical=True).eval()
    torch.manual_seed(11);base=NextBarrierModel(**options).eval()
    torch.testing.assert_close(base(x,q,symbol),old(x[:,:,:10],q,symbol),rtol=0,atol=0)
    masks=historical_mask(2,720,torch.Generator().manual_seed(42))
    assert masks.dtype==torch.bool and masks.shape==(2,720)
    assert masks.any(1).all() and (~masks).any(1).all()
    assert ((masks[:,1:]!=masks[:,:-1]).sum(1)<=6).all(), 'Masks are not physical intervals'
    assert torch.equal(masks,historical_mask(2,720,torch.Generator().manual_seed(42)))
    geometry=base.physical_levels(q)
    # Actual TP/SL divided by reference, in history's sigma*sqrt(60) units.
    expected=torch.tensor([[np.log(1.012),np.log(.992)],[np.log(.992),np.log(1.012)]])/(.002*np.sqrt(60))
    torch.testing.assert_close(geometry[0],expected.float(),atol=2e-6,rtol=2e-6)
    toy=torch.zeros(2,720,14);clock=torch.arange(720.)
    toy[:,:,7]=clock;toy[:,:,6]=clock*2;toy[:,:,4]=clock*3
    partial=torch.zeros_like(masks);partial[:,6:8]=True
    target,selected=base.reconstruction_targets(toy,partial)
    assert selected.sum(1).tolist()==[3,3], 'Every overlapping scale must have a reconstruction target'
    assert selected[0].nonzero().flatten().tolist()==[1,120,168], 'Physical patch alignment changed'
    torch.testing.assert_close(target[selected],torch.tensor([[6.5,13.,19.5]]).expand(6,-1))
    for flag in ({},{'activity':True},{'age':True},{'level_attention':True},{'local':True},
                 dict(activity=True,age=True,level_attention=True,local=True)):
        torch.manual_seed(11);model=NextBarrierModel(**options,**flag).eval()
        for name,param in old.named_parameters():
            # Activity extends the input convolution without changing its original weights.
            actual=dict(model.named_parameters())[name]
            if name.startswith('patch.') and name.endswith('weight'):actual=actual[:,:10]
            torch.testing.assert_close(actual,param,rtol=0,atol=0)
        logits=model(x,q,symbol)
        assert logits.shape==(2,2,5,3) and torch.isfinite(logits).all()
        p=incidence(logits)
        torch.testing.assert_close(p.sum(-1),torch.ones(2,2,5))
        assert (p[...,:2].diff(dim=-2)>=-1e-6).all() and (p[...,2].diff(dim=-1)<=1e-6).all()
        if not flag.get('activity'):
            other=x.clone();other[:,:,10:]+=100
            torch.testing.assert_close(logits,model(other,q,symbol),rtol=0,atol=0)
        leak_mask=masks.clone();leak_mask[:,-30:]=True
        changed=x.clone();changed[leak_mask]+=1000
        memory,_=model.encode_history(x,symbol,leak_mask)
        memory2,_=model.encode_history(changed,symbol,leak_mask)
        torch.testing.assert_close(memory,memory2,rtol=0,atol=0)
        target,selected=model.reconstruction_targets(x,masks)
        # An unmasked future minute within a partially hidden patch cannot alter its target.
        partial=torch.zeros_like(masks);partial[:,6:8]=True
        target,_=model.reconstruction_targets(x,partial)
        other=x.clone();other[:,8:]+=999
        other_target,_=model.reconstruction_targets(other,partial)
        torch.testing.assert_close(target,other_target,rtol=0,atol=0)
        assert selected.any() and selected.shape==memory.shape[:2]
        model.zero_grad();loss=model.pretrain_loss(x,symbol,masks)
        assert loss.ndim==0 and torch.isfinite(loss)
        loss.backward()
        assert model.patch[0].weight.grad.abs().sum()>0, 'Pretraining bypassed history convolution'
        if flag.get('activity'):
            assert model.patch[0].weight.grad[:,10:].abs().sum()>0, 'Activity inputs cannot learn'
        if flag.get('age'):
            assert model.age_projection.weight.grad.abs().sum()>0, 'Age inputs cannot learn'
        if flag.get('local'):
            assert model.local_encoder[0].weight.grad.abs().sum()>0, 'Pretraining bypassed local branch'
        parameter=model.encoder.layers[0].linear1.weight
        assert parameter.grad.abs().sum()>0, 'Pretraining bypassed shared transformer'
        before=parameter.detach().clone()
        torch.optim.SGD(model.parameters(),lr=.01).step()
        assert not torch.equal(before,parameter), 'Pretraining did not update encoder'
        state=io.BytesIO();torch.save(dict(config=model.config,weights=model.state_dict()),state);state.seek(0)
        saved=torch.load(state,weights_only=True);restored=NextBarrierModel(**saved['config']).eval()
        restored.load_state_dict(saved['weights'])
        torch.testing.assert_close(model(x,q,symbol),restored(x,q,symbol),rtol=0,atol=0)
        torch.testing.assert_close(model.pretrain_loss(x,symbol,masks),restored.pretrain_loss(x,symbol,masks),rtol=0,atol=0)
        model.zero_grad();z=torch.zeros(2,2,dtype=torch.long)
        risk_nll(model(x,q,symbol),z,z,z).backward()
        assert parameter.grad.abs().sum()>0, 'Downstream encoder differs from pretrained encoder'
        if flag.get('level_attention'):
            assert model.level_strength.grad.abs().sum()>0, 'Physical level bias cannot learn'
    level=NextBarrierModel(**options,level_attention=True).eval()
    near=geometry[:,:1,:1].expand(-1,-1,2)
    far=near+10
    assert (level.level_bias(q[:,:1],near)>level.level_bias(q[:,:1],far)).all(), 'Level proximity bias reversed'
    print('PASS vNext baseline parity, independent flags, physical geometry, coherent incidence, interval masks, leak safety, shared gradients and saved weights')


if __name__=='__main__':main()

```
