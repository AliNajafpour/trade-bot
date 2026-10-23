"""Fixed HGB/GRU/Kronos horizon forecasts, with no VAL/TEST model selection.

Reuses trees.TrainClip, neural._model_inputs/fit_predict and the pinned genuine
Kronos encoder adapter. All learned transforms fit purged TRAIN rows only.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import features, kronos, neural, trees
from .features import _validated_frame, forward_returns
from .trees import TrainClip

_ENCODER = None


def _encoder_features(frame, directory, device):
    """One bounded in-memory encoder; immutable hash-checked per-frame disk cache."""
    global _ENCODER
    indices = kronos._sample_indices(frame,64,1,20000)
    model_dir, tokenizer_dir, provenance = kronos._verify_assets({})
    bars,times,_,interval = _validated_frame(frame)
    digest=hashlib.sha256(json.dumps(dict(context=64,interval_seconds=interval,
        provenance=provenance,adapter=kronos._sha256(kronos.__file__),
        helper=kronos._sha256(__file__)),sort_keys=True).encode())
    for values in (bars.to_numpy(dtype=np.float64),times.as_unit('ns').asi8,indices):
        digest.update(np.ascontiguousarray(values).tobytes())
    key=digest.hexdigest()
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    cache=directory/f'{key}.npz'
    receipt=cache.with_suffix('.json')
    hit=cache.is_file()
    if hit:
        if not receipt.is_file() or json.loads(receipt.read_text())['sha256']!=kronos._sha256(cache):
            raise ValueError('Kronos horizon cache hash mismatch')
        with np.load(cache,allow_pickle=False) as saved:
            if not np.array_equal(saved['indices'],indices):
                raise ValueError('Kronos horizon cache indices mismatch')
            latent=saved['latent'].copy()
    else:
        encoder_key=(str(model_dir),str(tokenizer_dir),device,json.dumps(provenance['verified_files'],sort_keys=True))
        if _ENCODER is None or _ENCODER[0]!=encoder_key:
            _ENCODER=None
            if str(device).startswith('cuda'):
                import torch
                torch.cuda.empty_cache()
            model,tokenizer=kronos._load_encoder(model_dir,tokenizer_dir,device)
            _ENCODER=(encoder_key,model,tokenizer)
        latent=kronos._encode_windows(frame,indices,64,16,device,_ENCODER[1],_ENCODER[2])
        np.savez_compressed(cache,indices=indices,latent=latent)
        receipt.write_text(json.dumps(dict(cache_key=key,sha256=kronos._sha256(cache),provenance=provenance),indent=2),encoding='utf-8')
    width=json.loads((model_dir/'config.json').read_text())['d_model']
    if latent.shape!=(len(indices),width) or not np.isfinite(latent).all():
        raise ValueError('Invalid horizon encoder hidden states')
    return indices,latent,dict(cache_hit=hit,cache_path=str(cache),cache_key=key,
                              context=64,stride=1,batch_size=16,provenance=provenance)


def fit_forecasters(frame,horizon,train_end,val_end,run_dir,device='cuda',encoder_cache_dir=None):
    """Return length-N gross fractional forecasts and persisted fixed-model fits.

    NaN means unsupported warmup/gap history. Pre-TRAIN-end predictions are
    retrospective diagnostics. VAL/TEST labels never reach any learned model.
    GRU early stopping uses a separately purged final 20% of TRAIN only.
    """
    import torch
    n=len(frame)
    for name,value in (('horizon',horizon),('train_end',train_end),('val_end',val_end)):
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    if not 0<train_end<val_end<n or n>20000:
        raise ValueError('Require chronological nonempty splits and at most 20000 bars')
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'cpu'
    if not str(device).startswith(('cpu','cuda')):
        raise ValueError('Only cpu/cuda devices supported')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    directory=Path(run_dir)
    directory.mkdir(parents=True,exist_ok=True)
    source_hashes={str(Path(path).name):kronos._sha256(path)
                   for path in (__file__,features.__file__,neural.__file__,kronos.__file__,trees.__file__)}
    bars,times,_,interval=_validated_frame(frame)
    identity=hashlib.sha256(json.dumps(dict(horizon=int(horizon),train_end=int(train_end),
        val_end=int(val_end),device=str(device),interval_seconds=interval,sources=source_hashes),sort_keys=True).encode())
    for values in (bars.to_numpy(dtype=np.float64),times.as_unit('ns').asi8):
        identity.update(np.ascontiguousarray(values).tobytes())
    input_key=identity.hexdigest()
    saved_meta=directory/'metadata.json'
    if saved_meta.is_file():
        saved=json.loads(saved_meta.read_text(encoding='utf-8'))
        if saved.get('input_key')==input_key:
            for name,path in saved['artifacts'].items():
                if not Path(path).is_file() or kronos._sha256(path)!=saved['artifact_sha256'][name]:
                    raise ValueError(f'Completed forecast artifact hash mismatch: {path}')
            with np.load(directory/'forecasts.npz',allow_pickle=False) as stored:
                predictions={name:stored[name].copy() for name in ('hgb','gru','kronos')}
            if any(values.shape!=(n,) or np.isinf(values).any() for values in predictions.values()):
                raise ValueError('Completed forecast arrays are invalid')
            saved['resume_hit']=True
            return dict(predictions=predictions,metadata=saved)
    X,sequences,valid,columns=neural._model_inputs(frame,16)
    labels=np.full(n,np.nan)
    labels[:train_end]=forward_returns(frame.iloc[:train_end],int(horizon)).to_numpy()
    rows=np.arange(n)
    train=rows[valid & np.isfinite(labels) & (rows+horizon<train_end)]
    inner=int(train_end*.8)
    fit=rows[valid & np.isfinite(labels) & (rows+horizon<inner)][-12000:]
    holdout=rows[valid & np.isfinite(labels) & (rows>=inner) & (rows+horizon<train_end)]
    if len(train)<32 or len(fit)<24 or len(holdout)<8:
        raise ValueError('Insufficient purged TRAIN/inner holdout history')
    hgb=Pipeline([('clip',TrainClip()),('scale',StandardScaler()),
        ('model',HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,min_samples_leaf=30,
            learning_rate=.05,early_stopping=False,random_state=17))])
    with threadpool_limits(limits=2):
        hgb.fit(X[train],labels[train])
        tree_prediction=hgb.predict(X).astype(float)
    tree_prediction[~valid]=np.nan
    joblib.dump(dict(model=hgb,feature_columns=columns,horizon=int(horizon),train_indices=train),directory/'hgb.joblib')
    fitted=neural.fit_predict('gru',X[fit],sequences[fit],labels[fit]*10000,
        X[holdout],sequences[holdout],labels[holdout]*10000,X,sequences,seed=17,
        config=dict(epochs=8,patience=3,hidden=32,batch_size=256,device=device))
    bundle=fitted['bundle']
    bundle.update(sequence_length=16,feature_columns=columns,horizon=int(horizon),
                  fit_indices=fit.tolist(),holdout_indices=holdout.tolist())
    torch.save(bundle,directory/'gru.pt')
    gru_prediction=fitted['prediction'].astype(float)/10000
    gru_prediction[~valid]=np.nan
    indices,latent,encoder_meta=_encoder_features(frame,encoder_cache_dir or directory.parent/'encoder-cache',device)
    mask=(indices+horizon<train_end)&np.isfinite(labels[indices])
    if mask.sum()<18:
        raise ValueError('At least 18 purged TRAIN Kronos windows required')
    components=min(16,latent.shape[1],int(mask.sum())-1)
    with threadpool_limits(limits=2):
        scaler=StandardScaler().fit(latent[mask])
        scaled=scaler.transform(latent)
        pca=PCA(n_components=components,svd_solver='full').fit(scaled[mask])
        projected=pca.transform(scaled)
        head=Ridge(alpha=100).fit(projected[mask],labels[indices[mask]])
        kronos_prediction=np.full(n,np.nan)
        kronos_prediction[indices]=head.predict(projected)
    np.savez_compressed(directory/'head.npz',scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
        pca_mean=pca.mean_,pca_components=pca.components_,coefficient=head.coef_,
        intercept=np.asarray(head.intercept_),train_indices=indices[mask],ridge_alpha=np.asarray(100.))
    predictions=dict(hgb=tree_prediction,gru=gru_prediction,kronos=kronos_prediction)
    np.savez_compressed(directory/'forecasts.npz',**predictions)
    artifacts={name:str(directory/name) for name in ('hgb.joblib','gru.pt','head.npz','forecasts.npz')}
    metadata=dict(status='ok',input_key=input_key,resume_hit=False,horizon=int(horizon),train_end=int(train_end),val_end=int(val_end),
        train_last_index=int(train[-1]),train_rows=len(train),gru_internal_boundary=inner,
        gru_fit_last_index=int(fit[-1]),gru_holdout_last_index=int(holdout[-1]),
        gru_fit_rows=len(fit),gru_holdout_rows=len(holdout),gru_training=fitted['training'],
        kronos_train_last_index=int(indices[mask][-1]),kronos_train_rows=int(mask.sum()),
        kronos_pca_components=components,kronos_ridge_alpha=100,encoder=encoder_meta,
        artifacts=artifacts,artifact_sha256={name:kronos._sha256(path) for name,path in artifacts.items()},
        source_sha256=source_hashes,
        target='gross future close/observed close - 1; no fees deducted',
        selection='Fixed models; only GRU early stopping on purged inner TRAIN holdout; VAL/TEST unused',
        unsupported_history='NaN warmup and histories crossing gaps; no forward fill',
        caveats=['Frozen Kronos pretraining overlap remains unverified',
                 'Predicted return is not executable profit; caller owns fee-aware execution and held-out selection'])
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    return dict(predictions=predictions,metadata=metadata)
