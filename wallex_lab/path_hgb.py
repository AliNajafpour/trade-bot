"""Supervised quantile boosting for future HIGH, LOW drawdown and terminal price."""
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .supervised_dataset import HORIZONS,QUANTILES,sha,save


class PathHGB:
    def __init__(self,models,metadata):self.models=models;self.metadata=metadata

    @classmethod
    def fit(cls,x,y,fit_mask,known,*,leaves=15,iterations=100):
        x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);fit_mask=np.asarray(fit_mask,bool);known=np.asarray(known,bool)
        if x.ndim!=2 or y.shape!=(len(x),5,3) or known.shape!=(len(x),5) or not np.isfinite(x).all():raise ValueError('Invalid shared path arrays')
        models=[];started=time.perf_counter()
        with threadpool_limits(limits=2):
            for h in range(5):
                selected=fit_mask&known[:,h]
                if selected.sum()<100 or not np.isfinite(y[selected,h]).all():raise ValueError('Insufficient finite base-fit labels')
                for target in range(3):
                    for quantile in QUANTILES:
                        model=HistGradientBoostingRegressor(loss='quantile',quantile=quantile,max_iter=iterations,max_leaf_nodes=leaves,
                            min_samples_leaf=40,l2_regularization=3.,learning_rate=.06,early_stopping=False,random_state=17)
                        model.fit(x[selected],y[selected,h,target]);models.append(model)
        metadata=dict(family='hgb_path',leaves=leaves,iterations=iterations,quantiles=list(QUANTILES),horizons_hours=list(HORIZONS),
            fit_rows=int(fit_mask.sum()),n_features=x.shape[1],runtime_seconds=time.perf_counter()-started,
            quantile_ordering='Deterministic marginal rearrangement along quantile axis only; no target/stop inflation or clipping')
        return cls(models,metadata)

    def predict_quantiles(self,x):
        x=np.asarray(x,np.float32)
        if x.ndim!=2 or x.shape[1]!=self.metadata['n_features'] or not np.isfinite(x).all():raise ValueError('Invalid inference feature schema')
        with threadpool_limits(limits=2):prediction=np.column_stack([m.predict(x) for m in self.models]).reshape(len(x),5,3,3)
        return np.sort(prediction,axis=-1).astype(np.float32)

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True);joblib.dump(self.models,path/'model.joblib',compress=3)
        self.metadata['weights_sha256']=sha(path/'model.joblib');save(path/'model.json',self.metadata)

    @classmethod
    def load(cls,path):
        path=Path(path);metadata=json.loads((path/'model.json').read_text())
        if sha(path/'model.joblib')!=metadata['weights_sha256']:raise ValueError('HGB weight hash mismatch')
        return cls(joblib.load(path/'model.joblib'),metadata)
