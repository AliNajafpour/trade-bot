"""TRAIN-only first-touch and conditional-timeout heads for learned price extrema."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .supervised_dataset import QUANTILES,sha,save

GEOMETRIES={'g0':(1,1),'g1':(2,0),'g2':(0,0)}


def geometry_levels(quantiles,reference_close,geometry):
    q=np.asarray(quantiles,float);reference=float(reference_close)
    if q.shape!=(3,3) or not np.isfinite(q).all() or not np.isfinite(reference) or reference<=0 or geometry not in GEOMETRIES:raise ValueError('Invalid learned geometry input')
    target,stop=GEOMETRIES[geometry]
    return dict(target_price=float(reference*np.exp(q[0,target])),stop_price=float(reference*np.exp(-q[1,stop])),
        target_quantile=QUANTILES[target],stop_quantile=QUANTILES[stop])


def signal_features(features,quantiles,horizon_minutes,geometry):
    x=np.asarray(features,np.float32);q=np.asarray(quantiles,np.float32)
    if x.ndim==1:x=x[None,:]
    if q.ndim==2:q=q[None,:,:]
    if x.ndim!=2 or q.shape!=(len(x),3,3) or not np.isfinite(x).all() or not np.isfinite(q).all() or geometry not in GEOMETRIES:raise ValueError('Invalid path-head input')
    a,b=GEOMETRIES[geometry]
    return np.column_stack([x,q.reshape(len(x),9),np.full(len(x),np.log1p(horizon_minutes)),np.full(len(x),QUANTILES[a]),np.full(len(x),QUANTILES[b])]).astype(np.float32)


class PathSignalHead:
    def __init__(self,classifier,timeout_model,metadata):self.classifier=classifier;self.timeout_model=timeout_model;self.metadata=metadata

    @classmethod
    def fit(cls,x,labels,returns,horizons):
        x=np.asarray(x,np.float32);labels=np.asarray(labels,int);returns=np.asarray(returns,float);horizons=np.asarray(horizons,int)
        if len(x)<100 or not np.isfinite(x).all() or not np.isfinite(returns).all() or not np.isin(labels,[0,1,2]).all():raise ValueError('Insufficient finite calibration TRAIN labels')
        classifier=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=80,l2_regularization=5.,learning_rate=.06,early_stopping=False,random_state=17)
        timeout=labels==2;timeout_model=None
        with threadpool_limits(limits=2):
            classifier.fit(x,labels)
            if timeout.sum()>=100:
                timeout_model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=5.,learning_rate=.06,early_stopping=False,random_state=17)
                timeout_model.fit(x[timeout],returns[timeout])
        fallback={str(h):float(returns[timeout&(horizons==h)].mean()) for h in np.unique(horizons) if (timeout&(horizons==h)).any()}
        fallback['pooled']=float(returns[timeout].mean()) if timeout.any() else 0.
        metadata=dict(family='path_first_touch_hgb',training_rows=len(x),class_counts={str(i):int((labels==i).sum()) for i in [0,1,2]},
            timeout_rows=int(timeout.sum()),conditional_timeout_model=timeout_model is not None,timeout_means=fallback,n_features=x.shape[1],
            training='Later TRAIN calibration only; base predictors trained on earlier disjoint TRAIN. VAL/TEST never fit.',
            probabilities='Uncalibrated boosted softmax; held-out Brier/reliability must be inspected, not confidence guarantees.')
        return cls(classifier,timeout_model,metadata)

    def predict(self,features,quantiles,horizon_minutes,geometry):
        x=signal_features(features,quantiles,horizon_minutes,geometry)
        if x.shape[1]!=self.metadata['n_features']:raise ValueError('Head feature-schema mismatch')
        with threadpool_limits(limits=2):raw=self.classifier.predict_proba(x)
        result=np.zeros((len(x),3));result[:,self.classifier.classes_]=raw;return result

    def timeout_return(self,horizon_minutes,X_features=None,quantiles=None,geometry=None):
        fallback=self.metadata['timeout_means'].get(str(int(horizon_minutes)),self.metadata['timeout_means']['pooled'])
        if X_features is None:return fallback
        x=signal_features(X_features,quantiles,horizon_minutes,geometry)
        if self.timeout_model is None:return np.full(len(x),fallback)
        with threadpool_limits(limits=2):return self.timeout_model.predict(x)

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True);joblib.dump(dict(classifier=self.classifier,timeout_model=self.timeout_model),path/'head.joblib',compress=3)
        self.metadata['weights_sha256']=sha(path/'head.joblib');save(path/'head.json',self.metadata)

    @classmethod
    def load(cls,path):
        path=Path(path);metadata=json.loads((path/'head.json').read_text())
        if sha(path/'head.joblib')!=metadata['weights_sha256']:raise ValueError('Path head hash mismatch')
        data=joblib.load(path/'head.joblib');return cls(data['classifier'],data['timeout_model'],metadata)
