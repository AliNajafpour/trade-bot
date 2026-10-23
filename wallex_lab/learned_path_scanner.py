"""Paper integration of marginal path extrema and a TRAIN-fitted outcome head.

Reuses ForecastScanner's serialized asynchronous cache and the existing market
freshness gate. Marginal extrema are never represented as sampled joint paths.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import fast_market
from .forecast_scanner import ForecastScanner
from .path_signal import PathSignalHead, geometry_levels
from .spot import canonical_symbol, protection_geometry
from .supervised_dataset import HORIZONS, TARGET_NAMES, QUANTILES, live_inputs


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LearnedPathScanner(ForecastScanner):
    def __init__(self,directory,policy_path,model=None,head=None):
        self.policy_path=Path(policy_path).resolve();self.policy_sha256=_sha(self.policy_path)
        self.policy=json.loads(self.policy_path.read_text(encoding='utf-8'));p=self.policy
        if p.get('entry_review_mode','legacy') not in ('legacy','model_context'):raise ValueError('Unknown entry review mode')
        if type(p.get('minimum_net_rr',1.)) not in (int,float):raise ValueError('Numeric minimum net RR required')
        self.minimum_net_rr=float(p.get('minimum_net_rr',1.))
        threshold=p['threshold'];hours=p['horizon_hours']
        if (type(hours) not in (int,float) or hours not in HORIZONS or p['geometry'] not in ('g0','g1','g2')
            or type(threshold) not in (int,float) or not np.isfinite(threshold) or not 0<=threshold<=1
            or not np.isfinite(self.minimum_net_rr) or self.minimum_net_rr<0):raise ValueError('Invalid frozen learned-path policy')
        self.model_family=p['model_family']
        if self.model_family not in ('hgb','tcn'):raise ValueError('Unsupported learned-path predictor family')
        if model is None:
            if self.model_family=='hgb':
                from .path_hgb import PathHGB
                model=PathHGB.load(p['model_path'])
            else:
                from .path_tcn import PathTCN
                model=PathTCN.load(p['model_path'],device='cpu')
        self.model=model;self.head=head if head is not None else PathSignalHead.load(p['calibrator_path'])
        for artifact,expected in ((self.model,p['model_weights_sha256']),(self.head,p['calibrator_sha256'])):
            if not expected or artifact.metadata.get('weights_sha256')!=expected:raise ValueError('Weights differ from frozen validation policy')
        self.artifact_receipt=dict(policy_sha256=self.policy_sha256,model_weights_sha256=p['model_weights_sha256'],calibrator_sha256=p['calibrator_sha256'])
        for key,path in (('model_metadata_sha256',Path(p['model_path'])/'model.json'),('calibrator_metadata_sha256',Path(p['calibrator_path'])/'head.json')):
            actual=_sha(path) if path.exists() else None
            if key in p and actual!=p[key]:raise ValueError('Frozen artifact metadata hash differs')
            self.artifact_receipt[key]=actual
        if p.get('dataset_metadata_path'):
            if _sha(p['dataset_metadata_path'])!=p['dataset_metadata_sha256']:raise ValueError('Dataset metadata hash differs')
            schema=json.loads(Path(p['dataset_metadata_path']).read_text(encoding='utf-8'))
            if p.get('feature_names')!=schema['feature_names']:raise ValueError('Frozen dataset feature schema differs')
        self.view=p.get('view',p.get('family_alias',self.model_family));self.steps=int(hours*4)
        super().__init__(directory,family='chronos2',context=128,horizon=self.steps)

    def _forecast(self,frame,symbol,asof,key):
        folder=self.directory/symbol.replace('/','')/key;receipt=folder/'forecast.json'
        if receipt.exists():
            existing=json.loads(receipt.read_text(encoding='utf-8'))
            if existing['policy_sha256']!=self.policy_sha256:raise ValueError('Cached forecast policy differs')
            return existing
        features,sequence,names=live_inputs(frame,symbol.replace('/',''))
        if self.policy.get('feature_names') and list(names)!=self.policy['feature_names']:raise ValueError('Frozen feature schema differs')
        if self.head.metadata.get('feature_names') and list(names)!=self.head.metadata['feature_names']:raise ValueError('Trained head feature schema differs')
        q=np.asarray(self.model.predict_quantiles(features[None] if self.model_family=='hgb' else sequence[None]),float)
        if q.shape!=(1,5,3,3) or not np.isfinite(q).all() or (np.diff(q,axis=-1)<0).any():raise ValueError('Invalid learned marginal quantiles')
        folder.mkdir(parents=True,exist_ok=True)
        frame.tail(128).to_csv(folder/'input.csv',index=False)
        artifact=folder/'quantiles.npz';np.savez_compressed(artifact,quantiles=q[0],features=features,feature_names=np.asarray(names))
        origin=pd.Timestamp(frame.timestamp.iloc[-1])+pd.Timedelta(minutes=15)
        result=dict(model_family='learned_path',predictor_family=self.model_family,view=self.view,
            clock=dict(last_completed_at=origin.isoformat(),requested_at=str(asof)),reference_price=float(frame.close.iloc[-1]),
            quantiles_file=str(artifact.resolve()),quantiles_sha256=_sha(artifact),input_sha256=_sha(folder/'input.csv'),
            policy_sha256=self.policy_sha256,model_weights_sha256=self.policy['model_weights_sha256'],
            calibrator_sha256=self.policy['calibrator_sha256'],horizons_hours=list(HORIZONS),targets=list(TARGET_NAMES),
            quantile_levels=list(QUANTILES),joint_paths_available=False,
            assumption='Marginal log extrema and terminal quantiles; no native joint path or first-touch probability.')
        receipt.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        return result

    def evaluate(self,frame,symbol,asof,book,config):
        result=fast_market.evaluate(frame,symbol,asof,book,config)
        if not (result['status']=='setup' or result['status']=='no_setup' and result['reason']=='latest completed candle has no eligible setup'):return result
        result['setup']=None
        try:
            symbol=canonical_symbol(symbol)
            if result['clock']['interval_seconds']!=900:raise ValueError('Learned path model requires completed M15 inputs')
            bars=frame.loc[pd.to_datetime(frame.timestamp,utc=True)+pd.Timedelta(minutes=15)<=pd.Timestamp(asof)].copy()
            bars.attrs['interval_seconds']=900
            key,forecast=self.cached_forecast(bars,symbol,asof)
            if forecast is None:
                result.update(status='unavailable',reason='learned path forecast is being generated');return result
            if forecast['policy_sha256']!=self.policy_sha256 or _sha(forecast['quantiles_file'])!=forecast['quantiles_sha256']:raise ValueError('Forecast artifact receipt differs')
            origin=pd.Timestamp(forecast['clock']['last_completed_at']);p=self.policy
            minutes=int(p['horizon_hours']*60);deadline=origin+pd.Timedelta(minutes=minutes)
            if pd.Timestamp(asof)>=deadline:raise ValueError('Original forecast horizon expired')
            with np.load(forecast['quantiles_file'],allow_pickle=False) as raw:
                q=raw['quantiles'][list(HORIZONS).index(p['horizon_hours'])].copy();x=raw['features'][None].copy()
            if q.shape!=(3,3) or not np.isfinite(q).all() or (np.diff(q,axis=-1)<0).any():raise ValueError('Invalid marginal quantiles')
            geometry=geometry_levels(q,forecast['reference_price'],p['geometry'])
            sl,tp=float(geometry['stop_price']),float(geometry['target_price'])
            ask,bid=float(book['ask']),float(book['bid'])
            stop,target,_,_=protection_geometry({'stop_price':sl,'target_price':tp},ask)
            probabilities=np.asarray(self.head.predict(x,q[None],minutes,p['geometry']),float)[0]
            if probabilities.shape!=(3,) or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.isclose(probabilities.sum(),1):raise ValueError('Invalid observed-outcome probabilities')
            fees=config['fee_bps'];f=float(fees[symbol.split('/')[1]] if isinstance(fees,dict) else fees)/10000
            if not np.isfinite(f) or not 0<=f<1:raise ValueError('Invalid per-side fee')
            reward=tp*(1-f)/(ask*(1+f))-1;loss=sl*(1-f)/(ask*(1+f))-1
            train_timeout=float(np.asarray(self.head.timeout_return(minutes,X_features=x,quantiles=q[None],geometry=p['geometry'])).reshape(-1)[0])
            baseline_fee=float(self.head.metadata.get('fee_bps',13.))/10000
            if not np.isfinite(baseline_fee) or not 0<=baseline_fee<1 or not np.isfinite(train_timeout) or train_timeout<=-1:raise ValueError('Invalid trained timeout economics')
            # ponytail: next-open/reference drift is unmodelled; a future head can learn exit/reference directly.
            timeout_exit_ratio=(1+train_timeout)*(1+baseline_fee)/(1-baseline_fee)
            timeout=timeout_exit_ratio*forecast['reference_price']*(1-f)/(ask*(1+f))-1
            net=float(probabilities@np.array([reward,loss,timeout]));rr=float(reward/-loss) if loss<0 else 0.
            if not np.isfinite([timeout,net,rr]).all():raise ValueError('Nonfinite learned payoff')
            score=dict(p_tp_first=float(probabilities[0]),p_sl_first=float(probabilities[1]),p_timeout=float(probabilities[2]),
                expected_net_return=net,threshold=p['threshold'],horizon_hours=p['horizon_hours'],geometry=p['geometry'],view=self.view,
                head_family='hgb_observed_outcome',head_path=p['calibrator_path'],policy_selection_status=p['policy_selection_status'],
                target_price=tp,stop_price=sl,reward_risk_after_cost=rr,minimum_net_rr=self.minimum_net_rr,
                origin_at=origin.isoformat(),deadline=deadline.isoformat(),policy_sha256=self.policy_sha256,
                train_timeout_net_return=train_timeout,quote_adjusted_timeout_net_return=timeout,
                assumption='TRAIN-fitted observed-M1 outcome head; timeout quote adjustment approximates training next-open as reference close; probabilities are not native joint-path samples or a profit guarantee.')
            learned={**forecast,'model_path':p['model_path'],'weights_sha256':p['model_weights_sha256'],
                'horizon_hours':p['horizon_hours'],'terminal_close_quantiles':
                    {name:float(forecast['reference_price']*np.exp(q[2,i])) for i,name in enumerate(('q50','q75','q90'))},
                'mfe_log_quantiles':dict(zip(('q50','q75','q90'),q[0].tolist())),
                'mae_log_quantiles':dict(zip(('q50','q75','q90'),q[1].tolist()))}
            result['model_forecasts']=dict(model_family='learned_path',learned_path=learned,first_touch=score,aggregate_net_return=net,horizon_minutes=minutes)
            result['summary'].update(model_p_tp_first=score['p_tp_first'],model_p_sl_first=score['p_sl_first'],model_p_timeout=score['p_timeout'],
                model_net_return=net,model_margin_after_cost=net,model_horizon_minutes=minutes,predicted_target_price=tp,predicted_stop_price=sl,
                predicted_terminal_close_median=float(forecast['reference_price']*np.exp(q[2,0])),model_current_roundtrip_cost=2*f,
                model_tp_probability_threshold=p['threshold'],minimum_net_rr=self.minimum_net_rr,
                model_validation_qualified=float(p['policy_selection_status']=='validation_qualified'))
            if p.get('entry_review_mode')=='model_context':result['summary']['model_context_entry']=1
            if sl>=bid:reason='forecast stop already breached by current bid'
            elif probabilities[0]<p['threshold']:reason='learned TP-before-SL probability below frozen validation threshold'
            elif net<=0:reason='learned payoff does not clear current entry costs'
            elif reward<=0:reason='target price does not clear current entry costs'
            elif rr<self.minimum_net_rr:reason='post-cost payoff geometry below minimum'
            else:reason=None
            if reason:result.update(status='no_setup',reason=reason);return result
            volumes=bars.volume.tail(12);volumes=volumes[volumes>0]
            if not len(volumes):raise ValueError('No positive observed confirmation volume')
            result.update(status='setup',reason='learned path outcome head clears frozen policy and fresh quote costs',fingerprint=key,
                setup=dict(template='learned_path',family='learned_path',symbol=symbol,reference_price=ask,current_close=float(bars.close.iloc[-1]),
                    current_volume=float(bars.volume.iloc[-1]),stop_price=sl,target_price=tp,stop_fraction=stop,target_fraction=target,
                    roundtrip_cost_fraction=2*f,potential_net_reward_fraction=reward,reward_risk_after_cost=rr,
                    confirmation_price=float(bars.high.tail(4).max()),reclaim_price=float(bars.close.tail(12).mean()),minimum_volume=float(volumes.median()),
                    timestamp=origin.isoformat(),expires_at=deadline.isoformat(),horizon_bars=self.steps,max_hold_bars=self.steps,
                    horizon_minutes=minutes,holding_policy='validated_horizon',first_touch_probabilities=probabilities.tolist(),train_timeout_return=timeout))
        except (KeyError,ValueError,TypeError,IndexError,OSError,OverflowError) as error:
            result.update(status='unavailable',reason='learned path forecast unavailable: '+str(error),setup=None)
        return result
