"""Paper entries scored by an observed-outcome head over immutable foundation forecasts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import fast_market
from .forecast_scanner import ForecastScanner
from .first_touch import forecast_features, model_geometry
from .spot import canonical_symbol, protection_geometry


class FirstTouchScanner(ForecastScanner):
    def __init__(self,directory,policy_path,head=None,predictor=None,fundamental_manifest=None):
        self.policy_path=Path(policy_path).resolve()
        self.policy=json.loads(self.policy_path.read_text(encoding='utf-8'))
        policy=self.policy
        if policy.get('entry_review_mode','legacy') not in ('legacy','model_context'):raise ValueError('Unknown entry review mode')
        self.minimum_net_rr=float(policy.get('minimum_net_rr',1.))
        if not np.isfinite(self.minimum_net_rr) or self.minimum_net_rr<0:raise ValueError('Nonnegative finite minimum net RR required')
        self.review_all_positive=policy.get('review_all_positive',False)
        if type(self.review_all_positive) is not bool:raise ValueError('Boolean review_all_positive required')
        if policy['horizon_hours'] not in (1,3,6,12,24) or policy['geometry'] not in ('g0','g1','g2'):
            raise ValueError('Unsupported frozen first-touch policy')
        self.view=policy['view'];self.steps=int(policy['horizon_hours']*4)
        if self.view=='kronos_scaled_finetuned' and not policy.get('checkpoint'):
            raise ValueError('Selected fine-tuned view requires its exact checkpoint')
        family='kronos-path' if self.view.startswith('kronos') else 'chronos2'
        super().__init__(directory,family=family,context=128 if family=='kronos-path' else 512,horizon=96,
            checkpoint=policy.get('checkpoint'),predictor=predictor,
            fundamental_manifest=fundamental_manifest if 'fundamental' in self.view else None)
        if head is None:
            from .first_touch import FirstTouchHead
            head=FirstTouchHead.load(policy['head_path'])
        self.head=head
        expected=policy.get('head_weights_sha256')
        if expected and self.head.metadata.get('weights_sha256')!=expected:
            raise ValueError('Selected head weights differ from the frozen validation policy')

    def _forecast(self,frame,symbol,asof,key):
        folder=self.directory/symbol.replace('/','')/key
        if (folder/'forecast.json').exists():return json.loads((folder/'forecast.json').read_text(encoding='utf-8'))
        if self.family=='kronos-path':
            if self.predictor is None:
                from .scaled_kronos import ScaledKronos
                self.predictor=ScaledKronos(checkpoint=self.checkpoint)
            return self.predictor.forecast(frame,asof,folder,context=self.context,pred_len=96,paths=8,roundtrip_cost_bps=26)
        if self.predictor is None:
            from .chronos_market import ChronosMarket
            self.predictor=ChronosMarket()
        fundamental=None
        if 'fundamental' in self.view:
            # The verified input retains availability clocks; align completed M5 to completed M15.
            source=self.fundamentals.get(symbol.replace('/',''))
            if source is None:raise ValueError('Selected fundamental view has no observed covariate source')
            # Parent loader already strips provenance columns; preserve its causally checked values.
            left=pd.DataFrame({'timestamp':pd.to_datetime(frame.timestamp,utc=True)})
            left['decision_at']=left.timestamp+pd.Timedelta(minutes=15)
            right=source.copy();right['decision_at']=pd.to_datetime(right.timestamp,utc=True)+pd.Timedelta(minutes=5)
            fundamental=pd.merge_asof(left.sort_values('decision_at'),right.drop(columns='timestamp').sort_values('decision_at'),
                on='decision_at',direction='backward',tolerance=pd.Timedelta(days=2)).drop(columns='decision_at')
            fundamental.attrs=source.attrs.copy()
        return self.predictor.forecast(frame,symbol,asof,folder,context=self.context,pred_len=96,
            technical=self.view!='close_calendar',fundamentals=fundamental)

    def evaluate(self,frame,symbol,asof,book,config):
        # Review experiment defers spread/stop entry checks to fresh post-LLM risk sizing.
        scan_config={**config,'max_spread_bps':20000.} if self.review_all_positive else config
        result=fast_market.evaluate(frame,symbol,asof,book,scan_config)
        if not (result['status']=='setup' or result['status']=='no_setup' and result['reason']=='latest completed candle has no eligible setup'):
            return result
        result['setup']=None
        try:
            symbol=canonical_symbol(symbol)
            if result['clock']['interval_seconds']!=900:raise ValueError('First-touch policy requires native completed M15 inputs')
            bars=frame.loc[pd.to_datetime(frame.timestamp,utc=True)+pd.Timedelta(minutes=15)<=pd.Timestamp(asof)].copy()
            bars.attrs['interval_seconds']=900
            key,forecast=self.cached_forecast(bars,symbol,asof)
            if forecast is None:
                result.update(status='unavailable',reason='price forecast is being generated');return result
            raw_name='direct_chronos2' if self.family=='chronos2' else 'direct_kronos'
            result['model_forecasts']={raw_name:forecast,'model_family':'first-touch'}
            origin=pd.Timestamp(forecast['clock']['last_completed_at'])
            deadline=origin+pd.Timedelta(hours=self.policy['horizon_hours'])
            if pd.Timestamp(asof)>=deadline:raise ValueError('Forecast horizon expired')
            if self.family=='chronos2':
                path=Path(forecast['quantiles_file'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=forecast['quantiles_sha256']:raise ValueError('Forecast hash differs')
                with np.load(path,allow_pickle=False) as raw:q=raw['quantiles'][0,:self.steps].copy();levels=raw['levels'].copy()
            else:
                path=Path(forecast['raw_paths_file'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=forecast['raw_paths_sha256']:raise ValueError('Forecast hash differs')
                from .chronos_market import QUANTILES
                levels=np.asarray(QUANTILES)
                with np.load(path,allow_pickle=False) as raw:paths=raw['paths'][:,:self.steps,3].copy()
                if not np.isfinite(paths).all() or (paths<=0).any():raise ValueError('Invalid sampled close paths')
                q=np.quantile(paths,levels,axis=0).T
            indices=[int(np.flatnonzero(np.isclose(levels,level))[0]) for level in (.05,.1,.25,.5,.75,.9,.95)]
            central=q[:,indices]
            if not np.isfinite(central).all() or (central<=0).any() or (np.diff(central,axis=1)<0).any():
                raise ValueError('Invalid/crossed trained decision quantiles')
            geometry=model_geometry(q,levels,self.policy['geometry'])
            origin_bars=bars.loc[pd.to_datetime(bars.timestamp,utc=True)+pd.Timedelta(minutes=15)<=origin]
            if origin_bars.empty:raise ValueError('Original forecast context unavailable')
            features,names=forecast_features(q,levels,float(origin_bars.close.iloc[-1]),self.policy['horizon_hours']*60,
                geometry,origin,self.view,origin_bars.close.to_numpy())
            if list(names)!=list(self.head.feature_names):raise ValueError('Trained feature schema differs')
            probabilities=np.asarray(self.head.predict_proba(features[None,:]),float)[0]
            if probabilities.shape!=(3,) or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.isclose(probabilities.sum(),1):
                raise ValueError('Invalid learned first-touch probabilities')
            ask,bid=float(book['ask']),float(book['bid'])
            fees=config['fee_bps'];fee=float(fees[symbol.split('/')[1]] if isinstance(fees,dict) else fees)
            sl,tp=geometry['stop_price'],geometry['target_price']
            stop,target,_,_=protection_geometry({'stop_price':sl,'target_price':tp},ask)
            f=fee/10000;reward=tp*(1-f)/(ask*(1+f))-1;loss=sl*(1-f)/(ask*(1+f))-1
            timeout=float(self.head.timeout_return(self.policy['horizon_hours']*60,self.view))
            net=float(probabilities@np.array([reward,loss,timeout]));rr=reward/(-loss) if loss<0 else 0.
            score=dict(p_tp_first=float(probabilities[0]),p_sl_first=float(probabilities[1]),p_timeout=float(probabilities[2]),
                expected_net_return=net,threshold=float(self.policy['threshold']),horizon_hours=self.policy['horizon_hours'],
                geometry=self.policy['geometry'],view=self.view,head_path=self.policy.get('head_path'),
                policy_selection_status=self.policy['policy_selection_status'],target_price=tp,stop_price=sl,
                reward_risk_after_cost=float(rr),minimum_net_rr=self.minimum_net_rr,
                origin_at=origin.isoformat(),deadline=deadline.isoformat(),policy_sha256=hashlib.sha256(self.policy_path.read_bytes()).hexdigest(),
                assumption='Observed-M1 trained head; fixed one-minute historical latency differs from measured live inference latency.')
            result['model_forecasts'].update(first_touch=score,aggregate_net_return=net,horizon_minutes=self.policy['horizon_hours']*60)
            result['summary'].update(model_p_tp_first=score['p_tp_first'],model_p_sl_first=score['p_sl_first'],model_p_timeout=score['p_timeout'],
                model_net_return=net,model_margin_after_cost=net,model_horizon_minutes=self.policy['horizon_hours']*60,
                predicted_target_price=tp,predicted_stop_price=sl,model_current_roundtrip_cost=2*f,
                model_tp_probability_threshold=float(self.policy['threshold']),
                minimum_net_rr=self.minimum_net_rr,positive_net_review=float(self.review_all_positive),
                model_validation_qualified=float(self.policy['policy_selection_status']=='validation_qualified'))
            if self.policy.get('entry_review_mode')=='model_context':result['summary']['model_context_entry']=1
            if sl>=bid and not self.review_all_positive:reason='forecast stop already breached by current bid'
            elif probabilities[0]<self.policy['threshold']:reason='learned TP-before-SL probability below frozen validation threshold'
            elif net<=0:reason='learned first-touch payoff does not clear current entry costs'
            elif reward<=0:reason='target price does not clear current entry costs'
            elif rr<self.minimum_net_rr:reason='post-cost payoff geometry below minimum'
            else:reason=None
            if reason:result.update(status='no_setup',reason=reason);return result
            volumes=bars.volume.tail(12);volumes=volumes[volumes>0]
            if not len(volumes):raise ValueError('No positive observed confirmation volume')
            result.update(status='setup',reason='learned TP-before-SL score clears frozen validation policy and current costs',fingerprint=key,
                setup=dict(template='first-touch',family='first-touch',symbol=symbol,reference_price=ask,current_close=float(bars.close.iloc[-1]),
                    current_volume=float(bars.volume.iloc[-1]),stop_price=sl,target_price=tp,stop_fraction=stop,target_fraction=target,
                    roundtrip_cost_fraction=2*f,potential_net_reward_fraction=reward,reward_risk_after_cost=rr,
                    confirmation_price=float(bars.high.tail(4).max()),reclaim_price=float(bars.close.tail(12).mean()),minimum_volume=float(volumes.median()),
                    timestamp=origin.isoformat(),expires_at=deadline.isoformat(),horizon_bars=self.steps,max_hold_bars=self.steps,
                    horizon_minutes=self.policy['horizon_hours']*60,holding_policy='validated_horizon',
                    first_touch_probabilities=probabilities.tolist(),train_timeout_return=timeout))
        except (KeyError,ValueError,TypeError,IndexError,OSError) as error:
            result.update(status='unavailable',reason='first-touch forecast unavailable: '+str(error),setup=None)
        return result
