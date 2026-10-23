"""Preserve an original trained entry signal while applying separate paper-only exits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value,name):
    if type(value) not in (int,float) or not np.isfinite(value):raise ValueError('Finite numeric '+name+' required')
    return float(value)


class BarrierLiveScanner:
    def __init__(self,directory,policy_path,base_scanner=None):
        self.policy_path=Path(policy_path).resolve();self.policy=json.loads(self.policy_path.read_text(encoding='utf-8'))
        p=self.policy;self.base_policy_path=Path(p['base_policy_path']).resolve()
        if _sha(self.base_policy_path)!=p['base_policy_sha256']:raise ValueError('Original model policy hash differs')
        source_policy=json.loads(self.base_policy_path.read_text(encoding='utf-8'))
        if source_policy.get('checkpoint_sha256'):
            from .kronos import _sha256
            if _sha256(source_policy['checkpoint'])!=source_policy['checkpoint_sha256']:raise ValueError('Original checkpoint hash differs')
        if p['base_family'] not in ('learned_path','first-touch'):raise ValueError('Calibrated original model family required')
        spec=p['exit_spec'];self.reward_risk=_number(spec['reward_risk'],'reward/risk')
        if self.reward_risk<=0:raise ValueError('Positive reward/risk required')
        self.stop_mode=spec['stop_mode']
        if self.stop_mode not in ('fixed','learned_mae_q50'):raise ValueError('Unsupported barrier stop source')
        if self.stop_mode=='fixed':
            self.fixed_stop=_number(spec['stop_fraction'],'stop fraction')
            if not 0<self.fixed_stop<1:raise ValueError('Stop fraction must be between zero and one')
        if base_scanner is None:
            if p['base_family']=='learned_path':
                from .learned_path_scanner import LearnedPathScanner
                base_scanner=LearnedPathScanner(directory,self.base_policy_path)
            else:
                from .first_touch_scanner import FirstTouchScanner
                base_scanner=FirstTouchScanner(directory,self.base_policy_path)
        self.base=base_scanner;self.minimum_net_rr=0.
        self.artifact_receipt=dict(wrapper_policy_sha256=_sha(self.policy_path),base_policy_sha256=p['base_policy_sha256'],
            base_artifact_receipt=deepcopy(getattr(self.base,'artifact_receipt',{})),exit_spec=deepcopy(spec))

    def __getattr__(self,name):return getattr(self.base,name)

    def close(self):self.base.close()

    def evaluate(self,frame,symbol,asof,book,config):
        result=deepcopy(self.base.evaluate(frame,symbol,asof,book,config))
        if result.get('status')!='setup':return result
        try:
            original=deepcopy(result['setup']);ask=_number(book['ask'],'ask');bid=_number(book['bid'],'bid')
            fees=config['fee_bps'];fee=_number(fees[symbol.split('/')[1]] if isinstance(fees,dict) else fees,'fee bps')
            if ask<=0 or not 0<bid<ask or not 0<=fee<10000:raise ValueError('Invalid original entry quote/cost')
            f=fee/10000;sl=_number(original['stop_price'],'original stop');tp=_number(original['target_price'],'original target')
            probabilities=np.asarray(original['first_touch_probabilities'],float)
            if probabilities.shape!=(3,) or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.isclose(probabilities.sum(),1):
                raise ValueError('Original calibrated outcome probabilities required')
            timeout=_number(original['train_timeout_return'],'original timeout return')
            if not 0<sl<bid<ask<tp or timeout<=-1:raise ValueError('Original trained barriers no longer bracket the quote')
            threshold=_number(self.base.policy.get('threshold',0.),'original threshold');minimum_rr=float(self.base.minimum_net_rr)
            reward=tp*(1-f)/(ask*(1+f))-1;loss=sl*(1-f)/(ask*(1+f))-1
            net=float(probabilities@np.array([reward,loss,timeout]));rr=reward/-loss if loss<0 else 0.
            if probabilities[0]<threshold or net<=0 or reward<=0 or rr<minimum_rr:
                raise ValueError('Original frozen model signal fails probability/payoff entry gate')
            original_net=_number(result['summary']['model_net_return'],'original learned expected return')
            if not np.isclose(original_net,net,rtol=1e-6,atol=1e-10):raise ValueError('Original learned payoff receipt differs from current quote')
            signal=dict(reference_price=ask,stop_price=sl,target_price=tp,first_touch_probabilities=probabilities.tolist(),
                train_timeout_return=timeout,expected_net_return=net,fee_bps=fee,minimum_net_rr=minimum_rr,probability_threshold=threshold,
                deadline=result.get('model_forecasts',{}).get('first_touch',{}).get('deadline'),
                original_policy_path=str(self.base_policy_path),original_policy_sha256=self.policy['base_policy_sha256'])
            source={}
            if self.stop_mode=='fixed':stop=self.fixed_stop
            else:
                forecast=result['model_forecasts']['learned_path']
                if forecast['predictor_family']!='tcn':raise ValueError('Learned stop requires the actual TCN extrema forecast')
                value=_number(forecast['mae_log_quantiles']['q50'],'TCN MAE q50')
                stop=float(-np.expm1(-value));source=dict(mae_log_quantile=value,quantile=.5,
                    model_weights_sha256=forecast.get('model_weights_sha256'),quantiles_file=forecast.get('quantiles_file'),
                    quantiles_sha256=forecast.get('quantiles_sha256'),horizon_hours=forecast.get('horizon_hours'))
            target=stop*self.reward_risk
            if not np.isfinite([stop,target]).all() or not 0<stop<1 or target<=0:raise ValueError('Learned/fixed stop does not define valid barriers')
            target_price=ask*(1+target);stop_price=ask*(1-stop)
            if stop_price>=bid or target_price*(1-f)/(ask*(1+f))<=1:raise ValueError('New paper exit geometry fails fresh quote/cost guard')
            setup={k:v for k,v in original.items() if k not in ('first_touch_probabilities','train_timeout_return','horizon_bars','max_hold_bars')}
            setup.update(family='barrier-live',template='barrier-live',exit_mode='barriers_only',expires_at=None,
                holding_policy='barriers_only',original_entry_signal=signal,barrier_stop_fraction=stop,barrier_target_fraction=target,
                stop_fraction=stop,target_fraction=target,stop_price=stop_price,target_price=target_price,
                potential_net_reward_fraction=target_price*(1-f)/(ask*(1+f))-1,
                reward_risk_after_cost=(target_price*(1-f)/(ask*(1+f))-1)/(1-stop_price*(1-f)/(ask*(1+f))))
            result['setup']=setup
            result['model_forecasts']['barrier_exit']=dict(exit_mode='barriers_only',stop_mode=self.stop_mode,stop_fraction=stop,
                target_fraction=target,gross_reward_risk=self.reward_risk,source=source,expires_at=None,
                original_signal=deepcopy(signal),new_barrier_probabilities=None,
                assumption='Original model probabilities gate entry only; these separate exits have no calibrated probability or finite model holding deadline.')
            result['summary'].update(original_model_net_return=net,barrier_stop_fraction=stop,barrier_target_fraction=target,
                barrier_gross_reward_risk=self.reward_risk,barriers_only_exit=1.,predicted_target_price=target_price,predicted_stop_price=stop_price)
            result['reason']='Original trained entry signal approved; separate paper barriers remain active until TP or SL'
            return result
        except (KeyError,ValueError,TypeError,IndexError,OverflowError) as error:
            result.update(status='no_setup',setup=None,reason='barrier entry rejected: '+str(error));return result
