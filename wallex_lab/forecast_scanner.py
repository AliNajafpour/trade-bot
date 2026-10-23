"""Paper-only integration of genuine marginal forecasts and sampled path payoffs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Lock
import numpy as np
import pandas as pd
from . import fast_market
from .fees import official_public_rates
from .spot import canonical_symbol,protection_geometry

# Native predictors change process-global Torch RNG and allocate shared CUDA memory.
NATIVE_FORECAST_LOCK=Lock()


class ForecastScanner:
    def __init__(self,directory,family='chronos2',context=2048,horizon=36,checkpoint=None,predictor=None,fundamental_manifest=None):
        if family not in ('chronos2','kronos-path'): raise ValueError('Unknown forecast family')
        self.directory=Path(directory);self.family=family;self.context=context;self.horizon=horizon
        self.checkpoint=checkpoint;self.predictor=predictor;self.pool=ThreadPoolExecutor(max_workers=1)
        self.jobs={};self.cache={};self.refresh_errors={};self.lock=Lock();self.fundamentals={}
        if fundamental_manifest:
            from .chronos_market import prepare_fundamentals
            manifest=json.loads(Path(fundamental_manifest).read_text(encoding='utf-8'))
            for symbol,item in manifest['symbols'].items():
                source=item['outputs']['retrospective_assumed_48h'];path=Path(source['path'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']: raise ValueError('Fundamental source hash differs')
                self.fundamentals[symbol]=prepare_fundamentals(pd.read_csv(path),'retrospective_assumed_48h',source['sha256'])

    def close(self): self.pool.shutdown(wait=True)

    def _run_forecast(self,*args):
        if getattr(self,'model_family',None) in ('hgb','tcn'):
            return self._forecast(*args)
        with NATIVE_FORECAST_LOCK:
            return self._forecast(*args)

    def cached_forecast(self,bars,symbol,asof):
        """Share immutable async forecasts across decision policies; None means still pending."""
        key=hashlib.sha256((symbol+bars.tail(self.context+20).to_csv(index=False)).encode()).hexdigest()
        with self.lock:
            job=self.jobs.get(symbol)
            if job and job[1].done():
                try:
                    completed=job[1].result()
                    if not isinstance(completed,dict) or 'error' in completed:raise ValueError(str(completed))
                    self.cache[symbol]=(job[0],completed);self.refresh_errors.pop(symbol,None)
                except Exception as error:self.refresh_errors[symbol]=dict(key=job[0],error=str(error),observed_at=str(asof))
                del self.jobs[symbol]
            cached=self.cache.get(symbol)
            error=self.refresh_errors.get(symbol)
            if cached is None or cached[0]!=key:
                if symbol not in self.jobs and (not error or error['key']!=key):
                    self.jobs[symbol]=(key,self.pool.submit(self._run_forecast,bars,symbol,asof,key))
            if cached is None:
                if error:raise ValueError(error['error'])
                return key,None
            served_key,forecast=cached[0],deepcopy(cached[1])
            if served_key!=key or error:forecast['cache_refresh']=dict(requested_key=key,serving_key=served_key,pending=symbol in self.jobs,last_error=error)
        if 'error' in forecast:raise ValueError(forecast['error'])
        if pd.Timestamp(forecast['clock']['last_completed_at'])>pd.Timestamp(asof) or (forecast.get('completed_at') and pd.Timestamp(forecast['completed_at'])>pd.Timestamp(asof)):
            raise ValueError('Forecast receipt is not yet observable')
        return served_key,forecast

    def _forecast(self,frame,symbol,asof,key):
        folder=self.directory/symbol.replace('/','')/key
        if (folder/'forecast.json').exists(): return json.loads((folder/'forecast.json').read_text(encoding='utf-8'))
        if self.predictor is None:
            if self.family=='chronos2':
                from .chronos_market import ChronosMarket
                self.predictor=ChronosMarket()
            else:
                from .scaled_kronos import ScaledKronos
                self.predictor=ScaledKronos(checkpoint=self.checkpoint)
        if self.family=='chronos2':
            fundamental=self.fundamentals.get(symbol.replace('/',''))
            if fundamental is not None:
                # Carry only an already available past value into later observed bars.
                stamps=pd.DataFrame({'timestamp':pd.to_datetime(frame.timestamp,utc=True)})
                aligned=pd.merge_asof(stamps.sort_values('timestamp'),fundamental.sort_values('timestamp'),
                    on='timestamp',direction='backward',tolerance=pd.Timedelta(days=2))
                aligned.attrs=fundamental.attrs.copy()
            else: aligned=None
            return self.predictor.forecast(frame,symbol,asof,folder,context=self.context,pred_len=self.horizon,fundamentals=aligned)
        return self.predictor.forecast(frame,asof,folder,context=self.context,pred_len=self.horizon,paths=16,roundtrip_cost_bps=26)

    def evaluate(self,frame,symbol,asof,book,config):
        result=fast_market.evaluate(frame,symbol,asof,book,config)
        if not (result['status']=='setup' or result['status']=='no_setup' and result['reason']=='latest completed candle has no eligible setup'): return result
        result['setup']=None
        try:
            symbol=canonical_symbol(symbol);interval=result['clock']['interval_seconds']
            bars=frame.loc[pd.to_datetime(frame.timestamp,utc=True)+pd.Timedelta(seconds=interval)<=pd.Timestamp(asof)].copy()
            bars.attrs['interval_seconds']=interval
            key,forecast=self.cached_forecast(bars,symbol,asof)
            if forecast is None:
                result.update(status='unavailable',reason='price forecast is being generated');return result
            result['model_forecasts']={'direct_chronos2' if self.family=='chronos2' else 'direct_kronos':forecast,'model_family':self.family}
            deadline=pd.Timestamp(forecast['clock']['last_completed_at'])+pd.Timedelta(minutes=forecast['horizon_minutes'])
            if pd.Timestamp(asof)>=deadline:raise ValueError('forecast horizon expired')
            fees=config.get('fee_bps',official_public_rates());fee=float(fees[symbol.split('/')[1]] if isinstance(fees,dict) else fees)
            ask,bid=float(book['ask']),float(book['bid']);cost=2*fee/10000
            price=forecast['price_summary']
            if self.family=='chronos2':
                point=price['terminal_close_median'];q=price['close_quantiles_by_step']
                names=[name for name in ('q05','q10','q25','q50','q75','q90','q95') if name in q]
                if not {'q05','q50','q75'}<=set(names):raise ValueError('required decision quantiles missing')
                central=np.column_stack([q[name] for name in names])
                if not np.isfinite(central).all() or (central<=0).any() or (np.diff(central,axis=1)<0).any():
                    raise ValueError('invalid or crossed Chronos-2 decision quantiles')
                # Unused q01/q99 crossing cannot invalidate independently ordered central decision levels.
                result['model_forecasts']['decision_quantiles']=dict(names=names,ordered=True,
                    original_all_quantile_status=forecast['status'],native_crossed_steps=forecast.get('quantile_crossed_steps',0))
                future=np.asarray(pd.to_datetime(forecast['clock']['forecast_close_at'],utc=True)>pd.Timestamp(asof))
                if not future.any():raise ValueError('forecast horizon expired')
                tp=max(np.asarray(q['q75'])[future]);sl=min(np.asarray(q['q05'])[future])
                net=point*(1-fee/10000)/(ask*(1+fee/10000))-1
                if net<=0:
                    reason='Chronos-2 median price does not clear current ask and official fees'
                else:reason=None
            else:
                from .path_policy import select_path_policy
                with np.load(forecast['raw_paths_file'],allow_pickle=False) as saved:
                    future=pd.to_datetime(saved['forecast_open_ns'],utc=True)+pd.Timedelta(seconds=interval)>pd.Timestamp(asof)
                    policy=select_path_policy(saved['paths'][:,future,:],ask,fee_bps=fee)
                result['model_forecasts']['path_policy']=policy
                point=price['terminal_close_mean']
                if not policy['approved']:
                    result.update(status='no_setup',reason='sampled path payoff rejected: '+'; '.join(policy['reasons']));return result
                chosen=policy['selected'];tp=chosen['take_profit_price'];sl=chosen['stop_loss_price'];net=chosen['expected_net_return'];reason=None
            gross=point/ask-1;stop,target,_,_=protection_geometry({'stop_price':sl,'target_price':tp},ask)
            rr=(target-cost)/(stop+cost)
            summary=result['summary'];summary.update(model_net_return=float(net),model_median_net_return=float(net),
                model_horizon_minutes=forecast['horizon_minutes'],model_horizon_bars=self.horizon,model_current_roundtrip_cost=cost,
                model_margin_after_cost=float(net),predicted_terminal_close_median=float(price['terminal_close_quantiles']['q50']),
                predicted_target_price=float(tp),predicted_stop_price=float(sl),forecast_point_is_median=float(self.family=='chronos2'))
            result['model_forecasts'].update(aggregate_gross_return=float(gross),aggregate_net_return=float(net),
                horizon_minutes=forecast['horizon_minutes'],models={self.family:{'gross_return':float(gross),'net_return':float(net)}})
            if reason:result.update(status='no_setup',reason=reason);return result
            if sl>=bid:result.update(status='no_setup',reason='forecast stop already breached by current bid');return result
            if rr<1:result.update(status='no_setup',reason='post-cost payoff geometry below minimum');return result
            volumes=bars.volume.tail(12);volumes=volumes[volumes>0]
            if not len(volumes):raise ValueError('no observed positive confirmation volume')
            close=forecast['clock'].get('last_completed_at');deadline=pd.Timestamp(close)+pd.Timedelta(minutes=forecast['horizon_minutes'])
            result.update(status='setup',reason='model forecast clears current entry costs and independent price geometry',fingerprint=key,
                setup=dict(template=self.family,family=self.family,symbol=symbol,reference_price=ask,current_close=float(bars.close.iloc[-1]),
                    current_volume=float(bars.volume.iloc[-1]),stop_price=float(sl),target_price=float(tp),stop_fraction=stop,target_fraction=target,
                    roundtrip_cost_fraction=cost,potential_net_reward_fraction=target-cost,reward_risk_after_cost=rr,
                    confirmation_price=float(bars.high.tail(4).max()),reclaim_price=float(bars.close.tail(12).mean()),minimum_volume=float(volumes.median()),
                    timestamp=close,expires_at=deadline.isoformat(),horizon_bars=self.horizon,max_hold_bars=self.horizon,horizon_minutes=forecast['horizon_minutes']))
        except (KeyError,ValueError,TypeError) as error:result.update(status='unavailable',reason='forecast unavailable: '+str(error),setup=None)
        return result
