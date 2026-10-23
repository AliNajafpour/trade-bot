"""Asynchronous direct-price Kronos scan for the existing paper orchestrator."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Lock

import pandas as pd

from . import fast_market
from .direct_kronos import DirectKronos, causal_window
from .fees import official_public_rates
from .forecast_scanner import NATIVE_FORECAST_LOCK
from .spot import canonical_symbol, protection_geometry


class KronosMarket:
    def __init__(self, directory, family='small', checkpoint=None, paths=16, horizon=12, predictor=None):
        if type(paths) is not int or not 8 <= paths <= 256 or type(horizon) is not int or not 1 <= horizon <= 72:
            raise ValueError('Use 8..256 sampled paths and 1..72 forecast bars')
        self.directory=Path(directory)
        self.family,self.checkpoint,self.paths,self.horizon=family,checkpoint,paths,horizon
        self.predictor=predictor
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.jobs={}; self.results={}; self.refresh_errors={}; self.lock=Lock()

    def _forecast(self, window, asof, directory, cost_bps):
        if self.predictor is None:
            self.predictor=DirectKronos(family=self.family,checkpoint=self.checkpoint)
        return self.predictor.forecast(window,asof,directory,context=128,pred_len=self.horizon,paths=self.paths,
            roundtrip_cost_bps=cost_bps)

    def _run_forecast(self,*args):
        with NATIVE_FORECAST_LOCK:
            return self._forecast(*args)

    def close(self):
        self.pool.shutdown(wait=True)

    def evaluate(self, frame, symbol, asof, book, config):
        result=fast_market.evaluate(frame,symbol,asof,book,config)
        eligible=result['status']=='setup' or (result['status']=='no_setup' and result['reason']=='latest completed candle has no eligible setup')
        if not eligible: return result
        result['setup']=None
        try:
            window,clock=causal_window(frame,asof,128)
            symbol=canonical_symbol(symbol)
            fee=config.get('fee_bps',official_public_rates())
            fee=float(fee[symbol.split('/')[1]] if isinstance(fee,dict) else fee)
            cost=2*(fee+float(config.get('slippage_bps',0)))/10000
            key=hashlib.sha256((symbol+window.to_csv(index=False)).encode()).hexdigest()
            with self.lock:
                job=self.jobs.get(symbol)
                if job and job[1].done():
                    try:
                        completed=job[1].result()
                        if not isinstance(completed,dict) or 'error' in completed:raise ValueError(str(completed))
                        self.results[symbol]=(job[0],completed);self.refresh_errors.pop(symbol,None)
                    except Exception as error:self.refresh_errors[symbol]=dict(key=job[0],error=str(error),observed_at=str(asof))
                    del self.jobs[symbol]
                cached=self.results.get(symbol)
                error=self.refresh_errors.get(symbol)
                if cached is None or cached[0]!=key:
                    if symbol not in self.jobs and (not error or error['key']!=key):
                        self.jobs[symbol]=(key,self.pool.submit(self._run_forecast,window.copy(),asof,self.directory/symbol.replace('/','')/key,cost*10000))
                if cached is None:
                    if error:raise ValueError(error['error'])
                    result.update(status='unavailable',reason='direct Kronos price forecast is being generated')
                    return result
                served_key,forecast=cached[0],deepcopy(cached[1])
                if served_key!=key or error:forecast['cache_refresh']=dict(requested_key=key,serving_key=served_key,pending=symbol in self.jobs,last_error=error)
            if 'error' in forecast: raise ValueError(forecast['error'])
            origin=pd.Timestamp(forecast['clock']['last_completed_at'])
            if origin>pd.Timestamp(asof) or (forecast.get('completed_at') and pd.Timestamp(forecast['completed_at'])>pd.Timestamp(asof)):
                raise ValueError('Forecast receipt is not yet observable')
            deadline=origin+pd.Timedelta(minutes=forecast['horizon_minutes'])
            if pd.Timestamp(asof)>=deadline:raise ValueError('forecast horizon has expired')
            price=forecast.get('price_summary',{})
            result['model_forecasts']={'direct_kronos':forecast,'model_family':'kronos','model_ready':False}
            if price.get('status')!='ok' or price.get('valid_paths',0)<8:
                raise ValueError('fewer than 8 finite positive sampled close-price paths')
            # Use the actual current ask; do not move a frozen price forecast with the market.
            ask=float(book['ask']); terminal=float(price['terminal_close_mean'])
            gross=terminal/ask-1; net=gross-cost
            barriers=price['tp_sl']; sl=float(barriers['stop_loss_price']); tp=float(barriers['take_profit_price'])
            prediction=result['model_forecasts']
            prediction.update(model_ready=True,aggregate_gross_return=gross,aggregate_net_return=net,
                horizon_minutes=forecast['horizon_minutes'],entry_margin=0.,
                models={'kronos':{'gross_return':gross,'net_return':net}},
                probability_gross_exceeds_cost=None,
                provenance={'distribution':'uncalibrated direct sampled close-price paths'})
            summary=result['summary']
            summary.update(kronos_gross_return=gross,model_net_return=net,model_median_net_return=net,
                model_horizon_minutes=forecast['horizon_minutes'],model_horizon_bars=self.horizon,
                model_current_roundtrip_cost=cost,model_entry_margin=0.,model_margin_after_cost=net,
                predicted_terminal_close_mean=terminal,predicted_terminal_close_median=float(price['terminal_close_quantiles']['q50']),
                predicted_close_q10=float(price['terminal_close_quantiles']['q10']),predicted_close_q90=float(price['terminal_close_quantiles']['q90']),
                predicted_target_price=tp,predicted_stop_price=sl,forecast_sample_count=float(price['valid_paths']))
            if net<=0:
                result.update(status='no_setup',reason='direct Kronos expected terminal price does not clear current ask and official fees')
                return result
            stop,target,_,_=protection_geometry({'stop_price':sl,'target_price':tp},ask)
            positive=window.volume.tail(12); positive=positive[positive>0]
            if not len(positive): raise ValueError('no observed positive volume for a possible WATCH confirmation')
            result.update(status='setup',reason='direct sampled Kronos price forecast clears observed ask and official fees',fingerprint=served_key,
                setup=dict(template='kronos_direct_price',family='kronos_direct_price',symbol=symbol,
                    reference_price=ask,current_close=float(window.close.iloc[-1]),current_volume=float(window.volume.iloc[-1]),
                    stop_price=sl,target_price=tp,stop_fraction=stop,target_fraction=target,
                    roundtrip_cost_fraction=cost,potential_net_reward_fraction=target-cost,
                    reward_risk_after_cost=(target-cost)/(stop+cost),
                    # These levels are used only if the news/regime LLM explicitly requests WATCH.
                    confirmation_price=float(window.high.tail(4).max()),reclaim_price=float(window.close.tail(12).mean()),
                    minimum_volume=float(positive.median()),timestamp=origin.isoformat(),expires_at=deadline.isoformat(),
                    horizon_bars=self.horizon,max_hold_bars=self.horizon,horizon_minutes=forecast['horizon_minutes'],
                    geometry_caveat='Fixed price quantiles of sampled future CLOSES; uncalibrated, not intrabar touch probabilities'))
        except (ValueError,KeyError,TypeError) as error:
            result.update(status='unavailable',reason='direct Kronos forecast unavailable: '+str(error),setup=None)
        return result
