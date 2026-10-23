"""Completed-bar setup discovery; potential reward is geometry, not a forecast."""
import math

import numpy as np

from .features import _validated_frame


def scan_with_diagnostics(frame,config):
    data,timestamps,gaps,interval=_validated_frame(frame)
    stride=config.get('scan_stride_bars',4)
    hold=config.get('max_hold_bars',24)
    if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1 for v in (stride,hold)) or hold>24:
        raise ValueError('Positive integer scan clock and holding cap <=24 bars required')
    fee,slip=float(config.get('fee_bps',10)),float(config.get('slippage_bps',5))
    rr_min=float(config.get('minimum_net_rr',1.0))
    if not np.isfinite([fee,slip,rr_min]).all() or min(fee,slip,rr_min)<0:
        raise ValueError('Finite nonnegative costs and reward/risk gate required')
    cost=2*(fee+slip)/10000
    close=data.close.to_numpy();high=data.high.to_numpy();low=data.low.to_numpy();volume=data.volume.to_numpy()
    cumulative=np.cumsum(gaps)
    diagnostics=dict(scan_stride_bars=int(stride),interval_seconds=interval,
        insufficient_history=0,off_cadence=0,no_setup=0,cost_geometry=0,eligible=0,
        geometry_caveat='Targets use trailing movement scale; no expected return or success probability is inferred')
    candidates=[]
    for i in range(0,len(frame),stride):
        if i<48:
            diagnostics['insufficient_history']+=1;continue
        if cumulative[i]!=cumulative[i-48]:
            diagnostics['off_cadence']+=1;continue
        c=close[i];window=close[i-48:i+1]
        returns=np.diff(np.log(window))
        prior_close=close[i-48:i]
        true_range=np.maximum(high[i-47:i+1]-low[i-47:i+1],
            np.maximum(np.abs(high[i-47:i+1]-prior_close),np.abs(low[i-47:i+1]-prior_close)))
        bar_range=high[i]-low[i]
        features=dict(return_1=float(c/close[i-1]-1),return_3=float(c/close[i-3]-1),
            return_12=float(c/close[i-12]-1),return_48=float(c/close[i-48]-1),
            volatility_48=float(np.std(returns)),volume_ratio_12=float(np.mean(volume[i-11:i+1])/max(np.mean(volume[i-47:i+1]),1e-12)),
            close_location=float((c-low[i])/bar_range) if bar_range else .5,
            range_48=float((max(high[i-47:i+1])-min(low[i-47:i+1]))/c))
        if features['return_12']>0 and c>=max(close[i-20:i])*.999 and features['volume_ratio_12']>=.8:
            template='trend_breakout'
        elif features['return_48']>0 and features['return_3']>0 and c<max(window)*.999:
            template='trend_pullback'
        elif features['return_12']<-.005 and features['return_1']>0 and features['close_location']>.5:
            template='reversion'
        else:
            diagnostics['no_setup']+=1;continue
        stop=max(.003,min(.02,2*float(np.median(true_range))/c))
        # ponytail: trailing range/volatility gives payoff geometry, not a calibrated move forecast; replace only after a causal forecast comparison.
        target=min(.08,max(.65*features['range_48'],1.5*features['volatility_48']*math.sqrt(hold)))
        potential_net=target-cost
        rr=potential_net/(stop+cost)
        if potential_net<=0 or rr<rr_min:
            diagnostics['cost_geometry']+=1;continue
        candidates.append(dict(index=int(i),id=f'{template}-{i}',template=template,features=features,
            evidence=list(features),stop_fraction=stop,target_fraction=target,max_hold_bars=int(hold),
            roundtrip_cost_fraction=cost,potential_net_reward_fraction=potential_net,
            reward_risk_after_cost=rr,observation_hours=48*interval/3600,
            planned_max_hours=hold*interval/3600,
            geometry_caveat='Potential target payoff only; not expected profit, calibrated probability or a promise of fills'))
        diagnostics['eligible']+=1
    return dict(candidates=candidates,diagnostics=diagnostics)


def scan(frame,config):
    return scan_with_diagnostics(frame,config)['candidates']
