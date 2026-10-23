"""Ten causal long-spot rule sources. Setup geometry is not a probability forecast."""
import hashlib
import json

import numpy as np
import pandas as pd

from .data import aggregate_complete
from .features import _validated_frame

RULES=(('ema_crossover','trend',42),('trend_pullback','trend',60),
       ('donchian_breakout','breakout',21),('breakout_retest','breakout',25),
       ('bollinger_reentry','reversion',22),('rsi_recovery','reversion',22),
       ('vwap_reclaim','volume',22),('squeeze_breakout','volatility',42),
       ('support_sweep','reversion',21),('volume_momentum','breakout',22))


def scan_bars(bars,symbol,timeframe,asof,quote):
    """One completed, aggregated frame; also used unchanged in historical replay."""
    at=pd.Timestamp(asof);tf=int(timeframe)
    if at.tzinfo is None:raise ValueError('Aware strategy clock required')
    bars=bars.loc[bars.timestamp+pd.Timedelta(minutes=tf)<=at].tail(240).copy()
    rows=[];candidates=[]
    if len(bars):
        gaps=bars.timestamp.diff().dt.total_seconds().ne(tf*60)
        bars=bars.loc[bars.index>=bars.index[gaps][-1]].reset_index(drop=True)
    fresh=len(bars)>0 and 0<=(at-bars.timestamp.iloc[-1]-pd.Timedelta(minutes=tf)).total_seconds()<=tf*60
    if len(bars)>=2:
        c,h,l,v=bars.close,bars.high,bars.low,bars.volume
        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr=tr.rolling(14).mean()
        fast,slow=c.ewm(span=9,adjust=False).mean(),c.ewm(span=21,adjust=False).mean()
        ema20,ema50=c.ewm(span=20,adjust=False).mean(),c.ewm(span=50,adjust=False).mean()
        mean=c.rolling(20).mean();std=c.rolling(20).std(ddof=0)
        upper,lower=mean+2*std,mean-2*std
        resistance,support=h.shift().rolling(20).max(),l.shift().rolling(20).min()
        gain=c.diff().clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
        loss=(-c.diff().clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
        rsi=(100-100/(1+gain/loss.replace(0,np.nan))).where(loss>0,100.).where((gain+loss)>0,50.)
        vwap=(((h+l+c)/3)*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
        volume_ratio=v/v.shift().rolling(20).mean().replace(0,np.nan)
        width=(upper-lower)/mean
        price=float(c.iloc[-1]);a=float(atr.iloc[-1])
    for name,family,required in RULES:
        row=dict(strategy=name,family=family,timeframe_minutes=tf,status='unavailable',
                 stance='neutral',triggered=False,reason='Insufficient contiguous completed bars',
                 completed_at=(bars.timestamp.iloc[-1]+pd.Timedelta(minutes=tf)).isoformat() if len(bars) else None)
        rows.append(row)
        if not fresh or len(bars)<required:continue
        trigger=False;bull=False;bear=False;target_mean=False;evidence={}
        if name=='ema_crossover':
            bull=fast.iloc[-1]>slow.iloc[-1];bear=fast.iloc[-1]<slow.iloc[-1]
            trigger=bull and fast.iloc[-2]<=slow.iloc[-2]
            evidence=dict(fast_ema=float(fast.iloc[-1]),slow_ema=float(slow.iloc[-1]),periods=[9,21])
        elif name=='trend_pullback':
            bull=ema20.iloc[-1]>ema50.iloc[-1];bear=ema20.iloc[-1]<ema50.iloc[-1]
            trigger=bull and l.iloc[-2]<=ema20.iloc[-2] and price>ema20.iloc[-1] and price>c.iloc[-2]
            evidence=dict(ema20=float(ema20.iloc[-1]),ema50=float(ema50.iloc[-1]))
        elif name=='donchian_breakout':
            bull=price>resistance.iloc[-1];bear=price<support.iloc[-1];trigger=bull
            evidence=dict(prior_high=float(resistance.iloc[-1]),prior_low=float(support.iloc[-1]),lookback=20)
        elif name=='breakout_retest':
            crossed=(c>resistance)&(c.shift()<=resistance.shift())
            positions=np.flatnonzero(crossed.to_numpy())
            previous=[i for i in positions if len(c)-4<=i<len(c)-1]
            level=float(resistance.iloc[previous[-1]]) if previous else None
            trigger=level is not None and l.iloc[-1]<=level+.25*a and price>level
            bull=trigger;bear=level is not None and price<level
            evidence=dict(retest_level=level,lookback=20)
        elif name=='bollinger_reentry':
            trigger=c.iloc[-2]<lower.iloc[-2] and price>lower.iloc[-1]
            bull=trigger;bear=price<lower.iloc[-1];target_mean=True
            evidence=dict(lower=float(lower.iloc[-1]),middle=float(mean.iloc[-1]),periods=20,stddevs=2)
        elif name=='rsi_recovery':
            trigger=rsi.iloc[-2]<30<=rsi.iloc[-1]
            bull=rsi.iloc[-1]>50;bear=rsi.iloc[-1]<30;target_mean=True
            evidence=dict(rsi=float(rsi.iloc[-1]),previous_rsi=float(rsi.iloc[-2]),periods=14)
        elif name=='vwap_reclaim':
            trigger=bool(np.isfinite(vwap.iloc[-1]) and np.isfinite(vwap.iloc[-2]) and c.iloc[-2]<=vwap.iloc[-2] and price>vwap.iloc[-1] and v.iloc[-1]>0)
            bull=bool(np.isfinite(vwap.iloc[-1]) and price>vwap.iloc[-1]);bear=bool(np.isfinite(vwap.iloc[-1]) and price<vwap.iloc[-1])
            evidence=dict(rolling_vwap=float(vwap.iloc[-1]) if np.isfinite(vwap.iloc[-1]) else None,window_bars=20)
        elif name=='squeeze_breakout':
            threshold=float(width.iloc[-22:-2].quantile(.2))
            squeezed=width.iloc[-2]<=threshold
            trigger=squeezed and price>upper.iloc[-1]
            bull=trigger;bear=squeezed and price<lower.iloc[-1]
            evidence=dict(previous_bandwidth=float(width.iloc[-2]),trailing_bandwidth_quantile20=threshold,upper=float(upper.iloc[-1]))
        elif name=='support_sweep':
            trigger=l.iloc[-1]<support.iloc[-1]<price and price>bars.open.iloc[-1]
            bull=trigger;bear=price<support.iloc[-1]
            evidence=dict(prior_support=float(support.iloc[-1]),observed_low=float(l.iloc[-1]),lookback=20)
        elif name=='volume_momentum':
            trigger=bool(np.isfinite(volume_ratio.iloc[-1]) and volume_ratio.iloc[-1]>=1.5 and price>resistance.iloc[-1])
            bull=trigger;bear=bool(np.isfinite(volume_ratio.iloc[-1]) and volume_ratio.iloc[-1]>=1.5 and price<support.iloc[-1])
            evidence=dict(relative_volume=float(volume_ratio.iloc[-1]) if np.isfinite(volume_ratio.iloc[-1]) else None,minimum_relative_volume=1.5,prior_high=float(resistance.iloc[-1]))
        row.update(status='ok',stance='long' if trigger or bull else 'avoid_or_exit' if bear else 'neutral',
                   triggered=bool(trigger),reason='Fresh long setup' if trigger else 'Observed rule state; no fresh long trigger',evidence=evidence)
        if not trigger or v.iloc[-1]<=0:continue
        ref=float(quote['ask']);bid=float(quote['bid'])
        # Fixed research defaults; observed ATR/structure sets protection, not a promised win rate.
        risk=max(.005*ref,1.5*a,ref-float(l.tail(5).min())+.25*a)
        target=float(mean.iloc[-1]) if target_mean else ref+2*risk
        rr=(target-ref)/risk
        if risk/ref>.03 or not 1<=rr<=4 or not 0<ref-risk<bid<target:
            row['candidate_status']='Geometry outside supported entry domain';continue
        candidate=dict(stop_price=float(ref-risk),target_price=target,horizon_minutes={5:60,15:180,60:720}[tf],
                       strategy_sources=[dict(strategy=name,family=family,timeframe_minutes=tf,completed_at=row['completed_at'])])
        candidate['setup_id']=hashlib.sha256(json.dumps([symbol,name,tf,row['completed_at'],ref],sort_keys=True).encode()).hexdigest()
        candidates.append(candidate);row['candidate_status']='Submitted for model scoring';row['setup_id']=candidate['setup_id']
    return rows,candidates


def evaluate(frame,symbol,asof,quote):
    at=pd.Timestamp(asof)
    if at.tzinfo is None:raise ValueError('Aware strategy clock required')
    receipt=pd.Timestamp(quote['available_at'])
    if receipt.tzinfo is None or not 0<=(at-receipt).total_seconds()<=30:raise ValueError('Stale/future strategy quote')
    if not np.isfinite([quote['ask'],quote['bid']]).all() or not 0<quote['bid']<=quote['ask']:raise ValueError('Invalid strategy quote')
    _,times,_,cadence=_validated_frame(frame)
    if cadence!=60:raise ValueError('Native M1 strategy input required')
    native=frame.loc[times+pd.Timedelta(minutes=1)<=at].copy()
    rows=[];candidates=[]
    for tf in (5,15,60):
        bars,_=aggregate_complete(native,tf*60)
        r,c=scan_bars(bars,symbol,tf,asof,quote);rows.extend(r);candidates.extend(c)
    merged={}
    for candidate in candidates:
        key=tuple(candidate[k] for k in ('stop_price','target_price','horizon_minutes'))
        if key in merged:merged[key]['strategy_sources'].extend(candidate['strategy_sources'])
        else:merged[key]=candidate
    return dict(asof=at.isoformat(),symbol=symbol,status='ok' if any(r['status']=='ok' for r in rows) else 'unavailable',
                strategies=rows,candidates=list(merged.values()),
                active_families=sorted({r['family'] for r in rows if r['triggered']}),
                interpretation='Correlated rule evidence from the same prices, not independent votes or calibrated probabilities; no forced time exit')
