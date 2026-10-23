"""Candidate-conditioned, calibrated within-horizon forecasts for spot paper use.

The finite forecast horizon is a probability question, never a forced exit.
Torch and sklearn are loaded only when a fitted engine is opened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .data import aggregate_complete, load_market
from .features import _validated_frame
from .supervised_dataset import sequence_window, summary_features

SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT')
HORIZONS = (60, 180, 360, 720, 1440)
GRID = tuple((s, s*r, h) for s in (.005, .01, .02, .03) for r in (1, 2, 3, 4) for h in HORIZONS)
MANAGEMENT_GEOMETRY = dict(stop_fraction=[.000001, .5], target_fraction=[.000001, 1.], horizon_minutes=[60, 1440])
ROLES = {'base_fit': ('2026-04-03', '2026-05-19'), 'early_stop': ('2026-05-22', '2026-06-01'),
         'head_fit': ('2026-06-04', '2026-07-31'), 'calibration': ('2026-08-03', '2026-08-31'),
         'test': ('2026-09-03', '2026-09-30')}
GAP_POLICY = dict(context_minutes=2100, maximum_missing_fraction=.01, maximum_missing_run=3,
                  maximum_staleness_minutes=2, method='causal_previous_observed_close_zero_volume',
                  quality_features=['missing_fraction','recent60_missing_fraction','maximum_missing_run','affected_m15_fraction'])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def training_role(at):
    at = pd.Timestamp(at)
    if at.tzinfo is None: raise ValueError('Timezone-aware origin required')
    return next((role for role, (start, end) in ROLES.items()
                 if pd.Timestamp(start, tz='UTC') <= at < pd.Timestamp(end, tz='UTC')), None)


def first_touch(high, low, close, reference, stop_fraction, target_fraction, horizon):
    """Class0 TP,1 SL,2 unresolved; include the immediately following minute."""
    high, low, close = (np.asarray(x)[:horizon] for x in (high, low, close))
    if len(close) != horizon or horizon < 1: raise ValueError('Complete future horizon required')
    tp = np.flatnonzero(high >= reference*(1+target_fraction))
    sl = np.flatnonzero(low <= reference*(1-stop_fraction))
    if len(sl) and (not len(tp) or sl[0] <= tp[0]): return 1, -float(stop_fraction)
    if len(tp): return 0, float(target_fraction)
    return 2, float(close[-1]/reference-1)


def validate_candidate(candidate, reference, purpose='entry'):
    s, t, h = float(candidate['stop_price']), float(candidate['target_price']), candidate['horizon_minutes']
    if isinstance(h, bool) or not isinstance(h, (int, np.integer)) or not 60 <= h <= 1440:
        raise ValueError('Forecast horizon must be an integer60..1440 minutes')
    if not np.isfinite([reference, s, t]).all() or reference <= 0 or not 0 < s < reference < t:
        raise ValueError('Long spot levels must satisfy0<stop<reference<target')
    stop, target = (reference-s)/reference, (t-reference)/reference
    if purpose == 'position_management':
        if not .000001-1e-12 <= stop <= .5+1e-12 or not .000001-1e-12 <= target <= 1+1e-12:
            raise ValueError('Management head supports stop distance0.0001..50% and target distance0.0001..100%')
        return stop, target, int(h)
    if purpose != 'entry': raise ValueError('Unknown model purpose')
    if not .005-1e-12 <= stop <= .03+1e-12 or not 1-1e-12 <= target/stop <= 4+1e-12:
        raise ValueError('Supported stop distance0.5..3% and target/stop ratio1..4')
    return stop, target, int(h)


def geometry_features(features, quantiles, stops, targets, horizons):
    x = np.asarray(features, np.float32); q = np.asarray(quantiles, np.float32)
    s, t, h = (np.asarray(v, float).reshape(-1) for v in (stops, targets, horizons))
    if x.ndim != 2 or q.shape != (len(x), 5, 3, 3) or any(len(v) != len(x) for v in (s,t,h)):
        raise ValueError('Candidate feature shapes differ')
    if not np.isfinite(x).all() or not np.isfinite(q).all() or not np.isfinite([s,t,h]).all() or (s<=0).any() or (t<=0).any() or (h<=0).any():
        raise ValueError('Candidate inputs must be finite and positive')
    # Interpolate only the neural covariates; the classifier always sees exact geometry.
    clocks = np.log(np.asarray(HORIZONS)); wanted = np.log(h)
    lower = np.clip(np.searchsorted(clocks, wanted, side='right')-1, 0, 3)
    weight = np.clip((wanted-clocks[lower])/(clocks[lower+1]-clocks[lower]),0,1)
    selected = q[np.arange(len(q)),lower]*(1-weight[:,None,None])+q[np.arange(len(q)),lower+1]*weight[:,None,None]
    return np.column_stack((x, selected.reshape(len(x),9), s, t, t/s, np.log1p(h), s/np.sqrt(h/60), t/np.sqrt(h/60))).astype(np.float32)


def probability_metrics(labels, probabilities):
    y, p = np.asarray(labels, int), np.asarray(probabilities, float)
    onehot = np.eye(3)[y]
    return {'count': len(y), 'class_counts': np.bincount(y,minlength=3).tolist(),
            'brier_multiclass': float(np.mean(np.sum((p-onehot)**2,axis=1))),
            'nll': float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
            'accuracy': float((p.argmax(axis=1)==y).mean())}


def calibrated(raw, temperature):
    logits = np.log(np.clip(raw,1e-12,1))/temperature
    logits -= logits.max(axis=1,keepdims=True)
    result = np.exp(logits)
    return result/result.sum(axis=1,keepdims=True)


def candidate_temperatures(metadata, horizons):
    fitted=metadata.get('horizon_temperatures')
    if fitted is None:return metadata['temperature']
    return np.exp(np.interp(np.log(horizons),np.log(HORIZONS),np.log(fitted)))[:,None]


def market_features(m15, minutes, symbol, reference):
    """Reuse finite-window path features, add completed-minute state and venue drift."""
    summary = summary_features(m15, 'BTCUSDT')
    summary = summary.loc[:, [c for c in summary if not c.startswith('asset_')]]
    return np.r_[summary.iloc[-1].to_numpy(float), minute_state(m15,minutes,reference),
                 [float(symbol==s) for s in SYMBOLS]].astype(np.float32)


def minute_state(m15, minutes, reference):
    c = minutes.close.to_numpy(float); last = c[-1]
    ret = [last/c[-lag-1]-1 for lag in (1,5,15,60)]
    recent = minutes.iloc[-60:]; returns = np.diff(np.log(c[-61:]))
    clock = pd.Timestamp(minutes.timestamp.iloc[-1])+pd.Timedelta(minutes=1)
    higher = pd.Timestamp(m15.timestamp.iloc[-1])+pd.Timedelta(minutes=15)
    minute = ret+[float(returns.std()), float(recent.high.max()/recent.low.min()-1),
                  float(np.log1p(recent.volume.sum())), float(reference/m15.close.iloc[-1]-1),
                  float(last/reference-1), float((clock-higher).total_seconds()/900)]
    return np.asarray(minute,np.float32)


def repair_m1_context(candles, asof, context_minutes):
    """Causal bounded M1 input repair, with an explicit mask and no source writes.

    Callers using a different model must validate its sensitivity to these inputs.
    """
    if type(context_minutes) is not int or not 60<=context_minutes<=20160:raise ValueError('Context must be60..20160 minutes')
    at=pd.Timestamp(asof)
    if at.tzinfo is None:raise ValueError('Timezone-aware decision required')
    frame=candles.copy() if isinstance(candles,pd.DataFrame) else pd.DataFrame(candles)
    if pd.DatetimeIndex(pd.to_datetime(frame.timestamp,errors='raise')).tz is None:raise ValueError('Timezone-aware source timestamps required')
    frame.timestamp=pd.to_datetime(frame.timestamp,utc=True)
    frame=frame.loc[frame.timestamp+pd.Timedelta(minutes=1)<=at].reset_index(drop=True);frame.attrs['interval_seconds']=60
    _,clocks,_,interval=_validated_frame(frame)
    if interval!=60 or (clocks.as_unit('ns').asi8%60_000_000_000).any():raise ValueError('Aligned M1 bars required')
    last_completed=clocks[-1]+pd.Timedelta(minutes=1)
    if at-last_completed>pd.Timedelta(minutes=2):raise ValueError('Completed M1 context stale')
    index=pd.date_range(end=clocks[-1],periods=context_minutes,freq='min')
    if clocks[0]>index[0]:raise ValueError(f'At least{context_minutes} minutes of observed context required')
    frame=frame.set_index('timestamp').reindex(index).rename_axis('timestamp').reset_index()
    observed=frame.close.notna().to_numpy();missing=~observed
    edges=np.flatnonzero(np.diff(np.r_[False,missing,False].astype(int)));longest=int(np.max(edges[1::2]-edges[::2],initial=0))
    if not observed[0] or missing.mean()>GAP_POLICY['maximum_missing_fraction'] or longest>GAP_POLICY['maximum_missing_run']:
        raise ValueError('Context exceeds bounded missingness limits: <=1% missing and <=3 consecutive minutes')
    previous=frame.close.ffill()
    frame.loc[missing,['open','high','low','close']]=np.repeat(previous.to_numpy()[missing,None],4,axis=1)
    frame.loc[missing,'volume']=0.;frame['observed_mask']=observed;frame.attrs['interval_seconds']=60
    quality=dict(observed_minutes=int(observed.sum()),input_minutes=context_minutes,missing_minutes=int(missing.sum()),
        missing_fraction=float(missing.mean()),recent60_missing_fraction=float(missing[-60:].mean()),maximum_missing_run=longest,
        reconstructed_timestamps=[value.isoformat() for value in frame.loc[missing,'timestamp']],
        reconstruction_method=GAP_POLICY['method'] if missing.any() else None,reconstruction_is_observed=False,
        observed_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),last_completed_at=last_completed.isoformat())
    return frame,quality


def prepare_context(candles, asof, gap_policy=None):
    """Validate observed bars first; bounded reconstruction is model input, never source data."""
    at=pd.Timestamp(asof)
    if at.tzinfo is None:raise ValueError('Timezone-aware decision required')
    frame=candles.copy() if isinstance(candles,pd.DataFrame) else pd.DataFrame(candles)
    if pd.DatetimeIndex(pd.to_datetime(frame.timestamp,errors='raise')).tz is None:raise ValueError('Timezone-aware source timestamps required')
    frame.timestamp=pd.to_datetime(frame.timestamp,utc=True)
    frame=frame.loc[frame.timestamp+pd.Timedelta(minutes=1)<=at].reset_index(drop=True)
    frame.attrs['interval_seconds']=60
    _,clocks,gaps,interval=_validated_frame(frame)
    if interval!=60 or (clocks.as_unit('ns').asi8%60_000_000_000).any():raise ValueError('Aligned M1 bars required')
    last_completed=clocks[-1]+pd.Timedelta(minutes=1)
    if at-last_completed>pd.Timedelta(minutes=2):raise ValueError('Completed M1 context stale')
    if gap_policy is None:
        if len(frame)<1935 or gaps[-1935:].any():raise ValueError('At least1935 contiguous completed M1 bars required')
        frame=frame.iloc[-2100:].copy();frame['observed_mask']=True
    else:
        if gap_policy!=GAP_POLICY:raise ValueError('Unsupported unvalidated missingness policy')
        frame,_=repair_m1_context(frame,at,GAP_POLICY['context_minutes'])
    missing=~frame.observed_mask.to_numpy();edges=np.flatnonzero(np.diff(np.r_[False,missing,False].astype(int)))
    longest=int(np.max(edges[1::2]-edges[::2],initial=0))
    frame.attrs['interval_seconds']=60
    higher,_=aggregate_complete(frame,900);higher=higher.iloc[-128:].reset_index(drop=True)
    if len(higher)!=128:raise ValueError('128 completed M15 input buckets required')
    affected=frame.loc[missing,'timestamp'].dt.floor('15min').nunique()
    quality=dict(observed_minutes=int((~missing).sum()),input_minutes=len(frame),missing_minutes=int(missing.sum()),
                 missing_fraction=float(missing.mean()),recent60_missing_fraction=float(missing[-60:].mean()),
                 maximum_missing_run=longest,affected_m15_fraction=float(affected/len(higher)),
                 reconstructed_timestamps=[value.isoformat() for value in frame.loc[missing,'timestamp']],
                 reconstruction_method=GAP_POLICY['method'] if missing.any() else None,
                 reconstruction_is_observed=False,observed_mask_sha256=hashlib.sha256((~missing).tobytes()).hexdigest(),
                 last_completed_at=last_completed.isoformat())
    return frame,higher,quality


def quality_features(quality):
    return np.asarray([quality[key] for key in GAP_POLICY['quality_features']],np.float32)


def detect_named_patterns(frame):
    """Describe simple 15m swing patterns for review; this is not a trained signal."""
    if len(frame) < 24:
        return []
    high, low = frame.high.to_numpy(float), frame.low.to_numpy(float)
    atr = float(np.mean(high[-15:] - low[-15:]))
    if not np.isfinite(atr) or atr <= 0:
        return []
    pivots = []
    for i in range(3, len(frame)-3):
        top = high[i] > np.max(high[i-3:i]) and high[i] > np.max(high[i+1:i+4])
        bottom = low[i] < np.min(low[i-3:i]) and low[i] < np.min(low[i+1:i+4])
        if top == bottom:
            continue
        point = ('H' if top else 'L', float(high[i] if top else low[i]), i)
        if pivots and pivots[-1][0] == point[0]:
            if (point[0] == 'H' and point[1] > pivots[-1][1]) or (point[0] == 'L' and point[1] < pivots[-1][1]):
                pivots[-1] = point
        else:
            pivots.append(point)
    found = []
    if len(pivots) >= 5:
        recent = pivots[-5:]
        types = ''.join(point[0] for point in recent)
        values = [point[1] for point in recent]
        if types == 'HLHLH' and abs(values[0]-values[4]) <= .75*atr and values[2]-max(values[0],values[4]) >= .6*atr:
            found.append(dict(name='head_and_shoulders_possible', direction='bearish', timeframe='15m',
                neckline_price=(values[1]+values[3])/2, invalidation_price=values[2]+.25*atr,
                quality='heuristic_unvalidated'))
        elif types == 'LHLHL' and abs(values[0]-values[4]) <= .75*atr and min(values[0],values[4])-values[2] >= .6*atr:
            found.append(dict(name='inverse_head_and_shoulders_possible', direction='bullish', timeframe='15m',
                neckline_price=(values[1]+values[3])/2, invalidation_price=values[2]-.25*atr,
                quality='heuristic_unvalidated'))
    if len(pivots) >= 3:
        recent = pivots[-3:]
        if ''.join(point[0] for point in recent) == 'HLH' and abs(recent[0][1]-recent[2][1]) <= .5*atr:
            found.append(dict(name='double_top_possible', direction='bearish', timeframe='15m',
                neckline_price=recent[1][1], invalidation_price=max(recent[0][1],recent[2][1])+.25*atr,
                quality='heuristic_unvalidated'))
        elif ''.join(point[0] for point in recent) == 'LHL' and abs(recent[0][1]-recent[2][1]) <= .5*atr:
            found.append(dict(name='double_bottom_possible', direction='bullish', timeframe='15m',
                neckline_price=recent[1][1], invalidation_price=min(recent[0][1],recent[2][1])-.25*atr,
                quality='heuristic_unvalidated'))
    return found


class TechnicalEngine:
    def __init__(self, model_dir):
        import joblib
        self.root = Path(model_dir)
        self.metadata = json.loads((self.root/'model.json').read_text(encoding='utf-8'))
        if self.metadata['format_version'] not in (1,2) or sha(self.root/'head.joblib') != self.metadata['head_sha256']:
            raise ValueError('Council model integrity check failed')
        self.purpose = self.metadata.get('purpose', 'entry')
        if self.purpose not in ('entry', 'position_management') or (self.purpose == 'position_management' and
                (self.metadata.get('supported_geometry') != MANAGEMENT_GEOMETRY or self.metadata.get('release_checks_passed') is not True)):
            raise ValueError('Unvalidated management model domain')
        if (self.root/'neural'/'numpy-export.json').exists():
            from .path_tcn_numpy import PathTCNNumpy
            self.neural = PathTCNNumpy.load(self.root/'neural')
            self.neural_backend = 'numpy_scipy'
        else:
            import torch
            from .path_tcn import PathTCN
            torch.set_num_threads(2)
            self.neural = PathTCN.load(self.root/'neural',device='cpu')
            self.neural_backend = 'torch'
        if self.neural.metadata['weights_sha256'] != self.metadata['neural_weights_sha256']:
            raise ValueError('Council neural weights differ from fitted head')
        self.head = joblib.load(self.root/'head.joblib')
        self.cache = {}

    def predict(self, candles, symbol, asof, quote, candidates=None):
        began = time.perf_counter(); at = pd.Timestamp(asof)
        result = dict(status='unavailable',symbol=symbol,model_id=self.metadata['model_id'],
                      model_available_at=self.metadata['available_at'],signals=[])
        try:
            if at.tzinfo is None or at < pd.Timestamp(self.metadata['available_at']):
                raise ValueError('Forecast predates fitted model availability')
            compact = symbol.replace('/','').upper()
            if compact not in SYMBOLS: raise ValueError('Unsupported reference asset/quote; USDT spot required')
            reference, bid = float(quote['ask']), float(quote['bid'])
            fee = float(quote.get('fee_bps',13))/10000
            if not np.isfinite([reference,bid,fee]).all() or not 0<bid<=reference or not 0<=fee<.01:
                raise ValueError('Invalid executable bid/ask or per-side fee')
            if self.purpose == 'position_management' and (candidates is None or fee != 0 or bid != reference):
                raise ValueError('Management requires explicit owned/proposed barriers, current bid reference and zero sunk entry fees')
            policy=self.metadata.get('gap_policy') if self.metadata['format_version']==2 else None
            frame,higher,quality=prepare_context(candles,at,policy)
            last_completed=pd.Timestamp(quality['last_completed_at'])
            sequence=sequence_window(higher)
            key=hashlib.sha256(sequence.tobytes()).hexdigest()
            cache_hit=compact in self.cache and self.cache[compact][0]==key
            if not cache_hit:
                self.cache[compact]=(key,self.neural.predict_quantiles(sequence[None])[0])
            q=self.cache[compact][1]
            features=market_features(higher,frame,compact,reference)
            if policy is not None:features=np.r_[features,quality_features(quality)].astype(np.float32)
            features_hash=hashlib.sha256(features.tobytes()).hexdigest()
            if candidates is None:
                candidates=[dict(stop_price=reference*(1-s),target_price=reference*(1+t),horizon_minutes=h) for s,t,h in GRID]
            if not isinstance(candidates,list) or not 1<=len(candidates)<=200: raise ValueError('1..200 candidate geometries required')
            geometries=[validate_candidate(candidate,reference,self.purpose) for candidate in candidates]
            s,t,h=np.asarray(geometries).T
            x=geometry_features(np.repeat(features[None],len(s),axis=0),np.repeat(q[None],len(s),axis=0),s,t,h)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2):
                raw=self.head['classifier'].predict_proba(x)
                unresolved=self.head['unresolved'].predict(x)
            p=calibrated(raw,candidate_temperatures(self.metadata,h))
            unresolved=np.clip(unresolved,-s,t)
            for i,candidate in enumerate(candidates):
                geometry=dict(reference_price=reference,stop_price=float(candidate['stop_price']),
                              target_price=float(candidate['target_price']),horizon_minutes=int(h[i]))
                identity=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                reward=(1+t[i])*(1-fee)/(1+fee)-1
                loss=(1-s[i])*(1-fee)/(1+fee)-1
                marked=(1+unresolved[i])*(1-fee)/(1+fee)-1
                gross=float(p[i]@np.array([t[i],-s[i],unresolved[i]]))
                signal_identity=hashlib.sha256((self.metadata['model_id']+identity+at.isoformat()+key+features_hash).encode()).hexdigest()
                result['signals'].append(dict(signal_id=signal_identity,symbol=symbol,
                    **geometry,geometry_id=identity,p_tp_first=float(p[i,0]),p_sl_first=float(p[i,1]),p_unresolved=float(p[i,2]),
                    expected_marked_net_return=float(p[i]@np.array([reward,loss,marked])),net_reward_risk=float(reward/-loss),
                    expected_marked_gross_return=gross,round_trip_fee_drag=float((1+gross)*(1-(1-fee)/(1+fee))),
                    expected_tp_contribution=float(p[i,0]*reward),expected_sl_contribution=float(p[i,1]*loss),
                    expected_unresolved_mark_contribution=float(p[i,2]*marked),reward_net_return=float(reward),loss_net_return=float(loss),
                    raw_p_tp_first=float(raw[i,0]),raw_p_sl_first=float(raw[i,1]),raw_p_unresolved=float(raw[i,2]),
                    unresolved_marked_net_return=float(marked),forecast_expires_at=(at+pd.Timedelta(minutes=int(h[i]))).isoformat(),
                    forced_exit_at=None,holding_policy='no_time_exit',fee_bps=fee*10000))
            result.update(status='ok',purpose=self.purpose,diagnostics=dict(runtime_seconds=time.perf_counter()-began,neural_cache_hit=cache_hit,neural_backend=self.neural_backend,
                last_completed_at=last_completed.isoformat(),higher_frame_closed_at=(pd.Timestamp(higher.timestamp.iloc[-1])+pd.Timedelta(minutes=15)).isoformat(),
                latest_candle_quote_basis=float(reference/frame.close.iloc[-1]-1),candidate_count=len(candidates),
                quoted_spread_bps=float((reference-bid)/reference*10000),input_quality=quality,
                same_minute_touch_policy='SL first',forecast_horizon_is_forced_exit=False),
                provenance=dict(training_source='Official Binance spot archives Apr-Sep2026',execution_source=quote.get('source','caller candles and executable quote'),
                    candle_source=candles.attrs.get('source','caller completed M1 candles; venue supplied by caller') if isinstance(candles,pd.DataFrame) else 'caller completed M1 candles',
                    neural_context_sha256=key,refreshed_features_sha256=features_hash,
                    model_weights_sha256=self.neural.metadata['weights_sha256'],head_sha256=self.metadata['head_sha256'],
                    heldout_metrics=self.metadata['evaluation']['test']['calibrated'],
                    limitation='Cross-venue transfer is unvalidated. Unresolved is pending MTM, never a realized win; MTM expectation assumes mark liquidation costs, not a scheduled sale. Barrier-level fill economics excludes slippage/gap overshoot. No named-pattern validation.'))
        except (ValueError,KeyError,TypeError,IndexError,OverflowError) as error:
            result.update(status='unavailable',signals=[],reason=str(error))
        return result


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    predict=sub.add_parser('predict');predict.add_argument('--model-dir',required=True)
    predict.add_argument('--request',required=True);predict.add_argument('--output',required=True)
    args=parser.parse_args();request=json.loads(Path(args.request).read_text(encoding='utf-8'))
    engine=TechnicalEngine(args.model_dir)
    results=[]
    for market in request['markets']:
        frame=load_market(market['candles_path'],60)
        frame.attrs['source']=market.get('reference_source','caller M1 CSV; consult source receipt for venue')
        result=engine.predict(frame,market['symbol'],request['asof'],market['quote'],market.get('candidates'))
        result.setdefault('provenance',{}).update(input_candles_path=str(Path(market['candles_path']).resolve()),input_candles_sha256=sha(market['candles_path']))
        results.append(result)
    Path(args.output).write_text(json.dumps(dict(asof=request['asof'],results=results),indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__': main()
