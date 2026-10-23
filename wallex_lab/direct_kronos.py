"""Direct frozen Kronos OHLCV sampling; no fitted return head or signal substitute."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from threading import Lock

import numpy as np
import pandas as pd

from .features import _validated_frame
from .kronos import _verify_assets, _load_encoder, _sha256, _positive_int, TRADE_BOT

UPSTREAM = TRADE_BOT / 'third_party/kronos-67b630e/model/kronos.py'
UPSTREAM_SHA256 = '0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032'
COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']
_SAMPLING_LOCK = Lock()


def _predictor_class():
    """Load only pinned upstream predictor/sampling definitions, without import collisions."""
    import torch
    import torch.nn.functional as F
    if _sha256(UPSTREAM) != UPSTREAM_SHA256:
        raise ValueError('Original Kronos predictor source hash differs from pinned upstream')
    tree = ast.parse(UPSTREAM.read_text(encoding='utf-8'))
    names = {'top_k_top_p_filtering', 'sample_from_logits', 'auto_regressive_inference', 'calc_time_stamps', 'KronosPredictor'}
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError('Pinned upstream predictor definitions missing')
    namespace = dict(torch=torch, F=F, np=np, pd=pd, trange=range)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(UPSTREAM), 'exec'), namespace)
    return namespace['KronosPredictor']


def causal_window(frame, asof, context=128):
    context = _positive_int(context, 'context')
    if not 2 <= context <= 512:
        raise ValueError('context must be in [2,512]')
    clock = pd.Timestamp(asof)
    if clock.tzinfo is None:
        raise ValueError('asof must include a timezone')
    clock = clock.tz_convert('UTC')
    stamps = pd.DatetimeIndex(pd.to_datetime(frame['timestamp'], errors='raise'))
    if stamps.tz is None:
        raise ValueError('Input timestamps must include a timezone')
    interval = float(frame.attrs.get('interval_seconds', 300))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('interval_seconds must be positive finite')
    completed = frame.loc[stamps + pd.Timedelta(seconds=interval) <= clock].tail(context).copy().reset_index(drop=True)
    completed.attrs['interval_seconds'] = interval
    if len(completed) != context:
        raise ValueError('Insufficient completed context')
    _, times, gaps, _ = _validated_frame(completed)
    if gaps.any():
        raise ValueError('Kronos context must be contiguous')
    return completed, dict(asof=clock.isoformat(), interval_seconds=interval,
        first_open_at=times[0].isoformat(), last_open_at=times[-1].isoformat(),
        last_completed_at=(times[-1] + pd.Timedelta(seconds=interval)).isoformat(),
        bar_age_seconds=float((clock-times[-1]).total_seconds()-interval))


def summarize_paths(raw_paths, last_close, roundtrip_cost_bps=0):
    """Reject whole physically invalid paths; never repair model-generated prices."""
    paths = np.asarray(raw_paths, dtype=float)
    if paths.ndim != 3 or paths.shape[2] != 6 or paths.shape[0] < 1 or paths.shape[1] < 1:
        raise ValueError('paths must have shape [samples,horizon,6]')
    if not np.isfinite(last_close) or last_close <= 0 or not np.isfinite(roundtrip_cost_bps) or roundtrip_cost_bps < 0:
        raise ValueError('positive last_close and nonnegative finite costs required')
    finite = np.isfinite(paths).all(axis=(1,2))
    prices = (paths[:,:,:4] > 0).all(axis=(1,2))
    activity = (paths[:,:,4:] >= 0).all(axis=(1,2))
    physical = ((paths[:,:,1] >= paths[:,:,[0,2,3]].max(axis=2)) &
                (paths[:,:,2] <= paths[:,:,[0,1,3]].min(axis=2))).all(axis=1)
    valid = finite & prices & activity & physical
    result = dict(status='ok' if valid.any() else 'invalid_forecasts', requested_paths=len(paths),
        valid_paths=int(valid.sum()), discarded_paths=int((~valid).sum()), valid_path_indices=np.flatnonzero(valid).tolist(),
        invalid_path_indices=np.flatnonzero(~valid).tolist(), discard_counts=dict(nonfinite=int((~finite).sum()),
        nonpositive_price=int((~prices).sum()), negative_volume_or_amount=int((~activity).sum()), invalid_ohlc=int((~physical).sum())),
        distribution_kind='independent autoregressive sampled paths; empirical uncalibrated')
    if not valid.any():
        return result
    good = paths[valid]
    def quantiles(a, axis=None):
        return {f'q{int(q*100):02d}':np.quantile(a,q,axis=axis).tolist() for q in (.1,.25,.5,.75,.9)}
    terminal = good[:,-1,3]
    maxima, minima = good[:,:,1].max(axis=1), good[:,:,2].min(axis=1)
    tp, sl = float(np.quantile(maxima,.75)), float(np.quantile(minima,.25))
    result.update(close_quantiles_by_step=quantiles(good[:,:,3],axis=0), terminal_close_quantiles=quantiles(terminal),
        path_max_high_quantiles=quantiles(maxima), path_min_low_quantiles=quantiles(minima),
        terminal_close_mean=float(terminal.mean()), terminal_return_mean=float(terminal.mean()/last_close-1),
        probability_terminal_close_above_last=float((terminal>last_close).mean()),
        probability_terminal_return_above_cost=float((terminal/last_close-1 > roundtrip_cost_bps/10000).mean()),
        roundtrip_cost_bps=float(roundtrip_cost_bps), path_terminal_close=terminal.tolist(),
        path_max_high=maxima.tolist(), path_min_low=minima.tolist())
    barriers = dict(take_profit_price=tp,stop_loss_price=sl,derivation='TP=q75(path max high), SL=q25(path min low)',
        long_brackets_last_close=bool(sl<last_close<tp),same_bar_policy='SL first',
        calibration='Empirical model-sample frequencies, not observed trade probabilities')
    if sl<last_close<tp:
        tp_first=sl_first=ambiguous=unresolved=0
        for path in good:
            resolved=False
            for bar in path:
                hit_tp,hit_sl=bar[1]>=tp,bar[2]<=sl
                if hit_tp and hit_sl: ambiguous+=1
                if hit_sl: sl_first+=1;resolved=True;break
                if hit_tp: tp_first+=1;resolved=True;break
            if not resolved: unresolved+=1
        barriers.update(p_tp_before_sl=tp_first/len(good),p_sl_before_tp=sl_first/len(good),
            p_neither_within_horizon=unresolved/len(good),ambiguous_same_bar_paths=ambiguous)
    result['tp_sl']=barriers
    return result


def summarize_close_paths(raw_paths, last_close, roundtrip_cost_bps=0):
    """Price-only output validity is independent of generated wicks/volume validity."""
    paths=np.asarray(raw_paths,dtype=float)
    if paths.ndim!=3 or paths.shape[2]!=6 or min(paths.shape[:2])<1:
        raise ValueError('paths must have shape [samples,horizon,6]')
    if not np.isfinite(last_close) or last_close<=0 or not np.isfinite(roundtrip_cost_bps) or roundtrip_cost_bps<0:
        raise ValueError('positive last_close and nonnegative finite costs required')
    close=paths[:,:,3]
    valid=(np.isfinite(close)&(close>0)).all(axis=1)
    result=dict(status='ok' if valid.any() else 'invalid_price_forecasts',requested_paths=len(paths),
        valid_paths=int(valid.sum()),discarded_paths=int((~valid).sum()),valid_path_indices=np.flatnonzero(valid).tolist(),
        distribution_kind='independent autoregressive CLOSE paths; empirical uncalibrated',
        validity='Only finite positive generated CLOSE prices; OHLC/volume quality reported separately')
    if not valid.any():return result
    close=close[valid];terminal=close[:,-1];maxima=close.max(axis=1);minima=close.min(axis=1)
    def quantiles(a,axis=None):
        return {f'q{int(q*100):02d}':np.quantile(a,q,axis=axis).tolist() for q in (.1,.25,.5,.75,.9)}
    tp,sl=float(np.quantile(maxima,.75)),float(np.quantile(minima,.25))
    result.update(close_quantiles_by_step=quantiles(close,axis=0),terminal_close_quantiles=quantiles(terminal),
        path_max_close_quantiles=quantiles(maxima),path_min_close_quantiles=quantiles(minima),
        terminal_close_mean=float(terminal.mean()),terminal_return_mean=float(terminal.mean()/last_close-1),
        probability_terminal_close_above_last=float((terminal>last_close).mean()),
        probability_terminal_return_above_cost=float((terminal/last_close-1>roundtrip_cost_bps/10000).mean()),
        roundtrip_cost_bps=float(roundtrip_cost_bps),path_terminal_close=terminal.tolist(),
        path_max_close=maxima.tolist(),path_min_close=minima.tolist())
    barriers=dict(take_profit_price=tp,stop_loss_price=sl,derivation='TP=q75(path max close), SL=q25(path min close)',
        long_brackets_last_close=bool(sl<last_close<tp),barrier_event='predicted CLOSE crossing only; not intrabar touches',
        calibration='Uncalibrated model sample frequencies, not observed trade probabilities')
    if sl<last_close<tp:
        tp_first=sl_first=unresolved=0
        for path in close:
            resolved=False
            for price in path:
                if price<=sl:sl_first+=1;resolved=True;break
                if price>=tp:tp_first+=1;resolved=True;break
            if not resolved:unresolved+=1
        barriers.update(p_tp_before_sl=tp_first/len(close),p_sl_before_tp=sl_first/len(close),
            p_neither_within_horizon=unresolved/len(close),same_bar_policy='single close cannot cross disjoint barriers')
    result['tp_sl']=barriers
    return result


class DirectKronos:
    def __init__(self, family='small', device='cuda:0', predictor=None, provenance=None, checkpoint=None, checkpoint_metadata=None):
        self.device = device
        if predictor is None:
            model_dir, tokenizer_dir, self.provenance = _verify_assets({'kronos_model':family})
            model, tokenizer = _load_encoder(model_dir, tokenizer_dir, device)
            if checkpoint is not None:
                import torch
                path=Path(checkpoint).resolve()
                if path.suffix=='.safetensors':
                    from safetensors.torch import load_file
                    state=load_file(str(path),device=device)
                elif path.suffix in ('.pt','.pth'):
                    state=torch.load(path,map_location=device,weights_only=True)
                else:raise ValueError('checkpoint must be a direct state_dict .pt/.pth or .safetensors')
                model.load_state_dict(state,strict=True)
                self.provenance.update(checkpoint_path=str(path),checkpoint_sha256=_sha256(path),
                    checkpoint_load='strict model state_dict after verified pretrained initialization',
                    checkpoint_training_status='User-supplied checkpoint; training evidence must be inspected separately')
                if checkpoint_metadata is not None:
                    metadata=Path(checkpoint_metadata).resolve()
                    self.provenance.update(checkpoint_metadata_path=str(metadata),checkpoint_metadata_sha256=_sha256(metadata))
            predictor = _predictor_class()(model,tokenizer,device=device,max_context=512)
            self.provenance.update(predictor_source=str(UPSTREAM),predictor_source_sha256=UPSTREAM_SHA256)
        else:
            self.provenance = provenance or {'test_double':True}
        self.predictor = predictor

    def forecast(self, frame, asof, run_dir, context=128, pred_len=12, paths=4, seed=20261021, roundtrip_cost_bps=0):
        import torch
        pred_len,paths = _positive_int(pred_len,'pred_len'),_positive_int(paths,'paths')
        if pred_len>512 or paths>256:
            raise ValueError('pred_len<=512 and paths<=256 required')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=2**32-paths:
            raise ValueError('seed must allow distinct uint32 path seeds')
        window, clock = causal_window(frame,asof,context)
        folder=Path(run_dir).resolve();folder.mkdir(parents=True,exist_ok=True)
        window.to_csv(folder/'input.csv',index=False)
        xstamps=pd.Series(pd.to_datetime(window.timestamp,utc=True))
        ystamps=pd.Series(pd.date_range(xstamps.iloc[-1]+pd.Timedelta(seconds=clock['interval_seconds']),
            periods=pred_len,freq=pd.Timedelta(seconds=clock['interval_seconds'])))
        generated=[];latencies=[];seeds=list(range(seed,seed+paths));started=datetime.now(timezone.utc).isoformat()
        for draw_seed in seeds:
            # Torch sampling uses process-global RNG; serialize draws across forecast callers.
            with _SAMPLING_LOCK:
                torch.manual_seed(draw_seed)
                if self.device.startswith('cuda') and torch.cuda.is_available(): torch.cuda.manual_seed_all(draw_seed)
                before=time.perf_counter()
                with torch.inference_mode():
                    output=self.predictor.predict(window[COLUMNS[:5]],xstamps,ystamps,pred_len,
                        T=1.,top_k=0,top_p=.9,sample_count=1,verbose=False)
            generated.append(output[COLUMNS].to_numpy(dtype=float));latencies.append(time.perf_counter()-before)
        raw=np.stack(generated)
        np.savez_compressed(folder/'raw-paths.npz',paths=raw,seeds=np.asarray(seeds),columns=np.asarray(COLUMNS),
            forecast_open_ns=pd.DatetimeIndex(ystamps).as_unit('ns').asi8)
        result=summarize_paths(raw,float(window.close.iloc[-1]),roundtrip_cost_bps)
        result['price_summary']=summarize_close_paths(raw,float(window.close.iloc[-1]),roundtrip_cost_bps)
        result.update(clock=clock,context=context,pred_len=pred_len,horizon_minutes=pred_len*clock['interval_seconds']/60,
            seeds=seeds,sample_count_per_call=1,sampling=dict(temperature=1.,top_k=0,top_p=.9),
            started_at=started,completed_at=datetime.now(timezone.utc).isoformat(),device=self.device,
            path_latency_seconds=latencies,total_inference_seconds=sum(latencies),
            raw_paths_file=str(folder/'raw-paths.npz'),raw_paths_sha256=_sha256(folder/'raw-paths.npz'),
            input_file=str(folder/'input.csv'),input_sha256=_sha256(folder/'input.csv'),
            adapter_sha256=_sha256(__file__),provenance=self.provenance,
            caveats=['Frozen pretrained training overlap is unknown; this retrospective diagnostic may have pretraining contamination.',
                'Sample quantiles are uncalibrated and noisy at small sample counts; rejected paths condition the distribution.',
                'Amount is upstream OHLC mean times volume, an approximation; no fills or profit are inferred.'])
        (folder/'forecast.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        return result
