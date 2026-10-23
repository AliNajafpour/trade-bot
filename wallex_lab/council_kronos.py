"""Bounded Kronos neural adaptation, paired path audit and council prediction CLI."""
import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .direct_kronos import COLUMNS, DirectKronos, causal_window, summarize_paths, summarize_close_paths
from .features import _validated_frame
from .finetune_kronos import WindowDataset, calendar_splits, train, _write_json
from .kronos import _sha256, ASSETS, TRADE_BOT
from .locking import exclusive

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'runs/council-foundation-upgrade-20261022/kronos'
STATUS = ROOT/'runs/market-council-20261022/model-status/kronos.json'
MANIFEST = ROOT/'data/council-reference-20261022/manifest.json'
LOCK = ROOT/'runs/foundation-training.gpu.lock'
HOURS = (1, 3, 6, 12, 24)
QUANTILES = (.1, .5, .9)


def now():
    return datetime.now(timezone.utc).isoformat()


def status(state, **fields):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    if state != 'queued' and 'message' not in fields:
        old.pop('message',None)
    old.update(model='kronos', status=state, updated_at=now(), **fields)
    old.setdefault('started_at', now())
    _write_json(STATUS, old)


def reference_frames(manifest=MANIFEST):
    manifest = Path(manifest)
    receipt = json.loads(manifest.read_text())
    if receipt['status'] != 'ready':
        raise ValueError('Reference manifest is not ready')
    frames, paths = {}, []
    for symbol, item in sorted(receipt['symbols'].items()):
        for minutes in (15, 60):
            path = Path(item[f'path_{minutes}m'])
            if _sha256(path) != item[f'sha256_{minutes}m']:
                raise ValueError(f'Reference hash mismatch: {path}')
            frame = pd.read_csv(path)
            _, _, gaps, interval = _validated_frame(frame)
            if gaps.any() or interval != minutes*60:
                raise ValueError('Reference cadence must be gap-free and exact')
            frame.attrs['interval_seconds'] = interval
            frames[path.stem] = frame
            paths.append(path)
    return frames, paths


@contextlib.contextmanager
def cached_decoder(model):
    """Reuse causal transformer K/V; preserve the native dependency decoder."""
    original = model.decode_s1
    had_override = 'decode_s1' in model.__dict__
    cache, hidden = [], None

    def decode(s1_ids, s2_ids, stamp=None, padding_mask=None):
        nonlocal cache, hidden
        if model.training or padding_mask is not None:
            raise ValueError('Cached inference requires eval and unpadded equal-length contexts')
        offset = 0 if hidden is None else hidden.shape[1]
        if offset and s1_ids.shape[1] != offset+1:
            cache, hidden, offset = [], None, 0
        s1, s2 = s1_ids[:,offset:], s2_ids[:,offset:]
        x = model.embedding([s1,s2])
        if stamp is not None:
            x = x+model.time_emb(stamp[:,offset:])
        x = model.token_drop(x)
        for index, layer in enumerate(model.transformer):
            attention = layer.self_attn
            normalized = layer.norm1(x)
            batch,length,_ = normalized.shape
            def project(projection):
                return projection(normalized).view(batch,length,attention.n_heads,attention.head_dim).transpose(1,2)
            q,k,v = (project(projection) for projection in (attention.q_proj,attention.k_proj,attention.v_proj))
            cosine,sine = attention.rotary._update_cos_sin_cache(q,offset+length)
            cosine,sine = cosine[:,:,offset:],sine[:,:,offset:]
            q = q*cosine+attention.rotary._rotate_half(q)*sine
            k = k*cosine+attention.rotary._rotate_half(k)*sine
            if offset:
                k = torch.cat((cache[index][0],k),dim=2)
                v = torch.cat((cache[index][1],v),dim=2)
                cache[index] = (k,v)
            else:
                cache.append((k,v))
            attended = torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=not offset)
            attended = attended.transpose(1,2).contiguous().view(batch,length,attention.d_model)
            x = x+attention.resid_dropout(attention.out_proj(attended))
            x = x+layer.ffn(layer.norm2(x))
        current = model.norm(x)
        hidden = current if hidden is None else torch.cat((hidden,current),dim=1)
        return model.head(current[:,-1:]),hidden

    model.decode_s1 = decode
    try:
        yield
    finally:
        if had_override:
            model.decode_s1 = original
        else:
            del model.decode_s1


def draws(predictor, windows, interval, paths, seed):
    """Duplicate series as batch members: upstream sample_count averages paths."""
    if not windows or not 2 <= paths <= 64:
        raise ValueError('Nonempty windows and2..64 paths required')
    x, xt, yt = [], [], []
    length = int(24*3600/interval)
    for frame in windows:
        stamps = pd.Series(pd.to_datetime(frame.timestamp, utc=True))
        future = pd.Series(pd.date_range(stamps.iloc[-1]+pd.Timedelta(seconds=interval),
            periods=length, freq=pd.Timedelta(seconds=interval)))
        for _ in range(paths):
            x.append(frame[COLUMNS[:5]])
            xt.append(stamps)
            yt.append(future)
    torch.manual_seed(seed)
    if predictor.device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), cached_decoder(predictor.predictor.model):
        generated = predictor.predictor.predict_batch(x, xt, yt, length,
            T=1., top_k=0, top_p=.9, sample_count=1, verbose=False)
    return np.stack([frame[COLUMNS].to_numpy() for frame in generated]).reshape(len(windows),paths,length,6)


def first_touch(path, reference, fraction=.01):
    for bar in path:
        # Intrabar ordering is unobserved; count a simultaneous touch as SL first.
        if bar[2] <= reference*(1-fraction):
            return 1
        if bar[1] >= reference*(1+fraction):
            return 0
    return 2


def path_metrics(raw, actual, reference, interval):
    close = np.asarray(raw)[:,:,3]
    valid = np.isfinite(close).all(axis=1) & (close > 0).all(axis=1)
    quality = summarize_paths(raw, reference)
    rows = []
    for hour in HOURS:
        count = int(hour*3600/interval)
        truth = actual[:count,3]/reference-1
        row = dict(horizon_hours=hour, valid_close_paths=int(valid.sum()),
            physically_valid_paths=quality['valid_paths'], requested_paths=len(raw))
        if valid.any():
            future = close[valid,:count]/reference-1
            q = np.quantile(future, QUANTILES, axis=0)
            error = truth[None,:]-q
            loss = np.maximum(np.array(QUANTILES)[:,None]*error,
                (np.array(QUANTILES)[:,None]-1)*error)
            median = q[1]
            actual_range = float((actual[:count,1].max()-actual[:count,2].min())/reference)
            # Close amplitude avoids interpreting malformed generated OHLC as valid.
            close_range = np.median((future.max(axis=1)-future.min(axis=1)))
            row.update(path_pinball_bps=float(loss.mean()*10000),
                terminal_pinball_bps=float(loss[:,-1].mean()*10000),
                terminal_absolute_error_bps=float(abs(median[-1]-truth[-1])*10000),
                direction_correct=bool(np.sign(median[-1]) == np.sign(truth[-1])),
                actual_return=float(truth[-1]), median_return=float(median[-1]),
                interval_80_coverage=bool(q[0,-1] <= truth[-1] <= q[2,-1]),
                median_close_path_amplitude=float(close_range),
                actual_close_path_amplitude=float(truth.max()-truth.min()),
                close_amplitude_absolute_error_bps=float(abs(close_range-(truth.max()-truth.min()))*10000),
                actual_high_low_amplitude=actual_range)
        indices = quality['valid_path_indices']
        if indices:
            outcomes = np.array([first_touch(raw[i,:count],reference) for i in indices])
            actual_event = first_touch(actual[:count],reference)
            frequencies = np.bincount(outcomes,minlength=3)/len(outcomes)
            row.update(first_touch_sample_frequencies=frequencies.tolist(),actual_first_touch=actual_event,
                uncalibrated_first_touch_brier=float(np.square(frequencies-np.eye(3)[actual_event]).sum()))
        rows.append(row)
    return rows


def aggregate(rows):
    result = {}
    for hour in HOURS:
        group = [row for row in rows if row['horizon_hours']==hour]
        metrics = dict(windows=len(group),requested_paths=sum(r['requested_paths'] for r in group),
            physically_valid_paths=sum(r['physically_valid_paths'] for r in group),
            valid_close_paths=sum(r['valid_close_paths'] for r in group))
        for key in ('path_pinball_bps','terminal_pinball_bps','terminal_absolute_error_bps','direction_correct',
                'interval_80_coverage','close_amplitude_absolute_error_bps','uncalibrated_first_touch_brier'):
            values = [r[key] for r in group if key in r]
            metrics[key] = float(np.mean(values)) if values else None
            metrics[key+'_windows'] = len(values)
        result[str(hour)] = metrics
    return result


def evaluate(output=OUTPUT, device='cuda:0', paths=16, origins=8, resume=False):
    output = Path(output)
    training = json.loads((output/'training/training.json').read_text())
    frames, _ = reference_frames()
    bounds = [pd.Timestamp(t) for t in training['splits']['boundaries']]
    scheduled = []
    for split, begin, end in zip(('validation','test'),bounds[1:3],bounds[2:4]):
        for key, frame in frames.items():
            interval = _validated_frame(frame)[3]
            length = int(86400/interval)
            dataset = WindowDataset({key:frame},begin,end,128,length,length)
            # Fixed clock-spread origins selected without looking at any score.
            chosen = np.linspace(0,len(dataset)-1,min(origins,len(dataset)),dtype=int)
            for index in chosen:
                symbol,start = dataset.windows[index]
                scheduled.append(dict(split=split,key=key,interval=interval,start=start,
                    forecast_start=dataset.times[symbol][start+128].isoformat(),
                    target_end_exclusive=(dataset.times[symbol][start+128+length-1]+pd.Timedelta(seconds=interval)).isoformat()))
    _write_json(output/'evaluation-origins.json',scheduled)
    checkpoint = Path(training['selected_weights'])
    variants = {'pretrained':None,'adapted_validation_selected':checkpoint}
    all_metrics = {}
    with exclusive(LOCK,timeout=3600) if device.startswith('cuda') else contextlib.nullcontext():
        print('KRONOS_EVALUATION_GPU_LOCK_ACQUIRED' if device.startswith('cuda') else 'KRONOS_EVALUATION_CPU',flush=True)
        try:
            for variant, weights in variants.items():
                rows_path = output/(variant+'-rows.json')
                rows = json.loads(rows_path.read_text()) if resume and rows_path.exists() else []
                completed = {(r['split'],r['key'],r['start']) for r in rows}
                for identity in completed:
                    previous = [r for r in rows if (r['split'],r['key'],r['start'])==identity]
                    if {r['horizon_hours'] for r in previous} != set(HOURS) or any(r['requested_paths']!=paths for r in previous):
                        raise ValueError('Resume requires every completed origin to have all horizons and identical path count')
                resumed_origins = len(completed)
                predictor, latency = None, []
                for interval in (900,3600):
                    group = [item for item in scheduled if item['interval']==interval]
                    for batch_start in range(0,len(group),2):
                        batch = group[batch_start:batch_start+2]
                        done = [(item['split'],item['key'],item['start']) in completed for item in batch]
                        if all(done):
                            continue
                        if any(done):
                            raise ValueError('Cannot resume a partial RNG batch without changing sampling protocol')
                        if predictor is None:
                            predictor = DirectKronos(device=device,checkpoint=weights)
                        windows = [frames[item['key']].iloc[item['start']:item['start']+128].copy() for item in batch]
                        started = time.perf_counter()
                        raw = draws(predictor,windows,interval,paths,20261022+batch_start+int(interval))
                        latency.append(time.perf_counter()-started)
                        for item,window,samples in zip(batch,windows,raw):
                            length = int(86400/interval)
                            actual = frames[item['key']].iloc[item['start']+128:item['start']+128+length][COLUMNS[:5]].to_numpy()
                            for row in path_metrics(samples,actual,float(window.close.iloc[-1]),interval):
                                rows.append({**item,**row})
                        print(json.dumps(dict(variant=variant,interval=interval,completed_origins=batch_start+len(batch),total_origins=len(group))),flush=True)
                        _write_json(rows_path,rows)
                all_metrics[variant] = dict(validation=aggregate([r for r in rows if r['split']=='validation']),
                    test=aggregate([r for r in rows if r['split']=='test']),
                    inference_seconds=sum(latency) if not resumed_origins else None,
                    new_work_inference_seconds=sum(latency),resumed_completed_origins=resumed_origins,
                    by_cadence={str(minutes):{split:aggregate([r for r in rows if r['split']==split and r['interval']==minutes*60])
                        for split in ('validation','test')} for minutes in (15,60)})
                del predictor
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
        finally:
            print('KRONOS_EVALUATION_GPU_LOCK_RELEASED' if device.startswith('cuda') else 'KRONOS_EVALUATION_CPU_DONE',flush=True)
    baseline = np.mean([v['path_pinball_bps'] for v in all_metrics['pretrained']['by_cadence']['60']['validation'].values()])
    adapted = np.mean([v['path_pinball_bps'] for v in all_metrics['adapted_validation_selected']['by_cadence']['60']['validation'].values()])
    selected = 'adapted_validation_selected' if training['best_epoch']>0 and adapted < baseline else 'pretrained'
    report = dict(status='complete',completed_at=now(),metrics=all_metrics,selected_variant=selected,
        selection='Lower mean H1 VALIDATION free-running path pinball across1/3/6/12/24h, matching the hourly runtime adapter; M15 and TEST are diagnostics and never select',
        selected_checkpoint=str(checkpoint) if selected!='pretrained' else str(TRADE_BOT/'models'/ASSETS['small']['directory']/'model.safetensors'),
        selected_finetuned_weights=selected!='pretrained',paths_per_origin=paths,origins_per_asset_cadence_split=origins,
        first_touch='Uncalibrated empirical samples; +1%/-1% fixed barriers, SL first on same-bar ambiguity; physically invalid OHLC paths excluded',
        caveats=['No historical macro/news supplied; exchange candles only.','Pretrained training overlap unknown.',
            'Retrospective paired audit with a modest fixed sample; no trading profitability or calibration established.'])
    _write_json(output/'comparison.json',report)
    return report


def input_window(market, clock, source_interval_seconds=None):
    frame = pd.read_csv(market['candles_path'])
    cadence = market.get('source_interval_seconds',source_interval_seconds)
    if cadence is not None:
        frame.attrs['interval_seconds'] = cadence
    _, stamps, gaps, interval = _validated_frame(frame)
    if interval not in (60,300,900,3600):
        raise ValueError('Supported source cadence is1/5/15/60minutes')
    quality = dict(missing_minutes=0,reconstruction_method=None,
        robustness='Gap-free training; reconstructed-input robustness is unvalidated and advisory only')
    if interval==60:
        from .council_technical import repair_m1_context
        frame, repaired = repair_m1_context(frame,clock.floor('h'),context_minutes=7800)
        quality.update(repaired)
        quality['affected_hourly_buckets'] = int(frame.loc[~frame.observed_mask,'timestamp'].dt.floor('60min').nunique())
        _, stamps, gaps, interval = _validated_frame(frame)
    completed = frame.loc[stamps+pd.Timedelta(seconds=interval)<=clock].copy()
    completed.index = pd.to_datetime(completed.timestamp,utc=True)
    grouped = completed.resample('60min',closed='left',label='left')
    hourly = grouped.agg(dict(open='first',high='max',low='min',close='last',volume='sum'))
    hourly = hourly.loc[grouped.size().eq(int(3600/interval))].reset_index()
    hourly.attrs['interval_seconds'] = 3600
    window, timing = causal_window(hourly,clock,128)
    if timing['bar_age_seconds'] >= 3600:
        raise ValueError('Latest completed hourly candle is stale')
    return window, timing, quality


def predict(request, checkpoint=None, device='cpu', paths=8):
    before = time.perf_counter()
    if not isinstance(request,dict) or not isinstance(request.get('markets'),list) or not request['markets']:
        raise ValueError('request requires nonempty markets list')
    settings = request.get('settings',{})
    paths = settings.get('paths',paths)
    threads = settings.get('cpu_threads',2)
    if isinstance(paths,bool) or not isinstance(paths,int) or not 2 <= paths <= 64:
        raise ValueError('paths must be an integer in2..64')
    if isinstance(threads,bool) or not isinstance(threads,int) or not 1 <= threads <= 8:
        raise ValueError('cpu_threads must be an integer in1..8')
    if settings.get('context',128) != 128 or settings.get('prediction',24) != 24 or settings.get('interval_minutes',60) != 60:
        raise ValueError('Kronos council adapter requires context128/prediction24/interval_minutes60')
    clock = pd.Timestamp(request['asof'])
    if clock.tzinfo is None:
        raise ValueError('asof must include timezone')
    windows, clocks, symbols, qualities = [], [], [], []
    for market in request['markets']:
        window, timing, quality = input_window(market,clock,settings.get('interval_seconds'))
        windows.append(window)
        clocks.append(timing)
        symbols.append(market['symbol'])
        qualities.append(quality)
    input_preparation_seconds = time.perf_counter()-before
    with exclusive(LOCK,timeout=3600) if device.startswith('cuda') else contextlib.nullcontext():
        loading_started = time.perf_counter()
        predictor = DirectKronos(device=device,checkpoint=checkpoint)
        variant = 'finetuned' if checkpoint and predictor.provenance.get('checkpoint_sha256') != ASSETS['small']['weights'] else 'pretrained'
        if device=='cpu':
            torch.set_num_threads(threads)
        load_seconds = time.perf_counter()-loading_started
        started = time.perf_counter()
        raw = draws(predictor,windows,3600,paths,20261022)
        seconds = time.perf_counter()-started
    forecasts = {}
    for symbol,window,timing,samples,quality in zip(symbols,windows,clocks,raw,qualities):
        last = float(window.close.iloc[-1])
        horizons = {str(hour):summarize_close_paths(samples[:,:hour],last) for hour in HOURS}
        forecasts[symbol] = dict(status='ok',source='kronos',model_id='kronos-small-council-20261022',
            pretrained_or_finetuned=variant,clock=timing,
            reference_close=last,horizons=horizons,quality=summarize_paths(samples,last),
            input_quality=quality,
            probability_calibration='uncalibrated empirical sample frequencies; advisory only')
    return dict(status='ok',model='kronos',asof=clock.isoformat(),forecast_origin='completed hourly candles',
        forecasts=forecasts,pretrained_or_finetuned=variant,input_preparation_seconds=input_preparation_seconds,
        load_seconds=load_seconds,inference_seconds=seconds,total_seconds=time.perf_counter()-before,
        adapter_module='wallex_lab.council_kronos',checkpoint=str(checkpoint) if checkpoint else None,
        inference_settings=dict(context=128,prediction=24,interval_minutes=60,paths=paths,cpu_threads=threads),
        checkpoint_sha256=_sha256(checkpoint) if checkpoint else None,model_available_at=now(),promotion_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('train','evaluate','predict'))
    parser.add_argument('--request')
    parser.add_argument('--output')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--paths',type=int,default=8)
    parser.add_argument('--resume',action='store_true',help='Resume completed paired audit batches with unchanged origins/seeds')
    args = parser.parse_args()
    try:
        if args.mode=='train':
            _, paths = reference_frames()
            OUTPUT.mkdir(parents=True,exist_ok=True)
            status('queued',data_manifest=str(MANIFEST),data_manifest_sha256=_sha256(MANIFEST),
                adapter_module='wallex_lab.council_kronos',promotion_allowed=False)
            result = train(paths,OUTPUT/'training',prediction_hours=24,context=128,stride=16,
                eval_stride=96,epochs=8,batch_size=4,accumulation=4,patience=2,learning_rate=2e-5,
                seed=20261022,device=args.device,gpu_lock=LOCK,status_callback=status)
            status('evaluating',checkpoint=result['selected_weights'],best_epoch=result['best_epoch'],
                token_metrics=dict(baseline=result['baseline'],final=result['final']))
            result = evaluate(device=args.device,paths=16)
            status('evaluating',comparison_path=str(OUTPUT/'comparison.json'),selected_checkpoint=result['selected_checkpoint'],
                selected_variant=result['selected_variant'],message='Path comparison ready; CPU adapter benchmark pending')
        elif args.mode=='evaluate':
            result = evaluate(device=args.device,paths=args.paths,resume=args.resume)
        else:
            result = predict(json.loads(Path(args.request).read_text()),args.checkpoint,args.device,args.paths)
        if args.output:
            _write_json(Path(args.output),result)
        print(json.dumps(result,allow_nan=False),flush=True)
    except BaseException as error:
        if args.mode!='predict':
            status('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    main()
