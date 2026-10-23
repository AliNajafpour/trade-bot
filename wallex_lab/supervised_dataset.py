"""Shared causal path-extreme dataset for supervised tabular and sequence models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_market
from .features import make_features,_validated_frame

HORIZONS=(1,3,6,12,24)
QUANTILES=(.5,.75,.9)
TARGET_NAMES=('log_future_high_over_reference','log_reference_over_future_low','log_terminal_over_reference')
SEQUENCE_NAMES=('log_open_over_reference','log_high_over_reference','log_low_over_reference','log_close_over_reference','log1p_volume','hour_sin','hour_cos','dow_sin','dow_cos')
SYMBOLS=('BTCUSDT','ETHUSDT','SOLUSDT','SUSDT')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def summary_features(frame,symbol):
    """Finite-window features: no fitted scaler, future fill, or infinite EMA state."""
    if symbol not in SYMBOLS:raise ValueError('Unknown asset')
    features=make_features(frame);features=features.drop(columns=[c for c in features if c.startswith('ema_gap_')])
    close=frame.close
    for width in (24,48,96):
        features['return_'+str(width)]=close.pct_change(width,fill_method=None)
        features['high_distance_'+str(width)]=frame.high.rolling(width).max()/close-1
        features['low_distance_'+str(width)]=frame.low.rolling(width).min()/close-1
    clock=pd.DatetimeIndex(frame.timestamp)+pd.Timedelta(minutes=15)
    hour=(clock.hour+clock.minute/60)*2*np.pi/24;day=clock.dayofweek*2*np.pi/7
    features['hour_sin']=np.sin(hour);features['hour_cos']=np.cos(hour);features['dow_sin']=np.sin(day);features['dow_cos']=np.cos(day)
    features['log1p_volume']=np.log1p(frame.volume)
    for asset in SYMBOLS:features['asset_'+asset]=float(asset==symbol)
    return features.replace([np.inf,-np.inf],np.nan).fillna(0.)


def sequence_window(window):
    reference=float(window.close.iloc[-1]);clock=pd.DatetimeIndex(window.timestamp)+pd.Timedelta(minutes=15)
    hour=(clock.hour+clock.minute/60)*2*np.pi/24;day=clock.dayofweek*2*np.pi/7
    return np.column_stack((np.log(window[['open','high','low','close']].to_numpy()/reference),np.log1p(window.volume),
        np.sin(hour),np.cos(hour),np.sin(day),np.cos(day))).astype(np.float32)


def live_inputs(frame,symbol,context=128):
    """Caller supplies completed M15 bars ending at the actual decision clock."""
    if len(frame)<context or context<128:raise ValueError('At least128 completed M15 bars required')
    window=frame.iloc[-context:].copy();window.attrs['interval_seconds']=900
    _,_,gaps,interval=_validated_frame(window)
    if interval!=900 or gaps.any():raise ValueError('Contiguous completed M15 context required')
    summary=summary_features(window,symbol)
    return summary.iloc[-1].to_numpy(np.float32),sequence_window(window),list(summary.columns)


def build_dataset(root,output,context=128,eligibility_context=512):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Preserve dataset artifacts')
    manifest=json.loads((root/'manifest.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    if manifest['status']!='ready' or sha(root/'protocol.json')!=manifest['protocol_sha256']:raise ValueError('Verified complete frozen source required')
    if context<128 or eligibility_context<context:raise ValueError('Invalid context sizes')
    bounds=protocol['common_clock_boundaries'];train_start=pd.Timestamp(bounds['train_start']);train_end=pd.Timestamp(bounds['train_end'])
    calibration_start=train_start+(train_end-train_start)*.7;inner_stop_start=train_start+(calibration_start-train_start)*.8
    xs=[];sequences=[];ys=[];known=[];rows=[];sources=[];rejected=Counter();names=None
    for symbol in SYMBOLS:
        source=manifest['symbols'][symbol]['data'];m15=source['derived']['15m'];m1path=source['native_path']
        if sha(m1path)!=source['native_sha256'] or sha(m15['path'])!=m15['sha256']:raise ValueError('Source hash mismatch')
        frame=load_market(m15['path'],900);minute=load_market(m1path,60);_validated_frame(minute);_validated_frame(frame)
        sources.append(dict(symbol=symbol,m1_path=m1path,m1_sha256=source['native_sha256'],m15_path=m15['path'],m15_sha256=m15['sha256']))
        summary=summary_features(frame,symbol);names=list(summary.columns)
        clocks=pd.DatetimeIndex(frame.timestamp).as_unit('ns').asi8;ends=clocks+900_000_000_000
        native=pd.DatetimeIndex(minute.timestamp).as_unit('ns').asi8;high=minute.high.to_numpy();low=minute.low.to_numpy();close=minute.close.to_numpy()
        bad=np.r_[0,np.cumsum(np.diff(native)!=60_000_000_000)]
        for index in range(eligibility_context-1,len(frame)):
            at=pd.Timestamp(int(ends[index]),tz='UTC');split=next((s for s in ('train','validation','test') if pd.Timestamp(bounds[s+'_start'])<=at<pd.Timestamp(bounds[s+'_end'])),None)
            if split is None:continue
            stride=1 if split=='train' else 6
            if at.minute or at.second or at.hour%stride:continue
            beginning=clocks[index-eligibility_context+1];end=pd.Timestamp(bounds[split+'_end'])
            if beginning<pd.Timestamp(bounds[split+'_start']).value or at+pd.Timedelta(hours=24)>end:rejected['split_context_or_future_purge']+=1;continue
            if clocks[index]-beginning!=(eligibility_context-1)*900_000_000_000:rejected['context_gap']+=1;continue
            role=split
            if split=='train':
                if at+pd.Timedelta(hours=24)<=inner_stop_start:role='base_fit'
                elif at>=inner_stop_start and at+pd.Timedelta(hours=24)<=calibration_start:role='inner_stop'
                elif at>=calibration_start:role='calibration'
                else:rejected['inner_train_label_purge']+=1;continue
            reference=float(frame.close.iloc[index]);target=np.full((len(HORIZONS),3),np.nan,np.float32);valid=np.zeros(len(HORIZONS),bool)
            start=at.value+60_000_000_000;i=int(np.searchsorted(native,start))
            for hindex,hours in enumerate(HORIZONS):
                stop=at.value+hours*3_600_000_000_000;j=int(np.searchsorted(native,stop))
                if i>=len(native) or native[i]!=start or j<=i or native[j-1]+60_000_000_000!=stop or bad[j-1]!=bad[i]:continue
                target[hindex]=np.log([high[i:j].max()/reference,reference/low[i:j].min(),close[j-1]/reference]);valid[hindex]=True
            xs.append(summary.iloc[index].to_numpy(np.float32));sequences.append(sequence_window(frame.iloc[index-context+1:index+1]));ys.append(target);known.append(valid)
            rows.append(dict(origin_id=symbol+'-'+str(at.value),symbol=symbol,decision_at=at.isoformat(),origin_ns=at.value,split=split,
                training_role=role,reference_close=reference,source_m1=m1path,source_m15=m15['path']))
    if not rows:raise ValueError('No eligible source origins')
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['origin_ns'],rows[i]['symbol']))
    output.mkdir(parents=True)
    arrays=dict(X_features=np.asarray(xs)[order],X_sequence=np.asarray(sequences)[order],y_targets=np.asarray(ys)[order],y_known=np.asarray(known)[order])
    ordered=[rows[i] for i in order];np.savez_compressed(output/'dataset.npz',**arrays)
    with (output/'rows.jsonl').open('w') as handle:
        for row in ordered:handle.write(json.dumps(row)+'\n')
    metadata=dict(source_root=str(root.resolve()),protocol_sha256=sha(root/'protocol.json'),source_manifest_sha256=sha(root/'manifest.json'),
        dataset_sha256=sha(output/'dataset.npz'),rows_sha256=sha(output/'rows.jsonl'),context=context,eligibility_context=eligibility_context,
        feature_names=names,sequence_names=list(SEQUENCE_NAMES),horizon_hours=list(HORIZONS),quantiles=list(QUANTILES),target_names=list(TARGET_NAMES),
        calibration_start=calibration_start.isoformat(),inner_stop_start=inner_stop_start.isoformat(),counts=dict(Counter(row['training_role'] for row in ordered)),rejected=dict(rejected),
        known_target_counts=np.asarray(known)[order].sum(axis=0).tolist(),source_files=sources,source_code_sha256=sha(__file__),
        targets='Actual M1 future bars start strictly next minute and end at horizon; missing future labels remain NaN plus mask, never imputed.',
        training='Base fits first70%TRAIN with24h boundary purge; later30%TRAIN exclusively first-touch calibration; VAL/TEST never fit.',
        prediction_clock='Features contain completed bars only; input summary has maximum96bar lookback; sequence128; eligibility512 complete bars inside outer split.')
    save(output/'metadata.json',metadata);print(json.dumps(metadata),flush=True);return metadata


def load_dataset(path):
    path=Path(path);metadata=json.loads((path/'metadata.json').read_text())
    if sha(path/'dataset.npz')!=metadata['dataset_sha256'] or sha(path/'rows.jsonl')!=metadata['rows_sha256']:raise ValueError('Dataset hash mismatch')
    arrays=dict(np.load(path/'dataset.npz',allow_pickle=False));rows=[json.loads(line) for line in (path/'rows.jsonl').read_text().splitlines()]
    return arrays,rows,metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/fresh-history-20261022');parser.add_argument('--output',default='runs/supervised-dataset-20261022')
    args=parser.parse_args();build_dataset(args.root,args.output)
