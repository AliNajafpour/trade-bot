"""Point-in-time joins for numeric crypto covariates; historical availability is explicit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHAINS={'BTCUSDT':'Bitcoin','ETHUSDT':'Ethereum','SOLUSDT':'Solana','SUSDT':'Sonic'}
FUNDAMENTAL_COLUMNS=['fund_tvl_usd','fund_log_tvl','fund_tvl_change_1d']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tvl_series(payload,receipt,lag_hours=48):
    """Historical date is measurement time, never proof of publication availability."""
    if type(payload) is not list or not payload: raise ValueError('Nonempty historical chain TVL array required')
    frame=pd.DataFrame(payload)
    if not {'date','tvl'}<=set(frame): raise ValueError('Historical chain TVL schema differs')
    frame=frame[['date','tvl']].copy()
    frame['measurement_at']=pd.to_datetime(pd.to_numeric(frame['date'],errors='raise'),unit='s',utc=True)
    frame['fund_tvl_usd']=pd.to_numeric(frame['tvl'],errors='raise')
    if frame.measurement_at.duplicated().any() or not np.isfinite(frame.fund_tvl_usd).all() or (frame.fund_tvl_usd<0).any():
        raise ValueError('Duplicate, nonfinite or negative TVL observation')
    frame=frame.sort_values('measurement_at').reset_index(drop=True)
    observed=pd.Timestamp(receipt['response_received_utc'])
    if (frame.measurement_at>observed).any(): raise ValueError('Future-dated TVL measurement')
    frame['fund_log_tvl']=np.log1p(frame.fund_tvl_usd)
    change=frame.fund_tvl_usd.pct_change(fill_method=None)
    consecutive=frame.measurement_at.diff().dt.total_seconds().between(1,86400*1.5)
    frame['fund_tvl_change_1d']=change.where(consecutive&np.isfinite(change))
    frame['assumed_available_at']=frame.measurement_at+pd.Timedelta(hours=lag_hours)
    frame['verified_available_at']=observed
    return frame.drop(columns=['date','tvl'])


def join_covariates(bars,facts,mode='retrospective_assumed_48h',max_age_hours=72):
    if mode not in ('retrospective_assumed_48h','prospectively_verified'): raise ValueError('Unknown availability mode')
    frame=bars.copy();frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True)
    if not frame.timestamp.is_monotonic_increasing or frame.timestamp.duplicated().any(): raise ValueError('Sorted unique market timestamps required')
    frame['decision_at']=frame.timestamp+pd.Timedelta(minutes=5)
    if facts is None or facts.empty:
        for name in FUNDAMENTAL_COLUMNS:frame[name]=np.nan
        frame['fund_available_at']=pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns, UTC]')
        frame['fund_measurement_at']=pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns, UTC]')
    else:
        right=facts.copy()
        if mode=='retrospective_assumed_48h':right['fund_available_at']=right['assumed_available_at']
        else:
            # The current download proves only the latest reported measurement known at receipt.
            right=right.sort_values('measurement_at').tail(1)
            right['fund_available_at']=right['verified_available_at']
        right=right.rename(columns={'measurement_at':'fund_measurement_at'})
        frame=pd.merge_asof(frame,right[['fund_available_at','fund_measurement_at']+FUNDAMENTAL_COLUMNS]
            .sort_values('fund_available_at'),left_on='decision_at',right_on='fund_available_at',direction='backward',
            tolerance=pd.Timedelta(hours=max_age_hours))
    frame['fund_missing']=frame['fund_tvl_usd'].isna().astype('int8')
    frame['fund_availability_verified']=((frame.fund_missing==0)&(mode=='prospectively_verified')).astype('int8')
    known=frame.fund_available_at.notna()
    if (frame.loc[known,'fund_available_at']>frame.loc[known,'decision_at']).any(): raise AssertionError('Future covariate leaked')
    return frame
