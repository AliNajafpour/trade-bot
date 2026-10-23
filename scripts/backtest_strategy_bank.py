"""Standalone rule ablation on native Wallex candles; hypothetical bid-path fills."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.data import aggregate_complete
from wallex_lab.first_touch import FirstTouchBook
from wallex_lab.strategy_bank import RULES,scan_bars


def replay(source,seed,output,hours=24):
    source=Path(source);seed=json.loads(Path(seed).read_text(encoding='utf-8'))
    state=json.loads((source/'runtime-state.json').read_text(encoding='utf-8'))
    fee=float(seed['fee_bps']['USDT']);paths={};frames={};receipts={}
    for symbol in state['symbols']:
        base=source/'market-data'/symbol.replace('/','')/'snapshots'
        for directory in sorted(base.iterdir(),reverse=True):
            p=directory/'reference.csv';r=directory/'provenance.json'
            if p.is_file() and r.is_file():
                receipt=json.loads(r.read_text(encoding='utf-8'))
                if 'wallex' not in receipt['source'].lower():raise ValueError('Wallex-only replay required')
                frame=pd.read_csv(p,parse_dates=['timestamp']);frame.timestamp=pd.to_datetime(frame.timestamp,utc=True)
                frame.attrs['interval_seconds']=60
                paths[symbol]=dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),provenance=receipt)
                frames[symbol]=frame;receipts[symbol]=state['quotes'][symbol];break
    end=min(f.timestamp.iloc[-1]+pd.Timedelta(minutes=1) for f in frames.values()).floor('min')
    start=end-pd.Timedelta(hours=hours);all_rows=[]
    for scenario,multiplier in [('fees_only',0),('recorded_spread_proxy',1),('double_spread_stress',2)]:
        stats=defaultdict(lambda:dict(setups=0,closed_trades=0,wins=0,losses=0,open_at_end=0,censored=0,returns=[]))
        for symbol,frame in frames.items():
            q=receipts[symbol];spread=multiplier*(q['ask']/q['bid']-1)
            book=FirstTouchBook(frame.loc[frame.timestamp+pd.Timedelta(minutes=1)<=end].copy())
            for tf in (5,15,60):
                bars,_=aggregate_complete(frame,tf*60);next_free={}
                for i in range(len(bars)):
                    at=bars.timestamp.iloc[i]+pd.Timedelta(minutes=tf)
                    if not start<=at<end-pd.Timedelta(minutes=2):continue
                    close=float(bars.close.iloc[i]);quote=dict(bid=close,ask=close*(1+spread))
                    _,candidates=scan_bars(bars.iloc[max(0,i-240):i+1],symbol,tf,at.isoformat(),quote)
                    for c in candidates:
                        name=c['strategy_sources'][0]['strategy'];key=(name,tf);s=stats[key];s['setups']+=1
                        if at<next_free.get(name,start):continue
                        entry_at=at+pd.Timedelta(minutes=1)
                        entry_bar=frame.loc[frame.timestamp==entry_at]
                        if entry_bar.empty:s['censored']+=1;next_free[name]=end;continue
                        entry=float(entry_bar.open.iloc[0])*(1+spread)
                        origin=entry_at-pd.Timedelta(minutes=1)
                        outcome=book.label(origin,c['target_price'],c['stop_price'],int((end-origin).total_seconds()/60),fee,
                                           origin_entry_price=entry)
                        if not outcome['known_outcome']:
                            s['censored']+=1;next_free[name]=end;continue
                        next_free[name]=pd.Timestamp(outcome['exit_at'])
                        if outcome['class_index']==2:s['open_at_end']+=1;continue
                        value=float(outcome['net_return']);s['returns'].append(value);s['closed_trades']+=1
                        s['wins']+=int(value>0);s['losses']+=int(value<=0)
            print(scenario,symbol,'complete',flush=True)
        for name,_,_ in RULES:
            for tf in (5,15,60):
                s=stats[(name,tf)];returns=s.pop('returns')
                all_rows.append(dict(scenario=scenario,strategy=name,timeframe_minutes=tf,**s,
                    mean_closed_trade_net_return=float(np.mean(returns)) if returns else None,
                    profit_factor=float(sum(r for r in returns if r>0)/-sum(r for r in returns if r<0)) if any(r<0 for r in returns) else None))
    report=dict(period_start=start.isoformat(),period_end=end.isoformat(),hours=hours,fee_bps=fee,
        sources=paths,spread_receipts=receipts,results=all_rows,
        interpretation='Exploratory standalone rule ablation, not the AI/news manager or an executed portfolio. Non-overlapping per symbol/rule/timeframe; no forced time exit; unresolved positions remain open.',
        limitations=['Historical BBO unavailable: M1 prices proxy bids; recorded current spread is a constant scenario, not historical spread.',
            'Fees are the actual seed fee schedule; slippage is zero, so fills remain optimistic. Gap/invalid-entry outcomes are censored, never counted as wins.',
            'Returns are per closed trade, not account P&L; open and censored outcomes prevent profitability conclusions. No selection/promotion from this period.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--seed',required=True);p.add_argument('--output',required=True);p.add_argument('--hours',type=int,default=24)
    a=p.parse_args()
    if not 1<=a.hours<=168:p.error('Use 1..168 replay hours')
    replay(a.source,a.seed,a.output,a.hours)
