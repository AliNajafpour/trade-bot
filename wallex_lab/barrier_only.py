"""Shared-cash spot replay: TP/SL exits only; data-end positions remain OPEN."""
from collections import Counter
import heapq
import math
import numpy as np
import pandas as pd


def economics(stop_fraction,reward_multiple,fee_bps=13.,slippage_bps=0.,ratio_basis='gross'):
    s=float(stop_fraction);r=float(reward_multiple);f=fee_bps/10000;a=slippage_bps/10000
    if not all(math.isfinite(x) for x in (s,r,f,a)) or not 0<s<1 or r<=0 or not 0<=f<1 or not 0<=a<1 or ratio_basis not in ('gross','net'):raise ValueError('Invalid barrier economics')
    paid=(1+a)*(1+f);stop=1-s;net_stop=stop*(1-a)*(1-f)
    target=1+r*s if ratio_basis=='gross' else (paid+r*(paid-net_stop))/((1-a)*(1-f))
    reward=target*(1-a)*(1-f)/paid-1;loss=1-net_stop/paid
    return dict(stop_ratio=stop,target_ratio=target,net_reward=reward,net_loss=loss,net_rr=reward/loss,
        breakeven_win_rate=loss/(reward+loss) if reward>0 else None)


def first_barrier(book,entry_ns,end_ns,target,stop):
    i=int(np.searchsorted(book.ns,entry_ns));j=int(np.searchsorted(book.ns,end_ns))
    if i>=j or book.ns[i]!=entry_ns:return dict(kind='unknown',at=entry_ns,reason='missing_entry_minute')
    gaps=np.flatnonzero(np.diff(book.ns[i:j])!=60_000_000_000)
    gap=int(book.ns[i+gaps[0]]+60_000_000_000) if len(gaps) else None
    hits=np.flatnonzero((book.high[i:j]>=target)|(book.low[i:j]<=stop))
    k=i+int(hits[0]) if len(hits) else None
    if gap is not None and (k is None or gap<=book.ns[k]):return dict(kind='unknown',at=gap,reason='missing_minute')
    if k is None:
        if book.ns[j-1]+60_000_000_000<end_ns:return dict(kind='unknown',at=int(book.ns[j-1]+60_000_000_000),reason='missing_terminal_minutes')
        return dict(kind='open',at=end_ns,reason='open_at_data_end')
    opening=float(book.opens[k])
    if opening<=stop:price=opening;kind='sl';stamp=int(book.ns[k]);ambiguous=False
    elif opening>=target:price=target;kind='tp';stamp=int(book.ns[k]);ambiguous=False
    elif book.low[k]<=stop:price=stop;kind='sl';stamp=int(book.ns[k]+60_000_000_000);ambiguous=bool(book.high[k]>=target)
    else:price=target;kind='tp';stamp=int(book.ns[k]+60_000_000_000);ambiguous=False
    return dict(kind=kind,at=stamp,price=float(price),ambiguous_same_bar=ambiguous)


def simulate(candidates,books,start_at,end_at,*,stop_spec,reward_multiple,ratio_basis='gross',fee_bps=13.,slippage_bps=0.,initial_cash=100.,reviewer=None,cooldown_minutes=0.):
    start=pd.Timestamp(start_at);end=pd.Timestamp(end_at)
    if start.tzinfo is None or end.tzinfo is None or start>=end:raise ValueError('Aware increasing replay bounds required')
    start_ns=start.value;end_ns=end.value;f=fee_bps/10000;a=slippage_bps/10000
    economics(.01,reward_multiple,fee_bps,slippage_bps,ratio_basis)
    if not math.isfinite(initial_cash) or initial_cash<=0:raise ValueError('Positive finite cash required')
    if not math.isfinite(cooldown_minutes) or cooldown_minutes<0:raise ValueError('Nonnegative cooldown required')
    cash=float(initial_cash);positions={};all_positions=[];pending=[];fills=[];trades=[];unknown=[];skips=Counter();sequence=0;cooldowns={};reviews=[]
    def mark(symbol,at):
        b=books[symbol];i=int(np.searchsorted(b.ns+60_000_000_000,at,side='right'))-1
        return float(b.close[i]) if i>=0 else None
    def equity(at):return cash+sum(p['quantity']*p['entry_reference_price'] for p in positions.values())
    def advance(at):
        nonlocal cash
        while pending and pending[0][0]<=at:
            stamp,_,symbol=heapq.heappop(pending);p=positions[symbol];event=p['outcome']
            if event['kind']=='unknown':unknown.append(dict(symbol=symbol,at=pd.Timestamp(stamp,tz='UTC').isoformat(),reason=event['reason']));continue
            positions.pop(symbol);price=event['price']*(1-a);proceeds=p['quantity']*price*(1-f);fee=p['quantity']*price*f;cash+=proceeds
            trade={**{k:v for k,v in p.items() if k!='outcome'},'exit_at':pd.Timestamp(stamp,tz='UTC').isoformat(),'exit_ns':stamp,
                'exit_price':price,'exit_reason':event['kind'],'exit_fee':fee,'pnl':proceeds-p['cost_basis'],'ambiguous_same_bar':event['ambiguous_same_bar']}
            trades.append(trade);fills.append(dict(at_ns=stamp,side='sell',cash_after=cash,fee=fee,symbol=symbol))
            cooldowns[symbol]=stamp+int(cooldown_minutes*60_000_000_000)
    prepared=[]
    for row in candidates:
        try:
            at=pd.Timestamp(row['entry_at']);decision=pd.Timestamp(row['decision_at']);score=float(row['original_expected_net'])
            if at.tzinfo is None or decision.tzinfo is None or at<=decision or not math.isfinite(score):raise ValueError('Invalid candidate clock/score')
            if start_ns<=at.value<end_ns and score>0:prepared.append((at.value,-score,row['symbol'],row))
        except (ValueError,KeyError,TypeError):skips['invalid_candidate']+=1
    decision_accounts={}
    for at,_,symbol,row in sorted(prepared,key=lambda x:x[:3]):
        if reviewer is not None:
            decision_ns=pd.Timestamp(row['decision_at']).value
            if decision_ns not in decision_accounts:
                advance(decision_ns)
                decision_accounts[decision_ns]=dict(account_cash_quote=float(cash),account_position_count=float(len(positions)))
        advance(at)
        if unknown:skips['stopped_after_unknown_path']+=1;continue
        if symbol in positions:skips['same_symbol_position_open']+=1;continue
        if at<cooldowns.get(symbol,0):skips['cooldown']+=1;continue
        if len(positions)>=5:skips['position_cap']+=1;continue
        b=books[symbol];i=int(np.searchsorted(b.ns,at))
        if i>=len(b.ns) or b.ns[i]!=at:skips['missing_actual_entry']+=1;continue
        reference=float(b.opens[i]);entry=reference*(1+a)
        try:
            if stop_spec=='candidate':
                stop,target=float(row['stop_price']),float(row['target_price'])
                if row.get('exit_mode')=='barriers_only':
                    from .spot import protection_geometry
                    _,_,stop,target=protection_geometry(dict(exit_mode='barriers_only',
                        barrier_stop_fraction=row['stop_fraction'],barrier_target_fraction=row['target_fraction']),reference)
                if not 0<stop<entry<target:raise ValueError('Entry outside actual candidate barriers')
                reward=target*(1-a)*(1-f)/(entry*(1+f))-1;loss=1-stop*(1-a)*(1-f)/(entry*(1+f))
                if reward<=0 or reward/loss<float(row.get('minimum_net_rr',0)):raise ValueError('Cost adjusted geometry rejected')
                d=1-stop/reference;econ=dict(net_reward=reward,net_loss=loss,net_rr=reward/loss,breakeven_win_rate=loss/(reward+loss))
                if row.get('original_signal'):
                    from .spot import entry_signal_net_return
                    signal=row['original_signal'];net=entry_signal_net_return(signal,reference,fee_bps,slippage_bps)
                    if net<=0:skips['original_signal_no_longer_positive']+=1;continue
                fresh=row.get('fresh_entry_signal')
                if fresh:
                    if fresh['kind']=='kronos_point':net=float(fresh['terminal_close_mean'])/reference-1-2*(f+a)
                    elif fresh['kind']=='kronos_path_payoff':net=float(fresh['mean_scored_exit_price'])*(1-a)*(1-f)/(entry*(1+f))-1
                    else:raise ValueError('Unknown native fresh-entry signal')
                    if not math.isfinite(net):raise ValueError('Nonfinite native fresh-entry payoff')
                    if net<=0:skips['native_signal_no_longer_positive']+=1;continue
            else:
                d=float(row['learned_MAEq50']) if stop_spec=='learned' else float(stop_spec);econ=economics(d,reward_multiple,fee_bps,slippage_bps,ratio_basis)
                stop=reference*econ['stop_ratio'];target=reference*econ['target_ratio']
        except (ValueError,KeyError,TypeError):skips['invalid_learned_stop']+=1;continue
        paid=entry*(1+f);loss_per_unit=paid-stop*(1-a)*(1-f);account=equity(at)
        if stop_spec=='candidate' and row.get('exit_mode')!='barriers_only':loss_per_unit=reference*(d+2*(f+a))
        multiplier=float(row.get('risk_multiplier',1.));review=None
        if reviewer is not None:
            packet=dict(row,numeric_summary={**row.get('numeric_summary',{}),
                **decision_accounts[decision_ns],
                'max_position_fraction':.15,'risk_per_trade_fraction':.005})
            review=reviewer(packet);reviews.append(dict(origin_id=row['origin_id'],symbol=symbol,decision_at=row['decision_at'],**review))
            if not review.get('approved'):skips[review.get('reason','context_rejected')]+=1;continue
            multiplier=min(multiplier,float(review.get('risk_multiplier',0.)))
        if not math.isfinite(multiplier) or not 0<multiplier<=1:skips['invalid_risk_multiplier']+=1;continue
        expiry=row.get('expires_at')
        horizon=row.get('max_holding_minutes')
        try:
            stamp=pd.Timestamp(expiry) if expiry else None
            if stamp is not None and (stamp.tzinfo is None or pd.isna(stamp)):raise ValueError('Aware expiry required')
            if horizon is not None and (type(horizon) not in (int,float) or not math.isfinite(horizon) or horizon<=0):raise ValueError('Positive horizon required')
            expiry_ns=stamp.value if stamp is not None else (pd.Timestamp(row['decision_at']).value+int(horizon*60_000_000_000) if horizon else None)
        except (ValueError,TypeError,OverflowError):skips['invalid_expiry']+=1;continue
        if review and review.get('holding_minutes') and row.get('exit_mode')!='barriers_only':
            reviewed_expiry=at+int(review['holding_minutes']*60_000_000_000)
            expiry_ns=min(expiry_ns,reviewed_expiry) if expiry_ns else reviewed_expiry
        if expiry_ns is not None and expiry_ns<=at:skips['expired_before_entry']+=1;continue
        quantity=min(account*.15/reference,account*.005*multiplier/loss_per_unit,cash/paid)
        if quantity*reference<1:skips['cash_or_min_notional']+=1;continue
        basis=quantity*paid;charge=quantity*entry*f;cash-=basis
        if cash< -1e-8:raise ArithmeticError('Shared cash negative')
        outcome=first_barrier(b,at,min(end_ns,expiry_ns) if expiry_ns else end_ns,target,stop)
        if outcome['kind']=='open' and expiry_ns is not None and expiry_ns<=end_ns:
            k=int(np.searchsorted(b.ns,expiry_ns))-1
            if k<0 or b.ns[k]+60_000_000_000!=expiry_ns:outcome=dict(kind='unknown',at=expiry_ns,reason='missing_expiry_minute')
            else:outcome=dict(kind='expiry',at=expiry_ns,price=float(b.close[k]),ambiguous_same_bar=False)
        p=dict(origin_id=row['origin_id'],symbol=symbol,entry_at=pd.Timestamp(at,tz='UTC').isoformat(),entry_ns=at,entry_price=entry,entry_reference_price=reference,
            quantity=quantity,entry_fee=charge,cost_basis=basis,stop_price=stop,target_price=target,stop_fraction=d,planned_net_reward=econ['net_reward'],
            planned_net_loss=econ['net_loss'],planned_net_rr=econ['net_rr'],breakeven_win_rate=econ['breakeven_win_rate'],original_expected_net=float(row['original_expected_net']),outcome=outcome)
        positions[symbol]=p;all_positions.append(p);fills.append(dict(at_ns=at,side='buy',cash_after=cash,fee=charge,symbol=symbol))
        if outcome['kind']!='open':heapq.heappush(pending,(outcome['at'],sequence,symbol));sequence+=1
    advance(end_ns);fills.sort(key=lambda r:r['at_ns']);clock=np.arange(start_ns,end_ns+1,60_000_000_000,dtype=np.int64)
    values=np.full(len(clock),initial_cash);valid=np.ones(len(clock),bool);mark_valid=np.ones(len(clock),bool)
    if fills:
        ticks=np.array([r['at_ns'] for r in fills]);balance=np.array([r['cash_after'] for r in fills]);indices=np.searchsorted(ticks,clock,side='right')-1
        values=np.where(indices>=0,balance[np.maximum(indices,0)],initial_cash).astype(float)
    for p in all_positions:
        o=p['outcome'];exit_ns=o['at'] if o['kind'] in ('tp','sl','expiry') else end_ns+1
        left=int(np.searchsorted(clock,p['entry_ns']));right=int(np.searchsorted(clock,exit_ns));times=clock[left:right];b=books[p['symbol']]
        indices=np.searchsorted(b.ns+60_000_000_000,times,side='right')-1;safe=np.maximum(indices,0)
        okay=(indices>=0)&(times==b.ns[safe]+60_000_000_000)
        values[left:right]+=p['quantity']*b.close[safe]*(1-a)*(1-f);valid[left:right]&=okay;mark_valid[left:right]&=okay
        if o['kind']=='unknown':valid[int(np.searchsorted(clock,o['at'])):]=False
    observed=values[valid];dd=float(np.max(1-np.r_[initial_cash,observed]/np.maximum.accumulate(np.r_[initial_cash,observed]))) if len(observed) else None
    opens=[]
    for symbol,p in positions.items():
        price=mark(symbol,end_ns);net=p['quantity']*price*(1-a)*(1-f)
        opens.append({**{k:v for k,v in p.items() if k!='outcome'},'status':'unresolved_path' if p['outcome']['kind']=='unknown' else 'open',
            'end_mark_price':price,'net_liquidation_value':net,'unrealized_net_pnl':net-p['cost_basis']})
    complete=not unknown and bool(valid.all());realized=sum(t['pnl'] for t in trades);end_equity=float(values[-1]);wins=sum(t['pnl']>0 for t in trades);losses=sum(t['pnl']<0 for t in trades)
    summary=dict(initial_cash=initial_cash,entries=len(all_positions),closed_trades=len(trades),wins=wins,losses=losses,win_rate=wins/len(trades) if trades else None,
        realized_pnl=realized,open_positions=len(opens),unrealized_net_pnl=sum(p['unrealized_net_pnl'] for p in opens),cash=cash,
        end_net_liquidation_equity=end_equity if complete else None,total_net_pnl=end_equity-initial_cash if complete else None,
        return_pct=100*(end_equity/initial_cash-1) if complete else None,max_drawdown_pct=100*dd if complete and dd is not None else None,
        observed_prefix_drawdown_pct=100*dd if dd is not None else None,fees_paid=sum(r['fee'] for r in fills),unknown_paths=len(unknown),missing_mark_minutes=int((~mark_valid).sum()),
        unknown_equity_minutes=int((~valid).sum()),
        valid_for_comparison=complete,exit_counts=dict(Counter(t['exit_reason'] for t in trades)),skipped=dict(skips),
        fee_bps_per_side=fee_bps,slippage_bps_per_side=slippage_bps,stop_spec=stop_spec,reward_multiple=reward_multiple,ratio_basis=ratio_basis,
        timeout_exits=sum(t['exit_reason']=='expiry' for t in trades),open_marks='Hypothetical net liquidation mark only; no data-end sale or win/loss recorded.')
    return dict(summary=summary,trades=trades,open_positions=opens,unknown_paths=unknown,
        equity_clock_ns=clock,equity=values,equity_known=valid,reviews=reviews)
