"""Prospective, paper-only residual calibration with a frozen execution champion.

Counterfactual M1 labels measure price paths, not exchange fills or real profit.
The original model/risk rules remain the execution boundary.
"""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading

import numpy as np
import pandas as pd

from .context import save_json
from .first_touch import FirstTouchBook

FEATURE_NAMES=('bias','technical_tp_logit','technical_sl_logit','stop_distance','target_distance',
    'horizon','return_15m','volatility_60m','spread','market_bias','market_risk','market_confidence',
    'macro_bias','macro_risk','macro_confidence','market_trend','macro_trend')
DEFAULTS=dict(min_training_groups=32,min_evaluation_groups=16,evaluation_seconds=3600,
    minimum_brier_gain=.005,minimum_utility_gain=0.,learning_rate=.05,max_pending=50000)


def probabilities(value):
    a=np.asarray(value,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all() or (a<0).any() or not math.isclose(float(a.sum()),1.,abs_tol=1e-6):
        raise ValueError('Three normalized finite probabilities required')
    return np.maximum(a,1e-8)/np.maximum(a,1e-8).sum()


def residual_probabilities(base,x,weights):
    p=probabilities(base);x=np.asarray(x,float);w=np.asarray(weights,float)
    if x.shape!=(len(FEATURE_NAMES),) or w.shape!=(3,len(x)) or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError('Invalid residual model shape/values')
    logits=np.log(p)+np.clip(w@x,-2,2);logits-=logits.max()
    p=np.exp(logits)
    return p/p.sum()


def update_residual(weights,base,x,label,rate):
    if type(label) is not int or label not in (0,1,2) or not 0<float(rate)<=1:
        raise ValueError('Invalid residual update')
    w=np.asarray(weights,float);x=np.asarray(x,float)
    gradient=np.outer(residual_probabilities(base,x,w)-np.eye(3)[label],x)+.001*w
    gradient/=max(1.,float(np.linalg.norm(gradient)))
    return np.clip(w-float(rate)*gradient,-1,1)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _stamp(value):
    t=pd.Timestamp(value)
    if t.tzinfo is None:raise ValueError('Timezone-aware clock required')
    return t.tz_convert('UTC')


class AdaptiveCouncil:
    def __init__(self,root,settings=None,clock=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'learning.sqlite';self.lock=threading.RLock()
        self.clock=clock
        self.settings=dict(DEFAULTS,**(settings or {}))
        if self.settings.keys()!=DEFAULTS.keys():raise ValueError('Unknown adaptive setting')
        for key in ('min_training_groups','min_evaluation_groups','evaluation_seconds','max_pending'):
            if type(self.settings[key]) is not int or self.settings[key]<1:raise ValueError('Positive adaptive integer required')
        for key in ('minimum_brier_gain','minimum_utility_gain','learning_rate'):
            v=self.settings[key]
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('Invalid adaptive rate/gain')
        if self.settings['learning_rate']==0:raise ValueError('Positive learning rate required')
        if self.settings['min_evaluation_groups']>512:raise ValueError('Evaluation gate exceeds the 512-group window')
        with self._db() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY,body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY,symbol TEXT,role TEXT,origin TEXT,
                    end_at TEXT,split TEXT,version INTEGER,candidate INTEGER,status TEXT DEFAULT 'pending');
                CREATE TABLE IF NOT EXISTS examples(id TEXT PRIMARY KEY,gid TEXT,body TEXT,
                    status TEXT DEFAULT 'pending',outcome TEXT);
                CREATE INDEX IF NOT EXISTS pending_groups ON groups(status,end_at);
                CREATE INDEX IF NOT EXISTS group_examples ON examples(gid);
                CREATE TABLE IF NOT EXISTS evaluations(gid TEXT PRIMARY KEY,version INTEGER,candidate INTEGER,body TEXT);
                CREATE INDEX IF NOT EXISTS evaluation_versions ON evaluations(version,candidate);
                CREATE TABLE IF NOT EXISTS versions(version INTEGER PRIMARY KEY,body TEXT);
                CREATE TABLE IF NOT EXISTS candidates(version INTEGER PRIMARY KEY,body TEXT);''')
            row=db.execute('select body from state where id=1').fetchone()
            if row is None:
                w=np.zeros((3,len(FEATURE_NAMES))).tolist()
                self._save(db,dict(format=2,features=list(FEATURE_NAMES),settings=self.settings,
                    champion=w,challenger=w,unresolved=[0.]*len(FEATURE_NAMES),champion_unresolved=[0.]*len(FEATURE_NAMES),
                    candidate=w,candidate_unresolved=[0.]*len(FEATURE_NAMES),candidate_version=0,
                    champion_version=0,active=False,training_groups=0,evaluation_groups=0,
                    counters=dict(pending=0,mature=0,censored=0),last_evaluation_at=None,history=[],saturated=0))
            else:
                state=json.loads(row[0])
                if state.get('format')!=2 or state.get('features')!=list(FEATURE_NAMES) or state.get('settings')!=self.settings:
                    raise ValueError('Adaptive state identity differs; use a new run directory')

    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.path,timeout=5)
        try:
            with db:yield db
        finally:db.close()

    @staticmethod
    def _load(db):return json.loads(db.execute('select body from state where id=1').fetchone()[0])

    @staticmethod
    def _save(db,state):
        db.execute('insert or replace into state values(1,?)',(json.dumps(state,allow_nan=False),))

    @staticmethod
    def features(signal,market,reports,asof,frame=None):
        at=_stamp(asof);frame=pd.read_csv(market['candles_path']) if frame is None else frame
        times=pd.to_datetime(frame.timestamp,utc=True)
        frame=frame.loc[times+pd.Timedelta(minutes=1)<=at].tail(61)
        if len(frame)<16:raise ValueError('Observed pre-origin prices required')
        close=frame.close.to_numpy(float)
        p=probabilities([signal[k] for k in ('p_tp_first','p_sl_first','p_unresolved')])
        ref=float(signal['reference_price']);quote=market['quote']
        x=[1.,np.clip(np.log(p[0]/p[2])/5,-1,1),np.clip(np.log(p[1]/p[2])/5,-1,1),
           np.clip((1-signal['stop_price']/ref)/.03,0,2),np.clip((signal['target_price']/ref-1)/.12,0,2),
           np.clip(np.log1p(signal['horizon_minutes'])/np.log1p(1440),0,1),
           np.clip((close[-1]/close[-16]-1)/.02,-1,1),
           np.clip(np.std(np.diff(np.log(close)))/.01,0,1),
           np.clip((quote['ask']/quote['bid']-1)/.01,0,1)]
        for kind in ('market','macro'):
            r=reports.get(kind);values=[0.,0.,0.]
            if r and r.get('status') in ('ok','partial') and not r.get('invalidated_at'):
                if _stamp(r['generated_at'])<=at<_stamp(r['valid_until']):
                    a=r['assessment'];confidence=float(a['confidence'])
                    values=[{'bullish':1,'bearish':-1}.get(a['market_bias'],0)*confidence,
                            {'risk_on':1,'risk_off':-1}.get(a['risk_regime'],0)*confidence,confidence]
            x.extend(values)
        x.extend([x[6]*x[9],x[6]*x[12]])
        a=np.asarray(x,float)
        if not np.isfinite(a).all() or (np.abs(a)>2).any():raise ValueError('Invalid bounded adaptive features')
        return a

    @staticmethod
    def _timeout(signal,adjustment=0.):
        ref=float(signal['reference_price'])
        return float(np.clip(signal['unresolved_marked_net_return']+adjustment,
                             signal['stop_price']/ref-1,signal['target_price']/ref-1))

    @staticmethod
    def _returns(signal,quote,role,adjustment=0.):
        fee=float(quote.get('fee_bps',13))/10000
        slip=float(quote.get('slippage_bps',0))/10000
        ratio=(1-fee)/(1+fee)*(1-slip)/(1+slip)/quote['ask'] if role=='entry' else (1-slip)/quote['bid']
        # Same bid-referenced economics as entry_fee_signal/_holding_value; no second spread.
        return np.array([signal['target_price']*ratio-1,signal['stop_price']*ratio-1,
                         (1+AdaptiveCouncil._timeout(signal,adjustment))*signal['reference_price']*ratio-1])

    @staticmethod
    def _validate_market(market,at,fresh=True):
        source=market.get('reference_source',{})
        if 'wallex' not in source.get('source','').lower():
            raise ValueError('Adaptive learning requires Wallex-only observed prices')
        if _stamp(source['available_at'])>at:raise ValueError('Future candle receipt')
        if fresh and not 0<=(at-_stamp(market['quote']['available_at'])).total_seconds()<=30:
            raise ValueError('Adaptive quote is stale or in the future')

    def process(self,result,markets,reports,asof,role='entry'):
        with self.lock:
            if self.clock:asof=max(_stamp(asof),_stamp(self.clock())).isoformat()
            return self._process(result,markets,reports,asof,role)

    def _process(self,result,markets,reports,asof,role):
        if role not in ('entry','holding'):raise ValueError('Invalid learning role')
        at=_stamp(asof)
        for market in markets.values():self._validate_market(market,at)
        if _stamp(result['asof'])>at:raise ValueError('Future forecast cutoff')
        self.advance(markets,at.isoformat())
        if self.clock:at=max(at,_stamp(self.clock()))
        for market in markets.values():self._validate_market(market,at)
        output=deepcopy(result)
        with self.lock,self._db() as db:
            state=self._load(db)
            pending=state['counters']['pending']
            prepared=[]
            for row in output.get('results',[]):
                if row.get('status')!='ok' or row.get('symbol') not in markets:continue
                market=markets[row['symbol']];quote=dict(market['quote'])
                quote['fee_bps']=float(quote.get('fee_bps',13))
                history=pd.read_csv(market['candles_path'])
                for signal in row.get('signals',[]):
                    base=probabilities([signal[k] for k in ('p_tp_first','p_sl_first','p_unresolved')])
                    x=self.features(signal,market,reports,at.isoformat(),history)
                    champion=residual_probabilities(base,x,state['champion']) if state['active'] else base
                    challenger=residual_probabilities(base,x,state['candidate'])
                    base_timeout=float(signal['unresolved_marked_net_return'])
                    adjustment=float(np.clip(np.dot(state['champion_unresolved'],x),-.02,.02)) if state['active'] else 0.
                    alt_adjustment=float(np.clip(np.dot(state['candidate_unresolved'],x),-.02,.02))
                    adjusted_timeout=self._timeout(signal,adjustment)
                    candidate_timeout=self._timeout(signal,alt_adjustment)
                    returns=self._returns(signal,quote,role)
                    item=dict(symbol=row['symbol'],role=role,origin=at.floor('min').isoformat(),recorded_at=at.isoformat(),
                        horizon=int(signal['horizon_minutes']),entry_price=quote['ask'] if role=='entry' else quote['bid'],
                        fee_bps=quote['fee_bps'],slippage_bps=float(quote.get('slippage_bps',0)),reference=float(signal['reference_price']),
                        stop=float(signal['stop_price']),target=float(signal['target_price']),features=x.tolist(),
                        base=base.tolist(),champion=champion.tolist(),challenger=challenger.tolist(),
                        base_timeout=base_timeout,champion_timeout=adjusted_timeout if state['active'] else base_timeout,
                        challenger_timeout=candidate_timeout,
                        base_score=float(base@returns),champion_score=float(champion@self._returns(signal,quote,role,adjustment)),
                        challenger_score=float(challenger@self._returns(signal,quote,role,alt_adjustment)),
                        candidate_version=state['candidate_version'],
                        feature_asof=at.isoformat(),forecast_asof=result['asof'],
                        quote_available_at=quote['available_at'],candle_receipt=deepcopy(market['reference_source']),
                        candles_path=str(market['candles_path']),
                        report_versions={k:v.get('report_id') if v else None for k,v in reports.items()},
                        model_id=row.get('model_id'),signal_id=signal.get('signal_id'))
                    prepared.append(item)
                    if state['active']:
                        for k,v in zip(('p_tp_first','p_sl_first','p_unresolved'),champion):signal[k]=float(v)
                        signal['unresolved_marked_net_return']=adjusted_timeout
                        signal['expected_tp_contribution']=signal['p_tp_first']*signal['reward_net_return']
                        signal['expected_sl_contribution']=signal['p_sl_first']*signal['loss_net_return']
                        signal['expected_unresolved_mark_contribution']=signal['p_unresolved']*signal['unresolved_marked_net_return']
                        signal['expected_marked_net_return']=sum(signal[k] for k in ('expected_tp_contribution','expected_sl_contribution','expected_unresolved_mark_contribution'))
                        if signal.get('fee_bps')==0:signal['expected_marked_gross_return']=signal['expected_marked_net_return']
                    signal['adaptive_version']=state['champion_version']
            # Label only complete minutes after the prediction has actually been computed.
            recorded=max(at,_stamp(self.clock())) if self.clock else at
            for market in markets.values():self._validate_market(market,recorded)
            slot=int(recorded.timestamp())//900;groups={}
            for item in prepared:
                item.update(origin=recorded.floor('min').isoformat(),recorded_at=recorded.isoformat())
                key=_digest([item['symbol'],role,slot,item['horizon'],item['model_id']])
                groups.setdefault(key,[]).append(item)
            for gid,items in groups.items():
                if db.execute('select 1 from groups where id=?',(gid,)).fetchone():continue
                if pending+len(items)>self.settings['max_pending']:
                    state['saturated']+=1;continue
                end=(_stamp(items[0]['origin'])+pd.Timedelta(minutes=items[0]['horizon'])).isoformat()
                split='evaluation' if slot%4==3 else 'training'
                db.execute('insert into groups(id,symbol,role,origin,end_at,split,version,candidate) values(?,?,?,?,?,?,?,?)',
                    (gid,items[0]['symbol'],role,items[0]['origin'],end,split,state['champion_version'],state['candidate_version']))
                for index,item in enumerate(items):
                    db.execute('insert into examples(id,gid,body) values(?,?,?)',(_digest([gid,index]),gid,json.dumps(item,allow_nan=False)))
                pending+=len(items)
            state['last_asof']=recorded.isoformat()
            state['counters']['pending']=pending
            self._save(db,state)
        return output

    def advance(self,markets,asof):
        at=_stamp(asof);books={}
        with self.lock,self._db() as db:
            state=self._load(db)
            if state.get('last_asof') and at<_stamp(state['last_asof']):
                raise ValueError('Adaptive model cannot consume a historical origin after learning its future')
            state['last_asof']=at.isoformat()
            # Bounded work; whole horizon groups remain intact within the 2,000-example budget.
            groups=db.execute("select id,symbol,role,origin,end_at,split,version,candidate from groups where status='pending' and end_at<=? order by end_at limit 125",
                              (at.isoformat(),)).fetchall()
            used=0
            for gid,symbol,role,origin,end,split,version,candidate_version in groups:
                if symbol not in markets:continue
                self._validate_market(markets[symbol],at,fresh=False)
                rows=db.execute('select id,body from examples where gid=?',(gid,)).fetchall()
                if used+len(rows)>2000:break
                used+=len(rows)
                if symbol not in books:
                    frame=pd.read_csv(markets[symbol]['candles_path'])
                    frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True)
                    frame=frame.loc[frame.timestamp+pd.Timedelta(minutes=1)<=at].copy()
                    if frame.empty:continue
                    frame.attrs['interval_seconds']=60
                    books[symbol]=FirstTouchBook(frame)
                # A delayed source tail is still pending, regardless of any earlier barrier hit.
                # Finalize all geometries on equal horizon coverage; only interior gaps censor.
                if books[symbol].times[-1]+pd.Timedelta(minutes=1)<_stamp(end):continue
                outcomes=[];items=[json.loads(r[1]) for r in rows]
                for item in items:
                    outcome=books[symbol].label(item['origin'],item['target'],item['stop'],item['horizon'],item['fee_bps'],
                                               origin_entry_price=item['entry_price'])
                    if outcome['known_outcome']:
                        fee=item['fee_bps']/10000
                        slip=item.get('slippage_bps',0)/10000
                        ratio=(1-fee)/(1+fee)*(1-slip)/(1+slip) if role=='entry' else (1-slip)
                        outcome['proxy_net_return']=outcome['exit_price']/item['entry_price']*ratio-1
                    outcomes.append(outcome)
                known=all(o['known_outcome'] for o in outcomes)
                for (eid,_),outcome in zip(rows,outcomes):
                    state['counters']['pending']-=1
                    state['counters']['mature' if outcome['known_outcome'] else 'censored']+=1
                    db.execute('update examples set status=?,outcome=? where id=?',
                               ('mature' if outcome['known_outcome'] else 'censored',json.dumps(outcome),eid))
                db.execute('update groups set status=? where id=?',('mature' if known else 'censored',gid))
                if not known:continue
                if split=='training':
                    rate=self.settings['learning_rate']/math.sqrt(1+state['training_groups']/100)/len(items)
                    for item,outcome in zip(items,outcomes):
                        state['challenger']=update_residual(state['challenger'],item['base'],item['features'],outcome['class_index'],rate).tolist()
                        if outcome['class_index']==2:
                            x=np.array(item['features']);w=np.array(state['unresolved'])
                            truth=outcome['exit_price']/item['reference']-1
                            residual=np.clip(truth-item['base_timeout']-w@x,-.02,.02)
                            state['unresolved']=np.clip(w+rate*residual*x,-.02,.02).tolist()
                    state['training_groups']+=1
                else:
                    evaluation={}
                    for name in ('base','champion','challenger'):
                        evaluation[name+'_brier']=float(np.mean([np.square(np.array(item[name])-np.eye(3)[o['class_index']]).sum()
                            for item,o in zip(items,outcomes)]))
                        choice=max(range(len(items)),key=lambda i:items[i][name+'_score'])
                        evaluation[name+'_utility']=outcomes[choice]['proxy_net_return'] if items[choice][name+'_score']>0 else 0.
                    evaluation.update(asof=at.isoformat(),examples=len(items),interpretation='Counterfactual candle-path selection proxy, not executed portfolio P&L')
                    db.execute('insert into evaluations values(?,?,?,?)',(gid,version,candidate_version,json.dumps(evaluation)))
                    state['evaluation_groups']+=1
            self._consider_version(db,state,at)
            self._save(db,state)
        save_json(self.root/'status.json',self.status())

    def _consider_version(self,db,state,at):
        last=state['last_evaluation_at']
        if last and (at-_stamp(last)).total_seconds()<self.settings['evaluation_seconds']:return
        state['last_evaluation_at']=at.isoformat()
        if state['candidate_version']==0:
            if state['training_groups']>=self.settings['min_training_groups']:self._freeze_candidate(db,state,at)
            return
        rows=[json.loads(r[0]) for r in db.execute('select body from evaluations where version=? and candidate=? order by rowid desc limit 512',
                                                 (state['champion_version'],state['candidate_version']))]
        if not rows:return
        metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k.endswith(('_brier','_utility'))}
        state['evaluation']=dict(groups=len(rows),**metrics)
        if len(rows)<self.settings['min_evaluation_groups']:return
        gain=self.settings['minimum_brier_gain'];utility=self.settings['minimum_utility_gain']
        reason=None
        if state['active'] and metrics['champion_brier']>metrics['base_brier']+gain and metrics['champion_utility']<metrics['base_utility']-utility:
            state['champion']=np.zeros((3,len(FEATURE_NAMES))).tolist();state['champion_unresolved']=[0.]*len(FEATURE_NAMES)
            state['active']=False;reason='rollback_to_frozen_base'
        elif state['training_groups']>=self.settings['min_training_groups'] and metrics['challenger_brier']<metrics['champion_brier']-gain and metrics['challenger_utility']>max(0.,metrics['champion_utility']+utility):
            state['champion']=deepcopy(state['candidate']);state['champion_unresolved']=deepcopy(state['candidate_unresolved'])
            state['active']=True;reason='promote_exact_evaluated_checkpoint'
        if reason:
            state['champion_version']+=1
            event=dict(version=state['champion_version'],candidate=state['candidate_version'],candidate_sha256=state['candidate_sha256'],
                       at=at.isoformat(),reason=reason,evidence=dict(groups=len(rows),**metrics))
            state['history'].append(event)
            state['history']=state['history'][-100:]
            db.execute('insert into versions values(?,?)',(state['champion_version'],json.dumps(dict(event=event,weights=state['champion'],unresolved=state['champion_unresolved']))))
        self._freeze_candidate(db,state,at)

    @staticmethod
    def _freeze_candidate(db,state,at):
        state['candidate_version']+=1
        state['candidate']=deepcopy(state['challenger']);state['candidate_unresolved']=deepcopy(state['unresolved'])
        snapshot=dict(weights=state['candidate'],unresolved=state['candidate_unresolved'])
        state['candidate_sha256']=_digest(snapshot)
        snapshot.update(at=at.isoformat(),training_groups=state['training_groups'],sha256=state['candidate_sha256'])
        db.execute('insert into candidates values(?,?)',(state['candidate_version'],json.dumps(snapshot)))

    @staticmethod
    def _influence(weights):
        w=np.asarray(weights,float)
        values=np.array([1+np.linalg.norm(w[:,:9]),np.linalg.norm(w[:,[9,10,11,15]]),np.linalg.norm(w[:,[12,13,14,16]])])
        return dict(zip(('technical','market','macro'),(values/values.sum()).tolist()))

    def status(self):
        with self.lock,self._db() as db:
            state=self._load(db)
            counts=state['counters'];evaluation_groups=state['evaluation_groups']
        mode='adaptive_active' if state['active'] else 'warming_up'
        if state['history'] and state['history'][-1]['reason'].startswith('rollback'):mode='rolled_back'
        return dict(mode=mode,champion_version=state['champion_version'],training_groups=state['training_groups'],
            evaluation_groups=evaluation_groups,pending_examples=counts.get('pending',0),mature_examples=counts.get('mature',0),
            censored_examples=counts.get('censored',0),saturated_origins=state['saturated'],settings=self.settings,
            active_input_influence=self._influence(state['champion']),learning_input_influence=self._influence(state['challenger']),
            candidate_version=state['candidate_version'],candidate_sha256=state.get('candidate_sha256'),
            influence_interpretation='Normalized coefficient magnitudes plus the frozen technical prior; descriptive sensitivity, not causal attribution or LLM-declared weights',
            evaluation=state.get('evaluation'),version_history=state['history'][-10:],
            outcome_interpretation='Matured continuous Wallex M1 barrier outcomes; unresolved is a horizon mark, not a closed trade; all utility is counterfactual')
