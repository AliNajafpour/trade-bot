"""Small, persistent source-weight learner from settled bot fills; no order capability.

Directional agreement with realized net PnL is observational, not causal attribution
or a calibrated TP/profit probability. Base forecasts and execution gates stay frozen.
"""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading

SOURCES=('weekly_late','barrier_pretrained8','kronos','chronos2','strategy_bank','news','macro')
UNIT={name:1. for name in SOURCES}
MIN_TRAIN=20
MIN_EVAL=10


def stamp():return datetime.now(timezone.utc).isoformat()


def clock(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Aware learning timestamp required')
    return result


def bounded(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('Finite learning evidence required')
    return max(-1.,min(1.,value))


def fused(scores,weights):
    return sum(value*weights[name] for name,value in scores.items())/sum(weights[name] for name in scores) if scores else 0.


def source_scores(prediction,reports):
    scores={};route=prediction.get('technical_mode','weekly_late')
    forecasts=prediction.get('forecasts') or []
    if route in ('weekly_late','barrier_pretrained8') and forecasts:
        scores[route]=bounded(float(forecasts[0]['p_tp_first'])-float(forecasts[0]['p_sl_first']))
    legacy=prediction.get('legacy_technical') or {}
    ask=float(prediction['execution_ask'])
    for name in ('kronos','chronos2'):
        model=legacy.get('models',{}).get(name,{})
        if model.get('status')=='ok':
            scores[name]=bounded((float(model['close_quantiles_by_step']['q50'][-1])/ask-1)/.02)
    rules=[r for r in legacy.get('rules',{}).get('strategies',[]) if r.get('status')=='ok']
    if rules:scores['strategy_bank']=sum({'long':1.,'avoid_or_exit':-1.}.get(r.get('stance'),0.) for r in rules)/len(rules)
    for kind,name in (('market','news'),('macro','macro')):
        report=reports.get(kind)
        if report and report.get('status') in ('ok','partial'):
            assessment=report['assessment']
            scores[name]=bounded({'bullish':1.,'bearish':-1.}.get(assessment['market_bias'],0.)*float(assessment['confidence']))
    return scores


def receipt_observation(root,trade):
    evidence=trade['context']['evidence']
    path=Path(evidence['manager_receipt_path']).resolve();allowed=(Path(root)/'reviews').resolve()
    if path.name!='receipt.json' or not path.is_relative_to(allowed):raise ValueError('Learning receipt outside this ledger')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=evidence['manager_receipt_sha256']:raise ValueError('Learning receipt changed')
    receipt=json.loads(raw)
    for name in ('packet.json','final.json'):
        source=(path.parent/name).resolve()
        if not source.is_relative_to(allowed):raise ValueError('Learning evidence escaped this ledger')
        data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=receipt['files_sha256'][name]:raise ValueError('Learning input changed')
        if name=='packet.json':packet=json.loads(data)
        else:response=json.loads(data)
    if (receipt.get('status')!='ok' or response!=receipt['response'] or response.get('action')!='enter'
            or packet.get('mode')!='real_spot_user_armed' or packet.get('paper_only') is not False
            or packet.get('position') is not None or packet.get('symbol')!=trade['symbol']):
        raise ValueError('Only original real entry reviews can label a trade')
    return packet['adaptive_learning']['observation']


class LiveLearning:
    def __init__(self,root,symbols,now=stamp):
        self.root=Path(root);self.path=self.root/'adaptive-live.sqlite'
        self.symbols=list(symbols);self.now=now;self.lock=threading.RLock()
        with self.db() as db:
            db.executescript('CREATE TABLE IF NOT EXISTS learning_state(id INTEGER PRIMARY KEY,body TEXT);'
                'CREATE TABLE IF NOT EXISTS learning_trades(seq INTEGER PRIMARY KEY AUTOINCREMENT,entry_id TEXT UNIQUE,body TEXT);')
            if not db.execute('SELECT 1 FROM learning_state').fetchone():
                self.save(db,dict(format=1,weights=UNIT,version=0,next_candidate=1,candidate=None,
                    count=0,last_fit=0,watch=[],history=[],by_symbol={s:0 for s in self.symbols},skipped=0))

    @contextmanager
    def db(self):
        self.root.mkdir(parents=True,exist_ok=True)
        with self.lock:
            db=sqlite3.connect(self.path,timeout=5)
            try:
                with db:yield db
            finally:db.close()

    @staticmethod
    def load(db):return json.loads(db.execute('SELECT body FROM learning_state WHERE id=1').fetchone()[0])

    @staticmethod
    def save(db,state):db.execute('INSERT OR REPLACE INTO learning_state VALUES(1,?)',(json.dumps(state,allow_nan=False),))

    def observation(self,symbol,prediction,reports):
        with self.db() as db:state=self.load(db)
        scores=source_scores(prediction,reports);candidate=state['candidate']
        return dict(symbol=symbol,origin=self.now(),route=prediction.get('technical_mode','weekly_late'),
            scores=scores,champion_version=state['version'],weights=state['weights'],
            candidate_id=candidate['id'] if candidate else None,
            candidate_weights=candidate['weights'] if candidate else UNIT,
            weighted_direction=fused(scores,state['weights']),
            report_ids={k:v.get('report_id') if v else None for k,v in reports.items()},
            semantics='Directional evidence weights; not calibrated trade or TP-first probabilities')

    def sync(self,trades):
        with self.db() as db:
            state=self.load(db);skipped=0
            known={row[0] for row in db.execute('SELECT entry_id FROM learning_trades')}
            if known-{trade['entry_id'] for trade in trades}:
                raise ValueError('Previously learned trade is no longer fully settled; reconcile learning')
            for trade in trades:
                closure_digest=hashlib.sha256(json.dumps(trade,sort_keys=True,allow_nan=False).encode()).hexdigest()
                prior=db.execute('SELECT body FROM learning_trades WHERE entry_id=?',(trade['entry_id'],)).fetchone()
                if prior:
                    old=json.loads(prior[0])
                    if old['closure_digest']!=closure_digest:
                        raise ValueError('Settled outcome changed; learning requires reconciliation')
                    receipt_observation(self.root,trade)
                    continue
                try:
                    observation=receipt_observation(self.root,trade)
                    if observation['symbol'] not in self.symbols or observation['symbol']!=trade['symbol']:raise ValueError('Unknown learning symbol')
                    if clock(observation['origin'])>clock(self.now()):raise ValueError('Future learning origin')
                    scores=observation['scores']
                    if not scores or any(k not in SOURCES or bounded(v)!=v for k,v in scores.items()):raise ValueError('Invalid directional evidence')
                    net=float(trade['net_return'])
                    if not math.isfinite(net):raise ValueError('Invalid realized outcome')
                    for field in ('weights','candidate_weights'):
                        if set(observation[field])!=set(SOURCES) or any(not .75<=float(v)<=1.25 for v in observation[field].values()):
                            raise ValueError('Invalid stored source weights')
                except (KeyError,ValueError,TypeError,OSError,json.JSONDecodeError):
                    skipped+=1;continue
                label=1. if net>0 else -1. if net<0 else 0.
                item=dict(entry_id=trade['entry_id'],symbol=trade['symbol'],net_return=net,pnl_usdt=trade['pnl_usdt'],
                    first_observed_closed_at=self.now(),observation=observation,label=label,closure_digest=closure_digest)
                db.execute('INSERT INTO learning_trades(entry_id,body) VALUES(?,?)',(trade['entry_id'],json.dumps(item,allow_nan=False)))
                state['count']+=1;state['by_symbol'][trade['symbol']]=state['by_symbol'].get(trade['symbol'],0)+1
                # A candidate is frozen before its future validation entries are opened.
                candidate=state['candidate']
                if candidate and observation['candidate_id']==candidate['id'] and clock(observation['origin'])>=clock(candidate['created_at']):
                    candidate['evaluation'].append(item)
                    if len(candidate['evaluation'])>=MIN_EVAL:
                        evaluation=candidate['evaluation'];proposal=dict(state['weights'])
                        for name in SOURCES:
                            if sum(name in row['observation']['scores'] for row in evaluation)>=5:proposal[name]=candidate['weights'][name]
                        old=sum((fused(row['observation']['scores'],state['weights'])-row['label'])**2 for row in evaluation)/len(evaluation)
                        new=sum((fused(row['observation']['scores'],proposal)-row['label'])**2 for row in evaluation)/len(evaluation)
                        promoted=new<=old-.01
                        state['history'].append(dict(at=self.now(),candidate=candidate['id'],evaluation_count=len(evaluation),
                            baseline_directional_loss=old,candidate_directional_loss=new,promoted=promoted))
                        if promoted:state.update(weights=proposal,version=state['version']+1,watch=[])
                        state['candidate']=None;state['last_fit']=state['count']
                if state['version'] and observation['champion_version']==state['version']:
                    baseline=(fused(scores,UNIT)-label)**2;active=(fused(scores,observation['weights'])-label)**2
                    state['watch']=(state['watch']+[active-baseline])[-MIN_EVAL:]
                    if len(state['watch'])==MIN_EVAL and sum(state['watch'])/MIN_EVAL>.02:
                        state['history'].append(dict(at=self.now(),rollback=True,reason='Forward directional loss worsened versus equal weights'))
                        state.update(weights=dict(UNIT),version=state['version']+1,watch=[],candidate=None,last_fit=state['count'])
                if not state['candidate'] and state['count']>=MIN_TRAIN and state['count']-state['last_fit']>=10:
                    rows=[json.loads(r[0]) for r in db.execute('SELECT body FROM learning_trades ORDER BY seq DESC LIMIT 200')]
                    weights=dict(UNIT)
                    for name in SOURCES:
                        values=[row['observation']['scores'][name]*row['label'] for row in rows if name in row['observation']['scores']]
                        if len(values)>=8:weights[name]=1.+.25*sum(values)/(len(values)+10)
                    state['candidate']=dict(id=state['next_candidate'],created_at=self.now(),weights=weights,evaluation=[])
                    state['next_candidate']+=1;state['last_fit']=state['count']
            state['skipped']=skipped;state['history']=state['history'][-30:];self.save(db,state)
        return self.status()

    def status(self):
        with self.db() as db:state=self.load(db)
        candidate=state['candidate']
        return dict(enabled=True,status='advisory_active' if state['weights']!=UNIT else 'validating' if candidate else 'collecting',
            settled_trades=state['count'],by_symbol={s:state['by_symbol'].get(s,0) for s in self.symbols},
            source_weights=state['weights'],champion_version=state['version'],
            candidate_id=candidate['id'] if candidate else None,evaluation_trades=len(candidate['evaluation']) if candidate else 0,
            min_training_trades=MIN_TRAIN,min_forward_evaluation_trades=MIN_EVAL,
            skipped_without_valid_entry_evidence=state['skipped'],recent_changes=state['history'][-3:],
            base_model_weights_updated=False,
            limitations='Executed-trade selection bias and correlated inputs remain; directional loss is not proof of improved PnL.')
