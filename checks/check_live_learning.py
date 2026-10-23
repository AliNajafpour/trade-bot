"""Offline genuine-entry provenance, forward-only adaptation and rollback checks."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.live_learning import LiveLearning,UNIT,source_scores
from wallex_lab.live_execution import DEFAULTS


def check():
    with TemporaryDirectory() as temporary:
        root=Path(temporary);at=datetime(2026,10,24,tzinfo=timezone.utc)
        learner=LiveLearning(root,DEFAULTS['symbols'],now=lambda:at.isoformat())
        assert learner.status()['source_weights']==UNIT and len(learner.status()['by_symbol'])==10
        assert learner.status()['settled_trades']==0
        prediction=dict(technical_mode='weekly_late',execution_ask=100,forecasts=[dict(p_tp_first=.9,p_sl_first=.05)])
        reports=dict(market=dict(status='ok',report_id='immutable-report',assessment=dict(market_bias='bearish',confidence=1.)),macro=None)
        trades=[]
        def trade(index,net=.02,observation=None):
            nonlocal at
            at+=timedelta(minutes=1)
            symbol=DEFAULTS['symbols'][index%10]
            observation=observation or learner.observation(symbol,prediction,reports)
            observation=deepcopy(observation);observation['symbol']=symbol
            folder=root/'reviews'/str(index);folder.mkdir(parents=True)
            packet=dict(mode='real_spot_user_armed',paper_only=False,symbol=symbol,position=None,
                adaptive_learning=dict(observation=observation))
            response=dict(action='enter');files={}
            for name,value in (('packet.json',packet),('final.json',response)):
                target=folder/name;target.write_text(json.dumps(value));files[name]=hashlib.sha256(target.read_bytes()).hexdigest()
            receipt=folder/'receipt.json';receipt.write_text(json.dumps(dict(status='ok',response=response,files_sha256=files)))
            at+=timedelta(minutes=1)
            row=dict(entry_id=str(index),symbol=symbol,cost_usdt='100',proceeds_usdt=str(100*(1+net)),
                pnl_usdt=str(net*100),net_return=net,context=dict(evidence=dict(manager_receipt_path=str(receipt),
                    manager_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())))
            trades.append(row)
            return row
        old=learner.observation(DEFAULTS['symbols'][0],prediction,reports)
        for i in range(20):
            row=trade(i);learner.sync(trades)
        assert learner.status()['candidate_id']==1 and learner.status()['source_weights']==UNIT
        # Outcomes of entries opened before candidate freeze never count as validation.
        row=trade(20,observation=old);learner.sync(trades);assert learner.status()['evaluation_trades']==0
        for i in range(21,31):trade(i);learner.sync(trades)
        status=learner.status()
        assert status['champion_version']==1 and status['status']=='advisory_active'
        assert status['source_weights']['weekly_late']>1 and status['source_weights']['news']<1
        assert all(.75<=value<=1.25 for value in status['source_weights'].values())
        assert all(status['by_symbol'][s]>0 for s in DEFAULTS['symbols'])
        learner.sync(trades);assert learner.status()['settled_trades']==31,'Restart/rescan duplicated trades'
        reopened=LiveLearning(root,DEFAULTS['symbols'],now=lambda:at.isoformat())
        assert reopened.status()['source_weights']==status['source_weights']
        for i in range(31,41):trade(i,net=-.02);reopened.sync(trades)
        assert reopened.status()['source_weights']==UNIT and reopened.status()['recent_changes'][-1]['rollback']
        # Missing/changed/public evidence is never silently reconstructed from current news.
        invalid=trade(50);path=Path(invalid['context']['evidence']['manager_receipt_path'])
        (path.parent/'packet.json').write_text('{}')
        before=reopened.status()['settled_trades'];reopened.sync(trades)
        assert reopened.status()['settled_trades']==before and reopened.status()['skipped_without_valid_entry_evidence']==1
        malformed=deepcopy(invalid);malformed['entry_id']='51';malformed['context']={};reopened.sync(trades+[malformed])
        assert reopened.status()['settled_trades']==before
        legacy=dict(technical_mode='legacy_ensemble',execution_ask=100,legacy_technical=dict(models={
            'kronos':dict(status='ok',close_quantiles_by_step=dict(q50=[102.])),
            'chronos2':dict(status='ok',close_quantiles_by_step=dict(q50=[98.]))},
            rules=dict(strategies=[dict(status='ok',stance='long'),dict(status='ok',stance='avoid_or_exit')])) )
        scores=source_scores(legacy,{})
        assert scores==dict(kronos=1.,chronos2=-1.,strategy_bank=0.)
        assert 'p_tp_first' not in scores
        breakeven=trade(60,net=0.);reopened.sync(trades)
        corrected=dict(breakeven,cost_usdt='200',proceeds_usdt='200')
        try:reopened.sync(trades[:-1]+[corrected])
        except ValueError:pass
        else:raise AssertionError('Changed closure with unchanged zero PnL was silently reused')
        try:reopened.sync(trades[1:])
        except ValueError:pass
        else:raise AssertionError('Previously settled label disappeared without blocking learning')
    print('PASS all-ten accounting, immutable entry receipts, 20 training + 10 genuinely future validations, bounded source weights, restart idempotence, rollback and no invented TP probabilities')


if __name__=='__main__':check()
