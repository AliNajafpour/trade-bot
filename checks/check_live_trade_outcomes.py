"""Closed-trade labels use confirmed fills only; no network or exchange actions."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.live_execution import DEFAULTS,Gateway


def check():
    with TemporaryDirectory() as temporary:
        gateway=Gateway(Path(temporary),dict(DEFAULTS,account_binding='fixture'),object())
        def order(key,side,quantity,price,*,parent=None,role=None,fee='0',asset='USDT',status='FILLED',original=None):
            original=original or quantity
            key=gateway._record(key,role or ('entry' if side=='BUY' else 'exit'),
                dict(symbol='BTCUSDT',side=side,type='LIMIT',quantity=original,price=price),
                dict(reason='target') if parent else dict(evidence=dict(technical_mode='weekly_late')),parent)
            response=dict(clientOrderId=key,symbol='BTCUSDT',side=side,type='LIMIT',origQty=original,
                executedQty=quantity,price=price,executedSum=str(Decimal(quantity)*Decimal(price)),status=status,active=False,
                fills=[dict(quantity=quantity,price=price,fee=fee,feeAsset=asset)] if Decimal(quantity) else [])
            gateway._set(key,'acknowledged',response)
            return key
        def latest(entry):
            return next((trade for trade in gateway.closed_trades() if trade['entry_id']==entry),None)
        try:
            entry=order('cash-fee','BUY','1','100',fee='1')
            exit_id=order('cash-fee-sell','SELL','1','110',parent=entry,fee='1.1')
            trade=latest(entry)
            assert Decimal(trade['cost_usdt'])==101 and Decimal(trade['proceeds_usdt'])==Decimal('108.9')
            assert Decimal(trade['pnl_usdt'])==Decimal('7.9') and abs(trade['net_return']-float(Decimal('7.9')/101))<1e-12
            assert trade['symbol']=='BTCUSDT' and trade['context']==gateway.order(entry)['context']
            assert trade['exit_reasons']==['target']
            before=[tuple(row) for row in gateway.db.execute('SELECT * FROM orders')]
            assert gateway.closed_trades()==gateway.closed_trades()
            assert before==[tuple(row) for row in gateway.db.execute('SELECT * FROM orders')]
            base=order('base-fee','BUY','1','100',fee='.01',asset='BTC')
            order('base-fee-sell','SELL','.98','110',parent=base,fee='.01',asset='BTC')
            assert Decimal(latest(base)['pnl_usdt'])==Decimal('7.8')
            split=order('partial-buy','BUY','.6','100',status='CANCELED',original='1')
            order('partial-sell','SELL','.2','105',parent=split,status='CANCELED',original='.4',fee='.1')
            assert latest(split) is None,'Open residual became a closed learning label'
            order('rest-sell','SELL','.4','95',parent=split,role='stop',fee='.2')
            assert Decimal(latest(split)['pnl_usdt'])==Decimal('-1.3')
            assert latest(split)['exit_reasons']==['target','native_stop']
            nofill=order('no-fill','BUY','0','100',status='CANCELED',original='1')
            assert latest(nofill) is None
            for state in ('sending','unknown','cancel_unknown'):
                gateway._set(exit_id,state)
                assert latest(entry) is None,'Unresolved exit produced a label: '+state
            gateway._set(exit_id,'acknowledged')
            original=gateway.order(exit_id)['response']
            for field,value in (('fills',[]),('executedSum','111'),('active',True),('status','NEW')):
                changed=deepcopy(original);changed[field]=value;gateway._set(exit_id,'acknowledged',changed)
                assert latest(entry) is None,'Incomplete or unsettled fill accepted: '+field
            changed=deepcopy(original);changed['fills'][0]['feeAsset']='UNKNOWN'
            gateway._set(exit_id,'acknowledged',changed)
            assert latest(entry) is None,'Unsupported fee currency was estimated'
            gateway._set(exit_id,'acknowledged',original)
            assert latest(entry) is not None
            gateway._set(entry,'unknown')
            assert latest(entry) is None,'Unresolved entry produced a label'
        finally:gateway.db.close()
    print('PASS settled real-fill labels: USDT/base fees, partial cancellations, split exits, open residuals, unresolved/missing fills, and idempotent read-only outcomes')


if __name__=='__main__':check()
