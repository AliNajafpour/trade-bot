"""Offline exchange-boundary checks; never uses an API key or the network."""
import asyncio
from copy import deepcopy
from contextlib import ExitStack
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.live_execution import Gateway, OrderNotFound, normalize, DEFAULTS


def stamp(): return datetime.now(timezone.utc).isoformat()
def snapshot():
    return dict(account_scope='credit_subaccount',account_binding='fixture',observed_at=stamp(),
        balances={'USDT':dict(available='100',total='100'),'BTC':dict(available='5',total='5')},
        markets={'BTCUSDT':dict(symbol='BTCUSDT',baseAsset='BTC',quoteAsset='USDT',stepSize=3,tickSize=2,minQty='.001',minNotional='1',isMarketTypeEnable=True)},
        fees={'BTCUSDT':dict(takerFeeRate='.0013',makerFeeRate='.0013')},
        quotes={'BTCUSDT':dict(ask=100.,bid=99.99,ask_quantity=5.,bid_quantity=5.,available_at=stamp())},open_orders=[])


class Exchange:
    allow_writes=True
    def __init__(self): self.orders={};self.writes=[];self.timeout=False;self.cancel_race=False;self.partial=False;self.pending=False;self.reject_stop=False
    async def snapshot(self):
        value=snapshot();value['open_orders']=[deepcopy(r) for r in self.orders.values() if r['active']]
        return value
    async def get(self,key):
        await asyncio.sleep(0)
        if key not in self.orders: raise OrderNotFound()
        return deepcopy(self.orders[key])
    async def place(self,payload):
        await asyncio.sleep(0)
        self.writes.append(deepcopy(payload))
        row=dict(payload,clientOrderId=payload['client_id'],origQty=payload['quantity'],executedQty='0',executedSum='0',fills=[],status='NEW',active=True,stopPrice=payload.get('stop_Price'))
        if payload['side']=='BUY':
            qty=Decimal(payload['quantity'])*(Decimal('.4') if self.partial else 1)
            row.update(executedQty=str(qty),executedSum=str(qty*100),executedPrice='100',status='NEW' if self.partial else 'FILLED',active=self.partial,
                fills=[dict(quantity=str(qty),price='100',fee=str(qty*100*Decimal('.0013')),feeAsset='USDT')])
        self.orders[payload['client_id']]=row
        if self.pending and payload['side']=='BUY': row.update(executedQty='0',executedSum='0',fills=[],status='NEW',active=True)
        if self.reject_stop and payload['type']=='STOP_LIMIT': row.update(status='REJECTED',active=False)
        if self.timeout: self.timeout=False;raise TimeoutError('secret must not be persisted')
        return deepcopy(row)
    async def cancel(self,key):
        row=self.orders[key]
        if self.cancel_race:
            total=Decimal(row['origQty'])*Decimal(row['price'])
            row.update(status='FILLED',active=False,executedQty=row['origQty'],executedSum=str(total),
                fills=[dict(quantity=row['origQty'],price=row['price'],fee=str(total*Decimal('.0013')),feeAsset='USDT')])
            self.cancel_race=False
        else: row.update(status='CANCELED',active=False)


def prepare(gateway,key='entry'):
    return gateway.prepare_entry(key,'BTCUSDT','.1','100','98','102',snapshot(),
        dict(decision_at=stamp(),horizon_minutes=1440,p_tp_first=.6,p_sl_first=.25,p_no_touch=.15,execution_ask=100.,tp=102.,sl=98.),risk_multiplier=1.)


async def check():
    assert normalize('1.2349',3)=='1.234'
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(g.db.close)
        key=prepare(g)
        try:
            g.prepare_entry('entry','BTCUSDT','.1','100','97','102',snapshot(),
                dict(decision_at=stamp(),horizon_minutes=1440,p_tp_first=.6,p_sl_first=.25,p_no_touch=.15,execution_ask=100.,tp=102.,sl=97.),risk_multiplier=1.)
        except ValueError: pass
        else: raise AssertionError('An intent silently reused different protective barriers')
        exchange.pending=True;await g.send(key)
        try: prepare(g,'second')
        except ValueError: pass
        else: raise AssertionError('Pending unfilled BUY escaped the exposure guard')
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(g.db.close)
        foreign=snapshot();foreign['open_orders']=[dict(clientOrderId='unrelated')]
        try:
            g.prepare_entry('foreign','BTCUSDT','.1','100','98','102',foreign,
                dict(decision_at=stamp(),horizon_minutes=1440,p_tp_first=.6,p_sl_first=.25,p_no_touch=.15,execution_ask=100.,tp=102.,sl=98.),risk_multiplier=1.)
        except ValueError: pass
        else: raise AssertionError('Foreign account orders were ignored')
        key=prepare(g);exchange.reject_stop=True
        try: await g.execute_entry(key)
        except RuntimeError: pass
        else: raise AssertionError('Rejected protection reported successful execution')
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
        other=Gateway(Path(temporary),cfg,exchange);cleanup.callback(g.db.close);cleanup.callback(other.db.close)
        key=prepare(g);await g.send(key)
        await asyncio.gather(g.protect_entry(key),other.protect_entry(key),return_exceptions=True)
        assert len(exchange.writes)==2, 'Concurrent protection submitted duplicate SELLs'
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(g.db.close)
        key=prepare(g)
        assert prepare(g)==key
        try: await g.send(key)
        except PermissionError: pass
        else: raise AssertionError('Disabled live gateway transmitted an order')
        assert not exchange.writes
        try: await g.monitor_once()
        except PermissionError: pass
        else: raise AssertionError('Disabled monitor admitted live execution')
        cfg['enabled']=True
        await g.send(key);await g.send(key)
        assert len(exchange.writes)==1
        await g.protect_entry(key)
        assert len(exchange.writes)==2 and exchange.writes[-1]['type']=='STOP_LIMIT'
        assert exchange.writes[-1]['quantity']=='.1' or exchange.writes[-1]['quantity']=='0.100'
        await g.protect_entry(key);assert len(exchange.writes)==2
        # A stop filled between detailed reconciliation and the account snapshot cannot leave stale equity usable.
        stale=snapshot();stale['markets']['ETHUSDT']=dict(stale['markets']['BTCUSDT'],symbol='ETHUSDT',baseAsset='ETH')
        stale['fees']['ETHUSDT']=stale['fees']['BTCUSDT'];stale['quotes']['ETHUSDT']=stale['quotes']['BTCUSDT']
        cfg['symbols']=[*cfg['symbols'],'ETHUSDT']
        try:
            g._risk(dict(g.order(key)['payload'],symbol='ETHUSDT'),g.order(key)['context'],stale)
        except ValueError: pass
        else: raise AssertionError('Locally active stop absent from exchange open orders escaped admission')
        exchange.cancel_race=True
        await g.close_position(key,'target')
        assert len(exchange.writes)==2, 'Exit sold again after protective stop filled during cancellation'
        assert not g.positions()
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(g.db.close)
        key=prepare(g);exchange.timeout=True
        await g.send(key)
        assert g.order(key)['state']=='unknown'
        # A process restart cannot turn an uncertain submission into a second buy.
        restarted=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(restarted.db.close)
        await restarted.send(key);assert len(exchange.writes)==1
        await restarted.reconcile();assert restarted.order(key)['response']['status']=='FILLED'
        await restarted.protect_entry(key)
        assert len(exchange.writes)==2
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();exchange.partial=True;g=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(g.db.close)
        key=prepare(g);await g.send(key)
        await g.reconcile();await g.protect_entry(key)
        assert exchange.orders[key]['status']=='CANCELED'
        assert exchange.writes[-1]['quantity']=='0.040', 'Stop covered unfilled entry quantity'
        # Exchange holds 5 BTC, but this bot may sell only its own filled .04 BTC.
        await g.close_position(key,'manager')
        assert exchange.writes[-1]['quantity']=='0.040'
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(g.db.close)
        key=prepare(g);await g.execute_entry(key)
        # An unsent exit persisted before a failed network read must never be revived after protection is restored.
        old=g._record('aborted-exit','exit',dict(symbol='BTCUSDT',side='SELL',type='LIMIT',quantity='0.100',price='99.99'),{},key)
        try: await g.send(old)
        except ValueError: pass
        else: raise AssertionError('A sell was allowed alongside an active protective stop')
        await g.close_position(key,'retry')
        assert g.order(old)['state']=='abandoned'
        count=len(exchange.writes);await g.send(old);assert len(exchange.writes)==count
        stop=next(r for r in exchange.orders.values() if r['active'])
        assert stop['type']=='STOP_LIMIT'
        wrong=deepcopy(stop);wrong['stopPrice']='97'
        try: g._observe(stop['clientOrderId'],wrong)
        except ValueError: pass
        else: raise AssertionError('Exchange trigger mismatch was accepted')
        await g.monitor_once()
        assert len(exchange.writes)==count, 'Recovery monitor duplicated existing protection'
    for fail_on in (1,2,3):
        with TemporaryDirectory() as temporary, ExitStack() as cleanup:
            cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
            cleanup.callback(g.db.close)
            key=prepare(g);await g.execute_entry(key)
            original=exchange.snapshot;calls=0
            async def failing_snapshot():
                nonlocal calls
                calls+=1
                if calls==fail_on: raise TimeoutError('transient pre-submission read failure')
                return await original()
            exchange.snapshot=failing_snapshot
            try: await g.close_position(key,'manager')
            except (TimeoutError,RuntimeError): pass
            else: raise AssertionError('Injected close failure was not reached')
            assert len([r for r in exchange.orders.values() if r['active'] and r['type']=='STOP_LIMIT'])==1, 'Pre-submission exit failure lost native protection'
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(g.db.close)
        key=prepare(g);await g.execute_entry(key);exchange.timeout=True
        try: await g.close_position(key,'manager')
        except RuntimeError: pass
        else: raise AssertionError('Uncertain exit did not halt')
        assert len(exchange.writes)==3, 'Uncertain exit caused a second sell or protective over-reservation'
        assert g.db.execute("SELECT count(*) FROM orders WHERE role='exit' AND state='unknown'").fetchone()[0]==1
    with TemporaryDirectory() as temporary, ExitStack() as cleanup:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True);exchange=Exchange();g=Gateway(Path(temporary),cfg,exchange)
        cleanup.callback(g.db.close)
        key=prepare(g);await g.execute_entry(key)
        original=exchange.snapshot
        async def target_snapshot():
            value=await original();value['quotes']['BTCUSDT'].update(bid=103.,ask=103.01)
            return value
        exchange.snapshot=target_snapshot
        status=await g.monitor_once()
        assert status['actions'][0]['action']=='exit_attempt'
        assert any(p['side']=='SELL' and p['type']=='LIMIT' and p['price']=='103.00' for p in exchange.writes)
        assert len([r for r in exchange.orders.values() if r['active'] and r['type']=='STOP_LIMIT'])==1, 'Unfilled target exit lost protection'
    print('live execution: disabled writes, pending exposure, idempotency, concurrent dispatch, timeout recovery, partial fills, native protection, stale orders and owned inventory OK')


if __name__=='__main__': asyncio.run(check())
