"""Replay Wallex's triggered-stop shape and fee dust; no keys or network."""
import asyncio
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.live_execution import DEFAULTS
from wallex_lab.live_runner import LiveGateway
from check_live_execution import snapshot

class ReadOnlyExchange:
    allow_writes=True
    def __init__(self):self.orders={};self.writes=[]
    async def get(self,key):return deepcopy(self.orders[key])
    async def snapshot(self):
        s=snapshot();s['markets']={'SUSDT':dict(symbol='SUSDT',stepSize=2,tickSize=4,minQty='.01',minNotional='1')}
        s['balances']['S']=dict(available='.005875',total='.005875')
        s['fees']={'SUSDT':dict(takerFeeRate='.0013')}
        s['quotes']={'SUSDT':dict(s['quotes']['BTCUSDT'],bid=.0473,ask=.0475)}
        return s
    async def place(self,payload):raise AssertionError('Recovery must not submit a duplicate order or sell dust')
    async def cancel(self,key):raise AssertionError('Recovery must not cancel a settled order')

async def check():
    with TemporaryDirectory() as temporary:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True)
        exchange=ReadOnlyExchange();gateway=LiveGateway(Path(temporary),cfg,exchange)
        entry=gateway._record('entry','entry',dict(symbol='SUSDT',side='BUY',type='LIMIT',quantity='26.25',price='.0477'),
            dict(barriers=dict(stop_price='.0467',target_price='.0486'),evidence=dict(technical_mode='weekly_late')))
        stop=gateway._record('stop','stop',dict(symbol='SUSDT',side='SELL',type='STOP_LIMIT',quantity='26.21',price='.0466',stop_Price='.0467'),{},entry)
        buy=dict(clientOrderId=entry,symbol='SUSDT',side='BUY',type='LIMIT',price='.0477',origQty='26.25',
            executedQty='26.25',executedSum='1.252125',active=False,status='FILLED',
            fills=[dict(quantity='26.25',price='.0477',sum='1.252125',fee='.034125',feeAsset='S')])
        sold=dict(clientOrderId=stop,symbol='SUSDT',side='SELL',type='LIMIT',price='.0466',origQty='26.21',
            executedQty='26.21',executedSum='1.226628',active=False,status='FILLED',
            fills=[dict(quantity='26.21',price='.0468',sum='1.226628',fee='.0015946164',feeAsset='USDT')])
        exchange.orders={entry:buy,stop:sold}
        try:
            gateway._observe(entry,buy);gateway._set(stop,'unknown',error='ValueError')
            await gateway.reconcile()
            assert gateway.order(stop)['state']=='acknowledged','Confirmed triggered LIMIT fill remains unknown'
            for field,value in [('clientOrderId','foreign'),('symbol','BTCUSDT'),('side','BUY'),('price','.0465'),
                    ('origQty','26.22'),('fills',[]),('stopPrice','.0400'),('type','MARKET')]:
                wrong=deepcopy(sold);wrong[field]=value
                try:gateway._observe(stop,wrong)
                except (ValueError,KeyError):pass
                else:raise AssertionError('Stop identity/fill guard relaxed: '+field)
            fresh=gateway._record('fresh-stop','stop',dict(gateway.order(stop)['payload']),{},entry)
            waiting=dict(sold,clientOrderId=fresh,executedQty='0',executedSum='0',fills=[],active=True,status='NEW')
            try:gateway._observe(fresh,waiting)
            except ValueError:pass
            else:raise AssertionError('Untriggered LIMIT without fill/trigger evidence was accepted')
            gateway._set(fresh,'abandoned')
            await gateway.monitor_once()
            assert gateway.positions()==[],'Sub-lot fee remainder is displayed as an open position'
            dust=gateway.positions(include_dust=True)
            assert len(dust)==1 and dust[0]['dust'] is True and Decimal(dust[0]['quantity'])==Decimal('.005875')
            trades=gateway.closed_trades()
            assert len(trades)==1 and trades[0]['exit_reasons']==['native_stop']
            assert trades[0]['settlement']=='closed_with_dust' and Decimal(trades[0]['dust_quantity'])==Decimal('.005875')
            assert abs(Decimal(trades[0]['pnl_usdt'])-Decimal('-.02681101411703214'))<Decimal('1e-15')
            assert abs(trades[0]['net_return']-(-.02141720981132075))<1e-14
            # A quantity of one full lot is not dust, even if its notional is tiny.
            partial=dict(sold,status='CANCELED',executedQty='26.20',executedSum='1.22616',
                fills=[dict(quantity='26.20',price='.0468',sum='1.22616',fee='.001594008',feeAsset='USDT')])
            gateway._set(stop,'acknowledged',partial)
            assert gateway.positions() and gateway.closed_trades()==[],'Tradable remaining lots were silently retired'
            gateway._set(stop,'acknowledged',sold)
            gateway._set(fresh,'unknown')
            assert gateway.positions() and gateway.closed_trades()==[],'Uncertain child was hidden as dust'
            gateway._set(fresh,'abandoned')
            gateway.db.close();gateway=LiveGateway(Path(temporary),cfg,exchange)
            for _ in range(2):await gateway.monitor_once()
            assert not gateway.positions() and len(gateway.closed_trades())==1
            assert exchange.writes==[] and cfg['enabled'] is True
        finally:gateway.db.close()
    print('PASS triggered native-stop recovery, identity/fill guards, explicit fee dust, realized label, restart and zero exchange mutations')

if __name__=='__main__':asyncio.run(check())
