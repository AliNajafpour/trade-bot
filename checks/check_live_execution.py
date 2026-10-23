"""Offline exchange-boundary checks; never uses an API key or the network."""
import asyncio
from copy import deepcopy
from contextlib import ExitStack
from decimal import Decimal
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.live_execution import Gateway, OrderNotFound, WallexMCP, normalize, DEFAULTS


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


async def check_snapshot():
    class Fixture(WallexMCP):
        def __init__(self,optional_symbols=()):
            super().__init__(Path('unused'),DEFAULTS['symbols'],optional_symbols=optional_symbols)
            self.binding='fixture';self.active=0;self.peak=0;self.pages=[];self.failure=None
            original=snapshot()
            self.markets={s:dict(original['markets']['BTCUSDT'],symbol=s,baseAsset=s[:-4]) for s in self.symbols}
            self.balances={s[:-4]:dict(available='0',total='0') for s in self.symbols}
            self.balances['USDT']=dict(available='100',total='100')
            self.fees={s:dict(takerFeeRate='.0013',makerFeeRate='.0013') for s in self.symbols}
            self._limits_cache=(asyncio.get_running_loop().time(),dict(success=True,result=dict(symbols=self.markets)),{})
        async def _call(self,name,args):
            assert name in self.READS, 'Snapshot attempted an exchange mutation'
            self.active+=1;self.peak=max(self.peak,self.active)
            try:
                if self.failure=='slow-LTCUSDT' and args.get('symbol')=='LTCUSDT': await asyncio.sleep(30)
                await asyncio.sleep(10 if self.failure=='timeout' else .005)
                if name=='wallex_credit_sub_account_balances_details': result=self.balances
                elif name=='wallex_basic_get_account_fees': result=self.fees
                elif name=='wallex_spot_get_markets': result=dict(symbols=self.markets)
                elif name=='wallex_spot_get_open_order':
                    self.pages.append(args['page'])
                    result=[dict(clientOrderId='fixture')]* (100 if args['page']==1 else 1)
                else:
                    if self.failure==args['symbol']: raise ValueError('One book failed')
                    result=dict(bid=[dict(price='99.99',quantity='5')],ask=[dict(price='100',quantity='5')])
                return dict(success=True,result=result)
            finally: self.active-=1
    fixture=Fixture()
    value=await fixture.snapshot()
    assert set(value['quotes'])==set(value['markets'])==set(value['fees'])==set(DEFAULTS['symbols'])
    assert fixture.peak==4 and not fixture.active, 'Snapshot reads were serial or unbounded'
    assert fixture.pages==[1,2] and len(value['open_orders'])==101, 'Open orders were truncated'
    for failure in ('book','fee','balance','metadata','timeout'):
        fixture=Fixture()
        if failure=='book': fixture.failure='XRPUSDT'
        elif failure=='fee': fixture.fees.pop('LTCUSDT')
        elif failure=='balance': fixture.balances.pop('LTC')
        elif failure=='metadata': fixture.markets['XRPUSDT'].pop('minQty')
        else: fixture.failure='timeout'
        try: await asyncio.wait_for(fixture.snapshot(),timeout=.02 if failure=='timeout' else 2)
        except (ValueError,KeyError,TimeoutError): pass
        else: raise AssertionError('Incomplete ten-market snapshot was accepted: '+failure)
        assert not fixture.active, 'A failed/cancelled snapshot left reads running'
    fixture=Fixture();fixture.failure='timeout'
    slow=asyncio.create_task(fixture.snapshot())
    try:
        await asyncio.sleep(.01)
        fixture.failure=None
        await asyncio.wait_for(fixture.snapshot(),timeout=2)
        assert not slow.done(), 'Protection fixture did not overlap a stalled scan snapshot'
    finally:
        slow.cancel()
        await asyncio.gather(slow,return_exceptions=True)
    assert not fixture.active, 'Stalled scan cleanup left reads running'
    try: Fixture(optional_symbols=['BTCUSDT'])
    except ValueError: pass
    else: raise AssertionError('An original market could be optional')
    for failure in ('book','empty-book','fee','balance','metadata','slow'):
        fixture=Fixture(optional_symbols=['LTCUSDT'])
        if failure=='book': fixture.failure='LTCUSDT'
        elif failure=='empty-book':
            original_call=fixture._call
            async def empty_book(name,args):
                if name=='wallex_basic_get_market_depth' and args['symbol']=='LTCUSDT': return dict(result=dict(bid=[],ask=[]))
                return await original_call(name,args)
            fixture._call=empty_book
        elif failure=='fee': fixture.fees.pop('LTCUSDT')
        elif failure=='balance': fixture.balances.pop('LTC')
        elif failure=='metadata': fixture.markets['LTCUSDT'].pop('minQty')
        else: fixture.failure='slow-LTCUSDT'
        degraded=await fixture.snapshot()
        assert set(degraded['market_errors'])=={'LTCUSDT'}
        assert degraded['market_errors']['LTCUSDT']['error_code']=='market_unavailable'
        for field in ('quotes','markets','fees'):
            assert set(degraded[field])==set(DEFAULTS['symbols'])-{'LTCUSDT'}
        assert not fixture.active
    for missing in ('BTCUSDT','USDT'):
        fixture=Fixture(optional_symbols=['LTCUSDT'])
        if missing=='BTCUSDT': fixture.fees.pop(missing)
        else: fixture.balances.pop(missing)
        try: await fixture.snapshot()
        except (ValueError,KeyError): pass
        else: raise AssertionError('Optional markets hid missing mandatory data')
    await check_degraded_protection(degraded)


async def check_degraded_protection(degraded):
    with TemporaryDirectory() as temporary:
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True)
        raw=Exchange();gateway=Gateway(Path(temporary),cfg,raw)
        original=raw.snapshot
        async def unavailable_ltc():
            value=deepcopy(degraded)
            value['observed_at']=stamp()
            for quote in value['quotes'].values(): quote['available_at']=stamp()
            value['balances']['BTC']=dict(available='5',total='5')
            value['open_orders']=[deepcopy(r) for r in raw.orders.values() if r['active']]
            return value
        try:
            key=prepare(gateway);await gateway.execute_entry(key)
            stop=next(k for k,r in raw.orders.items() if r['type']=='STOP_LIMIT')
            raw.snapshot=unavailable_ltc
            before=len(raw.writes)
            result=await gateway.monitor_once()
            assert result['actions']==[dict(parent=key,action='protected')] and len(raw.writes)==before
            # Existing venue inventory in the unavailable market must halt before a mutation.
            with gateway.db:
                entry=gateway.order(key);entry['payload']['symbol']='LTCUSDT'
                gateway.db.execute('UPDATE orders SET payload=? WHERE id=?',(json.dumps(entry['payload']),key))
            raw.orders[key]['symbol']='LTCUSDT'
            for operation in (gateway.monitor_once(),gateway.close_position(key,'unavailable-market'),gateway.protect_entry(key)):
                try: await operation
                except ValueError as error: assert 'Owned or pending market data' in str(error)
                else: raise AssertionError('Unavailable owned LTC allowed a protection/exit mutation')
            try: prepare(gateway,'new-entry-with-missing-owned-market')
            except ValueError as error: assert 'Owned or pending market data' in str(error)
            else: raise AssertionError('Missing owned-market valuation allowed another entry')
            # A partly filled entry remainder must also stay untouched when its market disappears.
            entry=gateway.order(key);entry['payload']['quantity']='.2'
            with gateway.db:
                gateway.db.execute('UPDATE orders SET payload=? WHERE id=?',(json.dumps(entry['payload']),key))
            raw.orders[key].update(origQty='.2',active=True,status='PARTIALLY_FILLED')
            try: await gateway.protect_entry(key)
            except ValueError as error: assert 'Owned or pending market data' in str(error)
            else: raise AssertionError('Unavailable pending market reached cancellation')
            assert raw.orders[key]['active'], 'Missing market cancelled a pending entry remainder'
            assert len(raw.writes)==before and raw.orders[stop]['active'], 'Missing-market check cancelled native protection'
        finally:
            raw.snapshot=original
            gateway.db.close()


def check_llm_entries(technical_mode='llm_chart',selected_symbol='XRPUSDT'):
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        field='legacy_technical_symbols' if technical_mode=='legacy_ensemble' else 'llm_chart_symbols'
        cfg=dict(DEFAULTS,enabled=True,account_binding='fixture',stop_capabilities_verified=True)
        cfg[field]=[selected_symbol]
        raw=Exchange();gateway=Gateway(root,cfg,raw)
        directory=root/'reviews'/'xrp'/'inference';directory.mkdir(parents=True)
        packet=dict(symbol=selected_symbol,mode='real_spot_user_armed',paper_only=False,quote=dict(ask=100),
            admission=dict(eligible=True),prediction=dict(technical_mode=technical_mode,tp=102,sl=98))
        if technical_mode=='legacy_ensemble':
            packet['prediction']['legacy_technical']=dict(status='ok',models=dict(kronos=dict(status='ok'),chronos2=dict(status='ok')),
                rules=dict(status='ok',strategies=['offline strategy fixture']))
        response=dict(action='enter',risk_multiplier=.5,reason='offline validated fixture')
        path=directory/'receipt.json'
        def proof(**changes):
            (directory/'packet.json').write_text(json.dumps(packet))
            (directory/'final.json').write_text(json.dumps(response))
            receipt=dict(status='ok',response=response,ended_utc=stamp(),files_sha256={
                name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('packet.json','final.json')})
            receipt.update(changes);path.write_text(json.dumps(receipt))
            return dict(technical_mode=technical_mode,manager_receipt_path=str(path),manager_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                decision_at=stamp(),execution_ask=100,tp=102,sl=98)
        value=snapshot()
        value['markets'][selected_symbol]=dict(value['markets']['BTCUSDT'],symbol=selected_symbol,baseAsset=selected_symbol[:-4])
        value['quotes'][selected_symbol]=deepcopy(value['quotes']['BTCUSDT'])
        value['fees'][selected_symbol]=deepcopy(value['fees']['BTCUSDT'])
        def prepare_llm(key,evidence,multiplier=.5,symbol=None):
            if symbol is None: symbol=selected_symbol
            return gateway.prepare_entry(key,symbol,'.1',evidence['execution_ask'],evidence['sl'],evidence['tp'],value,evidence,risk_multiplier=multiplier)
        def reject(key,evidence,**kwargs):
            try: prepare_llm(key,evidence,**kwargs)
            except (ValueError,KeyError,OSError): pass
            else: raise AssertionError('Chart entry bypassed '+key)
        try:
            evidence=proof();key=prepare_llm('valid',evidence)
            assert gateway.order(key)['context']['evidence']==evidence and not raw.writes
            if technical_mode=='legacy_ensemble':
                original=deepcopy(packet['prediction']['legacy_technical'])
                for name in ('kronos','chronos2'):
                    for state in ('missing','error'):
                        if state=='missing': packet['prediction']['legacy_technical']['models'].pop(name)
                        else: packet['prediction']['legacy_technical']['models'][name]={'status':'error'}
                        reject(name+'-'+state,proof())
                        packet['prediction']['legacy_technical']=deepcopy(original)
                for rules in ({'status':'error','strategies':['fixture']},{'status':'ok','strategies':[]},{'status':'ok','strategies':'fixture'},None):
                    packet['prediction']['legacy_technical']['rules']=rules
                    reject('missing-or-failed-rules',proof())
                packet['prediction']['legacy_technical']=deepcopy(original)
                packet['prediction']['legacy_technical']['status']='error'
                reject('failed-legacy-analysis',proof())
                packet['prediction']['legacy_technical']=original
                for eligible in (False,1,None):
                    packet['admission']['eligible']=eligible
                    reject('ineligible-legacy-candidate',proof())
                packet['admission']['eligible']=True
                reject('wrong-evidence-mode',dict(proof(),technical_mode='llm_chart'))
                for symbol in ('XRPUSDT','DOGEUSDT','BNBUSDT'):
                    cfg['probability_symbols']=[symbol]
                    cfg[field]=[symbol]
                    cfg['llm_chart_symbols']=[symbol]
                    value['markets'][symbol]=dict(value['markets'][selected_symbol],symbol=symbol,baseAsset=symbol[:-4])
                    value['quotes'][symbol]=deepcopy(value['quotes'][selected_symbol])
                    value['fees'][symbol]=deepcopy(value['fees'][selected_symbol])
                    packet['symbol']=symbol
                    low=dict(proof(),horizon_minutes=1440,p_tp_first=.55,p_sl_first=.3,p_no_touch=.15)
                    reject('probability-cannot-bypass-'+symbol,low,symbol=symbol)
                cfg['probability_symbols']=[];cfg['llm_chart_symbols']=[];packet['symbol']=selected_symbol
            for symbol in ('BTCUSDT',selected_symbol):
                cfg[field]=['BTCUSDT']
                reject('unconfigured-or-original-'+symbol,proof(),symbol=symbol)
            low=dict(proof(),horizon_minutes=1440,p_tp_first=.55,p_sl_first=.3,p_no_touch=.15)
            reject('original-four-strict55',low,symbol='BTCUSDT')
            cfg[field]=[selected_symbol]
            reject('missing-receipt',dict(proof(),manager_receipt_path=str(directory/'absent.json')))
            reject('wrong-receipt-hash',dict(proof(),manager_receipt_sha256='0'*64))
            reject('stale-review',proof(ended_utc='2000-01-01T00:00:00+00:00'))
            reject('naive-review',proof(ended_utc='2026-10-24T00:00:00'))
            reject('failed-provider',proof(status='error'))
            reject('risk-increase',proof(),multiplier=.6)
            packet['mode']='public_chart_analysis'
            reject('public-analysis-after-switch-on',proof());packet['mode']='real_spot_user_armed'
            packet['paper_only']=True
            reject('paper-analysis',proof());packet['paper_only']=False
            packet['prediction']['technical_mode']='weekly'
            reject('model-analysis-as-chart',proof());packet['prediction']['technical_mode']=technical_mode
            packet['entry_policy']=dict(reference_ask=100,max_reprice_bps=10,tp_fraction=.02,sl_fraction=.02)
            def repriced(price):
                evidence=proof()
                evidence.update(execution_ask=str(price),tp=normalize(Decimal(str(price))*Decimal('1.02'),2),sl=normalize(Decimal(str(price))*Decimal('.98'),2))
                value['quotes'][selected_symbol].update(ask=float(price),bid=float(price)-.01)
                return evidence
            evidence=repriced('100.05')
            key=prepare_llm('five-bps-reprice',evidence)
            assert gateway.order(key)['payload']['price']=='100.05'
            assert gateway.order(key)['context']['barriers']==dict(stop_price='98.04',target_price='102.05')
            prepare_llm('exact-ten-bps-reprice',repriced('100.10'))
            prepare_llm('lower-five-bps-reprice',repriced('99.95'))
            outside=repriced('100.11')
            persisted=gateway.order(key)
            persisted['payload']['price']='100.11'
            persisted['context'].update(evidence=outside,barriers=dict(stop_price=outside['sl'],target_price=outside['tp']))
            try: gateway._risk(persisted['payload'],persisted['context'],value)
            except ValueError as error: assert 'price band' in str(error)
            else: raise AssertionError('Dispatch bypassed manager price band')
            reject('beyond-ten-bps',repriced('100.11'))
            reject('below-ten-bps-band',repriced('99.89'))
            packet['entry_policy']['max_reprice_bps']=11
            reject('widened-price-policy',repriced('100.05'))
            packet['entry_policy']['max_reprice_bps']=10
            packet['entry_policy']['tp_fraction']=.03
            reject('changed-relative-target-policy',repriced('100.05'))
            packet['entry_policy']['tp_fraction']=.02
            packet['entry_policy']['reference_ask']=101
            reject('mismatched-policy-reference',repriced('100.05'))
            packet.pop('entry_policy')
            value['quotes'][selected_symbol].update(ask=100,bid=99.99)
            packet['quote']['ask']=99
            reject('changed-ask',proof());packet['quote']['ask']=100
            packet['prediction']['tp']=103
            reject('changed-target',proof());packet['prediction']['tp']=102
            packet['symbol']='DOGEUSDT'
            reject('changed-symbol',proof());packet['symbol']=selected_symbol
            evidence=proof();(directory/'packet.json').write_text('{}')
            reject('changed-packet',evidence)
            evidence=proof();(directory/'final.json').write_text('{}')
            reject('changed-final',evidence)
            reject('manager-wait',proof(response=dict(action='wait',risk_multiplier=.5)))
            reject('final-response-mismatch',proof(response=dict(response,reason='changed validated response')))
            evidence=proof();outside=root/'receipt.json';outside.write_bytes(path.read_bytes())
            reject('outside-reviews',dict(evidence,manager_receipt_path=str(outside)))
            evidence=proof();key=prepare_llm('review-expires-before-send',evidence)
            stored=gateway.order(key)['context'];stored['evidence']['decision_at']=stamp()
            expired=proof(ended_utc='2000-01-01T00:00:00+00:00');stored['evidence'].update(expired)
            try: gateway._risk(gateway.order(key)['payload'],stored,value)
            except ValueError: pass
            else: raise AssertionError('Persisted chart intent bypassed fresh review check')
            assert not raw.writes and DEFAULTS['max_open_positions']==4
        finally: gateway.db.close()


async def check():
    await check_snapshot()
    check_llm_entries()
    check_llm_entries('legacy_ensemble','ADAUSDT')
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
    print('live execution: optional-market isolation, owned/pending fail-closed, BTC protection preserved; receipt-bound chart/legacy entries, both forecasters and strategy bank, strict55, bounded snapshots, disabled writes and risk caps OK')


if __name__=='__main__': asyncio.run(check())
