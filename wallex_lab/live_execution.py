"""Disabled-by-default credit-account execution boundary. No paper balance migration."""
import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import hashlib
import json
from pathlib import Path
import sqlite3
from .locking import exclusive

from .spot import _quote, load_exclusions, market_allowed

REPO=Path(__file__).resolve().parents[1]
DEFAULTS=dict(enabled=False,account_scope='credit_subaccount',account_binding=None,
    symbols=['BTCUSDT','ETHUSDT','SOLUSDT','SUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','BNBUSDT','TRXUSDT','LTCUSDT'],max_position_fraction=.15,
    risk_per_trade_fraction=.005,max_account_stop_risk_fraction=.02,max_open_positions=4,
    max_spread_bps=100,max_signal_age_seconds=120,snapshot_max_age_seconds=30,
    stop_limit_gap_bps=20,stop_capabilities_verified=False)


class OrderNotFound(Exception): pass


def number(value, positive=False):
    try: result=Decimal(str(value))
    except (InvalidOperation,ValueError): raise ValueError('Invalid exchange number') from None
    if not result.is_finite() or result<0 or (positive and result==0): raise ValueError('Invalid exchange number')
    return result


def normalize(value, decimals):
    if type(decimals) is not int or not 0<=decimals<=18: raise ValueError('Invalid venue precision')
    return format(number(value).quantize(Decimal(1).scaleb(-decimals),rounding=ROUND_DOWN),'f')


def age(value):
    stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
    if stamp.tzinfo is None: raise ValueError('Aware exchange timestamp required')
    return (datetime.now(timezone.utc)-stamp).total_seconds()


class Gateway:
    def __init__(self,root,config,exchange):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.config=config;self.exchange=exchange
        if config.get('account_scope')!='credit_subaccount': raise ValueError('Only the selected credit account is supported')
        if not config.get('account_binding'): raise ValueError('Read-only account binding required')
        self.exclusions=load_exclusions(REPO/'config/credit-excluded-markets.txt')
        self.db=sqlite3.connect(self.root/'orders.sqlite',timeout=5)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS identity(binding TEXT PRIMARY KEY)')
        prior=self.db.execute('SELECT binding FROM identity').fetchone()
        if prior and prior[0]!=config['account_binding']: raise ValueError('Refusing another account in this execution ledger')
        self.db.execute('INSERT OR IGNORE INTO identity VALUES(?)',(config['account_binding'],))
        self.db.execute('CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,intent_key TEXT UNIQUE,role TEXT,parent TEXT,payload TEXT,context TEXT,state TEXT,response TEXT,error TEXT)')
        self.db.commit()

    def order(self,key):
        row=self.db.execute('SELECT * FROM orders WHERE id=?',(key,)).fetchone()
        if not row: raise ValueError('Unknown local order')
        row=dict(row)
        for field in ('payload','context','response'): row[field]=json.loads(row[field]) if row[field] else None
        return row

    def _record(self,key,role,payload,context,parent=None):
        client='wlx-'+hashlib.sha256((self.config['account_binding']+'\n'+key).encode()).hexdigest()[:32]
        payload=dict(payload,client_id=client)
        serialized=json.dumps(payload,sort_keys=True,allow_nan=False)
        prior=self.db.execute('SELECT id,payload,context FROM orders WHERE intent_key=?',(key,)).fetchone()
        if prior:
            previous=json.loads(prior['context'])
            if prior['payload']!=serialized or any(previous.get(k)!=context.get(k) for k in ('barriers','risk_multiplier')):
                raise ValueError('Intent identity reused with changed order parameters')
            return prior['id']
        with self.db:
            self.db.execute('INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?)',(client,key,role,parent,serialized,json.dumps(context,allow_nan=False),'prepared',None,None))
        return client

    def _set(self,key,state,response=None,error=None):
        with self.db:
            self.db.execute('UPDATE orders SET state=?,response=COALESCE(?,response),error=? WHERE id=?',
                (state,json.dumps(response,allow_nan=False) if response is not None else None,error,key))

    def _snapshot(self,snapshot):
        if snapshot.get('account_scope')!='credit_subaccount' or snapshot.get('account_binding')!=self.config['account_binding']:
            raise ValueError('Credit account binding mismatch')
        if not 0<=age(snapshot['observed_at'])<=self.config['snapshot_max_age_seconds']: raise ValueError('Stale account snapshot')
        required={p['symbol'] for p in self.positions()}
        for row in self.db.execute("SELECT id FROM orders WHERE role='entry' AND state IN ('acknowledged','sending','unknown','cancel_unknown')"):
            order=self.order(row['id'])
            if order['state']!='acknowledged' or (order['response'] or {}).get('active'): required.add(order['payload']['symbol'])
        if (any(s not in snapshot.get(field,{}) for s in required for field in ('quotes','markets','fees'))
                or any(s[:-4] not in snapshot.get('balances',{}) for s in required)):
            raise ValueError('Owned or pending market data is unavailable')

    def _writes(self):
        if self.config.get('enabled') is not True or self.exchange.allow_writes is not True:
            raise PermissionError('Real exchange writes are disabled')

    def positions(self):
        result=[]
        for row in self.db.execute("SELECT id FROM orders WHERE role='entry'").fetchall():
            entry=self.order(row['id']);response=entry['response'] or {}
            bought=number(response.get('executedQty',0))
            if not bought: continue
            symbol=entry['payload']['symbol'];base=symbol[:-4]
            bought-=sum((number(f['fee']) for f in response.get('fills',[]) if f.get('feeAsset')==base),Decimal(0))
            sold=Decimal(0)
            for child in self.db.execute('SELECT id FROM orders WHERE parent=?',(entry['id'],)).fetchall():
                fill=self.order(child['id'])['response'] or {}
                sold+=number(fill.get('executedQty',0))
                sold+=sum((number(f['fee']) for f in fill.get('fills',[]) if f.get('feeAsset')==base),Decimal(0))
            remaining=bought-sold
            if remaining<0: raise ValueError('Owned-inventory accounting mismatch; reconciliation required')
            if remaining:
                cost=number(response.get('executedSum',0))+sum((number(f['fee']) for f in response.get('fills',[]) if f.get('feeAsset')=='USDT'),Decimal(0))
                result.append(dict(parent=entry['id'],symbol=symbol,quantity=str(remaining),cost_basis=str(cost*remaining/bought),**entry['context']['barriers']))
        return result

    def closed_trades(self):
        """Read fully settled roundtrips; cash flows include confirmed fees only."""
        # ponytail: scans historical orders; add a settled-trade index if the retained ledger becomes large.
        result=[]
        for item in self.db.execute("SELECT id FROM orders WHERE role='entry' ORDER BY rowid").fetchall():
            try:
                entry=self.order(item['id']);symbol=entry['payload']['symbol'];base=symbol[:-4]
                if not symbol.endswith('USDT'): continue
                def settled(row,side):
                    response=row['response'];payload=row['payload']
                    if (row['state']!='acknowledged' or not isinstance(response,dict)
                            or response.get('active') is not False
                            or response.get('status') not in ('FILLED','CANCELED','CANCELLED','REJECTED','EXPIRED')
                            or payload['symbol']!=symbol or payload['side']!=side
                            or response.get('clientOrderId')!=row['id']
                            or any(response.get(k)!=payload[k] for k in ('symbol','side','type'))):
                        raise ValueError('Unsettled trade')
                    quantity=number(response['executedQty']);original=number(response['origQty'],True)
                    if (quantity>original or original!=number(payload['quantity'],True)
                            or number(response['price'])!=number(payload['price'])
                            or (response['status']=='FILLED' and quantity!=original)):
                        raise ValueError('Invalid fill counters')
                    fills=response.get('fills',[])
                    if not isinstance(fills,list) or sum((number(f['quantity'],True) for f in fills),Decimal(0))!=quantity:
                        raise ValueError('Incomplete fills')
                    costs=[];base_fee=quote_fee=Decimal(0);unit=Decimal('.00000001')
                    for fill in fills:
                        fee=number(fill['fee']);asset=fill['feeAsset']
                        if asset not in (base,'USDT'): raise ValueError('Unsupported fee currency')
                        calculated=number(fill['quantity'],True)*number(fill['price'],True)
                        cost=number(fill.get('sum',calculated))
                        if abs(cost-calculated)>unit: raise ValueError('Fill cost mismatch')
                        costs.append(cost)
                        if asset==base: base_fee+=fee
                        else: quote_fee+=fee
                    cash=number(response['executedSum'])
                    if abs(cash-sum(costs,Decimal(0)))>unit: raise ValueError('Incomplete fill costs')
                    return quantity,base_fee,cash+quote_fee if side=='BUY' else cash-quote_fee
                bought,base_fee,cost=settled(entry,'BUY');bought-=base_fee
                if bought<=0 or cost<=0: continue
                sold=proceeds=Decimal(0);reasons=[]
                for child in self.db.execute('SELECT id FROM orders WHERE parent=? ORDER BY rowid',(entry['id'],)).fetchall():
                    row=self.order(child['id'])
                    if row['state'] in ('prepared','abandoned') and row['response'] is None: continue
                    if row['role'] not in ('stop','exit'): raise ValueError('Unknown sell role')
                    quantity,base_fee,cash=settled(row,'SELL')
                    sold+=quantity+base_fee;proceeds+=cash
                    if quantity:
                        reason='native_stop' if row['role']=='stop' else row['context'].get('reason','unknown_exit')
                        if reason not in reasons: reasons.append(reason)
                if sold!=bought: continue
                result.append(dict(entry_id=entry['id'],symbol=symbol,cost_usdt=str(cost),proceeds_usdt=str(proceeds),
                    pnl_usdt=str(proceeds-cost),net_return=float((proceeds-cost)/cost),
                    context=entry['context'],exit_reasons=reasons))
            except (ValueError,KeyError,TypeError,ArithmeticError):
                # An incomplete or inconsistent receipt cannot become a training label.
                continue
        return result

    def _risk(self,payload,context,snapshot):
        self._snapshot(snapshot)
        known={r['id'] for r in self.db.execute('SELECT id FROM orders')}
        open_ids={r.get('clientOrderId') for r in snapshot['open_orders']}
        if not open_ids<=known:
            raise ValueError('Unmanaged open orders require separate account reconciliation')
        for item in self.db.execute("SELECT id FROM orders WHERE state NOT IN ('prepared','abandoned')").fetchall():
            row=self.order(item['id'])
            if (row['response'] or {}).get('active') and row['id'] not in open_ids:
                raise ValueError('Open-order snapshot changed during reconciliation; refresh before entry')
            if row['state']!='acknowledged' or (row['role']!='stop' and (row['response'] or {}).get('active')):
                raise ValueError('Unsettled order blocks new entries until reconciled and protected')
        symbol=payload['symbol'];market=snapshot['markets'][symbol]
        if symbol not in self.config['symbols'] or not market_allowed(symbol[:-4]+'/USDT',self.exclusions): raise ValueError('Market outside credit-account policy')
        if market['baseAsset']!=symbol[:-4] or market['quoteAsset']!='USDT': raise ValueError('Only USDT spot buys are supported')
        for field,precision in (('quantity','stepSize'),('price','tickSize')):
            if normalize(payload[field],market[precision])!=payload[field]: raise ValueError('Venue precision changed; replan required')
        quote=snapshot['quotes'][symbol];_quote(quote,datetime.now(timezone.utc))
        q=number(payload['quantity'],True);price=number(payload['price'],True)
        if price!=number(quote['ask']): raise ValueError('Fresh executable ask changed; rescore exact levels')
        if q>number(quote['ask_quantity']): raise ValueError('Insufficient displayed entry depth')
        if (price-number(quote['bid']))/price*10000>number(self.config['max_spread_bps']): raise ValueError('Spread limit exceeded')
        if q<number(market['minQty']) or q*price<number(market['minNotional']): raise ValueError('Order below venue minimum')
        fee=number(snapshot['fees'][symbol]['takerFeeRate'])
        if fee>=1: raise ValueError('Invalid authenticated fee')
        available=number(snapshot['balances']['USDT']['available'])
        equity=number(snapshot['balances']['USDT']['total']);used=Decimal(0)
        positions=self.positions()
        if any(p['symbol']==symbol for p in positions): raise ValueError('Existing managed position cannot be duplicated')
        if len(positions)>=self.config['max_open_positions']: raise ValueError('Open-position cap reached')
        for pos in positions:
            stops=[self.order(r['id']) for r in self.db.execute("SELECT id FROM orders WHERE parent=? AND role='stop'",(pos['parent'],)).fetchall()]
            if len([r for r in stops if r['state']=='acknowledged' and (r['response'] or {}).get('active')])!=1:
                raise ValueError('Unprotected inventory blocks new entries')
            mark=snapshot['quotes'][pos['symbol']];_quote(mark,datetime.now(timezone.utc))
            equity+=number(pos['quantity'])*number(mark['bid'])
            rate=number(snapshot['fees'][pos['symbol']]['takerFeeRate'])
            limit=number(pos['stop_price'])*(1-number(self.config['stop_limit_gap_bps'])/10000)
            used+=max(Decimal(0),number(pos['cost_basis'])-number(pos['quantity'])*limit*(1-rate))
        stop=number(context['barriers']['stop_price'],True);target=number(context['barriers']['target_price'],True)
        if not stop<number(quote['bid'])<=price<target: raise ValueError('Invalid or breached protective barriers')
        cost=q*price*(1+fee)
        loss=cost-q*stop*(1-number(self.config['stop_limit_gap_bps'])/10000)*(1-fee)
        multiplier=number(context['risk_multiplier'])
        if multiplier>1: raise ValueError('Risk multiplier must be at most one')
        if cost>available or q*price>equity*number(self.config['max_position_fraction']): raise ValueError('Cash or allocation cap exceeded')
        if loss>equity*number(self.config['risk_per_trade_fraction'])*multiplier: raise ValueError('Per-trade stop-risk cap exceeded')
        if used+loss>equity*number(self.config['max_account_stop_risk_fraction']): raise ValueError('Portfolio stop-risk cap exceeded')
        if target*(1-fee)<=price*(1+fee): raise ValueError('Target does not clear authenticated fees')
        if self.db.execute("SELECT 1 FROM orders WHERE state IN ('sending','unknown','cancel_unknown') LIMIT 1").fetchone(): raise ValueError('Uncertain order blocks new entries until reconciled')
        evidence=context['evidence']
        if not 0<=age(evidence['decision_at'])<=self.config['max_signal_age_seconds']: raise ValueError('Signal expired')
        self._entry_evidence(payload,context['barriers']['target_price'],context['barriers']['stop_price'],evidence,context['risk_multiplier'],market)

    def _entry_evidence(self,payload,tp,sl,evidence,risk_multiplier,market):
        symbol=payload['symbol']
        technical_mode=None
        if symbol in DEFAULTS['symbols'] and symbol not in ('BTCUSDT','ETHUSDT','SOLUSDT','SUSDT') and symbol not in self.config.get('probability_symbols',[]):
            if symbol in self.config.get('legacy_technical_symbols',[]): technical_mode='legacy_ensemble'
            elif symbol in self.config.get('llm_chart_symbols',[]): technical_mode='llm_chart'
        if technical_mode:
            if evidence.get('technical_mode')!=technical_mode: raise ValueError('Configured technical market requires its reviewed analysis mode')
            path=Path(evidence['manager_receipt_path']).resolve()
            if not path.is_relative_to((self.root/'reviews').resolve()) or path.name!='receipt.json':
                raise ValueError('Manager receipt must be stored in this execution ledger')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=evidence['manager_receipt_sha256']: raise ValueError('Manager receipt integrity failed')
            receipt=json.loads(raw);response=receipt['response']
            if receipt.get('status')!='ok' or response.get('action')!='enter': raise ValueError('Manager did not approve this entry')
            for name in ('packet.json','final.json'):
                source=(path.parent/name).resolve()
                if not source.is_relative_to((self.root/'reviews').resolve()): raise ValueError('Manager evidence escaped execution ledger')
                data=source.read_bytes()
                if hashlib.sha256(data).hexdigest()!=receipt['files_sha256'][name]: raise ValueError('Manager evidence integrity failed')
                if name=='packet.json': packet=json.loads(data)
                elif json.loads(data)!=response: raise ValueError('Manager validated response changed')
            if packet.get('mode')!='real_spot_user_armed' or packet.get('paper_only') is not False or packet['prediction'].get('technical_mode')!=technical_mode:
                raise ValueError('Public or model-only analysis cannot authorize a chart entry')
            if technical_mode=='legacy_ensemble':
                legacy=packet['prediction'].get('legacy_technical')
                if not isinstance(legacy,dict) or legacy.get('status')!='ok': raise ValueError('Legacy technical analysis failed')
                models=legacy.get('models');rules=legacy.get('rules')
                if not isinstance(models,dict) or any(not isinstance(models.get(name),dict) or models[name].get('status')!='ok' for name in ('kronos','chronos2')):
                    raise ValueError('Both legacy forecasting models must succeed')
                if not isinstance(rules,dict) or rules.get('status')!='ok' or not isinstance(rules.get('strategies'),list) or not rules['strategies']:
                    raise ValueError('Legacy strategy bank must succeed with nonempty strategies')
                admission=packet.get('admission')
                if not isinstance(admission,dict) or admission.get('eligible') is not True: raise ValueError('Legacy technical candidate is not eligible')
            if not 0<=age(receipt['ended_utc'])<=self.config['max_signal_age_seconds']: raise ValueError('Manager review expired')
            if packet['symbol']!=symbol: raise ValueError('Manager reviewed another symbol')
            policy=packet.get('entry_policy')
            if policy is not None:
                reference=number(policy['reference_ask'],True);band=number(policy['max_reprice_bps'])
                if band>10 or (number(policy['tp_fraction']),number(policy['sl_fraction']))!=(Decimal('.02'),Decimal('.02')):
                    raise ValueError('Manager repricing policy exceeds reviewed limits')
                if (number(packet['quote']['ask']),number(packet['prediction']['tp']),number(packet['prediction']['sl']))!=(reference,
                        number(normalize(reference*Decimal('1.02'),market['tickSize'])),number(normalize(reference*Decimal('.98'),market['tickSize']))):
                    raise ValueError('Manager reference levels disagree with repricing policy')
                if abs(number(payload['price'])/reference-1)*10000>band:
                    raise ValueError('Executable ask exceeds manager-approved price band')
                if (number(tp),number(sl))!=(number(normalize(number(payload['price'])*Decimal('1.02'),market['tickSize'])),
                        number(normalize(number(payload['price'])*Decimal('.98'),market['tickSize']))):
                    raise ValueError('Repriced entry requires fixed two-percent barriers')
            elif (number(packet['quote']['ask']),number(packet['prediction']['tp']),number(packet['prediction']['sl']))!=(number(payload['price']),number(tp),number(sl)):
                raise ValueError('Executable levels changed; fresh manager review required')
            if not 0<number(risk_multiplier)<=number(response['risk_multiplier'])<=1: raise ValueError('Manager-approved risk was exceeded')
        else:
            probs=[number(evidence[k]) for k in ('p_tp_first','p_sl_first','p_no_touch')]
            if any(p>1 for p in probs) or abs(sum(probs)-1)>Decimal('.00001'): raise ValueError('Invalid competing-risk probabilities')
            if evidence.get('support_flags') or evidence['horizon_minutes']!=1440 or probs[0]<=Decimal('.55'): raise ValueError('Weekly entry gate failed')
        if (number(evidence['execution_ask']),number(evidence['tp']),number(evidence['sl']))!=(number(payload['price']),number(tp),number(sl)):
            raise ValueError('Rounded executable levels require a fresh review')

    def prepare_entry(self,key,symbol,quantity,price,stop,target,snapshot,evidence,*,risk_multiplier):
        with exclusive(self.root/'dispatch.lock'):
            return self._prepare_entry(key,symbol,quantity,price,stop,target,snapshot,evidence,risk_multiplier=risk_multiplier)

    def _prepare_entry(self,key,symbol,quantity,price,stop,target,snapshot,evidence,*,risk_multiplier):
        market=snapshot['markets'][symbol]
        payload=dict(symbol=symbol,side='BUY',type='LIMIT',quantity=normalize(quantity,market['stepSize']),price=normalize(price,market['tickSize']))
        sl=normalize(stop,market['tickSize']);tp=normalize(target,market['tickSize'])
        if (sl,tp)!=(normalize(number(payload['price'])*Decimal('.98'),market['tickSize']),normalize(number(payload['price'])*Decimal('1.02'),market['tickSize'])):
            raise ValueError('Weekly live policy requires fixed two-percent barriers')
        context=dict(barriers=dict(stop_price=sl,target_price=tp),evidence=evidence,risk_multiplier=float(risk_multiplier))
        existing=self.db.execute('SELECT id FROM orders WHERE intent_key=?',(key,)).fetchone()
        if not existing: self._risk(payload,context,snapshot)
        else: self._entry_evidence(payload,tp,sl,evidence,risk_multiplier,market)
        return self._record(key,'entry',payload,context)

    def _observe(self,key,response):
        row=self.order(key);payload=row['payload']
        if response.get('clientOrderId')!=key or any(response.get(k)!=payload[k] for k in ('symbol','side','type')):
            raise ValueError('Exchange order identity mismatch')
        executed=number(response['executedQty']);original=number(response['origQty'],True)
        previous=number((row['response'] or {}).get('executedQty',0))
        if executed<previous or executed>original or original!=number(payload['quantity']): raise ValueError('Invalid exchange execution counters')
        if type(response.get('active')) is not bool or response.get('status') not in ('NEW','PARTIALLY_FILLED','FILLED','CANCELED','CANCELLED','REJECTED','EXPIRED'): raise ValueError('Unknown exchange order state')
        if response['active']!=(response['status'] in ('NEW','PARTIALLY_FILLED')) or (response['status']=='FILLED' and executed!=original):
            raise ValueError('Inconsistent exchange order state')
        if number(response['price'])!=number(payload['price']): raise ValueError('Exchange limit price mismatch')
        if payload['type']=='STOP_LIMIT' and number(response['stopPrice'])!=number(payload['stop_Price']):
            raise ValueError('Exchange protective trigger mismatch')
        if executed:
            fills=response.get('fills')
            if not isinstance(fills,list) or sum((number(f['quantity']) for f in fills),Decimal(0))!=executed:
                raise ValueError('Confirmed fill and fee breakdown required')
            for fill in fills:
                number(fill['fee']);number(fill['price'],True)
                if fill['feeAsset'] not in (payload['symbol'][:-4],'USDT'): raise ValueError('Unsupported fee currency')
            # USDT totals are rounded by the venue; never demand sub-unit exactness from multiplied fill decimals.
            unit=Decimal('0.00000001')
            costs=[]
            for fill in fills:
                calculated=number(fill['quantity'])*number(fill['price'])
                cost=number(fill.get('sum',calculated))
                if abs(cost-calculated)>unit: raise ValueError('Fill cost mismatch')
                costs.append(cost)
            if abs(number(response['executedSum'])-sum(costs,Decimal(0)))>unit:
                raise ValueError('Execution cost mismatch')
        self._set(key,'acknowledged',response)

    async def send(self,key):
        self._writes()
        with exclusive(self.root/'dispatch.lock'):
            return await self._send(key)

    async def _send(self,key):
        self._writes();row=self.order(key)
        if row['state']!='prepared': return row
        await self._reconcile()
        snapshot=await self.exchange.snapshot();self._snapshot(snapshot)
        if row['role']=='entry':
            if self.config.get('stop_capabilities_verified') is not True: raise PermissionError('Native protective stop has not been verified for this account')
            self._risk(row['payload'],row['context'],snapshot)
        else:
            position=next((p for p in self.positions() if p['parent']==row['parent']),None)
            if not position or number(row['payload']['quantity'])>number(position['quantity']): raise ValueError('Sell exceeds confirmed bot-owned inventory')
            entry=self.order(row['parent'])
            if entry['state']!='acknowledged' or entry['response']['active']: raise ValueError('Entry remainder must be settled before a sell')
            for child in self.db.execute('SELECT id FROM orders WHERE parent=? AND id<>?',(row['parent'],key)).fetchall():
                other=self.order(child['id'])
                if other['state'] in ('sending','unknown','cancel_unknown') or (other['response'] or {}).get('active'):
                    raise ValueError('Existing exit reservation must be reconciled before another sell')
            base=row['payload']['symbol'][:-4]
            if number(row['payload']['quantity'])>number(snapshot['balances'][base]['available']): raise ValueError('Owned asset is locked or unavailable')
        with self.db:
            claimed=self.db.execute("UPDATE orders SET state='sending' WHERE id=? AND state='prepared'",(key,)).rowcount
        if not claimed: return self.order(key)
        try:
            await self.exchange.place(row['payload'])
            # Creation responses may omit fills/fees. The detailed GET is authoritative for owned inventory.
            self._observe(key,await self.exchange.get(key))
        except Exception as error: self._set(key,'unknown',error=type(error).__name__)
        return self.order(key)

    async def execute_entry(self,key):
        self._writes()
        with exclusive(self.root/'dispatch.lock'):
            if self.order(key)['role']!='entry': raise ValueError('Entry intent required')
            response=await self._send(key)
            if response['state']!='acknowledged': raise RuntimeError('Entry acceptance unresolved; reconcile before further action')
            await self._protect_entry(key)
            return self.order(key)

    async def reconcile(self):
        with exclusive(self.root/'dispatch.lock'):
            return await self._reconcile()

    async def _reconcile(self):
        for item in self.db.execute("SELECT id FROM orders WHERE state NOT IN ('prepared','abandoned')").fetchall():
            key=item['id']
            row=self.order(key)
            if row['state']=='acknowledged' and not row['response']['active']: continue
            try: self._observe(key,await self.exchange.get(key))
            except OrderNotFound:
                # Even 404 after a timed-out POST does not justify a second POST.
                self._set(key,'unknown',error='not_found_after_possible_submission')
            except Exception as error: self._set(key,'unknown',error=type(error).__name__)
        return self.positions()

    async def _cancel(self,key):
        self._writes();row=self.order(key)
        if not (row['response'] or {}).get('active'): return
        self._set(key,'cancel_unknown')
        try:
            await self.exchange.cancel(key)
            self._observe(key,await self.exchange.get(key))
        except Exception as error:
            self._set(key,'cancel_unknown',error=type(error).__name__);raise RuntimeError('Cancellation unresolved; reconcile before another order') from None
        if self.order(key)['response']['active']: raise RuntimeError('Cancellation not confirmed')

    async def protect_entry(self,parent):
        self._writes()
        with exclusive(self.root/'dispatch.lock'):
            return await self._protect_entry(parent)

    async def _protect_entry(self,parent):
        self._writes()
        await self._reconcile()
        entry=self.order(parent)
        if entry['state']!='acknowledged': raise RuntimeError('Entry acceptance unresolved')
        snapshot=await self.exchange.snapshot();self._snapshot(snapshot)
        await self._cancel(parent)
        await self._reconcile()
        position=next((p for p in self.positions() if p['parent']==parent),None)
        if not position: return None
        existing=[self.order(r['id']) for r in self.db.execute("SELECT id FROM orders WHERE parent=? AND role='stop'",(parent,)).fetchall()]
        if any(r['state'] in ('sending','unknown','cancel_unknown') for r in existing): raise RuntimeError('Protective order acceptance unresolved')
        market=snapshot['markets'][position['symbol']]
        qty=normalize(position['quantity'],market['stepSize']);stop=position['stop_price']
        active=[r for r in existing if (r['response'] or {}).get('active')]
        if len(active)>1: raise RuntimeError('Multiple active protective orders require reconciliation')
        if active:
            if number(active[0]['response']['origQty'])-number(active[0]['response']['executedQty'])!=number(qty):
                raise RuntimeError('Protective coverage mismatch; manual reconciliation required')
            return active[0]
        limit=normalize(number(stop)*(1-number(self.config['stop_limit_gap_bps'])/10000),market['tickSize'])
        if number(qty)<number(market['minQty']) or number(qty)*number(limit)<number(market['minNotional']): raise RuntimeError('Partial fill below protective-order minimum; manual attention required')
        payload=dict(symbol=position['symbol'],side='SELL',type='STOP_LIMIT',quantity=qty,price=limit,stop_Price=stop)
        prepared=[r for r in existing if r['state']=='prepared']
        if prepared:
            if len(prepared)!=1 or any(prepared[0]['payload'][k]!=v for k,v in payload.items()): raise RuntimeError('Prepared protection changed; reconciliation required')
            key=prepared[0]['id']
        else: key=self._record(parent+'-stop-'+str(len(existing)),'stop',payload,{},parent)
        await self._send(key)
        result=self.order(key)
        if result['state']!='acknowledged' or (not result['response']['active'] and result['response']['status']!='FILLED'):
            raise RuntimeError('Native protection not confirmed; new entries must remain halted')
        return result

    async def close_position(self,parent,reason):
        self._writes()
        with exclusive(self.root/'dispatch.lock'):
            return await self._close_position(parent,reason)

    async def _close_position(self,parent,reason):
        await self._reconcile()
        position=next((p for p in self.positions() if p['parent']==parent),None)
        if position:
            snapshot=await self.exchange.snapshot();self._snapshot(snapshot)
            market=snapshot['markets'][position['symbol']];quote=snapshot['quotes'][position['symbol']]
            _quote(quote,datetime.now(timezone.utc))
            qty=normalize(min(number(position['quantity']),number(quote['bid_quantity'])),market['stepSize'])
            if number(qty)<number(market['minQty']) or number(qty)*number(quote['bid'])<number(market['minNotional']):
                raise RuntimeError('No executable exit size; existing protection retained')
        try:
            return await self._close_orders(parent,reason)
        except Exception:
            uncertain=self.db.execute("SELECT 1 FROM orders WHERE (parent=? OR id=?) AND state IN ('sending','unknown','cancel_unknown')",(parent,parent)).fetchone()
            if not uncertain:
                with self.db:
                    self.db.execute("UPDATE orders SET state='abandoned' WHERE parent=? AND state='prepared'",(parent,))
                try: await self._protect_entry(parent)
                except Exception: raise RuntimeError('Exit failed and protective recovery is unconfirmed; manual reconciliation required') from None
            raise

    async def _close_orders(self,parent,reason):
        self._writes();await self._reconcile()
        await self._cancel(parent)
        with self.db:
            self.db.execute("UPDATE orders SET state='abandoned' WHERE parent=? AND state='prepared'",(parent,))
        for child in self.db.execute('SELECT id FROM orders WHERE parent=?',(parent,)).fetchall(): await self._cancel(child['id'])
        await self._reconcile()
        if self.db.execute("SELECT 1 FROM orders WHERE (parent=? OR id=?) AND state IN ('sending','unknown','cancel_unknown')",(parent,parent)).fetchone():
            raise RuntimeError('Unresolved order prevents a second exit')
        position=next((p for p in self.positions() if p['parent']==parent),None)
        if not position: return None
        snapshot=await self.exchange.snapshot();market=snapshot['markets'][position['symbol']]
        quote=snapshot['quotes'][position['symbol']];_quote(quote,datetime.now(timezone.utc))
        qty=normalize(min(number(position['quantity']),number(snapshot['balances'][position['symbol'][:-4]]['available']),number(quote['bid_quantity'])),market['stepSize'])
        price=normalize(quote['bid'],market['tickSize'])
        if number(qty)<number(market['minQty']) or number(qty)*number(price)<number(market['minNotional']): raise RuntimeError('Exit below venue minimum; manual attention required')
        generation=self.db.execute("SELECT count(*) FROM orders WHERE parent=? AND role='exit'",(parent,)).fetchone()[0]
        payload=dict(symbol=position['symbol'],side='SELL',type='LIMIT',quantity=qty,price=price)
        key=self._record(parent+'-exit-'+str(generation),'exit',payload,dict(reason=reason),parent)
        result=await self._send(key)
        if result['state']!='acknowledged': raise RuntimeError('Exit acceptance unresolved; reconcile before restoring protection')
        if result['state']=='acknowledged':
            # Emulate an immediate-or-cancel exit: an unfilled LIMIT must not replace the stop indefinitely.
            await self._cancel(key)
            await self._reconcile()
            await self._protect_entry(parent)
        return self.order(key)

    async def monitor_once(self):
        """Recover protection and manage targets before admitting persisted manager intents."""
        self._writes()
        with exclusive(self.root/'dispatch.lock'):
            await self._reconcile()
            if self.db.execute("SELECT 1 FROM orders WHERE state IN ('sending','unknown','cancel_unknown') LIMIT 1").fetchone():
                raise RuntimeError('Uncertain exchange outcome; monitor halted for reconciliation')
            entries=[self.order(r['id']) for r in self.db.execute("SELECT id FROM orders WHERE role='entry' AND state='acknowledged'").fetchall()]
            for entry in entries:
                if entry['response']['active']: await self._protect_entry(entry['id'])
            snapshot=await self.exchange.snapshot();self._snapshot(snapshot)
            actions=[]
            for position in self.positions():
                quote=snapshot['quotes'][position['symbol']];_quote(quote,datetime.now(timezone.utc))
                bid=number(quote['bid'])
                if bid>=number(position['target_price']) or bid<=number(position['stop_price']):
                    result=await self._close_position(position['parent'],'target' if bid>=number(position['target_price']) else 'stop_breached')
                    actions.append(dict(parent=position['parent'],action='exit_attempt',order=result['id'] if result else None))
                else:
                    await self._protect_entry(position['parent'])
                    actions.append(dict(parent=position['parent'],action='protected'))
            for row in self.db.execute("SELECT id FROM orders WHERE role='entry' AND state='prepared'").fetchall():
                result=await self._send(row['id'])
                if result['state']!='acknowledged': raise RuntimeError('Entry unresolved; monitor halted')
                await self._protect_entry(row['id'])
                actions.append(dict(parent=row['id'],action='entry_processed'))
            return dict(actions=actions,positions=self.positions())


class WallexMCP:
    """Dedicated trading boundary; the existing public/read-only client stays read-only."""
    READS=frozenset({'wallex_credit_sub_account_balances_details','wallex_basic_get_account_fees',
        'wallex_spot_get_markets','wallex_spot_get_open_order','wallex_spot_get_orders_client','wallex_basic_get_market_depth'})
    WRITES=frozenset({'wallex_spot_post_account_orders','wallex_spot_delete_orders_client'})

    def __init__(self,root,symbols,*,allow_writes=False,optional_symbols=()):
        self.root=Path(root);self.symbols=symbols;self.allow_writes=allow_writes
        self.optional_symbols=frozenset(optional_symbols)
        if not self.optional_symbols<=set(symbols)&set(DEFAULTS['symbols'][4:]): raise ValueError('Only configured reviewed chart markets may be optional')

    async def __aenter__(self):
        import os
        from .mcp_client import connect
        key=os.environ.get('WALLEX_API_KEY','')
        if not key or '\n' in key or '\r' in key: raise ValueError('Local WALLEX_API_KEY required')
        self.binding=hashlib.sha256(('credit_subaccount\n'+key).encode()).hexdigest()
        self.stack=AsyncExitStack()
        try:
            self.session,_,self.tools=await self.stack.enter_async_context(connect(dict(
                transport='streamable_http',url='https://api.wallex.ir/mcp',private_reads=True,timeout_seconds=35),self.root/'discovery'))
        except BaseException:
            await self.stack.aclose();raise
        return self

    async def __aexit__(self,*args): await self.stack.aclose()

    async def _call(self,name,args):
        if name not in self.READS and not (self.allow_writes is True and name in self.WRITES): raise PermissionError('Tool outside dedicated exchange boundary')
        if name not in self.tools: raise RuntimeError('Required official exchange capability is unavailable')
        result=await self.session.call_tool(name,arguments=args)
        if result.isError: raise RuntimeError('Wallex tool reported failure; remote text omitted')
        value=result.structuredContent
        if value is None:
            texts=[b.text for b in result.content if getattr(b,'type',None)=='text']
            if len(texts)!=1: raise ValueError('Expected one exchange JSON response')
            value=json.loads(texts[0])
        if not isinstance(value,dict) or value.get('success') is not True or 'result' not in value:
            raise RuntimeError('Exchange did not confirm success')
        return value

    async def place(self,payload): return (await self._call('wallex_spot_post_account_orders',payload))['result']
    async def get(self,key): return (await self._call('wallex_spot_get_orders_client',dict(client_id=key)))['result']
    async def cancel(self,key): return await self._call('wallex_spot_delete_orders_client',dict(client_id=key))

    async def snapshot(self):
        from .data import PublicClient
        observed_at=datetime.now(timezone.utc).isoformat()
        slots=asyncio.Semaphore(4)
        market_errors={}
        async def read(name,args):
            async with slots: return await self._call(name,args)
        async def depth(symbol):
            book=(await read('wallex_basic_get_market_depth',dict(symbol=symbol)))['result']
            bids=book.get('bid',book.get('bids'));asks=book.get('ask',book.get('asks'))
            if not bids or not asks: raise ValueError('Executable depth is unavailable')
            bid=max(bids,key=lambda r:number(r['price']));ask=min(asks,key=lambda r:number(r['price']))
            return dict(bid=float(number(bid['price'],True)),ask=float(number(ask['price'],True)),
                bid_quantity=float(number(bid['quantity'],True)),ask_quantity=float(number(ask['quantity'],True)),
                available_at=datetime.now(timezone.utc).isoformat(),source='https://api.wallex.ir/mcp')
        async def market_depth(symbol):
            if symbol not in self.optional_symbols: return await depth(symbol)
            try: return await asyncio.wait_for(depth(symbol),timeout=12)
            except Exception as error:
                market_errors[symbol]=dict(error_type=type(error).__name__,error_code='market_unavailable')
                return None
        async def official_limits():
            # MCP's market listing omits minimum order sizes; use the official public limits endpoint.
            clock=asyncio.get_running_loop().time()
            if not hasattr(self,'_limits_cache') or clock-self._limits_cache[0]>=300:
                client=PublicClient(self.root/'public',timeout=15)
                try:
                    async with slots: limits,receipt=await asyncio.to_thread(client.get,'/v1/markets',label='execution_limits')
                finally: client.session.close()
                self._limits_cache=(clock,limits,receipt)
            return self._limits_cache[1:]
        async def open_orders():
            orders=[]
            for page in range(1,101):
                response=await read('wallex_spot_get_open_order',dict(page=page,per_page=100))
                result=response['result']
                if isinstance(result,list): batch=result
                elif isinstance(result,dict) and isinstance(result.get('orders'),list): batch=result['orders']
                elif isinstance(result,dict) and all(isinstance(v,list) for v in result.values()): batch=[r for rows in result.values() for r in rows]
                else: raise ValueError('Unknown open-order response shape')
                orders.extend(batch)
                if len(batch)<100: return orders
            raise ValueError('Open-order pagination limit reached; snapshot incomplete')
        # Separate snapshots have their own bounded reads so a scan cannot queue out the protection monitor.
        tasks=[asyncio.create_task(job) for job in (
            read('wallex_credit_sub_account_balances_details',{}),
            read('wallex_basic_get_account_fees',{}),read('wallex_spot_get_markets',{}),
            official_limits(),open_orders(),
            *(market_depth(s) for s in self.symbols))]
        try: balances,fees,raw,limits_result,orders,*books=await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
        balances,fees,raw=balances['result'],fees['result'],raw['result']
        limits,receipt=limits_result
        listed=raw.get('symbols',raw.get('markets',raw))
        if isinstance(listed,list): listed={r['symbol']:r for r in listed}
        if limits.get('success') is not True: raise ValueError('Official venue limits unavailable')
        markets=limits['result']['symbols']
        quotes=dict(zip(self.symbols,books))
        if 'USDT' not in balances or any(k not in balances['USDT'] for k in ('available','total')): raise ValueError('Credit balance is incomplete')
        if number(balances['USDT']['available'])>number(balances['USDT']['total']): raise ValueError('Invalid credit balance')
        for symbol in self.symbols:
            if symbol in market_errors:
                quotes.pop(symbol);continue
            try:
                market=markets[symbol]
                for required in ('baseAsset','quoteAsset','stepSize','tickSize','minQty','minNotional'):
                    if required not in market: raise ValueError('Official market precision/limits unavailable')
                for required in ('makerFeeRate','takerFeeRate'): number(fees[symbol][required])
                current=listed[symbol]
                if current.get('base_asset',current.get('baseAsset'))!=market['baseAsset'] or current.get('quote_asset',current.get('quoteAsset'))!=market['quoteAsset']:
                    raise ValueError('Official market metadata disagrees')
                for field,other in (('stepSize','amount_precision'),('tickSize','price_precision')):
                    if current.get(other,current.get(field))!=market[field]: raise ValueError('Official venue precisions disagree')
                asset=symbol[:-4]
                if number(balances[asset]['available'])>number(balances[asset]['total']): raise ValueError('Invalid credit balance')
            except Exception as error:
                if symbol not in self.optional_symbols: raise
                market_errors[symbol]=dict(error_type=type(error).__name__,error_code='market_unavailable')
                quotes.pop(symbol)
        return dict(account_scope='credit_subaccount',account_binding=self.binding,
            observed_at=observed_at,balances=balances,
            markets={s:markets[s] for s in quotes},fees={s:fees[s] for s in quotes},quotes=quotes,open_orders=orders,market_errors=market_errors,
            limits_receipt=receipt)


async def preflight(config_path):
    from dotenv import load_dotenv
    from .context import save_json as save
    load_dotenv(REPO/'.env',override=True)
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    root=Path(config['run_dir']);root.mkdir(parents=True,exist_ok=True)
    report=dict(checked_at=datetime.now(timezone.utc).isoformat(),account_scope='credit_subaccount',
        writes_enabled=False,real_orders_submitted=0,ready_for_automatic_trading=False)
    try:
        async with WallexMCP(root,config['symbols'],allow_writes=False) as exchange:
            snapshot=await exchange.snapshot()
            save(root/'account-snapshot.json',snapshot)
            report.update(status='read_only_verified',account_binding=snapshot['account_binding'],
                available_usdt=snapshot['balances']['USDT']['available'],open_order_count=len(snapshot['open_orders']),
                markets={s:{k:snapshot['markets'][s][k] for k in ('stepSize','tickSize','minQty','minNotional','isMarketTypeEnable') if k in snapshot['markets'][s]} for s in config['symbols']})
            if config.get('account_binding') not in (None,snapshot['account_binding']): raise ValueError('Configured account changed')
            config['account_binding']=snapshot['account_binding']
            save(Path(config_path),config)
        report['remaining_checks']=['Native STOP_LIMIT execution/trigger behavior not live-verified','Agent-to-live execution is not armed']
    except Exception as error:
        report.update(status='blocked',error_type=type(error).__name__)
    save(root/'preflight.json',report)
    print(json.dumps(report,ensure_ascii=True))
    return report


async def monitor(config_path):
    """Explicitly started execution actor. Never installed or started by preflight."""
    from dotenv import load_dotenv
    from .context import save_json
    load_dotenv(REPO/'.env',override=True)
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config.get('enabled') is not True or config.get('stop_capabilities_verified') is not True:
        raise PermissionError('Execution monitor remains disabled; no exchange writes attempted')
    root=Path(config['run_dir']);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'monitor.lock'):
        async with WallexMCP(root,config['symbols'],allow_writes=True) as exchange:
            if exchange.binding!=config.get('account_binding'): raise ValueError('Preflight account binding changed')
            gateway=Gateway(root,config,exchange)
            try:
                while True:
                    latest=json.loads(Path(config_path).read_text(encoding='utf-8'))
                    if latest.get('enabled') is not True:
                        save_json(root/'monitor.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),status='stopped_by_config'))
                        break
                    if latest!=config: raise ValueError('Execution configuration changed; restart required')
                    try:
                        status=await gateway.monitor_once()
                    except Exception as error:
                        save_json(root/'monitor.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),status='halted',error_type=type(error).__name__))
                        raise RuntimeError('Execution monitor halted; inspect ledger and reconcile before restart') from None
                    save_json(root/'monitor.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),status='running',**status))
                    await asyncio.sleep(15)
            finally: gateway.db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Credit-account preflight (default); explicit disabled-by-default execution monitor.')
    parser.add_argument('--config',default=str(REPO/'config/live-execution.json'))
    parser.add_argument('--monitor',action='store_true',help='Requires separately enabled execution and verified native stops; never enabled by preflight')
    args=parser.parse_args()
    asyncio.run(monitor(args.config) if args.monitor else preflight(args.config))
