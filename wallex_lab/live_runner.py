"""User-armed Weekly/late spot actor. Deployment never arms this process."""
import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

from .context import save_json
from .live_execution import DEFAULTS, Gateway, WallexMCP, normalize, number
from .locking import exclusive


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_config(path):
    config = dict(DEFAULTS, **json.loads(Path(path).read_text(encoding='utf-8')))
    if type(config['enabled']) is not bool:
        raise ValueError('enabled must be boolean')
    return config


def validate_activation(config, binding):
    """Shared with the user control: no credentials or network required."""
    if config.get('enabled') is not True:
        raise PermissionError('Execution is disabled')
    if not binding or config.get('account_binding') != binding:
        raise ValueError('Authenticated account binding must match the selected account')
    if config.get('account_scope') != 'credit_subaccount':
        raise ValueError('Only the selected credit subaccount is supported')
    if config.get('stop_capabilities_verified') is not True:
        raise PermissionError('User must verify native STOP_LIMIT behavior before activation')
    number(config.get('max_capital_usdt', 0), True)
    for name in ('max_position_fraction', 'risk_per_trade_fraction', 'max_account_stop_risk_fraction'):
        if not 0 < number(config.get(name, DEFAULTS[name])) <= number(DEFAULTS[name]):
            raise ValueError('Risk caps cannot exceed the reviewed execution policy')
    count = config.get('max_open_positions', DEFAULTS['max_open_positions'])
    if type(count) is not int or not 1 <= count <= DEFAULTS['max_open_positions']:
        raise ValueError('Invalid maximum open positions')
    symbols = config.get('symbols')
    if not isinstance(symbols, list) or not symbols or len(set(symbols)) != len(symbols) or any(s not in DEFAULTS['symbols'] for s in symbols):
        raise ValueError('Only reviewed USDT spot markets are supported')


class ArmedExchange:
    """Hot-read the user's switch immediately before every remote mutation."""
    def __init__(self, exchange, config_path):
        self.exchange, self.config_path = exchange, Path(config_path)

    @property
    def allow_writes(self):
        try:
            validate_activation(read_config(self.config_path), self.exchange.binding)
            return True
        except (OSError, ValueError, TypeError, KeyError, PermissionError):
            return False

    async def snapshot(self):
        # An incomplete multi-market snapshot is never allowed to block protection indefinitely.
        return await asyncio.wait_for(self.exchange.snapshot(), timeout=25)

    async def get(self, key):
        return await self.exchange.get(key)

    async def place(self, payload):
        if not self.allow_writes:
            raise PermissionError('User execution switch is off or invalid')
        return await self.exchange.place(payload)

    async def cancel(self, key):
        if not self.allow_writes:
            raise PermissionError('User execution switch is off or invalid')
        return await self.exchange.cancel(key)


def size_entry(config, snapshot, positions, symbol, stop, multiplier):
    """Bound both total deployed cash and loss at the stop, using actual account fees."""
    price = number(snapshot['quotes'][symbol]['ask'], True)
    fee = number(snapshot['fees'][symbol]['takerFeeRate'])
    cap = number(config['max_capital_usdt'], True)
    cost = sum((number(p['cost_basis']) for p in positions), Decimal(0))
    marked = sum((number(p['quantity']) * number(snapshot['quotes'][p['symbol']]['bid']) for p in positions), Decimal(0))
    equity = min(cap, number(snapshot['balances']['USDT']['total']) + marked)
    available = min(number(snapshot['balances']['USDT']['available']), max(Decimal(0), cap - cost))
    gap = number(config['stop_limit_gap_bps']) / 10000
    loss = price * (1 + fee) - number(stop) * (1 - gap) * (1 - fee)
    if fee >= 1 or loss <= 0:
        raise ValueError('Invalid authenticated economics')
    used = sum((max(Decimal(0), number(p['cost_basis']) - number(p['quantity']) * number(p['stop_price']) * (1 - gap) * (1 - number(snapshot['fees'][p['symbol']]['takerFeeRate']))) for p in positions), Decimal(0))
    quantity = min(available / (price * (1 + fee)),
                   equity * number(config['max_position_fraction']) / price,
                   equity * number(config['risk_per_trade_fraction']) * number(multiplier) / loss,
                   max(Decimal(0), equity * number(config['max_account_stop_risk_fraction']) - used) / loss,
                   number(snapshot['quotes'][symbol]['ask_quantity']))
    market = snapshot['markets'][symbol]
    quantity = normalize(quantity, market['stepSize'])
    if number(quantity) < number(market['minQty']) or number(quantity) * price < number(market['minNotional']):
        raise ValueError('Approved cash/risk cannot fund the venue minimum')
    return quantity


class LiveGateway(Gateway):
    def __init__(self, root, config, exchange):
        super().__init__(root, config, exchange)
        # Restart never grants an old, unsubmitted manager decision a new execution window.
        self.retire_prepared_entries()

    def _retire_prepared_entries(self, key=None):
        with self.db:
            self.db.execute("UPDATE orders SET state='abandoned',error='fresh_manager_review_required' "
                            "WHERE role='entry' AND state='prepared' AND (? IS NULL OR id=?)", (key, key))

    def retire_prepared_entries(self, key=None):
        with exclusive(self.root / 'dispatch.lock'):
            self._retire_prepared_entries(key)

    def _risk(self, payload, context, snapshot):
        super()._risk(payload, context, snapshot)
        # Re-check the absolute user budget on persisted intents too, including after restart.
        maximum = size_entry(self.config, snapshot, self.positions(), payload['symbol'],
                             context['barriers']['stop_price'], context['risk_multiplier'])
        if number(payload['quantity']) > number(maximum):
            raise ValueError('Intent exceeds current user capital or stop-risk budget')

    async def execute_entry(self, key):
        try:
            return await super().execute_entry(key)
        finally:
            # Sending/unknown orders may have reached the venue and MUST retain their state.
            if self.order(key)['state'] == 'prepared':
                self.retire_prepared_entries(key)

    async def monitor_once(self):
        """Protect/reconcile inventory; only the current manager turn can dispatch a BUY."""
        self._writes()
        with exclusive(self.root / 'dispatch.lock'):
            self._retire_prepared_entries()
            await self._reconcile()
            if self.db.execute("SELECT 1 FROM orders WHERE state IN ('sending','unknown','cancel_unknown') LIMIT 1").fetchone():
                raise RuntimeError('Uncertain exchange outcome; monitor halted for reconciliation')
            entries = [self.order(r['id']) for r in self.db.execute("SELECT id FROM orders WHERE role='entry' AND state='acknowledged'")]
            for entry in entries:
                if entry['response']['active']:
                    await self._protect_entry(entry['id'])
            snapshot = await self.exchange.snapshot()
            self._snapshot(snapshot)
            actions = []
            for position in self.positions():
                from .spot import _quote
                quote = snapshot['quotes'][position['symbol']]
                _quote(quote, datetime.now(timezone.utc))
                bid = number(quote['bid'])
                if bid >= number(position['target_price']) or bid <= number(position['stop_price']):
                    result = await self._close_position(position['parent'], 'target' if bid >= number(position['target_price']) else 'stop_breached')
                    actions.append(dict(parent=position['parent'], action='exit_attempt', order=result['id'] if result else None))
                else:
                    await self._protect_entry(position['parent'])
                    actions.append(dict(parent=position['parent'], action='protected'))
            return dict(actions=actions, positions=self.positions())


class LiveRunner:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = read_config(self.config_path)
        self.root = Path(self.config['run_dir']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = dict(updated_at=stamp(), status='starting_disabled', enabled=False,
                          execution_ready=False, positions=[], recent_orders=[], last_scan=None)
        self.gateway = None
        self.predictor = None
        self.monitor_ok = False

    def publish(self):
        self.state['updated_at'] = stamp()
        if self.gateway:
            self.state['positions'] = self.gateway.positions()
            orders = []
            for row in self.gateway.db.execute('SELECT id FROM orders ORDER BY rowid DESC LIMIT 30'):
                order = self.gateway.order(row['id'])
                orders.append({k: order[k] for k in ('id', 'role', 'parent', 'payload', 'state', 'response', 'error')})
            self.state['recent_orders'] = orders
        save_json(self.root / 'status.json', self.state)

    def refresh_config(self):
        config = read_config(self.config_path)
        if Path(config['run_dir']).resolve() != self.root:
            raise ValueError('Execution ledger directory cannot change while running')
        if self.gateway and config['account_binding'] != self.gateway.config['account_binding']:
            raise ValueError('Execution ledger account cannot change while running')
        self.config = config
        if self.gateway:
            self.gateway.config = config
        return config

    def arm_check(self):
        config = self.refresh_config()
        validate_activation(config, self.exchange.exchange.binding)
        observed = self.state.get('monitor', {}).get('updated_at')
        fresh = observed and 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(observed)).total_seconds() <= 30
        if not self.monitor_ok or not fresh:
            raise PermissionError('Protection monitor is not healthy')
        return config

    async def protection(self):
        while True:
            try:
                self.refresh_config()
                if self.gateway and self.exchange.allow_writes:
                    await self.gateway.monitor_once()
                    self.monitor_ok = True
                    self.state['monitor'] = dict(status='healthy', updated_at=stamp())
                else:
                    self.monitor_ok = False
                    self.state['monitor'] = dict(status='disabled', updated_at=stamp())
            except Exception as error:
                self.monitor_ok = False
                self.state['monitor'] = dict(status='blocked', error_type=type(error).__name__, updated_at=stamp())
            self.publish()
            await asyncio.sleep(max(3, float(self.config.get('monitor_seconds', 5))))

    def history(self, symbol, decision):
        # Reuse only the existing history methods; never construct its PaperBroker.
        from .survival_paper import Runner, completed_history
        adapter = object.__new__(Runner)
        adapter.config, adapter.root, adapter.proxy_sources = self.config, self.root, {}
        needed = self.predictor.metadata['minimum_history_minutes']
        if self.config.get('history_mode') in ('kucoin_proxy', 'binance_proxy'):
            if self.config['history_mode'] != 'kucoin_proxy':
                raise ValueError('Use native Wallex or explicitly configured KuCoin/MEXC history')
            return adapter.history(symbol, decision, needed)
        import pandas as pd
        from urllib.parse import urlencode
        from .context import fetch_public
        from .data import load_market, normalize_history
        path = Path(self.config['source']) / 'market-data' / symbol / 'wallex-cache.csv'
        try:
            return completed_history(path, decision, needed), dict(source='Wallex native M1')
        except (OSError, ValueError):
            frame = load_market(path, 60) if path.exists() else pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        begin = int(decision.timestamp()) - needed * 60
        # Fetch the latest section on every retry; historical holes are never fabricated.
        if not frame.empty:
            begin = max(begin, int(frame.timestamp.max().timestamp()) - 60)
            expected = pd.date_range(decision - pd.Timedelta(minutes=needed), decision - pd.Timedelta(minutes=1), freq='min')
            missing = expected.difference(pd.DatetimeIndex(frame.timestamp))
            if len(missing): begin = min(begin, int(missing[0].timestamp()))
        receipts = []
        while begin < int(decision.timestamp()):
            end = min(int(decision.timestamp()), begin + 1000 * 60)
            query = urlencode(dict(symbol=symbol, resolution='1', **{'from': begin, 'to': end - 1}))
            raw, receipt = fetch_public('https://api.wallex.ir/v1/udf/history?' + query, self.root / 'history-receipts' / symbol, 'wallex_m1', 12)
            if receipt.get('status') != 200:
                raise ValueError('Wallex history unavailable')
            fresh, _ = normalize_history(json.loads(raw), 60, decision.timestamp())
            frame = pd.concat([frame, fresh]).drop_duplicates('timestamp', keep='last').sort_values('timestamp').tail(needed + 5)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            frame.to_csv(temporary, index=False)
            from .locking import atomic_replace
            atomic_replace(temporary, path)
            receipts.append(receipt)
            begin = end
        return completed_history(path, decision, needed), dict(source='Wallex native M1', receipts=receipts)

    def forecast(self, symbol, snapshot, position=None):
        import pandas as pd
        import torch
        from .barrier_survival import SurvivalPredictor
        if self.predictor is None:
            torch.set_num_threads(1)
            model = Path(self.config['model'])
            expected = self.config.get('model_metadata_sha256')
            if expected and hashlib.sha256((model / 'model.json').read_bytes()).hexdigest() != expected:
                raise ValueError('Weekly model identity mismatch')
            self.predictor = SurvivalPredictor.load(model)
        decision = pd.Timestamp.now(tz='UTC').floor('min')
        frame, provenance = self.history(symbol, decision)
        ask = normalize(snapshot['quotes'][symbol]['ask'], snapshot['markets'][symbol]['tickSize'])
        precision = snapshot['markets'][symbol]['tickSize']
        tp = position['target_price'] if position else normalize(number(ask) * Decimal('1.02'), precision)
        sl = position['stop_price'] if position else normalize(number(ask) * Decimal('.98'), precision)
        prediction = self.predictor.predict(frame, symbol, decision, float(frame.close.iloc[-1]), tp=float(tp), sl=float(sl))
        prediction.update(anchor_close=float(frame.close.iloc[-1]), execution_ask=float(ask),
                          probability_semantics='Trade-price first touch; not guaranteed executable fill')
        prediction['forecasts'] = [f for f in prediction['forecasts'] if f['horizon_minutes'] == 1440]
        if len(prediction['forecasts']) != 1:
            raise ValueError('One exact 24h forecast required')
        return frame, provenance, prediction, decision

    def reports(self):
        from .intelligence_reports import ReportStore
        store = ReportStore(Path(self.config['source']) / 'reports')
        return {kind: store.get(kind, stamp()) for kind in ('market', 'macro')}

    def review(self, symbol, frame, provenance, prediction, decision, snapshot, position, gates):
        from .council_charts import render_chart
        from .providers import run_one
        from .survival_news import PROMPT, PRESET, manager_schema
        reports = self.reports()
        directory = self.root / 'reviews' / f'{decision.value}-{symbol}'
        chart = render_chart(frame, symbol[:-4] + '/USDT', decision, directory / 'chart.png',
                             signals=[dict(stop_price=prediction['sl'], target_price=prediction['tp'])], history=provenance)
        packet = dict(paper_only=False, mode='real_spot_user_armed', asof=stamp(), symbol=symbol,
                      preset=PRESET, quote=snapshot['quotes'][symbol], prediction=prediction,
                      position=position, account_balances=snapshot['balances'],
                      managed_positions=self.gateway.positions() if self.gateway else [],
                      max_capital_usdt=self.config.get('max_capital_usdt'),
                      risk_policy=dict({k:self.config[k] for k in ('max_position_fraction', 'risk_per_trade_fraction', 'max_account_stop_risk_fraction', 'max_open_positions')},
                                       min_notional=snapshot['markets'][symbol]['minNotional'],
                                       authenticated_taker_fee=snapshot['fees'][symbol]['takerFeeRate']),
                      reports=reports, history_provenance=provenance, admission=gates)
        # Only replace the account mode sentence; retain the reviewed gates and no-tool contract.
        prompt = PROMPT.replace('one isolated SPOT PAPER experiment', 'one user-armed cash-funded REAL SPOT account')
        receipt = run_one(packet, prompt, manager_schema(bool(position)), directory / 'inference',
                          timeout=min(120, max(1, float(self.config.get('manager_timeout_seconds', 60)))), images=[chart])
        if receipt.get('status') != 'ok':
            raise ValueError('Manager did not return a validated decision')
        return receipt['response'], reports, receipt.get('receipt_path')

    async def enter(self, symbol, response, reviewed_reports, row):
        config = self.arm_check()
        snapshot = await self.exchange.snapshot()
        frame, provenance, prediction, decision = await asyncio.to_thread(self.forecast, symbol, snapshot)
        import pandas as pd
        if pd.Timestamp.now(tz='UTC').floor('min') != decision:
            raise ValueError('A new candle closed during scoring; next scan must rescore')
        from .survival_news import admission
        quote = snapshot['quotes'][symbol]
        gates = admission(prediction, prediction['forecasts'][0], quote['ask'], quote['bid'], float(number(snapshot['fees'][symbol]['takerFeeRate']) * 10000))
        if not gates['eligible']:
            raise ValueError('Fresh executable score failed the entry gate')
        current_reports = self.reports()
        if any((reviewed_reports[k] or {}).get('report_id') != (current_reports[k] or {}).get('report_id') for k in reviewed_reports):
            raise ValueError('Reports changed after manager review')
        multiplier = min(number(response['risk_multiplier']), *(number((r or {}).get('assessment', {}).get('risk_multiplier', 1)) for r in current_reports.values()))
        if multiplier <= 0 or multiplier > 1:
            raise ValueError('No positive approved risk budget')
        config = self.arm_check()
        quantity = size_entry(config, snapshot, self.gateway.positions(), symbol, prediction['sl'], multiplier)
        forecast = prediction['forecasts'][0]
        ask = normalize(quote['ask'], snapshot['markets'][symbol]['tickSize'])
        evidence = dict(decision_at=stamp(), horizon_minutes=1440, execution_ask=ask, tp=prediction['tp'], sl=prediction['sl'],
                        support_flags=prediction['support_flags'], **{k: forecast[k] for k in ('p_tp_first', 'p_sl_first', 'p_no_touch')})
        key = self.gateway.prepare_entry(f'weekly-live-{decision.value}-{symbol}', symbol, quantity, ask, prediction['sl'], prediction['tp'], snapshot, evidence, risk_multiplier=float(multiplier))
        try:
            self.arm_check()
            result = await self.gateway.execute_entry(key)
        finally:
            if self.gateway.order(key)['state'] == 'prepared':
                self.gateway.retire_prepared_entries(key)
        row.update(status='entry_processed', order_id=result['id'], final_prediction=prediction)

    async def scan(self, snapshot):
        from .survival_news import admission
        result = dict(started_at=stamp(), completed=False, results=[])
        self.state['last_scan'] = result
        for symbol in self.config['symbols']:
            row = dict(symbol=symbol, status='checking')
            result['results'].append(row)
            try:
                position = next((p for p in self.gateway.positions() if p['symbol'] == symbol), None) if self.gateway else None
                frame, provenance, prediction, decision = await asyncio.to_thread(self.forecast, symbol, snapshot, position)
                row.update(prediction=prediction, history_provenance=provenance)
                quote = snapshot['quotes'][symbol]
                gates = admission(prediction, prediction['forecasts'][0], quote['ask'], quote['bid'], float(number(snapshot['fees'][symbol]['takerFeeRate']) * 10000))
                row['admission'] = gates
                if not position and not gates['eligible']:
                    row['status'] = 'below_entry_gate'
                    continue
                if not self.exchange.allow_writes:
                    row['status'] = 'eligible_disabled' if not position else 'holding_disabled'
                    continue
                self.arm_check()
                # Each symbol gets a fresh account/book snapshot, not an earlier scan's price.
                snapshot = await self.exchange.snapshot()
                response, reports, receipt = await asyncio.to_thread(self.review, symbol, frame, provenance, prediction, decision, snapshot, position, gates)
                row.update(manager=response, manager_receipt=receipt, status=response['action'])
                self.arm_check()
                if position and response['action'] == 'close':
                    closed = await self.gateway.close_position(position['parent'], 'news_manager')
                    row['order_id'] = closed['id'] if closed else None
                elif not position and response['action'] == 'enter':
                    await self.enter(symbol, response, reports, row)
            except Exception as error:
                row.update(status='blocked', error_type=type(error).__name__)
            self.publish()
        result.update(completed=True, finished_at=stamp())
        self.publish()

    async def run(self):
        """Only user-controlled persisted activation can grant mutation capability."""
        with exclusive(self.root / 'live-runner.lock'):
            while True:
                monitor = None
                try:
                    self.refresh_config()
                    async with WallexMCP(self.root / 'exchange', self.config['symbols'], allow_writes=True) as raw:
                        self.exchange = ArmedExchange(raw, self.config_path)
                        snapshot = await self.exchange.snapshot()
                        self.state.update(account_binding=snapshot['account_binding'], account=dict(observed_at=snapshot['observed_at'], balances=snapshot['balances'], fees=snapshot['fees']))
                        if self.config.get('account_binding') == snapshot['account_binding']:
                            self.gateway = LiveGateway(self.root, self.config, self.exchange)
                        elif self.config['enabled']:
                            raise ValueError('Real account has not been bound by user activation')
                        monitor = asyncio.create_task(self.protection())
                        next_scan = 0.
                        while True:
                            self.refresh_config()
                            armed = self.exchange.allow_writes
                            snapshot = await self.exchange.snapshot()
                            self.state.update(status='running' if armed else 'ready_disabled', enabled=armed,
                                execution_ready=armed and self.monitor_ok,
                                account_binding=snapshot['account_binding'],
                                account=dict(observed_at=snapshot['observed_at'], balances=snapshot['balances'], fees=snapshot['fees']))
                            if self.gateway is None and self.config.get('account_binding') == snapshot['account_binding']:
                                self.gateway = LiveGateway(self.root, self.config, self.exchange)
                            self.publish()
                            if time.monotonic() >= next_scan:
                                await self.scan(snapshot)
                                next_scan = time.monotonic() + max(60, float(self.config.get('scan_seconds', 60)))
                            await asyncio.sleep(5)
                except Exception as error:
                    self.state.update(status='blocked', enabled=False, execution_ready=False, error_type=type(error).__name__)
                    self.publish()
                    await asyncio.sleep(10)
                finally:
                    if monitor:
                        monitor.cancel()
                        try: await monitor
                        except asyncio.CancelledError: pass
                    if self.gateway:
                        self.gateway.db.close()
                        self.gateway = None
                    self.monitor_ok = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        asyncio.run(LiveRunner(args.config).run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
