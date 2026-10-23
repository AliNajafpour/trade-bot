"""Offline user-switch, bounded sizing and real-ledger bridge checks; no keys/network."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_live_execution import Exchange, snapshot, prepare
from wallex_lab.live_execution import DEFAULTS
from wallex_lab.live_runner import ArmedExchange, LiveGateway, LiveRunner, size_entry, validate_activation
from wallex_lab.live_runner import stamp, error_details


def check_diagnostics():
    try:
        raise ValueError('history_does_not_end_at_decision')
    except ValueError as error:
        details = error_details(error)
    assert details['error_code'] == 'history_tail_pending'
    assert details['error_source'].startswith('check_live_runner.py:')
    assert details['reason'] and details['error_type'] == 'ValueError'
    try:
        raise ValueError('Bearer private-credential-must-never-leak')
    except ValueError as error:
        details = error_details(error)
    assert 'private-credential' not in json.dumps(details)


async def check():
    import pandas as pd
    assert len(DEFAULTS['symbols'])==10 and DEFAULTS['max_open_positions']==4
    for name in ('live-cloud.example.json','live-execution.json','weekly-source.example.json'):
        template=json.loads((Path(__file__).resolve().parents[1]/'config'/name).read_text())
        assert [s.replace('/','') for s in template['symbols']]==DEFAULTS['symbols']
        if name!='weekly-source.example.json':
            assert template['enabled'] is False and template['max_open_positions']==4
            for cap in ('max_position_fraction','risk_per_trade_fraction','max_account_stop_risk_fraction'):
                assert template[cap]==DEFAULTS[cap]
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / 'config.json'
        config = dict(DEFAULTS, run_dir=str(root), symbols=['BTCUSDT'], model='unused-fixture',
                      source=str(root / 'source'), account_binding='fixture', max_capital_usdt=10)
        def write(**changes):
            config.update(changes)
            path.write_text(json.dumps(config), encoding='utf-8')
        write()
        raw = Exchange()
        raw.binding = 'fixture'
        guarded = ArmedExchange(raw, path)
        runner = LiveRunner(path)
        runner.exchange = guarded
        runner.gateway = LiveGateway(root, config, guarded)
        runner.monitor_ok = True
        runner.state['monitor'] = dict(status='healthy', updated_at=stamp())
        forecast = dict(tp=102., sl=98., support_flags=[], forecasts=[dict(horizon_minutes=1440,
                        p_tp_first=.6, p_sl_first=.25, p_no_touch=.15)])
        runner.forecast = lambda *_: (None, dict(source='offline fixture'), deepcopy(forecast), pd.Timestamp.now(tz='UTC').floor('min'))
        runner.reports = lambda: dict(market=None, macro=None)
        runner.review = lambda *_: (_ for _ in ()).throw(AssertionError('Disabled runner called the manager'))
        try:
            ten=snapshot()
            for symbol in DEFAULTS['symbols']:
                ten['markets'][symbol]=dict(ten['markets']['BTCUSDT'],symbol=symbol,baseAsset=symbol[:-4])
                ten['quotes'][symbol]=deepcopy(ten['quotes']['BTCUSDT'])
                ten['fees'][symbol]=deepcopy(ten['fees']['BTCUSDT'])
            write(symbols=DEFAULTS['symbols'])
            runner.refresh_config()
            await runner.scan(ten)
            assert len(runner.state['last_scan']['results'])==10
            assert all(r['status']=='eligible_disabled' for r in runner.state['last_scan']['results'])
            assert not raw.writes and runner.gateway.db.execute('SELECT count(*) FROM orders').fetchone()[0]==0
            validate_activation(dict(config,enabled=True,stop_capabilities_verified=True),'fixture')
            try: validate_activation(dict(config,enabled=True,stop_capabilities_verified=True,max_open_positions=5),'fixture')
            except ValueError: pass
            else: raise AssertionError('Ten markets increased the four-position risk cap')
            for operation in (guarded.place({}),guarded.cancel('anything')):
                try: await operation
                except PermissionError: pass
                else: raise AssertionError('Disabled ten-market configuration crossed exchange boundary')
            write(symbols=['BTCUSDT'])
            runner.refresh_config()
            await runner.scan(snapshot())
            assert runner.state['last_scan']['results'][0]['status'] == 'eligible_disabled'
            assert not raw.writes and runner.gateway.db.execute('SELECT count(*) FROM orders').fetchone()[0] == 0
            working_forecast = runner.forecast
            runner.forecast = lambda *_: (_ for _ in ()).throw(ValueError('Wallex completed candle is still being published; retrying this decision'))
            await runner.scan(snapshot())
            delayed = runner.state['last_scan']['results'][0]
            assert delayed['status'] == 'data_pending' and delayed['error_code'] == 'wallex_anchor_pending'
            assert not raw.writes
            runner.forecast = working_forecast
            for operation in (guarded.place({}), guarded.cancel('anything')):
                try: await operation
                except PermissionError: pass
                else: raise AssertionError('Disabled command crossed exchange boundary')
            write(enabled=True)
            assert not guarded.allow_writes, 'Native-stop attestation was bypassed'
            write(stop_capabilities_verified=True)
            validate_activation(config, 'fixture')
            forecast['forecasts'][0].update(p_tp_first=.4, p_sl_first=.45)
            await runner.scan(snapshot())
            assert runner.state['last_scan']['results'][0]['status'] == 'below_entry_gate'
            assert not raw.writes and runner.gateway.db.execute('SELECT count(*) FROM orders').fetchone()[0] == 0
            forecast['forecasts'][0].update(p_tp_first=.6, p_sl_first=.25)
            runner.review = lambda *_: (dict(action='wait', risk_multiplier=.5, reason='fixture'), runner.reports(), None)
            await runner.scan(snapshot())
            assert runner.state['last_scan']['results'][0]['status'] == 'wait'
            assert not raw.writes and runner.gateway.db.execute('SELECT count(*) FROM orders').fetchone()[0] == 0
            try: validate_activation(dict(config, max_capital_usdt=0), 'fixture')
            except ValueError: pass
            else: raise AssertionError('Unbounded capital was accepted')
            try: validate_activation(dict(config, risk_per_trade_fraction=.01), 'fixture')
            except ValueError: pass
            else: raise AssertionError('Reviewed risk limit was increased')
            quantity = size_entry(config, snapshot(), [], 'BTCUSDT', 98, 1)
            assert quantity == '0.015', quantity
            assert size_entry(dict(config, max_capital_usdt=100), snapshot(), [], 'BTCUSDT', 98, .1) == '0.020'
            old = runner.gateway.prepare_entry('earlier-budget', 'BTCUSDT', '.015', '100', '98', '102', snapshot(),
                dict(decision_at=stamp(), horizon_minutes=1440, execution_ask=100., tp=102., sl=98.,
                     p_tp_first=.6, p_sl_first=.25, p_no_touch=.15), risk_multiplier=1)
            write(max_capital_usdt=1)
            runner.refresh_config()
            try: await runner.gateway.execute_entry(old)
            except ValueError: pass
            else: raise AssertionError('Persisted intent bypassed a reduced user capital budget')
            assert not raw.writes
            assert runner.gateway.order(old)['state'] == 'abandoned', 'Failed unsubmitted entry remained queued'
            write(max_capital_usdt=10)
            runner.refresh_config()
            # A valid manager-approved signal goes through the actual Gateway and native stop.
            row = {}
            await runner.enter('BTCUSDT', dict(action='enter', risk_multiplier=1, reason='fixture'), runner.reports(), row)
            assert row['status'] == 'entry_processed'
            assert len(raw.writes) == 2 and raw.writes[0]['side'] == 'BUY' and raw.writes[1]['type'] == 'STOP_LIMIT'
            assert runner.gateway.positions()[0]['quantity'] == '0.015'
            write(enabled=False)
            before = len(raw.writes)
            try: await runner.gateway.close_position(runner.gateway.positions()[0]['parent'], 'queued_before_disable')
            except PermissionError: pass
            else: raise AssertionError('Queued manager close bypassed a disabled switch')
            try: await guarded.cancel(raw.writes[1]['client_id'])
            except PermissionError: pass
            else: raise AssertionError('Disable switch cancelled native protection')
            assert len(raw.writes) == before
        finally:
            runner.gateway.db.close()
        failed_root = root / 'postprepare-failure'
        failed_root.mkdir()
        failed_path = failed_root / 'config.json'
        failed_path.write_text(json.dumps(dict(config, enabled=True, run_dir=str(failed_root))), encoding='utf-8')
        failed = LiveRunner(failed_path)
        failed_raw = Exchange()
        failed_raw.binding = 'fixture'
        failed.exchange = ArmedExchange(failed_raw, failed_path)
        failed.gateway = LiveGateway(failed_root, failed.config, failed.exchange)
        failed.forecast, failed.reports = runner.forecast, runner.reports
        failed.monitor_ok = True
        failed.state['monitor'] = dict(status='healthy', updated_at=stamp())
        arm = failed.arm_check
        calls = 0
        def expires_after_prepare():
            nonlocal calls
            calls += 1
            if calls == 3: raise PermissionError('Fixture switch changed after prepare')
            return arm()
        failed.arm_check = expires_after_prepare
        try:
            try: await failed.enter('BTCUSDT', dict(action='enter', risk_multiplier=1), failed.reports(), {})
            except PermissionError: pass
            else: raise AssertionError('Postprepare switch change was ignored')
            assert calls == 3 and not failed_raw.writes
            assert failed.gateway.db.execute("SELECT count(*) FROM orders WHERE role='entry' AND state='abandoned'").fetchone()[0] == 1
            await failed.gateway.monitor_once()
            assert not failed_raw.writes, 'Failed fresh entry was later submitted by protection'
        finally:
            failed.gateway.db.close()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = dict(DEFAULTS, enabled=True, stop_capabilities_verified=True,
                      account_binding='fixture', max_capital_usdt=100)
        raw = Exchange()
        gateway = LiveGateway(root, config, raw)
        try:
            key = prepare(gateway, 'before-disable')
            config['enabled'] = False
            try: await gateway.monitor_once()
            except PermissionError: pass
            else: raise AssertionError('Disabled monitor executed a queued entry')
            config['enabled'] = True
            await gateway.monitor_once()
            assert gateway.order(key)['state'] == 'abandoned' and not raw.writes
            stale = prepare(gateway, 'expired')
            context = gateway.order(stale)['context']
            context['evidence']['decision_at'] = '2000-01-01T00:00:00+00:00'
            with gateway.db:
                gateway.db.execute('UPDATE orders SET context=? WHERE id=?', (json.dumps(context), stale))
            await gateway.monitor_once()
            await gateway.monitor_once()
            assert gateway.order(stale)['state'] == 'abandoned' and not raw.writes
            restart = prepare(gateway, 'crashed-before-post')
            uncertain = prepare(gateway, 'possibly-posted')
            gateway._set(uncertain, 'sending')
            protected_child = gateway._record('protective-child', 'stop', dict(symbol='BTCUSDT', side='SELL', type='STOP_LIMIT', quantity='.1', price='97.8', stop_Price='98'), {}, parent=uncertain)
        finally:
            gateway.db.close()
        gateway = LiveGateway(root, config, raw)
        try:
            assert gateway.order(restart)['state'] == 'abandoned'
            assert gateway.order(uncertain)['state'] == 'sending'
            assert gateway.order(protected_child)['state'] == 'prepared', 'Native protection intent was discarded'
            try: await gateway.monitor_once()
            except RuntimeError: pass
            else: raise AssertionError('Uncertain order was ignored')
            assert gateway.order(uncertain)['state'] == 'unknown' and not raw.writes
        finally:
            gateway.db.close()
    print('PASS: ten-market OFF scan submits/cancels nothing; four-position/risk caps preserved; capital/native-stop gates; BUY+STOP; prepared entries retired; uncertain orders and protective children preserved')


if __name__ == '__main__':
    check_diagnostics()
    asyncio.run(check())
