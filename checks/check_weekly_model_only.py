"""Four-market model-only execution: no provider, reports or real exchange calls."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_live_execution import Exchange,snapshot
from wallex_lab.cloud import bootstrap
from wallex_lab.live_execution import DEFAULTS
from wallex_lab.live_runner import ArmedExchange,LiveGateway,LiveRunner,stamp,validate_activation

FOUR=['BTCUSDT','ETHUSDT','SOLUSDT','SUSDT']


async def check():
    import pandas as pd
    with TemporaryDirectory() as temporary:
        root=Path(temporary);path=root/'config.json'
        config=dict(DEFAULTS,enabled=True,stop_capabilities_verified=True,
            account_binding='fixture',max_capital_usdt=100,run_dir=str(root),source=str(root/'source'),
            model='unused-fixture',symbols=FOUR,decision_mode='weekly_model_only',adaptive_learning=False)
        path.write_text(json.dumps(config))
        raw=Exchange();raw.binding='fixture';book=snapshot()
        for symbol in FOUR:
            book['balances'][symbol[:-4]]=dict(available='5',total='5')
            book['quotes'][symbol]=deepcopy(book['quotes']['BTCUSDT'])
            book['markets'][symbol]=dict(book['markets']['BTCUSDT'],symbol=symbol,baseAsset=symbol[:-4])
            book['fees'][symbol]=deepcopy(book['fees']['BTCUSDT'])
        async def current_book():
            value=deepcopy(book)
            value['open_orders']=[deepcopy(r) for r in raw.orders.values() if r['active']]
            return value
        raw.snapshot=current_book
        runner=LiveRunner(path);runner.exchange=ArmedExchange(raw,path)
        runner.gateway=LiveGateway(root,config,runner.exchange)
        runner.monitor_ok=True;runner.state['monitor']=dict(status='healthy',updated_at=stamp())
        probability=.55
        def forecast(*_):
            return None,{},dict(tp=102.,sl=98.,support_flags=[],forecasts=[dict(
                horizon_minutes=1440,p_tp_first=probability,p_sl_first=.25,p_no_touch=.75-probability)]),pd.Timestamp.now(tz='UTC').floor('min')
        runner.forecast=forecast
        runner.review=lambda *_:(_ for _ in ()).throw(AssertionError('Model-only route called BoxAPI'))
        try:
            validate_activation(config,'fixture')
            await runner.scan(deepcopy(book))
            assert all(r['status']=='below_entry_gate' for r in runner.state['last_scan']['results'])
            assert not raw.writes,'Exactly 55 percent must not enter'
            probability=.6
            config['enabled']=False;path.write_text(json.dumps(config));runner.refresh_config()
            await runner.scan(deepcopy(book))
            assert all(r['status']=='eligible_disabled' for r in runner.state['last_scan']['results']) and not raw.writes
            config['enabled']=True;path.write_text(json.dumps(config));runner.refresh_config()
            probability=.4
            try:await runner.enter('BTCUSDT',dict(action='enter',risk_multiplier=1),{}, {})
            except ValueError as error:assert 'Fresh executable score failed' in str(error)
            else:raise AssertionError('Rescoring below the gate still submitted an entry')
            assert not raw.writes
            probability=.6
            with patch('wallex_lab.intelligence_reports.ReportStore',side_effect=AssertionError('Model-only route read stale reports')):
                await runner.scan(deepcopy(book))
            rows=runner.state['last_scan']['results']
            assert all(r['status']=='entry_processed' for r in rows),rows
            assert len(raw.writes)==8 and len(runner.gateway.positions())==4
            assert all(raw.writes[i]['side']=='BUY' and raw.writes[i+1]['type']=='STOP_LIMIT' for i in range(0,8,2))
            for row in rows:
                context=runner.gateway.order(row['order_id'])['context']
                assert context['evidence']['decision_mode']=='weekly_model_only'
                assert 'manager_receipt_path' not in context['evidence']
            before=len(raw.writes)
            runner.forecast=lambda *_:(_ for _ in ()).throw(AssertionError('Owned model-only trade was rescored for discretionary exit'))
            await runner.scan(deepcopy(book))
            assert all(r['status']=='holding_model_only' for r in runner.state['last_scan']['results'])
            assert len(raw.writes)==before,'Model-only hold must not close early'
            assert runner.state['decision_mode']=='weekly_model_only'
            for change in [dict(symbols=FOUR+['XRPUSDT']),dict(decision_mode='typo'),dict(legacy_technical_symbols=['BTCUSDT'])]:
                try:validate_activation(dict(config,**change),'fixture')
                except ValueError:pass
                else:raise AssertionError('Invalid model-only configuration was accepted')
        finally:runner.gateway.db.close()

    with TemporaryDirectory() as temporary,patch.dict('os.environ',{'WALLEX_EXECUTION_PRESET':'weekly_model_only','WALLEX_FORCE_DISABLE':'false'}):
        path,_=bootstrap(temporary);config=json.loads(path.read_text())
        assert config['symbols']==FOUR and config['decision_mode']=='weekly_model_only'
        assert config['enabled'] is False and config['adaptive_learning'] is False
        assert not any(config[k] for k in ('llm_chart_symbols','probability_symbols','legacy_technical_symbols'))
        config.update(account_binding='owner',max_capital_usdt=17,stop_capabilities_verified=True)
        path.write_text(json.dumps(config));bootstrap(temporary)
        assert json.loads(path.read_text())==config,'Restart must preserve capital, binding and model-only mode'
        config.update(enabled=True);path.write_text(json.dumps(config));bootstrap(temporary)
        assert json.loads(path.read_text())==config,'A matching mode must not disable an owner-armed account'
        config.update(decision_mode='council');path.write_text(json.dumps(config))
        try:bootstrap(temporary)
        except ValueError:pass
        else:raise AssertionError('Preset changed while account was armed')
        config.update(enabled=False);path.write_text(json.dumps(config))
        from wallex_lab.live_execution import Gateway
        ledger=Gateway(Path(config['run_dir']),config,None)
        try:
            key=ledger._record('old-extra-market','entry',dict(symbol='XRPUSDT'),
                dict(barriers=dict(stop_price='98',target_price='102')))
            ledger._set(key,'acknowledged',dict(executedQty='1',executedSum='100',fills=[],active=False))
            try:bootstrap(temporary)
            except ValueError as error:assert 'Reconcile all positions' in str(error)
            else:raise AssertionError('A disabled open position was dropped from protection')
            ledger._set(key,'unknown',dict(executedQty='0',fills=[],active=False))
            try:bootstrap(temporary)
            except ValueError as error:assert 'Reconcile all positions' in str(error)
            else:raise AssertionError('An uncertain order was discarded by the preset change')
            ledger._set(key,'acknowledged',dict(executedQty='0',fills=[],active=False))
            bootstrap(temporary)
        finally:ledger.db.close()
    print('PASS: four assets, strict >55%, real Gateway BUY+STOP on fake exchange, no BoxAPI/report dependency, owned barriers only, guarded OFF migration')


if __name__=='__main__':asyncio.run(check())
