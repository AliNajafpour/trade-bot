"""Exercise real threaded chart-review assembly with fake providers/exchange; no network."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_live_execution import Exchange,snapshot
from wallex_lab.live_execution import DEFAULTS
from wallex_lab.live_runner import LiveRunner,LiveGateway,ArmedExchange,stamp


async def check(mode='llm_chart'):
    import pandas as pd
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        path=root/'config.json'
        extras=['ADAUSDT','TRXUSDT','LTCUSDT'] if mode=='legacy_ensemble' else DEFAULTS['symbols'][4:]
        symbol=extras[0]
        config=dict(DEFAULTS,run_dir=str(root),source=str(root/'source'),model='unused',
            account_binding='fixture',max_capital_usdt=100,llm_chart_symbols=extras if mode=='llm_chart' else [],
            legacy_technical_symbols=extras if mode=='legacy_ensemble' else [])
        path.write_text(json.dumps(config))
        raw=Exchange();raw.binding='fixture'
        current=snapshot()
        for symbol in DEFAULTS['symbols']:
            current['markets'][symbol]=dict(current['markets']['BTCUSDT'],symbol=symbol,baseAsset=symbol[:-4])
            current['quotes'][symbol]=deepcopy(current['quotes']['BTCUSDT'])
            current['fees'][symbol]=deepcopy(current['fees']['BTCUSDT'])
            current['balances'][symbol[:-4]]=dict(available='5',total='5')
        async def book():
            value=deepcopy(current)
            value['open_orders']=[deepcopy(order) for order in raw.orders.values() if order['active']]
            return value
        raw.snapshot=book
        runner=LiveRunner(path)
        runner.exchange=ArmedExchange(raw,path)
        runner.gateway=LiveGateway(root,config,runner.exchange)
        runner.reports=lambda:dict(market=None,macro=None)
        calls=[];overlays=[]
        def forecast(symbol,account,position=None):
            assert position is None,'OFF analysis read private positions'
            return None,dict(source='public fixture'),dict(symbol=symbol,technical_mode='llm_chart',
                tp=102.,sl=98.,entry=100.,execution_ask=100.,decision_at=stamp(),forecasts=[],support_flags=[]),pd.Timestamp.now(tz='UTC').floor('min')
        runner.chart_forecast=forecast
        async def legacy(*_):
            origin=pd.Timestamp.now(tz='UTC').floor('h').isoformat()
            return dict(status='ok',models={name:dict(status='ok',history=dict(forecast_origin_at=origin),
                close_quantiles_by_step=dict(q50=[103.]*24)) for name in ('kronos','chronos2')},
                rules=dict(status='ok',strategies=[dict(strategy='fixture')],candidates=[]))
        runner.legacy_forecast=legacy
        def chart(*args,**kwargs):
            overlays.append(kwargs['signals'])
            return root/'chart.png'
        def provider(packet,prompt,schema,output,**kwargs):
            calls.append((deepcopy(packet),schema))
            output.mkdir(parents=True,exist_ok=True)
            response=dict(action='enter',risk_multiplier=1.,reason='Offline chart fixture')
            files={}
            for name,value in (('packet.json',packet),('final.json',response)):
                target=output/name;target.write_text(json.dumps(value))
                files[name]=hashlib.sha256(target.read_bytes()).hexdigest()
            receipt=dict(status='ok',response=response,ended_utc=stamp(),files_sha256=files,
                receipt_path=str(output/'receipt.json'))
            (output/'receipt.json').write_text(json.dumps(receipt))
            return receipt
        async def stop(_):raise asyncio.CancelledError
        try:
            with patch('wallex_lab.council_charts.render_chart',side_effect=chart),patch('wallex_lab.providers.run_one',side_effect=provider):
                # Passing private barriers deliberately must still produce a public chart while OFF.
                _,_,prediction,decision=forecast(symbol,current)
                private=dict(prediction,tp=140.,sl=80.)
                await asyncio.to_thread(runner.review,symbol,None,{},private,decision,current,
                    dict(symbol=symbol,target_price=140.,stop_price=80.),{},[dict(private='owned')])
                assert overlays[-1]==[dict(stop_price=98.,target_price=102.)]
                assert calls[-1][1]['properties']['action']['enum']==['enter','wait','reject']
                calls.clear()
                with patch('wallex_lab.live_runner.asyncio.sleep',side_effect=stop):
                    try:await runner.chart_analysis()
                    except asyncio.CancelledError:pass
                assert len(calls)==len(extras) and not raw.writes
                assert runner.gateway.db.execute('SELECT count(*) FROM orders').fetchone()[0]==0
                for packet,_ in calls:
                    assert packet['mode']=='public_chart_analysis'
                    assert not {'account_balances','managed_positions','position','max_capital_usdt'} & packet.keys()
                    assert 'authenticated_taker_fee' not in packet['risk_policy']
                    assert packet['prediction']['forecasts']==[]
                assert all(row['status']=='analysis_only_disabled' for row in runner.chart_rows.values())
                # User activation requires a fresh private receipt, then the normal BUY + native STOP path.
                config.update(enabled=True,stop_capabilities_verified=True)
                path.write_text(json.dumps(config));runner.refresh_config()
                runner.monitor_ok=True;runner.state['monitor']=dict(status='healthy',updated_at=stamp())
                _,_,prediction,decision=forecast(symbol,current)
                gates={}
                if mode=='legacy_ensemble':
                    prediction.update(technical_mode=mode,legacy_technical=await legacy())
                    gates=runner.legacy_admission(symbol,prediction,current)
                    assert gates['eligible']
                    expired=deepcopy(prediction)
                    for model in expired['legacy_technical']['models'].values():
                        model['history']['forecast_origin_at']=(pd.Timestamp.now(tz='UTC').floor('h')-pd.Timedelta(hours=1)).isoformat()
                        model['close_quantiles_by_step']['q50']=[200.]+[90.]*23
                    assert not runner.legacy_admission(symbol,expired,current)['eligible'],'Past forecast peak admitted entry'
                    expired['legacy_technical']['rules']['candidates']=[dict(target_price=103.,strategy_sources=['fixture'])]
                    assert runner.legacy_admission(symbol,expired,current)['eligible'],'Positive rule setup was ignored'
                response,reports,receipt=await asyncio.to_thread(runner.review,symbol,None,{},prediction,
                    decision,current,None,gates,[])
                assert calls[-1][0]['mode']=='real_spot_user_armed'
                row=dict(prediction=prediction,manager_receipt=receipt)
                await runner.enter(symbol,response,reports,row)
                assert row['status']=='entry_processed'
                assert [order['type'] for order in raw.writes]==['LIMIT','STOP_LIMIT']
        finally:runner.gateway.db.close()
    print('PASS',mode,'threaded review; OFF privacy/zero intents; all configured review paths; receipt-backed fake BUY+STOP')

if __name__=='__main__':
    asyncio.run(check())
    asyncio.run(check('legacy_ensemble'))
