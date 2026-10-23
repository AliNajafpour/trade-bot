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


async def check():
    import pandas as pd
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        path=root/'config.json'
        extras=DEFAULTS['symbols'][4:]
        config=dict(DEFAULTS,run_dir=str(root),source=str(root/'source'),model='unused',
            account_binding='fixture',max_capital_usdt=100,llm_chart_symbols=extras)
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
                _,_,prediction,decision=forecast('XRPUSDT',current)
                private=dict(prediction,tp=140.,sl=80.)
                await asyncio.to_thread(runner.review,'XRPUSDT',None,{},private,decision,current,
                    dict(symbol='XRPUSDT',target_price=140.,stop_price=80.),{},[dict(private='owned')])
                assert overlays[-1]==[dict(stop_price=98.,target_price=102.)]
                assert calls[-1][1]['properties']['action']['enum']==['enter','wait','reject']
                calls.clear()
                with patch('wallex_lab.live_runner.asyncio.sleep',side_effect=stop):
                    try:await runner.chart_analysis()
                    except asyncio.CancelledError:pass
                assert len(calls)==6 and not raw.writes
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
                _,_,prediction,decision=forecast('XRPUSDT',current)
                response,reports,receipt=await asyncio.to_thread(runner.review,'XRPUSDT',None,{},prediction,
                    decision,current,None,{},[])
                assert calls[-1][0]['mode']=='real_spot_user_armed'
                row=dict(prediction=prediction,manager_receipt=receipt)
                await runner.enter('XRPUSDT',response,reports,row)
                assert row['status']=='entry_processed'
                assert [order['type'] for order in raw.writes]==['LIMIT','STOP_LIMIT']
        finally:runner.gateway.db.close()
    print('PASS threaded review without SQLite access; OFF chart privacy/zero intents; six real review paths; receipt-backed fake BUY+STOP')

if __name__=='__main__':asyncio.run(check())
