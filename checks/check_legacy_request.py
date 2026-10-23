"""Runner/worker clock contract and cancellation check; no models, network or orders."""
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.live_runner import LiveRunner


async def check():
    with TemporaryDirectory() as temporary:
        root=Path(temporary);config=root/'config.json'
        config.write_text(json.dumps(dict(run_dir=str(root),legacy_model_dirs=dict(kronos='unused',tokenizer='unused',chronos2='unused'))))
        runner=LiveRunner(config)
        decision=pd.Timestamp.now(tz='UTC').floor('min')
        frame=pd.DataFrame(dict(timestamp=[decision-pd.Timedelta(minutes=1)],close=[100.]))
        quote=dict(ask=100.,bid=99.9,available_at=pd.Timestamp.now(tz='UTC').isoformat())
        requests=[];calls=[]
        class Worker:
            returncode=None
            pid=987654
            def __init__(self,command):self.command=command
            async def communicate(self):
                request=json.loads(Path(self.command[self.command.index('--request')+1]).read_text())
                requests.append(request)
                assert pd.Timestamp(request['decision_at'])==decision
                assert pd.Timestamp(request['asof'])>=pd.Timestamp(request['quote']['available_at'])
                output=Path(self.command[self.command.index('--output')+1])
                output.write_text(json.dumps(dict(status='ok',symbol='ADAUSDT',models={k:dict(status='ok') for k in ('kronos','chronos2')})))
                self.returncode=0
                return b'',b''
            async def wait(self):self.returncode=0
        async def spawn(*command,**kwargs):
            calls.append(command)
            if command[0]!='taskkill':assert kwargs['env']['OMP_NUM_THREADS']=='1'
            return Worker(command)
        with patch('wallex_lab.live_runner.asyncio.create_subprocess_exec',side_effect=spawn):
            result=await runner.legacy_forecast('ADAUSDT',frame,dict(quotes=dict(ADAUSDT=quote)))
        assert result['status']=='ok' and len(requests)==1
        async def cancelled(self):raise asyncio.CancelledError
        with patch.object(Worker,'communicate',cancelled),patch('wallex_lab.live_runner.asyncio.create_subprocess_exec',side_effect=spawn),\
                patch('wallex_lab.live_runner.os.killpg',create=True) as group:
            try:await runner.legacy_forecast('ADAUSDT',frame,dict(quotes=dict(ADAUSDT=quote)))
            except asyncio.CancelledError:pass
            else:raise AssertionError('Cancellation was swallowed')
            if sys.platform=='win32':assert calls[-1][:2]==('taskkill','/PID') and '/T' in calls[-1] and '/F' in calls[-1]
            else:group.assert_called_once()
    print('PASS separate completed-candle/current-quote clocks; isolated worker request; cancellation cleans process tree')


if __name__=='__main__':asyncio.run(check())
