"""Offline clock/composition checks. Real foundation benchmarks use the CLI separately."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wallex_lab.legacy_technical import inputs, forecast, run_stage


def check():
    with TemporaryDirectory() as temporary:
        root=Path(temporary);path=root/'history.csv'
        decision=pd.Timestamp('2026-10-24T12:23:00Z')
        close=100+np.sin(np.arange(10080)/50)
        frame=pd.DataFrame(dict(timestamp=pd.date_range(decision-pd.Timedelta(minutes=10080),periods=10080,freq='min'),
            open=close,high=close+1,low=close-1,close=close,volume=10))
        frame.to_csv(path,index=False)
        request=dict(symbol='ADAUSDT',history_path=str(path),asof=decision.isoformat(),
            quote=dict(ask=101,bid=100.99,available_at=decision.isoformat()),
            model_dirs=dict(kronos='unused',tokenizer='unused',chronos2='unused'),run_dir=str(root/'output'))
        native,hourly,history=inputs(request)
        assert len(native)==10080 and len(hourly)==128
        assert hourly.timestamp.iloc[-1]+pd.Timedelta(hours=1)==decision.floor('h')
        assert not history['interpolated'] and history['used_m1_rows']==10080
        current=dict(request,decision_at=decision.isoformat(),asof=(decision+pd.Timedelta(seconds=20)).isoformat())
        assert inputs(current)[2]==history, 'Wall clock seconds changed completed history decision'
        for broken in (dict(request,symbol='BTCUSDT'),dict(request,asof='2026-10-24T12:23:00'),
                dict(request,asof=(decision+pd.Timedelta(minutes=1)).isoformat())):
            try: inputs(broken)
            except ValueError: pass
            else: raise AssertionError('Unsupported symbol or incomplete/stale history accepted')
        frame.drop(index=500).to_csv(path,index=False)
        try: inputs(request)
        except ValueError: pass
        else: raise AssertionError('Missing minute was compressed or filled')
        frame.to_csv(path,index=False)
        request_path=root/'request.json';request_path.write_text(json.dumps(request))
        calls=[]
        def isolated_stage(command,env):
            stage=command[command.index('--stage')+1];calls.append(stage)
            assert env['OMP_NUM_THREADS']=='1'
            output=Path(command[command.index('--output')+1])
            output.write_text(json.dumps(dict(status='ok',symbol='ADAUSDT',model=stage,history=history)))
            return subprocess.CompletedProcess(command,0)
        with patch('wallex_lab.legacy_technical.run_stage',side_effect=isolated_stage):
            result=forecast(request,request_path)
        assert calls==['kronos','chronos2'] and set(result['models'])==set(calls)
        assert result['technical_mode']=='legacy_ensemble' and result['rules']['status']=='ok'
        current_path=root/'current-request.json';current_path.write_text(json.dumps(current))
        with patch('wallex_lab.legacy_technical.run_stage',side_effect=isolated_stage):
            current_result=forecast(current,current_path)
        assert current_result['decision_at']==decision.isoformat() and current_result['rules']['asof']==current['asof']
        assert len(result['rules']['strategies'])==30
        assert {row['strategy'] for row in result['rules']['strategies']}=={
            'ema_crossover','trend_pullback','donchian_breakout','breakout_retest','bollinger_reentry',
            'rsi_recovery','vwap_reclaim','squeeze_breakout','support_sweep','volume_momentum'}
        def changed_history(command,env):
            completed=isolated_stage(command,env)
            output=Path(command[command.index('--output')+1]);result=json.loads(output.read_text())
            result['history']['history_sha256']='0'*64;output.write_text(json.dumps(result))
            return completed
        with patch('wallex_lab.legacy_technical.run_stage',side_effect=changed_history):
            try: forecast(request,request_path)
            except ValueError: pass
            else: raise AssertionError('Two models could use different source history silently')
        with patch('wallex_lab.legacy_technical.run_stage',side_effect=subprocess.CalledProcessError(1,['fixture'])):
            try: forecast(request,request_path)
            except subprocess.CalledProcessError: pass
            else: raise AssertionError('A missing foundation model silently became rule-only output')
        child=MagicMock();child.pid=123;child.communicate.side_effect=[subprocess.TimeoutExpired(['fixture'],180),('','')]
        with patch('wallex_lab.legacy_technical.subprocess.Popen') as create,patch('wallex_lab.legacy_technical.subprocess.run') as terminate:
            create.return_value.__enter__.return_value=child
            try: run_stage(['fixture'],{})
            except subprocess.TimeoutExpired: pass
            else: raise AssertionError('Timed-out model stage was accepted')
            assert 'start_new_session' not in create.call_args.kwargs
            if sys.platform=='win32': assert terminate.call_args.args[0]==['taskkill','/PID','123','/T','/F']
            else: child.kill.assert_called_once()
    print('legacy technical: strict completed history, genuine-stage composition contract, isolated single-thread workers, all ten retained rules, and failed-model rejection OK')


if __name__=='__main__': check()
