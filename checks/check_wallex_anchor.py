"""Replay delayed native candle publication. No keys, network or exchange writes."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wallex_lab.survival_paper import Runner


def main():
    decision = pd.Timestamp('2026-10-23T21:19:00Z')
    target = int((decision-pd.Timedelta(minutes=1)).timestamp())
    def payload(times):
        return json.dumps(dict(s='ok', t=times, o=[100]*len(times), h=[101]*len(times),
                               l=[99]*len(times), c=[100]*len(times), v=[1]*len(times))).encode()
    pending = (payload([target-60]), {'status':200})
    published = (payload([target-60,target]), {'status':200})
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        runner=object.__new__(Runner)
        runner.root=root;runner.config={'source':str(root/'source')}
        with patch('wallex_lab.survival_paper.fetch_public',side_effect=[pending,pending,published]) as fetch, \
             patch('wallex_lab.survival_paper.time.sleep') as sleep:
            anchor,path,receipts=runner.wallex_anchor('BTCUSDT',decision)
        assert len(anchor)==1 and anchor.timestamp.iloc[0]==decision-pd.Timedelta(minutes=1)
        assert fetch.call_count==3 and [call.args[0] for call in sleep.call_args_list]==[8,8]
        assert len(receipts)==3
        with patch('wallex_lab.survival_paper.fetch_public',side_effect=AssertionError('Valid cache fetched again')):
            assert len(runner.wallex_anchor('BTCUSDT',decision)[0])==1
        with patch('wallex_lab.survival_paper.fetch_public',return_value=pending) as fetch, \
             patch('wallex_lab.survival_paper.time.sleep'):
            try:runner.wallex_anchor('ETHUSDT',decision)
            except ValueError as error:assert 'still being published' in str(error)
            else:raise AssertionError('An unpublished candle was fabricated')
        assert fetch.call_count==3, 'Publication retry is unbounded'
    print('PASS: delayed publication retries same minute, valid cache reused, absent candles never fabricated')


if __name__=='__main__':main()
