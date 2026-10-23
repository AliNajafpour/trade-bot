"""Initialize the actual bundled paper runner offline. No provider, quote or exchange calls."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wallex_lab.survival_news import Runner

with TemporaryDirectory() as tmp:
    config=json.loads((ROOT/'config/weekly-paper.example.json').read_text(encoding='utf-8'))
    config.update(run_dir=str(Path(tmp)/'account'),source=str(Path(tmp)/'source'),
                  capital_seed=str(ROOT/config['capital_seed']),model=str(ROOT/config['model']))
    config['model_metadata_sha256']=hashlib.sha256((Path(config['model'])/'model.json').read_bytes()).hexdigest()
    runner=Runner(config)
    runner.publish()
    state=json.loads((Path(config['run_dir'])/'status.json').read_text(encoding='utf-8'))
    assert state['paper_only'] is True and state['account']['cash']['USDT']==100
    assert not state['account']['fills'] and not state['account']['positions']
    assert len(runner.predictor.models)==3
    assert runner.predictor.metadata['minimum_history_minutes']==10080
    assert state['experiment']['preset']['timed_exit'] is False
    assert state['reports']['market'] is None and state['reports']['macro'] is None
print('bundled runner: real three-member model loaded, paper account initialized and dashboard state published offline')
