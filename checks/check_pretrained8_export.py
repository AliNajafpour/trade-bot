"""Check the exported eight-asset bundle and optional original-model equivalence."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from scripts.export_pretrained8 import CHECKPOINT_SHA256, SYMBOLS, UNUSED
from wallex_lab.barrier_dataset import sha256, history_inputs, barrier_query, write_json
from wallex_lab.barrier_survival import SurvivalPredictor
from wallex_lab.data import load_market


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/barrier-survival/pretrained8')
    parser.add_argument('--source-root', help='Also verify against original NextBarrierModel and observed week contexts')
    parser.add_argument('--output', help='Save full original/export equivalence and observed forecasts')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.model)
    predictor = SurvivalPredictor.load(root)
    metadata = predictor.metadata
    assert metadata['symbols'] == SYMBOLS and len(predictor.models) == 1
    assert metadata['context'] == metadata['minimum_history_minutes'] == 10080
    assert metadata['source_checkpoint_sha256'] == CHECKPOINT_SHA256
    assert metadata['new_training_performed'] is False
    assert metadata['temperature'] == 1.0 and metadata['feature_version'] == 'relative-v2'
    assert set(metadata['compatibility']['removed_unused_tensors']) == UNUSED
    provenance = root/'provenance'
    for filename, field in [('source-protocol.json', 'protocol_sha256'), ('source-fit.json', 'source_fit_sha256'),
                            ('source-calibration.json', 'source_calibration_sha256'), ('source-test.json', 'source_test_sha256')]:
        assert sha256(provenance/filename) == metadata[field]
    print('PASS bundle, saved weights, original symbol order and historical provenance hashes', flush=True)
    if not args.source_root:
        return
    original_root = Path(args.source_root)
    source = original_root/'runs/barrier-vnext-20261023'
    checkpoint = source/'early/pretrained/seed-42/best.pt'
    assert sha256(checkpoint) == CHECKPOINT_SHA256
    identity = json.loads((provenance/'source-identity.json').read_text())
    for filename in ('wallex_lab/barrier_vnext_model.py', 'wallex_lab/barrier_survival.py'):
        assert sha256(original_root/filename) == identity['sources'][filename]
    # Load the original package separately, so both original model classes retain their original code.
    spec = importlib.util.spec_from_file_location('_archived_wallex', original_root/'wallex_lab/__init__.py',
                                                submodule_search_locations=[str(original_root/'wallex_lab')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    original_class = importlib.import_module('_archived_wallex.barrier_vnext_model').NextBarrierModel
    original = original_class(**identity['config']).eval()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    original.load_state_dict(weights, strict=True)
    exported = predictor.models[0]
    assert set(weights)-set(exported.state_dict()) == UNUSED
    assert all(torch.equal(value, weights[key]) for key, value in exported.state_dict().items())
    rows = []
    for symbol, venue in [('XRPUSDT', 'mexc'), ('DOGEUSDT', 'kucoin'), ('BNBUSDT', 'kucoin')]:
        path = original_root/f'runs/ten-assets-proxy-review/{venue}/{symbol}/{venue}-cache.csv'
        history = load_market(path, 60).iloc[-10080:].copy()
        assert len(history) == 10080
        decision = pd.Timestamp(history.timestamp.iloc[-1])+pd.Timedelta(minutes=1)
        entry = float(history.close.iloc[-1])
        sequence, sigma, reference = history_inputs(history, version='relative-v2')
        query = barrier_query(entry, entry*1.02, entry*.98, sigma, reference, 'long')
        x, q, asset = torch.from_numpy(sequence[None]), torch.from_numpy(query[None]), torch.tensor([SYMBOLS.index(symbol)])
        with torch.inference_mode():
            first, second = original(x, q, asset), exported(x, q, asset)
        assert torch.equal(first, second), (symbol, float((first-second).abs().max()))
        forecast = predictor.predict(history, symbol, decision, entry, tp_fraction=.02, sl_fraction=.02)
        probability = np.array([[row[key] for key in ('p_tp_first', 'p_sl_first', 'p_no_touch')] for row in forecast['forecasts']])
        assert np.isfinite(probability).all() and (probability >= 0).all() and np.allclose(probability.sum(1), 1, atol=2e-6)
        assert (np.diff(probability[:, :2], axis=0) >= -2e-6).all() and (np.diff(probability[:, 2]) <= 2e-6).all()
        rows.append(dict(symbol=symbol, observed_history_path=str(path), observed_history_sha256=sha256(path), venue=venue,
                         original_export_max_abs_logit_difference=float((first-second).abs().max()),
                         cross_venue_calibration='Unmeasured: trained on Binance, observed context from '+venue,
                         forecast=forecast))
        print('PASS exact original/export '+symbol+' '+json.dumps(forecast['forecasts'][-1]), flush=True)
    result = dict(source_checkpoint_sha256=CHECKPOINT_SHA256, exported_member_sha256=metadata['members'][0]['sha256'],
                  used_tensors_equal=True, new_training_performed=False, actual_observed_histories=True, symbols=rows)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)


if __name__ == '__main__':
    main()
