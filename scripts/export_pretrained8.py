"""Package an existing eight-asset checkpoint for inference. Performs no training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from wallex_lab.barrier_dataset import sha256, write_json
from wallex_lab.barrier_survival import MultiScaleSurvival, SurvivalPredictor

UNUSED = {'mask_token', 'reconstruction.weight', 'reconstruction.bias'}
CHECKPOINT_SHA256 = 'fdc447607006f8c72b25df36e94b2f00bcd85eb367dfd4758cbb60505d6c5042'
SYMBOLS = ['ADAUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETHUSDT', 'LINKUSDT', 'SOLUSDT', 'XRPUSDT']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, help='Original workspace containing the archived run')
    parser.add_argument('--output', default='models/barrier-survival/pretrained8')
    args = parser.parse_args()
    source = Path(args.source_root)/'runs/barrier-vnext-20261023'
    variant = source/'early/pretrained'
    checkpoint = variant/'seed-42/best.pt'
    fit = json.loads((variant/'seed-42/fit.json').read_text())
    identity = json.loads((variant/'seed-42/identity.json').read_text())
    calibration = json.loads((variant/'calibration.json').read_text())
    test = json.loads((variant/'test.json').read_text())
    frozen = json.loads((source/'choices-frozen.json').read_text())
    if sha256(checkpoint) != CHECKPOINT_SHA256 or fit['checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint differs from the selected, pinned historical weights')
    if fit['identity'] != identity or fit['config'] != identity['config']:
        raise ValueError('Historical fit identity mismatch')
    if (identity['data']['metadata']['symbols'] != SYMBOLS or
            any(identity['config'][flag] for flag in ('activity', 'age', 'level_attention', 'local'))):
        raise ValueError('Only the exact eight-asset plain survival variant is compatible')
    if (sha256(source/'protocol.json') != frozen['protocol_sha256'] or
            sha256(source/'choices-frozen.json') != calibration['freeze_sha256'] or
            test['receipt']['freeze_sha256'] != calibration['freeze_sha256'] or
            sha256(variant/'calibration.json') != test['receipt']['calibration_sha256']):
        raise ValueError('Historical calibration/test freeze receipt mismatch')
    for path in (checkpoint, variant/'seed-42/fit.json'):
        if frozen['files'].get(str(path)) != sha256(path):
            raise ValueError('Historical frozen artifact changed: '+str(path))
    config = {key: fit['config'][key] for key in ('symbols', 'context', 'width', 'layers')}
    model = MultiScaleSurvival(**config, hierarchical=True).eval()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if set(weights)-set(model.state_dict()) != UNUSED or set(model.state_dict())-set(weights):
        raise ValueError('Unexpected missing or unused checkpoint tensors')
    used = {key: weights[key] for key in model.state_dict()}
    model.load_state_dict(used, strict=True)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in used.items())
    root = Path(args.output)
    if root.exists():
        raise FileExistsError('Preserve the previous exported artifact')
    source_data = identity['data']['metadata']
    fold = next(fold for fold in source_data['folds'] if fold['name'] == 'early')
    metadata = dict(symbols=SYMBOLS, **{key: config[key] for key in ('context', 'width', 'layers')},
                    minimum_history_minutes=config['context'], feature_version='relative-v2',
                    temperature=calibration['temperature'], geometry_support=source_data['geometry_support'],
                    semantics='price_touch', protocol_sha256=sha256(source/'protocol.json'),
                    dataset_manifest_sha256=identity['data']['manifest_sha256'],
                    fold=dict(fold, fit=fit['fit_origins'], selection=fit['selection_origins'],
                              calibration=calibration['origins'], test=test['calibrated']['origins']),
                    model_family='Existing eight-asset early-fold pretrained variant, seed 42',
                    source_checkpoint_sha256=CHECKPOINT_SHA256, source_fit_sha256=sha256(variant/'seed-42/fit.json'),
                    source_calibration_sha256=sha256(variant/'calibration.json'), source_test_sha256=sha256(variant/'test.json'),
                    export_created_utc=datetime.now(timezone.utc).isoformat(), new_training_performed=False,
                    compatibility=dict(removed_unused_tensors=sorted(UNUSED), symbol_remapping=False,
                                       reason='All optional flags false; original forward directly calls MultiScaleSurvival on the same ten channels.'),
                    training_history=dict(selected_epoch=fit['selected_epoch'], stopped_for_deadline=fit['stopped_for_deadline'],
                                          pretraining=fit['pretraining']),
                    limitations=['One member, not the three-member Weekly late ensemble.',
                                 'Historical metrics pool eight Binance assets and 128 barrier queries, not this fixed 2% policy or a per-asset validation.',
                                 'Deployment uses other venues; cross-venue probability calibration has not been measured.',
                                 'No prospective profitability or execution performance claim.'])
    SurvivalPredictor.save(root, [model], metadata)
    provenance = root/'provenance'
    provenance.mkdir()
    for name, original in [('source-protocol.json', source/'protocol.json'), ('source-fit.json', variant/'seed-42/fit.json'),
                           ('source-identity.json', variant/'seed-42/identity.json'), ('source-calibration.json', variant/'calibration.json'),
                           ('source-test.json', variant/'test.json'), ('source-choices-frozen.json', source/'choices-frozen.json')]:
        shutil.copy2(original, provenance/name)
    summary = dict(source_checkpoint_sha256=CHECKPOINT_SHA256, exported_member_sha256=sha256(root/'member-0.pt'),
                   used_tensor_count=len(used), omitted_unused_tensors=sorted(UNUSED),
                   tensor_values_changed=False, new_training_performed=False,
                   historical_test=dict(origins=test['calibrated']['origins'], origin_query_visits=test['calibrated']['origin_query_visits'],
                                        partial_label_nll=test['calibrated']['partial_label_nll'],
                                        by_horizon=[{key: value for key, value in row.items() if key != 'reliability'} for row in test['calibrated']['by_horizon']]))
    write_json(root/'export-summary.json', summary)
    print(json.dumps(dict(output=str(root.resolve()), symbols=SYMBOLS, **summary)), flush=True)


if __name__ == '__main__':
    main()
