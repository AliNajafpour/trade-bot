"""Frozen genuine Kronos hidden states with a train-only PCA/Ridge return head.

No forecast interpolation, momentum substitute, downloads or missing-weight
fallback. Vendored upstream code and license are recorded in vendor_kronos/.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .features import _validated_frame, forward_returns

TRADE_BOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'small': dict(directory='kronos-small-901c26c', revision='901c26c1332695a2a8f243eb2f37243a37bea320',
        config='5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396',
        weights='b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020'),
    'base': dict(directory='kronos-base-2b55474', revision='2b554741eca47781b64468546e77fef3e85130e6',
        config='77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2',
        weights='abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83'),
}
TOKENIZER = dict(directory='kronos-tokenizer-base-0e01173', revision='0e0117387f39004a9016484a186a908917e22426',
    config='2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2',
    weights='59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _sample_indices(frame, context, stride, max_samples):
    """Fixed-clock scheduled observations with no history crossing a gap."""
    context = _positive_int(context, 'kronos_context')
    stride = _positive_int(stride, 'kronos_stride')
    max_samples = _positive_int(max_samples, 'kronos_max_samples')
    _, _, gaps, _ = _validated_frame(frame)
    if context < 2 or context > 512:
        raise ValueError('kronos_context must be between 2 and 512')
    indices = np.arange(context - 1, len(frame), stride, dtype=np.int64)
    cumulative = np.cumsum(gaps)
    indices = indices[(cumulative[indices] - cumulative[indices - context + 1]) == 0]
    if len(indices) > max_samples:
        raise ValueError(f'{len(indices)} Kronos windows exceeds kronos_max_samples={max_samples}; increase kronos_stride or budget explicitly')
    if not len(indices):
        raise ValueError('No complete contiguous Kronos history windows')
    return indices


def _window_batch(frame, indices, context):
    """Normalize only bars in [decision-context+1, decision], following upstream.

    This is a causal per-window transform, not a fitted whole-dataset scaler.
    Amount is an explicitly approximate OHLC mean times volume, as upstream.
    """
    data, times, _, _ = _validated_frame(frame)
    values = data.to_numpy(dtype=np.float64)
    values = np.column_stack((values, values[:, :4].mean(axis=1) * values[:, 4]))
    rows = np.asarray(indices)[:, None] - np.arange(context - 1, -1, -1)[None, :]
    if (rows < 0).any() or (rows >= len(frame)).any():
        raise ValueError('Kronos window outside completed candle history')
    windows = values[rows]
    normalized = np.clip((windows - windows.mean(axis=1, keepdims=True)) /
        (windows.std(axis=1, keepdims=True) + 1e-5), -5, 5).astype(np.float32)
    stamps = np.column_stack((times.minute, times.hour, times.weekday, times.day, times.month))
    return normalized, stamps[rows].astype(np.float32)


def _encode_windows(frame, indices, context, batch_size, device, model, tokenizer):
    """Adapt upstream encode/decode_s1 and keep the final normalized hidden state."""
    import torch
    batch_size = _positive_int(batch_size, 'kronos_batch_size')
    model.eval().requires_grad_(False)
    tokenizer.eval().requires_grad_(False)
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            values, stamps = _window_batch(frame, indices[start:start + batch_size], context)
            x = torch.as_tensor(values, dtype=torch.float32, device=device)
            stamp = torch.as_tensor(stamps, dtype=torch.float32, device=device)
            ids = tokenizer.encode(x, half=True)
            _, hidden = model.decode_s1(ids[0], ids[1], stamp=stamp)
            output.append(hidden[:, -1, :].float().cpu().numpy())
    latent = np.concatenate(output)
    if latent.shape != (len(indices), model.d_model) or not np.isfinite(latent).all():
        raise RuntimeError('Kronos encoder did not return finite final hidden states')
    return latent


def _verify_assets(config):
    family = config.get('kronos_model', 'small')
    if family not in ASSETS:
        raise ValueError('kronos_model must be small or base')
    model_info = ASSETS[family]
    model_dir = Path(config.get('kronos_model_dir', TRADE_BOT / 'models' / model_info['directory']))
    tokenizer_dir = Path(config.get('kronos_tokenizer_dir', TRADE_BOT / 'models' / TOKENIZER['directory']))
    files = {}
    for directory, info in ((model_dir, model_info), (tokenizer_dir, TOKENIZER)):
        for name, expected in (('config.json', info['config']), ('model.safetensors', info['weights'])):
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(f'Local Kronos asset is missing: {path}. Production has no synthetic fallback.')
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f'Kronos asset hash differs from pinned download receipt: {path}')
            files[str(path)] = actual
    vendor = Path(__file__).parent / 'vendor_kronos'
    origins = json.loads((vendor / 'ORIGINS.json').read_text(encoding='utf-8'))
    for name, info in origins['files'].items():
        path = vendor / name
        actual = _sha256(path)
        if actual != info['copied_sha256']:
            raise ValueError(f'Vendored Kronos source hash changed: {path}')
        files[str(path)] = actual
    return model_dir, tokenizer_dir, dict(family=family, model_revision=model_info['revision'],
        tokenizer_revision=TOKENIZER['revision'], verified_files=files, source_reuse=origins)


def _load_encoder(model_dir, tokenizer_dir, device):
    try:
        import torch
        from safetensors.torch import load_file
        from .vendor_kronos.kronos import Kronos, KronosTokenizer
    except ImportError as exc:
        raise RuntimeError(f'Kronos requires the existing torch/einops/safetensors/huggingface_hub runtime: {exc}') from exc
    torch.set_num_threads(2)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'Kronos device {device} requested but CUDA is unavailable')
    model = Kronos(**json.loads((model_dir / 'config.json').read_text()))
    tokenizer = KronosTokenizer(**json.loads((tokenizer_dir / 'config.json').read_text()))
    model.load_state_dict(load_file(str(model_dir / 'model.safetensors')), strict=True)
    tokenizer.load_state_dict(load_file(str(tokenizer_dir / 'model.safetensors')), strict=True)
    return model.to(device).eval().requires_grad_(False), tokenizer.to(device).eval().requires_grad_(False)


def _held_targets(frame, score, threshold, exposure, start):
    """Hold allocation until the next observed score; gaps clear the allocation."""
    _, _, gaps, _ = _validated_frame(frame)
    result = np.zeros(len(frame), dtype=float)
    held = 0.0
    for i in range(start, len(frame)):
        if gaps[i]:
            held = 0.0
        if np.isfinite(score[i]):
            held = exposure if score[i] > threshold else 0.0
        result[i] = held
    return result


def _learn_head(frame, indices, latent, train_end, val_end, config, run_dir):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from .engine import backtest

    n = len(frame)
    if not (0 < train_end < val_end < n):
        raise ValueError('Kronos requires chronological train_end < val_end < number of candles')
    horizon = _positive_int(config.get('horizon', 3), 'horizon')
    labels = forward_returns(frame, horizon).to_numpy()
    train = (indices + horizon < train_end) & np.isfinite(labels[indices])
    validation = (indices >= train_end) & (indices + horizon < val_end) & np.isfinite(labels[indices])
    if train.sum() < 12 or validation.sum() < 4:
        raise ValueError('Kronos needs at least 12 purged train and 4 purged validation windows')
    exposure = float(config.get('max_exposure', 1.0))
    if not np.isfinite(exposure) or not 0 <= exposure <= 1:
        raise ValueError('max_exposure must be finite in [0, 1]')
    components = min(_positive_int(config.get('kronos_pca_components', 16), 'kronos_pca_components'),
        latent.shape[1], int(train.sum()) - 1)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # ponytail: a small fixed Ridge/threshold grid bounds validation search; expand only with more history.
    trials = []
    best = None
    with threadpool_limits(limits=2):
        scaler = StandardScaler().fit(latent[train])
        scaled = scaler.transform(latent)
        pca = PCA(n_components=components, svd_solver='full').fit(scaled[train])
        projected = pca.transform(scaled)
        roundtrip_cost = 2 * (float(config.get('fee_bps', 10)) + float(config.get('slippage_bps', 5))) / 10000
        for alpha in (1.0, 10.0, 100.0):
            head = Ridge(alpha=alpha).fit(projected[train], labels[indices[train]])
            predicted = head.predict(projected)
            observed_score = np.full(n, np.nan)
            observed_score[indices[validation]] = predicted[validation]
            for threshold in sorted(set((0.0, roundtrip_cost, roundtrip_cost * 2))):
                targets = _held_targets(frame.iloc[:val_end], observed_score[:val_end], threshold, exposure, train_end)
                metrics = backtest(frame.iloc[:val_end], targets, config, start=train_end)['metrics']
                value = float(metrics['total_return'])
                trials.append(dict(alpha=alpha, threshold=threshold, validation_total_return=value))
                if best is None or value > best[0]:
                    best = value, alpha, threshold, head, predicted
    value, alpha, threshold, head, predicted = best
    score = np.full(n, np.nan)
    score[indices] = predicted
    targets = _held_targets(frame, score, threshold, exposure, val_end)
    np.savez_compressed(run_dir / 'head.npz', scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        pca_mean=pca.mean_, pca_components=pca.components_, coefficient=head.coef_, intercept=np.asarray(head.intercept_))
    metadata = dict(status='ok', strategy='kronos', train_end=int(train_end), val_end=int(val_end),
        representation='frozen Kronos.decode_s1 final normalized hidden state',
        selected=dict(ridge_alpha=alpha, entry_threshold=threshold, pca_components=components),
        train_indices=indices[train].tolist(), validation_indices=indices[validation].tolist(),
        label_horizon=horizon, boundary_purge_bars=horizon, head_fit='TRAIN only, no validation/test refit',
        scaler_fit='TRAIN latent rows only; input-window normalization uses completed bars in each window',
        selection='next-open validation backtest with configured each-side fees/slippage and end liquidation',
        validation_total_return=value, trials=trials, decisions_start=val_end,
        sparse_observation_policy='score is NaN on unsampled rows; hold target until next observed score; clear after a gap',
        head_artifact=str(run_dir / 'head.npz'), head_sha256=_sha256(run_dir / 'head.npz'),
        caveats=['Pretrained model training dates and overlap with market history are not independently verified; pretrained contamination remains possible.',
            'Head/scaler/PCA have no test-label access; historical pre-test scores are diagnostic in-sample/selection outputs, and targets start at test boundary.',
            'OHLC mean times volume approximates traded amount; volume semantics and real fills are not guaranteed.'])
    (run_dir / 'selection.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return dict(target=targets, score=score, metadata=metadata)


def generate(frame, train_end, val_end, config, run_dir):
    """Generate sparse genuine-model observations and test-period held allocations.

    Controls: kronos_model=small|base; kronos_context=64; kronos_stride=4;
    kronos_max_samples=2048; kronos_batch_size=16; kronos_device=cpu;
    kronos_pca_components=16. Asset directories and cache directory may be moved,
    but pinned weight/config hashes must still match. Budget overruns fail.
    """
    context = _positive_int(config.get('kronos_context', 64), 'kronos_context')
    stride = _positive_int(config.get('kronos_stride', 4), 'kronos_stride')
    batch_size = _positive_int(config.get('kronos_batch_size', 16), 'kronos_batch_size')
    indices = _sample_indices(frame, context, stride, config.get('kronos_max_samples', 2048))
    model_dir, tokenizer_dir, provenance = _verify_assets(config)
    device = str(config.get('kronos_device', config.get('device', 'cpu')))
    data, times, _, _ = _validated_frame(frame)
    digest = hashlib.sha256(json.dumps(dict(context=context, files=provenance['verified_files'],
        adapter=_sha256(__file__)), sort_keys=True).encode())
    for values in (data.to_numpy(dtype=np.float64), times.as_unit('ns').asi8, indices):
        digest.update(np.ascontiguousarray(values).tobytes())
    cache_key = digest.hexdigest()
    cache_dir = Path(config.get('kronos_cache_dir', Path(run_dir) / 'encoder-cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'{cache_key}.npz'
    receipt = cache.with_suffix('.json')
    cache_hit = cache.exists()
    if cache_hit:
        if not receipt.exists() or json.loads(receipt.read_text())['sha256'] != _sha256(cache):
            raise ValueError(f'Kronos encoder cache receipt mismatch: {cache}')
        with np.load(cache, allow_pickle=False) as stored:
            if not np.array_equal(stored['indices'], indices):
                raise ValueError('Kronos cached observation indices differ')
            latent = stored['latent'].copy()
        width = json.loads((model_dir / 'config.json').read_text())['d_model']
        if latent.shape != (len(indices), width) or not np.isfinite(latent).all():
            raise ValueError('Invalid Kronos cached hidden states')
    else:
        model, tokenizer = _load_encoder(model_dir, tokenizer_dir, device)
        latent = _encode_windows(frame, indices, context, batch_size, device, model, tokenizer)
        np.savez_compressed(cache, indices=indices, latent=latent)
        receipt.write_text(json.dumps(dict(cache_key=cache_key, sha256=_sha256(cache),
            context=context, stride=stride, observation_count=len(indices), provenance=provenance), indent=2) + '\n')
        del model, tokenizer
        if device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
    result = _learn_head(frame, indices, latent, train_end, val_end, config, run_dir)
    result['metadata'].update(provenance=provenance, context=context, stride=stride, batch_size=batch_size,
        latent_width=latent.shape[1], observation_count=len(indices), observation_indices=indices.tolist(),
        device=device, cache_key=cache_key, cache_path=str(cache), cache_hit=cache_hit)
    Path(run_dir, 'selection.json').write_text(json.dumps(result['metadata'], indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result
