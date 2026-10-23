"""Fine-tune pinned Kronos token autoregression; frozen tokenizer, no trading policy.

Adapted from upstream 67b630e finetune/{dataset,train_predictor,config}.py.
Adaptations: common-calendar purged splits, forecast-only CE, explicit s1 teacher
forcing, fixed AdamW LR and gradient accumulation for a 6 GB laptop. TEST never
chooses an epoch. Pretrained contamination is unknown; no live-model promotion.
"""
import contextlib
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch

from .features import _validated_frame
from .kronos import _load_encoder, _sha256, _verify_assets
from .locking import atomic_replace

UPSTREAM = 'https://github.com/shiyu-coder/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/finetune/'


def calendar_splits(frames):
    """Shared clock prevents later training in one asset leaking into another's VAL."""
    validated = [_validated_frame(frame) for frame in frames.values()]
    times = [item[1] for item in validated]
    if not times:
        raise ValueError('At least one market is required')
    begin = max(t[0] for t in times)
    cadence = pd.Timedelta(seconds=min(item[3] for item in validated))
    end = min(item[1][-1] + pd.Timedelta(seconds=item[3]) for item in validated)
    bars = int((end-begin) / cadence)
    if bars < 10:
        raise ValueError('Insufficient common market history')
    return [begin, begin + cadence*int(bars*.7), begin + cadence*int(bars*.85), end]


class WindowDataset(torch.utils.data.Dataset):
    """Each context AND target belongs wholly to one split and contiguous block."""
    def __init__(self, frames, begin, end, context, prediction, stride, limit=0):
        predictions = prediction.values() if isinstance(prediction, dict) else [prediction]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (context, stride, *predictions)):
            raise ValueError('context, prediction, stride must be positive integers')
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError('limit must be a nonnegative integer')
        self.context, self.prediction = context, prediction
        self.values, self.stamps, self.windows = {}, {}, []
        self.times, self.intervals, self.predictions = {}, {}, {}
        for symbol, frame in sorted(frames.items()):
            data, times, gaps, interval = _validated_frame(frame)
            if interval not in (300, 900, 3600) or (times.as_unit('ns').asi8 % int(interval*1e9)).any():
                raise ValueError('Only aligned M5, M15 or H1 candles are supported')
            self.intervals[symbol] = pd.Timedelta(seconds=interval)
            self.predictions[symbol] = prediction[symbol] if isinstance(prediction, dict) else prediction
            width = context + self.predictions[symbol]
            values = data.to_numpy(dtype=np.float64)
            self.values[symbol] = np.column_stack((values, values[:, :4].mean(axis=1)*values[:, 4]))
            self.stamps[symbol] = np.column_stack((times.minute, times.hour, times.weekday, times.day, times.month)).astype(np.float32)
            self.times[symbol] = times
            cumulative = np.cumsum(gaps)
            starts = np.arange(0, len(frame)-width+1, stride)
            for start in starts:
                last = start + width - 1
                if times[start] >= begin and times[last] + self.intervals[symbol] <= end and cumulative[last] == cumulative[start]:
                    self.windows.append((symbol, int(start)))
        if limit and len(self.windows) > limit:
            # Fixed evenly spaced subset, never a performance-based sample.
            self.windows = [self.windows[i] for i in np.linspace(0, len(self.windows)-1, limit, dtype=int)]
        if not self.windows:
            raise ValueError(f'No complete context+target windows in split {begin} to {end}')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        symbol, start = self.windows[index]
        stop = start + self.context + self.predictions[symbol]
        values = self.values[symbol][start:stop]
        prefix = values[:self.context]
        normalized = np.clip((values-prefix.mean(axis=0))/(prefix.std(axis=0)+1e-5), -5, 5).astype(np.float32)
        return normalized, self.stamps[symbol][start:stop]

    def receipt(self):
        return [dict(symbol=symbol, start_index=start, context_start=self.times[symbol][start].isoformat(),
            forecast_start=self.times[symbol][start+self.context].isoformat(),
            prediction_bars=self.predictions[symbol], interval_seconds=self.intervals[symbol].total_seconds(),
            target_end_exclusive=(self.times[symbol][start+self.context+self.predictions[symbol]-1]+self.intervals[symbol]).isoformat())
            for symbol, start in self.windows]


def causal_mode(model, training):
    model.train(training)
    # Pinned module.py:387 disables causal CROSS attention in eval. Set only the
    # attention's flag; dropout child modules stay in eval. Its own SDPA dropout
    # is zero in Kronos.dep_layer. Never mutate the vendored source.
    attention = model.dep_layer.cross_attn
    if attention.attn_dropout_p != 0:
        raise ValueError('Causal eval adapter requires zero dependency attention dropout')
    attention.training = True


def token_loss(model, tokenizer, values, stamps, context):
    with torch.no_grad():
        s1, s2 = tokenizer.encode(values.float(), half=True)
    logits = model(s1[:, :-1], s2[:, :-1], stamp=stamps[:, :-1],
        use_teacher_forcing=True, s1_targets=s1[:, 1:])
    # Output position context-1 predicts the first future token, at raw context.
    loss, _, _ = model.head.compute_loss(logits[0][:, context-1:], logits[1][:, context-1:],
        s1[:, context:], s2[:, context:])
    return loss


def run_epoch(model, tokenizer, loader, context, device, optimizer=None, accumulation=8, amp=False):
    training = optimizer is not None
    causal_mode(model, training)
    tokenizer.eval().requires_grad_(False)
    dtype = torch.bfloat16 if str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    iterator = iter(loader)
    total, count, steps = 0., 0, 0
    gradient_context = contextlib.nullcontext if training else torch.no_grad
    with gradient_context():
        while group := list(itertools.islice(iterator, accumulation if training else 1)):
            group_size = sum(len(values) for values, _ in group)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for values, stamps in group:
                values, stamps = values.to(device), stamps.to(device)
                # Tokenizer remains float32 even when the predictor uses AMP.
                with torch.no_grad():
                    ids = tokenizer.encode(values.float(), half=True)
                with torch.autocast(device_type='cuda', dtype=dtype, enabled=amp):
                    logits = model(ids[0][:, :-1], ids[1][:, :-1], stamp=stamps[:, :-1],
                        use_teacher_forcing=True, s1_targets=ids[0][:, 1:])
                    loss, _, _ = model.head.compute_loss(logits[0][:, context-1:], logits[1][:, context-1:],
                        ids[0][:, context:], ids[1][:, context:])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite token CE')
                size = len(values)
                total += float(loss.detach()) * size
                count += size
                if training:
                    scaler.scale(loss * size / group_size).backward()
            if training:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                steps += 1
    if not count:
        raise ValueError('Empty epoch')
    return dict(token_ce=total/count, windows=count, optimizer_steps=steps)


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    atomic_replace(temporary,path)


class LengthBatches:
    """Keep different forecast cadences in separate batches; shuffle only TRAIN."""
    def __init__(self, dataset, size, shuffle, generator):
        self.dataset, self.size, self.shuffle, self.generator = dataset, size, shuffle, generator

    def __iter__(self):
        groups = {}
        for index, (symbol, _) in enumerate(self.dataset.windows):
            groups.setdefault(self.dataset.predictions[symbol], []).append(index)
        batches = []
        for indices in groups.values():
            if self.shuffle:
                indices = [indices[i] for i in torch.randperm(len(indices), generator=self.generator).tolist()]
            batches.extend(indices[i:i+self.size] for i in range(0, len(indices), self.size))
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=self.generator).tolist()]
        yield from batches


def train(paths, output, *, context=128, prediction=12, stride=16, eval_stride=128,
          epochs=3, batch_size=2, accumulation=8, patience=1, learning_rate=4e-5,
          seed=20261021, device='cuda:0', train_limit=0, eval_limit=0, smoke=False,
          prediction_hours=None, gpu_lock=None, status_callback=None):
    """Fresh output only. Validation selects epoch; TEST is diagnostic, never a gate."""
    for name, value in dict(epochs=epochs, batch_size=batch_size, accumulation=accumulation, patience=patience).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if epochs > 10 or context not in (128, 256) or not 1 <= prediction <= 128 or context+prediction > 512:
        raise ValueError('Require max10 epochs, context128/256, prediction1..128, total<=512')
    if prediction_hours is not None and (isinstance(prediction_hours, bool) or not 0 < prediction_hours <= 24):
        raise ValueError('prediction_hours must be in (0,24]')
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 4e-5:
        raise ValueError('learning_rate must be in (0, 4e-5]')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite training output: {output}')
    sources = [Path(path).resolve() for path in paths]
    if len(set(sources)) != len(sources) or len({p.stem for p in sources}) != len(sources):
        raise ValueError('Training source paths and market stems must be unique')
    frames, inputs = {}, {}
    for path in sources:
        digest = _sha256(path)
        frames[path.stem] = pd.read_csv(path)
        _validated_frame(frames[path.stem])
        if _sha256(path) != digest:
            raise RuntimeError(f'Training source changed while reading: {path}')
        inputs[str(path)] = digest
    bounds = calendar_splits(frames)
    if not smoke and bounds[-1]-bounds[0] < pd.Timedelta(days=80):
        raise ValueError('Production fine-tuning requires >=80 days common history; short data needs --smoke')
    lengths = {symbol: int(prediction_hours*3600/_validated_frame(frame)[3]) for symbol,frame in frames.items()} if prediction_hours else prediction
    if isinstance(lengths,dict) and any(v < 1 or context+v > 512 or v*_validated_frame(frames[s])[3] != prediction_hours*3600 for s,v in lengths.items()):
        raise ValueError('Forecast must divide cadence and context+forecast must be <=512')
    datasets = {name: WindowDataset(frames, bounds[i], bounds[i+1], context, lengths,
        stride if i == 0 else eval_stride, train_limit if i == 0 else eval_limit)
        for i,name in enumerate(('train','validation','test'))}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    from .locking import exclusive
    lock_context = exclusive(gpu_lock, timeout=3600) if gpu_lock and str(device).startswith('cuda') else contextlib.nullcontext()
    if status_callback:
        status_callback('queued', message='CPU data preparation complete; waiting for GPU lock')
    with lock_context:
        print('KRONOS_GPU_LOCK_ACQUIRED' if gpu_lock and str(device).startswith('cuda') else 'KRONOS_CPU_RUN', flush=True)
        try:
            if status_callback:
                status_callback('training',message='GPU lock held; loading verified weights and scoring pretrained baseline')
            model_dir, tokenizer_dir, provenance = _verify_assets(dict(kronos_model='small'))
            model, tokenizer = _load_encoder(model_dir, tokenizer_dir, device)
            model.requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(.9,.95), weight_decay=.1)
            generator = torch.Generator().manual_seed(seed)
            loaders = {name: torch.utils.data.DataLoader(dataset,
                batch_sampler=LengthBatches(dataset,batch_size,name=='train',generator), num_workers=0)
                for name,dataset in datasets.items()}
            output.mkdir(parents=True)
            best_dir = output/'best_model'
            best_dir.mkdir()
            shutil.copyfile(model_dir/'config.json', best_dir/'config.json')
            config = dict(context=context,prediction=prediction,stride=stride,eval_stride=eval_stride,epochs=epochs,
                batch_size=batch_size,accumulation=accumulation,patience=patience,learning_rate=learning_rate,
                seed=seed,device=device,train_limit=train_limit,eval_limit=eval_limit,smoke=smoke,
                prediction_hours=prediction_hours,prediction_bars=lengths)
            source_paths = [Path(__file__), Path(__file__).resolve().parents[1]/'scripts/finetune_kronos.py',
                Path(__file__).resolve().parents[1]/'checks/check_finetune_kronos.py']
            data_receipts = {str(p): _sha256(p) for p in {source.parent.parent/'manifest.json' for source in sources} if p.is_file()}
            metadata = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                config=config, inputs=inputs, data_receipts=data_receipts, pretrained=provenance,
                sources={str(p.resolve()): _sha256(p) for p in source_paths},
                upstream=[UPSTREAM+name for name in ('dataset.py','train_predictor.py','config.py')],
                adaptations=['forecast target tokens only (upstream trains all shifted tokens)',
                    'explicit s1 teacher forcing (upstream default samples s1)', 'fixed AdamW LR; no upstream OneCycle scheduler',
                    'common 70/15/15 calendar splits; entire context+target purged; no gap crossing',
                    'gradient accumulation; float32 frozen tokenizer; mixed-precision predictor on CUDA',
                    'eval dependency cross attention causal flag restored without enabling dropout'],
                splits=dict(boundaries=[t.isoformat() for t in bounds],
                    windows={name: dataset.receipt() for name,dataset in datasets.items()}),
                selection='lowest VALIDATION token CE including unchanged pretrained epoch0; TEST never selects epochs',
                promotion='none; token CE does not establish profitable or physically valid price forecasts',
                caveats=['Pretrained training overlap is unknown; heldout fine-tune data is not proof of independent pretraining.',
                    'Amount is approximate mean OHLC times exchange volume.', 'Teacher-forced CE differs from free-running path error.'],
                epochs=[])
            receipt = output/'training.json'
            _write_json(receipt, metadata)
            from safetensors.torch import save_file, load_file
            def save_weights():
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(best_dir/'model.safetensors'))
            started = time.perf_counter()
            amp = str(device).startswith('cuda')
            if amp:
                torch.cuda.reset_peak_memory_stats(device)
            try:
                baseline_val = run_epoch(model,tokenizer,loaders['validation'],context,device,amp=amp)
                baseline_test = run_epoch(model,tokenizer,loaders['test'],context,device,amp=amp)
                metadata['baseline'] = dict(validation=baseline_val,test=baseline_test)
                best_loss, best_epoch, misses = baseline_val['token_ce'], 0, 0
                save_weights()
                _write_json(receipt,metadata)
                for epoch in range(1,epochs+1):
                    if status_callback:
                        status_callback('training',epoch=epoch,maximum_epochs=epochs,best_epoch=best_epoch,best_validation_token_ce=best_loss)
                    epoch_started = time.perf_counter()
                    training = run_epoch(model,tokenizer,loaders['train'],context,device,optimizer,accumulation,amp)
                    validation = run_epoch(model,tokenizer,loaders['validation'],context,device,amp=amp)
                    improved = validation['token_ce'] < best_loss
                    if improved:
                        best_loss,best_epoch,misses = validation['token_ce'],epoch,0
                        save_weights()
                    else:
                        misses += 1
                    checkpoint = output/f'epoch-{epoch}.pt'
                    torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                        epoch=epoch,seed=seed,config=config,input_hashes=inputs,split_boundaries=metadata['splits']['boundaries'],
                        random_state=random.getstate(),numpy_state=np.random.get_state(),torch_rng_state=torch.get_rng_state(),
                        loader_generator_state=generator.get_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if amp else None),checkpoint)
                    entry = dict(epoch=epoch,train=training,validation=validation,selected=improved,
                        elapsed_seconds=time.perf_counter()-epoch_started,checkpoint=str(checkpoint),checkpoint_sha256=_sha256(checkpoint))
                    metadata['epochs'].append(entry)
                    metadata.update(best_epoch=best_epoch,best_validation_token_ce=best_loss)
                    _write_json(receipt,metadata)
                    print(json.dumps(entry),flush=True)
                    if misses >= patience:
                        break
                model.load_state_dict(load_file(str(best_dir/'model.safetensors'),device=device),strict=True)
                if status_callback:
                    status_callback('evaluating',best_epoch=best_epoch,best_validation_token_ce=best_loss)
                final_val = run_epoch(model,tokenizer,loaders['validation'],context,device,amp=amp)
                final_test = run_epoch(model,tokenizer,loaders['test'],context,device,amp=amp)
                if not math.isclose(final_val['token_ce'],best_loss,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError('Selected weights failed validation loss replay')
                if any(_sha256(path) != digest for path,digest in inputs.items()):
                    raise RuntimeError('Training input changed during run')
                metadata.update(status='complete',best_epoch=best_epoch,best_validation_token_ce=best_loss,
                    final=dict(validation=final_val,test=final_test),selected_weights=str(best_dir/'model.safetensors'),
                    selected_weights_sha256=_sha256(best_dir/'model.safetensors'),elapsed_seconds=time.perf_counter()-started,
                    completed_utc=datetime.now(timezone.utc).isoformat(), selected_finetuned_weights=best_epoch > 0,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if amp else 0,
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if amp else 0)
            except BaseException as exc:
                metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-started)
                _write_json(receipt,metadata)
                raise
            _write_json(receipt,metadata)
            return metadata
        finally:
            if gpu_lock and str(device).startswith('cuda'):
                print('KRONOS_GPU_LOCK_RELEASED', flush=True)
