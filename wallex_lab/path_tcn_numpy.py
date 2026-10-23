"""Inference-only NumPy/SciPy evaluation of the existing seven-layer PathTCN."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import erf
from threadpoolctl import threadpool_limits


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PathTCNNumpy:
    def __init__(self, weights, metadata):
        self.metadata = metadata
        width, channels = metadata['width'], metadata['sequence_shape'][1]
        shapes = {'projection.weight': (width, channels, 1), 'projection.bias': (width,),
                  'head.weight': (45, width), 'head.bias': (45,)}
        for i in range(7):
            shapes.update({f'layers.{i}.weight': (width,width,3), f'layers.{i}.bias': (width,)})
        if metadata['format_version'] != 1 or metadata['model'] != 'causal_tcn_quantiles' or set(weights) != set(shapes):
            raise ValueError('Unsupported NumPy TCN architecture')
        self.weights = {}
        for name, shape in shapes.items():
            value = np.asarray(weights[name])
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid NumPy TCN tensor: '+name)
            self.weights[name] = value

    @classmethod
    def load(cls, path, device='cpu'):
        if device != 'cpu': raise ValueError('NumPy TCN supports CPU inference only')
        path = Path(path)
        metadata = json.loads((path/'model.json').read_text(encoding='utf-8'))
        receipt = json.loads((path/'numpy-export.json').read_text(encoding='utf-8'))
        errors = [receipt['verification'][key] for key in (
            'quantile_max_abs_error','probability_max_abs_error','expected_return_max_abs_error')]
        if not np.isfinite(errors).all() or any(value < 0 or value > 1e-6 for value in errors):
            raise ValueError('Invalid NumPy equivalence metrics')
        if (receipt['format_version'] != 1 or
                file_sha256(path/'model.json') != receipt['metadata_sha256'] or
                file_sha256(path/'weights.pt') != metadata['weights_sha256'] or
                metadata['weights_sha256'] != receipt['original_weights_sha256'] or
                file_sha256(path/'weights.numpy.npz') != receipt['exported_weights_sha256'] or
                receipt['verification']['quantile_max_abs_error'] > 1e-6 or
                receipt['verification']['probability_max_abs_error'] > 1e-6 or
                receipt['verification']['expected_return_max_abs_error'] > 1e-6):
            raise ValueError('NumPy TCN artifact integrity/equivalence check failed')
        with np.load(path/'weights.numpy.npz', allow_pickle=False) as archive:
            return cls(dict(archive), metadata)

    def predict_quantiles(self, x, batch_size=256):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError('Causal sequences must be a finite 3-dimensional array')
        if tuple(x.shape[1:]) != tuple(self.metadata['sequence_shape']):
            raise ValueError('Causal sequence shape differs from TRAIN')
        if type(batch_size) is not int or not 1 <= batch_size <= 1024:
            raise ValueError('Batch size must be1..1024')
        scaler = self.metadata['sequence_scaler']
        clipped = np.clip(x, scaler['lower'], scaler['upper'])
        values = np.ascontiguousarray(np.clip((clipped-scaler['mean'])/scaler['scale'],-10,10),dtype=np.float32)
        values[..., ~np.asarray(scaler['active'],bool)] = 0
        w = self.weights
        parts = []
        with threadpool_limits(limits=2):
            for start in range(0,len(values),batch_size):
                encoded = values[start:start+batch_size] @ w['projection.weight'][:,:,0].T + w['projection.bias']
                length = encoded.shape[1]
                for i in range(7):
                    dilation = 2**i
                    padded = np.pad(encoded,((0,0),(2*dilation,0),(0,0)))
                    kernel = w[f'layers.{i}.weight']
                    projected = padded[:,:length] @ kernel[:,:,0].T
                    projected += padded[:,dilation:dilation+length] @ kernel[:,:,1].T
                    projected += padded[:,2*dilation:2*dilation+length] @ kernel[:,:,2].T
                    projected += w[f'layers.{i}.bias']
                    encoded += projected*np.float32(.5)*(np.float32(1)+erf(projected*np.float32(2**-.5)))
                raw = (encoded[:,-1] @ w['head.weight'].T+w['head.bias']).reshape(-1,5,3,3)
                ordered = np.concatenate((raw[...,:1],raw[...,:1]+np.cumsum(np.logaddexp(np.float32(0),raw[...,1:]),axis=-1)),axis=-1)
                parts.append(ordered)
        raw = np.concatenate(parts) if parts else np.empty((0,5,3,3),np.float32)
        return (raw*np.asarray(self.metadata['target_scale'])[None,...,None]+
                np.asarray(self.metadata['target_mean'])[None,...,None]).astype(np.float32)
