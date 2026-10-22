"""Optional execution optimizations without changing the locked training source.

The default CLI runs only a short validation job. It never edits or resumes a
formal experiment directory. Numerical precision, effective batch size, TD
objective, replay sampling and BatchNorm semantics remain unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .model import E2EModel, E2ERunner


@dataclass(frozen=True)
class PerformanceOptions:
    checkpoint_mode: str = 'full'
    microbatch: int = 64
    gpu_pixels: bool = False
    pixel_load_batch: int = 256

    def __post_init__(self):
        if self.checkpoint_mode not in ('full', 'none', 'late'):
            raise ValueError('checkpoint_mode must be full, none or late')
        if self.microbatch < 1 or self.pixel_load_batch < 1:
            raise ValueError('Image microbatch and pixel-load batch must be positive')


class PerformanceModel(E2EModel):
    """Same model parameters; choose where CNN activations are recomputed.

    ``full`` reproduces the original entire-encoder checkpoint. ``none`` keeps
    activations for backward. ``late`` checkpoints only ResNet stages 3 and 4,
    retaining the stem and early-stage activations. All options retain the
    complete gradient and fixed BatchNorm running statistics.
    """
    def __init__(self, *args, checkpoint_mode='full', **kwargs):
        if checkpoint_mode not in ('full', 'none', 'late'):
            raise ValueError('Unknown checkpoint mode')
        kwargs['checkpoint_encoder'] = checkpoint_mode == 'full'
        super().__init__(*args, **kwargs)
        self.checkpoint_mode = checkpoint_mode

    def _encode_chunk(self, images):
        if self.checkpoint_mode != 'late':
            return super()._encode_chunk(images)
        encoder = self.encoder
        dtype = torch.bfloat16 if self.encoder_precision == 'bfloat16' else torch.float16
        with torch.autocast(images.device.type, dtype=dtype,
                            enabled=self.amp and images.device.type == 'cuda'):
            x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(images))))
            x = encoder.layer2(encoder.layer1(x))
            for stage in (encoder.layer3, encoder.layer4):
                if torch.is_grad_enabled():
                    x = checkpoint(stage, x, use_reentrant=False, preserve_rng_state=False)
                else:
                    x = stage(x)
            return encoder.avgpool(x).flatten(1)


class TensorPixelStore:
    """Immutable uint8 RGB observations resident on one torch device.

    GPU caching removes per-update CPU gathers and host-to-device pixel copies.
    It caches only pixels: online neural features are still recomputed after
    every optimizer step. CPU support exists for tests of indexing semantics.
    """
    def __init__(self, images, count, device, load_batch=256):
        if count < 1 or load_batch < 1:
            raise ValueError('Pixel count and load batch must be positive')
        self.device = torch.device(device)
        first = images.get(np.arange(min(count, load_batch), dtype=np.int64))
        if first.ndim != 4 or first.shape[-1] != 3 or first.dtype != np.uint8:
            raise ValueError('Expected uint8 NHWC RGB source pixels')
        self.tensor = torch.empty((count, *first.shape[1:]), dtype=torch.uint8, device=self.device)
        self.tensor[:len(first)].copy_(torch.from_numpy(first))
        for start in range(len(first), count, load_batch):
            ids = np.arange(start, min(count, start + load_batch), dtype=np.int64)
            self.tensor[start:start + len(ids)].copy_(torch.from_numpy(images.get(ids)))
        self.bytes = self.tensor.numel() * self.tensor.element_size()

    def get(self, indices):
        ids = np.asarray(indices, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self.tensor)):
            raise ValueError('Pixel indices outside cached observations')
        return self.tensor[torch.as_tensor(ids, device=self.device)]


class PerformanceRunner(E2ERunner):
    def __init__(self, data, images, device='cuda', acting_batch=128,
                 gpu_pixels=False, pixel_load_batch=256):
        super().__init__(data, images, device, acting_batch)
        self.pixel_cache = None
        if gpu_pixels:
            self.pixel_cache = TensorPixelStore(images, len(data.boxes), self.device, pixel_load_batch)

    def _encode_ids(self, model, ids):
        if self.pixel_cache is None:
            return super()._encode_ids(model, ids)
        rgb = self.pixel_cache.get(ids)
        tensor = rgb.permute(0, 3, 1, 2).float()
        tensor = (tensor / 255.0 - self.mean) / self.std
        return model.encode(tensor)


@contextmanager
def runtime_adapters(options, identity):
    """Scope execution adapters to one explicit training call.

    Locked source files are not edited. Original configuration fields retain
    their meaning; a separate, checked performance identity is embedded at the
    top level of every model/resume checkpoint and beside training.json.
    """
    from . import experiment
    original_model, original_runner, original_save = experiment.E2EModel, experiment.E2ERunner, experiment.atomic_save

    def make_model(*args, **kwargs):
        kwargs['microbatch'] = options.microbatch
        return PerformanceModel(*args, checkpoint_mode=options.checkpoint_mode, **kwargs)

    def make_runner(*args, **kwargs):
        return PerformanceRunner(*args, gpu_pixels=options.gpu_pixels,
                                 pixel_load_batch=options.pixel_load_batch, **kwargs)

    def save(value, path):
        return original_save({**value, 'performance': identity}, path)

    experiment.E2EModel, experiment.E2ERunner, experiment.atomic_save = make_model, make_runner, save
    try:
        yield
    finally:
        experiment.E2EModel, experiment.E2ERunner, experiment.atomic_save = original_model, original_runner, original_save


def performance_identity(options):
    root = Path(__file__).resolve().parents[1]
    sources = ('e2e_map/performance.py', 'e2e_map/experiment.py', 'e2e_map/model.py', 'e2e_map/images.py')
    return {'schema_version': 1, 'options': asdict(options),
            'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
            'semantics': 'unchanged effective batch, TD loss, action/reward protocol and BN running statistics',
            'precision': 'CUDA BF16 encoder, FP32 feature boundary and policy heads',
            'scope': 'independent short training validation; excluded from formal policy comparisons'}


def train(args, options):
    """Run the original training loop under recorded, isolated adapters."""
    from . import experiment
    if not args.stop_after_updates or not 1 <= args.stop_after_updates <= 200:
        raise ValueError('Performance validation requires a positive stop_after_updates limit of at most 200')
    if args.batch_size != 256:
        raise ValueError('This comparison keeps the effective replay batch fixed at 256')
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    identity_path = args.out / 'performance.json'
    identity = performance_identity(options)
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding='utf8'))
        if previous != identity:
            raise ValueError('Performance settings or source changed; use a new output directory')
        if not args.resume:
            raise ValueError('Performance output exists; use --resume or a new directory')
    elif any(args.out.glob('*.pt')) or (args.out / 'training.json').exists():
        raise ValueError('Refusing to reuse an existing non-performance training directory')
    else:
        identity_path.write_text(json.dumps(identity, indent=2), encoding='utf8')
    args.microbatch = options.microbatch
    args.no_activation_checkpoint = options.checkpoint_mode == 'none'
    with runtime_adapters(options, identity):
        summary = experiment.train(args)
    summary['performance'] = identity
    experiment.write_json(args.out / 'training.json', summary)
    return summary


def main():
    from .experiment import parser, ROOT
    p = parser()
    p.description = __doc__
    p.set_defaults(stop_after_updates=100, save_every=0, microbatch=64)
    p.add_argument('--checkpoint-mode', choices=('full', 'none', 'late'), default='full')
    p.add_argument('--gpu-pixels', action='store_true')
    p.add_argument('--pixel-load-batch', type=int, default=256)
    args = p.parse_args()
    if args.cache is None:
        args.cache = ROOT / ('output/ibe_map/cache' if args.variant in ('sam_map', 'ibe_map') else 'output/map/cache')
    if args.resume and not (args.out / 'performance.json').exists():
        p.error('Only performance jobs created in this independent directory can be resumed')
    train(args, PerformanceOptions(args.checkpoint_mode, args.microbatch, args.gpu_pixels, args.pixel_load_batch))


if __name__ == '__main__':
    main()
