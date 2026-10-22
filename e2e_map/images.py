"""Memory-mapped RGB observations with explicit source and cache identities.

These caches store pixels, never trainable-network features. Their ordering is
identical to the immutable observation/classifier cache used by every policy.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from map_baseline.features import read_scene, sha256, write_json


PREPROCESS = {'version': 1, 'color': 'RGB', 'source_size': [256, 256],
              'size': [224, 224], 'resize': 'PIL bilinear',
              'storage': 'uint8 NHWC; normalize only at model input'}


def _read_source(path):
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        if image.size != (256, 256):
            raise ValueError(f'Expected 256x256 source: {path}')
        array = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)).copy()
    return array, hashlib.sha256(content).hexdigest()


class RGBImageStore:
    def __init__(self, data, data_root, decomposed_root, cache_root, variant,
                 workers=8, verify_cache=True, verbose=True, observation_cache=None):
        self.scenes = list(data.scenes)
        self.arrays, self.metadata = [], []
        self.kind = 'sam' if variant in ('sam_map', 'ibe_map') else 'raw'
        source_is_sam = 'scene_decomposition' in data.observation_signature['preprocess']
        if source_is_sam != (self.kind == 'sam'):
            raise ValueError('RGB source kind differs from policy observation cache')
        data_root, decomposed_root, cache_root = map(Path, (data_root, decomposed_root, cache_root))
        if data_root.resolve() != Path(data.observation_signature['dataset_path']).resolve():
            raise ValueError('RGB dataset path differs from fixed observation cache')
        folder = cache_root / self.kind
        folder.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for slot, scene in enumerate(self.scenes):
                names = [str(name) for name in data.names[slot * 605:(slot + 1) * 605]]
                if len(names) != 605 or any(Path(name).name != name for name in names):
                    raise ValueError('Expected 605 local observation names per scene')
                source_dir = (decomposed_root / f'{scene:03d}' if self.kind == 'sam'
                              else data_root / 'multiview_imgs' / f'{scene:02d}')
                if observation_cache is not None:
                    fixed_metadata_path = Path(observation_cache) / 'scenes' / f'{scene:03d}.json'
                    fixed_metadata = json.loads(fixed_metadata_path.read_text(encoding='utf8'))
                    records, original_source = read_scene(data_root, scene)
                    if original_source != fixed_metadata['signature']['source']:
                        raise ValueError(f'Original images changed since frozen observation extraction: {scene}')
                    if names != [record['name'] for record in records]:
                        raise ValueError(f'RGB/observation name ordering differs: {scene}')
                if self.kind == 'sam':
                    progress = json.loads((source_dir / 'progress.json').read_text(encoding='utf8'))
                    sig = progress['signature']
                    common = {k: sig[k] for k in ('pretrained_sha256', 'classifier_sha256', 'preprocess')}
                    common['dataset_path'] = sig['source']['dataset_path']
                    if progress.get('completed') != 605 or common != data.observation_signature:
                        raise ValueError(f'Incomplete/mismatched SAM decomposition: {source_dir}')
                paths = [source_dir / name for name in names]
                source_stats = []
                for name, path in zip(names, paths):
                    stat = path.stat()
                    source_stats.append([name, stat.st_size, stat.st_mtime_ns])
                stats_digest = hashlib.sha256(json.dumps(source_stats, separators=(',', ':')).encode()).hexdigest()
                signature = {'scene': scene, 'kind': self.kind, 'preprocess': PREPROCESS,
                             'source_dir': str(source_dir.resolve()), 'names': names,
                             'source_name_size_mtime_sha256': stats_digest,
                             'observation_signature': data.observation_signature}
                path = folder / f'{scene:03d}.npy'
                meta_path = path.with_suffix('.json')
                if path.exists() or meta_path.exists():
                    if not path.exists() or not meta_path.exists():
                        raise ValueError(f'Incomplete image cache; remove only this incomplete pair: {path}')
                    metadata = json.loads(meta_path.read_text(encoding='utf8'))
                    if metadata['signature'] != signature:
                        raise ValueError(f'RGB source/cache signature changed: {path}')
                    if verify_cache and sha256(path) != metadata['npy_sha256']:
                        raise ValueError(f'RGB cache content hash mismatch: {path}')
                else:
                    temporary = path.with_suffix('.npy.tmp')
                    cache = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                                     shape=(605, 224, 224, 3))
                    source_hashes = []
                    for i, (array, digest) in enumerate(pool.map(_read_source, paths)):
                        cache[i] = array
                        source_hashes.append(digest)
                    cache.flush()
                    del cache
                    temporary.replace(path)
                    metadata = {'signature': signature, 'npy_sha256': sha256(path),
                                'source_png_sha256': dict(zip(names, source_hashes)),
                                'source_verification': 'Content SHA256 at build; names/size/mtime on reuse.'}
                    write_json(meta_path, metadata)
                    if verbose:
                        print(f'RGB {self.kind} scene {scene:03d}: 605 images cached', flush=True)
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.dtype != np.uint8 or array.shape != (605, 224, 224, 3):
                    raise ValueError(f'Invalid RGB cache shape/dtype: {path}')
                self.arrays.append(array)
                self.metadata.append(metadata)
        identity = [{'scene': s, 'npy_sha256': m['npy_sha256'],
                     'source_name_size_mtime_sha256': m['signature']['source_name_size_mtime_sha256']}
                    for s, m in zip(self.scenes, self.metadata)]
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def get(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.scenes) * 605):
            raise ValueError('Image indices outside loaded observation scenes')
        batch = np.empty((len(indices), 224, 224, 3), dtype=np.uint8)
        slots = indices // 605
        for slot in np.unique(slots):
            rows = np.flatnonzero(slots == slot)
            batch[rows] = self.arrays[int(slot)][indices[rows] % 605]
        return batch
