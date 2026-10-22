"""Cache frozen ImageNet ResNet50 observations and fixed classifier outputs.

This is a documented implementation choice for this runnable MAP baseline;
the available article does not fully specify encoder fine-tuning details.
No target class name or scene identity is passed into either image network.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

import numpy as np
from PIL import Image
import torch
from torch import nn

from demo.classifier import CLASSES, load_classifier, predict_images


WEIGHT_URL = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
WEIGHT_NAME = 'resnet50-0676ba61.pth'
POSITION_RE = re.compile(r'^x_(-?\d+(?:\.\d+)?)_y_(-?\d+(?:\.\d+)?)_z_(-?\d+(?:\.\d+)?)_')
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
PREPROCESS = {
    'cache_version': 1,
    'encoder': 'standard ResNet50 [3,4,6,3] bottleneck, stride in 3x3 conv (torchvision v1.5 convention)',
    'weights': 'ImageNet1K_V1 resnet50-0676ba61.pth',
    'encoder_training': 'frozen eval mode; no encoder fine-tuning',
    'encoder_input': 'full RGB image resized directly from 256x256 to 224x224 with PIL bilinear',
    'normalization_mean': [0.485, 0.456, 0.406],
    'normalization_std': [0.229, 0.224, 0.225],
    'features': 'global average pooled 2048-vector stored as float16',
    'classifier_input': 'supplied visible-target bbox crop without padding; demo.classifier preprocessing',
    'classifier_invalid_bbox': 'all-zero probabilities, visible=False',
    'sort': 'ascending (x,y,z) integer coordinates; only bboxes.json entries',
    'coordinate_convention': 'AirSim NED; z < 0, altitude = -z meters',
    'scope_note': 'Frozen encoder is this runnable baseline implementation choice; not an official released MAP implementation or an exact paper reproduction.',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(channels * 4),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.relu(self.bn3(self.conv3(out)) + identity)


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._layer(64, 64, 3, 1)
        self.layer2 = self._layer(256, 128, 4, 2)
        self.layer3 = self._layer(512, 256, 6, 2)
        self.layer4 = self._layer(1024, 512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)

    @staticmethod
    def _layer(in_channels, channels, count, stride):
        return nn.Sequential(Bottleneck(in_channels, channels, stride),
                             *[Bottleneck(channels * 4, channels) for _ in range(count - 1)])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def get_weights(out):
    directory = out / 'weights'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / WEIGHT_NAME
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        if temporary.exists() and sha256(temporary).startswith('0676ba61'):
            temporary.replace(path)
        else:
            print(f'Downloading {WEIGHT_URL}', flush=True)
            urllib.request.urlretrieve(WEIGHT_URL, temporary)
            if not sha256(temporary).startswith('0676ba61'):
                raise ValueError('Downloaded ResNet50 weights failed the official SHA256 prefix check')
            temporary.replace(path)
    digest = sha256(path)
    if not digest.startswith('0676ba61'):
        raise ValueError(f'Invalid official ResNet50 weights: {path}')
    return path, digest


def read_scene(data, scene):
    folder = data / 'multiview_imgs' / f'{scene:02d}'
    annotation = folder / 'bboxes.json'
    entries = json.loads(annotation.read_text(encoding='utf-8'))
    records = []
    for name, box in entries.items():
        if Path(name).name != name:
            raise ValueError(f'Annotation name is not a local filename: {name}')
        match = POSITION_RE.match(name)
        if match is None:
            raise ValueError(f'Unrecognized coordinates: {name}')
        floating = tuple(float(value) for value in match.groups())
        position = tuple(int(value) for value in floating)
        if floating != position:
            raise ValueError(f'Non-integer sampling position: {name}')
        if not all(value % 10 == 0 for value in position) or not (-50 <= position[0] <= 50 and -50 <= position[1] <= 50 and -100 <= position[2] <= -60):
            raise ValueError(f'Position outside the expected sampling lattice: {name}')
        if len(box) != 4 or not np.isfinite(box).all():
            raise ValueError(f'Invalid bbox: {name}: {box}')
        x1, y1, x2, y2 = box
        if not (0 <= x1 <= x2 <= 256 and 0 <= y1 <= y2 <= 256):
            raise ValueError(f'Out-of-bounds bbox: {name}: {box}')
        path = folder / name
        stat = path.stat()
        records.append({'name': name, 'path': path, 'position': position, 'box': box,
                        'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    records.sort(key=lambda row: row['position'])
    if len(records) != 605 or len({row['position'] for row in records}) != 605:
        raise ValueError(f'Scene {scene} does not contain 605 distinct annotated lattice positions')
    stats = [[row['name'], row['bytes'], row['mtime_ns']] for row in records]
    image_fingerprint = hashlib.sha256(json.dumps(stats, separators=(',', ':')).encode()).hexdigest()
    annotation_stat = annotation.stat()
    source = {
        'dataset_path': str(data.resolve()), 'scene_path': str(folder.resolve()),
        'annotation_path': str(annotation.resolve()), 'annotation_sha256': sha256(annotation),
        'annotation_bytes': annotation_stat.st_size, 'annotation_mtime_ns': annotation_stat.st_mtime_ns,
        'annotated_image_count': len(records), 'annotated_image_bytes': sum(row['bytes'] for row in records),
        'images_name_size_mtime_sha256': image_fingerprint,
        'fingerprint_note': 'Image fingerprint covers names, file byte sizes and mtimes, not image-content hashes.',
    }
    source['fingerprint'] = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return records, source


def load_image(record):
    with Image.open(record['path']) as opened:
        image = opened.convert('RGB')
    if image.size != (256, 256):
        raise ValueError(f'Unexpected image dimensions {image.size}: {record["path"]}')
    resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy((array - MEAN) / STD)
    x1, y1, x2, y2 = record['box']
    crop = image.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else None
    return tensor, crop


def validate_arrays(arrays, count=605):
    expected = {'features': (count, 2048), 'probs': (count, 5), 'boxes': (count, 4),
                'positions': (count, 3), 'names': (count,), 'visible': (count,)}
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f'{key} has shape {arrays[key].shape}; expected {shape}')
    for key in ('features', 'probs', 'boxes'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'Non-finite {key}')
    visible = arrays['visible']
    if not np.allclose(arrays['probs'][visible].sum(1), 1.0, atol=1e-5):
        raise ValueError('Classifier probabilities do not sum to one')
    if np.any(arrays['probs'][~visible]):
        raise ValueError('Invisible targets must have zero probabilities')
    if (arrays['probs'] < 0).any() or (arrays['probs'] > 1).any():
        raise ValueError('Classifier probabilities outside [0,1]')


def valid_cache(npz_path, json_path, signature):
    if not (npz_path.is_file() and json_path.is_file()):
        return False
    try:
        metadata = json.loads(json_path.read_text(encoding='utf-8'))
        if metadata['signature'] != signature or metadata['npz_sha256'] != sha256(npz_path):
            return False
        with np.load(npz_path, allow_pickle=False) as arrays:
            validate_arrays(arrays)
        return True
    except (KeyError, OSError, ValueError):
        return False


@torch.inference_mode()
def extract_scene(records, encoder, classifier, device, batch_size, workers, amp):
    feature_parts = []
    probabilities = np.zeros((len(records), len(CLASSES)), dtype=np.float32)
    visible = np.zeros(len(records), dtype=bool)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(records), batch_size):
            batch_records = records[start:start + batch_size]
            loaded = list(pool.map(load_image, batch_records))
            batch = torch.stack([item[0] for item in loaded]).to(device)
            context = torch.autocast('cuda', dtype=torch.float16) if amp and device.type == 'cuda' else nullcontext()
            with context:
                embeddings = encoder(batch)
            feature_parts.append(embeddings.float().cpu().numpy().astype(np.float16))
            crop_indices = [i for i, item in enumerate(loaded) if item[1] is not None]
            if crop_indices:
                crops = [loaded[i][1] for i in crop_indices]
                indices = np.asarray(crop_indices) + start
                probabilities[indices] = predict_images(classifier, crops, device, batch_size)
                visible[indices] = True
    arrays = {
        'features': np.concatenate(feature_parts), 'probs': probabilities,
        'boxes': np.asarray([record['box'] for record in records], dtype=np.float32),
        'positions': np.asarray([record['position'] for record in records], dtype=np.int16),
        'names': np.asarray([record['name'] for record in records]), 'visible': visible,
    }
    validate_arrays(arrays, len(records))
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('dataset'))
    parser.add_argument('--classifier', type=Path, default=Path('output/demo/classifier.pt'))
    parser.add_argument('--out', type=Path, default=Path('output/map/cache'))
    parser.add_argument('--scenes', type=int, nargs='+', default=list(range(1, 151)))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    scenes = sorted(set(args.scenes))
    if not scenes or min(scenes) < 1 or max(scenes) > 150 or args.batch_size < 1 or args.workers < 1:
        parser.error('Scenes must be in 1..150; batch-size and workers must be positive')
    torch.set_num_threads(8)
    # Keep classifier probabilities stable across different batch shapes.
    # Encoder autocast remains explicit and separately recorded below.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    scene_out = args.out / 'scenes'
    scene_out.mkdir(exist_ok=True)
    weights_path, weights_sha = get_weights(args.out)
    classifier_sha = sha256(args.classifier)
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    amp = not args.no_amp and device.type == 'cuda'
    preprocess = {**PREPROCESS, 'encoder_compute_precision': 'CUDA float16 autocast' if amp else 'float32',
                  'classifier_compute_precision': 'float32; TF32 disabled', 'classifier_classes': CLASSES}
    encoder = ResNet50()
    encoder.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    encoder.requires_grad_(False).to(device).eval()
    classifier, device = load_classifier(args.classifier, str(device))
    classifier.requires_grad_(False)
    started = time.perf_counter()
    extracted, skipped = [], []
    print(f'Frozen ResNet50 feature extraction: {len(scenes)} scenes on {device}; amp={amp}', flush=True)
    for index, scene in enumerate(scenes):
        records, source = read_scene(args.data, scene)
        signature = {'source': source, 'pretrained_sha256': weights_sha,
                     'classifier_sha256': classifier_sha, 'preprocess': preprocess}
        npz_path, json_path = scene_out / f'{scene:03d}.npz', scene_out / f'{scene:03d}.json'
        if not args.force and valid_cache(npz_path, json_path, signature):
            skipped.append(scene)
        else:
            scene_started = time.perf_counter()
            arrays = extract_scene(records, encoder, classifier, device, args.batch_size, args.workers, amp)
            temporary = npz_path.with_suffix('.npz.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, **arrays)
            temporary.replace(npz_path)
            metadata = {'scene': scene, 'signature': signature, 'npz_sha256': sha256(npz_path),
                        'image_count': len(records), 'visible_count': int(arrays['visible'].sum()),
                        'extract_seconds': time.perf_counter() - scene_started,
                        'batch_size': args.batch_size, 'workers': args.workers,
                        'device': str(device), 'torch_version': str(torch.__version__),
                        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                        'array_dtypes': {key: str(value.dtype) for key, value in arrays.items()}}
            write_json(json_path, metadata)
            extracted.append(scene)
        if (index + 1) % 10 == 0 or index == 0 or index + 1 == len(scenes):
            print(f'[{index + 1}/{len(scenes)}] scene={scene:03d}; extracted={len(extracted)} skipped={len(skipped)} elapsed={time.perf_counter() - started:.1f}s', flush=True)
    available = []
    for path in sorted(scene_out.glob('*.json')):
        metadata = json.loads(path.read_text(encoding='utf-8'))
        if metadata['signature']['source']['dataset_path'] == str(args.data.resolve()) and metadata['signature']['pretrained_sha256'] == weights_sha and metadata['signature']['classifier_sha256'] == classifier_sha and metadata['signature']['preprocess'] == preprocess:
            available.append({'scene': metadata['scene'], 'path': str(path.with_suffix('.npz').resolve()),
                              'metadata_path': str(path.resolve()), 'npz_sha256': metadata['npz_sha256'],
                              'source_fingerprint': metadata['signature']['source']['fingerprint'],
                              'image_count': metadata['image_count'], 'visible_count': metadata['visible_count']})
    manifest = {'cache_version': PREPROCESS['cache_version'], 'dataset_path': str(args.data.resolve()),
                'pretrained_url': WEIGHT_URL, 'pretrained_sha256': weights_sha,
                'classifier_path': str(args.classifier.resolve()), 'classifier_sha256': classifier_sha,
                'preprocess': preprocess, 'scenes': available, 'total_scenes': len(available),
                'total_images': sum(row['image_count'] for row in available),
                'last_run': {'requested_scenes': scenes, 'extracted': extracted, 'skipped': skipped,
                             'elapsed_seconds': time.perf_counter() - started}}
    write_json(args.out / 'cache_manifest.json', manifest)
    print(f'Manifest: {args.out / "cache_manifest.json"}; available={len(available)} scenes', flush=True)


if __name__ == '__main__':
    main()
