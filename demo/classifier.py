"""Small, reproducible ResNet18 vehicle classifier for the UEVAVD demo.

Implements the standard ResNet18 architecture without a torchvision dependency.
Pretrained weights: https://download.pytorch.org/models/resnet18-f37072fd.pth
The validation split is internal to train_605, not a paper benchmark score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn

CLASSES = ['Hatchback', 'Pickup', 'Sedan', 'Sportscar', 'SUV']
FOLDERS = ['SM_Hatchback_17', 'SM_Pickup_5', 'SM_Sedan_01a_14', 'SM_SportsCar_2', 'SM_SUV_01a_17']
IMAGE_SIZE = 128
TRAINING_CONFIG = {
    'optimizer': 'AdamW', 'learning_rate': 0.0003, 'weight_decay': 0.0001,
    'scheduler': 'CosineAnnealingLR', 'label_smoothing': 0.03,
    'augmentation': {'brightness_factor': [0.65, 1.35], 'contrast_factor': [0.75, 1.25], 'color_factor': [0.5, 1.4], 'horizontal_flip': False},
    'resize': '128x128 bilinear', 'normalization_mean': [0.485, 0.456, 0.406],
    'normalization_std': [0.229, 0.224, 0.225], 'validation_fraction': 0.2,
    'fine_tuning': 'all layers', 'precision': 'float32',
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


def image_tensor(image, augment=False):
    image = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    if augment:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.35))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.4))
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((array - MEAN) / STD)


def load_classifier(checkpoint_path, device='cuda'):
    torch.set_num_threads(8)
    device = torch.device(device if device != 'cuda' or torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES:
        raise ValueError('Unexpected classifier class order')
    model = ResNet18(len(CLASSES))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, device


@torch.inference_mode()
def predict_images(model, images, device, batch_size=128):
    model.eval()
    probs = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([image_tensor(im) for im in images[start:start + batch_size]]).to(device)
        probs.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.empty((0, len(CLASSES)), dtype=np.float32)


def train(data, out, epochs=8, seed=42, batch_size=96):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    pretrained = cache / 'resnet18-f37072fd.pth'
    if not pretrained.exists():
        print('Downloading official ImageNet ResNet18 weights...', flush=True)
        urllib.request.urlretrieve('https://download.pytorch.org/models/resnet18-f37072fd.pth', pretrained)
    digest = hashlib.sha256(pretrained.read_bytes()).hexdigest()
    if not digest.startswith('f37072fd'):
        raise RuntimeError('Pretrained ResNet18 weight hash mismatch')
    model = ResNet18(1000)
    model.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True))
    model.fc = nn.Linear(512, len(CLASSES))
    model.to(device)
    print(f'Device: {device}; loading training images...', flush=True)

    rng = np.random.default_rng(seed)
    train_images, train_labels, val_images, val_labels, train_names, val_names = [], [], [], [], [], []
    for label, folder in enumerate(FOLDERS):
        paths = sorted((data / folder).glob('*.png'))
        if not paths:
            raise ValueError(f'No training images in {data / folder}')
        order = rng.permutation(len(paths))
        split = max(1, round(len(paths) * 0.2))
        for rank, idx in enumerate(order):
            with Image.open(paths[int(idx)]) as image:
                image = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            if rank < split:
                val_images.append(image)
                val_labels.append(label)
                val_names.append(str(paths[int(idx)].relative_to(data)))
            else:
                train_images.append(image)
                train_labels.append(label)
                train_names.append(str(paths[int(idx)].relative_to(data)))
    train_y = torch.tensor(train_labels, dtype=torch.long, device=device)
    val_y = np.array(val_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    history, best, best_nll = [], -1.0, float('inf')
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_images))
        loss_sum, correct = 0.0, 0
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            batch = torch.stack([image_tensor(train_images[int(i)], augment=True) for i in ids]).to(device)
            target = train_y[torch.tensor(ids, dtype=torch.long, device=device)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(ids)
            correct += (logits.argmax(1) == target).sum().item()
        val_probs = predict_images(model, val_images, device, batch_size)
        accuracy = float((val_probs.argmax(1) == val_y).mean())
        val_nll = float(-np.log(np.clip(val_probs[np.arange(len(val_y)), val_y], 1e-8, 1)).mean())
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / len(train_images), 'train_accuracy': correct / len(train_images), 'validation_accuracy': accuracy, 'validation_nll': val_nll, 'seconds': time.perf_counter() - epoch_started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best or (accuracy == best and val_nll < best_nll):
            best, best_nll = accuracy, val_nll
            temp_checkpoint = out / 'classifier.pt.tmp'
            torch.save({'architecture': 'resnet18', 'classes': CLASSES, 'image_size': IMAGE_SIZE, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'epoch': epoch + 1, 'validation_accuracy': best, 'validation_nll': best_nll}, temp_checkpoint)
            temp_checkpoint.replace(out / 'classifier.pt')
        scheduler.step()
        summary = {'architecture': 'ResNet18', 'initialization': str(pretrained), 'pretrained_sha256': digest, 'classes': CLASSES, 'seed': seed, 'image_size': IMAGE_SIZE, 'epochs_requested': epochs, 'train_count': len(train_images), 'validation_count': len(val_images), 'validation_note': 'Fixed stratified 80/20 split within unoccluded train_605. Neighboring viewpoints share a scene; this is a sanity check, not independent-context or paper benchmark accuracy.', 'checkpoint_selection': 'Highest validation accuracy, then lowest validation negative log likelihood.', 'best_validation_accuracy': best, 'best_validation_nll': best_nll, 'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None, 'history': history, 'train_files': train_names, 'validation_files': val_names, 'elapsed_seconds': time.perf_counter() - started}
        summary['hyperparameters'] = {**TRAINING_CONFIG, 'batch_size': batch_size, 'scheduler_T_max': epochs}
        (out / 'training.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Checkpoint: {out / "classifier.pt"}; best internal validation={best:.4f}; elapsed={time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('dataset/train_605'))
    parser.add_argument('--out', type=Path, default=Path('output/demo'))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=96)
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.seed, args.batch_size)
