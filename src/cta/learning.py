"""Offline DINOv2 + PyTorch head training. Never opens cameras or robot ports.

python -m cta.learning --help
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from cta.learning_data import audit_sessions, iter_rgb, read_take, split_takes
from cta.paths import data_dir

MODEL_ID = 'facebook/dinov2-small'


def prepare_image(rgb: np.ndarray) -> np.ndarray:
    """Letterbox the full human view; center-cropping can remove reaching hands."""
    h, w = rgb.shape[:2]
    scale = 224 / max(h, w)
    resized = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
    image = np.zeros((224, 224, 3), np.uint8)
    rh, rw = resized.shape[:2]
    y, x = (224 - rh) // 2, (224 - rw) // 2
    image[y:y + rh, x:x + rw] = resized
    return image


class JointHead(nn.Module):
    """Custom trainable head; outputs six simulation-space joint targets in radians."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Linear(128, 6))

    def forward(self, features):
        return self.net((features - self.feature_mean) / self.feature_std)


def load_head(path: Path) -> JointHead:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = JointHead(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def train_head(takes: dict, output: Path, epochs: int = 100, seed: int = 42) -> dict:
    """Train on cached frozen features, select on validation, evaluate test once."""
    if epochs < 1:
        raise ValueError('epochs must be positive')
    split = split_takes(list(takes), seed)
    dim = None
    for name, take in takes.items():
        x, y = take['features'], take['targets']
        if x.ndim != 2 or len(x) == 0 or y.shape != (len(x), 6):
            raise ValueError(f'invalid feature/target shapes: {name}')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'non-finite cache: {name}')
        dim = x.shape[1] if dim is None else dim
        if x.shape[1] != dim:
            raise ValueError('feature dimensions differ between takes')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    # Small heads are faster on CPU; expensive DINOv2 extraction uses CUDA separately.
    arrays = {}
    for part, names in split.items():
        arrays[part] = tuple(torch.from_numpy(np.concatenate([takes[n][key] for n in names])).float()
                             for key in ('features', 'targets'))
    x_train, y_train = arrays['train']
    model = JointHead(dim)
    model.feature_mean.copy_(x_train.mean(0))
    model.feature_std.copy_(x_train.std(0, unbiased=False).clamp_min(1e-6))
    mean_target = y_train.mean(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                         batch_size=128, shuffle=True, num_workers=0)
    best, history = float('inf'), []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(arrays['val'][0]), arrays['val'][1]).item()
        history.append({'epoch': epoch + 1, 'train_mse_rad2': loss_sum / len(x_train), 'val_mse_rad2': val})
        if val < best:
            best = val
            torch.save({'state_dict': model.state_dict(), 'input_dim': dim, 'seed': seed,
                        'epoch': epoch + 1, 'target': 'q_baseline', 'units': 'radians',
                        'mean_target': mean_target.tolist(), 'split': split}, output / 'best.pt')
    model = load_head(output / 'best.pt')
    with torch.no_grad():
        x_test, y_test = arrays['test']
        per_joint = torch.rad2deg((model(x_test) - y_test).abs().mean(0)).tolist()
        baseline = torch.rad2deg((mean_target - y_test).abs().mean()).item()
    metrics = {'target': 'q_baseline', 'meaning': 'imitation error vs classical teacher; not physical accuracy',
               'test_mae_deg': float(np.mean(per_joint)), 'test_per_joint_mae_deg': per_joint,
               'joint_order': ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_flex', 'wrist_roll', 'gripper'],
               'mean_baseline_test_mae_deg': baseline, 'best_val_mse_rad2': best,
               'split': split, 'frames': {k: len(v[0]) for k, v in arrays.items()},
               'seed': seed, 'epochs': epochs, 'torch_version': str(torch.__version__)}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    with (output / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return metrics


def load_encoder(device: str):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    encoder.requires_grad_(False)
    return processor, encoder


def encode_images(images: list, processor, encoder, device: str) -> np.ndarray:
    inputs = processor(images=images, do_resize=False, do_center_crop=False, return_tensors='pt').to(device)
    with torch.no_grad():
        features = encoder(**inputs).last_hidden_state[:, 0]
    return features.cpu().float().numpy()


def validate_setups(metadata: list[dict]) -> None:
    """A single-image student cannot infer a hidden calibration/setup change."""
    keys = ('hand', 'mirrored', 'camera_mode', 'calibration', 'gains', 'control', 'reach_mapping', 'reach_home_n')
    first = metadata[0]
    for meta in metadata[1:]:
        for key in keys:
            if meta.get(key) != first.get(key):
                raise ValueError(f'incompatible {key} across takes; record one fixed setup per experiment')


def extract(root: Path, output: Path, control='reach', stride=6, batch_size=16, device='cuda') -> dict:
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    report = audit_sessions(root, control, stride)
    if report['rejected']:
        raise ValueError(f'fix or move rejected takes before extraction: {report["rejected"]}')
    if len(report['accepted']) < 3:
        raise ValueError('record at least 3 usable takes before extraction; prefer 12-20')
    validate_setups([read_take(Path(root) / entry['name'], control, stride)[0]
                     for entry in report['accepted']])
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    processor, encoder = load_encoder(device)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'model_id': MODEL_ID, 'model_revision': getattr(encoder.config, '_commit_hash', None),
                'preprocessing': 'RGB full-frame letterbox 224; official normalization; CLS token',
                'control': control, 'stride': stride, 'target': 'q_baseline', 'units': 'radians', 'takes': []}
    for entry in report['accepted']:
        name = entry['name']
        path = Path(root) / name
        meta, rows = read_take(path, control, stride)
        embeddings, batch = [], []
        for rgb in iter_rgb(path, rows):
            batch.append(prepare_image(rgb))
            if len(batch) == batch_size:
                embeddings.append(encode_images(batch, processor, encoder, device))
                batch = []
        if batch:
            embeddings.append(encode_images(batch, processor, encoder, device))
        with np.load(path / 'frames.npz', allow_pickle=False) as z:
            targets = z['q_baseline'][rows].astype('float32')
        np.savez_compressed(output / f'{name}.npz', features=np.concatenate(embeddings), targets=targets, rows=rows)
        digest = hashlib.sha256((path / 'frames.npz').read_bytes()).hexdigest()
        manifest['takes'].append({'name': name, 'metadata': meta, 'source_npz_sha256': digest, 'frames': len(rows)})
        print(f'extracted {name}: {len(rows)} frames', flush=True)
    # Written last: no manifest means extraction was interrupted and cache is unusable.
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit', help='validate recordings; no hardware is opened')
    audit.add_argument('--sessions', type=Path)
    audit.add_argument('--stride', type=int, default=6)
    ext = sub.add_parser('extract', help='cache frozen DINOv2 features from completed takes')
    ext.add_argument('--sessions', type=Path)
    ext.add_argument('--output', type=Path, required=True)
    ext.add_argument('--stride', type=int, default=6)
    ext.add_argument('--batch-size', type=int, default=16)
    ext.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    train = sub.add_parser('train', help='train custom head and evaluate held-out takes')
    train.add_argument('--cache', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--seed', type=int, default=42)
    smoke = sub.add_parser('smoke-model', help='download pretrained model; check a head training step (not robot results)')
    smoke.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)
    try:
        if args.command == 'audit':
            report = audit_sessions(args.sessions or data_dir() / 'sessions', stride=args.stride)
            print(json.dumps(report, indent=2))
            return 0 if report['accepted'] and not report['rejected'] else 1
        if args.command == 'extract':
            extract(args.sessions or data_dir() / 'sessions', args.output, stride=args.stride,
                    batch_size=args.batch_size, device=args.device)
        elif args.command == 'train':
            manifest = json.loads((args.cache / 'manifest.json').read_text())
            takes = {}
            for take in manifest['takes']:
                name = take['name']
                if Path(name).name != name or '/' in name or '\\' in name:
                    raise ValueError('invalid take name in manifest')
                with np.load(args.cache / f'{name}.npz', allow_pickle=False) as z:
                    takes[name] = {k: z[k].copy() for k in ('features', 'targets')}
            metrics = train_head(takes, args.output, args.epochs, args.seed)
            (args.output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
            print(json.dumps(metrics, indent=2))
        elif args.command == 'smoke-model':
            processor, encoder = load_encoder(args.device)
            features = encode_images([prepare_image(np.zeros((480, 640, 3), np.uint8))], processor, encoder, args.device)
            head = JointHead(features.shape[1])
            loss = head(torch.from_numpy(features)).square().mean()
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
                raise ValueError('head gradients failed')
            print(json.dumps({'model': MODEL_ID, 'device': args.device, 'features': list(features.shape),
                              'head_gradients_finite': True, 'robot_performance_measured': False}))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
