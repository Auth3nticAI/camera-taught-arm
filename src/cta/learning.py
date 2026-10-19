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


HAND_MARGIN = 1.6     # crop side = landmark box side x this (fingers + a little context)
HAND_MIN_SIDE = 0.05  # of the longer image side, so a far-away hand is not blown up to noise


def hand_crop(rgb: np.ndarray, lm_img: np.ndarray) -> np.ndarray:
    """224x224 square close-up centred on the 21 hand landmarks (normalized image coords, may exceed 0..1).
    Off-image parts are black; no landmarks gives a black image (D-U)."""
    h, w = rgb.shape[:2]
    uv = np.asarray(lm_img, dtype=np.float64)[:, :2]
    if not np.isfinite(uv).all():
        return np.zeros((224, 224, 3), np.uint8)
    px = uv * (w, h)
    (x0, y0), (x1, y1) = px.min(0), px.max(0)
    side = max(max(x1 - x0, y1 - y0) * HAND_MARGIN, HAND_MIN_SIDE * max(h, w), 1.0)
    k = 224 / side
    m = np.array([[k, 0, 112 - k * (x0 + x1) / 2], [0, k, 112 - k * (y0 + y1) / 2]])
    return cv2.warpAffine(rgb, m, (224, 224), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def select_features(z, input_mode: str = 'full') -> dict:
    """Pick the head input from a cached take: 'full' frame, 'hand' crop, or 'both' concatenated (D-U)."""
    if input_mode not in ('full', 'hand', 'both'):
        raise ValueError(f'unknown input {input_mode!r}')
    keys = {'full': [''], 'hand': ['_hand'], 'both': ['', '_hand']}[input_mode]
    if any(f'features{k}' not in z for k in keys):
        raise ValueError(f'input {input_mode!r} needs features_hand in the cache (extract --hand-crop)')
    out = {'targets': z['targets'], 'features': np.concatenate([z[f'features{k}'] for k in keys], axis=-1)}
    if all(f'features{k}_aug' in z for k in keys):
        out['features_aug'] = np.concatenate([z[f'features{k}_aug'] for k in keys], axis=-1)
    return out


def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ch. 12 augmentation for a letterboxed 224x224 frame: small scale/shift/rotation + brightness/contrast.
    NEVER a horizontal flip - that would swap the user's left/right and corrupt the joint targets."""
    h, w = img.shape[:2]
    scale = rng.uniform(0.9, 1.1)
    angle = rng.uniform(-5.0, 5.0)
    tx, ty = rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    contrast, brightness = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    return np.clip(out.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)


class JointHead(nn.Module):
    """Custom trainable head; outputs six simulation-space joint targets in radians."""

    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))
        layers = [nn.Linear(input_dim, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers, nn.Linear(hidden, 6))

    def forward(self, features):
        return self.net((features - self.feature_mean) / self.feature_std)


def load_head(path: Path) -> JointHead:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = JointHead(checkpoint['input_dim'], checkpoint.get('hidden', 128), checkpoint.get('dropout', 0.0))
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def train_head(takes: dict, output: Path, epochs: int = 100, seed: int = 42, dropout: float = 0.0,
               weight_decay: float = 0.01, feature_noise: float = 0.0, use_aug: bool = False,
               hidden: int = 128, split: dict | None = None) -> dict:
    """Train on cached frozen features, select on validation, evaluate test once.
    Regularization (ch. 12): dropout, AdamW weight decay, Gaussian noise on standardized features, and
    augmented-image feature copies ('features_aug', TRAIN takes only - never val/test)."""
    if epochs < 1:
        raise ValueError('epochs must be positive')
    if split is None:
        split = split_takes(list(takes), seed)
    elif set(sum(split.values(), [])) - set(takes) or set(split['train']) & set(split['val'] + split['test']):
        raise ValueError('split must use known takes and keep train disjoint from val/test')
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
    model = JointHead(dim, hidden, dropout)
    model.feature_mean.copy_(x_train.mean(0))  # normalization from ORIGINAL train frames only
    model.feature_std.copy_(x_train.std(0, unbiased=False).clamp_min(1e-6))
    mean_target = y_train.mean(0)
    n_aug = 0
    if use_aug:
        missing = [n for n in split['train'] if 'features_aug' not in takes[n]]
        if missing:
            raise ValueError(f'use_aug needs features_aug in the cache (extract --augment N): {missing}')
        xa = np.concatenate([takes[n]['features_aug'].reshape(-1, dim) for n in split['train']])
        ya = np.concatenate([np.repeat(takes[n]['targets'], takes[n]['features_aug'].shape[1], axis=0)
                             for n in split['train']])
        n_aug = len(xa)
        x_train = torch.cat([x_train, torch.from_numpy(xa).float()])
        y_train = torch.cat([y_train, torch.from_numpy(ya).float()])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    noise_std = model.feature_std * feature_noise
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                         batch_size=128, shuffle=True, num_workers=0)
    best, history = float('inf'), []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            if feature_noise > 0:
                x = x + torch.randn_like(x) * noise_std
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
                        'hidden': hidden, 'dropout': dropout,
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
               'split': split, 'frames': {**{k: len(v[0]) for k, v in arrays.items()},
                                          **({'train_aug': n_aug} if use_aug else {})},
               'config': {'dropout': dropout, 'weight_decay': weight_decay, 'feature_noise': feature_noise,
                          'use_aug': use_aug, 'hidden': hidden},
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

    def norm(key, value):
        # camera_mode carries the MEASURED fps, which jitters run to run (59.41 vs 59.47) on the same setup
        if key == 'camera_mode' and isinstance(value, dict):
            return (value.get('width'), value.get('height'), value.get('fourcc'), round(float(value.get('fps', 0))))
        return value

    first = metadata[0]
    for meta in metadata[1:]:
        for key in keys:
            if norm(key, meta.get(key)) != norm(key, first.get(key)):
                raise ValueError(f'incompatible {key} across takes; record one fixed setup per experiment')


def extract(root: Path, output: Path, control='reach', stride=6, batch_size=16, device='cuda',
            augment: int = 0, seed: int = 42, hand: bool = False) -> dict:
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
                'control': control, 'stride': stride, 'target': 'q_baseline', 'units': 'radians',
                'augment_copies': augment, 'augment': 'affine scale 0.9-1.1, rot +-5 deg, shift +-5%, '
                'contrast 0.8-1.2, brightness +-20; no horizontal flip',
                'hand_crop': f'lm_img box x{HAND_MARGIN}, min side {HAND_MIN_SIDE} of frame, 224 square' if hand
                else None, 'takes': []}
    rng = np.random.default_rng(seed)
    for entry in report['accepted']:
        name = entry['name']
        path = Path(root) / name
        meta, rows = read_take(path, control, stride)
        with np.load(path / 'frames.npz', allow_pickle=False) as z:
            targets = z['q_baseline'][rows].astype('float32')
            lms = z['lm_img'][rows] if hand else None
        streams = ['features'] + (['features_hand'] if hand else [])
        pending = {k: [] for k in streams + [f'{k}_aug' for k in streams]}
        done = {k: [] for k in pending}

        def flush(min_size):
            for k, imgs in pending.items():
                if imgs and len(imgs) >= min_size:
                    done[k].append(encode_images(imgs, processor, encoder, device))
                    pending[k] = []

        for i, rgb in enumerate(iter_rgb(path, rows)):
            views = {'features': prepare_image(rgb)}
            if hand:
                views['features_hand'] = hand_crop(rgb, lms[i])
            for k, img in views.items():
                pending[k].append(img)
                pending[f'{k}_aug'].extend(augment_image(img, rng) for _ in range(augment))
            flush(batch_size)
        flush(1)
        arrays = {k: np.concatenate(v) for k, v in done.items() if v}
        for k in [k for k in arrays if k.endswith('_aug')]:
            arrays[k] = arrays[k].reshape(len(rows), augment, -1)
        np.savez_compressed(output / f'{name}.npz', targets=targets, rows=rows, **arrays)
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
    ext.add_argument('--augment', type=int, default=0, help='augmented copies per frame (ch. 12; no flips)')
    ext.add_argument('--hand-crop', action='store_true', help='also cache features of a hand close-up (D-U)')
    train = sub.add_parser('train', help='train custom head and evaluate held-out takes')
    train.add_argument('--cache', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--dropout', type=float, default=0.0)
    train.add_argument('--weight-decay', type=float, default=0.01)
    train.add_argument('--feature-noise', type=float, default=0.0, help='Gaussian noise x feature std')
    train.add_argument('--use-aug', action='store_true', help='add augmented copies from the cache (train takes only)')
    train.add_argument('--hidden', type=int, default=128)
    train.add_argument('--input', choices=('full', 'hand', 'both'), default='full',
                       help='head input: full frame, hand crop, or both concatenated (D-U)')
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
                    batch_size=args.batch_size, device=args.device, augment=args.augment,
                    hand=args.hand_crop)
        elif args.command == 'train':
            manifest = json.loads((args.cache / 'manifest.json').read_text())
            takes = {}
            for take in manifest['takes']:
                name = take['name']
                if Path(name).name != name or '/' in name or '\\' in name:
                    raise ValueError('invalid take name in manifest')
                with np.load(args.cache / f'{name}.npz', allow_pickle=False) as z:
                    takes[name] = {k: v.copy() for k, v in select_features(z, args.input).items()}
            metrics = train_head(takes, args.output, args.epochs, args.seed, dropout=args.dropout,
                                 weight_decay=args.weight_decay, feature_noise=args.feature_noise,
                                 use_aug=args.use_aug, hidden=args.hidden)
            metrics['config']['input'] = args.input
            (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
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
