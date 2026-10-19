"""Validated, whole-take data for webcam-controller distillation (no leader)."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np


def read_take(path: Path, control: str = 'reach', stride: int = 6) -> tuple[dict, np.ndarray]:
    """Return metadata and original row indices; never reindex filtered video."""
    path = Path(path)
    if stride < 1:
        raise ValueError('stride must be positive')
    meta = json.loads((path / 'meta.json').read_text())
    if meta.get('mode') != 'free' or meta.get('control') != control:
        raise ValueError(f'need free webcam {control} take, got {meta.get("mode")}/{meta.get("control")}')
    if meta.get('mirrored'):
        # D-T (2026-10-18): the OBSBOT image is not mirrored; --mirrored inverted the left/right targets.
        raise ValueError('recorded with --mirrored: left/right targets are inverted (decision D-T); re-record '
                         'or re-derive targets before training')
    with np.load(path / 'frames.npz', allow_pickle=False) as z:
        q, t, status = z['q_baseline'], z['t'], z['status']
        n = len(t)
        if q.shape != (n, 6) or status.shape != (n,) or t.shape != (n,):
            raise ValueError('invalid target, status or timestamp shape')
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('timestamps must be finite and strictly increasing')
        keep = (status == 2) & np.isfinite(q).all(axis=1)
    cap = cv2.VideoCapture(str(path / 'human.mp4'))
    try:
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != n:
            raise ValueError('video frame count does not match NPZ rows')
    finally:
        cap.release()
    # Subsample the original timeline, rather than compressing gaps in tracking.
    rows = np.flatnonzero(keep & (np.arange(n) % stride == 0))
    if not len(rows):
        raise ValueError('no finite TRACKING targets at this stride')
    return {**meta, 'total_frames': n}, rows


def iter_rgb(path: Path, rows: np.ndarray):
    """Decode sequentially for reliable frame correspondence, yielding RGB arrays."""
    cap = cv2.VideoCapture(str(Path(path) / 'human.mp4'))
    try:
        cursor = 0
        for row in rows:
            while cursor <= row:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'truncated video at row {cursor}: {path}')
                cursor += 1
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def audit_sessions(root: Path, control: str = 'reach', stride: int = 6) -> dict:
    report = {'root': str(root), 'control': control, 'stride': stride,
              'accepted': [], 'rejected': [], 'usable_frames': 0}
    if stride < 1:
        raise ValueError('stride must be positive')
    for path in sorted(Path(root).glob('*')):
        if not path.is_dir():
            continue
        try:
            meta, rows = read_take(path, control, stride)
            # Decode all frames, not just the declared container frame count.
            count = sum(1 for _ in iter_rgb(path, np.arange(meta['total_frames'])))
            report['accepted'].append({'name': path.name, 'total_frames': count,
                                       'usable_frames': len(rows), 'robot': meta.get('robot')})
            report['usable_frames'] += len(rows)
        except (ValueError, OSError, KeyError, EOFError, BadZipFile) as exc:
            report['rejected'].append({'name': path.name, 'reason': str(exc)})
    return report


def split_takes(names: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Deterministic 70/15/15-ish split, with at least one take per partition."""
    names = sorted(names)
    if len(names) < 3 or len(set(names)) != len(names):
        raise ValueError('need at least 3 distinct usable takes (prefer 12-20)')
    ordered = np.random.default_rng(seed).permutation(names).tolist()
    n = max(1, int(len(names) * 0.15))
    return {'train': ordered[2 * n:], 'val': ordered[n:2 * n], 'test': ordered[:n]}
