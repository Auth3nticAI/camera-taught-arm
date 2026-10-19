"""The trained network as a live arm driver (--learned RUN_DIR, decision D-V).

Each frame: the same views used in training (hand crop and/or letterboxed full frame) -> frozen DINOv2 CLS ->
the trained JointHead -> 6 joint targets (radians, sim frame, same space as q_baseline). A One Euro filter
damps the network's frame-to-frame jitter. The gripper stays on the pinch rule (blend_gripper). Everything
downstream (joint limits, speed limit, lead bound, table clearance) is the existing send path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cta.filters import OneEuro
from cta.learning import encode_images, hand_crop, load_encoder, load_head, prepare_image

NET_MIN_CUTOFF = 1.0  # Hz: still pose -> heavy smoothing of the ~20 deg prediction noise
NET_BETA = 0.3        # faster motion -> less smoothing (less lag)


def blend_gripper(q_net: np.ndarray, q_pipe: np.ndarray) -> np.ndarray:
    """Network drives joints 0-4; the gripper (5) stays on the pinch rule (its learned error is the worst)."""
    q = np.asarray(q_net, dtype=float).copy()
    q[5] = q_pipe[5]
    return q


class LearnedDriver:
    def __init__(self, run_dir: Path, device: str | None = None, encode=None, smooth: bool = True,
                 min_cutoff: float = NET_MIN_CUTOFF, beta: float = NET_BETA):
        run_dir = Path(run_dir)
        self.input = json.loads((run_dir / 'metrics.json').read_text()).get('config', {}).get('input', 'full')
        if self.input not in ('full', 'hand', 'both'):
            raise ValueError(f'unknown head input {self.input!r} in {run_dir}')
        self.head = load_head(run_dir / 'best.pt').eval()
        self.head.requires_grad_(False)
        if encode is None:
            device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            processor, encoder = load_encoder(device)
            encode = lambda images: encode_images(images, processor, encoder, device)  # noqa: E731
        self.encode = encode
        self.smoother = OneEuro(min_cutoff=min_cutoff, beta=beta) if smooth else None
        self.run_dir = run_dir

    def head_predict(self, rgb: np.ndarray, lm_img: np.ndarray) -> np.ndarray:
        """Raw (unsmoothed) network output for one frame."""
        if lm_img is None or not np.isfinite(np.asarray(lm_img, dtype=float)[:, :2]).all():
            raise ValueError('hand landmarks required for the learned driver')
        views = []
        if self.input in ('full', 'both'):
            views.append(prepare_image(rgb))
        if self.input in ('hand', 'both'):
            views.append(hand_crop(rgb, lm_img))
        x = np.asarray(self.encode(views), dtype=np.float32).reshape(1, -1)  # full then hand, as in training
        with torch.no_grad():
            return self.head(torch.from_numpy(x))[0].numpy().astype(float)

    def predict(self, rgb: np.ndarray, lm_img: np.ndarray, t: float) -> np.ndarray:
        q = self.head_predict(rgb, lm_img)
        return q if self.smoother is None else self.smoother(q, t)

    def reset(self) -> None:
        if self.smoother is not None:
            self.smoother.reset()


def learned_command(driver: LearnedDriver, status: str, obs, rgb: np.ndarray, t: float, q_pipe: np.ndarray,
                    last_net: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """-> (q to drive, network q to remember). The pipeline's status still gates everything (the network was
    trained on TRACKING frames only): TRACKING + hand -> network; COAST -> hold the last network pose;
    otherwise the pipeline q (the send path holds the arm anyway) and the smoother restarts."""
    if status == 'TRACKING' and obs is not None:
        net = driver.predict(rgb, obs.lm_img, t)
        return blend_gripper(net, q_pipe), net
    if status == 'COAST' and last_net is not None:
        return blend_gripper(last_net, q_pipe), last_net
    driver.reset()
    return np.asarray(q_pipe, dtype=float), None
