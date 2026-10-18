"""--control reach (Tray, 2026-10-18): the user's arm is only the WHERE. Pure numpy.

The shoulder->wrist vector in body coordinates, divided by the user's arm length, is a normalized reach n
(|n| <= 1). The robot's gripper goes to ROBOT_SHOULDER + ROBOT_REACH * n (base frame: x fwd, y left, z up),
clipped to the workspace box, and the robot gets there with its own IK - no joint copying.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from cta.arm_angles import body_frame, side_indices, unmirror

REACH_FRACTION = 0.95             # ROBOT_REACH = this x the robot's measured max reach (Kinematics.max_reach)
NO_HAND_PITCH_RAD = -np.pi / 4    # claw pitch with no hand: ~45 deg down
ARM_LENGTH_WINDOW = 60            # running-median window, frames (~2 s at 30 fps)


def reach_vector(world: np.ndarray, side: str, mirrored: bool, hips_ok: bool) -> tuple[np.ndarray, float]:
    """(r_body, L_frame): wrist - shoulder of the user's `side` arm in body coords (fwd, left, up), and
    |upper arm| + |forearm| this frame. Same body frame as arm mode (un-mirroring, hidden-hips fallback)."""
    up, left, fwd = body_frame(world, mirrored, hips_ok)
    w = unmirror(world, image_coords=False) if mirrored else np.asarray(world, dtype=float)
    sh, el, wr = side_indices(side, False)  # labels are already in the unmirrored frame
    r = w[wr] - w[sh]
    length = float(np.linalg.norm(w[el] - w[sh]) + np.linalg.norm(w[wr] - w[el]))
    return np.array([r @ fwd, r @ left, r @ up]), length


class ArmLength:
    """User arm length: running median of the per-frame segment sum over the last ARM_LENGTH_WINDOW
    well-visible frames, so a single foreshortened / mis-tracked frame doesn't change it."""

    def __init__(self, window: int = ARM_LENGTH_WINDOW):
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, length_frame: float) -> float:
        if np.isfinite(length_frame) and length_frame > 0:
            self._buf.append(float(length_frame))
        return self.value

    @property
    def value(self) -> float:
        return float(np.median(self._buf)) if self._buf else float("nan")


def reach_target(n: np.ndarray, robot_shoulder: np.ndarray, robot_reach: float,
                 box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Gripper target (base frame): robot_shoulder + robot_reach * n, |n| clamped to 1, clipped to the box."""
    n = np.asarray(n, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm > 1.0:
        n = n / norm
    return np.clip(np.asarray(robot_shoulder, float) + robot_reach * n, box_min, box_max)
