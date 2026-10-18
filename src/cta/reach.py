"""--control reach (Tray, 2026-10-18): the user's arm is only the WHERE. Pure numpy.

How "where" is measured (spec 2026-10-18-reach-measurement-fix, from recorded data):
  - The arm is LOCKED at calibration (the Pose side whose wrist is nearest the hand tracker's wrist) and never
    re-picked per frame (re-picking switched sides 33-131 times per take).
  - The hand position comes from the HAND TRACKER (wrist or palm centre, image coords): the Pose wrist
    teleports ~0.15 image widths in one frame while the hand tracker moves 0.016-0.05.
  - Anchor: the locked side's Pose shoulder (image); scale: the image shoulder separation (aspect-corrected),
    so n is in "arm lengths" = ARM_PER_SHOULDER shoulder widths.
  - Forward: metric depth (Depth Anything V2) hand minus shoulder, divided by ARM_PER_SHOULDER x the Pose
    WORLD shoulder width; without a depth model, the hand-scale cue (fwd_from_hand_scale).
n = (fwd, left, up) in the user's body terms. Robot target = p0 + ROBOT_REACH * (n - n_home) (relative,
the CLI default) or ROBOT_SHOULDER + ROBOT_REACH * n (absolute, reach_target), clipped to the box.
"""
from __future__ import annotations

from collections import deque

import numpy as np

REACH_FRACTION = 0.95             # ROBOT_REACH = this x the robot's measured max reach (Kinematics.max_reach)
NO_HAND_PITCH_RAD = -np.pi / 4    # claw pitch with no hand: ~45 deg down
ARM_PER_SHOULDER = 1.7            # human arm length (shoulder -> wrist) / shoulder width
SCALE_WINDOW_S = 0.5              # median window for the shoulder separation / width / shoulder depth
FALLBACK_FOCAL_H = 1.2            # hand-scale fallback only: assumed webcam focal length in image heights
                                  # (~45 deg vertical FOV); the depth model doesn't need it
PALM_IDX = (0, 5, 9, 13, 17)      # wrist + the four MCPs


def palm_centre(lm_img: np.ndarray) -> np.ndarray:
    """Mean image position (u, v) of the wrist and the four MCP landmarks."""
    return np.asarray(lm_img, dtype=float)[list(PALM_IDX), :2].mean(axis=0)


def shoulder_sep(pose_img: np.ndarray, aspect: float) -> float:
    """Pose shoulder separation in the image, in image-HEIGHT units (u scaled by the aspect ratio)."""
    d = np.asarray(pose_img, dtype=float)[11, :2] - np.asarray(pose_img, dtype=float)[12, :2]
    return float(np.hypot(d[0] * aspect, d[1]))


def plane_reach(hand_uv, shoulder_uv, sep: float, aspect: float, mirrored: bool) -> np.ndarray:
    """(n_left, n_up): hand - shoulder in the image, in arm lengths (ARM_PER_SHOULDER x sep), user's terms.

    Unmirrored camera: the user faces the camera, so their left is image +u; a mirrored image flips that.
    Up is image -v in both."""
    du = (float(hand_uv[0]) - float(shoulder_uv[0])) * aspect
    dv = float(hand_uv[1]) - float(shoulder_uv[1])
    arm = ARM_PER_SHOULDER * sep
    return np.array([(-du if mirrored else du) / arm, -dv / arm])


def fwd_from_depth(d: float, d_home: float, shoulder_width_m: float) -> float:
    """n_fwd from d = depth_hand - depth_shoulder (m): closer to the camera than at home = forward."""
    return float(-(d - d_home) / (ARM_PER_SHOULDER * shoulder_width_m))


def fwd_from_hand_scale(ratio: float, sep_home: float, focal_h: float = FALLBACK_FOCAL_H) -> float:
    """Fallback n_fwd (relative to home) from ratio = (hand scale / shoulder sep) / (the same at home).

    Pinhole: the shoulders sit at Z = focal_h * w / sep_home (w = shoulder width), the hand starts at about
    that depth; a hand that looks `ratio` x bigger is at Z / ratio, i.e. Z (1 - 1/ratio) closer."""
    return float(focal_h / sep_home * (1.0 - 1.0 / ratio) / ARM_PER_SHOULDER)


class WindowMedian:
    """Median of the samples from the last `window_s` seconds (pruned when a sample arrives, so the value
    holds while no samples come in)."""

    def __init__(self, window_s: float = SCALE_WINDOW_S):
        self.window_s = window_s
        self._buf: deque[tuple[float, float]] = deque()

    def update(self, value: float, t: float) -> float:
        if np.isfinite(value):
            self._buf.append((float(t), float(value)))
            while self._buf and self._buf[0][0] < t - self.window_s:
                self._buf.popleft()
        return self.value

    @property
    def value(self) -> float:
        return float(np.median([v for _, v in self._buf])) if self._buf else float("nan")


def reach_target(n: np.ndarray, robot_shoulder: np.ndarray, robot_reach: float,
                 box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Gripper target (base frame): robot_shoulder + robot_reach * n, |n| clamped to 1, clipped to the box."""
    n = np.asarray(n, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm > 1.0:
        n = n / norm
    return np.clip(np.asarray(robot_shoulder, float) + robot_reach * n, box_min, box_max)
