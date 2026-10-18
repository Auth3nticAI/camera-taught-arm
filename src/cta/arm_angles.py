"""User's shoulder/elbow (MediaPipe Pose) -> SO-101 shoulder_pan, shoulder_lift, elbow_flex. Pure numpy.

Mirrored input is un-mirrored first (image x -> 1-x, world x -> -x, left/right landmark labels swapped), so
everything below works on an unmirrored view of the user. Body frame axes are (fwd, left, up); the robot
faces the same way as the user, so they map onto the robot base axes (x, y, z) one to one.

Offsets (measured from the MuJoCo SO-101 joint anchors, checked in tests/test_arm_angles.py):
  - shoulder_lift = 0 tilts the upper arm (lift->elbow joint) 13.97 deg forward of vertical, so
    robot lift = (user's upper-arm angle from vertical) - LIFT_OFFSET.
  - elbow_flex = 0 bends the forearm 73.82 deg from the upper-arm line (not 90), so
    robot elbow = (user's elbow flexion) - ELBOW_OFFSET.
"""
from __future__ import annotations

import numpy as np

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
_LR_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
             (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32))
_SWAP = np.arange(33)
for _i, _j in _LR_PAIRS:
    _SWAP[_i], _SWAP[_j] = _j, _i
_SIDE = {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}

LIFT_OFFSET_RAD = float(np.radians(13.968))
ELBOW_OFFSET_RAD = float(np.radians(73.825))
# azimuth is undefined for a near-vertical upper arm. Hysteresis on |horizontal part| / |upper arm|:
# a valid pan turns invalid below PAN_INVALID_FRAC, an invalid one turns valid again above PAN_VALID_FRAC
PAN_INVALID_FRAC = 0.30
PAN_VALID_FRAC = 0.40
PAN_MAX_RAD = 1.91986 - 0.087  # shoulder_pan limit (MuJoCo model, 110 deg) minus the 5 deg command margin
_CAM_UP = np.array([0.0, -1.0, 0.0])


def unmirror(lm: np.ndarray, image_coords: bool) -> np.ndarray:
    """Pose landmarks from a mirrored image -> the same body seen unmirrored."""
    out = np.array(lm, dtype=float)[_SWAP]
    out[:, 0] = 1.0 - out[:, 0] if image_coords else -out[:, 0]
    return out


def side_indices(side: str, mirrored: bool) -> tuple[int, int, int]:
    """(shoulder, elbow, wrist) landmark indices of the user's `side` arm in the RAW (camera) landmarks."""
    return tuple(int(_SWAP[i]) for i in _SIDE[side]) if mirrored else _SIDE[side]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _frame(w: np.ndarray, hips_ok: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = w[L_SH] - w[R_SH]
    if hips_ok:
        up = _unit((w[L_SH] + w[R_SH]) / 2 - (w[L_HIP] + w[R_HIP]) / 2)
    else:  # hips hidden (e.g. seated at a desk): camera up, made perpendicular to the shoulder line
        s = _unit(left)
        up = _unit(_CAM_UP - s * (_CAM_UP @ s))
    left = _unit(left - up * (left @ up))
    fwd = _unit(np.cross(left, up))
    if fwd @ np.array([0.0, 0.0, -1.0]) < 0:  # the chest faces the camera (camera z points away)
        fwd = -fwd
    # re-derive left from up x fwd: with the user facing the camera this is their true left even when
    # Pose swaps its shoulder labels, and it keeps the frame right-handed
    return up, np.cross(up, fwd), fwd


def body_frame(world: np.ndarray, mirrored: bool, hips_ok: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(up, left, fwd) unit vectors in unmirrored camera axes, from Pose world landmarks.

    hips_ok=False: the hip landmarks are guesses; `up` = camera up orthogonalized against the shoulders."""
    w = unmirror(world, image_coords=False) if mirrored else np.asarray(world, dtype=float)
    return _frame(w, hips_ok)


def choose_arm_side(pose_img: np.ndarray, hand_wrist_uv, last_side: str | None, want: str,
                    mirrored: bool) -> str:
    """The user's arm whose Pose wrist is nearest the hand tracker's wrist (both in camera image coords).

    No hand: keep `last_side`; no history: `want` (the --hand flag). Labels are in the unmirrored frame.
    """
    if hand_wrist_uv is None:
        return last_side if last_side is not None else want
    uv = np.asarray(hand_wrist_uv, dtype=float)[:2]
    d = {s: float(np.hypot(*(np.asarray(pose_img, float)[side_indices(s, mirrored)[2], :2] - uv)))
         for s in ("left", "right")}
    return min(d, key=d.get)


def arm_joint_angles(world: np.ndarray, side: str, mirrored: bool, prev_pan: float,
                     prev_valid: bool = True, hips_ok: bool = True) -> tuple[float, float, float, bool]:
    """(pan, lift, elbow, pan_valid) in robot radians for the user's `side` arm (lift/elbow unclamped).

    pan = -azimuth (azimuth: 0 = forward, + = toward the user's left; +pan swings the robot gripper
    toward its right, -y), unwrapped toward prev_pan. pan_valid False (and pan = prev_pan) when:
      - the upper arm is near vertical (azimuth undefined; hysteresis via prev_valid), or
      - the unwrapped pan is beyond the pan range (upper arm pointing behind the torso, where azimuth
        noise across +-180 deg would otherwise flip the base by ~200 deg).
    """
    w = unmirror(world, image_coords=False) if mirrored else np.asarray(world, dtype=float)
    up, left, fwd = _frame(w, hips_ok)
    sh, el, wr = _SIDE[side]
    a, b = w[el] - w[sh], w[wr] - w[el]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    lift = float(np.arccos(np.clip(a @ up / na, -1.0, 1.0))) - LIFT_OFFSET_RAD
    flex = float(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0)))
    elbow = flex - ELBOW_OFFSET_RAD
    af, al = float(a @ fwd), float(a @ left)
    if np.hypot(af, al) < (PAN_INVALID_FRAC if prev_valid else PAN_VALID_FRAC) * na:
        return float(prev_pan), lift, elbow, False
    pan = float(prev_pan + (-np.arctan2(al, af) - prev_pan + np.pi) % (2 * np.pi) - np.pi)
    if abs(pan) > PAN_MAX_RAD:
        return float(prev_pan), lift, elbow, False
    return pan, lift, elbow, True
