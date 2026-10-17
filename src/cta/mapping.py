"""Calibration and absolute same-side hand -> gripper target mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cta.config import READY_Q, SIM_LIMITS
from cta.features import HandFeatures
from cta.kinematics import Kinematics


@dataclass
class Calibration:
    u0: float
    v0: float
    s0: float
    pitch0: float
    roll0: float
    p0: np.ndarray  # ready-pose gripper position, base frame (3,)
    pitch_home: float
    pinch_closed: float = 0.25
    pinch_open: float = 0.9


def _default_box() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0.1, -0.25, 0.02]), np.array([0.4, 0.25, 0.35])


@dataclass
class MapGains:
    k_lat: float = 0.30
    k_vert: float = 0.30
    k_fwd: float = 0.25
    box_min: np.ndarray = field(default_factory=lambda: _default_box()[0])
    box_max: np.ndarray = field(default_factory=lambda: _default_box()[1])

    @classmethod
    def from_kinematics(cls, kin: Kinematics, n: int = 2000, seed: int = 0) -> "MapGains":
        """Workspace box from a reach analysis of the working band.

        FK of joint samples (limits shrunk 10%) is kept where 0.02<=z<=0.25 and x>=0.10;
        per-axis 2nd/98th percentiles give the box, z is fixed to [0.02, 0.25], x/y are
        shrunk 1 cm per side, then each x/y face is pulled in by 1 cm steps until IK
        reaches the face center (z=0.10) within 1 cm.
        """
        rng = np.random.default_rng(seed)
        lo, hi = SIM_LIMITS[:, 0], SIM_LIMITS[:, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9
        qs = mid + rng.uniform(-1, 1, (n, 6)) * half
        pts = np.array([kin.fk(q)[0] for q in qs])
        pts = pts[(pts[:, 2] >= 0.02) & (pts[:, 2] <= 0.25) & (pts[:, 0] >= 0.10)]
        bmin = np.percentile(pts, 2, axis=0)
        bmax = np.percentile(pts, 98, axis=0)
        bmin[2], bmax[2] = 0.02, 0.25
        bmin[:2] += 0.01
        bmax[:2] -= 0.01
        _, pitch_home = kin.fk(READY_Q)

        def reaches(p) -> bool:
            q = kin.ik(np.asarray(p), pitch_home, READY_Q, 0.0, READY_Q[5])
            return bool(np.linalg.norm(kin.fk(q)[0] - p) <= 0.01)

        def faces():
            m = (bmin + bmax) / 2
            return {
                (0, 1): [bmax[0], 0.0, 0.10], (0, 0): [bmin[0], 0.0, 0.10],
                (1, 1): [m[0], bmax[1], 0.10], (1, 0): [m[0], bmin[1], 0.10],
            }

        for _ in range(100):
            bad = [k for k, p in faces().items() if not reaches(p)]
            if not bad:
                break
            for ax, upper in bad:
                (bmax if upper else bmin)[ax] += -0.01 if upper else 0.01
        return cls(box_min=bmin, box_max=bmax)


def _check(s0: float, closed: float, open_: float) -> None:
    if abs(s0) < 1e-9:
        raise ValueError("calibration scale s0 must be non-zero (hand not detected?)")
    if abs(open_ - closed) < 1e-9:
        raise ValueError("pinch_open and pinch_closed must differ")


def calibrate(f: HandFeatures, kin: Kinematics) -> Calibration:
    _check(f.scale, 0.25, 0.9)
    p0, pitch_home = kin.fk(READY_Q)
    return Calibration(
        u0=f.u, v0=f.v, s0=f.scale, pitch0=f.pitch, roll0=f.roll,
        p0=np.asarray(p0, dtype=float), pitch_home=float(pitch_home),
    )


def map_to_target(
    f: HandFeatures, cal: Calibration, g: MapGains
) -> tuple[np.ndarray, float, float, float]:
    y = cal.p0[1] + g.k_lat * (f.u - cal.u0)
    z = cal.p0[2] - g.k_vert * (f.v - cal.v0)
    x = cal.p0[0] + g.k_fwd * (1.0 - cal.s0 / max(f.scale, 1e-6))
    pos = np.clip([x, y, z], g.box_min, g.box_max)
    pitch = cal.pitch_home + (f.pitch - cal.pitch0)
    roll = f.roll - cal.roll0
    t = float(np.clip((f.pinch - cal.pinch_closed) / (cal.pinch_open - cal.pinch_closed), 0, 1))
    lo, hi = SIM_LIMITS[5]
    return pos, float(pitch), float(roll), float(lo + t * (hi - lo))


def save_calibration(cal: Calibration, path) -> None:
    d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(cal).items()}
    Path(path).write_text(json.dumps(d, indent=2))


def load_calibration(path) -> Calibration:
    d = json.loads(Path(path).read_text())
    d["p0"] = np.array(d["p0"], dtype=float)
    cal = Calibration(**d)
    _check(cal.s0, cal.pinch_closed, cal.pinch_open)
    return cal
