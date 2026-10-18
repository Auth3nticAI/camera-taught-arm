"""Follower-only sim<->real alignment from table-touch poses.

The limp follower is placed with its closed jaws touching the table in several poses. In each pose
the lowest point of the gripper/jaw meshes must sit at the table height z0 (base frame), so the
small calibration offsets of shoulder_lift, elbow_flex and wrist_flex (which change that height)
are found by least squares. shoulder_pan and wrist_roll don't move the contact point vertically,
so their offsets stay 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_menagerie as mm
import numpy as np

from cta.config import JOINTS

SOLVED = (1, 2, 3)  # shoulder_lift, elbow_flex, wrist_flex
MAX_OFFSET_DEG = 15.0
MAX_Z0 = 0.05
MIN_POSES = 6
_CONTACT_BODIES = ("gripper", "moving_jaw_so101_v1")


class ContactModel:
    """Lowest point of the gripper + moving-jaw meshes, in the base frame, for a sim pose."""

    def __init__(self) -> None:
        self.m = mm.load("robotstudio_so101")
        self.d = mujoco.MjData(self.m)
        self._qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
        self._base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "baseframe")
        bodies = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in _CONTACT_BODIES}
        self._geoms = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] in bodies and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                self._geoms.append((g, self.m.mesh_vert[a:a + n].copy()))

    def lowest_z(self, q: np.ndarray) -> float:
        self.d.qpos[:] = 0.0
        for adr, v in zip(self._qadr, q):
            self.d.qpos[adr] = v
        mujoco.mj_kinematics(self.m, self.d)
        rb = self.d.site_xmat[self._base].reshape(3, 3)
        pb = self.d.site_xpos[self._base]
        z = np.inf
        for g, verts in self._geoms:
            world = self.d.geom_xpos[g] + verts @ self.d.geom_xmat[g].reshape(3, 3).T
            z = min(z, float(((world - pb) @ rb)[:, 2].min()))
        return z


@dataclass
class AlignResult:
    offset_deg: np.ndarray  # (5,) to store in SimRealMap.offset_deg (sign assumed +1)
    z0: float               # table height in the base frame (m)
    rms_before: float       # contact-height RMS (m) with zero offsets, z0 fit
    rms_after: float
    residuals_m: np.ndarray


def _residuals(params: np.ndarray, real_deg: np.ndarray, grip_q: np.ndarray, model: ContactModel) -> np.ndarray:
    off = np.zeros(5)
    off[list(SOLVED)] = params[:3]
    out = np.empty(len(real_deg))
    for i, (deg, g) in enumerate(zip(real_deg, grip_q)):
        q = np.append(np.radians(deg - off), g)
        out[i] = model.lowest_z(q) - params[3]
    return out


def solve_offsets(real_deg: np.ndarray, grip_q: np.ndarray, model: ContactModel, iters: int = 60) -> AlignResult:
    """real_deg: (N,5) LeRobot degrees for joints 1-5 at each touch pose; grip_q: (N,) sim gripper rad."""
    real_deg = np.asarray(real_deg, dtype=float)
    grip_q = np.asarray(grip_q, dtype=float)
    if len(real_deg) < MIN_POSES:
        raise ValueError(f"need at least {MIN_POSES} touch poses, got {len(real_deg)}")
    lo = np.array([-MAX_OFFSET_DEG] * 3 + [-MAX_Z0])
    hi = -lo

    def r(p):
        return _residuals(p, real_deg, grip_q, model)

    p = np.zeros(4)
    p[3] = float(np.mean(r(p)))  # best table height with zero offsets
    rms_before = float(np.sqrt(np.mean(r(p) ** 2)))
    steps = np.array([1e-2, 1e-2, 1e-2, 1e-5])
    mu = 1e-3
    res = r(p)
    for _ in range(iters):  # Levenberg-Marquardt with a numerical Jacobian and box bounds
        jac = np.stack([(r(p + np.eye(4)[k] * steps[k]) - res) / steps[k] for k in range(4)], axis=1)
        jtj, jtr = jac.T @ jac, jac.T @ res
        delta = np.linalg.solve(jtj + mu * np.diag(np.diag(jtj) + 1e-12), -jtr)
        cand = np.clip(p + delta, lo, hi)
        cand_res = r(cand)
        if cand_res @ cand_res < res @ res:
            p, res, mu = cand, cand_res, mu * 0.3
            if np.max(np.abs(delta[:3])) < 1e-4:
                break
        else:
            mu *= 10.0
    off = np.zeros(5)
    off[list(SOLVED)] = p[:3]
    return AlignResult(off, float(p[3]), rms_before, float(np.sqrt(np.mean(res ** 2))), res)
