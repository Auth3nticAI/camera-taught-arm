"""Follower-only sim<->real alignment from table-touch poses.

The limp follower is placed with its closed jaws touching the table in several poses. In each pose
the lowest point of the gripper/jaw meshes must sit at the table height z0 (base frame), so the
small calibration offsets of shoulder_lift, elbow_flex and wrist_flex (which change that height)
are found by least squares. shoulder_pan and wrist_roll don't move the contact point vertically,
so their offsets stay 0.

The table height is NOT a free parameter: the base is clamped flat on the table, so the table plane
is the bottom of the base mesh (ContactModel.table_z). Freeing it let lift/wrist offsets trade off
against it and run to their bounds on a real 6-pose capture (2026-10-17). A weak prior also pulls
offsets toward 0 (PRIOR_MM_PER_DEG), and per-joint standard deviations flag poorly varied poses.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_menagerie as mm
import numpy as np

from cta.config import JOINTS

SOLVED = (1, 2, 3)  # shoulder_lift, elbow_flex, wrist_flex
MAX_OFFSET_DEG = 15.0
MIN_POSES = 6
PRIOR_MM_PER_DEG = 1.0 / 3.0  # each degree of offset costs like 1/3 mm of contact error
MAX_TRUSTED_STD_DEG = 2.0
_CONTACT_BODIES = ("gripper", "moving_jaw_so101_v1")


_EXTREMES_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _extremes(verts: np.ndarray, n_dirs: int) -> np.ndarray:
    """Vertices that are lowest along at least one of n_dirs Fibonacci-sphere directions (a hull subset)."""
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * i / n_dirs)
    th = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
    return verts[np.unique(np.argmin(verts @ dirs.T, axis=0))]


class ContactModel:
    """Lowest point of the gripper + moving-jaw meshes (or the given bodies' meshes), in the base frame."""

    def __init__(self, bodies: tuple[str, ...] = _CONTACT_BODIES, extreme_dirs: int = 0) -> None:
        """extreme_dirs > 0: keep only each mesh's extreme vertices along that many directions (fast
        approximate lowest_z for per-frame use; exact when 0, the default used by the alignment solve)."""
        self.m = mm.load("robotstudio_so101")
        self.d = mujoco.MjData(self.m)
        self._qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
        self._base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "baseframe")
        self.bodies = tuple(bodies)
        bodies = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
        if -1 in bodies:
            raise ValueError(f"unknown body in {self.bodies}")
        self._geoms = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] in bodies and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                verts = self.m.mesh_vert[a:a + n].copy()
                if extreme_dirs:
                    key = (int(mid), extreme_dirs)
                    if key not in _EXTREMES_CACHE:
                        _EXTREMES_CACHE[key] = _extremes(verts, extreme_dirs)
                    verts = _EXTREMES_CACHE[key]
                self._geoms.append((g, verts))
        base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._base_geoms = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] == base and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                self._base_geoms.append((g, self.m.mesh_vert[a:a + n].copy()))
        self.table_z = self._lowest(np.zeros(len(JOINTS)), self._base_geoms)

    def lowest_z(self, q: np.ndarray) -> float:
        return self._lowest(q, self._geoms)

    def _lowest(self, q: np.ndarray, geoms) -> float:
        self.d.qpos[:] = 0.0
        for adr, v in zip(self._qadr, q):
            self.d.qpos[adr] = v
        mujoco.mj_kinematics(self.m, self.d)
        rb = self.d.site_xmat[self._base].reshape(3, 3)
        pb = self.d.site_xpos[self._base]
        z = np.inf
        for g, verts in geoms:
            world = self.d.geom_xpos[g] + verts @ self.d.geom_xmat[g].reshape(3, 3).T
            z = min(z, float(((world - pb) @ rb)[:, 2].min()))
        return z


@dataclass
class AlignResult:
    offset_deg: np.ndarray  # (5,) to store in SimRealMap.offset_deg (sign assumed +1)
    z0: float               # table height in the base frame (m) - fixed, from the base mesh
    rms_before: float       # contact-height RMS (m) with zero offsets
    rms_after: float
    residuals_m: np.ndarray
    std_deg: np.ndarray     # (3,) 1-sigma uncertainty of lift/elbow/wrist_flex offsets

    @property
    def trusted(self) -> bool:
        return bool(np.all(self.std_deg <= MAX_TRUSTED_STD_DEG)) and bool(
            np.all(np.abs(self.offset_deg) < MAX_OFFSET_DEG - 0.1)
        )


def _contact_residuals(off3: np.ndarray, real_deg: np.ndarray, grip_q: np.ndarray, model: ContactModel) -> np.ndarray:
    off = np.zeros(5)
    off[list(SOLVED)] = off3
    out = np.empty(len(real_deg))
    for i, (deg, g) in enumerate(zip(real_deg, grip_q)):
        out[i] = model.lowest_z(np.append(np.radians(deg - off), g)) - model.table_z
    return out


def solve_offsets(real_deg: np.ndarray, grip_q: np.ndarray, model: ContactModel, iters: int = 60) -> AlignResult:
    """real_deg: (N,5) LeRobot degrees for joints 1-5 at each touch pose; grip_q: (N,) sim gripper rad."""
    real_deg = np.asarray(real_deg, dtype=float)
    grip_q = np.asarray(grip_q, dtype=float)
    if len(real_deg) < MIN_POSES:
        raise ValueError(f"need at least {MIN_POSES} touch poses, got {len(real_deg)}")
    prior_w = PRIOR_MM_PER_DEG / 1000.0  # metres of residual per degree of offset

    def contact(p):
        return _contact_residuals(p, real_deg, grip_q, model)

    def full(p):  # data residuals + weak prior pulling offsets toward 0
        return np.concatenate([contact(p), prior_w * p])

    p = np.zeros(3)
    rms_before = float(np.sqrt(np.mean(contact(p) ** 2)))
    step = 1e-2
    mu = 1e-3
    res = full(p)
    jac = None
    for _ in range(iters):  # Levenberg-Marquardt with a numerical Jacobian and box bounds
        jac = np.stack([(full(p + np.eye(3)[k] * step) - res) / step for k in range(3)], axis=1)
        jtj, jtr = jac.T @ jac, jac.T @ res
        delta = np.linalg.solve(jtj + mu * np.diag(np.diag(jtj) + 1e-12), -jtr)
        cand = np.clip(p + delta, -MAX_OFFSET_DEG, MAX_OFFSET_DEG)
        cand_res = full(cand)
        if cand_res @ cand_res < res @ res:
            p, res, mu = cand, cand_res, mu * 0.3
            if np.max(np.abs(delta)) < 1e-4:
                break
        else:
            mu *= 10.0
    data_res = contact(p)
    dof = max(len(data_res) - 3, 1)
    sigma2 = max(float(data_res @ data_res) / dof, (0.5e-3) ** 2)  # floor: ~0.5 mm touch noise
    jac = np.stack([(full(p + np.eye(3)[k] * step) - full(p)) / step for k in range(3)], axis=1)
    std = np.sqrt(np.diag(np.linalg.inv(jac.T @ jac)) * sigma2)
    off = np.zeros(5)
    off[list(SOLVED)] = p
    return AlignResult(off, model.table_z, rms_before, float(np.sqrt(np.mean(data_res ** 2))), data_res, std)
