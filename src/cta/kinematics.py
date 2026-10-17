"""Forward kinematics and damped-least-squares IK for the MuJoCo SO-101."""
import mujoco
import mujoco_menagerie as mm
import numpy as np

from cta.config import APPROACH_AXIS, JOINTS, SIM_LIMITS

_EPS = 1e-4
_LAMBDA = 0.05
_MAX_ITERS = 15
_PITCH_WEIGHT = 0.05  # m/rad
_ARM = 4  # IK solves joints 0..3


class Kinematics:
    def __init__(self):
        self._m = mm.load("robotstudio_so101")
        self._d = mujoco.MjData(self._m)  # private; fully overwritten each call
        obj = mujoco.mjtObj
        self._base = mujoco.mj_name2id(self._m, obj.mjOBJ_SITE, "baseframe")
        self._grip = mujoco.mj_name2id(self._m, obj.mjOBJ_SITE, "gripperframe")
        self._adr = np.array([
            self._m.jnt_qposadr[mujoco.mj_name2id(self._m, obj.mjOBJ_JOINT, n)]
            for n in JOINTS
        ])

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """Gripper site position in the base frame and approach pitch (rad, + = up)."""
        d = self._d
        d.qpos[:] = 0.0
        d.qpos[self._adr] = q
        mujoco.mj_kinematics(self._m, d)
        rb = d.site_xmat[self._base].reshape(3, 3)
        p = rb.T @ (d.site_xpos[self._grip] - d.site_xpos[self._base])
        axis = rb.T @ d.site_xmat[self._grip].reshape(3, 3)[:, APPROACH_AXIS]
        return p, float(np.arctan2(axis[2], np.hypot(axis[0], axis[1])))

    def _pose_vec(self, q):
        p, pitch = self.fk(q)
        return np.concatenate([p, [_PITCH_WEIGHT * pitch]])

    def ik(self, target_pos: np.ndarray, target_pitch: float, q_init: np.ndarray,
           wrist_roll: float, gripper: float) -> np.ndarray:
        lo, hi = SIM_LIMITS[:, 0], SIM_LIMITS[:, 1]
        q = np.clip(np.asarray(q_init, dtype=float).copy(), lo, hi)
        q[4] = np.clip(wrist_roll, lo[4], hi[4])
        q[5] = np.clip(gripper, lo[5], hi[5])
        target = np.concatenate([np.asarray(target_pos, dtype=float),
                                 [_PITCH_WEIGHT * target_pitch]])
        for _ in range(_MAX_ITERS):
            x0 = self._pose_vec(q)
            J = np.zeros((4, _ARM))
            for j in range(_ARM):
                qj = q.copy()
                qj[j] += _EPS
                J[:, j] = (self._pose_vec(qj) - x0) / _EPS
            dq = J.T @ np.linalg.solve(J @ J.T + _LAMBDA**2 * np.eye(4), target - x0)
            q[:_ARM] = np.clip(q[:_ARM] + dq, lo[:_ARM], hi[:_ARM])
        return q
