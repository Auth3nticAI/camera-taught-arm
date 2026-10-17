"""MuJoCo digital twin rendered offscreen."""
from __future__ import annotations

import mujoco
import mujoco_menagerie as mm
import numpy as np

from cta.config import JOINTS


class Twin:
    def __init__(self, width: int = 640, height: int = 480):
        self._m = mm.load("robotstudio_so101", "scene")
        self._d = mujoco.MjData(self._m)
        self._adr = np.array([
            self._m.jnt_qposadr[mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINTS
        ])
        self._r = mujoco.Renderer(self._m, height=height, width=width)
        self._cam = mujoco.MjvCamera()
        self._cam.lookat[:] = (0.15, 0.0, 0.12)
        self._cam.distance = 0.9
        self._cam.azimuth = 150.0
        self._cam.elevation = -20.0

    def render(self, q: np.ndarray) -> np.ndarray:
        self._d.qpos[self._adr] = q
        mujoco.mj_forward(self._m, self._d)
        self._r.update_scene(self._d, camera=self._cam)
        return self._r.render().copy()

    def close(self) -> None:
        self._r.close()
