"""Hand observation -> joint command pipeline (hold-last on loss, rate-limited)."""
from __future__ import annotations

import numpy as np

from cta.config import READY_Q, SIM_LIMITS
from cta.features import HandFeatures, compute_features
from cta.filters import OneEuro
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta.mapping import Calibration, MapGains, calibrate, map_to_target


class MirrorPipeline:
    def __init__(self, kin: Kinematics, cal: Calibration | None, gains: MapGains,
                 smoother: OneEuro | None, max_step_rad: float = 0.15, aspect: float = 16 / 9):
        self.kin = kin
        self.cal = cal
        self.gains = gains
        self.smoother = smoother
        self.max_step_rad = max_step_rad
        self.aspect = aspect
        self.q_cmd = READY_Q.astype(float).copy()
        self.last_features: HandFeatures | None = None

    def reset_to(self, q: np.ndarray) -> None:
        """Seed the held command / IK warm start from the real pose."""
        self.q_cmd = np.clip(np.asarray(q, dtype=float), SIM_LIMITS[:, 0], SIM_LIMITS[:, 1])
        if self.smoother is not None:
            self.smoother.reset()

    def set_smoother(self, smoother: OneEuro | None) -> None:
        self.smoother = smoother
        if smoother is not None:
            smoother.reset()

    def _features(self, obs: HandObservation) -> HandFeatures:
        return compute_features(obs.lm_img, obs.lm_world, self.aspect)

    def calibrate(self, obs: HandObservation) -> None:
        self.cal = calibrate(self._features(obs), self.kin)
        if self.smoother is not None:
            self.smoother.reset()

    def step(self, obs: HandObservation | None, t: float) -> tuple[np.ndarray, str]:
        self.last_features = None
        if self.cal is None:
            return READY_Q.astype(float).copy(), "UNCALIBRATED"
        if obs is None:
            return self.q_cmd.copy(), "NO HAND"
        f = self._features(obs)
        pos, pitch, roll, grip = map_to_target(f, self.cal, self.gains)
        if not np.all(np.isfinite([*pos, pitch, roll, grip])):
            return self.q_cmd.copy(), "NO HAND"
        self.last_features = f
        q = self.kin.ik(pos, pitch, self.q_cmd, roll, grip)
        if self.smoother is not None:
            q = np.asarray(self.smoother(q, t), dtype=float)
        # note: smoother state tracks the unclamped target, not the rate-limited q_cmd
        q = self.q_cmd + np.clip(q - self.q_cmd, -self.max_step_rad, self.max_step_rad)
        q = np.clip(q, SIM_LIMITS[:, 0], SIM_LIMITS[:, 1])
        self.q_cmd = q
        return q.copy(), "TRACKING"
