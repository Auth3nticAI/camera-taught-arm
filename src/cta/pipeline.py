"""Hand observation -> joint command pipeline (hold-last on loss, rate-limited)."""
from __future__ import annotations

import numpy as np

from cta.config import READY_Q, SIM_LIMITS
from cta.features import HandFeatures, compute_features
from cta.filters import OneEuro
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta.mapping import Calibration, MapGains, calibrate, map_to_target, set_pinch_closed

# Snap gripper (Tray's decision 2026-10-17): pinch fraction below CLOSE -> closed, above OPEN -> open,
# in between keep the previous state (dead-zone stops flicker from noisy fingertip landmarks).
GRIP_CLOSE_FRAC = 0.35
GRIP_OPEN_FRAC = 0.55
COAST_S = 0.15  # hand lost for less than this: report COAST (arm keeps moving to the last target)
GRIP_STEP_RAD = 0.6  # per-frame gripper limit: full travel in ~4 frames, independent of the arm limit


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
        self._grip_closed = False
        self._last_seen_t: float | None = None

    def _grip_state(self, frac: float) -> bool:
        """Hysteresis on the pinch fraction (0 = pinched, 1 = open hand); returns True if closed."""
        if frac < GRIP_CLOSE_FRAC:
            self._grip_closed = True
        elif frac > GRIP_OPEN_FRAC:
            self._grip_closed = False
        return self._grip_closed

    def set_pinch_closed(self, obs: HandObservation) -> None:
        if self.cal is None:
            raise ValueError("calibrate (open hand, 'c') before setting the pinch")
        self.cal = set_pinch_closed(self.cal, self._features(obs))

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
        f = self._features(obs) if obs is not None else None
        if self.cal is None:
            self.last_features = f  # recorded even uncalibrated (paired takes still need them)
            return READY_Q.astype(float).copy(), "UNCALIBRATED"
        if obs is None:
            if self._last_seen_t is not None and t - self._last_seen_t <= COAST_S:
                return self.q_cmd.copy(), "COAST"
            return self.q_cmd.copy(), "NO HAND"
        pos, pitch, roll, grip = map_to_target(f, self.cal, self.gains)
        if not np.all(np.isfinite([*pos, pitch, roll, grip])):
            return self.q_cmd.copy(), "NO HAND"
        self.last_features = f
        self._last_seen_t = t
        lo, hi = SIM_LIMITS[5]
        grip = lo if self._grip_state((grip - lo) / (hi - lo)) else hi
        q = self.kin.ik(pos, pitch, self.q_cmd, roll, grip)
        arm = q[:5]
        if self.smoother is not None:  # arm joints only; the gripper must stay snappy
            arm = np.asarray(self.smoother(arm, t), dtype=float)
        # note: smoother state tracks the unclamped target, not the rate-limited q_cmd
        arm = self.q_cmd[:5] + np.clip(arm - self.q_cmd[:5], -self.max_step_rad, self.max_step_rad)
        g = self.q_cmd[5] + np.clip(grip - self.q_cmd[5], -GRIP_STEP_RAD, GRIP_STEP_RAD)
        q = np.clip(np.append(arm, g), SIM_LIMITS[:, 0], SIM_LIMITS[:, 1])
        self.q_cmd = q
        return q.copy(), "TRACKING"
