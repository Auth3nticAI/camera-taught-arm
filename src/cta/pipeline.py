"""Hand / arm-pose observation -> joint command pipelines (hold-last on loss, rate-limited)."""
from __future__ import annotations

import numpy as np

from cta.arm_angles import ELBOW_OFFSET_RAD, arm_joint_angles, choose_arm_side
from cta.config import READY_Q, SIM_LIMITS
from cta.features import HandFeatures, compute_features
from cta.filters import OneEuro
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta.mapping import Calibration, MapGains, calibrate, map_to_target, set_pinch_closed
from cta.robot_io import clamp_margin

# Snap gripper (Tray's decision 2026-10-17): pinch fraction below CLOSE -> closed, above OPEN -> open,
# in between keep the previous state (dead-zone stops flicker from noisy fingertip landmarks).
GRIP_CLOSE_FRAC = 0.35
GRIP_OPEN_FRAC = 0.55
COAST_S = 0.15  # hand lost for less than this: report COAST (arm keeps moving to the last target)
GRIP_STEP_RAD = 0.6  # per-frame gripper limit: full travel in ~4 frames, independent of the arm limit
TABLE_CLEARANCE_M = 0.02  # arm-pose mode: lowest gripper/jaw point stays this far above the table


class _PipelineBase:
    """Shared by both control modes: calibration, snap gripper, COAST/NO HAND, smoothing + rate limit."""

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

    def _snap_grip(self, grip: float) -> float:
        lo, hi = SIM_LIMITS[5]
        return lo if self._grip_state((grip - lo) / (hi - lo)) else hi

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

    def _lost(self, t: float) -> tuple[np.ndarray, str]:
        if self._last_seen_t is not None and t - self._last_seen_t <= COAST_S:
            return self.q_cmd.copy(), "COAST"
        return self.q_cmd.copy(), "NO HAND"

    def _finish(self, arm: np.ndarray, grip: float, t: float) -> np.ndarray:
        """Smooth (arm joints only; the gripper must stay snappy), rate-limit, clip; becomes q_cmd."""
        if self.smoother is not None:
            arm = np.asarray(self.smoother(arm, t), dtype=float)
        # note: smoother state tracks the unclamped target, not the rate-limited q_cmd
        arm = self.q_cmd[:5] + np.clip(arm - self.q_cmd[:5], -self.max_step_rad, self.max_step_rad)
        g = self.q_cmd[5] + np.clip(grip - self.q_cmd[5], -GRIP_STEP_RAD, GRIP_STEP_RAD)
        self.q_cmd = np.clip(np.append(arm, g), SIM_LIMITS[:, 0], SIM_LIMITS[:, 1])
        return self.q_cmd


class MirrorPipeline(_PipelineBase):
    """--control hand: the hand's image position/size drives the gripper position through IK."""

    def step(self, obs: HandObservation | None, t: float) -> tuple[np.ndarray, str]:
        self.last_features = None
        f = self._features(obs) if obs is not None else None
        if self.cal is None:
            self.last_features = f  # recorded even uncalibrated (paired takes still need them)
            return READY_Q.astype(float).copy(), "UNCALIBRATED"
        if obs is None:
            return self._lost(t)
        pos, pitch, roll, grip = map_to_target(f, self.cal, self.gains)
        if not np.all(np.isfinite([*pos, pitch, roll, grip])):
            return self.q_cmd.copy(), "NO HAND"
        self.last_features = f
        self._last_seen_t = t
        grip = self._snap_grip(grip)
        q = self.kin.ik(pos, pitch, self.q_cmd, roll, grip)
        return self._finish(q[:5], grip, t).copy(), "TRACKING"


def _bisect_joint(q: np.ndarray, joint: int, safe: float, ok, iters: int) -> np.ndarray:
    """Move q[joint] from its (failing) value toward `safe` (passing) just far enough for ok(q)."""
    out = q.copy()
    bad, good = q[joint], safe
    for _ in range(iters):
        out[joint] = (bad + good) / 2
        if ok(out):
            good = out[joint]
        else:
            bad = out[joint]
    out[joint] = good
    return out


def clear_table(q: np.ndarray, contact, clearance: float = TABLE_CLEARANCE_M, iters: int = 20) -> np.ndarray:
    """Keep the lowest gripper/jaw point at least `clearance` above the table.

    Too low: move shoulder_lift toward 0 (raises the arm) by bisection, just far enough. If even lift 0
    is too low (elbow folded down with the upper arm vertical), also unfold elbow_flex toward straight up
    (-ELBOW_OFFSET), which always clears."""
    q = np.asarray(q, dtype=float)

    def ok(x):
        return contact.lowest_z(x) >= contact.table_z + clearance

    if ok(q):
        return q
    up = q.copy()
    up[1] = 0.0
    if ok(up):
        return _bisect_joint(q, 1, 0.0, ok, iters)
    straight = max(-ELBOW_OFFSET_RAD, SIM_LIMITS[2, 0])
    return _bisect_joint(up, 2, straight, ok, iters)


class ArmPosePipeline(_PipelineBase):
    """--control arm (decision D-O): the user's shoulder/elbow (MediaPipe Pose) drive shoulder_pan,
    shoulder_lift and elbow_flex joint for joint; the hand still drives wrist_flex (world pitch),
    wrist_roll and the snap gripper. No pose = COAST / NO HAND. No hand: wrist and gripper hold."""

    def __init__(self, kin: Kinematics, cal: Calibration | None, gains: MapGains,
                 smoother: OneEuro | None, max_step_rad: float = 0.15, aspect: float = 16 / 9,
                 mirrored: bool = False, want: str = "left"):
        super().__init__(kin, cal, gains, smoother, max_step_rad, aspect)
        from cta.align import ContactModel  # loads its own MuJoCo model; only needed in this mode
        self.contact = ContactModel()
        self.mirrored = mirrored
        self.want = want
        self.side: str | None = None
        self._pan = float(self.q_cmd[0])

    def reset_to(self, q: np.ndarray) -> None:
        super().reset_to(q)
        self._pan = float(self.q_cmd[0])

    def _wrist_flex_for_pitch(self, q: np.ndarray, pitch: float) -> float:
        """wrist_flex giving world approach pitch `pitch` for this lift/elbow (planar chain: affine, slope +-1)."""
        q0, q1 = q.copy(), q.copy()
        q0[3], q1[3] = 0.0, 0.5
        p0 = self.kin.planar_pitch(q0)
        slope = np.sign((self.kin.planar_pitch(q1) - p0 + np.pi) % (2 * np.pi) - np.pi)
        err = (pitch - p0 + np.pi) % (2 * np.pi) - np.pi
        lo, hi = SIM_LIMITS[3]
        return float(np.clip(slope * err, lo, hi))

    def step(self, obs: HandObservation | None, t: float, pose_obs=None) -> tuple[np.ndarray, str]:
        self.last_features = None
        f = self._features(obs) if obs is not None else None
        if self.cal is None:
            self.last_features = f
            return READY_Q.astype(float).copy(), "UNCALIBRATED"
        if pose_obs is None or not (np.all(np.isfinite(pose_obs.world)) and np.all(np.isfinite(pose_obs.img))):
            return self._lost(t)
        wrist_uv = obs.lm_img[0, :2] if obs is not None else None
        side = choose_arm_side(pose_obs.img, wrist_uv, self.side, self.want, self.mirrored)
        pan, lift, elbow, pan_valid = arm_joint_angles(pose_obs.world, side, self.mirrored, self._pan)
        if not np.all(np.isfinite([pan, lift, elbow])):
            return self._lost(t)
        self.side = side
        self._last_seen_t = t
        if pan_valid:
            self._pan = pan
        held = self.q_cmd[3:].copy()  # hand missing: wrist_flex, wrist_roll and gripper hold their last values
        q = np.append(clamp_margin(np.append([pan, lift, elbow], held))[:3], held)
        hand = map_to_target(f, self.cal, self.gains) if f is not None else None
        tracking_hand = hand is not None and bool(np.all(np.isfinite(hand[1:])))
        if tracking_hand:
            _, pitch, roll, grip = hand
            self.last_features = f
            q[3] = self._wrist_flex_for_pitch(q, pitch)
            q[4] = np.clip(roll, *SIM_LIMITS[4])
            q[5] = self._snap_grip(grip)
        self._finish(q[:5], q[5], t)
        if not tracking_hand:
            self.q_cmd[3:] = held  # exact hold (the arm smoother could otherwise drift the wrist)
        self.q_cmd = clear_table(self.q_cmd, self.contact)
        return self.q_cmd.copy(), "TRACKING"
