"""Hand / arm-pose observation -> joint command pipelines (hold-last on loss, rate-limited)."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from cta.arm_angles import ELBOW_OFFSET_RAD, arm_joint_angles, choose_arm_side, side_indices
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
TABLE_CLEARANCE_M = 0.005  # D-Q (Tray): lowest point of CLEARANCE_BODIES stays this far above the table
# (0.5 cm lets the jaws reach a 1-inch block's middle); --table-margin overrides it
CLEARANCE_PAD_M = 0.001   # added to every check: covers the fast contact model's optimism (below)
# clearance checks the forearm and wrist too, not just the jaws (align.py's touch model stays jaws-only)
CLEARANCE_BODIES = ("gripper", "moving_jaw_so101_v1", "wrist", "lower_arm")
POSE_MIN_VIS = 0.5     # arm mode: tracked-arm + shoulder landmarks need min(visibility, presence) >= this
HAND_ENGAGE_S = 1.0    # arm mode: the body drives the robot only if the hand was seen this recently
CLEARANCE_DIRS = 500   # extreme-vertex directions: lowest_z ~0.1 ms; <= ~0.8 mm optimistic (reviewer
                       # measured 0.755 mm), compensated by CLEARANCE_PAD_M (+1 mm)


def clearance_model():
    """The per-frame table-clearance contact model (forearm, wrist, gripper, jaw; fast approximation)."""
    from cta.align import ContactModel  # loads its own MuJoCo model
    return ContactModel(CLEARANCE_BODIES, extreme_dirs=CLEARANCE_DIRS)


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
    """Move q[joint] from its failing value toward `safe` (caller checked it passes) just far enough for ok(q)."""
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


def _floor(contact, margin: float) -> float:
    """Lowest allowed z: table + margin + CLEARANCE_PAD_M."""
    return contact.table_z + margin + CLEARANCE_PAD_M


def _clears(q: np.ndarray, contact, margin: float = TABLE_CLEARANCE_M) -> bool:
    return contact.lowest_z(q) >= _floor(contact, margin)


def clear_table(q: np.ndarray, contact, clearance: float = TABLE_CLEARANCE_M, iters: int = 14) -> np.ndarray | None:
    """Keep the arm (CLEARANCE_BODIES meshes) `clearance` (+ CLEARANCE_PAD_M) above the table; None if impossible.

    Too low: move shoulder_lift toward 0 (raises the arm) by bisection, just far enough. If even lift 0
    is too low (elbow folded down with the upper arm vertical), unfold elbow_flex toward straight up
    (-ELBOW_OFFSET). If that fails too, return None (the caller holds its previous command)."""
    q = np.asarray(q, dtype=float)

    def ok(x):
        return _clears(x, contact, clearance)

    if ok(q):
        return q
    up = q.copy()
    up[1] = 0.0
    if ok(up):
        return _bisect_joint(q, 1, 0.0, ok, iters)
    straight = up.copy()
    straight[2] = max(-ELBOW_OFFSET_RAD, SIM_LIMITS[2, 0])
    if ok(straight):
        return _bisect_joint(up, 2, straight[2], ok, iters)
    return None


def guard_clearance(q_new: np.ndarray, prev: np.ndarray, contact,
                    margin: float = TABLE_CLEARANCE_M) -> tuple[np.ndarray, str | None]:
    """Last check on the command about to be SENT (after the speed limit and lead clip) -> (q, reason).

    Accepted (reason None): q_new clears table + margin (+ pad), or is no lower than prev (so an arm that
    starts low can still rise). Otherwise the moving joints that could lower the arm are frozen at prev -
    fewest first - (reason "froze"), and if no combination is acceptable, prev is held (reason "held")."""
    q_new = np.asarray(q_new, dtype=float)
    prev = np.asarray(prev, dtype=float)
    floor = _floor(contact, margin)
    z_prev = contact.lowest_z(prev)

    def accept(c):
        z = contact.lowest_z(c)
        return z >= floor or z >= z_prev - 1e-9

    if accept(q_new):
        return q_new, None
    moving = [j for j in range(1, 6) if q_new[j] != prev[j]]  # pan turns about z: never lowers the arm
    for k in range(1, len(moving)):  # freezing every moving joint is just prev: reported as "held" below
        for frozen in combinations(moving, k):
            c = q_new.copy()
            c[list(frozen)] = prev[list(frozen)]
            if accept(c):
                return c, "froze"
    return prev.copy(), "held"


class ArmPosePipeline(_PipelineBase):
    """--control arm (decision D-O): the user's shoulder/elbow (MediaPipe Pose) drive shoulder_pan,
    shoulder_lift and elbow_flex joint for joint; the hand still drives wrist_flex (world pitch),
    wrist_roll and the snap gripper.

    - No pose, or the tracked arm / shoulders only guessed (visibility or presence < pose_min_vis):
      COAST / NO HAND. Hidden hips are fine (camera-up fallback).
    - Engagement: the body drives the arm only while the hand was seen within HAND_ENGAGE_S; then the
      wrist and gripper hold. Longer without the hand: NO HAND, everything holds.
    - Every command keeps the arm table_margin (+ CLEARANCE_PAD_M) above the table (else the previous
      command holds)."""

    def __init__(self, kin: Kinematics, cal: Calibration | None, gains: MapGains,
                 smoother: OneEuro | None, max_step_rad: float = 0.15, aspect: float = 16 / 9,
                 mirrored: bool = False, want: str = "left", pose_min_vis: float = POSE_MIN_VIS,
                 contact=None, table_margin: float = TABLE_CLEARANCE_M, flip: bool = True):
        super().__init__(kin, cal, gains, smoother, max_step_rad, aspect)
        self.table_margin = table_margin
        self.flip = flip  # vertical-reflection arm mapping (see arm_angles.arm_joint_angles)
        if contact is None:
            contact = clearance_model()
        self.contact = contact
        self.mirrored = mirrored
        self.want = want
        self.pose_min_vis = pose_min_vis
        self.side: str | None = None
        self.clearance_failures = 0
        self._pan = float(self.q_cmd[0])
        self._pan_valid = True
        self._hand_seen_t: float | None = None
        self._pitch_filter: OneEuro | None = None
        self._pitch_filter_src: OneEuro | None = None

    def reset_to(self, q: np.ndarray) -> None:
        super().reset_to(q)
        self._pan = float(self.q_cmd[0])
        self._pitch_filter = None

    def _smooth_pitch(self, pitch: float, t: float) -> float:
        """The wrist is solved after the arm smoother, so the hand pitch gets a companion filter."""
        if self.smoother is None:
            return pitch
        if self._pitch_filter is None or self._pitch_filter_src is not self.smoother:
            s = self.smoother
            self._pitch_filter = OneEuro(min_cutoff=s.min_cutoff, beta=s.beta, d_cutoff=s.d_cutoff)
            self._pitch_filter_src = s
        return float(self._pitch_filter(np.array([pitch]), t)[0])

    def _wrist_flex_for_pitch(self, q: np.ndarray, pitch: float, ref: float | None = None) -> float:
        """wrist_flex giving world approach pitch `pitch` for this lift/elbow (planar chain: affine, slope +-1).

        Of the solutions 2 pi apart, the one nearest `ref` (default: the current wrist command) is clipped
        to the limits, so an unreachable target near the +-180 deg wrap doesn't flip the wrist end to end."""
        q0, q1 = q.copy(), q.copy()
        q0[3], q1[3] = 0.0, 0.5
        p0 = self.kin.planar_pitch(q0)
        slope = np.sign((self.kin.planar_pitch(q1) - p0 + np.pi) % (2 * np.pi) - np.pi)
        w = slope * ((pitch - p0 + np.pi) % (2 * np.pi) - np.pi)
        ref = self.q_cmd[3] if ref is None else ref
        w += 2 * np.pi * np.round((ref - w) / (2 * np.pi))
        lo, hi = SIM_LIMITS[3]
        return float(np.clip(w, lo, hi))

    def _usable(self, pose_obs, side: str) -> tuple[bool, bool]:
        """(arm landmarks trusted, hips trusted) from per-landmark min(visibility, presence)."""
        conf = np.minimum(np.asarray(pose_obs.vis, float), np.asarray(pose_obs.presence, float))
        arm = list(side_indices(side, self.mirrored)) + [11, 12]  # tracked arm + both shoulders
        return bool(np.all(conf[arm] >= self.pose_min_vis)), bool(np.all(conf[[23, 24]] >= self.pose_min_vis))

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
        arm_ok, hips_ok = self._usable(pose_obs, side)
        if not arm_ok:  # elbow/wrist/shoulder guessed (e.g. out of frame): don't let guesses drive the arm
            return self._lost(t)
        hand = map_to_target(f, self.cal, self.gains) if f is not None else None
        tracking_hand = hand is not None and bool(np.all(np.isfinite(hand[1:])))
        if tracking_hand:
            self._hand_seen_t = t
        elif self._hand_seen_t is None or t - self._hand_seen_t > HAND_ENGAGE_S:
            return self.q_cmd.copy(), "NO HAND"  # body alone is not engagement: present the hand
        pan, lift, elbow, pan_valid = arm_joint_angles(pose_obs.world, side, self.mirrored, self._pan,
                                                       self._pan_valid, hips_ok, flip=self.flip)
        if not np.all(np.isfinite([pan, lift, elbow])):
            return self._lost(t)
        self.side = side
        self._last_seen_t = t
        self._pan_valid = pan_valid
        if pan_valid:
            self._pan = pan
        prev = self.q_cmd.copy()
        held = prev[3:]  # hand missing: wrist_flex, wrist_roll and gripper hold their last values
        q = np.append(clamp_margin(np.append([pan, lift, elbow], held))[:3], held)
        pitch = None
        if tracking_hand:
            _, pitch, roll, grip = hand
            pitch = self._smooth_pitch(pitch, t)
            self.last_features = f
            q[3] = self._wrist_flex_for_pitch(q, pitch, ref=prev[3])
            q[4] = np.clip(roll, *SIM_LIMITS[4])
            q[5] = self._snap_grip(grip)
        out = self._finish(q[:5], q[5], t).copy()
        if pitch is None:
            out[3:] = held  # exact hold (the arm smoother could otherwise drift the wrist)
        else:
            out[3] = self._wrist_for_final(out, pitch, prev[3])
        cleared = clear_table(out, self.contact, self.table_margin)
        for _ in range(4):  # clearance moved lift/elbow: re-solve the wrist for the world pitch, re-check
            if cleared is None or pitch is None or np.array_equal(cleared, out):
                break
            out = cleared.copy()
            out[3] = self._wrist_for_final(out, pitch, prev[3])
            cleared = clear_table(out, self.contact, self.table_margin)
        if cleared is None:
            self.clearance_failures += 1
            if self.clearance_failures == 1:
                print("WARNING: no table clearance for this pose (even raising the arm fails); holding")
            self.q_cmd = prev
            return prev.copy(), "TRACKING"
        self.q_cmd = cleared
        return cleared.copy(), "TRACKING"

    def _wrist_for_final(self, q: np.ndarray, pitch: float, prev_w: float) -> float:
        """Re-solve wrist_flex on the final lift/elbow (so world pitch doesn't drift), rate-limited."""
        w = self._wrist_flex_for_pitch(q, pitch, ref=prev_w)
        return float(prev_w + np.clip(w - prev_w, -self.max_step_rad, self.max_step_rad))
