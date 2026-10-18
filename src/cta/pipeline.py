"""Hand / arm-pose / reach observation -> joint command pipelines (hold-last on loss, rate-limited)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from cta.arm_angles import ELBOW_OFFSET_RAD, arm_joint_angles, choose_arm_side, side_indices
from cta.config import READY_Q, SIM_LIMITS
from cta.features import HandFeatures, compute_features
from cta.filters import OneEuro
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta.mapping import Calibration, MapGains, calibrate, map_to_target, set_pinch_closed
from cta.depth import PATCH_RADIUS_FRAC, sample_patch
from cta.reach import (NO_HAND_PITCH_RAD, REACH_FRACTION, WindowMedian, fwd_from_depth, fwd_from_hand_scale,
                       palm_centre, plane_reach, reach_target, shoulder_sep)
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
CART_CLEAR_TOL_M = 0.001  # reach mode: Cartesian clearance lands within 1 mm above the floor
POSE_MIN_VIS = 0.5     # arm mode: tracked-arm + shoulder landmarks need min(visibility, presence) >= this
HAND_ENGAGE_S = 1.0    # arm mode: the body drives the robot only if the hand was seen this recently
TARGET_SPEED_MPS = 0.25  # reach (relative mapping): the smoothed 3-D target moves at most this fast
FWD_GAIN = 0.3           # reach (relative): forward gain. From replay (reach_train_01..04): at 1.0 forward
                         # saturated the x-box on 29-62% of tracked frames; 0.3 = largest gain in the sweep
                         # (1/0.6/0.4/0.35/0.3/0.2) with <= 10% saturation in every take (0/8.6/0.5/0%)
FWD_HOME_S = 0.5         # reach: the forward home is the median of >= this long of valid samples after 'c'
FWD_HOME_MIN_SAMPLES = 3
D_MIN_M, D_MAX_M = -1.0, 0.3  # reach: plausible hand - shoulder depth (hand <= 1 m in front, <= 0.3 m behind);
                              # outside = the patch missed the hand (take 04 frame 13: +0.78 m) -> rejected
SH_DEPTH_MAX_AGE_S = 1.0  # reach: shoulder depth older than this (hand covering it) -> forward held
TELEPORT_DT_CAP_S = 0.1   # reach: shoulder-teleport allowance 0.08 + 0.8 * min(dt, this)
MOVED_REASON = "arm/body moved - press c to re-lock"
DEPTH_MEDIAN_FRAMES = 3  # reach: median over the last N hand-shoulder depth differences (measurement denoise)
HAND_POINT = "wrist"     # reach: hand tracker point for "where" ("wrist" = lm 0, "palm" = mean of 0,5,9,13,17)
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

    def _finish(self, arm: np.ndarray, grip: float, t: float, smooth: bool = True) -> np.ndarray:
        """Smooth (arm joints only; the gripper must stay snappy), rate-limit, clip; becomes q_cmd.
        smooth=False: the caller already smoothed upstream (reach mode smooths the 3-D target)."""
        if smooth and self.smoother is not None:
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


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


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


@dataclass
class _Gate:
    """What the shared front half of a pose-driven step() found (see _BodyPipeline._gate)."""
    f: HandFeatures | None
    side: str
    hips_ok: bool
    hand: tuple | None  # map_to_target(...) when the hand is tracked this frame, else None


class _BodyPipeline(_PipelineBase):
    """Shared by the pose-driven modes (arm, reach): arm-side choice, visibility gating, hand engagement,
    the hand-pitch companion filter and the table-clearance hold.

    - No pose, or the tracked arm / shoulders only guessed (visibility or presence < pose_min_vis):
      COAST / NO HAND. Hidden hips are fine (camera-up fallback).
    - Engagement: the body drives the arm only while the hand was seen within HAND_ENGAGE_S.
      Longer without the hand: NO HAND, everything holds.
    - Every command keeps the arm table_margin (+ CLEARANCE_PAD_M) above the table (else the previous
      command holds)."""

    def __init__(self, kin: Kinematics, cal: Calibration | None, gains: MapGains,
                 smoother: OneEuro | None, max_step_rad: float = 0.15, aspect: float = 16 / 9,
                 mirrored: bool = False, want: str = "left", pose_min_vis: float = POSE_MIN_VIS,
                 contact=None, table_margin: float = TABLE_CLEARANCE_M):
        super().__init__(kin, cal, gains, smoother, max_step_rad, aspect)
        self.table_margin = table_margin
        if contact is None:
            contact = clearance_model()
        self.contact = contact
        self.mirrored = mirrored
        self.want = want
        self.pose_min_vis = pose_min_vis
        self.side: str | None = None
        self.clearance_failures = 0
        self._hand_seen_t: float | None = None
        self._pitch_filter: OneEuro | None = None
        self._pitch_filter_src: OneEuro | None = None

    def reset_to(self, q: np.ndarray) -> None:
        super().reset_to(q)
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

    def _usable(self, pose_obs, side: str) -> tuple[bool, bool]:
        """(arm landmarks trusted, hips trusted) from per-landmark min(visibility, presence)."""
        conf = np.minimum(np.asarray(pose_obs.vis, float), np.asarray(pose_obs.presence, float))
        arm = list(side_indices(side, self.mirrored)) + [11, 12]  # tracked arm + both shoulders
        return bool(np.all(conf[arm] >= self.pose_min_vis)), bool(np.all(conf[[23, 24]] >= self.pose_min_vis))

    def _gate(self, obs: HandObservation | None, t: float, pose_obs) -> tuple[tuple | None, _Gate | None]:
        """Front half of step(): (early (q, status), None) to return now, or (None, gate) to go on."""
        self.last_features = None
        f = self._features(obs) if obs is not None else None
        if self.cal is None:
            self.last_features = f
            return (READY_Q.astype(float).copy(), "UNCALIBRATED"), None
        if pose_obs is None or not (np.all(np.isfinite(pose_obs.world)) and np.all(np.isfinite(pose_obs.img))):
            return self._lost(t), None
        wrist_uv = obs.lm_img[0, :2] if obs is not None else None
        side = choose_arm_side(pose_obs.img, wrist_uv, self.side, self.want, self.mirrored)
        arm_ok, hips_ok = self._usable(pose_obs, side)
        if not arm_ok:  # elbow/wrist/shoulder guessed (e.g. out of frame): don't let guesses drive the arm
            return self._lost(t), None
        hand = map_to_target(f, self.cal, self.gains) if f is not None else None
        tracking_hand = hand is not None and bool(np.all(np.isfinite(hand[1:])))
        if tracking_hand:
            self._hand_seen_t = t
        elif self._hand_seen_t is None or t - self._hand_seen_t > HAND_ENGAGE_S:
            return (self.q_cmd.copy(), "NO HAND"), None  # body alone is not engagement: present the hand
        return None, _Gate(f, side, hips_ok, hand if tracking_hand else None)

    def _accept(self, gate: _Gate, t: float) -> None:
        """The frame is good: remember the side and the time (COAST is measured from here)."""
        self.side = gate.side
        self._last_seen_t = t
        if gate.hand is not None:
            self.last_features = gate.f

    def _clear_or_hold(self, out: np.ndarray, prev: np.ndarray, resolve=None) -> tuple[np.ndarray, str]:
        """clear_table on the output (resolve(q) -> q re-fixes the wrist after a clearance move, up to 4
        rounds); if no clear pose exists, hold prev and warn once."""
        cleared = clear_table(out, self.contact, self.table_margin)
        for _ in range(4):
            if cleared is None or resolve is None or np.array_equal(cleared, out):
                break
            out = resolve(cleared.copy())
            cleared = clear_table(out, self.contact, self.table_margin)
        if cleared is None:
            self.clearance_failures += 1
            if self.clearance_failures == 1:
                print("WARNING: no table clearance for this pose (even raising the arm fails); holding")
            self.q_cmd = prev
            return prev.copy(), "TRACKING"
        self.q_cmd = cleared
        return cleared.copy(), "TRACKING"


class ArmPosePipeline(_BodyPipeline):
    """--control arm (decision D-O): the user's shoulder/elbow (MediaPipe Pose) drive shoulder_pan,
    shoulder_lift and elbow_flex joint for joint; the hand still drives wrist_flex (world pitch),
    wrist_roll and the snap gripper. No hand (still engaged): wrist and gripper hold."""

    def __init__(self, kin: Kinematics, cal: Calibration | None, gains: MapGains,
                 smoother: OneEuro | None, max_step_rad: float = 0.15, aspect: float = 16 / 9,
                 mirrored: bool = False, want: str = "left", pose_min_vis: float = POSE_MIN_VIS,
                 contact=None, table_margin: float = TABLE_CLEARANCE_M, flip: bool = True):
        super().__init__(kin, cal, gains, smoother, max_step_rad, aspect, mirrored, want, pose_min_vis,
                         contact, table_margin)
        self.flip = flip  # vertical-reflection arm mapping (see arm_angles.arm_joint_angles)
        self._pan = float(self.q_cmd[0])
        self._pan_valid = True

    def reset_to(self, q: np.ndarray) -> None:
        super().reset_to(q)
        self._pan = float(self.q_cmd[0])

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

    def step(self, obs: HandObservation | None, t: float, pose_obs=None) -> tuple[np.ndarray, str]:
        early, g = self._gate(obs, t, pose_obs)
        if early is not None:
            return early
        pan, lift, elbow, pan_valid = arm_joint_angles(pose_obs.world, g.side, self.mirrored, self._pan,
                                                       self._pan_valid, g.hips_ok, flip=self.flip)
        if not np.all(np.isfinite([pan, lift, elbow])):
            return self._lost(t)
        self._accept(g, t)
        self._pan_valid = pan_valid
        if pan_valid:
            self._pan = pan
        prev = self.q_cmd.copy()
        held = prev[3:]  # hand missing: wrist_flex, wrist_roll and gripper hold their last values
        q = np.append(clamp_margin(np.append([pan, lift, elbow], held))[:3], held)
        pitch = None
        if g.hand is not None:
            _, pitch, roll, grip = g.hand
            pitch = self._smooth_pitch(pitch, t)
            q[3] = self._wrist_flex_for_pitch(q, pitch, ref=prev[3])
            q[4] = np.clip(roll, *SIM_LIMITS[4])
            q[5] = self._snap_grip(grip)
        out = self._finish(q[:5], q[5], t).copy()
        if pitch is None:
            out[3:] = held  # exact hold (the arm smoother could otherwise drift the wrist)
            return self._clear_or_hold(out, prev)
        out[3] = self._wrist_for_final(out, pitch, prev[3])

        def resolve(c):  # clearance moved lift/elbow: re-solve the wrist for the world pitch
            c[3] = self._wrist_for_final(c, pitch, prev[3])
            return c

        return self._clear_or_hold(out, prev, resolve)

    def _wrist_for_final(self, q: np.ndarray, pitch: float, prev_w: float) -> float:
        """Re-solve wrist_flex on the final lift/elbow (so world pitch doesn't drift), rate-limited."""
        w = self._wrist_flex_for_pitch(q, pitch, ref=prev_w)
        return float(prev_w + np.clip(w - prev_w, -self.max_step_rad, self.max_step_rad))


class ReachPipeline(_BodyPipeline):
    """--control reach (default, Tray 2026-10-18): the user's arm is the WHERE, the hand the WHAT.

    WHERE (spec 2026-10-18-reach-measurement-fix): the arm is locked at calibration ('c': the Pose side whose
    wrist is nearest the hand tracker's wrist) and never re-picked until the next 'c'. n = (fwd, left, up):
    left/up = hand tracker hand point - locked Pose shoulder (image), in arm lengths (ARM_PER_SHOULDER x the
    median image shoulder separation); fwd = metric depth (hand - shoulder) / (ARM_PER_SHOULDER x Pose world
    shoulder width), or the hand-scale cue (fwd_source="scale", e.g. no depth model). Pose wrist/elbow
    landmarks are not used. Target = p0 + ROBOT_REACH (n - n_home) (home_relative, the CLI default) or
    ROBOT_SHOULDER + ROBOT_REACH n, clipped to the workspace box (z floor = table + margin + pad). The arm
    smoother runs on that 3-D target, then Kinematics.ik picks the robot's own joints.
    Hand: claw pitch (calibrated, as in hand mode), wrapped roll, snap gripper.
    No hand, no pose, the locked/other shoulder guessed (< pose_min_vis) or the shoulders collapsing /
    teleporting: COAST / NO HAND (everything holds)."""

    def __init__(self, kin: Kinematics, cal: Calibration | None, gains: MapGains,
                 smoother: OneEuro | None, max_step_rad: float = 0.15, aspect: float = 16 / 9,
                 mirrored: bool = False, want: str = "left", pose_min_vis: float = POSE_MIN_VIS,
                 contact=None, table_margin: float = TABLE_CLEARANCE_M, home_relative: bool = False,
                 hand_point: str = HAND_POINT, fwd_source: str = "depth", fwd_gain: float = FWD_GAIN):
        super().__init__(kin, cal, gains, smoother, max_step_rad, aspect, mirrored, want, pose_min_vis,
                         contact, table_margin)
        if hand_point not in ("wrist", "palm") or fwd_source not in ("depth", "scale"):
            raise ValueError(f"bad hand_point={hand_point!r} / fwd_source={fwd_source!r}")
        self.robot_shoulder = kin.shoulder_anchor()
        self.robot_reach = REACH_FRACTION * kin.max_reach()
        # The CLI selects home-relative mapping; absolute mapping remains available for old experiments.
        self.home_relative = home_relative
        self.hand_point = hand_point
        self.fwd_source = fwd_source
        self.fwd_gain = float(fwd_gain)
        self.box_min = np.array(gains.box_min, dtype=float)
        self.box_max = np.array(gains.box_max, dtype=float)
        self.box_min[2] = self.contact.table_z + table_margin + CLEARANCE_PAD_M
        self.tracking_hold_reason: str | None = None
        self._reach_sample_t: float | None = None
        self._reset_measurement()
        self.target: np.ndarray | None = None
        self.pitch_target = NO_HAND_PITCH_RAD
        self.ik_target: np.ndarray | None = None  # target after the Cartesian clearance raise
        self.roll_target = 0.0
        self._hand_pitch: float | None = None
        self._roll_u: float | None = None  # unwrapped hand roll (for wrap-aware smoothing)
        self._roll_filter: OneEuro | None = None
        self._roll_filter_src: OneEuro | None = None

    def reset_to(self, q: np.ndarray) -> None:
        super().reset_to(q)
        self._roll_filter = None

    def _smooth_roll(self, roll: float, t: float) -> float:
        """Wrap-aware roll smoothing: One Euro on the unwrapped roll, wrapped back to [-pi, pi)."""
        self._roll_u = roll if self._roll_u is None else self._roll_u + _wrap(roll - self._roll_u)
        if self.smoother is None:
            return _wrap(self._roll_u)
        if self._roll_filter is None or self._roll_filter_src is not self.smoother:
            s = self.smoother
            self._roll_filter = OneEuro(min_cutoff=s.min_cutoff, beta=s.beta, d_cutoff=s.d_cutoff)
            self._roll_filter_src = s
        return _wrap(float(self._roll_filter(np.array([self._roll_u]), t)[0]))

    def _ik_clear(self, target: np.ndarray, pitch: float, q_init: np.ndarray, roll: float,
                  grip: float) -> tuple[np.ndarray, np.ndarray]:
        """IK, keeping table clearance in CARTESIAN space: if the solution would put the arm below
        table + margin (+ pad), raise only the target's z by bisection (x, y, pitch kept) until the lowest
        point sits just above it (CART_CLEAR_TOL_M). If even the box top can't clear, the plain solution is
        returned and clear_table / hold handles it. -> (q, the target actually solved for)."""
        q = self.kin.ik(target, pitch, q_init, roll, grip)
        if _clears(q, self.contact, self.table_margin):
            return q, target
        floor = _floor(self.contact, self.table_margin)
        # secant steps first: the lowest point rises roughly linearly with the target z (slope ~1 at first,
        # then measured), so 1-3 IK solves usually land within tolerance; bisection below is the fallback
        aim = floor + CART_CLEAR_TOL_M / 2
        qn, zn, low = q, float(target[2]), self.contact.lowest_z(q)
        prev_t = self.ik_target
        if prev_t is not None and prev_t[2] > zn and np.linalg.norm(prev_t[:2] - target[:2]) < 0.02:
            # temporal coherence: start from last frame's raised height (steady state: 2 IK solves)
            tn = target.copy()
            tn[2] = min(float(prev_t[2]), float(self.box_max[2]))
            q0 = self.kin.ik(tn, pitch, q_init, roll, grip)
            low0 = self.contact.lowest_z(q0)
            if floor <= low0 <= floor + CART_CLEAR_TOL_M:
                return q0, tn
            qn, zn, low = q0, float(tn[2]), low0
        slope = 1.0
        for _ in range(4):
            z_new = min(zn + (aim - low) / slope, float(self.box_max[2]))
            tn = target.copy()
            tn[2] = z_new
            q_new = self.kin.ik(tn, pitch, qn, roll, grip)
            low_new = self.contact.lowest_z(q_new)
            if floor <= low_new <= floor + CART_CLEAR_TOL_M:
                return q_new, tn
            if abs(z_new - zn) > 1e-6:
                slope = float(np.clip((low_new - low) / (z_new - zn), 0.3, 3.0))
            qn, zn, low = q_new, z_new, low_new
        lo, hi = float(target[2]), float(self.box_max[2])
        up = target.copy()
        up[2] = hi
        q_hi = self.kin.ik(up, pitch, q, roll, grip)
        if not _clears(q_hi, self.contact, self.table_margin):
            return q, target
        best_t = up.copy()
        while hi - lo > CART_CLEAR_TOL_M:
            mid = target.copy()
            mid[2] = (lo + hi) / 2
            qm = self.kin.ik(mid, pitch, q_hi, roll, grip)
            if _clears(qm, self.contact, self.table_margin):
                hi, q_hi, best_t = mid[2], qm, mid
            else:
                lo = mid[2]
        return q_hi, best_t

    def _reset_measurement(self) -> None:
        """Everything 'c' re-measures: the locked arm, the home, the medians, the depth reference."""
        self.side = None
        self.locked_side: str | None = None
        self.reach_home_n: np.ndarray | None = None
        self.last_n: np.ndarray | None = None
        self.raw_target: np.ndarray | None = None
        self.reach_line_uv: np.ndarray | None = None   # (2, 2): locked shoulder -> hand point (dashboard)
        self.depth_hand = float("nan")                  # metres, this frame's hand patch (recorder)
        self.depth_shoulder = float("nan")              # metres, the shoulder depth used (recorder)
        self._sep = WindowMedian()        # image shoulder separation (height units)
        self._width_m = WindowMedian()    # Pose world shoulder width (m)
        self._sh_depth = WindowMedian()   # shoulder patch depth (m)
        self._d: float | None = None      # depth_hand - depth_shoulder, short median (held between depth frames)
        self._d_buf: deque[float] = deque(maxlen=DEPTH_MEDIAN_FRAMES)
        self._d_t: float | None = None    # time of the last accepted d sample
        self._d_home: float | None = None
        self._home_samples: list[tuple] = []  # forward home candidates (t, d) or (t, ratio, sep)
        self._sh_depth_t: float | None = None  # time of the last shoulder depth sample
        self._ratio_home: float | None = None  # hand-scale fallback: median (hand scale / sep) at home
        self._sep_home: float | None = None
        self._fwd_scale = 0.0     # last hand-scale forward (relative, before gain)
        self._fwd_offset = 0.0    # forward carried over when depth fails mid-session (depth_failed)
        self._last_fwd_rel = 0.0
        self._shoulder_span: float | None = None
        self._prev_shoulders: np.ndarray | None = None
        self._prev_shoulders_t: float | None = None

    def calibrate(self, obs: HandObservation, pose_obs=None) -> None:
        """'c': hand calibration + lock the arm (now if the body is visible, else on the next frame with
        both); the home is recorded on the first tracked frame after this (same frame in the app)."""
        super().calibrate(obs)
        self._reset_measurement()
        self.tracking_hold_reason = None
        self._try_lock(obs, pose_obs)

    def _try_lock(self, obs: HandObservation | None, pose_obs) -> None:
        """Lock the arm whose Pose wrist is nearest the hand-tracker wrist; shoulders and wrists must be
        finite and trusted (min(visibility, presence) >= pose_min_vis)."""
        if self.locked_side is not None or obs is None or pose_obs is None:
            return
        idx = [11, 12, 15, 16]
        if not np.all(np.isfinite(np.asarray(pose_obs.img)[idx, :2])):
            return
        conf = np.minimum(np.asarray(pose_obs.vis, float), np.asarray(pose_obs.presence, float))
        if not np.all(conf[idx] >= self.pose_min_vis):
            return
        self.locked_side = choose_arm_side(pose_obs.img, obs.lm_img[0, :2], None, self.want, self.mirrored)
        self.side = self.locked_side
        print(f"arm locked: {self.locked_side} (press c to re-lock)")

    def depth_failed(self) -> None:
        """The depth model died mid-session: continue on the hand-scale cue from the current forward value
        (no jump; forward is held while the hand-scale home is measured)."""
        if self.fwd_source == "depth":
            self.fwd_source = "scale"
            self._fwd_offset = self._last_fwd_rel
            self._fwd_scale = 0.0
            self._ratio_home = self._sep_home = None
            self._home_samples = []

    def _home_ready(self, t: float) -> bool:
        h = self._home_samples
        return len(h) >= FWD_HOME_MIN_SAMPLES and t - h[0][0] >= FWD_HOME_S

    def _hold(self, reason: str, t: float) -> tuple[np.ndarray, str]:
        self.tracking_hold_reason = reason
        return self._lost(t)

    def _shoulder_glitch(self, pose_obs, t: float) -> str | None:
        """Confidence alone can't catch confident hallucinations (visibility reads ~1.0 throughout the
        recordings). Shoulders out of view, a collapsed/changed span or a teleport -> hold. The teleport
        allowance is 0.08 + 0.8 * min(dt, 0.1 s) from the last good shoulders: a body that moved far while
        unseen holds until it comes back or 'c' re-locks."""
        shoulders = np.asarray(pose_obs.img[[11, 12], :2], dtype=float)
        span = float(np.linalg.norm(shoulders[0] - shoulders[1]))
        if np.any(shoulders < 0.02) or np.any(shoulders > 0.98):
            return "shoulders outside camera view"
        if span < 0.06 or (self._shoulder_span is not None and
                           not 0.5 * self._shoulder_span <= span <= 2.0 * self._shoulder_span):
            return "shoulder span collapsed or changed"
        if self._prev_shoulders is not None:
            dt = min(max(0.0, t - self._prev_shoulders_t), TELEPORT_DT_CAP_S)
            if np.linalg.norm(shoulders - self._prev_shoulders, axis=1).max() > 0.08 + 0.8 * dt:
                return MOVED_REASON  # teleport, or the body moved while unseen: held until it returns / 'c'
        self._prev_shoulders, self._prev_shoulders_t = shoulders, t
        return None

    def _forward(self, f: HandFeatures, obs_lm: np.ndarray, sh_uv: np.ndarray, sep: float,
                 depth_map: np.ndarray | None, t: float) -> tuple[float, float]:
        """(n_fwd relative to home x fwd_gain, n_fwd absolute). Forward is held at home (0) until the home is
        the median of >= FWD_HOME_S of valid samples after 'c'.

        Depth: d = palm-centre patch depth - shoulder depth. Samples outside [D_MIN_M, D_MAX_M] (the patch
        missed the hand) are rejected; d = median of the last DEPTH_MEDIAN_FRAMES samples, restarted after a
        gap > COAST_S, held between depth frames. The shoulder depth is a 0.5 s median, not sampled while the
        hand covers the shoulder; older than SH_DEPTH_MAX_AGE_S -> d held.
        Scale fallback: relative only (absolute = relative); non-finite samples hold the last value."""
        if self.fwd_source == "scale":
            ratio = f.scale * self.aspect / sep
            if np.isfinite(ratio) and ratio > 0:
                if self._ratio_home is None:
                    self._home_samples.append((t, ratio, sep))
                    if self._home_ready(t):
                        self._ratio_home = float(np.median([h[1] for h in self._home_samples]))
                        self._sep_home = float(np.median([h[2] for h in self._home_samples]))
                if self._ratio_home is not None:
                    self._fwd_scale = fwd_from_hand_scale(ratio / self._ratio_home, self._sep_home)
            n = self._fwd_offset + self.fwd_gain * self._fwd_scale
            self._last_fwd_rel = n
            return n, n
        ds = self._sh_depth.value
        if depth_map is not None:
            palm = palm_centre(obs_lm)  # depth always at the palm centre: the wrist point sits on the hand's
            dh = sample_patch(depth_map, palm)  # edge and its patch hits the background (replay: 0.9-1.4 m p99)
            self.depth_hand = dh
            gap = np.hypot((palm[0] - sh_uv[0]) * self.aspect, palm[1] - sh_uv[1])
            if gap > 2.0 * f.scale * self.aspect + 2 * PATCH_RADIUS_FRAC:  # hand not over the shoulder
                sample = sample_patch(depth_map, sh_uv)
                if np.isfinite(sample):
                    ds = self._sh_depth.update(sample, t)
                    self._sh_depth_t = t
            fresh = self._sh_depth_t is not None and t - self._sh_depth_t <= SH_DEPTH_MAX_AGE_S
            d = dh - ds
            if fresh and np.isfinite(d) and D_MIN_M <= d <= D_MAX_M:
                if self._d_t is not None and t - self._d_t > COAST_S:
                    self._d_buf.clear()  # don't median across a gap
                self._d_buf.append(d)
                self._d_t = t
                self._d = float(np.median(self._d_buf))
                if self._d_home is None:
                    self._home_samples.append((t, d))
                    if self._home_ready(t):
                        self._d_home = float(np.median([h[1] for h in self._home_samples]))
        self.depth_shoulder = ds
        width = self._width_m.value
        if self._d is None or not np.isfinite(width) or width <= 0:
            return self._fwd_offset, 0.0
        n_abs = fwd_from_depth(self._d, 0.0, width)
        n_rel = 0.0 if self._d_home is None else self.fwd_gain * fwd_from_depth(self._d, self._d_home, width)
        self._last_fwd_rel = n_rel
        return n_rel, n_abs

    def step(self, obs: HandObservation | None, t: float, pose_obs=None,
             depth_map: np.ndarray | None = None) -> tuple[np.ndarray, str]:
        """depth_map: metric depth (any resolution, same framing as the image) for this frame, or None
        (no depth model, or a skipped depth frame: the last hand-shoulder depth is held)."""
        self.tracking_hold_reason = None
        self.reach_line_uv = None
        self.depth_hand = float("nan")
        self.depth_shoulder = float("nan")
        self.last_features = None
        sample_dt = 1 / 60 if self._reach_sample_t is None else max(0.0, min(t - self._reach_sample_t, 0.05))
        self._reach_sample_t = t
        f = self._features(obs) if obs is not None else None
        if self.cal is None:
            self.last_features = f
            return READY_Q.astype(float).copy(), "UNCALIBRATED"
        if obs is None:
            return self._lost(t)  # the hand tracker IS the "where": no hand, no target (hold)
        if pose_obs is None or not np.all(np.isfinite(np.asarray(pose_obs.img)[[11, 12], :2])):
            return self._hold("no body (shoulders) in view", t)
        self._try_lock(obs, pose_obs)
        if self.locked_side is None:
            return self._hold("arm not locked yet", t)
        conf = np.minimum(np.asarray(pose_obs.vis, float), np.asarray(pose_obs.presence, float))
        sh_i = side_indices(self.locked_side, self.mirrored)[0]
        if not np.all(conf[[11, 12]] >= self.pose_min_vis):  # locked shoulder (and the scale's other one)
            return self._hold("shoulder not visible", t)
        hand = map_to_target(f, self.cal, self.gains)
        if not np.all(np.isfinite(hand[1:])):
            return self._hold("hand orientation not reliable", t)
        reason = self._shoulder_glitch(pose_obs, t)
        if reason is not None:
            return self._hold(reason, t)

        img = np.asarray(pose_obs.img, dtype=float)
        sh_uv = img[sh_i, :2]
        hand_uv = palm_centre(obs.lm_img) if self.hand_point == "palm" else np.asarray(obs.lm_img[0, :2], float)
        sep = self._sep.update(shoulder_sep(img, self.aspect), t)
        w = np.asarray(pose_obs.world, dtype=float)
        self._width_m.update(float(np.linalg.norm(w[11] - w[12])), t)
        n_left, n_up = plane_reach(hand_uv, sh_uv, sep, self.aspect, self.mirrored)
        fwd_rel, fwd_abs = self._forward(f, obs.lm_img, sh_uv, sep, depth_map, t)
        if not np.all(np.isfinite([n_left, n_up, fwd_rel, fwd_abs])):
            return self._hold("reach not measurable", t)

        self.side = self.locked_side
        self._last_seen_t = t
        self.last_features = f
        if self._shoulder_span is None:
            self._shoulder_span = float(np.linalg.norm(img[11, :2] - img[12, :2]))
        self.reach_line_uv = np.array([sh_uv, hand_uv])
        if self.home_relative:
            n = np.array([fwd_rel, n_left, n_up])
            if self.reach_home_n is None:
                self.reach_home_n = n.copy()
            self.raw_target = np.clip(self.cal.p0 + self.robot_reach * (n - self.reach_home_n),
                                      self.box_min, self.box_max)
        else:
            n = np.array([fwd_abs, n_left, n_up])
            self.raw_target = reach_target(n, self.robot_shoulder, self.robot_reach, self.box_min, self.box_max)
        self.last_n = n
        target = self.raw_target
        if self.smoother is not None:  # smooth in space (the target), not the joints
            target = np.clip(np.asarray(self.smoother(target, t), dtype=float), self.box_min, self.box_max)
        if self.home_relative and self.target is not None:  # target speed limit (TARGET_SPEED_MPS)
            delta = target - self.target
            distance = float(np.linalg.norm(delta))
            if distance > 0:
                target = self.target + delta * min(1.0, TARGET_SPEED_MPS * sample_dt / distance)
        self.target = target
        prev = self.q_cmd.copy()
        _, pitch, roll, grip = hand
        self._hand_pitch = self._smooth_pitch(pitch, t)
        self.pitch_target = self._hand_pitch
        self.roll_target = self._smooth_roll(roll, t)
        roll, grip = float(np.clip(self.roll_target, *SIM_LIMITS[4])), self._snap_grip(grip)
        q, self.ik_target = self._ik_clear(target, self.pitch_target, prev, roll, grip)
        out = self._finish(q[:5], grip, t, smooth=False).copy()
        out, status = self._clear_or_hold(out, prev)
        if self.home_relative:
            # Apply after clearance too: the corrective solver must not bypass time-based limits.
            bounded = prev + np.clip(out - prev, -np.radians(90) * sample_dt, np.radians(90) * sample_dt)
            self.q_cmd, _ = guard_clearance(bounded, prev, self.contact, self.table_margin)
            return self.q_cmd.copy(), status
        return out, status
