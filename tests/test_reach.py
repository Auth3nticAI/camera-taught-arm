"""--control reach: the user's arm says WHERE, the robot gets there with its own IK.

WHERE (spec 2026-10-18-reach-measurement-fix): the hand tracker's hand point relative to the LOCKED arm's
Pose shoulder (image plane, in shoulder-widths), plus forward from Depth Anything V2 (hand vs shoulder
depth), or the hand-scale cue when there is no depth model."""
import mujoco
import numpy as np
import pytest
from helpers import bend, body_dir, make_body, make_hand

from cta.align import ContactModel
from cta.arm_angles import side_indices
from cta.config import SIM_LIMITS
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta.mapping import MapGains
from cta.pipeline import CLEARANCE_BODIES, CLEARANCE_PAD_M, COAST_S, TABLE_CLEARANCE_M, ReachPipeline
from cta.pose import PoseObservation
from cta.reach import (ARM_PER_SHOULDER, REACH_FRACTION, WindowMedian, fwd_from_depth, fwd_from_hand_scale,
                       palm_centre, plane_reach, reach_target, shoulder_sep)

D = np.radians
GLITCH = "shoulder jump rejected (glitch)"
ASPECT = 16 / 9


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


@pytest.fixture(scope="module")
def gains(kin):
    return MapGains.from_kinematics(kin)


@pytest.fixture(scope="module")
def contact():
    return ContactModel(CLEARANCE_BODIES)


# ---- robot constants, measured from the MuJoCo model -------------------------------------------------

def test_robot_shoulder_is_the_shoulder_lift_anchor(kin):
    kin.fk(np.zeros(6))
    m, d = kin._m, kin._d
    rb, pb = d.site_xmat[kin._base].reshape(3, 3), d.site_xpos[kin._base]
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift")
    assert np.allclose(kin.shoulder_anchor(), rb.T @ (d.xanchor[j] - pb), atol=1e-9)


def test_robot_reach_is_the_max_gripper_distance(kin):
    sh, reach = kin.shoulder_anchor(), kin.max_reach()
    rng = np.random.default_rng(0)
    qs = rng.uniform(SIM_LIMITS[:, 0], SIM_LIMITS[:, 1], (3000, 6))
    qs[:, 0] = 0.0  # the anchor is measured at pan 0 (pan turns the anchor with the arm)
    dists = [np.linalg.norm(kin.fk(q)[0] - sh) for q in qs]
    assert max(dists) <= reach + 1e-4  # nothing reaches farther
    assert max(dists) > reach - 0.02   # and it isn't a loose over-estimate
    assert 0.35 < reach < 0.5 and REACH_FRACTION == 0.95


def test_reach_target_clamps_and_clips():
    sh, reach = np.array([0.07, -0.02, 0.12]), 0.4
    lo, hi = np.array([0.2, -0.3, 0.01]), np.array([0.44, 0.3, 0.25])
    t = reach_target(np.array([2.0, 0.0, 0.0]), sh, reach, lo, hi)  # |n| > 1 is clamped to 1
    assert np.allclose(t, [0.44, -0.02, 0.12])
    t = reach_target(np.array([0.5, 0.0, -0.9]), sh, reach, lo, hi)
    assert t[2] == pytest.approx(0.01)  # never below the box floor


# ---- measurement units (pure) --------------------------------------------------------------------------

def _body(az, elev, flex=0.0, side="left", mirrored=False):
    a = body_dir(az, elev)
    return make_body(a, bend(a, flex), side=side, mirrored=mirrored)


def test_shoulder_sep_is_aspect_corrected():
    img, _ = _body(0, 180)
    assert shoulder_sep(img, ASPECT) == pytest.approx(0.35)  # synthetic: 1 image height per metre


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("az,elev,want", [(90, 90, (1, 0)), (-90, 90, (-1, 0)), (0, 0, (0, 1)), (0, 180, (0, -1))])
def test_plane_reach_is_in_user_terms_in_both_mirror_modes(mirrored, side, az, elev, want):
    """Arm straight to the user's left / right / up / down -> (n_left, n_up) = +-0.53 / (1.7 x 0.35)."""
    img, _ = _body(az, elev, side=side, mirrored=mirrored)
    sh, _, wr = side_indices(side, mirrored)
    n = plane_reach(img[wr, :2], img[sh, :2], shoulder_sep(img, ASPECT), ASPECT, mirrored)
    assert np.allclose(n, 0.53 / (ARM_PER_SHOULDER * 0.35) * np.array(want), atol=1e-9)


def test_palm_centre_is_the_mean_of_wrist_and_mcps():
    lm, _ = make_hand(wrist_uv=(0.3, 0.6))
    assert np.allclose(palm_centre(lm), lm[[0, 5, 9, 13, 17], :2].mean(axis=0))


def test_fwd_from_depth_hand_closer_is_forward():
    assert fwd_from_depth(-0.30, -0.10, 0.35) == pytest.approx(0.20 / (ARM_PER_SHOULDER * 0.35))
    assert fwd_from_depth(0.0, -0.10, 0.35) < 0  # hand pulled back toward the body
    assert np.isnan(fwd_from_depth(np.nan, 0.0, 0.35))


def test_fwd_from_hand_scale_fallback():
    assert fwd_from_hand_scale(1.0, 0.3) == pytest.approx(0.0)  # at home
    assert fwd_from_hand_scale(1.25, 0.3) > 0.1                  # hand looks bigger: closer: forward
    assert fwd_from_hand_scale(0.8, 0.3) < -0.1
    # pinhole: hand at Z_home (= f w / sep_home) moved to Z_home / r -> (1 - 1/r) Z_home / (ARM w)
    assert fwd_from_hand_scale(2.0, 0.3, focal_h=1.2) == pytest.approx(0.5 * 1.2 / 0.3 / ARM_PER_SHOULDER)


def test_window_median_is_time_based():
    m = WindowMedian(0.5)
    for i in range(10):
        m.update(1.0, i * 0.04)
    assert m.update(5.0, 0.40) == pytest.approx(1.0)  # one outlier doesn't move it
    assert m.update(2.0, 2.0) == pytest.approx(2.0)   # old samples expire


# ---- ReachPipeline -------------------------------------------------------------------------------------

def pose(az=0.0, elev=90.0, flex=0.0, side="left", mirrored=False, vis=1.0):
    img, world = _body(az, elev, flex, side, mirrored)
    img = img.copy()
    img[:, 1] += 0.4  # realistic framing: head (face) in view above the shoulders, not off the top
    po = PoseObservation(img, world, 0.9, np.full(33, vis), np.ones(33))
    po.mirrored_input = mirrored
    po.side = side
    return po


def hand(po=None, side="left", **kw):
    if po is not None:
        kw["wrist_uv"] = tuple(po.img[side_indices(side, po.mirrored_input)[2], :2])
    lm_img, lm_world = make_hand(**kw)
    return HandObservation(lm_img, lm_world, "left", 0.9)


def depth(po, side="left", base=1.5, size=(90, 160)):
    """Synthetic metric depth: the body plane at the shoulder's depth, a disc at the arm's wrist depth."""
    sh, _, wr = side_indices(side, po.mirrored_input)
    d = np.full(size, base + po.world[sh, 2])
    h, w = size
    yy, xx = np.mgrid[:h, :w]
    u, v = po.img[wr, :2]
    d[(xx - u * w) ** 2 + (yy - v * h) ** 2 <= 6 ** 2] = base + po.world[wr, 2]
    return d


def calibrated(kin, gains, po=None, **kw):
    """'c' in a neutral pose (arm hanging forward-down, hand clear of the shoulder) + that frame's step."""
    kw.setdefault("hand_point", "wrist")
    p = ReachPipeline(kin, None, gains, None, **kw)
    po = pose(0, 120, 30) if po is None else po
    p.calibrate(hand(po, side=po.side), po)
    p.step(hand(po, side=po.side), -1.0, po, depth(po, po.side))
    return p


def run(p, hand_obs, pose_obs, n=90, t0=0.0, dmap="auto"):
    dm = depth(pose_obs, pose_obs.side) if isinstance(dmap, str) else dmap
    for i in range(n):
        q, s = p.step(hand_obs, t0 + i / 60, pose_obs, dm)
    return q, s


def test_calibration_locks_the_arm_and_it_never_switches(kin, gains):
    p = calibrated(kin, gains)
    assert p.locked_side == "left"
    po = pose()
    run(p, hand(po), po, n=5)
    other = pose(side="right")  # the hand tracker's wrist now sits on the OTHER Pose wrist
    for i in range(20):
        _, s = p.step(hand(other, side="right"), 1.0 + i / 60, other, depth(other, "right"))
        assert p.locked_side == "left" and s == "TRACKING"
    # the hand is measured against the LOCKED (left) shoulder: it is now ~a shoulder width to the right
    assert p.last_n[1] < -0.3
    p.calibrate(hand(other, side="right"), other)  # a new 'c' with the OTHER arm's hand never locks it
    assert p.locked_side is None  # only the configured --hand arm (left) may lock


@pytest.mark.parametrize("mirrored", [False, True])
def test_lock_uses_unmirroring(kin, gains, mirrored):
    po = pose(side="right", mirrored=mirrored)
    p = calibrated(kin, gains, po=po, mirrored=mirrored, want="right")  # --hand right
    assert p.locked_side == "right"


def test_lock_waits_for_pose_when_c_has_no_body(kin, gains):
    p = ReachPipeline(kin, None, gains, None, want="right")
    p.calibrate(hand(), None)
    assert p.locked_side is None
    po = pose(side="right")
    _, s = p.step(hand(po, side="right"), 0.0, po, depth(po, "right"))
    assert p.locked_side == "right" and s == "TRACKING"


def test_locked_shoulder_not_visible_is_lost(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    q0, _ = p.step(hand(po), 0.0, po, depth(po))
    bad = pose()
    bad.vis[side_indices("left", False)[0]] = 0.1
    q1, s1 = p.step(hand(po), COAST_S / 2, bad, depth(po))
    assert s1 == "COAST" and np.array_equal(q0, q1)
    # the OTHER arm's wrist/elbow being guessed doesn't matter any more (it is not used)
    ok = pose()
    ok.vis[[14, 16]] = 0.1
    assert p.step(hand(po), COAST_S, ok, depth(po))[1] == "TRACKING"


def test_shoulder_sanity_checks_still_hold(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    q0, _ = p.step(hand(po), 0.0, po, depth(po))
    bad = pose()
    bad.img[[11, 12], :2] = [[0.481, 0.405], [0.502, 0.368]]  # recorded glitch at row 1315
    q1, _ = p.step(hand(po), 0.04, bad, depth(po))
    assert np.array_equal(q0, q1) and p.tracking_hold_reason == "shoulder span collapsed or changed"
    bad = pose()
    bad.img[[11, 12], 1] += 0.4
    q2, _ = p.step(hand(po), 0.08, bad, depth(po))
    assert np.array_equal(q0, q2) and p.tracking_hold_reason == GLITCH
    _, s = p.step(hand(po), 0.12, po, depth(po))
    assert s == "TRACKING" and p.tracking_hold_reason is None  # no multi-frame reacquisition hold


def test_real_body_move_is_accepted_after_consistent_frames(kin, gains):
    """Tray's rule (2026-10-18): a one-frame shoulder jump is a glitch - reject it and hold the last good
    command; a NEW position that stays consistent for SHOULDER_CONFIRM_FRAMES is a real move - accept it as
    the new baseline automatically (no 'c', no long pause)."""
    from cta.pipeline import SHOULDER_CONFIRM_FRAMES
    p = calibrated(kin, gains)
    po = pose()
    p.step(hand(po), 0.0, po, depth(po))
    moved = pose()
    moved.img[:, 0] += 0.25  # the whole body 0.25 image widths to the side
    t = 0.04
    for i in range(SHOULDER_CONFIRM_FRAMES - 1):  # still confirming: held
        _, s = p.step(hand(moved, side="left"), t, moved, depth(moved))
        assert s != "TRACKING" and p.tracking_hold_reason == GLITCH
        t += 0.04
    _, s = p.step(hand(moved, side="left"), t, moved, depth(moved))
    assert s == "TRACKING" and p.locked_side == "left"  # accepted, no re-lock needed


def test_flickering_shoulder_is_never_accepted(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    q0, _ = p.step(hand(po), 0.0, po, depth(po))
    a, b = pose(), pose()
    a.img[[11, 12], 0] += 0.25
    b.img[[11, 12], 0] -= 0.25
    t = 0.04
    for i in range(20):  # two glitch positions alternating: never consistent, never accepted
        bad = a if i % 2 == 0 else b
        q, s = p.step(hand(po), t, bad, depth(po))
        assert s != "TRACKING" and np.array_equal(q, q0)
        t += 0.04
    _, s = p.step(hand(po), t, po, depth(po))
    assert s == "TRACKING"  # the next valid frame near the last good position resumes immediately


def test_pose_wrist_teleport_does_not_move_the_target(kin, gains):
    """Root cause #2: the Pose wrist jumps 0.15 image widths; the hand tracker doesn't. Target must not care."""
    p = calibrated(kin, gains, home_relative=True)
    po = pose(0, 120, 30)
    run(p, hand(po), po, n=30)
    before = p.raw_target.copy()
    glitch = pose(0, 120, 30)
    glitch.img[15, :2] += [0.15, -0.1]
    glitch.world[15] += [0.2, -0.2, 0.3]
    p.step(hand(po), 1.0, glitch, depth(po))
    assert np.allclose(p.raw_target, before, atol=1e-9)


def test_home_relative_target_is_p0_plus_reach_times_delta_n(kin, gains):
    p = calibrated(kin, gains, home_relative=True)
    neutral = pose(0, 120, 30)
    run(p, hand(neutral), neutral)
    assert np.allclose(p.raw_target, p.cal.p0, atol=1e-6)  # home -> the calibrated ready-pose gripper
    up = HandObservation(*make_hand(wrist_uv=(neutral.img[15, 0], neutral.img[15, 1] - 0.0595)), "left", 0.9)
    run(p, up, neutral, n=5, t0=2.0, dmap=None)  # depth held (the disc stays put)
    dn_up = 0.0595 / (ARM_PER_SHOULDER * 0.35)
    assert p.raw_target[2] == pytest.approx(min(p.cal.p0[2] + p.robot_reach * dn_up, p.box_max[2]), abs=1e-3)
    assert np.allclose(p.raw_target[:2], p.cal.p0[:2], atol=2e-3)
    p.calibrate(up, neutral)  # a new 'c' makes this the home
    run(p, up, neutral, n=5, t0=4.0, dmap=None)
    assert np.allclose(p.raw_target, p.cal.p0, atol=1e-6)


def test_home_relative_forward_from_depth(kin, gains):
    p = calibrated(kin, gains, home_relative=True, fwd_gain=1.0)
    po = pose(0, 120, 30)
    run(p, hand(po), po, n=10)
    closer = depth(po)
    closer[closer < closer.max()] -= 0.10  # the hand disc 10 cm closer to the camera
    run(p, hand(po), po, n=5, t0=1.0, dmap=closer)
    want = p.robot_reach * 0.10 / (ARM_PER_SHOULDER * 0.35)
    assert p.raw_target[0] - p.cal.p0[0] == pytest.approx(min(want, p.box_max[0] - p.cal.p0[0]), abs=2e-3)


def test_depth_skipped_frames_hold_the_last_depth(kin, gains):
    p = calibrated(kin, gains, home_relative=True)
    po = pose(0, 120, 30)
    closer = depth(po)
    closer[closer < closer.max()] -= 0.10
    run(p, hand(po), po, n=10)
    run(p, hand(po), po, n=2, t0=1.0, dmap=closer)
    t1 = p.raw_target.copy()
    run(p, hand(po), po, n=3, t0=2.0, dmap=None)  # no depth this frame (every-2nd-frame mode): hold d
    assert np.allclose(p.raw_target, t1)


def test_depth_fallback_uses_hand_scale(kin, gains):
    p = calibrated(kin, gains, home_relative=True, fwd_source="scale")
    po = pose(0, 120, 30)
    wr = tuple(po.img[15, :2])
    run(p, HandObservation(*make_hand(wrist_uv=wr), "left", 0.9), po, n=10, dmap=None)
    assert np.allclose(p.raw_target, p.cal.p0, atol=1e-6)
    run(p, HandObservation(*make_hand(scale=1.3, wrist_uv=wr), "left", 0.9), po, n=5, t0=1.0, dmap=None)
    assert p.raw_target[0] > p.cal.p0[0] + 0.03  # bigger hand = closer = forward
    assert np.allclose(p.raw_target[1:], p.cal.p0[1:], atol=2e-3)


def test_absolute_extended_forward_reaches_forward(kin, gains):
    p = calibrated(kin, gains)
    po = pose(0, 90)
    q, s = run(p, hand(po), po)
    assert s == "TRACKING"
    n = np.array([0.53 / (ARM_PER_SHOULDER * 0.35), 0.0, 0.0])
    assert np.allclose(p.last_n, n, atol=1e-6)
    ideal = kin.shoulder_anchor() + REACH_FRACTION * kin.max_reach() * n
    assert np.allclose(p.target, np.clip(ideal, p.box_min, p.box_max), atol=1e-6)
    assert np.linalg.norm(kin.fk(q)[0] - p.target) < 0.01


@pytest.mark.parametrize("az,elev,flex", [(0, 90, 0), (60, 90, 0), (0, 120, 30), (-40, 70, 60), (0, 60, 90)])
def test_gripper_lands_on_target(kin, gains, contact, az, elev, flex):
    p = calibrated(kin, gains)
    po = pose(az, elev, flex)
    q, s = run(p, hand(po), po)
    assert s == "TRACKING"
    assert np.all(p.target >= p.box_min - 1e-9) and np.all(p.target <= p.box_max + 1e-9)
    pos = kin.fk(q)[0]
    assert np.linalg.norm(pos - p.ik_target) < 0.01           # lands on the (clearance-adjusted) IK target
    assert np.linalg.norm(pos[:2] - p.target[:2]) < 0.01      # and x, y are where the user reached
    assert p.ik_target[2] >= p.target[2] - 1e-9              # clearance only ever raises z


def test_reach_left_goes_to_robot_left(kin, gains):
    p = calibrated(kin, gains)
    po = pose(70, 90)
    q, _ = run(p, hand(po), po)
    assert p.target[1] > 0.1 and kin.fk(q)[0][1] > 0.1


def test_reach_down_lowers_target_but_not_below_floor(kin, gains, contact):
    p = calibrated(kin, gains)
    po = pose(0, 90)
    run(p, hand(po), po, n=5)
    z_level = p.target[2]
    po = pose(0, 150)  # down-forward
    q, _ = run(p, hand(po), po, t0=1.0)
    assert p.target[2] < z_level - 0.1
    assert p.target[2] >= contact.table_z + TABLE_CLEARANCE_M + CLEARANCE_PAD_M - 1e-9
    assert contact.lowest_z(q) >= contact.table_z + TABLE_CLEARANCE_M + CLEARANCE_PAD_M - 1e-3


@pytest.mark.parametrize("home_relative", [False, True])
def test_mirrored_gives_same_command(kin, gains, home_relative):
    a0, b0 = pose(0, 120, 30), pose(0, 120, 30, mirrored=True)
    pa = calibrated(kin, gains, po=a0, home_relative=home_relative)
    pb = calibrated(kin, gains, po=b0, mirrored=True, home_relative=home_relative)
    run(pa, hand(a0), a0, n=5)
    run(pb, hand(b0), b0, n=5)
    a, b = pose(25, 100, 40), pose(25, 100, 40, mirrored=True)
    qa, _ = run(pa, hand(a), a, t0=1.0)
    qb, _ = run(pb, hand(b), b, t0=1.0)
    assert np.allclose(pa.target, pb.target) and np.allclose(qa, qb)
    assert pa.target[1] > pa.cal.p0[1] + 0.02 if home_relative else pa.target[1] > 0.02  # user's left: robot's left


def test_engagement_and_no_hand_hold(kin, gains):
    from cta.pipeline import HAND_ENGAGE_S
    fresh = calibrated(kin, gains)
    assert fresh.step(None, 0.0, pose())[1] == "NO HAND"  # body alone never engages
    p = calibrated(kin, gains)
    po = pose(0, 90)
    q0, _ = run(p, hand(po), po, n=10)
    q1, s1 = p.step(None, 9 / 60 + HAND_ENGAGE_S + 0.1, pose(60, 90))
    assert s1 == "NO HAND" and np.array_equal(q0, q1)


def test_visibility_gating_and_coast(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    q0, s0 = p.step(hand(po), 0.0, po, depth(po))
    assert s0 == "TRACKING"
    bad = pose(60, 90, vis=0.1)
    q1, s1 = p.step(hand(po), COAST_S / 2, bad)
    assert s1 == "COAST" and np.array_equal(q0, q1)
    q2, s2 = p.step(hand(po), COAST_S * 3, None)
    assert s2 == "NO HAND" and np.array_equal(q0, q2)
    bare = PoseObservation(po.img, po.world, 0.9)  # no visibility at all: gated out (fails closed)
    assert p.step(hand(po), COAST_S * 4, bare)[1] == "NO HAND"


def test_snap_gripper(kin, gains):
    lo, hi = SIM_LIMITS[5]
    p = calibrated(kin, gains)
    po = pose()
    q, _ = run(p, hand(po, pinch_closed=True), po, n=8)
    assert q[5] == pytest.approx(lo)
    q, _ = run(p, hand(po), po, n=8, t0=1.0)
    assert q[5] == pytest.approx(hi)


def test_hand_pitch_drives_the_claw(kin, gains):
    p = calibrated(kin, gains)
    po = pose(0, 45, 90)  # mid-height target: no table clearance in play
    q, _ = run(p, hand(po, pitch=-0.3), po)
    assert kin.fk(q)[1] == pytest.approx(p.cal.pitch_home - 0.3, abs=D(4))


def test_no_hand_holds_every_joint(kin, gains):
    p = calibrated(kin, gains)
    po = pose(0, 45, 90)
    q0, _ = run(p, hand(po, pinch_closed=True, roll=0.4), po, n=40)
    q1, s1 = p.step(None, 40 / 60 + COAST_S / 2, po)
    assert s1 == "COAST" and np.array_equal(q1, q0)
    for i in range(30):
        q, s = p.step(None, 40 / 60 + 0.2 + i / 60, po)
    assert s == "NO HAND" and np.array_equal(q, q0)


class SpySmoother:
    """Records what it is asked to smooth; halves each step (heavy smoothing)."""

    def __init__(self):
        self.shapes, self.prev = [], None
        self.min_cutoff, self.beta, self.d_cutoff = 1.0, 0.0, 1.0

    def reset(self):
        self.prev = None

    def __call__(self, x, t):
        x = np.asarray(x, float)
        self.shapes.append(x.shape)
        self.prev = x if self.prev is None else (self.prev + x) / 2
        return self.prev.copy()


def test_smoothing_is_on_the_target_not_the_joints(kin, gains):
    p = calibrated(kin, gains)
    spy = SpySmoother()
    p.set_smoother(spy)
    a, b = pose(0, 90), pose(70, 90)
    run(p, hand(a), a, n=30)
    raw_a = p.raw_target.copy()
    p.step(hand(b), 1.0, b, depth(b))
    assert set(spy.shapes) == {(3,)}  # the 3-D target, before IK
    assert np.allclose(p.target, (raw_a + p.raw_target) / 2, atol=1e-6)


def test_table_margin_floor(kin, gains, contact):
    big = 0.06
    p = calibrated(kin, gains, table_margin=big)
    assert p.box_min[2] == pytest.approx(contact.table_z + big + CLEARANCE_PAD_M)
    po = pose(0, 170)
    q, _ = run(p, hand(po), po)
    assert p.target[2] >= contact.table_z + big + CLEARANCE_PAD_M - 1e-9
    assert contact.lowest_z(q) >= contact.table_z + big + CLEARANCE_PAD_M - 1e-3


def test_reach_to_table_gets_the_jaws_down_to_the_margin(kin, gains, contact):
    """Picking a ~2.5 cm block: near the table, clearance raises the TARGET (x, y kept), not the shoulder."""
    p = calibrated(kin, gains)
    po = pose(0, 120, 30)  # reaching down-forward to the table
    q, s = run(p, hand(po), po, n=120)
    assert s == "TRACKING"
    pos = kin.fk(q)[0]
    assert np.linalg.norm(pos[:2] - p.target[:2]) < 0.01
    clearance = contact.lowest_z(q) - contact.table_z  # exact meshes
    assert TABLE_CLEARANCE_M <= clearance <= TABLE_CLEARANCE_M + 0.003


def test_roll_is_smoothed_and_wrap_aware(kin, gains):
    from cta.filters import OneEuro
    p = calibrated(kin, gains)
    p.set_smoother(OneEuro(min_cutoff=0.5, beta=0.0))
    po = pose(0, 45, 90)
    dm = depth(po)
    outs = []
    for i in range(60):  # jitter +-0.1 rad around 0.4
        p.step(hand(po, roll=0.4 + (0.1 if i % 2 else -0.1)), i / 60, po, dm)
        outs.append(p.roll_target)
    assert np.ptp(outs[-20:]) < 0.05  # much less than the 0.2 rad input jitter
    for i in range(60):  # jitter across the +-pi wrap: must not average to ~0
        p.step(hand(po, roll=np.pi - 0.05 if i % 2 else -np.pi + 0.05), 1.0 + i / 60, po, dm)
    assert abs(p.roll_target) > 3.0


def test_reach_line_and_recorder_fields(kin, gains):
    p = calibrated(kin, gains)
    po = pose(0, 120, 30)
    run(p, hand(po), po, n=3)
    line = p.reach_line_uv
    assert line.shape == (2, 2)
    assert np.allclose(line[0], po.img[11, :2]) and np.allclose(line[1], po.img[15, :2])  # shoulder -> hand
    assert np.isfinite(p.depth_hand) and np.isfinite(p.depth_shoulder) and p.depth_hand < p.depth_shoulder
    p.step(None, 5.0, po)
    assert p.reach_line_uv is None and np.isnan(p.depth_hand)


# ---- review round 2026-10-18: forward home, depth sanity, staleness, lock visibility --------------------

def closer_map(po, dz, side="left"):
    m = depth(po, side)
    m[m < m.max()] -= dz  # the hand disc dz closer to the camera
    return m


def test_forward_home_is_a_half_second_median_and_rejects_impossible_depth(kin, gains):
    """take 04: the first depth sample hit the background (d = +0.78 m, hand 'behind' the shoulder) and
    became d_home. Now: impossible d is rejected, d_home = median of >= 0.5 s of valid samples, and
    forward is held at home until then."""
    p = ReachPipeline(kin, None, gains, None, home_relative=True, hand_point="wrist", fwd_gain=1.0)
    po = pose(0, 120, 30)
    p.calibrate(hand(po), po)
    bad = closer_map(po, -(0.78 + 0.37))  # hand disc reads 0.78 m BEHIND the shoulder
    p.step(hand(po), 0.0, po, bad)
    assert p._d is None  # rejected outright
    for i in range(1, 12):  # 0.04 .. 0.44 s: the hand drifts 5 cm closer; forward held until home exists
        p.step(hand(po), i * 0.04, po, closer_map(po, 0.05 * (i % 2)))
        assert p.raw_target[0] == pytest.approx(p.cal.p0[0])
    for i in range(12, 20):
        p.step(hand(po), i * 0.04, po, depth(po))
    assert p._d_home == pytest.approx(-0.37, abs=0.06)
    assert p.raw_target[0] == pytest.approx(p.cal.p0[0], abs=0.02)


def test_fwd_gain_scales_relative_forward(kin, gains):
    out = []
    for g in (1.0, 0.5):
        p = calibrated(kin, gains, home_relative=True, fwd_gain=g)
        po = pose(0, 120, 30)
        run(p, hand(po), po, n=40)
        run(p, hand(po), po, n=5, t0=2.0, dmap=closer_map(po, 0.05))
        out.append(p.raw_target[0] - p.cal.p0[0])
    assert out[0] > 0.02 and out[1] == pytest.approx(out[0] / 2, abs=1e-3)


def test_scale_fallback_home_is_a_median_not_one_frame(kin, gains):
    p = ReachPipeline(kin, None, gains, None, home_relative=True, fwd_source="scale", fwd_gain=1.0)
    po = pose(0, 120, 30)
    wr = tuple(po.img[15, :2])
    p.calibrate(hand(po), po)
    p.step(HandObservation(*make_hand(scale=1.6, wrist_uv=wr), "left", 0.9), 0.0, po)  # one odd first frame
    for i in range(1, 20):
        p.step(HandObservation(*make_hand(wrist_uv=wr), "left", 0.9), i * 0.04, po)
    assert np.allclose(p.raw_target, p.cal.p0, atol=1e-6)  # the outlier didn't become the home


def test_stale_shoulder_depth_holds_forward(kin, gains):
    """Hand over the shoulder (shoulder unsampled) for > 1 s after a gap: don't trust the old shoulder depth."""
    p = calibrated(kin, gains, home_relative=True, fwd_gain=1.0)
    po = pose(0, 120, 30)
    run(p, hand(po), po, n=40)
    x0 = p.raw_target[0]
    over = pose(0, 90)  # hand right in front of the shoulder: the shoulder patch is covered
    run(p, hand(over), over, n=10, t0=3.0, dmap=closer_map(over, 0.3))
    assert p.raw_target[0] == pytest.approx(x0)


def test_depth_median_restarts_after_a_gap(kin, gains):
    p = calibrated(kin, gains, home_relative=True, fwd_gain=1.0)
    po = pose(0, 120, 30)
    run(p, hand(po), po, n=40)
    p.step(None, 1.0, po)
    p.step(hand(po), 1.5, po, closer_map(po, 0.1))  # first frame after a > COAST_S gap
    assert p._d == pytest.approx(-0.37 - 0.1, abs=0.02)  # not a median with pre-gap samples


def test_depth_failed_switches_to_hand_scale_without_a_jump(kin, gains):
    p = calibrated(kin, gains, home_relative=True, fwd_gain=1.0)
    po = pose(0, 120, 30)
    run(p, hand(po), po, n=40)
    run(p, hand(po), po, n=5, t0=2.0, dmap=closer_map(po, 0.05))
    x0 = p.raw_target[0]
    assert x0 > p.cal.p0[0] + 0.02
    p.depth_failed()
    assert p.fwd_source == "scale"
    run(p, hand(po), po, n=40, t0=3.0, dmap=None)
    assert p.raw_target[0] == pytest.approx(x0, abs=1e-6)  # forward continues from where depth left it


def test_lock_needs_visible_landmarks_and_says_so(kin, gains, capsys):
    p = ReachPipeline(kin, None, gains, None)
    po = pose()
    po.vis[[15, 16]] = 0.1  # both wrists guessed
    p.calibrate(hand(po), po)
    assert p.locked_side is None
    ok = pose()
    p.step(hand(ok), 0.0, ok, depth(ok))
    assert p.locked_side == "left"
    assert "arm locked: left (press c to re-lock)" in capsys.readouterr().out


def test_depth_fields_are_per_frame(kin, gains):
    p = calibrated(kin, gains)
    po = pose(0, 120, 30)
    m = depth(po)
    m[:, :] = np.where(m < m.max(), m, np.nan)  # shoulder unreadable this frame
    p.step(hand(po), 0.0, po, m)
    assert np.isfinite(p.depth_hand)  # recorded whenever measured
    p.step(hand(po), 0.02, po, None)
    assert np.isnan(p.depth_hand) and np.isfinite(p.depth_shoulder)


# ---- face gate (review B1, 2026-10-18): shoulders only count when the face is in view ABOVE them ---------

def test_head_out_of_frame_holds(kin, gains):
    """Recorded take 02: Tray stood up, head/shoulders left the frame, Pose put confident 'shoulders' on
    the torso. With no face in view, shoulder readings are not trusted: hold."""
    from cta.pipeline import FACE_REASON
    p = calibrated(kin, gains)
    po = pose()
    q0, _ = p.step(hand(po), 0.0, po, depth(po))
    standing = pose()
    standing.img[[0, 2, 5], 1] = -0.05  # face above the top of the image
    q1, s = p.step(hand(po), 0.04, standing, depth(po))
    assert s != "TRACKING" and np.array_equal(q0, q1) and p.tracking_hold_reason == FACE_REASON


def test_phantom_shoulders_below_the_face_line_never_accepted(kin, gains):
    """Hallucinated shoulders on the torso that hold steady for > SHOULDER_CONFIRM_FRAMES are still
    rejected when the face is not above them (here: face below the phantom 'shoulders')."""
    from cta.pipeline import SHOULDER_CONFIRM_FRAMES
    p = calibrated(kin, gains)
    po = pose()
    q0, _ = p.step(hand(po), 0.0, po, depth(po))
    phantom = pose()
    phantom.img[[11, 12], 1] += 0.35            # 'shoulders' jumped down onto the torso ...
    phantom.img[[0, 2, 5], 1] = phantom.img[11, 1] + 0.05  # ... and the face is not above them
    t = 0.04
    for _ in range(SHOULDER_CONFIRM_FRAMES * 3):
        q, s = p.step(hand(po), t, phantom, depth(po))
        assert s != "TRACKING" and np.array_equal(q, q0)
        t += 0.04


def test_low_visibility_face_holds(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    p.step(hand(po), 0.0, po, depth(po))
    dim = pose()
    dim.vis[[0, 2, 5]] = 0.1
    assert p.step(hand(po), 0.04, dim, depth(po))[1] != "TRACKING"


def test_lock_only_the_configured_arm(kin, gains):
    """Tray (2026-10-18): it locked his RIGHT arm when he raised his right hand. Only the --hand arm may lock;
    a hand on the other arm waits (holds) with a message."""
    from cta.pipeline import WRONG_HAND_REASON
    p = ReachPipeline(kin, None, gains, None, want="left", hand_point="wrist")
    other = pose(side="right")
    p.calibrate(hand(other, side="right"), other)
    _, s = p.step(hand(other, side="right"), 0.0, other, depth(other, "right"))
    assert p.locked_side is None and s != "TRACKING" and p.tracking_hold_reason == WRONG_HAND_REASON
    po = pose(side="left")
    p.step(hand(po), 0.1, po, depth(po))
    assert p.locked_side == "left"


def test_lock_does_not_need_the_other_arms_wrist(kin, gains):
    """Tray (2026-10-19): seated with the right hand resting below the frame, 'c' never locked the left arm."""
    p = ReachPipeline(kin, None, gains, None, want="left", hand_point="wrist")
    po = pose(side="left")
    other_wr = side_indices("right", False)[2]
    po.vis[other_wr] = 0.05  # the other wrist is off-frame / guessed
    p.calibrate(hand(po), po)
    assert p.locked_side == "left"
    po2 = pose(side="left")
    po2.vis[side_indices("left", False)[2]] = 0.05  # but the configured arm's own wrist IS required
    q = ReachPipeline(kin, None, gains, None, want="left", hand_point="wrist")
    q.calibrate(hand(po2), po2)
    assert q.locked_side is None
