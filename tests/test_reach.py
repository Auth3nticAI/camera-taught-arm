"""--control reach: the user's arm says WHERE (reach vector), the robot gets there with its own IK."""
import mujoco
import numpy as np
import pytest
from helpers import bend, body_dir, make_body, make_hand

from cta.align import ContactModel
from cta.config import SIM_LIMITS
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta.mapping import MapGains
from cta.pipeline import CLEARANCE_BODIES, CLEARANCE_PAD_M, COAST_S, TABLE_CLEARANCE_M, ReachPipeline
from cta.pose import PoseObservation
from cta.reach import NO_HAND_PITCH_RAD, REACH_FRACTION, ArmLength, reach_target, reach_vector

D = np.radians


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


# ---- reach vector ------------------------------------------------------------------------------------

def test_reach_vector_forward_extended():
    _, w = make_body(body_dir(0, 90))
    r, L = reach_vector(w, "left", False, True)
    assert np.allclose(r, [0.53, 0.0, 0.0], atol=1e-9) and L == pytest.approx(0.53)


@pytest.mark.parametrize("scale", [0.8, 1.2])
def test_arm_length_invariance(scale):
    a = body_dir(30, 70)
    _, w = make_body(a, bend(a, 50))
    _, ws = make_body(a, bend(a, 50), upper_len=0.28 * scale, fore_len=0.25 * scale)
    r, L = reach_vector(w, "left", False, True)
    rs, Ls = reach_vector(ws, "left", False, True)
    assert np.allclose(r / L, rs / Ls, atol=1e-9)


@pytest.mark.parametrize("side", ["left", "right"])
def test_reach_vector_mirrored_is_the_same(side):
    a = body_dir(-20, 100)
    _, w = make_body(a, bend(a, 30), side=side)
    _, wm = make_body(a, bend(a, 30), side=side, mirrored=True)
    assert np.allclose(reach_vector(w, side, False, True)[0], reach_vector(wm, side, True, True)[0], atol=1e-9)


def test_arm_length_median_ignores_a_bent_frame():
    al = ArmLength()
    for _ in range(20):
        al.update(0.53)
    assert al.update(0.30) == pytest.approx(0.53)  # one foreshortened frame doesn't shrink it
    assert ArmLength().update(0.5) == pytest.approx(0.5)  # initialized on the first good frame


def test_reach_target_clamps_and_clips():
    sh, reach = np.array([0.07, -0.02, 0.12]), 0.4
    lo, hi = np.array([0.2, -0.3, 0.01]), np.array([0.44, 0.3, 0.25])
    t = reach_target(np.array([2.0, 0.0, 0.0]), sh, reach, lo, hi)  # |n| > 1 is clamped to 1
    assert np.allclose(t, [0.44, -0.02, 0.12])
    t = reach_target(np.array([0.5, 0.0, -0.9]), sh, reach, lo, hi)
    assert t[2] == pytest.approx(0.01)  # never below the box floor


# ---- ReachPipeline -------------------------------------------------------------------------------------

def pose(az=0.0, elev=90.0, flex=0.0, side="left", mirrored=False, vis=1.0):
    a = body_dir(az, elev)
    img, world = make_body(a, bend(a, flex), side=side, mirrored=mirrored)
    po = PoseObservation(img, world, 0.9, np.full(33, vis), np.ones(33))
    po.mirrored_input = mirrored
    return po


def hand(po=None, **kw):
    if po is not None:
        kw["wrist_uv"] = tuple(po.img[16 if po.mirrored_input else 15, :2])
    lm_img, lm_world = make_hand(**kw)
    return HandObservation(lm_img, lm_world, "left", 0.9)


def calibrated(kin, gains, **kw):
    p = ReachPipeline(kin, None, gains, None, **kw)
    p.calibrate(hand())
    return p


def run(p, hand_obs, pose_obs, n=90, t0=0.0):
    for i in range(n):
        q, s = p.step(hand_obs, t0 + i / 60, pose_obs)
    return q, s


def test_extended_forward_reaches_forward(kin, gains):
    p = calibrated(kin, gains)
    po = pose(0, 90)
    q, s = run(p, hand(po), po)
    assert s == "TRACKING"
    ideal = kin.shoulder_anchor() + REACH_FRACTION * kin.max_reach() * np.array([1.0, 0.0, 0.0])
    assert np.allclose(p.target, np.clip(ideal, p.box_min, p.box_max), atol=1e-6)
    assert np.linalg.norm(kin.fk(q)[0] - p.target) < 0.01


@pytest.mark.parametrize("az,elev,flex", [(0, 90, 0), (60, 90, 0), (0, 120, 30), (-40, 70, 60), (0, 60, 90)])
def test_gripper_lands_on_target(kin, gains, contact, az, elev, flex):
    p = calibrated(kin, gains)
    po = pose(az, elev, flex)
    q, s = run(p, hand(po), po)
    assert s == "TRACKING"
    assert np.all(p.target >= p.box_min - 1e-9) and np.all(p.target <= p.box_max + 1e-9)
    if contact.lowest_z(q) > contact.table_z + TABLE_CLEARANCE_M + CLEARANCE_PAD_M + 1e-3:  # clearance idle
        assert np.linalg.norm(kin.fk(q)[0] - p.target) < 0.01


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


def test_mirrored_gives_same_command(kin, gains):
    pa, pb = calibrated(kin, gains), calibrated(kin, gains, mirrored=True)
    a, b = pose(25, 100, 40), pose(25, 100, 40, mirrored=True)
    qa, _ = run(pa, hand(a), a)
    qb, _ = run(pb, hand(b), b)
    assert np.allclose(pa.target, pb.target) and np.allclose(qa, qb)


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
    q0, s0 = p.step(hand(po), 0.0, po)
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


def test_no_hand_default_pitch_and_holds_roll_gripper(kin, gains):
    p = calibrated(kin, gains)
    po = pose(0, 45, 90)
    q0, _ = run(p, hand(po, pinch_closed=True, roll=0.4), po, n=40)
    q1, s1 = p.step(None, 40 / 60 + COAST_S / 2, po)  # brief hand dropout: keep the last hand pitch
    assert s1 == "TRACKING" and p.pitch_target == pytest.approx(p._hand_pitch)
    for i in range(30):  # hand gone 0.2 .. 0.7 s (still engaged): settle toward the default ~45 deg down
        q, s = p.step(None, 40 / 60 + 0.2 + i / 60, po)
    assert s == "TRACKING" and p.pitch_target == NO_HAND_PITCH_RAD
    assert kin.fk(q)[1] == pytest.approx(NO_HAND_PITCH_RAD, abs=D(5))
    assert q[4] == q0[4] and q[5] == q0[5]  # roll and gripper held


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
    p.step(hand(b), 1.0, b)
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
