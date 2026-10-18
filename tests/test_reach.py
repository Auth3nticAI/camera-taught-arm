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


def test_reach_holds_when_tracker_switches_to_other_arm(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    q0, _ = p.step(hand(po), 0.0, po)
    other = pose(side='right')
    lm_img, lm_world = make_hand(wrist_uv=tuple(other.img[16, :2]))
    obs = HandObservation(lm_img, lm_world, 'right', 0.9)
    q1, status = p.step(obs, 0.03, other)
    assert np.array_equal(q0, q1)
    assert status == 'COAST' and p.tracking_hold_reason == 'arm identity changed'


def test_reach_holds_confident_collapsed_or_teleported_shoulders(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    q0, _ = p.step(hand(po), 0.0, po)
    bad = pose()
    bad.img[[11, 12], :2] = [[0.481, 0.405], [0.502, 0.368]]  # recorded glitch at row 1315
    q1, _ = p.step(hand(po), 0.04, bad)
    assert np.array_equal(q0, q1) and p.tracking_hold_reason == 'shoulder span collapsed or changed'
    bad = pose()
    bad.img[[11, 12], 1] += 0.4
    q2, _ = p.step(hand(po), 0.08, bad)
    assert np.array_equal(q0, q2) and p.tracking_hold_reason == 'shoulders jumped'


def test_reach_rejects_confident_world_jump_and_recovers_on_return(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    q0, _ = p.step(hand(po), 0.0, po)
    flipped = pose(0, 150)
    q1, _ = p.step(hand(flipped), 0.03, flipped)
    assert np.array_equal(q0, q1) and p.tracking_hold_reason == '3D reach jumped'
    for t in (0.06, 0.09, 0.12):
        _, status = p.step(hand(po), t, po)
    assert status == 'TRACKING' and p.tracking_hold_reason is None


def test_missing_wrist_holds_all_joints_and_requires_stable_reacquisition(kin, gains):
    p = calibrated(kin, gains, home_relative=True)
    po = pose(0, 60, 60)
    q0, _ = run(p, hand(po, roll=0.3), po, n=50)
    for t in (1.0, 1.2, 1.5):
        q, status = p.step(None, t, pose(50, 110))
        assert np.array_equal(q, q0) and status != 'TRACKING'
    changed = pose(45, 75, 45)
    for t in (1.53, 1.56):
        q, status = p.step(hand(changed), t, changed)
        assert np.array_equal(q, q0) and status != 'TRACKING'
    q, status = p.step(hand(changed), 1.59, changed)
    assert status == 'TRACKING'
    assert np.linalg.norm(p.target - p._held_target) < 1e-6  # clutch: no chase toward a new absolute hand position
    assert np.max(np.abs(q[:5] - q0[:5])) <= np.radians(90) * 0.03 + 1e-6


def test_alternating_depth_glitches_cannot_resume_by_waiting(kin, gains):
    p = calibrated(kin, gains, home_relative=True)
    good, bad = pose(0, 60, 60), pose(0, 150)
    q0, _ = p.step(hand(good), 0.0, good)
    for i in range(1, 80):
        po = bad if i % 2 else good
        q, status = p.step(hand(po), i / 30, po)
        assert np.array_equal(q, q0) and status != 'TRACKING'


def test_home_relative_reach_keeps_neutral_hand_above_table(kin, gains):
    p = calibrated(kin, gains, home_relative=True)
    neutral = pose(0, 120, 30)  # wrist naturally below the human shoulder
    run(p, hand(neutral), neutral)
    assert np.allclose(p.raw_target, p.cal.p0, atol=1e-6)
    assert p.raw_target[2] > 0.14  # neutral should not be clamped to table height
    higher = pose(0, 100, 30)
    run(p, hand(higher), higher, t0=2.0)
    assert p.raw_target[2] > p.cal.p0[2] + 0.04
    p.calibrate(hand(higher))
    run(p, hand(higher), higher, t0=4.0)
    assert np.allclose(p.raw_target, p.cal.p0, atol=1e-6)


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


def test_no_hand_holds_every_joint_without_fallback_pitch(kin, gains):
    p = calibrated(kin, gains)
    po = pose(0, 45, 90)
    q0, _ = run(p, hand(po, pinch_closed=True, roll=0.4), po, n=40)
    q1, s1 = p.step(None, 40 / 60 + COAST_S / 2, po)  # brief hand dropout: keep the last hand pitch
    assert s1 == "COAST" and np.array_equal(q1, q0)
    for i in range(30):  # hand gone 0.2 .. 0.7 s (still engaged): settle toward the default ~45 deg down
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
    outs = []
    for i in range(60):  # jitter +-0.1 rad around 0.4
        q, _ = p.step(hand(po, roll=0.4 + (0.1 if i % 2 else -0.1)), i / 60, po)
        outs.append(p.roll_target)
    assert np.ptp(outs[-20:]) < 0.05  # much less than the 0.2 rad input jitter
    for i in range(60):  # jitter across the +-pi wrap: must not average to ~0
        q, _ = p.step(hand(po, roll=np.pi - 0.05 if i % 2 else -np.pi + 0.05), 1.0 + i / 60, po)
    assert abs(p.roll_target) > 3.0
