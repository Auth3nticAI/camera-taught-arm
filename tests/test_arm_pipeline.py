import numpy as np
import pytest
from helpers import bend, body_dir, make_body, make_hand

from cta.align import ContactModel
from cta.config import READY_Q, SIM_LIMITS
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta.mapping import MapGains
from cta.pipeline import CLEARANCE_BODIES, COAST_S, TABLE_CLEARANCE_M, ArmPosePipeline, clear_table
from cta.pose import PoseObservation

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


def hand(po=None, **kw):
    """Synthetic hand; with a pose, its wrist sits on that pose's raised (left-label) wrist in the image."""
    if po is not None:
        kw["wrist_uv"] = tuple(po.img[16 if po.mirrored_input else 15, :2])
    lm_img, lm_world = make_hand(**kw)
    return HandObservation(lm_img, lm_world, "left", 0.9)


def pose(az=0.0, elev=90.0, flex=60.0, side="left", mirrored=False):
    a = body_dir(az, elev)
    img, world = make_body(a, bend(a, flex), side=side, mirrored=mirrored)
    po = PoseObservation(img, world, 0.9)
    po.mirrored_input = mirrored  # test-only tag for hand()
    return po


def calibrated(kin, gains, **kw):
    p = ArmPosePipeline(kin, None, gains, None, **kw)
    p.calibrate(hand())
    return p


def run(p, hand_obs, pose_obs, n=60, t0=0.0):
    for i in range(n):
        q, s = p.step(hand_obs, t0 + i / 60, pose_obs)
    return q, s


def test_uncalibrated_holds_ready(kin, gains):
    p = ArmPosePipeline(kin, None, gains, None)
    q, s = p.step(hand(), 0.0, pose())
    assert s == "UNCALIBRATED" and np.allclose(q, READY_Q)
    assert p.last_features is not None


def test_tracks_shoulder_and_elbow(kin, gains):
    p = calibrated(kin, gains)
    po = pose(az=30, elev=60, flex=40)
    q, s = run(p, hand(po), po)
    assert s == "TRACKING"
    from cta.arm_angles import ELBOW_OFFSET_RAD, LIFT_OFFSET_RAD
    assert q[0] == pytest.approx(D(-30), abs=1e-3)
    assert q[1] == pytest.approx(D(60) - LIFT_OFFSET_RAD, abs=1e-3)
    assert q[2] == pytest.approx(D(40) - ELBOW_OFFSET_RAD, abs=1e-3)


def test_no_pose_coasts_then_no_hand(kin, gains):
    p = calibrated(kin, gains)
    q0, s0 = p.step(hand(), 0.0, pose())
    assert s0 == "TRACKING"
    q1, s1 = p.step(hand(), COAST_S / 2, None)  # hand visible but no pose: still a lost body
    assert s1 == "COAST" and np.array_equal(q0, q1)
    q2, s2 = p.step(None, COAST_S * 3, None)
    assert s2 == "NO HAND" and np.array_equal(q0, q2)


def test_hand_missing_holds_wrist_and_gripper(kin, gains):
    p = calibrated(kin, gains)
    po = pose(az=0, elev=90, flex=60)
    q0, _ = run(p, hand(po, pinch_closed=True), po)
    q1, s1 = run(p, None, pose(az=-40, elev=70, flex=90), n=40, t0=1.0)  # hand seen < 1 s ago
    assert s1 == "TRACKING"
    assert not np.allclose(q0[:3], q1[:3], atol=0.1)  # shoulder/elbow still follow the body
    assert np.array_equal(q0[3:], q1[3:])             # wrist_flex, wrist_roll, gripper held
    assert p.last_features is None


def test_gripper_snaps(kin, gains):
    lo, hi = SIM_LIMITS[5]
    p = calibrated(kin, gains)
    q, _ = run(p, hand(pinch_closed=True), pose(), n=6)
    assert q[5] == pytest.approx(lo)
    q, _ = run(p, hand(), pose(), n=6, t0=1.0)
    assert q[5] == pytest.approx(hi)


@pytest.mark.parametrize("hand_pitch", [0.0, 0.4, -0.3])
def test_wrist_flex_holds_hand_pitch(kin, gains, hand_pitch):
    p = calibrated(kin, gains)  # calibrated at pitch 0
    po = pose(az=0, elev=50, flex=40)
    q, _ = run(p, hand(po, pitch=hand_pitch), po)
    _, world_pitch = kin.fk(q)
    assert world_pitch == pytest.approx(p.cal.pitch_home + hand_pitch, abs=D(1))


def test_rate_limited(kin, gains):
    p = calibrated(kin, gains, max_step_rad=0.15)
    prev, _ = p.step(hand(), 0.0, pose(az=0, elev=90, flex=60))
    q, _ = p.step(hand(), 1 / 60, pose(az=80, elev=90, flex=60))
    assert np.max(np.abs(q[:5] - prev[:5])) == pytest.approx(0.15)


def test_outputs_within_limits_and_above_table(kin, gains, contact):
    rng = np.random.default_rng(3)
    p = calibrated(kin, gains)
    for i in range(80):
        side = ["left", "right"][int(rng.integers(2))]
        po = pose(rng.uniform(-120, 120), rng.uniform(0, 180), rng.uniform(0, 160), side, bool(rng.integers(2)))
        q, _ = p.step(hand(pitch=rng.uniform(-1, 1)), i / 60, po)
        assert np.all(np.isfinite(q))
        assert np.all(q >= SIM_LIMITS[:, 0] - 1e-9) and np.all(q <= SIM_LIMITS[:, 1] + 1e-9)
        assert contact.lowest_z(q) >= contact.table_z + TABLE_CLEARANCE_M - 1e-3


def test_clear_table_raises_arm(contact):
    q = np.array([0.0, D(95), D(16), 0.0, 0.0, 0.5])  # arm hanging, forearm forward: jaws in the table
    assert contact.lowest_z(q) < contact.table_z + TABLE_CLEARANCE_M
    out = clear_table(q, contact)
    assert contact.lowest_z(out) >= contact.table_z + TABLE_CLEARANCE_M - 1e-3
    assert 0.0 <= out[1] < q[1]  # only shoulder_lift moved, toward 0 (up), and only as far as needed
    assert contact.lowest_z(out) < contact.table_z + TABLE_CLEARANCE_M + 0.005
    assert np.array_equal(np.delete(out, 1), np.delete(q, 1))


def test_clear_table_unfolds_elbow_when_lift_zero_is_not_enough(contact):
    q = np.array([0.0, D(90), D(40), D(60), 0.0, 0.5])  # elbow + wrist folded down: still low at lift 0
    q0 = q.copy()
    q0[1] = 0.0
    assert contact.lowest_z(q0) < contact.table_z + TABLE_CLEARANCE_M
    out = clear_table(q, contact)
    assert contact.lowest_z(out) >= contact.table_z + TABLE_CLEARANCE_M - 1e-3
    assert out[1] == 0.0 and out[2] < q[2]
    assert np.array_equal(out[3:], q[3:])


def test_clear_table_leaves_safe_pose(contact):
    assert np.array_equal(clear_table(READY_Q, contact), READY_Q)


def test_mirrored_pipeline_matches_unmirrored(kin, gains):
    pa = calibrated(kin, gains)
    pb = calibrated(kin, gains, mirrored=True)
    poa, pob = pose(az=20, elev=75, flex=50), pose(az=20, elev=75, flex=50, mirrored=True)
    qa, _ = run(pa, hand(poa), poa)
    qb, _ = run(pb, hand(pob), pob)
    assert np.allclose(qa, qb)


def test_reset_to_and_surface(kin, gains):
    p = ArmPosePipeline(kin, None, gains, None)
    p.reset_to(np.full(6, 10.0))
    assert np.all(p.q_cmd <= SIM_LIMITS[:, 1])
    for name in ("step", "calibrate", "set_pinch_closed", "reset_to", "set_smoother", "q_cmd", "cal",
                 "smoother", "last_features"):
        assert hasattr(p, name)


# ---- review fixes ------------------------------------------------------------------------------------

def low_vis(po, idx, v=0.1):
    po.vis = np.ones(33)
    po.vis[list(idx)] = v
    return po


def test_low_visibility_arm_is_no_pose(kin, gains):
    p = calibrated(kin, gains)
    po = pose()
    q0, s0 = p.step(hand(po), 0.0, po)
    assert s0 == "TRACKING"
    bad = low_vis(pose(az=-60), [13])  # elbow guessed (out of frame)
    q1, s1 = p.step(hand(po), COAST_S / 2, bad)
    assert s1 == "COAST" and np.array_equal(q0, q1)
    q2, s2 = p.step(hand(po), COAST_S * 3, bad)
    assert s2 == "NO HAND" and np.array_equal(q0, q2)
    bad2 = pose(az=-60)
    bad2.presence = np.ones(33)
    bad2.presence[12] = 0.2  # the other shoulder is also needed (body frame)
    assert p.step(hand(po), COAST_S * 4, bad2)[1] == "NO HAND"


def test_pose_min_vis_threshold_is_configurable(kin, gains):
    p = calibrated(kin, gains, pose_min_vis=0.05)
    po = low_vis(pose(), [13])
    assert p.step(hand(po), 0.0, po)[1] == "TRACKING"


def test_hidden_hips_still_track(kin, gains):
    from cta.arm_angles import ELBOW_OFFSET_RAD, LIFT_OFFSET_RAD
    p = calibrated(kin, gains)
    po = low_vis(pose(az=30, elev=60, flex=40), [23, 24])
    po.world[[23, 24]] = [[0.3, -0.2, 0.4], [0.1, 0.5, -0.3]]  # guessed hips
    q, s = run(p, hand(po), po)
    assert s == "TRACKING"
    assert q[0] == pytest.approx(D(-30), abs=1e-3)
    assert q[1] == pytest.approx(D(60) - LIFT_OFFSET_RAD, abs=1e-3)
    assert q[2] == pytest.approx(D(40) - ELBOW_OFFSET_RAD, abs=1e-3)


def test_body_without_hand_disengages_after_a_second(kin, gains):
    from cta.pipeline import HAND_ENGAGE_S
    p = calibrated(kin, gains)
    po = pose(az=0, elev=60, flex=40)
    q0, _ = run(p, hand(po), po, n=10)
    q1, s1 = p.step(None, 9 / 60 + HAND_ENGAGE_S / 2, pose(az=-40, elev=60, flex=40))
    assert s1 == "TRACKING" and not np.allclose(q0, q1)  # recently engaged: the body still drives
    q2, s2 = p.step(None, 9 / 60 + HAND_ENGAGE_S + 0.1, pose(az=40, elev=60, flex=40))
    assert s2 == "NO HAND" and np.array_equal(q1, q2)  # hand gone > 1 s: hold
    fresh = calibrated(kin, gains)
    assert fresh.step(None, 0.0, po)[1] == "NO HAND"  # body alone never engages


class _FloorEverywhere:
    table_z = 0.0

    def lowest_z(self, q):
        return -1.0  # every pose (even 'straight up') is in the table


def test_clear_table_reports_failure_instead_of_unsafe_pose():
    assert clear_table(np.array([0.0, 1.0, 1.0, 0.5, 0.0, 0.5]), _FloorEverywhere()) is None


def test_pipeline_holds_when_clearance_impossible(kin, gains, capsys):
    p = calibrated(kin, gains)
    po = pose(az=0, elev=60, flex=40)
    q0, _ = run(p, hand(po), po, n=5)
    p.contact = _FloorEverywhere()
    for i in range(3):
        q, s = p.step(hand(po), 1.0 + i / 60, pose(az=-50, elev=80, flex=60))
        assert np.array_equal(q, q0)
    assert p.clearance_failures == 3
    assert capsys.readouterr().out.count("clearance") == 1  # logged once


def test_clearance_checks_wrist_and_forearm(contact):
    assert set(CLEARANCE_BODIES) >= {"gripper", "moving_jaw_so101_v1", "wrist", "lower_arm"}
    q = D(np.array([0.0, 75.0, 48.0, 94.0, 0.0, 5.0]))  # jaws 2.1 cm up, but the wrist housing in the table
    assert ContactModel().lowest_z(q) >= contact.table_z + TABLE_CLEARANCE_M
    assert contact.lowest_z(q) < contact.table_z + TABLE_CLEARANCE_M
    assert contact.lowest_z(clear_table(q, contact)) >= contact.table_z + TABLE_CLEARANCE_M - 1e-3


def test_wrist_solve_stays_on_its_side_at_the_wrap(kin, gains):
    p = calibrated(kin, gains)
    lo, hi = SIM_LIMITS[3]
    q = np.array([0.0, D(30), D(-60), 0.0, 0.0, 0.5])
    base = kin.planar_pitch(q)
    q1 = q.copy()
    q1[3] = 0.5
    slope = np.sign(kin.planar_pitch(q1) - base)
    for w in (D(178), D(182), D(-178)):  # unreachable targets just either side of the +-180 wrap
        target = base + slope * w
        p.q_cmd[3] = hi
        assert p._wrist_flex_for_pitch(q, target) == pytest.approx(hi)
        p.q_cmd[3] = lo
        assert p._wrist_flex_for_pitch(q, target) == pytest.approx(lo)


def test_world_pitch_held_even_when_clearance_moves_the_arm(kin, gains):
    from cta.arm_angles import LIFT_OFFSET_RAD
    p = calibrated(kin, gains)
    po = pose(az=0, elev=80, flex=70)  # the forearm reaches down: clearance raises the arm
    q, s = run(p, hand(po), po)
    assert s == "TRACKING"
    assert q[1] < D(80) - LIFT_OFFSET_RAD - 0.05  # clearance did move the lift
    assert kin.fk(q)[1] == pytest.approx(p.cal.pitch_home, abs=D(1))


def test_fast_clearance_model_matches_exact(contact):
    from cta.pipeline import clearance_model
    fast = clearance_model()
    assert fast.table_z == contact.table_z
    rng = np.random.default_rng(7)
    for _ in range(300):
        q = rng.uniform(SIM_LIMITS[:, 0], SIM_LIMITS[:, 1])
        err = fast.lowest_z(q) - contact.lowest_z(q)
        assert 0.0 <= err < 2e-4  # never lower than exact; < 0.2 mm high
