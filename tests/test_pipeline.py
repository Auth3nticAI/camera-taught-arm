import numpy as np
import pytest
from helpers import make_hand

from cta.config import READY_Q, SIM_LIMITS
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta.mapping import MapGains
from cta.pipeline import MirrorPipeline


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


@pytest.fixture(scope="module")
def gains(kin):
    return MapGains.from_kinematics(kin)


def obs(**kw):
    lm_img, lm_world = make_hand(**kw)
    return HandObservation(lm_img, lm_world, "right", 0.9)


def test_step_uncalibrated_holds_ready(kin, gains):
    p = MirrorPipeline(kin, None, gains, None)
    q, status = p.step(obs(), 0.0)
    assert status == "UNCALIBRATED"
    assert np.allclose(q, READY_Q)


def test_step_holds_last_command_when_no_hand(kin, gains):
    p = MirrorPipeline(kin, None, gains, None)
    p.calibrate(obs())
    q1, s1 = p.step(obs(wrist_uv=(0.4, 0.5)), 0.0)
    assert s1 == "TRACKING"
    q_coast, s_coast = p.step(None, 0.01)  # brief gap: COAST, same command (no jump)
    assert s_coast == "COAST" and np.array_equal(q1, q_coast)
    q2, s2 = p.step(None, 1.0)  # hand really gone: NO HAND, still holding
    assert s2 == "NO HAND"
    assert np.array_equal(q1, q2)
    assert p.last_features is None


def test_step_rate_limits_joint_jumps(kin, gains):
    p = MirrorPipeline(kin, None, gains, None, max_step_rad=0.15)
    p.calibrate(obs())
    prev, _ = p.step(obs(), 0.0)
    q, status = p.step(obs(wrist_uv=(1.0, 0.5)), 0.01)  # u += 0.5
    assert status == "TRACKING"
    # arm joints only: the snap gripper has its own faster limit (GRIP_STEP_RAD)
    assert np.max(np.abs(q[:5] - prev[:5])) == pytest.approx(0.15)  # clamping kicked in


def test_step_outputs_within_limits(kin, gains):
    rng = np.random.default_rng(1)
    p = MirrorPipeline(kin, None, gains, None)
    p.calibrate(obs())
    for i in range(100):
        o = obs(scale=rng.uniform(0.6, 1.6), pitch=rng.uniform(-1, 1), roll=rng.uniform(-1.5, 1.5),
                pinch_closed=bool(rng.integers(2)), wrist_uv=tuple(rng.uniform(0.1, 0.9, 2)))
        q, _ = p.step(o, i * 0.03)
        assert q.shape == (6,) and np.all(np.isfinite(q))
        assert np.all(q >= SIM_LIMITS[:, 0] - 1e-9) and np.all(q <= SIM_LIMITS[:, 1] + 1e-9)


def test_step_nonfinite_target_holds(kin, gains):
    p = MirrorPipeline(kin, None, gains, None)
    p.calibrate(obs())
    q1, _ = p.step(obs(wrist_uv=(0.4, 0.5)), 0.0)
    bad = obs()
    bad.lm_img[:] = np.nan
    q2, status = p.step(bad, 0.01)
    assert status == "NO HAND" and np.array_equal(q1, q2)


def test_step_uncalibrated_still_records_features(kin, gains):
    p = MirrorPipeline(kin, None, gains, None)
    q, status = p.step(obs(), 0.0)
    assert status == "UNCALIBRATED" and np.allclose(q, READY_Q)
    assert p.last_features is not None
    p.step(None, 0.01)
    assert p.last_features is None


def _calibrated(kin, gains, **kw):
    p = MirrorPipeline(kin, None, gains, None, **kw)
    p.calibrate(obs())  # open hand
    return p


def test_gripper_snaps_closed_and_open_with_hysteresis(kin, gains):
    lo, hi = SIM_LIMITS[5]
    p = _calibrated(kin, gains)
    for i in range(6):
        q, _ = p.step(obs(pinch_closed=True), 0.1 * (i + 1))
    assert q[5] == pytest.approx(lo)  # fully closed within a few frames
    for i in range(6):
        q, _ = p.step(obs(), 1.0 + 0.1 * i)
    assert q[5] == pytest.approx(hi)  # fully open again


def test_gripper_holds_state_inside_deadzone(kin, gains):
    from cta.pipeline import GRIP_CLOSE_FRAC, GRIP_OPEN_FRAC

    assert GRIP_CLOSE_FRAC < GRIP_OPEN_FRAC
    lo, hi = SIM_LIMITS[5]
    p = _calibrated(kin, gains)
    for i in range(6):
        p.step(obs(pinch_closed=True), 0.1 * i)
    mid = (GRIP_CLOSE_FRAC + GRIP_OPEN_FRAC) / 2  # inside the dead-zone: stays closed
    assert p._grip_state(mid) is True


def test_gripper_bypasses_arm_smoother(kin, gains):
    from cta.filters import OneEuro

    lo, _ = SIM_LIMITS[5]
    p = _calibrated(kin, gains)
    p.set_smoother(OneEuro(min_cutoff=0.05))  # very heavy arm smoothing
    for i in range(6):
        q, _ = p.step(obs(pinch_closed=True), 0.016 * (i + 1))
    assert q[5] == pytest.approx(lo)  # gripper unaffected by the slow arm filter


def test_brief_dropout_coasts_then_reports_no_hand(kin, gains):
    from cta.pipeline import COAST_S

    p = _calibrated(kin, gains)
    q0, s0 = p.step(obs(), 0.0)
    assert s0 == "TRACKING"
    q1, s1 = p.step(None, COAST_S / 2)  # short gap: keep going toward the last target
    assert s1 == "COAST" and np.allclose(q1, q0)
    q2, s2 = p.step(None, COAST_S * 2)  # long gap: hold as before
    assert s2 == "NO HAND"
