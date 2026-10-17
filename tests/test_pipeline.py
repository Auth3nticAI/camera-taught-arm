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
    q2, s2 = p.step(None, 0.01)
    assert s2 == "NO HAND"
    assert np.array_equal(q1, q2)
    assert p.last_features is None


def test_step_rate_limits_joint_jumps(kin, gains):
    p = MirrorPipeline(kin, None, gains, None, max_step_rad=0.15)
    p.calibrate(obs())
    prev, _ = p.step(obs(), 0.0)
    q, status = p.step(obs(wrist_uv=(1.0, 0.5)), 0.01)  # u += 0.5
    assert status == "TRACKING"
    assert np.max(np.abs(q - prev)) <= 0.15 + 1e-9
    assert np.max(np.abs(q - prev)) > 0.0


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
