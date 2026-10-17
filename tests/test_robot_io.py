import numpy as np
import pytest

from cta.config import READY_Q, SIM_LIMITS
from cta.robot_io import FollowerArm, LeaderArm, NullArm, clamp_margin, offset_from_reference
from cta.units import SimRealMap, real_to_sim, sim_to_real


def test_clamp_margin_keeps_inside():
    lo, hi = SIM_LIMITS[:, 0], SIM_LIMITS[:, 1]
    for q in (lo - 1.0, hi + 1.0, lo, hi):
        out = clamp_margin(q)
        assert np.all(out >= lo + 0.087 - 1e-12) and np.all(out <= hi - 0.087 + 1e-12)
    assert np.allclose(clamp_margin(hi), hi - 0.087)
    assert np.allclose(clamp_margin(lo), lo + 0.087)


def test_clamp_margin_identity_on_mid_values():
    assert np.allclose(clamp_margin(READY_Q), READY_Q)


def test_null_arm_read_returns_last_sent():
    arm = NullArm()
    arm.connect()
    assert arm.read() is None
    arm.send(READY_Q)
    r = arm.read()
    assert r.shape == (6,) and np.allclose(r, READY_Q)
    arm.close()


class _Uncal:
    is_calibrated = False

    def connect(self, calibrate=False):
        pass

    def disconnect(self):
        pass


class _Cal(_Uncal):
    is_calibrated = True

    def __init__(self):
        self.sent = None
        self.obs = {f"{j}.pos": 0.0 for j in
                    ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}

    def send_action(self, a):
        self.sent = a

    def get_observation(self):
        return {**self.obs, "front": object()}  # extra camera-like key must be ignored

    def get_action(self):
        return self.obs


def test_follower_raises_with_calibrate_command_when_uncalibrated():
    arm = FollowerArm("COM3", device=_Uncal())
    with pytest.raises(RuntimeError, match="lerobot-calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=cta_follower"):
        arm.connect()


def test_leader_raises_with_calibrate_command_when_uncalibrated():
    arm = LeaderArm("COM4", device=_Uncal())
    with pytest.raises(RuntimeError, match="lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=cta_leader"):
        arm.connect()


def test_follower_send_and_read_roundtrip():
    dev = _Cal()
    m = SimRealMap()
    arm = FollowerArm("COM3", device=dev, sim_real=m)
    arm.connect()
    arm.send(READY_Q)
    assert set(dev.sent) == set(sim_to_real(READY_Q, m))
    assert np.allclose(real_to_sim(dev.sent, m), READY_Q)
    assert arm.read().shape == (6,)
    arm.close()


def test_leader_read_shape():
    arm = LeaderArm("COM4", device=_Cal())
    arm.connect()
    assert arm.read().shape == (6,)


def test_offset_from_reference():
    # joint reads 30 deg when sim says 0.5 rad, sign +1
    off = offset_from_reference(30.0, 0.5, 1.0)
    assert np.isclose(1.0 * np.degrees(0.5) + off, 30.0)
    off = offset_from_reference(30.0, 0.5, -1.0)
    assert np.isclose(-1.0 * np.degrees(0.5) + off, 30.0)
