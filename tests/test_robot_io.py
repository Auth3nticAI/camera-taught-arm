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


# ---- fix round 1: safety paths ----
from cta.robot_io import limit_speed, select_command  # noqa: E402

REST = np.array([0.3, -1.2, 1.4, 0.5, 0.0, 0.2])


def test_limit_speed_bounds_step_and_clamps_dt():
    out = limit_speed(np.zeros(6), np.ones(6), 0.01, 1.0)
    assert np.allclose(out, 0.01)
    assert np.allclose(limit_speed(np.zeros(6), np.ones(6), 5.0, 1.0), 0.1)  # dt clamped to 0.1
    assert np.allclose(limit_speed(np.zeros(6), np.ones(6), -1.0, 1.0), 0.0)
    assert np.allclose(limit_speed(np.zeros(6), np.full(6, 0.001), 0.1, 1.0), 0.001)  # no overshoot


def test_mirror_tracking_moves_toward_pipeline_speed_limited():
    out = select_command("mirror", "TRACKING", READY_Q, None, REST, 1 / 60, np.radians(90))
    assert np.all(np.abs(out - REST) <= np.radians(90) / 60 + 1e-12)
    assert np.linalg.norm(out - READY_Q) < np.linalg.norm(REST - READY_Q)


@pytest.mark.parametrize("status", ["UNCALIBRATED", "NO HAND"])
def test_mirror_not_tracking_holds_prev(status):
    out = select_command("mirror", status, READY_Q, None, REST, 1 / 60, np.radians(90))
    assert np.allclose(out, REST)


def test_paired_follows_leader_never_pipeline():
    leader = clamp_margin(REST + 0.5)
    out = select_command("paired", "TRACKING", READY_Q, leader, REST, 1 / 60, np.radians(90))
    assert np.all(np.abs(out - REST) <= np.radians(90) / 60 + 1e-12)
    assert np.all((out - REST) * (leader - REST) >= 0)  # toward leader
    held = select_command("paired", "TRACKING", READY_Q, None, REST, 1 / 60, np.radians(90))
    assert np.allclose(held, REST)


def test_follower_send_rejects_nonfinite_and_sends_nothing():
    dev = _Cal()
    arm = FollowerArm("COM3", device=dev)
    arm.connect()
    bad = READY_Q.copy()
    bad[2] = np.nan
    with pytest.raises(ValueError):
        arm.send(bad)
    assert dev.sent is None
    assert np.allclose(arm.send(READY_Q), READY_Q)


def test_simrealmap_validation(tmp_path):
    with pytest.raises(ValueError):
        SimRealMap(sign=np.array([1, 1, 1, 1, 2.0]))
    with pytest.raises(ValueError):
        SimRealMap(sign=np.ones(4))
    with pytest.raises(ValueError):
        SimRealMap(offset_deg=np.array([0, 0, np.inf, 0, 0.0]))
    p = tmp_path / "m.json"
    p.write_text('{"sign": [1,1,1,1,0.5], "offset_deg": [0,0,0,0,0]}')
    with pytest.raises(ValueError):
        SimRealMap.load(p)


def test_connect_failure_disconnects_bus():
    class Boom(_Uncal):
        disconnected = False

        def connect(self, calibrate=False):
            raise OSError("serial")

        def disconnect(self):
            self.disconnected = True

    d = Boom()
    with pytest.raises(OSError):
        FollowerArm("COM3", device=d).connect()
    assert d.disconnected


def test_make_configs(monkeypatch):
    import lerobot.robots.so_follower as sf
    seen = {}

    class FakeCfg:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(sf, "SO101FollowerConfig", FakeCfg)
    monkeypatch.setattr(sf, "SO101Follower", lambda cfg: "dev")
    assert FollowerArm("COM3")._make() == "dev"
    assert seen["max_relative_target"] == 10.0 and seen["id"] == "cta_follower" and seen["port"] == "COM3"


def test_pipeline_reset_to():
    from cta.kinematics import Kinematics
    from cta.mapping import MapGains
    from cta.pipeline import MirrorPipeline
    kin = Kinematics()
    p = MirrorPipeline(kin, None, MapGains.from_kinematics(kin), None)
    p.reset_to(REST)
    assert np.allclose(p.q_cmd, REST)


def test_send_returns_clipped_action_as_rad():
    class Clip(_Cal):
        def send_action(self, a):
            return {k: v * 0.5 for k, v in a.items()}

    arm = FollowerArm("COM3", device=Clip())
    arm.connect()
    out = arm.send(READY_Q)
    assert np.allclose(out[:5], READY_Q[:5] * 0.5)
