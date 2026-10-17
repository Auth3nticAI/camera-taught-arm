import time

import numpy as np
import pytest

from cta.apps import mirror
from cta.camera import CameraMode

REST = np.array([0.3, -1.2, 1.4, 0.5, 0.0, 0.2])
LEADER = np.array([-0.5, 0.4, -0.3, -0.8, 1.0, 1.2])  # far from REST
EDGE = np.array([0.3, -1.745, 1.69, 0.5, 0.0, -0.175])  # on/outside the 5 deg margin
STATE = {}
MAX_DPS = 90.0
STEP = np.radians(MAX_DPS) / 60 + 1e-9
closed = []


class FakeCam:
    def __init__(self, *a, **k):
        pass

    def is_open(self):
        return True

    def measure_mode(self):
        return CameraMode(1920, 1080, 60)

    def read(self):
        return np.zeros((8, 8, 3), np.uint8), time.perf_counter()

    def close(self):
        pass


class FakeTracker:
    def __init__(self, *a, **k):
        pass

    def detect(self, rgb, t):
        return None

    def close(self):
        pass


class FakeTwin:
    def render(self, q):
        return np.zeros((4, 4, 3), np.uint8)

    def close(self):
        pass


class FakeDash:
    def log_frame(self, *a, **k):
        pass


class FakeFollower:
    sent = []

    def __init__(self, port):
        pass

    def connect(self):
        pass

    def read(self):
        return STATE["pose"]

    def send(self, q):
        FakeFollower.sent.append(q.copy())
        if len(FakeFollower.sent) >= 4:
            raise OSError("serial died")

    def close(self):
        closed.append("follower")


class FakeLeader:
    def __init__(self, port):
        pass

    def connect(self):
        pass

    def read(self):
        return STATE["leader"]

    def close(self):
        closed.append("leader")


def _run(monkeypatch, tmp_path, pose, leader, extra):
    for name, obj in [("Camera", FakeCam), ("HandTracker", FakeTracker), ("Twin", FakeTwin),
                      ("Dashboard", FakeDash), ("FollowerArm", FakeFollower), ("LeaderArm", FakeLeader)]:
        monkeypatch.setattr(mirror, name, obj)
    monkeypatch.setattr(mirror, "_start_keys", lambda ev: None)
    closed.clear()
    FakeFollower.sent.clear()
    STATE.update(pose=pose, leader=leader)
    with pytest.raises(OSError):
        mirror.main(["--robot", "so101", "--follower-port", "COM3", "--max-speed-dps", str(MAX_DPS),
                     "--calib", str(tmp_path / "none.json"), *extra])


def test_finally_closes_arms_and_paired_does_not_lunge(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, REST, LEADER, ["--leader-port", "COM4", "--mode", "paired"])
    assert sorted(closed) == ["follower", "leader"]
    assert np.all(np.abs(FakeFollower.sent[0] - REST) <= STEP)  # leader far away: ramp, no jump
    assert np.any(np.abs(FakeFollower.sent[0] - REST) > 1e-4)


def test_mirror_uncalibrated_holds_rest_pose(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, REST, None, [])
    assert closed == ["follower"]
    for q in FakeFollower.sent:
        assert np.all(np.abs(q - REST) <= STEP)


def test_startup_pose_outside_margin_ramps_from_raw_pose(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, EDGE, None, [])
    assert np.all(np.abs(FakeFollower.sent[0] - EDGE) <= STEP)
