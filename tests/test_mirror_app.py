import time

import numpy as np
import pytest

from cta.apps import mirror
from cta.camera import CameraMode

REST = np.array([0.3, -1.2, 1.4, 0.5, 0.0, 0.2])
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
        return REST

    def send(self, q):
        FakeFollower.sent.append(q.copy())
        if len(FakeFollower.sent) >= 3:
            raise OSError("serial died")

    def close(self):
        closed.append("follower")


class FakeLeader:
    def __init__(self, port):
        pass

    def connect(self):
        pass

    def read(self):
        return REST

    def close(self):
        closed.append("leader")


def test_finally_closes_arms_when_send_raises(monkeypatch, tmp_path):
    for name, obj in [("Camera", FakeCam), ("HandTracker", FakeTracker), ("Twin", FakeTwin),
                      ("Dashboard", FakeDash), ("FollowerArm", FakeFollower), ("LeaderArm", FakeLeader)]:
        monkeypatch.setattr(mirror, name, obj)
    monkeypatch.setattr(mirror, "_start_keys", lambda ev: None)
    closed.clear()
    FakeFollower.sent.clear()
    with pytest.raises(OSError):
        mirror.main(["--robot", "so101", "--follower-port", "COM3", "--leader-port", "COM4",
                     "--mode", "paired", "--calib", str(tmp_path / "none.json")])
    assert sorted(closed) == ["follower", "leader"]
    # first command is at the real pose, not READY_Q
    assert np.max(np.abs(FakeFollower.sent[0] - REST)) < 0.05
