import json
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
    calls = 0
    fail = staticmethod(lambda n: n >= 4)  # n = 1-based send() call index -> raise?
    read_fail = staticmethod(lambda n: False)
    reads = 0

    def __init__(self, port):
        pass

    def connect(self):
        pass

    def read(self):
        FakeFollower.reads += 1
        if FakeFollower.reads > 1 and FakeFollower.read_fail(FakeFollower.reads - 1):  # call 1 = startup read
            raise ConnectionError("read failed")
        return STATE["pose"]

    def send(self, q):
        FakeFollower.calls += 1
        if FakeFollower.fail(FakeFollower.calls):
            raise OSError("serial died")
        FakeFollower.sent.append(q.copy())

    def close(self):
        closed.append("follower")


class FakeLeader:
    def __init__(self, port):
        pass

    def connect(self):
        pass

    calls = 0
    fail = staticmethod(lambda n: False)

    def read(self):
        FakeLeader.calls += 1
        if FakeLeader.fail(FakeLeader.calls):
            raise ConnectionError("leader read failed")
        return STATE["leader"]

    def close(self):
        closed.append("leader")


def _run(monkeypatch, tmp_path, pose, leader, extra, map_file=True, fail=None, leader_fail=None, read_fail=None):
    for name, obj in [("Camera", FakeCam), ("HandTracker", FakeTracker), ("Twin", FakeTwin),
                      ("Dashboard", FakeDash), ("FollowerArm", FakeFollower), ("LeaderArm", FakeLeader)]:
        monkeypatch.setattr(mirror, name, obj)
    monkeypatch.setattr(mirror, "_start_keys", lambda ev: None)
    monkeypatch.setenv("CTA_DATA_DIR", str(tmp_path))
    if map_file:
        (tmp_path / "sim_real_map.json").write_text(
            json.dumps({"sign": [1] * 5, "offset_deg": [0] * 5}))
    closed.clear()
    FakeFollower.sent.clear()
    FakeFollower.calls = FakeFollower.reads = FakeLeader.calls = 0
    FakeFollower.fail = staticmethod(fail or (lambda n: n >= 4))
    FakeFollower.read_fail = staticmethod(read_fail or (lambda n: False))
    FakeLeader.fail = staticmethod(leader_fail or (lambda n: False))
    STATE.update(pose=pose, leader=leader)
    return mirror.main(["--robot", "so101", "--follower-port", "COM3", "--max-speed-dps", str(MAX_DPS),
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


PAIRED = ["--leader-port", "COM4", "--mode", "paired"]


def test_intermittent_bus_errors_do_not_end_loop(monkeypatch, tmp_path, capsys):
    # sends 3..6 raise, then recover; loop must survive and run all 40 frames
    rc = _run(monkeypatch, tmp_path, REST, None, ["--max-frames", "40"], fail=lambda n: 3 <= n <= 6)
    assert rc == 0
    assert FakeFollower.calls == 40 and len(FakeFollower.sent) == 36
    assert closed == ["follower"]
    assert "bus error" in capsys.readouterr().out


def test_intermittent_leader_errors_send_nothing_that_frame(monkeypatch, tmp_path):
    rc = _run(monkeypatch, tmp_path, REST, LEADER, ["--max-frames", "30", *PAIRED],
              leader_fail=lambda n: 5 <= n <= 8, fail=lambda n: False)
    assert rc == 0 and FakeLeader.calls == 30
    assert len(FakeFollower.sent) == 26  # 4 leader failures -> 4 skipped sends
    assert sorted(closed) == ["follower", "leader"]


def test_fifteen_consecutive_failures_exit_and_close(monkeypatch, tmp_path):
    rc = _run(monkeypatch, tmp_path, REST, None, ["--max-frames", "500"], fail=lambda n: n >= 3)
    assert rc == 1
    assert FakeFollower.calls == 2 + 15  # 2 good sends, then exactly 15 failures
    assert closed == ["follower"]


def test_fourteen_failures_then_success_resets_counter(monkeypatch, tmp_path):
    rc = _run(monkeypatch, tmp_path, REST, None, ["--max-frames", "60"],
              fail=lambda n: 3 <= n <= 16 or 18 <= n <= 31)
    assert rc == 0 and FakeFollower.calls == 60


def test_failed_reads_record_nan_and_take_continues(monkeypatch, tmp_path):
    from cta.recorder import load_session
    rc = _run(monkeypatch, tmp_path, REST, LEADER,
              ["--max-frames", "20", "--record", "t", "--autostart-record", *PAIRED],
              fail=lambda n: False, read_fail=lambda n: n in (3, 4))
    assert rc == 0
    d = load_session(tmp_path / "sessions" / "t_01")
    assert d["t"].shape == (20,)
    assert np.isnan(d["q_follower"][[2, 3]]).all()
    assert np.isfinite(d["q_follower"][[0, 1, 4]]).all()
    assert (d["status"] == 0).all()  # paired + uncalibrated: allowed, flagged UNCALIBRATED
    assert (d["smoother_on"] == 1).all()
    assert np.isfinite(d["q_leader"]).all()


def test_mirror_mode_refuses_to_record_uncalibrated(monkeypatch, tmp_path, capsys):
    rc = _run(monkeypatch, tmp_path, REST, None, ["--max-frames", "5", "--record", "t", "--autostart-record"])
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "sessions").exists() or not list((tmp_path / "sessions").iterdir())
    assert closed == ["follower"]


def test_refuses_without_sim_real_map(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, REST, None, ["--max-frames", "3"], map_file=False)
    assert FakeFollower.calls == 0


def test_allow_identity_map_overrides(monkeypatch, tmp_path):
    rc = _run(monkeypatch, tmp_path, REST, None, ["--max-frames", "3", "--allow-identity-map"],
              map_file=False, fail=lambda n: False)
    assert rc == 0 and FakeFollower.calls == 3


def test_dispatch_keys_injected():
    ev = set()
    mirror.dispatch_keys(["C", "x", None, "q", " ", "r"], ev)
    assert ev == {"c", "q", "r"}


def test_console_keys_poll_swallows_special_prefix():
    ev, queue = set(), list("s") + [chr(0xE0), "H"] + list("c")
    k = mirror._ConsoleKeys(ev, kbhit=lambda: bool(queue), getwch=lambda: queue.pop(0))
    k.poll()
    assert ev == {"s", "c"} and not queue


def test_sync_data_help_lists_config():
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "sync_data.py"
    out = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "config" in out.stdout


def test_stalled_arm_commands_stay_near_measured_pose(monkeypatch, tmp_path, capsys):
    # follower never moves (e.g. motors unpowered): commands must not drift toward the target
    _run(monkeypatch, tmp_path, REST, LEADER, PAIRED + ["--max-frames", "60"], fail=lambda n: False)
    assert len(FakeFollower.sent) >= 50
    for q in FakeFollower.sent:
        assert np.all(np.abs(q - REST) <= STEP)
    assert "not moving" in capsys.readouterr().out
