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
        if STATE.get("clock") is not None:  # deterministic 60 fps timestamps
            STATE["clock"] += 1 / 60
            return np.zeros((8, 8, 3), np.uint8), STATE["clock"]
        return np.zeros((8, 8, 3), np.uint8), time.perf_counter()

    def close(self):
        pass


class FakeTracker:
    obs = None  # HandObservation returned every frame (None = no hand)

    def __init__(self, *a, **k):
        pass

    def detect(self, rgb, t):
        return FakeTracker.obs

    def close(self):
        pass


class FakePose:
    obs = None        # PoseObservation returned every frame (None = no body)
    constructed = 0

    def __init__(self, *a, **k):
        FakePose.constructed += 1

    def detect(self, rgb, t):
        return FakePose.obs

    def close(self):
        pass


class FakeTwin:
    def render(self, q):
        return np.zeros((4, 4, 3), np.uint8)

    def close(self):
        pass


class FakeDash:
    arm_points = []
    reach = []

    def log_frame(self, *a, **k):
        if k.get("arm_uv") is not None:
            FakeDash.arm_points.append(k["arm_uv"])
        if k.get("reach_n") is not None:
            FakeDash.reach.append(k["reach_n"])


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
        if STATE.get("deadband") is not None:  # real servo: ignores commands within its deadband
            err = np.asarray(q) - STATE["pose"]
            STATE["pose"] = STATE["pose"] + np.where(np.abs(err) > STATE["deadband"], err, 0.0)

    def set_servo_p(self, p):
        STATE["servo_p"] = p

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


def _run(monkeypatch, tmp_path, pose, leader, extra, map_file=True, fail=None, leader_fail=None, read_fail=None,
         calib=None):
    for name, obj in [("Camera", FakeCam), ("HandTracker", FakeTracker), ("Twin", FakeTwin),
                      ("Dashboard", FakeDash), ("FollowerArm", FakeFollower), ("LeaderArm", FakeLeader),
                      ("PoseTracker", FakePose)]:
        monkeypatch.setattr(mirror, name, obj)
    monkeypatch.setattr(mirror, "_start_keys", lambda ev: None)
    monkeypatch.setenv("CTA_DATA_DIR", str(tmp_path))
    if map_file:
        (tmp_path / "sim_real_map.json").write_text(
            json.dumps({"sign": [1] * 5, "offset_deg": [0] * 5}))
    closed.clear()
    FakeDash.arm_points.clear()
    FakeDash.reach.clear()
    FakePose.constructed = 0
    FakeFollower.sent.clear()
    FakeFollower.calls = FakeFollower.reads = FakeLeader.calls = 0
    FakeFollower.fail = staticmethod(fail or (lambda n: n >= 4))
    FakeFollower.read_fail = staticmethod(read_fail or (lambda n: False))
    FakeLeader.fail = staticmethod(leader_fail or (lambda n: False))
    STATE.update(pose=pose, leader=leader)
    return mirror.main(["--robot", "so101", "--follower-port", "COM3", "--max-speed-dps", str(MAX_DPS),
                        "--calib", str(calib or tmp_path / "none.json"), *extra])


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
    from cta.apps.mirror import MAX_LEAD_RAD
    for q in FakeFollower.sent:  # commands never run more than MAX_LEAD_RAD ahead of the stalled arm
        assert np.all(np.abs(q - REST) <= MAX_LEAD_RAD + 1e-9)
    assert "not moving" in capsys.readouterr().out


def test_servo_deadband_does_not_freeze_the_arm(monkeypatch, tmp_path):
    # servos ignore tiny errors: ramping one small step ahead of the MEASURED pose never exceeds the
    # deadband, so the arm crawls/freezes (seen live 2026-10-17). Ramping from the last command works.
    STATE["deadband"] = np.radians(1.0)
    STATE["clock"] = 0.0
    try:
        _run(monkeypatch, tmp_path, REST, LEADER, PAIRED + ["--max-frames", "300"], fail=lambda n: False)
        moved = np.max(np.abs(STATE["pose"] - REST))
    finally:
        STATE["deadband"] = STATE["clock"] = None
    assert moved > np.radians(5.0)


def test_servo_p_default_applied_on_connect(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, REST, None, ["--max-frames", "3"], fail=lambda n: False)
    from cta.apps.mirror import DEFAULT_SERVO_P
    assert STATE["servo_p"] == DEFAULT_SERVO_P


# ---- --control arm / hand ----------------------------------------------------------------------------

def _save_cal(tmp_path):
    from helpers import make_hand

    from cta.features import compute_features
    from cta.kinematics import Kinematics
    from cta.mapping import calibrate, save_calibration
    img, world = make_hand()
    path = tmp_path / "cal.json"
    save_calibration(calibrate(compute_features(img, world, 16 / 9), Kinematics()), path)
    return path


def test_control_hand_never_builds_pose_tracker(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, REST, None, ["--control", "hand", "--max-frames", "5"], fail=lambda n: False)
    assert FakePose.constructed == 0
    for q in FakeFollower.sent:
        assert np.all(np.abs(q - REST) <= STEP)


def _arm_run(monkeypatch, tmp_path, extra, frames=120, control="arm"):
    """A raised left arm (az 30, elev 60, flex 40) with a hand on its wrist, in --control arm (or `control`)."""
    from helpers import bend, body_dir, make_body, make_hand

    from cta.hands import HandObservation
    from cta.pose import PoseObservation
    a = body_dir(30, 60)
    FakePose.obs = PoseObservation(*make_body(a, bend(a, 40)), 0.9, np.ones(33), np.ones(33))
    FakeTracker.obs = HandObservation(*make_hand(wrist_uv=tuple(FakePose.obs.img[15, :2])), "left", 0.9)
    STATE["clock"] = 0.0
    STATE["deadband"] = 0.0  # the fake follower reaches each command, so the lead bound lets it move
    try:
        return _run(monkeypatch, tmp_path, REST, None,
                    ["--control", control, "--max-frames", str(frames), "--hand", "left", *extra],
                    fail=lambda n: False, calib=_save_cal(tmp_path))
    finally:
        FakePose.obs = FakeTracker.obs = None
        STATE["clock"] = STATE["deadband"] = None


def test_control_arm_drives_shoulder_and_elbow(monkeypatch, tmp_path):
    from cta.recorder import load_session
    rc = _arm_run(monkeypatch, tmp_path, ["--record", "arm", "--autostart-record", "--pose-min-vis", "0.5"])
    assert rc == 0 and FakePose.constructed == 1
    last = FakeFollower.sent[-1]
    assert last[0] < REST[0] - 0.3          # pan swung toward -30 deg (user's arm to their left)
    assert np.all(np.abs(np.diff(np.array(FakeFollower.sent), axis=0)) <= STEP)  # app speed limit intact
    assert FakeDash.arm_points and np.asarray(FakeDash.arm_points[-1]).shape == (3, 2)
    d = load_session(tmp_path / "sessions" / "arm_01")
    assert np.isfinite(d["pose_world"]).all() and d["pose_img"].shape == (120, 33, 3)
    assert (d["status"] == 2).all()  # TRACKING: pose + hand
    assert d["pose_world"].dtype == np.float32
    assert np.allclose(d["pose_vis"], 1.0) and np.allclose(d["pose_presence"], 1.0)
    meta = json.loads((tmp_path / "sessions" / "arm_01" / "meta.json").read_text())
    assert meta["control"] == "arm"



def test_safe_clearance_hold_is_not_a_power_warning(monkeypatch, tmp_path, capsys):
    """A clearance hold keeps the arm still on purpose: say so, and don't blame the 12 V supply."""
    monkeypatch.setattr(mirror, "guard_clearance", lambda q_new, prev, contact, margin: (prev.copy(), "held"))
    rc = _arm_run(monkeypatch, tmp_path, [], frames=150)
    out = capsys.readouterr().out
    assert rc == 0
    assert np.allclose(FakeFollower.sent[-1], REST)  # every step held
    assert "holding: next step would go below the table margin" in out
    assert out.count("holding: next step would go below the table margin") <= 3  # rate-limited
    assert "not moving" not in out


def test_table_margin_flag_reaches_pipeline_and_send_path(monkeypatch, tmp_path):
    seen = {"pipe": [], "send": []}
    real_pipe, real_next = mirror.ArmPosePipeline, mirror.next_command

    def pipe_spy(*a, **k):
        seen["pipe"].append(k.get("table_margin"))
        return real_pipe(*a, **k)

    def next_spy(*a, **k):
        seen["send"].append(k.get("margin"))
        return real_next(*a, **k)

    monkeypatch.setattr(mirror, "ArmPosePipeline", pipe_spy)
    monkeypatch.setattr(mirror, "next_command", next_spy)
    assert _arm_run(monkeypatch, tmp_path, ["--table-margin", "0.03"], frames=10) == 0
    assert seen["pipe"] == [0.03] and set(seen["send"]) == {0.03}


@pytest.mark.parametrize("bad", ["-0.01", "0.2"])
def test_table_margin_out_of_range_is_rejected(bad):
    with pytest.raises(SystemExit):
        mirror.main(["--robot", "none", "--table-margin", bad])



# ---- --control reach (the default) ---------------------------------------------------------------------

def test_control_defaults_to_reach(monkeypatch, tmp_path):
    seen = []
    real = mirror.ReachPipeline

    def spy(*a, **k):
        seen.append((k.get("table_margin"), k.get("home_relative")))
        return real(*a, **k)

    monkeypatch.setattr(mirror, "ReachPipeline", spy)
    _run(monkeypatch, tmp_path, REST, None, ["--max-frames", "3", "--table-margin", "0.02"], fail=lambda n: False)
    assert seen == [(0.02, True)] and FakePose.constructed == 1


def test_control_reach_moves_the_gripper_where_the_arm_reaches(monkeypatch, tmp_path):
    from cta.kinematics import Kinematics
    from cta.recorder import load_session
    rc = _arm_run(monkeypatch, tmp_path, ["--record", "reach", "--autostart-record", "--reach-mapping", "absolute"], frames=180, control="reach")
    assert rc == 0 and FakePose.constructed == 1
    sent = np.array(FakeFollower.sent)
    assert np.all(np.abs(np.diff(sent, axis=0)) <= STEP)  # app speed limit intact
    p, _ = Kinematics().fk(sent[-1])
    assert p[1] > 0.05  # the user reaches toward their left (az 30): the gripper goes to the robot's left
    assert FakeDash.arm_points and np.asarray(FakeDash.arm_points[-1]).shape == (3, 2)  # shoulder -> elbow -> wrist
    assert FakeDash.reach and np.asarray(FakeDash.reach[-1]).shape == (3,)
    d = load_session(tmp_path / "sessions" / "reach_01")
    assert (d["status"] == 2).all()
    meta = json.loads((tmp_path / "sessions" / "reach_01" / "meta.json").read_text())
    assert meta["control"] == "reach"
