import numpy as np
import pytest
from helpers import make_hand

from cta.features import HandFeatures
from cta.hands import HandObservation
from cta.recorder import SessionRecorder, load_session

SIZE = (64, 36)


def _obs():
    lm_img, lm_world = make_hand()
    return HandObservation(lm_img, lm_world, "right", 0.9)


def _feats():
    return HandFeatures(0.5, 0.5, 0.1, 0.2, 0.3, 0.4)


def _rgb():
    return np.zeros((72, 128, 3), np.uint8)


def _add(r, obs="hand", leader=None, follower=None):
    r.add(0.0, _obs() if obs == "hand" else None, _feats() if obs == "hand" else None,
          np.zeros(6), leader, follower, _rgb())


def test_roundtrip_with_missing_leader(tmp_path):
    with SessionRecorder(tmp_path, "s", {"mode": "free"}, video_size=SIZE) as r:
        for i in range(10):
            r.add(i / 60, _obs(), _feats(), np.full(6, 0.1), None, np.full(6, 0.2), _rgb())
    d = load_session(tmp_path / "s")
    assert d["t"].shape == (10,)
    assert d["hand_present"].dtype == bool and d["hand_present"].all()
    assert d["lm_img"].shape == (10, 21, 3) and d["lm_world"].shape == (10, 21, 3)
    assert d["feats"].shape == (10, 6)
    assert np.allclose(d["feats"][0], [0.5, 0.5, 0.1, 0.2, 0.3, 0.4])
    assert np.isnan(d["q_leader"]).all() and d["q_leader"].shape == (10, 6)
    assert np.allclose(d["q_baseline"], 0.1) and np.allclose(d["q_follower"], 0.2)
    assert (tmp_path / "s" / "human.mp4").exists() and (tmp_path / "s" / "meta.json").exists()


def test_no_hand_rows_are_nan(tmp_path):
    with SessionRecorder(tmp_path, "s", {}, video_size=SIZE) as r:
        _add(r, "hand")
        _add(r, None)
    d = load_session(tmp_path / "s")
    assert d["hand_present"].tolist() == [True, False]
    assert np.isnan(d["lm_img"][1]).all() and np.isnan(d["lm_world"][1]).all()
    assert np.isnan(d["feats"][1]).all()
    assert np.isfinite(d["lm_img"][0]).all()


def test_recorder_saves_on_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with SessionRecorder(tmp_path, "s", {}, autosave_s=1e9, video_size=SIZE) as r:
            for _ in range(5):
                _add(r)
            raise RuntimeError("serial link died")
    assert load_session(tmp_path / "s")["t"].shape == (5,)


def test_autosave_writes_partial(tmp_path):
    r = SessionRecorder(tmp_path, "s", {}, autosave_s=0, video_size=SIZE)
    try:
        _add(r)
        assert (tmp_path / "s" / "frames.npz").exists()
        assert load_session(tmp_path / "s")["t"].shape == (1,)
    finally:
        r.close()


def test_status_and_smoother_roundtrip(tmp_path):
    from cta.recorder import STATUS_CODES
    with SessionRecorder(tmp_path, "s", {}, video_size=SIZE) as r:
        for st, sm in [("UNCALIBRATED", 1), ("NO HAND", 0), ("TRACKING", 1)]:
            r.add(0.0, None, None, np.zeros(6), None, None, _rgb(), status=STATUS_CODES[st], smoother_on=sm)
    d = load_session(tmp_path / "s")
    assert d["status"].dtype == np.uint8 and d["status"].tolist() == [0, 1, 2]
    assert d["smoother_on"].dtype == np.int8 and d["smoother_on"].tolist() == [1, 0, 1]


def test_save_retries_replace(tmp_path, monkeypatch):
    import os
    real, calls = os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) <= 2:
            raise PermissionError("locked")
        return real(a, b)

    monkeypatch.setattr("cta.recorder.time.sleep", lambda s: None)
    monkeypatch.setattr("cta.recorder.os.replace", flaky)
    r = SessionRecorder(tmp_path, "s", {}, autosave_s=1e9, video_size=SIZE)
    _add(r)
    r.close()
    assert len(calls) == 3
    assert load_session(tmp_path / "s")["t"].shape == (1,)


def test_locked_file_never_raises_and_keeps_tmp(tmp_path, monkeypatch, capsys):
    def locked(a, b):
        raise PermissionError("locked")

    monkeypatch.setattr("cta.recorder.time.sleep", lambda s: None)
    monkeypatch.setattr("cta.recorder.os.replace", locked)
    r = SessionRecorder(tmp_path, "s", {}, autosave_s=0, video_size=SIZE)
    _add(r)  # autosave fails: must not raise
    _add(r)
    r.close()  # final save fails: must not raise
    assert (tmp_path / "s" / "frames.tmp.npz").exists()
    assert "frames.tmp.npz" in capsys.readouterr().out
    assert load_session(tmp_path / "s")["t"].shape == (2,)  # falls back to tmp


def test_load_session_missing_and_list_sessions(tmp_path):
    from cta.recorder import list_sessions
    (tmp_path / "orphan_01").mkdir()
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "orphan_01")
    with SessionRecorder(tmp_path, "good_01", {}, video_size=SIZE) as r:
        _add(r)
    assert [p.name for p in list_sessions(tmp_path, "*_01")] == ["good_01"]


def test_pose_rows_roundtrip_and_nan_when_absent(tmp_path):
    img, world = np.full((33, 3), 0.25), np.full((33, 3), -0.5)
    with SessionRecorder(tmp_path, "s", {"control": "arm"}, video_size=SIZE) as r:
        r.add(0.0, _obs(), _feats(), np.zeros(6), None, None, _rgb(), pose_img=img, pose_world=world)
        _add(r)  # old call signature: no pose
    d = load_session(tmp_path / "s")
    assert d["pose_img"].shape == (2, 33, 3) and d["pose_world"].shape == (2, 33, 3)
    assert np.allclose(d["pose_img"][0], 0.25) and np.allclose(d["pose_world"][0], -0.5)
    assert np.isnan(d["pose_img"][1]).all() and np.isnan(d["pose_world"][1]).all()
    assert d["pose_img"].dtype == np.float32 and d["pose_world"].dtype == np.float32
    import json
    assert json.loads((tmp_path / "s" / "meta.json").read_text())["control"] == "arm"



def test_pose_vis_and_presence_rows(tmp_path):
    vis, pres = np.linspace(0, 1, 33), np.full(33, 0.75)
    with SessionRecorder(tmp_path, "s", {}, video_size=SIZE) as r:
        r.add(0.0, None, None, np.zeros(6), None, None, _rgb(), pose_img=np.zeros((33, 3)),
              pose_world=np.zeros((33, 3)), pose_vis=vis, pose_presence=pres)
        _add(r)
    d = load_session(tmp_path / "s")
    assert d["pose_vis"].shape == (2, 33) and d["pose_presence"].shape == (2, 33)
    assert d["pose_vis"].dtype == np.float32 and d["pose_presence"].dtype == np.float32
    assert np.allclose(d["pose_vis"][0], vis) and np.allclose(d["pose_presence"][0], 0.75)
    assert np.isnan(d["pose_vis"][1]).all() and np.isnan(d["pose_presence"][1]).all()


def _count_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_robot_camera_video_has_one_frame_per_row(tmp_path):
    with SessionRecorder(tmp_path, "s", {}, video_size=SIZE, robot_video_size=SIZE) as r:
        for i in range(6):
            robot = _rgb() if i % 3 else None  # robot camera missing on some rows -> black frame, ok=False
            r.add(float(i), _obs(), _feats(), np.zeros(6), None, None, _rgb(),
                  robot_rgb=robot, robot_t=None if robot is None else float(i) - 0.01)
    s = load_session(tmp_path / "s")
    assert _count_frames(tmp_path / "s" / "robot.mp4") == 6 == len(s["t"])
    assert s["robot_cam_ok"].tolist() == [False, True, True, False, True, True]
    assert np.isnan(s["robot_frame_t"][0]) and s["robot_frame_t"][1] == pytest.approx(0.99)


def test_no_robot_video_by_default(tmp_path):
    with SessionRecorder(tmp_path, "s", {}, video_size=SIZE) as r:
        _add(r)
    assert not (tmp_path / "s" / "robot.mp4").exists()
    assert load_session(tmp_path / "s")["robot_cam_ok"].tolist() == [False]
