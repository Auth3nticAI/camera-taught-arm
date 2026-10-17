from cta.camera import CameraMode, camera_check


def test_camera_check_ok():
    assert camera_check(CameraMode(1920, 1080, 59.0)) == []


def test_camera_check_flags_wrong_mode():
    w = camera_check(CameraMode(1280, 720, 30.0))
    assert len(w) == 2
