import numpy as np

from cta.camera import CameraMode, camera_check


def test_camera_check_ok():
    assert camera_check(CameraMode(1920, 1080, 59.0)) == []


def test_camera_check_flags_wrong_mode():
    w = camera_check(CameraMode(1280, 720, 30.0))
    assert len(w) == 2


def test_camera_check_flags_non_mjpg():
    w = camera_check(CameraMode(1920, 1080, 59.0, "YUY2"))
    assert len(w) == 1 and "MJPG" in w[0]
    assert camera_check(CameraMode(1920, 1080, 59.0, "MJPG")) == []


def test_latest_frame_camera_returns_newest_frame_without_blocking():
    import time
    from cta.camera import LatestFrameCamera

    class Fake:
        def __init__(self):
            self.i = 0

        def read(self):
            time.sleep(0.005)
            self.i += 1
            return np.full((4, 4, 3), self.i % 255, np.uint8), time.perf_counter()

        def close(self):
            self.closed = True

    fake = Fake()
    cam = LatestFrameCamera(camera=fake)
    try:
        t0 = time.perf_counter()
        while cam.latest()[0] is None and time.perf_counter() - t0 < 2:
            time.sleep(0.01)
        a, ta, ida = cam.latest()
        time.sleep(0.05)
        b, tb, idb = cam.latest()
        assert a is not None and idb > ida and tb > ta
        t1 = time.perf_counter()
        cam.latest()
        assert time.perf_counter() - t1 < 0.005  # never waits for the camera
    finally:
        cam.stop()
    assert fake.closed
