import numpy as np
import pytest

from cta.paths import models_dir

MODEL = models_dir() / "pose_landmarker_full.task"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/download_models.py")


@needs_model
def test_pose_tracker_blank_frame_returns_none():
    from cta.pose import PoseTracker
    tr = PoseTracker(MODEL)
    try:
        rgb = np.zeros((1080, 1920, 3), np.uint8)
        assert tr.detect(rgb, 0.0) is None
        assert tr.detect(rgb, 0.0) is None  # repeated timestamp must not raise (VIDEO mode)
    finally:
        tr.close()


def test_pose_observation_shapes():
    from cta.pose import PoseObservation
    o = PoseObservation(np.zeros((33, 3)), np.zeros((33, 3)), 0.5)
    assert o.img.shape == (33, 3) and o.world.shape == (33, 3) and o.score == 0.5
