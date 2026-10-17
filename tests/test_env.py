def test_imports_and_cuda():
    import torch, mujoco, mediapipe, lerobot, rerun  # noqa
    assert torch.cuda.is_available()


def test_so101_model_loads():
    import mujoco_menagerie as mm
    m = mm.load("robotstudio_so101")
    assert m.nu == 6


def test_hand_model_present():
    from cta.paths import models_dir
    assert (models_dir() / "hand_landmarker.task").stat().st_size > 1_000_000
