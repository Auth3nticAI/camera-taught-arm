import numpy as np

from cta.hands import HandTracker, select_hand
from cta.paths import models_dir


def test_select_hand_flips_handedness_for_unmirrored():
    # MediaPipe labels assume a mirrored selfie; on an unmirrored frame its "Left" is the user's right
    assert select_hand(["Left", "Right"], [0.9, 0.9], want="right", mirrored_input=False) == 0
    assert select_hand(["Left", "Right"], [0.9, 0.9], want="right", mirrored_input=True) == 1


def test_select_hand_none_when_absent():
    assert select_hand(["Left"], [0.9], "left", False) is None


def test_tracker_returns_none_on_blank_image():
    t = HandTracker(models_dir() / "hand_landmarker.task")
    assert t.detect(np.zeros((1080, 1920, 3), np.uint8), 0.0) is None


def test_choose_hand_follows_position_when_label_flips():
    from cta.hands import choose_hand

    # tracked hand was at wrist (0.7, 0.6); this frame MediaPipe labels it the other side
    labels, scores = ["Left", "Right"], [0.9, 0.9]
    wrists = [(0.2, 0.5), (0.71, 0.61)]
    assert choose_hand(labels, scores, wrists, want="left", mirrored_input=True, last_wrist=(0.7, 0.6)) == 1


def test_choose_hand_uses_label_only_without_history():
    from cta.hands import choose_hand

    labels, scores, wrists = ["Left", "Right"], [0.9, 0.9], [(0.2, 0.5), (0.7, 0.6)]
    assert choose_hand(labels, scores, wrists, "left", True, last_wrist=None) == 0
    assert choose_hand(labels, scores, wrists, "left", True, last_wrist=(0.5, 0.1)) is None
