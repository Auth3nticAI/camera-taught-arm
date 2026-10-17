import numpy as np

from cta.features import compute_features
from helpers import make_hand

ASPECT = 16 / 9


def feats(**kw):
    img, w = make_hand(**kw)
    return compute_features(img, w, ASPECT)


def test_scale_doubles_when_hand_twice_as_big():
    assert abs(feats(scale=2.0).scale - 2 * feats().scale) < 1e-6


def test_pinch_small_when_tips_touch():
    assert feats(pinch_closed=True).pinch < 0.05
    assert feats().pinch > 0.3


def test_pitch_positive_when_fingers_point_up():
    f = feats(pitch=np.radians(30))
    assert abs(f.pitch - 0.52) < 0.05
    assert feats(pitch=np.radians(-30)).pitch < 0
    assert abs(feats().pitch) < 1e-6


def test_roll_changes_sign_with_hand_rotation():
    p, m = feats(roll=np.radians(30)).roll, feats(roll=np.radians(-30)).roll
    assert p > 0 > m
    assert abs(p - 0.52) < 0.05 and abs(m + 0.52) < 0.05
    assert abs(feats().roll) < 1e-6
