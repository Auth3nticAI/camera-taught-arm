import dataclasses

import numpy as np
import pytest

from cta.config import READY_Q, SIM_LIMITS
from cta.features import HandFeatures
from cta.kinematics import Kinematics
from cta.mapping import MapGains, calibrate, load_calibration, map_to_target, save_calibration

KIN = Kinematics()
HOME = HandFeatures(u=0.5, v=0.5, scale=0.2, pitch=0.1, roll=0.2, pinch=0.6)


_GAINS = MapGains.from_kinematics(KIN)


def _setup():
    return calibrate(HOME, KIN), _GAINS


def test_home_features_map_to_p0():
    cal, g = _setup()
    pos, pitch, roll, grip = map_to_target(HOME, cal, g)
    assert np.allclose(pos, cal.p0, atol=1e-9)
    assert abs(pitch - cal.pitch_home) < 1e-9
    assert abs(roll) < 1e-9
    frac = (0.6 - cal.pinch_closed) / (cal.pinch_open - cal.pinch_closed)
    lo, hi = SIM_LIMITS[5]
    assert np.isclose(grip, lo + frac * (hi - lo))


def test_user_left_moves_robot_plus_y():
    cal, g = _setup()
    base, *_ = map_to_target(HOME, cal, g)
    pos, *_ = map_to_target(dataclasses.replace(HOME, u=HOME.u + 0.1), cal, g)
    assert np.isclose(pos[1] - base[1], 0.03)
    assert np.isclose(pos[0], base[0]) and np.isclose(pos[2], base[2])


def test_hand_closer_moves_robot_forward():
    cal, g = _setup()
    base, *_ = map_to_target(HOME, cal, g)
    pos, *_ = map_to_target(dataclasses.replace(HOME, scale=HOME.scale * 1.25), cal, g)
    assert pos[0] > base[0]


def test_forward_gain_exact_value():
    cal, g = _setup()
    base, *_ = map_to_target(HOME, cal, g)
    pos, *_ = map_to_target(dataclasses.replace(HOME, scale=HOME.scale * 1.1), cal, g)
    assert np.isclose(pos[0] - base[0], 0.25 * (1 - 1 / 1.1))


def test_scale_saturates_x_at_box():
    cal, g = _setup()
    hi, *_ = map_to_target(dataclasses.replace(HOME, scale=HOME.scale * 1000), cal, g)
    lo, *_ = map_to_target(dataclasses.replace(HOME, scale=HOME.scale * 0.01), cal, g)
    assert hi[0] == g.box_max[0]
    assert lo[0] == g.box_min[0]


def test_box_faces_reachable_by_ik():
    cal, g = _setup()
    m = (g.box_min + g.box_max) / 2
    for p in ([g.box_max[0], 0, 0.1], [g.box_min[0], 0, 0.1],
              [m[0], g.box_min[1], 0.1], [m[0], g.box_max[1], 0.1]):
        p = np.array(p)
        q = KIN.ik(p, cal.pitch_home, READY_Q, 0.0, READY_Q[5])
        assert np.linalg.norm(KIN.fk(q)[0] - p) <= 0.01


def test_calibrate_rejects_zero_scale():
    with pytest.raises(ValueError):
        calibrate(dataclasses.replace(HOME, scale=0.0), KIN)


def test_pinch_closed_gives_min_gripper():
    cal, g = _setup()
    *_, grip = map_to_target(dataclasses.replace(HOME, pinch=0.1), cal, g)
    assert grip == SIM_LIMITS[5, 0]


def test_target_clipped_to_box():
    cal, g = _setup()
    for du in (5.0, -5.0):
        pos, *_ = map_to_target(dataclasses.replace(HOME, u=HOME.u + du, v=HOME.v + du), cal, g)
        assert np.all(pos >= g.box_min) and np.all(pos <= g.box_max)
    assert np.all(g.box_max > g.box_min)


def test_calibration_json_roundtrip(tmp_path):
    cal, _ = _setup()
    path = tmp_path / "cal.json"
    save_calibration(cal, path)
    back = load_calibration(path)
    assert isinstance(back.p0, np.ndarray)
    assert np.allclose(back.p0, cal.p0)
    for f in ("u0", "v0", "s0", "pitch0", "roll0", "pinch_closed", "pinch_open", "pitch_home"):
        assert getattr(back, f) == getattr(cal, f)


def test_roll_delta_wraps_across_pi():
    # home roll near +180 deg; a small wobble past the wrap reads as -179 deg - must map to ~+2 deg, not -358
    home = dataclasses.replace(HOME, roll=np.radians(179.0))
    cal = calibrate(home, KIN)
    _, _, roll, _ = map_to_target(dataclasses.replace(home, roll=np.radians(-179.0)), cal, _GAINS)
    assert np.isclose(roll, np.radians(2.0), atol=1e-9)
    _, _, roll, _ = map_to_target(dataclasses.replace(home, roll=np.radians(177.0)), cal, _GAINS)
    assert np.isclose(roll, np.radians(-2.0), atol=1e-9)


def test_calibrate_sets_pinch_open_from_home_hand():
    cal = calibrate(dataclasses.replace(HOME, pinch=0.8), KIN)
    assert cal.pinch_open == pytest.approx(0.8)


def test_calibrate_rejects_closed_hand():
    with pytest.raises(ValueError, match="open your hand"):
        calibrate(dataclasses.replace(HOME, pinch=0.3), KIN)


def test_set_pinch_closed_from_live_pinch():
    from cta.mapping import set_pinch_closed

    cal = calibrate(dataclasses.replace(HOME, pinch=0.8), KIN)
    cal = set_pinch_closed(cal, dataclasses.replace(HOME, pinch=0.12))
    assert cal.pinch_closed == pytest.approx(0.12)
    with pytest.raises(ValueError):
        set_pinch_closed(cal, dataclasses.replace(HOME, pinch=0.75))  # too close to open
