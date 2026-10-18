import numpy as np
import pytest

from cta.align import ContactModel, solve_offsets
from cta.config import SIM_LIMITS

TRUE_OFFSET = np.array([0.0, 3.0, -2.0, 4.0, 0.0])  # deg; only lift/elbow/wrist_flex are observable


def _touch_poses(model: ContactModel, n: int = 10, seed: int = 0) -> np.ndarray:
    """Sim poses whose lowest jaw point sits exactly on the table (lift found by bisection)."""
    rng = np.random.default_rng(seed)
    poses = []
    while len(poses) < n:
        q = np.zeros(6)
        q[0] = rng.uniform(-0.8, 0.8)
        q[2] = rng.uniform(-0.6, 1.4)
        q[3] = rng.uniform(0.0, 1.6)
        q[5] = SIM_LIMITS[5, 0]
        lo, hi = -1.6, 1.6

        def h(lift):
            q[1] = lift
            return model.lowest_z(q) - model.table_z

        if h(lo) * h(hi) > 0:
            continue
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if h(lo) * h(mid) <= 0:
                hi = mid
            else:
                lo = mid
        q[1] = 0.5 * (lo + hi)
        poses.append(q.copy())
    return np.array(poses)


@pytest.fixture(scope="module")
def model():
    return ContactModel()


def test_recovers_offsets_and_table_height(model):
    q = _touch_poses(model)
    real_deg = np.degrees(q[:, :5]) + TRUE_OFFSET
    res = solve_offsets(real_deg, q[:, 5], model)
    assert np.allclose(res.offset_deg[1:4], TRUE_OFFSET[1:4], atol=0.3)
    assert res.offset_deg[0] == 0.0 and res.offset_deg[4] == 0.0
    assert res.z0 == model.table_z
    assert res.rms_after < 0.0005 < res.rms_before
    assert res.trusted and np.all(res.std_deg < 1.0)


def test_zero_offsets_when_already_aligned(model):
    q = _touch_poses(model, seed=1)
    res = solve_offsets(np.degrees(q[:, :5]), q[:, 5], model)
    assert np.all(np.abs(res.offset_deg) < 0.3)


def test_requires_enough_poses(model):
    with pytest.raises(ValueError):
        solve_offsets(np.zeros((3, 5)), np.zeros(3), model)


def test_offsets_bounded(model):
    q = _touch_poses(model, n=8, seed=2)
    res = solve_offsets(np.degrees(q[:, :5]) + np.array([0, 40.0, 0, 0, 0]), q[:, 5], model)
    assert np.all(np.abs(res.offset_deg) <= 15.0 + 1e-9)
    assert not res.trusted


def test_table_is_base_bottom(model):
    assert -0.01 < model.table_z < 0.0  # base mesh bottom sits ~2.4 mm below the base frame
