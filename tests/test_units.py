import numpy as np

from cta.config import SIM_LIMITS
from cta.units import SimRealMap, real_to_sim, sim_to_real


def test_sim_real_roundtrip():
    rng = np.random.default_rng(1)
    m = SimRealMap(sign=np.array([1, -1, 1, -1, 1.0]), offset_deg=np.array([3, -5, 0, 7, 1.0]))
    for _ in range(20):
        q = rng.uniform(SIM_LIMITS[:, 0], SIM_LIMITS[:, 1])
        assert np.allclose(real_to_sim(sim_to_real(q, m), m), q)


def test_gripper_limits_map_to_0_and_100():
    m = SimRealMap()
    q = np.zeros(6)
    q[5] = SIM_LIMITS[5, 0]
    assert np.isclose(sim_to_real(q, m)["gripper.pos"], 0.0)
    q[5] = SIM_LIMITS[5, 1]
    assert np.isclose(sim_to_real(q, m)["gripper.pos"], 100.0)


def test_sign_flip_applies():
    q = np.array([0.1, 0.5, 0.2, 0.0, 0.0, 0.5])
    base = sim_to_real(q, SimRealMap())["shoulder_lift.pos"]
    m = SimRealMap()
    m.sign[1] = -1
    assert np.isclose(sim_to_real(q, m)["shoulder_lift.pos"], -base)


def test_missing_file_gives_defaults(tmp_path):
    m = SimRealMap.load(tmp_path / "nope.json")
    assert np.array_equal(m.sign, np.ones(5)) and np.array_equal(m.offset_deg, np.zeros(5))


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "m.json"
    SimRealMap(sign=np.array([1, -1, 1, 1, 1.0]), offset_deg=np.arange(5.0)).save(p)
    m = SimRealMap.load(p)
    assert np.array_equal(m.sign, [1, -1, 1, 1, 1]) and np.array_equal(m.offset_deg, np.arange(5.0))
