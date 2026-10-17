import numpy as np

from cta.config import READY_Q, SIM_LIMITS
from cta.kinematics import Kinematics


def test_fk_ready_pose_in_front_of_base():
    k = Kinematics()
    p, _ = k.fk(READY_Q)
    assert p[2] > 0.02 and np.linalg.norm(p[:2]) > 0.10


def test_ik_roundtrip_under_5mm():
    k = Kinematics()
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = READY_Q + rng.uniform(-0.4, 0.4, 6)
        q = np.clip(q, *SIM_LIMITS.T)
        p, pitch = k.fk(q)
        q2 = k.ik(p, pitch, READY_Q, q[4], q[5])
        p2, _ = k.fk(q2)
        assert np.linalg.norm(p2 - p) < 0.005


def test_ik_unreachable_target_stays_finite_and_in_limits():
    k = Kinematics()
    q = k.ik(np.array([2.0, 0.0, 2.0]), 0.0, READY_Q, 0.0, 0.5)
    assert np.all(np.isfinite(q))
    assert np.all(q >= SIM_LIMITS[:, 0]) and np.all(q <= SIM_LIMITS[:, 1])
