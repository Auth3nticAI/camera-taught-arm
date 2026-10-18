"""Table clearance on the command actually SENT (after the per-joint speed limit and the lead clip).

The pipeline's target can be clear while the per-joint ramp toward it is not: each joint moves at the same
speed, so the arm passes through poses the target never had (review 2026-10-17)."""
import numpy as np
import pytest

from cta.align import ContactModel
from cta.apps.mirror import next_command
from cta.pipeline import CLEARANCE_BODIES, TABLE_CLEARANCE_M, guard_clearance

D = np.radians
DT = 1 / 60
MAX_SPEED = D(90.0)


@pytest.fixture(scope="module")
def contact():
    return ContactModel(CLEARANCE_BODIES)


def floor(contact):
    return contact.table_z + TABLE_CLEARANCE_M


def replay(contact, start, target, frames=400, mode="mirror", use_contact=True):
    """Follower that reaches every command (the worst case for the lead clip)."""
    prev = start.copy()
    sent = []
    for _ in range(frames):
        q_new = next_command(mode, "TRACKING", target, None, prev, DT, MAX_SPEED, prev.copy(),
                             contact if use_contact else None)
        sent.append(q_new)
        prev = q_new
    return np.array(sent)


START = D(np.array([0.0, 24.0, -64.0, 0.0, 0.0, 30.0]))
TARGET = D(np.array([0.0, 84.0, 69.0, 48.0, 0.0, 30.0]))  # clear itself (3.3 cm), path is not


def test_unguarded_ramp_dips_into_table(contact):
    assert contact.lowest_z(START) >= floor(contact) and contact.lowest_z(TARGET) >= floor(contact)
    sent = replay(contact, START, TARGET, use_contact=False)
    assert min(contact.lowest_z(q) for q in sent) < floor(contact) - 0.05  # the hazard is real


def test_guarded_ramp_never_sent_below_table(contact):
    sent = replay(contact, START, TARGET)
    lows = np.array([contact.lowest_z(q) for q in sent])
    assert lows.min() >= floor(contact) - 1e-9
    assert np.all(np.abs(np.diff(np.vstack([START, sent]), axis=0)) <= MAX_SPEED * DT + 1e-9)  # still rate limited


def test_guard_passes_safe_moves_unchanged(contact):
    prev = D(np.array([0.0, 30.0, -30.0, 0.0, 0.0, 30.0]))
    q = prev + D(1.0)
    assert np.array_equal(guard_clearance(q, prev, contact), q)


def test_guard_lets_a_low_arm_rise(contact):
    low = D(np.array([0.0, 84.0, 0.0, 10.0, 0.0, 30.0]))  # already in the table (e.g. measured start pose)
    assert contact.lowest_z(low) < floor(contact)
    up = low.copy()
    up[1] -= D(1.5)
    out = guard_clearance(up, low, contact)
    assert np.array_equal(out, up)  # moving up is allowed even below the floor
    down = low.copy()
    down[2] += D(1.5)
    assert contact.lowest_z(guard_clearance(down, low, contact)) >= contact.lowest_z(low) - 1e-9


def test_paired_mode_is_not_guarded(contact):
    """Paired teleop (the leader drives) must still be able to touch the table."""
    prev = D(np.array([0.0, 84.0, 0.0, 10.0, 0.0, 30.0]))
    leader = prev.copy()
    leader[2] += D(5.0)  # pushing further into the table
    q = next_command("paired", "TRACKING", prev, leader, prev, DT, MAX_SPEED, prev.copy(), contact)
    assert q[2] > prev[2] and contact.lowest_z(q) < contact.lowest_z(prev)
