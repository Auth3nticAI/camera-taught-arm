import mujoco
import numpy as np
import pytest
from helpers import bend, body_dir, make_body

from cta.arm_angles import (ELBOW_OFFSET_RAD, LIFT_OFFSET_RAD, arm_joint_angles, body_frame,
                            choose_arm_side, side_indices)
from cta.kinematics import Kinematics

D = np.radians


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


def robot_arm_dirs(kin: Kinematics, q) -> tuple[np.ndarray, np.ndarray]:
    """Robot upper-arm (lift->elbow joint) and forearm (elbow->wrist joint) unit directions, base frame
    (x forward, y = robot's left, z up), straight from the MuJoCo model's joint anchors."""
    kin.fk(np.asarray(q, float))
    m, d = kin._m, kin._d
    rb, pb = d.site_xmat[kin._base].reshape(3, 3), d.site_xpos[kin._base]
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("shoulder_lift", "elbow_flex", "wrist_flex")]
    p = [rb.T @ (d.xanchor[j] - pb) for j in jid]
    a, b = p[1] - p[0], p[2] - p[1]
    return a / np.linalg.norm(a), b / np.linalg.norm(b)


def angle_deg(u, v) -> float:
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


# ---- verified robot conventions --------------------------------------------------------------------

def test_offsets_match_the_robot_model(kin):
    a, b = robot_arm_dirs(kin, [0, 0, 0, 0, 0, 0])
    assert angle_deg(a, [0, 0, 1]) == pytest.approx(np.degrees(LIFT_OFFSET_RAD), abs=0.2)
    assert angle_deg(a, b) == pytest.approx(np.degrees(ELBOW_OFFSET_RAD), abs=0.2)
    a90, _ = robot_arm_dirs(kin, [0, D(90), 0, 0, 0, 0])
    assert angle_deg(a90, [0, 0, 1]) == pytest.approx(90 + np.degrees(LIFT_OFFSET_RAD), abs=0.2)  # +lift = forward


def test_positive_pan_moves_gripper_to_robot_right(kin):
    p, _ = kin.fk(np.array([D(30), 0, 0, 0, 0, 0]))
    assert p[1] < -0.1


# ---- body frame ------------------------------------------------------------------------------------

@pytest.mark.parametrize("mirrored", [False, True])
def test_body_frame_facing_camera(mirrored):
    _, world = make_body(body_dir(0, 90), mirrored=mirrored)
    up, left, fwd = body_frame(world, mirrored)
    assert np.allclose(up, [0, -1, 0], atol=1e-6)
    assert np.allclose(fwd, [0, 0, -1], atol=1e-6)  # chest toward the camera
    # the user's true left in UNMIRRORED camera coords (+x); mirrored input is un-mirrored first
    assert np.allclose(left, [1, 0, 0], atol=1e-6)


# ---- joint angles ----------------------------------------------------------------------------------

def test_arm_straight_up_is_robot_upper_arm_vertical():
    _, world = make_body(body_dir(0, 0))
    pan, lift, elbow, valid = arm_joint_angles(world, "left", False, prev_pan=0.3)
    assert lift == pytest.approx(-LIFT_OFFSET_RAD, abs=1e-6)  # spec "lift ~ 0" + the model's 14 deg offset
    assert not valid and pan == 0.3


def test_arm_forward_horizontal():
    _, world = make_body(body_dir(0, 90))
    pan, lift, elbow, valid = arm_joint_angles(world, "left", False, prev_pan=0.0)
    assert valid
    assert lift == pytest.approx(D(90) - LIFT_OFFSET_RAD, abs=1e-6)
    assert pan == pytest.approx(0.0, abs=1e-6)


def test_arm_out_to_users_left_is_negative_pan():
    _, world = make_body(body_dir(90, 90))
    pan, lift, _, valid = arm_joint_angles(world, "left", False, prev_pan=0.0)
    assert valid and pan == pytest.approx(D(-90), abs=1e-6)


def test_arm_hanging_down_holds_previous_pan():
    _, world = make_body(body_dir(0, 180))
    pan, _, _, valid = arm_joint_angles(world, "left", False, prev_pan=-0.42)
    assert not valid and pan == -0.42


def test_elbow_straight_and_right_angle():
    up = body_dir(0, 90)
    _, w0 = make_body(up)
    _, w90 = make_body(up, bend(up, 90))
    assert arm_joint_angles(w0, "left", False, 0.0)[2] == pytest.approx(-ELBOW_OFFSET_RAD, abs=1e-6)
    assert arm_joint_angles(w90, "left", False, 0.0)[2] == pytest.approx(D(90) - ELBOW_OFFSET_RAD, abs=1e-6)


POSES = [  # (side, az, elev, flex) - all inside the robot's joint limits
    ("left", 0, 90, 90), ("left", 40, 60, 30), ("right", -35, 110, 80),
    ("right", 0, 45, 120), ("left", 70, 100, 100), ("right", -80, 70, 75),
]


@pytest.mark.parametrize("side,az,elev,flex", POSES)
def test_robot_arm_shape_matches_user_arm(kin, side, az, elev, flex):
    """Strongest check: the robot's upper arm and forearm (MuJoCo joint anchors) point where the user's do,
    with body (fwd, left, up) == robot base (x, y, z) - the robot faces the same way as the user."""
    a = body_dir(az, elev)
    b = bend(a, flex)
    _, world = make_body(a, b, side=side)
    pan, lift, elbow, valid = arm_joint_angles(world, side, False, 0.0)
    assert valid
    ra, rb = robot_arm_dirs(kin, [pan, lift, elbow, 0, 0, 0])
    assert angle_deg(ra, a) < 1.0
    assert angle_deg(rb, b) < 1.0


@pytest.mark.parametrize("side,az,elev,flex", POSES)
def test_mirrored_image_gives_same_robot_angles(side, az, elev, flex):
    a = body_dir(az, elev)
    _, w = make_body(a, bend(a, flex), side=side)
    _, wm = make_body(a, bend(a, flex), side=side, mirrored=True)
    assert np.allclose(arm_joint_angles(w, side, False, 0.0), arm_joint_angles(wm, side, True, 0.0), atol=1e-9)


# ---- which arm -------------------------------------------------------------------------------------

@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
def test_choose_arm_side_nearest_hand_wrist(side, mirrored):
    a = body_dir(0, 90)
    img, _ = make_body(a, side=side, mirrored=mirrored)  # `side` arm raised, the other hangs
    _, _, wr = side_indices(side, mirrored)
    hand_uv = img[wr, :2] + 0.01  # the hand tracker's wrist, in the camera's own image coords
    other = "right" if side == "left" else "left"
    assert choose_arm_side(img, hand_uv, last_side=other, want=other, mirrored=mirrored) == side


def test_choose_arm_side_ignores_labels():
    img, _ = make_body(body_dir(0, 90), side="left")
    swapped = img.copy()
    swapped[[11, 12, 13, 14, 15, 16]] = img[[12, 11, 14, 13, 16, 15]]  # Pose swapped its labels
    hand_uv = img[15, :2]
    assert choose_arm_side(swapped, hand_uv, None, "left", False) == "right"  # follows the position


def test_choose_arm_side_without_hand():
    img, _ = make_body(body_dir(0, 90))
    assert choose_arm_side(img, None, "right", "left", False) == "right"  # keep the last side
    assert choose_arm_side(img, None, None, "left", False) == "left"      # no history: --hand
    assert choose_arm_side(img, None, None, "right", True) == "right"


def test_side_indices():
    assert side_indices("left", False) == (11, 13, 15)
    assert side_indices("left", True) == (12, 14, 16)  # mirrored image: labels are swapped
    assert side_indices("right", False) == (12, 14, 16)


# ---- review fixes: hidden hips, pan behind the torso, hysteresis --------------------------------------

@pytest.mark.parametrize("mirrored", [False, True])
def test_hidden_hips_fall_back_to_camera_up(mirrored):
    a = body_dir(30, 70)
    _, world = make_body(a, bend(a, 50), mirrored=mirrored)
    ref = arm_joint_angles(world, "left", mirrored, 0.0)
    garbage = world.copy()
    garbage[[23, 24]] = [[0.3, -0.2, 0.4], [0.1, 0.5, -0.3]]  # occluded hips: Pose guesses nonsense
    up, left, fwd = body_frame(garbage, mirrored, hips_ok=False)
    assert np.allclose(up, [0, -1, 0], atol=1e-6) and np.allclose(left, [1, 0, 0], atol=1e-6)
    assert np.allclose(arm_joint_angles(garbage, "left", mirrored, 0.0, hips_ok=False), ref, atol=1e-6)


def test_pan_behind_torso_noise_does_not_flip():
    for az in (178, -178, 179, -177, 150, -150):  # upper arm pointing behind the user, jittering across 180
        _, world = make_body(body_dir(az, 90))
        pan, _, _, valid = arm_joint_angles(world, "left", False, prev_pan=0.5)
        assert not valid and pan == 0.5


def test_pan_unwraps_and_stays_in_range():
    _, world = make_body(body_dir(100, 90))  # slightly behind the shoulder line: still reachable (|pan| < 105)
    pan, _, _, valid = arm_joint_angles(world, "left", False, prev_pan=D(-90))
    assert valid and pan == pytest.approx(D(-100), abs=1e-6)
    pan, _, _, valid = arm_joint_angles(world, "left", False, prev_pan=D(100))  # 200 deg jump in one frame
    assert not valid and pan == pytest.approx(D(100))


def test_pan_valid_hysteresis():
    from cta.arm_angles import PAN_INVALID_FRAC, PAN_VALID_FRAC
    assert PAN_INVALID_FRAC < PAN_VALID_FRAC
    mid = np.degrees(np.arcsin((PAN_INVALID_FRAC + PAN_VALID_FRAC) / 2))  # horizontal fraction in the band
    _, world = make_body(body_dir(20, mid))
    assert arm_joint_angles(world, "left", False, 0.0, prev_valid=True)[3] is True
    assert arm_joint_angles(world, "left", False, 0.3, prev_valid=False)[3] is False


def test_pan_max_matches_joint_limit_and_margin():
    from cta.arm_angles import PAN_MAX_RAD
    from cta.config import SIM_LIMITS
    assert PAN_MAX_RAD == pytest.approx(min(-SIM_LIMITS[0, 0], SIM_LIMITS[0, 1]) - 0.087)
