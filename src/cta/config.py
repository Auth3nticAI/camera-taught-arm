"""Shared constants: joint names, sim joint limits, ready pose, approach axis."""
import mujoco
import mujoco_menagerie as mm
import numpy as np

JOINTS: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)


def _load_limits() -> np.ndarray:
    m = mm.load("robotstudio_so101")
    lim = np.zeros((len(JOINTS), 2))
    for i, name in enumerate(JOINTS):
        lim[i] = m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
    return lim


# (6, 2) rad, read from the MuJoCo model.
SIM_LIMITS: np.ndarray = _load_limits()

# Chosen empirically by printing FK (no viewer available): at q=0 the arm is stretched
# forward (+x) at z~0.25 with the approach axis level. This pose lowers the gripper in
# front of the base: gripper site ~(0.315, 0, 0.155) m in the base frame, approach
# pitch ~-17 deg, wrist_flex at 0 (mid-range), gripper partly open.
READY_Q: np.ndarray = np.array([0.0, -0.6, 0.9, 0.0, 0.0, 0.6])

# Column of the gripperframe site xmat pointing wrist -> jaw tips. Chosen empirically:
# at q=0 wrist body (0.232,-0.018,0.234) -> site (0.391,0,0.246) is along +x = column 0,
# and flexing wrist_flex by +/-0.5 rotates column 0 in the x-z plane (z = -/+0.48),
# while column 1 stays fixed (the flex hinge axis).
APPROACH_AXIS: int = 0
