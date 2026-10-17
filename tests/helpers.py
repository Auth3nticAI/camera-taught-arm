"""Shared test helpers: synthetic MediaPipe-style hand."""
import numpy as np

_MCP_R = 0.08  # wrist -> MCP distance, meters
_MCP_ANGLES = {5: -22.0, 9: 0.0, 13: 12.0, 17: 24.0}  # deg from hand axis, lateral
_SEGS = (0.04, 0.025, 0.02)  # proximal, middle, distal lengths
_FINGER_LEN_SCALE = {5: 1.0, 9: 1.1, 13: 1.0, 17: 0.8}


def _rodrigues(pts: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    return pts * c + np.cross(k, pts) * s + np.outer(pts @ k, k) * (1 - c)


def make_hand(
    scale: float = 1.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    pinch_closed: bool = False,
    wrist_uv: tuple[float, float] = (0.5, 0.5),
    aspect: float = 16 / 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic open right hand -> (lm_img (21,3), lm_world (21,3) meters).

    Camera-aligned axes: x right, y down, z away. At pitch=roll=0 the fingers point
    away from the camera (+z), palm level, index->pinky along +x. `roll` rotates about
    the hand axis (right-hand rule about wrist->middle-MCP); `pitch` > 0 tips the
    fingers up (-y). lm_img is an orthographic projection (0.08 m ~ 0.2 image heights
    at scale 1), multiplied by `scale`; u is divided by `aspect` so x is isotropic.
    """
    w = np.zeros((21, 3))
    for m, deg in _MCP_ANGLES.items():
        a = np.radians(deg)
        base = np.array([_MCP_R * np.sin(a), 0.0, _MCP_R * np.cos(a)])
        for j in range(4):
            if j == 0:
                w[m + j] = base
            else:
                w[m + j] = w[m + j - 1] + np.array(
                    [0.0, 0.0, _SEGS[j - 1] * _FINGER_LEN_SCALE[m]]
                )
    w[1] = [-0.025, 0.0, 0.02]
    w[2] = [-0.04, 0.0, 0.05]
    w[3] = [-0.05, 0.0, 0.08]
    w[4] = [-0.06, 0.0, 0.105]
    if pinch_closed:
        w[4] = w[8]
    w = _rodrigues(w, np.array([0.0, 0.0, 1.0]), roll)
    c, s = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # +pitch: +z axis -> (0,-sin,cos)
    w = w @ rx.T

    f = 2.5 * scale  # image heights per meter
    img = np.zeros((21, 3))
    img[:, 0] = wrist_uv[0] + w[:, 0] * f / aspect
    img[:, 1] = wrist_uv[1] + w[:, 1] * f
    img[:, 2] = w[:, 2] * f / aspect
    return img, w
