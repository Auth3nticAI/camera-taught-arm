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


# ---- synthetic MediaPipe-style body (pose landmarks) ----------------------------------------------
POSE_LR_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
                 (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32))
# body frame (fwd, left, up) -> camera axes (x right, y down, z away) for a user facing the camera in an
# UNMIRRORED image: their chest faces -z, their left appears on the image right (+x), up is -y
_BODY_TO_CAM = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])  # columns: fwd, left, up


def body_dir(az_deg: float, elev_deg: float) -> np.ndarray:
    """Unit vector in body coords (fwd, left, up): az from fwd toward the user's left, elev from up."""
    az, el = np.radians(az_deg), np.radians(elev_deg)
    return np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)])


def bend(upper: np.ndarray, flex_deg: float) -> np.ndarray:
    """Forearm direction: `upper` folded by flex_deg inside its vertical plane, the way the SO-101 folds
    (from an upper arm pointing up, folding swings the forearm forward/down)."""
    h = np.array([upper[0], upper[1], 0.0])
    h = h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-9 else np.array([1.0, 0.0, 0.0])
    n = np.cross([0.0, 0.0, 1.0], h)  # plane normal (az = 0 -> the user's left)
    return _rodrigues(upper[None, :], n, np.radians(flex_deg))[0]


def make_body(upper: np.ndarray, fore: np.ndarray | None = None, side: str = "left",
              mirrored: bool = False, aspect: float = 16 / 9, upper_len: float = 0.28,
              fore_len: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic pose -> (img (33,3) normalized, world (33,3) meters, hip-centred camera axes).

    The user faces the camera; shoulders 0.35 m apart, hips 0.5 m below. The `side` arm points along
    `upper` (body coords fwd/left/up) and its forearm along `fore` (default: straight); the other arm
    hangs down. mirrored=True returns what MediaPipe reports for the same body in a MIRRORED image:
    x flipped (image 1-u, world -x) and the left/right landmark labels swapped.
    """
    fore = upper if fore is None else fore
    B = {}
    B["l_sh"], B["r_sh"] = np.array([0.0, 0.175, 0.5]), np.array([0.0, -0.175, 0.5])
    B["l_hip"], B["r_hip"] = np.array([0.0, 0.1, 0.0]), np.array([0.0, -0.1, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    arms = {}
    for s, sh in (("left", B["l_sh"]), ("right", B["r_sh"])):
        a, b = (np.asarray(upper, float), np.asarray(fore, float)) if s == side else (down, down)
        el = sh + upper_len * a / np.linalg.norm(a)
        arms[s] = (sh, el, el + fore_len * b / np.linalg.norm(b))
    body = np.zeros((33, 3))
    body[:11] = (B["l_sh"] + B["r_sh"]) / 2 + np.array([0.0, 0.0, 0.2])  # head
    body[11], body[13], body[15] = arms["left"]
    body[12], body[14], body[16] = arms["right"]
    for i in (17, 19, 21):
        body[i] = body[15]
    for i in (18, 20, 22):
        body[i] = body[16]
    body[23], body[24] = B["l_hip"], B["r_hip"]
    for i in (25, 27, 29, 31):
        body[i] = B["l_hip"] + np.array([0.0, 0.0, -0.5])
    for i in (26, 28, 30, 32):
        body[i] = B["r_hip"] + np.array([0.0, 0.0, -0.5])
    world = body @ _BODY_TO_CAM.T
    world -= (world[23] + world[24]) / 2
    f = 1.0  # image heights per meter
    img = np.zeros((33, 3))
    img[:, 0] = 0.5 + world[:, 0] * f / aspect
    img[:, 1] = 0.6 + world[:, 1] * f
    img[:, 2] = world[:, 2] * f / aspect
    if mirrored:
        img[:, 0] = 1.0 - img[:, 0]
        world[:, 0] = -world[:, 0]
        for i, j in POSE_LR_PAIRS:
            img[[i, j]] = img[[j, i]]
            world[[i, j]] = world[[j, i]]
    return img, world
