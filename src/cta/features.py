"""Hand features from MediaPipe landmarks (pure numpy)."""
from dataclasses import dataclass

import numpy as np

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
_MCPS = (INDEX_MCP, MIDDLE_MCP, PINKY_MCP)


@dataclass
class HandFeatures:
    u: float  # wrist, normalized image x
    v: float  # wrist, normalized image y (down)
    scale: float  # mean wrist->MCP image distance, fraction of image width
    pitch: float  # rad, elevation of wrist->middle-MCP above camera x-z plane; up = +
    roll: float  # rad, see compute_features
    pinch: float  # thumb tip - index tip distance / palm size (world)


def compute_features(lm_img: np.ndarray, lm_world: np.ndarray, aspect: float) -> HandFeatures:
    """Compute features. lm_img (21,3) normalized; lm_world (21,3) meters; aspect = w/h.

    World axes (camera-aligned): x right, y down, z away from camera.
    pitch: asin(-axis_y) where axis = unit(wrist -> middle MCP); +ve = fingers up.
    roll: signed angle about the hand axis (right-hand rule about axis) from the
    camera x-axis (projected onto the plane normal to the axis) to the index-MCP ->
    pinky-MCP vector (projected likewise). Zero for a level palm (fingers along z,
    index->pinky along +x); ccw about the axis (fingers toward +z => from +x toward +y)
    is positive.
    """
    lm_img = np.asarray(lm_img, dtype=float)
    w = np.asarray(lm_world, dtype=float)

    d = lm_img[list(_MCPS), :2] - lm_img[WRIST, :2]
    dist_h = np.hypot(d[:, 0] * aspect, d[:, 1])  # in image-height units
    scale = float(dist_h.mean() / aspect)  # -> fraction of image width

    axis = w[MIDDLE_MCP] - w[WRIST]
    palm = float(np.linalg.norm(w[list(_MCPS)] - w[WRIST], axis=1).mean())
    axis = axis / np.linalg.norm(axis)
    pitch = float(np.arcsin(np.clip(-axis[1], -1.0, 1.0)))

    ref = np.array([1.0, 0.0, 0.0])
    ref = ref - axis * (ref @ axis)
    lat = w[PINKY_MCP] - w[INDEX_MCP]
    lat = lat - axis * (lat @ axis)
    roll = float(np.arctan2(axis @ np.cross(ref, lat), ref @ lat))

    pinch = float(np.linalg.norm(w[THUMB_TIP] - w[INDEX_TIP]) / palm)
    return HandFeatures(float(lm_img[WRIST, 0]), float(lm_img[WRIST, 1]), scale, pitch, roll, pinch)
