"""Rerun dashboard (single window; no HighGUI)."""
from __future__ import annotations

import numpy as np
import rerun as rr

from cta.config import JOINTS
from cta.hands import HandObservation


class Dashboard:
    def __init__(self, app_id: str = "cta"):
        rr.init(app_id, spawn=True)

    def log_frame(self, rgb: np.ndarray | None, obs: HandObservation | None, twin_rgb: np.ndarray,
                  q_cmd: np.ndarray, fps: float, latency_ms: float, status: str) -> None:
        if rgb is not None:
            rr.log("camera/image", rr.Image(rgb).compress(jpeg_quality=70))
            if obs is not None:
                h, w = rgb.shape[:2]
                pts = obs.lm_img[:, :2] * np.array([w, h])
                rr.log("camera/image/hand", rr.Points2D(pts, radii=4.0))
            else:
                rr.log("camera/image/hand", rr.Clear(recursive=False))
        rr.log("twin/image", rr.Image(twin_rgb))
        for i, name in enumerate(JOINTS):
            rr.log(f"joints/{name}", rr.Scalars(float(np.degrees(q_cmd[i]))))
        rr.log("stats/fps", rr.Scalars(float(fps)))
        rr.log("stats/latency_ms", rr.Scalars(float(latency_ms)))
        rr.log("status", rr.TextLog(status))
