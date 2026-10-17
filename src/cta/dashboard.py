"""Rerun dashboard (single window; no HighGUI)."""
from __future__ import annotations

import cv2
import numpy as np
import rerun as rr

from cta.config import JOINTS
from cta.hands import HandObservation

LOG_WIDTH = 960  # camera image is logged at the size MediaPipe sees


class Dashboard:
    def __init__(self, app_id: str = "cta"):
        rr.init(app_id, spawn=True)
        self._last_status: str | None = None

    def log_frame(self, rgb: np.ndarray | None, obs: HandObservation | None, twin_rgb: np.ndarray | None,
                  q_cmd: np.ndarray, fps: float, latency_ms: float, status: str,
                  loop_ms: float | None = None, extra_rad: dict[str, np.ndarray] | None = None) -> None:
        """Scalars/status every call; images only when rgb / twin_rgb are given (decimation)."""
        if rgb is not None:
            h, w = rgb.shape[:2]
            if w > LOG_WIDTH:
                h, w = round(h * LOG_WIDTH / w), LOG_WIDTH
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            rr.log("camera/image", rr.Image(rgb).compress(jpeg_quality=70))
            if obs is not None:
                rr.log("camera/image/hand", rr.Points2D(obs.lm_img[:, :2] * np.array([w, h]), radii=4.0))
            else:
                rr.log("camera/image/hand", rr.Clear(recursive=False))
        if twin_rgb is not None:
            rr.log("twin/image", rr.Image(twin_rgb))
        for i, name in enumerate(JOINTS):
            rr.log(f"joints/{name}", rr.Scalars(float(np.degrees(q_cmd[i]))))
        for prefix, vec in (extra_rad or {}).items():
            for i, name in enumerate(JOINTS):
                rr.log(f"{prefix}/{name}", rr.Scalars(float(np.degrees(vec[i]))))
        rr.log("stats/fps", rr.Scalars(float(fps)))
        rr.log("stats/latency_ms", rr.Scalars(float(latency_ms)))
        if loop_ms is not None:
            rr.log("stats/loop_ms", rr.Scalars(float(loop_ms)))
        if status != self._last_status:
            rr.log("status", rr.TextLog(status))
            self._last_status = status
