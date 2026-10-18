"""MediaPipe Tasks PoseLandmarker wrapper (one body, VIDEO mode)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

_ARM_LANDMARKS = (11, 12, 13, 14, 15, 16)  # shoulders, elbows, wrists


@dataclass
class PoseObservation:
    img: np.ndarray    # (33, 3) normalized image coords (in the camera's own, possibly mirrored, image)
    world: np.ndarray  # (33, 3) meters, hip-centred, camera axes: x right, y down, z away
    score: float       # mean visibility of the shoulder/elbow/wrist landmarks
    vis: np.ndarray = field(default_factory=lambda: np.ones(33))       # (33,) per-landmark visibility 0..1
    presence: np.ndarray = field(default_factory=lambda: np.ones(33))  # (33,) per-landmark presence 0..1


class PoseTracker:
    def __init__(self, model_path: Path, infer_width: int = 640):  # the nets themselves run at 224/256 px
        self.infer_width = infer_width
        self._last_ms = -1
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray, t: float) -> PoseObservation | None:
        h, w = rgb.shape[:2]
        if w > self.infer_width:
            rgb = cv2.resize(rgb, (self.infer_width, round(h * self.infer_width / w)), interpolation=cv2.INTER_AREA)
        ms = int(t * 1000)
        if ms <= self._last_ms:  # VIDEO mode needs strictly increasing timestamps
            ms = self._last_ms + 1
        self._last_ms = ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(img, ms)
        if not res.pose_landmarks or not res.pose_world_landmarks:
            return None
        lm, wl = res.pose_landmarks[0], res.pose_world_landmarks[0]
        vis = np.array([1.0 if p.visibility is None else p.visibility for p in lm], dtype=np.float64)
        pres = np.array([1.0 if p.presence is None else p.presence for p in lm], dtype=np.float64)
        return PoseObservation(
            img=np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64),
            world=np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float64),
            score=float(vis[list(_ARM_LANDMARKS)].mean()),
            vis=vis, presence=pres,
        )

    def close(self) -> None:
        self._lm.close()
