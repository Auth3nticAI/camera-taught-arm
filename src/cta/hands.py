"""MediaPipe Tasks HandLandmarker wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


@dataclass
class HandObservation:
    lm_img: np.ndarray    # (21, 3) normalized image coords
    lm_world: np.ndarray  # (21, 3) meters
    side: str             # user's actual hand: "right" or "left"
    score: float


def _actual_side(label: str, mirrored_input: bool) -> str:
    """MediaPipe assumes a mirrored selfie; on unmirrored frames its label is the opposite hand."""
    label = label.lower()
    if mirrored_input:
        return label
    return "left" if label == "right" else "right"


def select_hand(handedness_labels: list[str], scores: list[float], want: str, mirrored_input: bool) -> int | None:
    """Index of the best-scoring detected hand that is the user's `want` hand."""
    best, best_score = None, -1.0
    for i, (lab, sc) in enumerate(zip(handedness_labels, scores)):
        if _actual_side(lab, mirrored_input) == want and sc > best_score:
            best, best_score = i, sc
    return best


class HandTracker:
    def __init__(self, model_path: Path, want: str = "right", mirrored_input: bool = False, infer_width: int = 960):
        self.want = want
        self.mirrored_input = mirrored_input
        self.infer_width = infer_width
        self._last_ms = -1
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray, t: float) -> HandObservation | None:
        h, w = rgb.shape[:2]
        if w > self.infer_width:
            rgb = cv2.resize(rgb, (self.infer_width, round(h * self.infer_width / w)), interpolation=cv2.INTER_AREA)
        ms = int(t * 1000)
        if ms <= self._last_ms:  # VIDEO mode needs strictly increasing timestamps
            ms = self._last_ms + 1
        self._last_ms = ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(img, ms)
        if not res.hand_landmarks:
            return None
        labels = [c[0].category_name for c in res.handedness]
        scores = [c[0].score for c in res.handedness]
        i = select_hand(labels, scores, self.want, self.mirrored_input)
        if i is None:
            return None
        return HandObservation(
            lm_img=np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[i]], dtype=np.float64),
            lm_world=np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]], dtype=np.float64),
            side=self.want,
            score=float(scores[i]),
        )

    def close(self) -> None:
        self._lm.close()
