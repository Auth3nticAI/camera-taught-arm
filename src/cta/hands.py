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


MAX_WRIST_JUMP = 0.15  # normalized image units: farther than this from the last wrist = a different hand
LOST_FRAMES = 15        # after this many frames without the hand, re-acquire by handedness label


def choose_hand(handedness_labels: list[str], scores: list[float], wrists: list[tuple[float, float]],
                want: str, mirrored_input: bool, last_wrist: tuple[float, float] | None) -> int | None:
    """Keep following the SAME hand by wrist position; MediaPipe's left/right label flickers frame to
    frame on small or occluded hands (seen live 2026-10-17: TRACKING/NO HAND every other frame).
    The label is only used to acquire the hand when there is no recent position."""
    if last_wrist is not None:
        d = [np.hypot(w[0] - last_wrist[0], w[1] - last_wrist[1]) for w in wrists]
        i = int(np.argmin(d))
        if d[i] <= MAX_WRIST_JUMP:
            return i
    return select_hand(handedness_labels, scores, want, mirrored_input)


class HandTracker:
    def __init__(self, model_path: Path, want: str = "right", mirrored_input: bool = False, infer_width: int = 1280):
        self.want = want
        self._last_wrist: tuple[float, float] | None = None
        self._missed = 0
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
            return self._miss()
        labels = [c[0].category_name for c in res.handedness]
        scores = [c[0].score for c in res.handedness]
        wrists = [(h[0].x, h[0].y) for h in res.hand_landmarks]
        i = choose_hand(labels, scores, wrists, self.want, self.mirrored_input, self._last_wrist)
        if i is None:
            return self._miss()
        self._last_wrist, self._missed = wrists[i], 0
        return HandObservation(
            lm_img=np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[i]], dtype=np.float64),
            lm_world=np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]], dtype=np.float64),
            side=self.want,
            score=float(scores[i]),
        )

    def _miss(self) -> None:
        self._missed += 1
        if self._missed >= LOST_FRAMES:
            self._last_wrist = None
        return None

    def close(self) -> None:
        self._lm.close()
