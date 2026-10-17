"""Webcam capture with a startup mode check (never uses HighGUI windows)."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraMode:
    width: int
    height: int
    fps: float


def camera_check(actual: CameraMode, wanted: CameraMode = CameraMode(1920, 1080, 60)) -> list[str]:
    """Warnings if the measured mode differs from wanted; empty if OK."""
    warnings = []
    if (actual.width, actual.height) != (wanted.width, wanted.height):
        warnings.append(
            f"camera size {actual.width}x{actual.height}, wanted {wanted.width}x{wanted.height}"
            " - check the PTZ app preset (tracking, gesture, zoom off)"
        )
    if actual.fps < 0.85 * wanted.fps:
        warnings.append(
            f"camera fps {actual.fps:.1f}, wanted >= {0.85 * wanted.fps:.1f} (of {wanted.fps:g})"
        )
    return warnings


class Camera:
    def __init__(self, index: int, wanted: CameraMode):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, wanted.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wanted.height)
        self.cap.set(cv2.CAP_PROP_FPS, wanted.fps)

    def is_open(self) -> bool:
        return bool(self.cap.isOpened())

    def read(self) -> tuple[np.ndarray | None, float]:
        """RGB frame (or None on failure) and time.perf_counter() after the read."""
        ok, bgr = self.cap.read()
        t = time.perf_counter()
        if not ok or bgr is None:
            return None, t
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t

    def measure_mode(self, n: int = 60) -> CameraMode:
        """Measure real size and fps from n reads (CAP_PROP_FPS is not trusted)."""
        times: list[float] = []
        w = h = 0
        for _ in range(n):
            frame, t = self.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            times.append(t)
        if len(times) < 2 or times[-1] <= times[0]:
            return CameraMode(w, h, 0.0)
        return CameraMode(w, h, (len(times) - 1) / (times[-1] - times[0]))

    def close(self) -> None:
        self.cap.release()


def list_cameras(max_index: int = 6) -> list[tuple[int, int, int]]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append((i, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="probe indices 0-5")
    args = ap.parse_args()
    if args.list:
        cams = list_cameras()
        for i, w, h in cams:
            print(f"index {i}: opens, reports {w}x{h}")
        if not cams:
            print("no cameras opened on indices 0-5")
        return
    cam = Camera(args.index, CameraMode(1920, 1080, 60))
    try:
        if not cam.is_open():
            print(f"WARNING: camera {args.index} did not open")
            return
        mode = cam.measure_mode()
        print(f"measured mode: {mode.width}x{mode.height} @ {mode.fps:.1f} fps")
        warns = camera_check(mode)
        for w in warns:
            print(f"WARNING: {w}")
        if not warns:
            print("camera mode OK")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
