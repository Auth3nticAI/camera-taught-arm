"""Session recorder: per-frame hand/arm data + video for Plan 2 training.

Layout under <root>/<name>/: frames.npz, human.mp4 (frame i <-> row i), meta.json.
Joint vectors are radians; absent data is NaN.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from cta.features import HandFeatures
from cta.hands import HandObservation

VIDEO_FPS = 60


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                           cwd=Path(__file__).resolve().parent)
        return r.stdout.strip() or "unknown" if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def next_take_name(root: Path, name: str) -> str:
    """First unused '<name>_<NN>' under root."""
    i = 1
    while (Path(root) / f"{name}_{i:02d}").exists():
        i += 1
    return f"{name}_{i:02d}"


def _vec(q: np.ndarray | None) -> np.ndarray:
    return np.full(6, np.nan) if q is None else np.asarray(q, dtype=float).reshape(6)


class SessionRecorder:
    _SHAPES = {"t": (), "hand_present": (), "lm_img": (21, 3), "lm_world": (21, 3), "feats": (6,),
               "q_baseline": (6,), "q_leader": (6,), "q_follower": (6,)}

    def __init__(self, root: Path, name: str, meta: dict, autosave_s: float = 30.0,
                 video_size: tuple[int, int] = (1280, 720)):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.autosave_s = autosave_s
        self.video_size = tuple(video_size)
        self.n = 0
        self._cap = 1024
        self._buf = {k: self._alloc(k) for k in self._SHAPES}
        self._closed = False
        (self.dir / "meta.json").write_text(json.dumps({**meta, "name": name}, indent=2, default=str))
        self._video = cv2.VideoWriter(str(self.dir / "human.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                      VIDEO_FPS, self.video_size)
        if not self._video.isOpened():
            raise RuntimeError(f"cannot open video writer for {self.dir / 'human.mp4'}")
        self._last_save = time.monotonic()

    def _alloc(self, k: str) -> np.ndarray:
        dt = bool if k == "hand_present" else float
        return np.zeros((self._cap, *self._SHAPES[k]), dtype=dt)

    def add(self, t: float, obs: HandObservation | None, feats: HandFeatures | None,
            q_baseline: np.ndarray, q_leader: np.ndarray | None, q_follower: np.ndarray | None,
            rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("recorder is closed")
        if self.n == self._cap:
            self._cap *= 2
            for k, a in self._buf.items():
                self._buf[k] = np.concatenate([a, self._alloc(k)])
        i, b = self.n, self._buf
        b["t"][i] = t
        b["hand_present"][i] = obs is not None
        b["lm_img"][i] = np.nan if obs is None else obs.lm_img
        b["lm_world"][i] = np.nan if obs is None else obs.lm_world
        b["feats"][i] = np.nan if feats is None else [feats.u, feats.v, feats.scale, feats.pitch,
                                                      feats.roll, feats.pinch]
        b["q_baseline"][i] = _vec(q_baseline)
        b["q_leader"][i] = _vec(q_leader)
        b["q_follower"][i] = _vec(q_follower)
        w, h = self.video_size
        frame = rgb if (rgb.shape[1], rgb.shape[0]) == (w, h) else cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        self._video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # a row always gets a frame
        self.n += 1
        if time.monotonic() - self._last_save >= self.autosave_s:
            self.save()

    def save(self) -> None:
        tmp = self.dir / "frames.tmp.npz"
        np.savez(tmp, **{k: a[: self.n] for k, a in self._buf.items()})
        os.replace(tmp, self.dir / "frames.npz")
        self._last_save = time.monotonic()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.save()
        finally:
            self._closed = True
            self._video.release()

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_session(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path) / "frames.npz") as z:
        return {k: z[k] for k in z.files}
