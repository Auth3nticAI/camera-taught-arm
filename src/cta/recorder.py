"""Session recorder: per-frame hand/arm data + video for Plan 2 training.

Layout under <root>/<name>/: frames.npz, human.mp4 (frame i <-> row i), meta.json.
Joint vectors are radians; absent data is NaN.

Per-row codes:
  status (uint8): 0 UNCALIBRATED, 1 NO HAND, 2 TRACKING, 3 COAST (hand lost < 0.15 s; command =
    last target, feats NaN), 255 unknown. Mask rows with status != 2
    (or hand_present) before training on feats; UNCALIBRATED rows carry features but the pipeline
    command is the constant READY_Q.
  smoother_on (int8): 1 smoother on, 0 off, -1 unknown.
  pose_img, pose_world (33,3), pose_vis, pose_presence (33,): MediaPipe Pose landmarks and their
    per-landmark visibility/presence (--control arm), float32, NaN when absent. Sessions
    recorded before arm mode have no such keys. meta.json "control" = "reach" | "arm" | "hand".
  depth_hand, depth_shoulder (float32, metres): --control reach, the palm-centre depth this frame and the
    shoulder depth used (Depth Anything V2), NaN when absent / no depth model. locked_side (int8): the arm
    locked at calibration, 0 = left, 1 = right, -1 = none. Sessions before 2026-10-18 lack these keys.
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
STATUS_CODES = {"UNCALIBRATED": 0, "NO HAND": 1, "TRACKING": 2, "COAST": 3}
STATUS_UNKNOWN = 255
SIDE_CODES = {"left": 0, "right": 1}  # locked_side column; -1 = none
_REPLACE_TRIES = 5
_REPLACE_SLEEP_S = 0.1


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
               "q_baseline": (6,), "q_leader": (6,), "q_follower": (6,), "status": (), "smoother_on": (),
               "pose_img": (33, 3), "pose_world": (33, 3), "pose_vis": (33,), "pose_presence": (33,),
               "depth_hand": (), "depth_shoulder": (), "locked_side": ()}
    _DTYPES = {"hand_present": bool, "status": np.uint8, "smoother_on": np.int8,
               "pose_img": np.float32, "pose_world": np.float32, "pose_vis": np.float32,
               "pose_presence": np.float32, "depth_hand": np.float32, "depth_shoulder": np.float32,
               "locked_side": np.int8}

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
        return np.zeros((self._cap, *self._SHAPES[k]), dtype=self._DTYPES.get(k, float))

    def add(self, t: float, obs: HandObservation | None, feats: HandFeatures | None,
            q_baseline: np.ndarray, q_leader: np.ndarray | None, q_follower: np.ndarray | None,
            rgb: np.ndarray, status: int = STATUS_UNKNOWN, smoother_on: int = -1,
            pose_img: np.ndarray | None = None, pose_world: np.ndarray | None = None,
            pose_vis: np.ndarray | None = None, pose_presence: np.ndarray | None = None,
            depth_hand: float | None = None, depth_shoulder: float | None = None,
            locked_side: str | None = None) -> None:
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
        b["status"][i] = status
        b["smoother_on"][i] = smoother_on
        b["pose_img"][i] = np.nan if pose_img is None else pose_img
        b["pose_world"][i] = np.nan if pose_world is None else pose_world
        b["pose_vis"][i] = np.nan if pose_vis is None else pose_vis
        b["pose_presence"][i] = np.nan if pose_presence is None else pose_presence
        b["depth_hand"][i] = np.nan if depth_hand is None else depth_hand
        b["depth_shoulder"][i] = np.nan if depth_shoulder is None else depth_shoulder
        b["locked_side"][i] = SIDE_CODES.get(locked_side, -1)
        w, h = self.video_size
        frame = rgb if (rgb.shape[1], rgb.shape[0]) == (w, h) else cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        self._video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # a row always gets a frame
        self.n += 1
        if time.monotonic() - self._last_save >= self.autosave_s:
            try:
                self.save()
            except Exception as e:  # e.g. Defender / a sync holds frames.npz; the next autosave retries
                print(f"WARNING: autosave failed ({e}); will retry")
                self._last_save = time.monotonic()

    def save(self) -> None:
        tmp = self.dir / "frames.tmp.npz"
        np.savez(tmp, **{k: a[: self.n] for k, a in self._buf.items()})
        for attempt in range(_REPLACE_TRIES):
            try:
                os.replace(tmp, self.dir / "frames.npz")
                break
            except OSError:  # PermissionError on a locked target (Windows)
                if attempt == _REPLACE_TRIES - 1:
                    raise
                time.sleep(_REPLACE_SLEEP_S)
        self._last_save = time.monotonic()

    def close(self) -> None:
        """Final save. Never raises on a save failure (caller still has arms to close)."""
        if self._closed:
            return
        try:
            self.save()
        except Exception as e:
            bar = "!" * 70
            print(bar)
            print(f"!! FINAL SAVE FAILED ({e}). Your data is in:")
            print(f"!!   {self.dir / 'frames.tmp.npz'}")
            print("!! rename it to frames.npz once the lock is gone.")
            print(bar)
        finally:
            self._closed = True
            self._video.release()

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_session(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    f = path / "frames.npz"
    if not f.exists():
        tmp = path / "frames.tmp.npz"
        if not tmp.exists():
            raise FileNotFoundError(f"no frames.npz (or frames.tmp.npz) in {path}")
        print(f"WARNING: {f.name} missing in {path}; loading {tmp.name}")
        f = tmp
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def list_sessions(root: Path, pattern: str = "*") -> list[Path]:
    """Take directories under root matching pattern that actually hold frames (skips orphans)."""
    return sorted(p for p in Path(root).glob(pattern)
                  if p.is_dir() and ((p / "frames.npz").exists() or (p / "frames.tmp.npz").exists()))
