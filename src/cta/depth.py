"""Monocular metric depth (Depth Anything V2 Metric Indoor Small, via transformers) + robust patch sampling.

Used by --control reach for the FORWARD part of "where": depth at the hand minus depth at the locked
shoulder, in metres. Lazy: nothing is downloaded or loaded until load()/predict(). GPU + fp16 when CUDA
is available, else CPU fp32. If the model can't load (no network on first use, no transformers, ...),
load() returns False with `.error` set and predict() returns None; the reach pipeline then falls back to
the hand-scale cue (cta.reach.fwd_from_hand_scale).
"""
from __future__ import annotations

import time

import numpy as np

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEFAULT_INPUT_H = 518      # model input height (multiple of 14; 518 = its training size); width follows the
                           # frame aspect. Metric scale shifts with input size (replay: ~1.5 m at 266 vs ~1.1 m)
PATCH_RADIUS_FRAC = 0.02   # patch half-size, fraction of the depth map height


def sample_patch(depth: np.ndarray | None, uv, radius_frac: float = PATCH_RADIUS_FRAC) -> float:
    """Median of the finite depth values in a square patch around normalized image point uv (u right, v down).

    The patch is clipped at the image edges; uv outside [0, 1] (or non-finite / None) gives NaN, as does a
    patch with no finite values."""
    if depth is None or uv is None:
        return float("nan")
    u, v = float(uv[0]), float(uv[1])
    if not (np.isfinite(u) and np.isfinite(v)) or not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return float("nan")
    h, w = depth.shape[:2]
    r = max(1, int(round(radius_frac * h)))
    cx, cy = min(int(u * w), w - 1), min(int(v * h), h - 1)
    patch = np.asarray(depth[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1], dtype=float)
    vals = patch[np.isfinite(patch)]
    return float(np.median(vals)) if vals.size else float("nan")


class DepthEstimator:
    """Depth Anything V2 wrapper: predict(rgb uint8 HxWx3) -> metric depth (h x w float32, metres) or None."""

    def __init__(self, model_id: str = DEPTH_MODEL_ID, input_height: int = DEFAULT_INPUT_H,
                 device: str | None = None, local_only: bool = False):
        self.model_id = model_id
        self.input_height = int(round(input_height / 14) * 14)
        self.device = device
        self.local_only = local_only
        self.error: str | None = None
        self.last_ms = float("nan")
        self._model = None
        self._tried = False

    def load(self) -> bool:
        """Load once (download into the normal HF cache on first use). True when ready."""
        if self._model is not None:
            return True
        if self._tried:
            return False
        self._tried = True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            proc = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=self.local_only)
            model = AutoModelForDepthEstimation.from_pretrained(self.model_id, local_files_only=self.local_only)
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._dtype = torch.float16 if dev.startswith("cuda") else torch.float32
            self._model = model.to(dev, dtype=self._dtype).eval()
            self.device = dev
            self._torch = torch
            # the processor only supplies the normalization; resizing is done on the GPU in predict()
            self._mean = torch.tensor(proc.image_mean, device=dev, dtype=self._dtype).view(1, 3, 1, 1)
            self._std = torch.tensor(proc.image_std, device=dev, dtype=self._dtype).view(1, 3, 1, 1)
            return True
        except Exception as e:  # no network / no model / no torch: the caller falls back
            self.error = f"{type(e).__name__}: {e}"
            self._model = None
            return False

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict(self, rgb: np.ndarray) -> np.ndarray | None:
        if not self.load():
            return None
        torch = self._torch
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        ih = self.input_height
        iw = max(14, int(round(ih * w / h / 14)) * 14)
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
            x = x.permute(2, 0, 1).unsqueeze(0).to(self._dtype) / 255.0
            x = torch.nn.functional.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False,
                                                antialias=True)
            x = (x - self._mean) / self._std
            out = self._model(pixel_values=x).predicted_depth  # (1, ih, iw) metres
            d = out[0].float().cpu().numpy()
        self.last_ms = (time.perf_counter() - t0) * 1000
        return d.astype(np.float32, copy=False)
