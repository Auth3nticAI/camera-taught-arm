"""Sim (radians) <-> real (LeRobot SO-101: degrees for joints 1-5, 0-100 gripper)."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cta.config import JOINTS, SIM_LIMITS
from cta.paths import data_dir


def _default_path() -> Path:
    return data_dir() / "sim_real_map.json"


@dataclass
class SimRealMap:
    sign: np.ndarray = field(default_factory=lambda: np.ones(5))
    offset_deg: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self) -> None:
        self.sign = np.asarray(self.sign, dtype=float)
        self.offset_deg = np.asarray(self.offset_deg, dtype=float)
        if self.sign.shape != (5,) or self.offset_deg.shape != (5,):
            raise ValueError(f"SimRealMap sign/offset_deg must have shape (5,), got {self.sign.shape}/{self.offset_deg.shape}")
        if not np.all(np.abs(self.sign) == 1.0):
            raise ValueError(f"SimRealMap sign values must be +1 or -1, got {self.sign.tolist()}")
        if not np.all(np.isfinite(self.offset_deg)):
            raise ValueError(f"SimRealMap offset_deg must be finite, got {self.offset_deg.tolist()}")

    def save(self, path: Path | None = None) -> None:
        p = Path(path) if path else _default_path()
        p.write_text(json.dumps({"sign": self.sign.tolist(), "offset_deg": self.offset_deg.tolist()}))

    @classmethod
    def load(cls, path: Path | None = None) -> "SimRealMap":
        p = Path(path) if path else _default_path()
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        return cls(sign=np.array(d["sign"], dtype=float), offset_deg=np.array(d["offset_deg"], dtype=float))


def sim_to_real(q: np.ndarray, m: SimRealMap) -> dict[str, float]:
    q = np.asarray(q, dtype=float)
    deg = m.sign * np.degrees(q[:5]) + m.offset_deg
    lo, hi = SIM_LIMITS[5]
    out = {f"{j}.pos": float(deg[i]) for i, j in enumerate(JOINTS[:5])}
    out["gripper.pos"] = float(100.0 * (q[5] - lo) / (hi - lo))
    return out


def real_to_sim(obs: dict[str, float], m: SimRealMap) -> np.ndarray:
    deg = np.array([obs[f"{j}.pos"] for j in JOINTS[:5]], dtype=float)
    lo, hi = SIM_LIMITS[5]
    q = np.zeros(6)
    q[:5] = np.radians((deg - m.offset_deg) / m.sign)
    q[5] = lo + obs["gripper.pos"] / 100.0 * (hi - lo)
    return q
