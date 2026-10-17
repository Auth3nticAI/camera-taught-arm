"""Real-arm I/O (LeRobot SO-101). LeRobot imports are lazy so tests / --robot none need no serial SDK."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from cta.config import JOINTS, SIM_LIMITS
from cta.units import SimRealMap, real_to_sim, sim_to_real


def clamp_margin(q: np.ndarray, margin_rad: float = 0.087) -> np.ndarray:
    """Keep commands margin_rad (5 deg) inside SIM_LIMITS."""
    q = np.asarray(q, dtype=float)
    return np.clip(q, SIM_LIMITS[:, 0] + margin_rad, SIM_LIMITS[:, 1] - margin_rad)


def offset_from_reference(real_deg: float, sim_rad: float, sign: float) -> float:
    """offset_deg such that sign*degrees(sim_rad) + offset == real_deg."""
    return float(real_deg - sign * np.degrees(sim_rad))


class ArmIO(Protocol):
    def connect(self) -> None: ...
    def send(self, q: np.ndarray) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class NullArm:
    def __init__(self) -> None:
        self._last: np.ndarray | None = None

    def connect(self) -> None:
        pass

    def send(self, q: np.ndarray) -> None:
        self._last = np.asarray(q, dtype=float).copy()

    def read(self) -> np.ndarray | None:
        return None if self._last is None else self._last.copy()

    def close(self) -> None:
        pass


def _joint_obs(d: dict) -> dict[str, float]:
    return {f"{j}.pos": float(d[f"{j}.pos"]) for j in JOINTS}  # drops camera / extra keys


class _LeRobotArm:
    _what = ""
    _cal_cmd = ""

    def __init__(self, port: str, device=None, sim_real: SimRealMap | None = None):
        self.port = port
        self._dev = device
        self.map = sim_real or SimRealMap.load()
        self._connected = False

    def _make(self):  # pragma: no cover - needs lerobot
        raise NotImplementedError

    def connect(self) -> None:
        if self._dev is None:
            self._dev = self._make()
        self._dev.connect(calibrate=False)
        self._connected = True
        if not self._dev.is_calibrated:
            self.close()
            raise RuntimeError(
                f"{self._what} is not calibrated. Run:\n  {self._cal_cmd.format(port=self.port)}")

    def close(self) -> None:
        if self._dev is not None and self._connected:
            self._connected = False
            self._dev.disconnect()


class FollowerArm(_LeRobotArm):
    _what = "SO-101 follower (cta_follower)"
    _cal_cmd = "uv run lerobot-calibrate --robot.type=so101_follower --robot.port={port} --robot.id=cta_follower"

    def _make(self):  # pragma: no cover
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        return SO101Follower(SO101FollowerConfig(port=self.port, id="cta_follower", max_relative_target=10.0))

    def send(self, q: np.ndarray) -> None:
        self._dev.send_action(sim_to_real(q, self.map))

    def read(self) -> np.ndarray | None:
        return real_to_sim(_joint_obs(self._dev.get_observation()), self.map)


class LeaderArm(_LeRobotArm):
    _what = "SO-101 leader (cta_leader)"
    _cal_cmd = "uv run lerobot-calibrate --teleop.type=so101_leader --teleop.port={port} --teleop.id=cta_leader"

    def _make(self):  # pragma: no cover
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        return SO101Leader(SO101LeaderConfig(port=self.port, id="cta_leader"))

    def read(self) -> np.ndarray:
        return real_to_sim(_joint_obs(self._dev.get_action()), self.map)

    def read_raw(self) -> dict[str, float]:
        return _joint_obs(self._dev.get_action())
