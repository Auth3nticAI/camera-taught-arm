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


def limit_speed(prev: np.ndarray, target: np.ndarray, dt: float, max_speed_rad_s: float) -> np.ndarray:
    """Per-joint |delta| <= max_speed * dt (dt clamped to [0, 0.1])."""
    dt = float(np.clip(dt, 0.0, 0.1))
    prev = np.asarray(prev, dtype=float)
    step = max_speed_rad_s * dt
    return prev + np.clip(np.asarray(target, dtype=float) - prev, -step, step)


def select_command(mode: str, status: str, pipeline_q: np.ndarray, leader_q: np.ndarray | None,
                   prev: np.ndarray, dt: float, max_speed_rad_s: float) -> np.ndarray:
    """Per-frame safe command. mirror: follow pipeline while TRACKING/COAST, else hold prev.
    paired: follow the leader only (pipeline q is never sent)."""
    if mode == "paired":
        desired = prev if leader_q is None else leader_q
    else:
        desired = pipeline_q if status in ("TRACKING", "COAST") else prev  # COAST: brief dropout
    return limit_speed(prev, clamp_margin(desired), dt, max_speed_rad_s)


def offset_from_reference(real_deg: float, sim_rad: float, sign: float) -> float:
    """offset_deg such that sign*degrees(sim_rad) + offset == real_deg."""
    return float(real_deg - sign * np.degrees(sim_rad))


class ArmIO(Protocol):
    def connect(self) -> None: ...
    def send(self, q: np.ndarray) -> object: ...
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
        try:
            self._dev.connect(calibrate=False)
        except Exception:
            try:
                self._dev.disconnect()  # release a half-open bus / torque
            except Exception:
                pass
            raise
        self._connected = True
        if not self._dev.is_calibrated:
            self.close()
            raise RuntimeError(
                f"{self._what} is not calibrated. Run:\n  {self._cal_cmd.format(port=self.port)}")

    def calibration_info(self) -> dict:
        return {"id": getattr(self._dev, "id", None), "path": str(getattr(self._dev, "calibration_fpath", None))}

    def close(self) -> None:
        if self._dev is not None and self._connected:
            self._connected = False
            self._dev.disconnect()


MAX_RELATIVE_TARGET_DEG = 15.0  # LeRobot's per-command backstop (our own lead limit is tighter)
SERVO_P_RANGE = (8, 32)         # Feetech P_Coefficient: LeRobot sets 16 (soft); factory default is 32


class FollowerArm(_LeRobotArm):
    _what = "SO-101 follower (cta_follower)"
    _cal_cmd = "uv run lerobot-calibrate --robot.type=so101_follower --robot.port={port} --robot.id=cta_follower"

    def _make(self):  # pragma: no cover
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        return SO101Follower(SO101FollowerConfig(port=self.port, id="cta_follower",
                                                 max_relative_target=MAX_RELATIVE_TARGET_DEG))

    def set_servo_p(self, p: int) -> None:
        """Servo stiffness for the 5 arm joints (gripper untouched). Higher = faster, tighter tracking
        but more shake. Call right after connect while the arm rests: torque is briefly disabled
        (as LeRobot's own configure() does) because the register is in the servo's EEPROM area."""
        lo, hi = SERVO_P_RANGE
        if not lo <= int(p) <= hi:
            raise ValueError(f"servo P must be in [{lo}, {hi}], got {p}")
        bus = self._dev.bus
        with bus.torque_disabled():
            for motor in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
                bus.write("P_Coefficient", motor, int(p))

    def send(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError(f"refusing to send non-finite / mis-shaped command: {q}")
        action = sim_to_real(q, self.map)
        if not all(np.isfinite(v) for v in action.values()):
            raise ValueError(f"refusing to send non-finite action: {action}")
        ret = self._dev.send_action(action)  # LeRobot returns the action after max_relative_target clipping
        return real_to_sim(_joint_obs(ret if ret is not None else action), self.map)

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
