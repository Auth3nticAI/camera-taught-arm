"""Follower-only alignment: python -m cta.apps.align_follower --follower-port COM4

The follower stays LIMP the whole time (torque is disabled and never re-enabled), so it cannot move
on its own. Place the closed jaws touching the table, press SPACE to capture; repeat in varied poses
(near/far, left/right, wrist angled). Then 's' solves, 'w' writes sim_real_map.json (old one -> .bak).
Keys need this terminal focused (Windows console) - see _keys().
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time

import numpy as np
import rerun as rr

from cta.align import MIN_POSES, ContactModel, solve_offsets
from cta.config import JOINTS
from cta.paths import data_dir
from cta.twin import Twin
from cta.units import SimRealMap, real_to_sim

HELP = (
    "SPACE/ENTER = capture touch pose · u = undo last · s = solve · w = write map · q = quit\n"
    f"Capture at least {MIN_POSES} (aim for 8-10): near and far, left and right, wrist bent different amounts,\n"
    "jaws CLOSED and just touching the table (no pressing down)."
)


def _keys() -> list[str]:
    if sys.platform == "win32":
        import msvcrt

        out = []
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # special-key prefix: swallow the second code
                msvcrt.getwch()
                continue
            out.append(ch)
        return out
    return []  # non-Windows: use --line-mode


def _connect(port: str):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    bus = SO101Follower(SO101FollowerConfig(port=port, id="cta_follower")).bus
    bus.connect()
    bus.disable_torque()
    if not bus.is_calibrated:
        bus.disconnect(disable_torque=True)
        raise RuntimeError(
            "follower not calibrated: run uv run lerobot-calibrate --robot.type=so101_follower "
            f"--robot.port={port} --robot.id=cta_follower"
        )
    return bus


def _read_deg(bus) -> dict[str, float]:
    return {f"{k}.pos": float(v) for k, v in bus.sync_read("Present_Position").items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--follower-port", required=True)
    ap.add_argument("--line-mode", action="store_true", help="type commands + Enter (non-Windows)")
    args = ap.parse_args(argv)

    out_path = data_dir() / "sim_real_map.json"
    preview = SimRealMap()
    poses_deg: list[np.ndarray] = []
    poses_grip: list[float] = []
    result = None
    model = ContactModel()
    twin = Twin(640, 480)
    rr.init("cta-align", spawn=True)
    print(HELP)
    bus = _connect(args.follower_port)
    print("connected; torque is OFF (arm is limp).")
    try:
        frame = 0
        while True:
            obs = _read_deg(bus)
            q = real_to_sim(obs, preview)
            if frame % 3 == 0:
                rr.log("twin", rr.Image(twin.render(q)))
                for i, j in enumerate(JOINTS):
                    rr.log(f"joints/{j}", rr.Scalars(np.degrees(q[i])))
                rr.log("contact_z_mm", rr.Scalars(1000.0 * model.lowest_z(q)))
            frame += 1
            cmds = (input("> ").strip()[:1] or " ",) if args.line_mode else _keys()
            for c in cmds:
                if c in (" ", "\r", "\n"):
                    deg = np.array([obs[f"{j}.pos"] for j in JOINTS[:5]])
                    poses_deg.append(deg)
                    poses_grip.append(float(real_to_sim(obs, SimRealMap())[5]))
                    print(f"captured #{len(poses_deg)}  contact z (current map) = {1000 * model.lowest_z(q):+.1f} mm")
                elif c == "u" and poses_deg:
                    poses_deg.pop()
                    poses_grip.pop()
                    print(f"removed last; {len(poses_deg)} poses")
                elif c == "s":
                    try:
                        result = solve_offsets(np.array(poses_deg), np.array(poses_grip), model)
                    except ValueError as e:
                        print(f"cannot solve: {e}")
                        continue
                    preview = SimRealMap(offset_deg=result.offset_deg)
                    print("offsets (deg) lift/elbow/wrist_flex:", np.round(result.offset_deg[1:4], 2).tolist())
                    print(f"table height z0 = {1000 * result.z0:+.1f} mm (base frame)")
                    print(f"contact error RMS: {1000 * result.rms_before:.1f} mm -> {1000 * result.rms_after:.1f} mm")
                    print("per-pose residual (mm):", np.round(1000 * result.residuals_m, 1).tolist())
                    if result.rms_after > 0.004:
                        print("WARNING: residual > 4 mm - a touch pose may be bad (undo + recapture) or the jaws pressed into the table.")
                    if np.any(np.abs(result.offset_deg) >= 14.9):
                        print("WARNING: an offset hit the 15 deg bound - recheck calibration before writing.")
                    print("preview applied to the twin; press 'w' to write it.")
                elif c == "w":
                    if result is None:
                        print("solve first ('s').")
                        continue
                    if out_path.exists():
                        shutil.copy2(out_path, out_path.with_suffix(".json.bak"))
                    preview.save(out_path)
                    print(f"wrote {out_path} (previous saved as .bak if it existed)")
                elif c == "q":
                    return 0
            if not args.line_mode:
                time.sleep(0.03)
    finally:
        bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    raise SystemExit(main())
