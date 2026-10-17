"""cta-sim-real-check: twin driven by real_to_sim(leader); interactively fix per-joint sign / offset."""
from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

from cta.config import JOINTS
from cta.dashboard import Dashboard
from cta.robot_io import LeaderArm, offset_from_reference
from cta.twin import Twin
from cta.units import SimRealMap, real_to_sim


class _Shared:
    """Latest leader reading, written by the main (render) thread only; read by the prompt thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.raw: dict[str, float] | None = None
        self.stop = threading.Event()
        self.saved = False


def _prompt(map_: SimRealMap, shared: _Shared) -> None:
    """Thin stdin loop (background thread; never touches the serial bus or the renderer).
    'f N' flip sign of joint N (1-5), 'o N DEG' set offset, 'z N RAD' set offset so the leader's current
    reading of joint N maps to sim angle RAD, 'p' print, 's' save+quit, 'q' quit w/o save.
    NOTE: after a sign flip the joint's offset is stale - re-zero it with 'z' (or 'o')."""
    print("commands: f N | o N DEG | z N SIM_RAD | p | s (save) | q (quit w/o save); N = 1..5")
    print("NOTE: after 'f' re-zero that joint with 'z N SIM_RAD' (leader at a known pose) or 'o N DEG'.")
    try:
        while not shared.stop.is_set():
            try:
                parts = input("> ").split()
            except EOFError:
                return
            if not parts:
                continue
            try:
                c = parts[0]
                if c == "q":
                    return
                if c == "s":
                    map_.save()
                    shared.saved = True
                    print("saved sim_real_map.json")
                    return
                if c == "p":
                    print("sign", map_.sign, "offset_deg", map_.offset_deg)
                    continue
                i = int(parts[1]) - 1
                if not 0 <= i < 5:
                    print("N must be 1..5")
                    continue
                if c == "f":
                    map_.sign[i] *= -1
                elif c == "o":
                    map_.offset_deg[i] = float(parts[2])
                elif c == "z":
                    with shared.lock:
                        raw = None if shared.raw is None else shared.raw[f"{JOINTS[i]}.pos"]
                    if raw is None:
                        print("no leader reading yet")
                        continue
                    map_.offset_deg[i] = offset_from_reference(raw, float(parts[2]), map_.sign[i])
                else:
                    print("unknown command")
                    continue
                print(f"{JOINTS[i]}: sign {map_.sign[i]:+.0f} offset {map_.offset_deg[i]:.1f} deg")
            except (IndexError, ValueError) as e:
                print(f"bad input: {e}")
    finally:
        shared.stop.set()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cta-sim-real-check")
    ap.add_argument("--leader-port", required=True)
    args = ap.parse_args(argv)

    leader = LeaderArm(args.leader_port)
    twin = dash = None
    shared = _Shared()
    rc = 0
    try:
        leader.connect()
        twin, dash = Twin(), Dashboard("cta-sim-real")  # created and used on THIS thread (GL context)
        threading.Thread(target=_prompt, args=(leader.map, shared), daemon=True).start()
        while not shared.stop.is_set():
            raw = leader.read_raw()  # only this thread touches the serial bus
            with shared.lock:
                shared.raw = raw
            q = real_to_sim(raw, leader.map)
            dash.log_frame(None, None, twin.render(q), q, 0.0, 0.0, "SIM-REAL CHECK")
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"ERROR in render loop: {e!r}")
        rc = 1
    finally:
        shared.stop.set()
        if not shared.saved and (not np.array_equal(leader.map.sign, np.ones(5)) or np.any(leader.map.offset_deg)):
            # edits are never lost; the 's' command already saved the real file if requested
            from cta.units import _default_path
            part = _default_path().with_suffix(".partial.json")
            try:
                leader.map.save(part)
                print(f"unsaved edits kept in {part} (rename to sim_real_map.json to use)")
            except Exception as e:
                print(f"could not write partial map: {e}")
        leader.close()
        if twin is not None:
            twin.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
