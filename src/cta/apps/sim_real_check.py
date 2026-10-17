"""cta-sim-real-check: twin driven by real_to_sim(leader); interactively fix per-joint sign / offset."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from cta.config import JOINTS
from cta.dashboard import Dashboard
from cta.robot_io import LeaderArm, offset_from_reference
from cta.twin import Twin
from cta.units import SimRealMap, real_to_sim


def _prompt(map_: SimRealMap, leader: LeaderArm) -> None:
    """Thin stdin loop: 'f N' flip sign of joint N (1-5), 'o N DEG' set offset, 'z N RAD' offset so the
    leader's current reading of joint N maps to sim angle RAD, 'p' print, 's' save+quit, 'q' quit."""
    print("commands: f N | o N DEG | z N SIM_RAD | p | s (save) | q (quit w/o save); N = 1..5")
    while True:
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
                raw = leader.read_raw()[f"{JOINTS[i]}.pos"]
                map_.offset_deg[i] = offset_from_reference(raw, float(parts[2]), map_.sign[i])
            else:
                print("unknown command")
                continue
            print(f"{JOINTS[i]}: sign {map_.sign[i]:+.0f} offset {map_.offset_deg[i]:.1f} deg")
        except (IndexError, ValueError) as e:
            print(f"bad input: {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cta-sim-real-check")
    ap.add_argument("--leader-port", required=True)
    args = ap.parse_args(argv)

    leader = LeaderArm(args.leader_port)
    twin = dash = None
    try:
        leader.connect()
        twin, dash = Twin(), Dashboard("cta-sim-real")
        import threading
        stop = threading.Event()

        def pump():
            while not stop.is_set():
                q = leader.read()  # leader.map is mutated live by the prompt thread
                dash.log_frame(None, None, twin.render(q), q, 0.0, 0.0, "SIM-REAL CHECK")
                time.sleep(0.03)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        _prompt(leader.map, leader)
        stop.set()
        t.join(timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        leader.close()
        if twin is not None:
            twin.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
