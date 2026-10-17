"""cta-mirror: webcam hand -> SO-101 twin (and later the real arm)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from cta.camera import Camera, CameraMode, camera_check
from cta.dashboard import Dashboard
from cta.filters import OneEuro
from cta.hands import HandTracker
from cta.kinematics import Kinematics
from cta.mapping import MapGains, load_calibration, save_calibration
from cta.paths import data_dir, models_dir
from cta.pipeline import MirrorPipeline
from cta.robot_io import FollowerArm, LeaderArm, clamp_margin, select_command
from cta.twin import Twin


def _start_keys(events: set[str]):
    # NOTE: pynput listens globally - keys are captured even when another window has focus.
    try:
        from pynput import keyboard
    except Exception as e:  # headless / no backend
        print(f"keyboard control unavailable ({e}); c/s/q keys disabled")
        return None

    def on_press(key):
        ch = getattr(key, "char", None)
        if ch and ch.lower() in "csq":
            events.add(ch.lower())

    lis = keyboard.Listener(on_press=on_press)
    lis.start()
    return lis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cta-mirror")
    ap.add_argument("--robot", choices=["none", "so101"], default="none")
    ap.add_argument("--follower-port", default=None, help="follower serial port, e.g. COM3 (--robot so101)")
    ap.add_argument("--leader-port", default=None, help="leader serial port (required for --mode paired)")
    ap.add_argument("--mode", choices=["mirror", "paired"], default="mirror",
                    help="mirror: camera drives the follower; paired: leader drives the follower, camera only logged")
    ap.add_argument("--max-speed-dps", type=float, default=90.0,
                    help="joint speed limit for commands sent to the arm, deg/s (default 90)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--mirrored", action="store_true", help="camera image is already mirrored")
    ap.add_argument("--calib", default=None, help="calibration JSON path")
    ap.add_argument("--max-frames", type=int, default=0, help="exit after N frames (smoke run)")
    args = ap.parse_args(argv)

    if args.robot == "so101":
        if not args.follower_port:
            ap.error("--robot so101 requires --follower-port")
        if args.max_speed_dps <= 0:
            ap.error("--max-speed-dps must be > 0")
        if args.mode == "paired" and not args.leader_port:
            ap.error("--mode paired requires --leader-port")

    if args.robot == "none" and args.mode == "paired":
        ap.error("--mode paired requires --robot so101")
    max_speed = float(np.radians(args.max_speed_dps))

    calib_path = Path(args.calib) if args.calib else data_dir() / "calibration.json"
    cal = None
    if calib_path.exists():
        cal = load_calibration(calib_path)
        print(f"loaded calibration from {calib_path}")

    wanted = CameraMode(1920, 1080, 60)
    cam = Camera(args.camera, wanted)
    tracker = twin = None
    follower = leader = None
    keys = None
    rc = 0
    try:
        if not cam.is_open():
            print(f"ERROR: camera {args.camera} did not open")
            return 1
        mode = cam.measure_mode()
        print(f"measured camera mode: {mode.width}x{mode.height} @ {mode.fps:.1f} fps")
        warns = camera_check(mode, wanted)
        if warns:
            bar = "!" * 70
            print(bar)
            for w in warns:
                print(f"!! CAMERA WARNING: {w}")
            print(bar)
        if args.robot == "so101":
            follower = FollowerArm(args.follower_port)
            follower.connect()
            print(f"follower connected on {args.follower_port} (mode: {args.mode})")
            if args.mode == "paired":
                leader = LeaderArm(args.leader_port)
                leader.connect()
                print(f"leader connected on {args.leader_port}")
        aspect = mode.width / mode.height if mode.height else 16 / 9

        kin = Kinematics()
        gains = MapGains.from_kinematics(kin)
        smoother = OneEuro()
        pipe = MirrorPipeline(kin, cal, gains, smoother, aspect=aspect)
        prev = None
        if follower is not None:
            prev = clamp_margin(follower.read())  # actual pose: start from reality, never from READY_Q
            pipe.reset_to(prev)
            print(f"follower pose (deg): {np.round(np.degrees(prev), 1).tolist()}")
        warned_nonfinite = False
        last_cmd_t = None
        tracker = HandTracker(models_dir() / "hand_landmarker.task", want=args.hand,
                              mirrored_input=args.mirrored)
        twin = Twin()
        dash = Dashboard()
        events: set[str] = set()
        keys = _start_keys(events)
        print("keys: c = calibrate (hand at home), s = toggle smoother, q = quit")

        fps_ema = 0.0
        lat_ema = 0.0
        loop_ema = 0.0
        last_t = None
        last_ok = time.perf_counter()
        rc = 0
        n = 0
        while True:
            rgb, t_frame = cam.read()
            if "q" in events:
                break
            if rgb is None:
                if time.perf_counter() - last_ok > 2.0:
                    print("ERROR: no camera frames for 2 s; exiting")
                    rc = 1
                    break
                time.sleep(0.005)
                continue
            last_ok = time.perf_counter()
            obs = tracker.detect(rgb, t_frame)
            if "c" in events:
                events.discard("c")
                if obs is None:
                    print("cannot calibrate: no hand visible")
                else:
                    pipe.calibrate(obs)
                    save_calibration(pipe.cal, calib_path)
                    print(f"calibrated; saved to {calib_path}")
            if "s" in events:
                events.discard("s")
                if pipe.smoother is None:
                    pipe.set_smoother(OneEuro())
                else:
                    pipe.set_smoother(None)
                print(f"smoother {'ON' if pipe.smoother else 'OFF'}")
            q, status = pipe.step(obs, t_frame)
            extra = None
            if follower is not None:
                dt = 1 / 60 if last_cmd_t is None else t_frame - last_cmd_t
                last_cmd_t = t_frame
                q_leader = leader.read() if leader is not None else None
                if q_leader is not None:
                    extra = {"leader": q_leader, "camera_cmd": q}  # camera q: logged, not sent
                q_new = select_command(args.mode, status, q, q_leader, prev, dt, max_speed)
                try:
                    follower.send(q_new)
                    prev = q_new
                except ValueError as e:  # non-finite command: hold this frame
                    if not warned_nonfinite:
                        print(f"WARNING: command rejected, holding ({e})")
                        warned_nonfinite = True
                q = prev
            latency_ms = (time.perf_counter() - t_frame) * 1000
            now = time.perf_counter()
            if last_t is not None and now > last_t:
                inst = 1.0 / (now - last_t)
                fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
            lat_ema = latency_ms if lat_ema == 0.0 else 0.9 * lat_ema + 0.1 * latency_ms
            last_t = now
            n += 1
            heavy = n % 2 == 0  # twin render + images at most every 2nd frame
            twin_rgb = twin.render(q) if heavy else None
            dash.log_frame(rgb if heavy else None, obs if heavy else None, twin_rgb, q,
                           fps_ema, lat_ema, status, loop_ema, extra_rad=extra)
            loop_ms = (time.perf_counter() - t_frame) * 1000  # frame read -> end of iteration
            loop_ema = loop_ms if loop_ema == 0.0 else 0.9 * loop_ema + 0.1 * loop_ms
            if args.max_frames and n >= args.max_frames:
                print(f"done {n} frames: fps(EMA) {fps_ema:.1f}, latency(EMA) {lat_ema:.1f} ms, loop(EMA) {loop_ema:.1f} ms, status {status}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        for arm in (follower, leader):  # disconnect disables torque; never skip the rest
            if arm is not None:
                try:
                    arm.close()
                except Exception as e:
                    print(f"WARNING: arm disconnect failed: {e}")
        if keys is not None:
            keys.stop()
        cam.close()
        if tracker is not None:
            tracker.close()
        if twin is not None:
            twin.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
