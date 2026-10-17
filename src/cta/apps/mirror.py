"""cta-mirror: webcam hand -> SO-101 twin (and later the real arm)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cta.camera import Camera, CameraMode, camera_check
from cta.dashboard import Dashboard
from cta.filters import OneEuro
from cta.hands import HandTracker
from cta.kinematics import Kinematics
from cta.mapping import MapGains, load_calibration, save_calibration
from cta.paths import data_dir, models_dir
from cta.pipeline import MirrorPipeline
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
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--mirrored", action="store_true", help="camera image is already mirrored")
    ap.add_argument("--calib", default=None, help="calibration JSON path")
    ap.add_argument("--max-frames", type=int, default=0, help="exit after N frames (smoke run)")
    args = ap.parse_args(argv)

    if args.robot == "so101":
        print("SO-101 support arrives in Task 8")
        return 2

    calib_path = Path(args.calib) if args.calib else data_dir() / "calibration.json"
    cal = None
    if calib_path.exists():
        cal = load_calibration(calib_path)
        print(f"loaded calibration from {calib_path}")

    wanted = CameraMode(1920, 1080, 60)
    cam = Camera(args.camera, wanted)
    tracker = twin = None
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
        aspect = mode.width / mode.height if mode.height else 16 / 9

        kin = Kinematics()
        gains = MapGains.from_kinematics(kin)
        smoother = OneEuro()
        pipe = MirrorPipeline(kin, cal, gains, smoother, aspect=aspect)
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
                           fps_ema, lat_ema, status, loop_ema)
            loop_ms = (time.perf_counter() - t_frame) * 1000  # frame read -> end of iteration
            loop_ema = loop_ms if loop_ema == 0.0 else 0.9 * loop_ema + 0.1 * loop_ms
            if args.max_frames and n >= args.max_frames:
                print(f"done {n} frames: fps(EMA) {fps_ema:.1f}, latency(EMA) {lat_ema:.1f} ms, loop(EMA) {loop_ema:.1f} ms, status {status}")
                break
    except KeyboardInterrupt:
        pass
    finally:
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
