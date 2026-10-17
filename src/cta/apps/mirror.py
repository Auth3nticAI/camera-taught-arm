"""cta-mirror: webcam hand -> SO-101 twin (and later the real arm)."""
from __future__ import annotations

import argparse
import sys
import time

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

    from pathlib import Path
    calib_path = Path(args.calib) if args.calib else data_dir() / "calibration.json"
    cal = None
    if calib_path.exists():
        cal = load_calibration(calib_path)
        print(f"loaded calibration from {calib_path}")

    wanted = CameraMode(1920, 1080, 60)
    cam = Camera(args.camera, wanted)
    tracker = twin = None
    keys = None
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
        last_t = None
        n = 0
        while True:
            rgb, t_frame = cam.read()
            if rgb is None:
                continue
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
            if "q" in events:
                break
            q, status = pipe.step(obs, t_frame)
            latency_ms = (time.perf_counter() - t_frame) * 1000
            twin_rgb = twin.render(q)
            now = time.perf_counter()
            if last_t is not None and now > last_t:
                inst = 1.0 / (now - last_t)
                fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
            lat_ema = latency_ms if lat_ema == 0.0 else 0.9 * lat_ema + 0.1 * latency_ms
            last_t = now
            dash.log_frame(rgb, obs, twin_rgb, q, fps_ema, lat_ema, status)
            n += 1
            if args.max_frames and n >= args.max_frames:
                print(f"done {n} frames: fps(EMA) {fps_ema:.1f}, latency(EMA) {lat_ema:.1f} ms, status {status}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
