"""cta-mirror: webcam arm + hand -> SO-101 twin / real arm.

--control arm (default, decision D-O): the user's shoulder/elbow (MediaPipe Pose) drive the robot's
shoulder and elbow joint for joint; the hand drives the wrist and gripper. --control hand: the hand alone
drives the gripper position through IK (the original mode)."""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

from cta.arm_angles import side_indices
from cta.camera import Camera, CameraMode, camera_check
from cta.dashboard import Dashboard
from cta.filters import OneEuro
from cta.hands import HandTracker
from cta.kinematics import Kinematics
from cta.mapping import MapGains, load_calibration, save_calibration
from cta.paths import data_dir, models_dir
from cta.pipeline import ArmPosePipeline, MirrorPipeline
from cta.pose import PoseTracker
from cta.recorder import STATUS_CODES, STATUS_UNKNOWN, SessionRecorder, git_commit, next_take_name
from cta.robot_io import FollowerArm, LeaderArm, clamp_margin, select_command
from cta.twin import Twin
from cta.units import SimRealMap


KEY_CHARS = "csrqp"
STALL_GAP_RAD = np.radians(10.0)  # target this far from the measured pose ...
STALL_MOVE_RAD = np.radians(0.2)  # ... while the arm moves less than this per frame ...
STALL_FRAMES = 45                 # ... for this many frames (~0.75 s) -> warn: arm not moving
MAX_LEAD_RAD = np.radians(12.0)   # command may lead the measured pose by at most this (LeRobot clamps at 15)
# live-tuning defaults (2026-10-17, first real-arm run lagged): Tray tunes these with the CLI flags
DEFAULT_SERVO_P = 24      # LeRobot uses 16 (soft, laggy); 32 = factory (stiff, may shake)
DEFAULT_GAIN = 1.5        # scales hand motion -> arm motion (1.0 = 30 cm of arm per frame-width of hand)
DEFAULT_SMOOTH_BETA = 0.05  # One Euro beta: higher = less lag when moving (LeRobot-agnostic; 0.007 was laggy)
MAX_CONSEC_BUS_FAILS = 15
WARN_EVERY_S = 1.0


def dispatch_keys(chars, events: set[str]) -> None:
    """Map typed characters to app events (pure; unit-tested without a keyboard)."""
    for ch in chars:
        if ch and ch.lower() in KEY_CHARS:
            events.add(ch.lower())


class _ConsoleKeys:
    """Windows console keys via msvcrt: polled once per loop, only active while this terminal has focus."""

    def __init__(self, events: set[str], kbhit=None, getwch=None):
        import msvcrt
        self._events = events
        self._kbhit = kbhit or msvcrt.kbhit
        self._getwch = getwch or msvcrt.getwch

    def poll(self) -> None:
        chars = []
        while self._kbhit():
            ch = self._getwch()
            if ch in ("\x00", "\xe0"):  # function/arrow key prefix: swallow the second code
                self._getwch()
                continue
            chars.append(ch)
        dispatch_keys(chars, self._events)

    def stop(self) -> None:
        pass


class _PynputKeys:
    # NOTE: pynput listens globally (non-Windows fallback): keys are captured even when another window has focus.
    def __init__(self, listener):
        self._lis = listener

    def poll(self) -> None:
        pass

    def stop(self) -> None:
        self._lis.stop()


def _start_keys(events: set[str]):
    if sys.platform == "win32":
        try:
            return _ConsoleKeys(events)
        except Exception as e:
            print(f"console keys unavailable ({e}); c/s/r/q keys disabled")
            return None
    try:
        from pynput import keyboard
    except Exception as e:  # headless / no backend
        print(f"keyboard control unavailable ({e}); c/s/r/q keys disabled")
        return None

    def on_press(key):
        dispatch_keys([getattr(key, "char", None)], events)

    lis = keyboard.Listener(on_press=on_press)
    lis.start()
    return _PynputKeys(lis)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cta-mirror")
    ap.add_argument("--robot", choices=["none", "so101"], default="none")
    ap.add_argument("--follower-port", default=None, help="follower serial port, e.g. COM3 (--robot so101)")
    ap.add_argument("--leader-port", default=None, help="leader serial port (required for --mode paired)")
    ap.add_argument("--control", choices=["arm", "hand"], default="arm",
                    help="arm (default): your shoulder/elbow drive the robot's, hand drives wrist+gripper; "
                         "hand: the hand alone drives the gripper position (IK)")
    ap.add_argument("--mode", choices=["mirror", "paired"], default="mirror",
                    help="mirror: camera drives the follower; paired: leader drives the follower, camera only logged")
    ap.add_argument("--max-speed-dps", type=float, default=90.0,
                    help="joint speed limit for commands sent to the arm, deg/s (default 90)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--mirrored", action="store_true", help="camera image is already mirrored")
    ap.add_argument("--servo-p", type=int, default=DEFAULT_SERVO_P,
                    help=f"arm servo stiffness 8-32 (default {DEFAULT_SERVO_P}; LeRobot uses 16)")
    ap.add_argument("--gain", type=float, default=DEFAULT_GAIN,
                    help=f"hand->arm motion scale (default {DEFAULT_GAIN})")
    ap.add_argument("--smooth-beta", type=float, default=DEFAULT_SMOOTH_BETA,
                    help=f"One Euro beta: higher = snappier, lower = smoother (default {DEFAULT_SMOOTH_BETA})")
    ap.add_argument("--calib", default=None, help="calibration JSON path")
    ap.add_argument("--record", default=None, metavar="NAME",
                    help="enable recording; 'r' key starts/stops a take saved as sessions/NAME_NN")
    ap.add_argument("--autostart-record", action="store_true", help="start the first take immediately (smoke runs)")
    ap.add_argument("--allow-identity-map", action="store_true",
                    help="--robot so101: run without sim_real_map.json (identity sign/offset; unsafe unless verified)")
    ap.add_argument("--max-frames", type=int, default=0, help="exit after N frames (smoke run)")
    args = ap.parse_args(argv)

    if args.autostart_record and not args.record:
        ap.error("--autostart-record requires --record NAME")
    if args.robot == "so101":
        if not args.follower_port:
            ap.error("--robot so101 requires --follower-port")
        if args.max_speed_dps <= 0:
            ap.error("--max-speed-dps must be > 0")
        if args.mode == "paired" and not args.leader_port:
            ap.error("--mode paired requires --leader-port")

    map_path = data_dir() / "sim_real_map.json"
    if args.robot == "so101" and not map_path.exists():
        if not args.allow_identity_map:
            ap.error(f"{map_path} is missing; the sim<->real joint map would silently be identity. "
                     "Run `uv run python -m cta.apps.sim_real_check` (or sync_data.py pull --what config), "
                     "or pass --allow-identity-map to override.")
        print("!! WARNING: running with an IDENTITY sim_real_map (--allow-identity-map)")

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
    tracker = pose_tracker = twin = None
    follower = leader = None
    keys = None
    rec: SessionRecorder | None = None
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
            follower.set_servo_p(args.servo_p)  # arm is resting: brief torque-off is harmless here
            print(f"follower connected on {args.follower_port} (mode: {args.mode}, servo P {args.servo_p})")
            if args.mode == "paired":
                leader = LeaderArm(args.leader_port)
                leader.connect()
                print(f"leader connected on {args.leader_port}")
        aspect = mode.width / mode.height if mode.height else 16 / 9

        kin = Kinematics()
        gains = MapGains.from_kinematics(kin)
        gains = dataclasses.replace(gains, k_lat=gains.k_lat * args.gain, k_vert=gains.k_vert * args.gain,
                                    k_fwd=gains.k_fwd * args.gain)
        smoother = OneEuro(beta=args.smooth_beta)
        if args.control == "arm":
            pipe = ArmPosePipeline(kin, cal, gains, smoother, aspect=aspect, mirrored=args.mirrored, want=args.hand)
        else:
            pipe = MirrorPipeline(kin, cal, gains, smoother, aspect=aspect)
        print(f"control: {args.control}")
        prev = None
        last_meas = None
        stalled_frames = 0
        if follower is not None:
            prev = np.asarray(follower.read(), dtype=float)  # raw pose, unclamped: ramp into the margin
            if prev.shape != (6,) or not np.all(np.isfinite(prev)):
                raise RuntimeError(f"follower pose is not finite ({prev}); check calibration / sim_real_map")
            pipe.reset_to(clamp_margin(prev))
            print(f"follower pose (deg): {np.round(np.degrees(prev), 1).tolist()}")
        warned_nonfinite = False
        last_cmd_t = None
        tracker = HandTracker(models_dir() / "hand_landmarker.task", want=args.hand,
                              mirrored_input=args.mirrored)
        if args.control == "arm":
            pose_tracker = PoseTracker(models_dir() / "pose_landmarker_full.task")
        twin = Twin()
        dash = Dashboard()
        events: set[str] = set()
        keys = _start_keys(events)
        print("keys (click this terminal first): c = calibrate (hand at home), s = toggle smoother, "
              + ("r = start/stop take, " if args.record else "") + "q = quit")
        sessions_root = data_dir() / "sessions"

        def start_take() -> SessionRecorder | None:
            if pipe.cal is None:
                if args.mode == "mirror":
                    print("REFUSED: cannot record a free-hand take while uncalibrated "
                          "(commands would be a constant pose). Press c with the hand at home first.")
                    return None
                print("!! WARNING: recording a paired take while UNCALIBRATED: leader labels are valid, "
                      "but pipeline commands are constant; use the per-row 'status' column to mask.")
            name = next_take_name(sessions_root, args.record)
            sim_map = SimRealMap.load()
            meta = {
                "mode": "paired" if args.mode == "paired" else "free",
                "robot": args.robot, "camera_mode": dataclasses.asdict(mode),
                "calibration": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(pipe.cal).items()}
                if pipe.cal is not None else None,
                "gains": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(gains).items()},
                "sim_real_map": {"sign": sim_map.sign.tolist(), "offset_deg": sim_map.offset_deg.tolist()},
                "lerobot_calibration": {
                    "follower": getattr(follower, "calibration_info", lambda: None)(),
                    "leader": getattr(leader, "calibration_info", lambda: None)()},
                "git_commit": git_commit(), "hand": args.hand, "mirrored": args.mirrored, "control": args.control,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            r = SessionRecorder(sessions_root, name, meta)
            print(f"REC START: {r.dir}")
            return r

        def stop_take() -> None:
            nonlocal rec
            r, rec = rec, None
            if r is None:
                return
            try:
                r.close()
            finally:
                print(f"REC STOP: {r.n} frames saved to {r.dir}")
                print("reminder: run `uv run python scripts/sync_data.py pull` before recording on another machine; "
                      "push (`... push --what sessions`) only BETWEEN takes, never mid-take.")

        if args.autostart_record:
            rec = start_take()
            if rec is None:
                print("ERROR: --autostart-record refused (see above)")
                return 1

        fps_ema = 0.0
        lat_ema = 0.0
        loop_ema = 0.0
        last_t = None
        last_ok = time.perf_counter()
        rc = 0
        n = 0
        consec_fail = 0
        last_warn = -1e9
        while True:
            rgb, t_frame = cam.read()
            if keys is not None:
                keys.poll()
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
            pose_obs = pose_tracker.detect(rgb, t_frame) if pose_tracker is not None else None
            if "c" in events:
                events.discard("c")
                if obs is None:
                    print("cannot calibrate: no hand visible")
                else:
                    try:
                        pipe.calibrate(obs)
                        save_calibration(pipe.cal, calib_path)
                        print(f"calibrated (open hand = {pipe.cal.pinch_open:.2f}); saved to {calib_path}. "
                              "Now pinch thumb+index and press 'p'.")
                    except ValueError as e:
                        print(f"cannot calibrate: {e}")
            if "p" in events:
                events.discard("p")
                if obs is None:
                    print("cannot set pinch: no hand visible")
                else:
                    try:
                        pipe.set_pinch_closed(obs)
                        save_calibration(pipe.cal, calib_path)
                        print(f"pinch set: closed={pipe.cal.pinch_closed:.2f} open={pipe.cal.pinch_open:.2f}; saved")
                    except ValueError as e:
                        print(f"cannot set pinch: {e}")
            if "s" in events:
                events.discard("s")
                if pipe.smoother is None:
                    pipe.set_smoother(OneEuro(beta=args.smooth_beta))
                else:
                    pipe.set_smoother(None)
                print(f"smoother {'ON' if pipe.smoother else 'OFF'}")
            if "r" in events:
                events.discard("r")
                if not args.record:
                    print("recording not enabled; start with --record NAME")
                elif rec is None:
                    rec = start_take()
                else:
                    stop_take()
            q, status = pipe.step(obs, t_frame, pose_obs) if args.control == "arm" else pipe.step(obs, t_frame)
            q_baseline = q.copy()
            q_leader = None
            q_foll = None
            extra = None
            bus_err: Exception | None = None
            if follower is not None:
                dt = 1 / 60 if last_cmd_t is None else t_frame - last_cmd_t
                last_cmd_t = t_frame
                skip_send = False
                if leader is not None:
                    try:
                        q_leader = leader.read()
                    except Exception as e:  # one serial hiccup must not end the take
                        bus_err, skip_send = e, True
                if q_leader is not None:
                    extra = {"leader": q_leader, "camera_cmd": q}  # camera q: logged, not sent
                # ramp from the LAST COMMAND (ramping from the measured pose crawls: each step is inside
                # the servo deadband), but never let the command run more than MAX_LEAD_RAD ahead of the
                # MEASURED pose - so a stalled arm (e.g. no 12 V motor power, 2026-10-17) can't be outrun
                if not skip_send:
                    try:
                        q_meas = np.asarray(follower.read(), dtype=float)
                        if q_meas.shape == (6,) and np.all(np.isfinite(q_meas)):
                            q_foll = q_meas
                    except Exception as e:  # serial hiccup: hold, send nothing new
                        bus_err, skip_send = e, True
                if q_foll is not None:
                    target = q_leader if args.mode == "paired" else (q if status in ("TRACKING", "COAST") else None)
                    moved = last_meas is not None and np.max(np.abs(q_foll - last_meas)) > STALL_MOVE_RAD
                    gap = 0.0 if target is None else float(np.max(np.abs(clamp_margin(target) - q_foll)))
                    stalled_frames = stalled_frames + 1 if (gap > STALL_GAP_RAD and not moved) else 0
                    last_meas = q_foll
                    if stalled_frames == STALL_FRAMES:
                        print("WARNING: arm is not moving toward its commands - check the follower's 12 V "
                              "motor power supply (motors only read on USB power)")
                if not skip_send:
                    q_new = select_command(args.mode, status, q, q_leader, prev, dt, max_speed)
                    if q_foll is not None:
                        q_new = np.clip(q_new, q_foll - MAX_LEAD_RAD, q_foll + MAX_LEAD_RAD)
                    try:
                        sent = follower.send(q_new)
                        prev = q_new
                        if sent is not None:
                            extra = {**(extra or {}), "sent": sent}
                    except ValueError as e:  # non-finite command: hold this frame
                        if not warned_nonfinite:
                            print(f"WARNING: command rejected, holding ({e})")
                            warned_nonfinite = True
                    except Exception as e:  # bus error: hold prev, send nothing new
                        bus_err = e
                q = prev
            if rec is not None:
                rec.add(t_frame, obs, pipe.last_features, q_baseline, q_leader, q_foll, rgb,
                        status=STATUS_CODES.get(status, STATUS_UNKNOWN),
                        smoother_on=int(pipe.smoother is not None),
                        pose_img=None if pose_obs is None else pose_obs.img,
                        pose_world=None if pose_obs is None else pose_obs.world)
            if bus_err is None:
                consec_fail = 0
            else:
                consec_fail += 1
                now_w = time.perf_counter()
                if now_w - last_warn >= WARN_EVERY_S:
                    last_warn = now_w
                    print(f"WARNING: arm bus error, holding ({consec_fail} in a row): {bus_err!r}")
                if consec_fail >= MAX_CONSEC_BUS_FAILS:
                    print(f"ERROR: {consec_fail} consecutive arm bus errors; exiting (arms will be closed)")
                    rc = 1
                    break
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
            arm_uv = None
            if heavy and pose_obs is not None and getattr(pipe, "side", None) is not None:
                arm_uv = pose_obs.img[list(side_indices(pipe.side, args.mirrored)), :2]  # shoulder, elbow, wrist
            dash.log_frame(rgb if heavy else None, obs if heavy else None, twin_rgb, q,
                           fps_ema, lat_ema, f"{status} REC" if rec is not None else status, loop_ema, extra_rad=extra,
                           arm_uv=arm_uv)
            loop_ms = (time.perf_counter() - t_frame) * 1000  # frame read -> end of iteration
            loop_ema = loop_ms if loop_ema == 0.0 else 0.9 * loop_ema + 0.1 * loop_ms
            if args.max_frames and n >= args.max_frames:
                print(f"done {n} frames: fps(EMA) {fps_ema:.1f}, latency(EMA) {lat_ema:.1f} ms, loop(EMA) {loop_ema:.1f} ms, status {status}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if rec is not None:  # stop_take is defined whenever rec can be set
            try:
                stop_take()
            except Exception as e:
                print(f"WARNING: recorder close failed: {e}")
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
        if pose_tracker is not None:
            pose_tracker.close()
        if twin is not None:
            twin.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
