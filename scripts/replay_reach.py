"""Replay recorded reach sessions through the OLD and NEW ReachPipeline and report target stability.

Proof for docs/specs/2026-10-18-reach-measurement-fix.md. Per take:
  - side switches after 'c' (OLD: the arm side it picks per frame, nearest Pose wrist to the hand-tracker
    wrist; NEW: the locked side),
  - per-frame |delta target| p50 / p99 / max (cm) between consecutive TRACKING frames, at three stages:
    raw (before smoothing), smoothed (One Euro output), final (after the 0.25 m/s target speed limit),
  - frames with a jump > 3 cm (each stage); jumps across a non-TRACKING gap (re-acquisition) separately,
  - jitter (second-difference noise), which separates measurement noise from real motion,
  - held fraction: frames where hand AND pose were both present but the pipeline did not track.

Replay conventions (no 'c' press is recorded): 'c' = the first frame with both the hand and the pose present
(the arm lock and the reach home are taken there); the hand calibration (pitch/roll/pinch, p0) is the recorded
live one from meta.json (recomputing it fails on takes whose first frame is a pinch); relative ("home") mapping as in the CLI,
smoother = the CLI default One Euro (beta DEFAULT_SMOOTH_BETA). Depth: human.mp4 frame i <-> row i, through
the real Depth Anything V2 model (GPU fp16).

OLD = the ReachPipeline at --old-rev (git show; default: the spec commit), loaded with its own reach.py.

  uv run python scripts/replay_reach.py                  # reach_train_01..04, NEW wrist + palm
  uv run python scripts/replay_reach.py --depth none     # NEW with the hand-scale fallback
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

from cta.apps.mirror import DEFAULT_SMOOTH_BETA
from cta.arm_angles import choose_arm_side
from cta.depth import DepthEstimator
from cta.filters import OneEuro
from cta.hands import HandObservation
from cta.kinematics import Kinematics
from cta import pipeline as pipeline_mod
from cta.mapping import Calibration, MapGains
from cta.paths import data_dir
from cta.pipeline import ReachPipeline, clearance_model
from cta.pose import PoseObservation
from cta.recorder import load_session

REPO = Path(__file__).resolve().parents[1]
OLD_REV = "27488e3"  # the spec commit: the pipeline with per-frame side choice + world-landmark reach
JUMP_M = 0.03


def load_old_pipeline(rev: str) -> types.ModuleType:
    """The OLD pipeline module from git, with its own (old) reach.py, without touching the working tree."""
    def show(path):
        return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO, capture_output=True, text=True,
                              check=True, encoding="utf-8").stdout
    reach = types.ModuleType("cta_old_reach")
    exec(compile(show("src/cta/reach.py"), f"{rev}:reach.py", "exec"), reach.__dict__)
    sys.modules["cta_old_reach"] = reach
    src = show("src/cta/pipeline.py").replace("from cta.reach import", "from cta_old_reach import")
    mod = types.ModuleType("cta_old_pipeline")
    sys.modules["cta_old_pipeline"] = mod  # dataclasses look their module up here
    exec(compile(src, f"{rev}:pipeline.py", "exec"), mod.__dict__)
    return mod


class RecSmoother:
    """OneEuro wrapper that remembers its last output (the smoothed target before the speed limit)."""

    def __init__(self, inner: OneEuro):
        self.inner = inner
        self.min_cutoff, self.beta, self.d_cutoff = inner.min_cutoff, inner.beta, inner.d_cutoff
        self.last = None

    def reset(self):
        self.inner.reset()

    def __call__(self, x, t):
        self.last = self.inner(x, t)
        return self.last


def obs_at(d, i, hand):
    h = HandObservation(d["lm_img"][i].astype(float), d["lm_world"][i].astype(float), hand, 0.9) \
        if bool(d["hand_present"][i]) else None
    img = d["pose_img"][i].astype(float)
    if not np.all(np.isfinite(img[:, :2])):
        return h, None
    vis = d["pose_vis"][i].astype(float)
    p = PoseObservation(img, d["pose_world"][i].astype(float), float(np.nanmean(vis[11:17])), vis,
                        d["pose_presence"][i].astype(float))
    return h, p


def stats(targets: list) -> dict:
    """targets: per frame, the target (m) on TRACKING frames else None. Reports, in cm:
    - consecutive: |delta| between TRACKING frames i-1 and i (p50 / p99 / max, count > 3 cm),
    - re-acquisition: |delta| from the last TRACKING frame to the first one after a gap (max, count > 3 cm),
    - jitter: |second difference| / sqrt(6) over three consecutive TRACKING frames (noise s -> ~s; smooth
      motion -> ~0), p50 / p99."""
    cons, gap, jit, last = [], [], [], None
    for i, x in enumerate(targets):
        if x is None:
            continue
        if last is not None:
            dd = float(np.linalg.norm(x - targets[last])) * 100
            (cons if last == i - 1 else gap).append(dd)
        if i >= 2 and targets[i - 1] is not None and targets[i - 2] is not None:
            jit.append(float(np.linalg.norm(x - 2 * targets[i - 1] + targets[i - 2])) * 100 / np.sqrt(6))
        last = i

    def pct(v, q):
        return float(np.percentile(v, q)) if v else float("nan")
    return {"p50": pct(cons, 50), "p99": pct(cons, 99), "max": max(cons, default=float("nan")),
            "n_gt3": int(sum(v > JUMP_M * 100 for v in cons)), "n_pairs": len(cons),
            "reacq_n": len(gap), "reacq_max": max(gap, default=float("nan")),
            "reacq_gt3": int(sum(v > JUMP_M * 100 for v in gap)),
            "jit50": pct(jit, 50), "jit99": pct(jit, 99)}


class Runner:
    def __init__(self, name, pipe, kind):
        self.name, self.pipe, self.kind = name, pipe, kind
        self.status, self.raw, self.smooth, self.final, self.sides, self.reasons = [], [], [], [], [], {}
        self.n_fwd = []

    def step(self, obs, t, pose_obs, dmap):
        p = self.pipe
        if self.kind == "old":
            if pose_obs is not None and obs is not None:  # the side OLD picks this frame (per-frame choice)
                self.sides.append(choose_arm_side(pose_obs.img, obs.lm_img[0, :2], p.side, p.want, p.mirrored))
            _, s = p.step(obs, t, pose_obs)
        else:
            _, s = p.step(obs, t, pose_obs, dmap)
            self.sides.append(p.locked_side)
        tracking = s == "TRACKING"
        self.status.append(s)
        r = getattr(p, "tracking_hold_reason", None)
        if r:
            self.reasons[r] = self.reasons.get(r, 0) + 1
        self.raw.append(p.raw_target.copy() if tracking else None)
        self.smooth.append(np.clip(p.smoother.last, p.box_min, p.box_max) if tracking else None)
        self.final.append(p.target.copy() if tracking else None)
        self.n_fwd.append(p.last_n[0] if tracking and p.last_n is not None else np.nan)

    def summary(self, both):
        st = np.array(self.status)
        sides = [s for s in self.sides if s is not None]
        switches = int(sum(a != b for a, b in zip(sides, sides[1:])))
        held = float(np.mean(st[both] != "TRACKING")) if both.any() else float("nan")
        return {"pipeline": self.name, "side_switches": switches, "tracking_frac": float(np.mean(st == "TRACKING")),
                "held_frac_both_present": held, "raw": stats(self.raw), "smoothed": stats(self.smooth),
                "final": stats(self.final), "hold_reasons": self.reasons}


def replay(take: Path, old_mod, est, kin, gains, contact, hand_points, fwd_source) -> tuple[list, dict]:
    d = load_session(take)
    meta = json.loads((take / "meta.json").read_text())
    cm = meta["camera_mode"]
    aspect = cm["width"] / cm["height"]
    mirrored, hand = bool(meta["mirrored"]), meta["hand"]
    n = len(d["t"])
    both = d["hand_present"] & np.all(np.isfinite(d["pose_img"][:, :, 0]), axis=1)
    c = int(np.argmax(both))
    rc = dict(meta["calibration"])
    rec_cal = Calibration(**{**rc, "p0": np.asarray(rc["p0"], dtype=float)})

    def mk(cls, **kw):
        return cls(kin, None, gains, RecSmoother(OneEuro(beta=DEFAULT_SMOOTH_BETA)), aspect=aspect,
                   mirrored=mirrored, want=hand, contact=contact, home_relative=True, **kw)

    runners = [Runner("OLD", mk(old_mod.ReachPipeline), "old")]
    for hp in hand_points:
        runners.append(Runner(f"NEW-{hp}", mk(ReachPipeline, hand_point=hp, fwd_source=fwd_source), "new"))
    cap = cv2.VideoCapture(str(take / "human.mp4"))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    depth_ms, d_hand, d_sh = [], [], []
    for i in range(n):
        ok, bgr = cap.read()
        dmap = None
        if est is not None and ok and i >= c:
            dmap = est.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            depth_ms.append(est.last_ms)
        if i < c:
            continue
        obs, pose_obs = obs_at(d, i, hand)
        t = float(d["t"][i])
        for r in runners:
            if i == c:
                mod = old_mod if r.kind == "old" else pipeline_mod
                real, mod.calibrate = mod.calibrate, lambda f, k: rec_cal  # the recorded live hand calibration
                try:
                    r.pipe.calibrate(obs) if r.kind == "old" else r.pipe.calibrate(obs, pose_obs)
                finally:
                    mod.calibrate = real
            r.step(obs, t, pose_obs, dmap)
        p = runners[1].pipe
        d_hand.append(p.depth_hand)
        d_sh.append(p.depth_shoulder)
    cap.release()
    bw = both[c:]
    out = []
    for r in runners:
        s = r.summary(bw)
        s.update(take=take.name, frames=n - c, c_frame=c)
        out.append(s)
    info = {"take": take.name, "rows": n, "video_frames": n_video, "c_frame": c, "aspect": aspect,
            "mirrored": mirrored, "hand": hand, "fps": float(1 / np.median(np.diff(d["t"]))),
            "depth_ms_median": float(np.median(depth_ms)) if depth_ms else float("nan"),
            "hand_present_frac": float(d["hand_present"][c:].mean()), "both_present_frac": float(bw.mean()),
            "depth_hand_median": float(np.nanmedian(d_hand)) if np.isfinite(d_hand).any() else float("nan"),
            "depth_shoulder_median": float(np.nanmedian(d_sh)) if np.isfinite(d_sh).any() else float("nan")}
    return out, info


def fmt_table(rows: list[dict]) -> str:
    """Summary (final = the target IK aims at) + detail (raw / smoothed stages)."""
    out = ["Summary - final target (after One Euro + the 0.25 m/s target speed limit), consecutive TRACKING frames:",
           "", "| take | pipeline | side switches | tracking | held (hand+pose present) | final p50 / p99 / max cm "
           "| final >3 cm | final jitter p50 / p99 cm | re-acq. jumps (max cm) |", "|" + "---|" * 9]
    for r in rows:
        f = r["final"]
        out.append(f"| {r['take']} | {r['pipeline']} | {r['side_switches']} | {r['tracking_frac']:.1%} | "
                   f"{r['held_frac_both_present']:.1%} | {f['p50']:.2f} / {f['p99']:.2f} / {f['max']:.2f} | "
                   f"{f['n_gt3']} | {f['jit50']:.2f} / {f['jit99']:.2f} | {f['reacq_n']} ({f['reacq_max']:.1f}) |")
    out += ["", "Detail - raw (before smoothing) and smoothed (One Euro output, before the speed limit):", "",
            "| take | pipeline | raw p50 / p99 / max cm | raw >3 cm | raw jitter p50 / p99 | raw re-acq. max / >3 cm "
            "| smoothed p50 / p99 / max cm | smoothed >3 cm | smoothed jitter p50 / p99 | smoothed re-acq. max / >3 cm |",
            "|" + "---|" * 10]
    for r in rows:
        cells = []
        for k in ("raw", "smoothed"):
            x = r[k]
            cells += [f"{x['p50']:.2f} / {x['p99']:.2f} / {x['max']:.2f}", str(x["n_gt3"]),
                      f"{x['jit50']:.2f} / {x['jit99']:.2f}", f"{x['reacq_max']:.1f} / {x['reacq_gt3']}"]
        out.append(f"| {r['take']} | {r['pipeline']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="*", default=[f"reach_train_0{i}" for i in range(1, 5)])
    ap.add_argument("--root", default=str(data_dir() / "sessions"))
    ap.add_argument("--old-rev", default=OLD_REV)
    ap.add_argument("--depth", choices=["model", "none"], default="model")
    ap.add_argument("--depth-input-h", type=int, default=None)
    ap.add_argument("--hand-points", default="wrist,palm")
    ap.add_argument("--json", default=None, help="write all results here")
    args = ap.parse_args(argv)
    old_mod = load_old_pipeline(args.old_rev)
    est = None
    if args.depth == "model":
        est = DepthEstimator(**({"input_height": args.depth_input_h} if args.depth_input_h else {}))
        if not est.load():
            print(f"ERROR: depth model unavailable ({est.error}); use --depth none")
            return 1
        print(f"depth: {est.model_id} on {est.device} ({est._dtype}), input height {est.input_height}")
    kin = Kinematics()
    gains = MapGains.from_kinematics(kin)
    contact = clearance_model()
    rows, infos = [], []
    t0 = time.perf_counter()
    for take in args.takes:
        r, info = replay(Path(args.root) / take, old_mod, est, kin, gains, contact,
                         args.hand_points.split(","), "depth" if est is not None else "scale")
        rows += r
        infos.append(info)
        print(json.dumps(info))
        for x in r:
            print(f"  {x['pipeline']}: hold reasons {x['hold_reasons']}")
    print(fmt_table(rows))
    print(f"replay took {time.perf_counter() - t0:.0f} s")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "takes": infos, "old_rev": args.old_rev,
                                               "depth": args.depth}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
