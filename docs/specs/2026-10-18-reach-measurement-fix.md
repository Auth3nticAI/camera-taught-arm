# Reach Measurement Fix: Build Spec (root cause from recorded data)

*Tray approved this on 2026-10-18: "pull the models and fix the problems". Branch `codex/webcam-learning`. The concept stays as Tray approved it: the arm says *where*, the robot reaches with its own IK, and the hand is the claw. This changes only **how "where" is measured**. Every safety layer stays.*

## Root cause (measured on `cta-data/sessions/reach_train_01..04`, ~4,600 frames)

1. **The tracked arm keeps switching.** Arm side is re-picked every frame (pose wrist nearest the hand-tracker wrist). That gives 33–131 side switches per take, and the switches cause most frame-to-frame reach jumps above 15 cm (e.g. 25/41, 36/57, 25/36, 59/62).
2. **The Pose wrist teleports.** On same-side big-jump frames, the Pose wrist moved a median **0.14–0.15 image widths in one frame**, while the hand-tracker wrist for the same hand moved **0.016–0.05**.
3. **Pose depth is noisy:** about ±11–17 cm on the wrist−shoulder z.
4. **MediaPipe reports visibility of about 1.0 throughout**, so visibility gating can't catch any of this.

## Fix

### 1. Lock the arm at calibration
- When `c` is pressed, record which arm is in use: the Pose side whose wrist (image) is nearest the hand-tracker wrist at that moment, using the existing un-mirroring.
- **Never switch sides afterwards** until the next `c`.
- If the locked side's shoulder is not visible (below `--pose-min-vis`) or both shoulders collapse/teleport (keep the existing shoulder-sanity checks), treat the frame as lost (COAST/NO HAND).

### 2. "Where" from the hand tracker, anchored on the Pose shoulder
- **Hand position:** the hand tracker's wrist `lm_img[0]`, the stable source, in image coordinates. Optionally use the palm centre (mean of landmarks 0, 5, 9, 13, 17) if it measures steadier in replay; pick whichever replays better and say so in the report.
- **Anchor:** the **locked side's Pose shoulder** (image). Scale = Pose shoulder separation in the image (aspect-corrected), smoothed with a short median (~0.5 s).
- **Normalized in-plane reach**, in user/body terms (left = user's left, up = up), correct for both mirrored and unmirrored images:
  `n_left`, `n_up` = (hand − shoulder) / (ARM_PER_SHOULDER × shoulder_sep), where ARM_PER_SHOULDER ≈ 1.7 (human arm length ÷ shoulder width). Unit-test with synthetic frames in both mirror modes.
- **Forward (toward the camera = the user's forward), from Depth Anything V2:**
  - **Model:** `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` via `transformers` (already a dependency). Run on GPU when available (fp16 is fine) at a modest input size. Measure ms per frame on this desktop (RTX 4070 Ti) and report it.
  - Sample robust depth (median over a small patch) at the hand (palm centre) and at the locked shoulder. `d = depth_hand − depth_shoulder` in metres.
  - `n_fwd = −(d − d_home) / (ARM_PER_SHOULDER × shoulder_width_m)` with `shoulder_width_m` = Pose **world** shoulder distance (median-smoothed).
  - If the model is unavailable (no GPU, download failure) or `--depth none`: fall back to the hand-scale cue, i.e. hand scale ÷ shoulder separation relative to its calibration value. Tested.
  - The depth model may run every frame or every 2nd frame, with the last value held between runs. Pick based on measured fps; the target is ≥ 20 fps overall on the desktop.
- **Reference:** keep the CLI default `--reach-mapping relative`. At `c`, record the home values (`n_*_home`, `d_home`, scale). The target is `p0 + ROBOT_REACH × (n − n_home)`, clipped to the workspace box, with the existing smoothing on the target and the existing IK, Cartesian clearance, and all guards. `--reach-mapping absolute` keeps working with the new measurements.
- **Pose's wrist/elbow world landmarks are no longer used for "where".** They may still feed the existing sanity checks.

### 3. Keep
- The engagement rule and COAST/NO HAND semantics.
- Snap gripper, roll, pitch.
- clear_table, the sent-path guard, lead bound, speed limit, stall warning, table margin.
- The recorder: add per-row `depth_hand`, `depth_shoulder`, `locked_side` (NaN or −1 when absent).
- The dashboard: show the hand→shoulder line and `reach/n_*`.

## Proof (required, not optional)

- **`scripts/replay_reach.py`** replays recorded sessions (hand + pose landmarks + `human.mp4` frames for depth) through the OLD and NEW ReachPipeline and reports, per take:
  - side switches;
  - per-frame |Δtarget| p50/p99/max on TRACKING frames;
  - the fraction of frames held;
  - the number of frames with a target jump > 3 cm.
- Run it on all four takes and put the before/after table in the report.
- **Acceptance:** zero side switches after `c`, and per-frame target jumps p99 ≤ 1.5 cm and max ≤ 4 cm on TRACKING frames (the smoothing and rate limit allowed). If that isn't met, report the numbers honestly with the residual cause. **Don't loosen the acceptance or add holds to hide it.**
- Unit tests (TDD): side lock; mirrored/unmirrored direction; depth fallback; patch sampling robust to NaN or edges; the home-relative target.
- The full suite stays green.
