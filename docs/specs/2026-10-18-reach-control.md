# Reach Control Mode: Build Spec

*Tray approved this concept on 2026-10-18 ("perfect, let's get this built"). It extends `2026-10-17-arm-pose-control.md`, and every safety rule from that spec and the app applies unchanged.*

## Concept (Tray's words)

"I move my arm forward, the robot moves forward naturally with its own anatomy. I shouldn't have to make it mirror my arm. My arm doesn't look like the robot arm, so the joints don't match. It just needs to be able to get up and grab it."

- **The arm is the "where":** shoulder and elbow are watched only to see how far and which way the user is reaching.
- **The hand is the "what":** grip, turn, and tilt (wave) of the claw.

## Behaviour

- New `--control reach` becomes the **default**. `arm` (joint mirroring) and `hand` (image-space hand IK) stay available.
- **Reach vector, every frame, from Pose world landmarks:**
  - Build the body frame exactly as in arm mode: un-mirror, use the hidden-hips fallback, and pick the side by nearest hand wrist.
  - `r = wrist − shoulder` (the chosen side), expressed in body coordinates (fwd, left, up).
- **User arm length `L`:** a running median of |upper arm| + |forearm| over the last ~2 s of well-visible frames. It is initialized on the first good frame and updated slowly, so a bent arm doesn't shrink it.
- **Normalized reach:** `n = r / L`, clamped to |n| ≤ 1.
- **Robot gripper target (base frame, x fwd, y left, z up):** `target = ROBOT_SHOULDER + ROBOT_REACH * n`.
  - `ROBOT_SHOULDER` is the shoulder_lift joint anchor from the MuJoCo model (measure it; don't guess).
  - `ROBOT_REACH` is the measured max gripper distance from that anchor (fully extended), times 0.95.
  - Then clip to the reach-based workspace box (MapGains box; z floor per the table margin).
  - So: arm folded in → gripper near the base; fully extended forward → robot fully extended forward; reach down toward the table → robot reaches down to the table; reach to your left → robot's left.
- **Joints:** `Kinematics.ik(target, target_pitch, q_init=q_cmd, wrist_roll, gripper)`. The robot chooses its own shoulder and elbow; there is no joint copying.
  - `target_pitch` = `pitch_home + (hand_pitch − pitch0)`, as in hand mode (calibration `c`).
  - With no hand or calibration, `target_pitch` = a natural default: pitch that points the gripper ~45° down. Hold the previous wrist roll and gripper.
- **Roll and gripper:** exactly as in hand mode (wrapped roll delta; snap gripper; `c` open hand, `p` pinch).
- **Smoothing:** apply the arm smoother to the *target position* (3D, before IK) rather than the joints, so motion is smooth in space. Keep the per-frame rate limit and joint clip on the IK output.

## Safety (unchanged rules, apply all)

- Engagement: a body-only frame counts as TRACKING only if the hand was seen within 1.0 s.
- Visibility gating at `--pose-min-vis` for the chosen side's shoulder/elbow/wrist and both shoulders.
- COAST under 0.15 s, otherwise NO HAND hold.
- `clear_table` on the pipeline output.
- The sent-path guard, lead bound, speed limit and stall warning in the app.
- `--table-margin` 0.005.
- The IK target is always inside the workspace box.

## Interfaces

- `src/cta/reach.py` (pure):
  - `reach_vector(world, side, mirrored, hips_ok) -> (r_body (3,), L_frame)`
  - `class ArmLength` (running median; `.update(L_frame) -> L`)
  - `reach_target(n, robot_shoulder, robot_reach, box_min, box_max) -> (3,)`
- `src/cta/pipeline.py`: `ReachPipeline` on `_PipelineBase`, with the same public surface and `step(hand_obs, t, pose_obs)` signature as `ArmPosePipeline`. Share side choice, gating and engagement code with `ArmPosePipeline`; don't duplicate it.
- `src/cta/apps/mirror.py`: `--control {reach,arm,hand}`, default `reach`. Dashboard: draw the chosen shoulder→wrist reach line, and log `reach/n_x`, `n_y`, `n_z`. Recorder meta `control`.

## Tests (TDD)

- **Synthetic body, arm fully extended forward** → target ≈ ROBOT_SHOULDER + ROBOT_REACH·(1,0,0), clipped to the box. The gripper from `fk(ik(...))` lands within 1 cm of the clipped target.
- **Arm to the user's left** → target y > 0 (the robot's left). Reaching down-forward → target z decreases toward the floor, never below the table floor.
- **Arm-length invariance:** the same pose with 0.8× and 1.2× arm segment lengths gives the same target.
- **Mirrored image** → same target.
- **ArmLength** median ignores a single bent-arm frame.
- **ReachPipeline:** engagement, visibility gating, COAST/NO HAND, the snap gripper, no-hand default pitch, smoothing on target.
- **App:** `--control reach` runs with fakes; the existing modes are unchanged; all existing tests stay green.

## As built (2026-10-18)

- **Measured from the MuJoCo model** (`Kinematics.shoulder_anchor()` / `max_reach()`, checked in `tests/test_reach.py`):
  - `ROBOT_SHOULDER` = (0.0692, −0.0183, 0.1166) m, the shoulder_lift anchor at pan 0.
  - Max reach = 0.4113 m. This search includes wrist_roll, because the gripper site sits slightly off the roll axis. So `ROBOT_REACH` = 0.95 × 0.4113 = 0.391 m.
- **Box:** the MapGains box (`from_kinematics`), with its z floor replaced by `table_z + table_margin + 1 mm pad` (−0.0024 + 0.005 + 0.001 ≈ 3.6 mm). The box's lower x bound (~0.22 m) means "arm folded in" stops about 22 cm in front of the base. `clear_table` then guarantees the meshes.
- **No-hand pitch:** a brief hand dropout (≤ `COAST_S`, 0.15 s) keeps the last hand pitch, and only a longer absence (still within the 1 s engagement window) moves to ~45° down. Reason: MediaPipe hand flicker would otherwise swing the wrist every other frame. Without calibration the status is UNCALIBRATED (READY pose), as in every mode.
- **Smoothing:** the arm smoother runs on the 3-D target. The hand pitch has its own companion One Euro filter, shared with arm mode; roll is not smoothed.
- **Shared code:** `_BodyPipeline` (side choice, visibility gating, engagement, pitch filter, clearance hold) is the base of both `ArmPosePipeline` and `ReachPipeline`.
