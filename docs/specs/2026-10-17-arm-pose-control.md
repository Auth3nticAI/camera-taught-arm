# Arm-Pose Control Mode: Build Spec

*Decision D-O (Tray, 2026-10-17). It extends the Takeworthy design doc (`2026-10-16-camera-taught-arm-design.md`) without replacing anything. All existing safety rules stay in force.*

## Goal

Add a control mode where the user's own **shoulder and elbow** drive the robot's shoulder_pan, shoulder_lift and elbow_flex **joint for joint**, using MediaPipe Pose. The hand tracker keeps driving wrist_flex, wrist_roll and the snap gripper.

- New flag: `cta-mirror --control {arm,hand}`, default **`arm`**.
- `--control hand` is today's behaviour, unchanged.

## Inputs

- **MediaPipe PoseLandmarker** (Tasks API, mediapipe 1.1.1, `RunningMode.VIDEO`, `num_poses=1`).
  - Model: `pose_landmarker_full.task` from `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`, downloaded into `models_dir()`. Add it to `scripts/download_models.py`.
  - Use **world landmarks** (meters, hip-centred, camera-aligned axes: x right, y down, z away from the camera) for angles, and image landmarks for side selection.
  - Indices: shoulders 11 (L) / 12 (R), elbows 13 / 14, wrists 15 / 16, hips 23 / 24.
- **Hand tracker:** the existing `HandTracker` and `compute_features`, unchanged.

## Which arm

- **Pick the pose side whose wrist (image coords) is nearest the tracked hand's wrist** (`HandObservation.lm_img[0]`). Don't trust Pose's left/right labels: they swap on mirrored images.
- If no hand is visible, keep the side used last. With no history, use the side that matches `--hand` (with the mirrored flip applied the same way as `hands._actual_side`).

## Body frame (per frame, from world landmarks)

- `up` = normalize(mid_shoulder − mid_hip).
- `left` = the user's left: normalize(L_shoulder − R_shoulder), orthogonalized against `up`.
- `fwd` = out of the user's chest: normalize(cross(left, up)). Flip the sign if needed so that `fwd · (0, 0, −1) > 0` (the chest faces the camera, and camera z points away).
- If the image was mirrored (`--mirrored`), MediaPipe's L/R world landmarks are the anatomical mirror image. Correct this so that `left` always points to the user's true left. Unit-test with a synthetic mirrored body.

## Joint angles (the user's arm → robot joints)

`a` = elbow − shoulder (upper arm); `b` = wrist − elbow (forearm).

**Robot shoulder_lift** = elevation of the upper arm from vertical:
- `lift = arccos(a·up / |a|)`, so 0 = arm straight up and +90° = horizontal.
- Robot convention (verified): `shoulder_lift` 0 = upper arm vertical, +90° = upper arm forward horizontal.
- Clamp to joint limits via the existing margin clamp.

**Robot shoulder_pan** = azimuth of the upper arm toward the user's left:
- `az = atan2(a·left, a·fwd)`, so 0 = forward and +90° = to the user's left.
- The robot faces the same way as the user, and **+pan moves the robot's gripper toward −y (its right)**. Therefore `pan = −az`.
- When the horizontal component is small (`|a_h| < 0.35 |a|`, i.e. the arm is near vertical, up or down), azimuth is undefined: **hold the previous pan**.

**Robot elbow_flex:**
- Human flexion = angle between `a` and `b` (0 = straight, 180 = fully folded).
- Robot elbow 0 = forearm perpendicular to the upper arm (verified: lift 0 + elbow 0 = upper arm up, forearm forward; elbow +90 = folded down; elbow −90 = straight).
- Therefore `elbow = flexion − 90°`.

**Robot wrist_flex:** keep the hand-mode semantics. The gripper's world approach pitch should equal `pitch_home + (hand_pitch − pitch0)` (from the existing Calibration).
- Given the lift and elbow above, find `wrist_flex` so that `Kinematics.fk(q)[1]` matches. In this planar chain, world pitch is affine in wrist_flex with slope ±1, so evaluate FK at two wrist values to get the slope sign and offset, then solve.
- If there is no hand or calibration: hold the previous wrist_flex.

**Robot wrist_roll and gripper:** exactly as in hand mode (wrapped roll delta; snap gripper with hysteresis, calibrated via `c` and `p`).

## Safety (in addition to everything that already applies)

1. **Table clearance:** after computing q, if the lowest gripper/jaw point (`cta.align.ContactModel.lowest_z`) is below `table_z + 0.02` m, move shoulder_lift toward 0 (raise the arm) by bisection until it's at least 0.02 m above. Unit-test it.
2. The same pipeline rate limit, smoother (arm joints only), joint-limit clip, COAST (< 0.15 s) and NO HAND hold semantics as `MirrorPipeline`. **No pose = NO HAND / COAST.** Hand missing but pose present: shoulder and elbow still follow, while wrist and gripper hold their last values (status `TRACKING`).
3. The mirror app's send path is untouched: lead bound, speed limit, measured-pose seed, stall warning.

## Interfaces

- `src/cta/pose.py`:
  - `PoseObservation(img: (33,3), world: (33,3), score: float)`
  - `PoseTracker(model_path, infer_width=1280)` with `.detect(rgb, t) -> PoseObservation | None` and `.close()`
- `src/cta/arm_angles.py` (pure):
  - `body_frame(world, mirrored) -> (up, left, fwd)`
  - `choose_arm_side(pose_img, hand_wrist_uv, last_side, want, mirrored) -> "left" | "right"`
  - `arm_joint_angles(world, side, mirrored, prev_pan) -> (pan, lift, elbow, pan_valid)`
- `src/cta/pipeline.py`: new class `ArmPosePipeline(kin, cal, gains, smoother, max_step_rad=0.15, aspect=16/9, mirrored=False, want="left")`.
  - The same public surface as `MirrorPipeline`: `step`, `calibrate`, `set_pinch_closed`, `reset_to`, `set_smoother`, `q_cmd`, `cal`, `smoother`, `last_features`.
  - `step(hand_obs, t, pose_obs)` has `pose_obs` as an extra argument.
  - Factor shared code out of `MirrorPipeline` rather than duplicating it.
- `src/cta/apps/mirror.py`:
  - `--control {arm,hand}`, default `arm`.
  - Construct the matching pipeline.
  - In arm mode, run `PoseTracker` on the same frame and draw pose points (shoulder, elbow, wrist of the chosen side) on the dashboard under `camera/image/arm`.
- `src/cta/recorder.py`: add optional per-row `pose_world (N,33,3)` and `pose_img (N,33,3)`, NaN when absent, plus a `control` field in meta. This stays backward compatible for `load_session`.

## Tests (TDD)

- **Synthetic bodies built in a test helper:** shoulders 0.35 m apart, hips 0.5 m below, the user facing the camera.
  - Arm straight up → lift ≈ 0.
  - Arm forward horizontal → lift ≈ 90°, pan ≈ 0.
  - Arm out to the user's left, horizontal → pan ≈ −90° (robot pan; gripper to the robot's left).
  - Arm hanging down → pan held from the previous value.
  - Elbow straight → elbow ≈ −90°; elbow at 90° → ≈ 0.
- The mirrored-image version of the same body gives the same robot angles.
- `choose_arm_side` picks the side nearest the hand wrist, regardless of labels.
- Table clearance raises the arm when the target is below the table.
- `ArmPosePipeline`: COAST / NO HAND / hand-missing-wrist-hold semantics; the snap gripper still works.
- **The app:** `--control arm` runs with fakes (extend the fakes in `test_mirror_app.py`), and `--control hand` behaviour is unchanged (existing tests pass).
