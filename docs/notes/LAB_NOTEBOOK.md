# Takeworthy — Lab Notebook

Dated entries: what was tried, what broke, what was learned, numbers, and screenshot paths.

---

## 2026-10-16 · Research and design
- Researched four areas: human-to-robot retargeting, webcam joint-angle accuracy, the robotics-data market, and Windows tooling. Report: `reports/Webcam motion capture for robotics.md`.
- **Learned:**
  - A webcam gets big joints (elbow, knee) within a few degrees of lab mocap, but fingers are only ~11–22° accurate, and single-camera depth isn't metric.
  - The One Euro filter removes jitter but also cuts real motion (path-length ratio ≈ 0.48 in a 2026 study). That gap is the opening for a learned smoother.
  - Retargeting = converting human pose into robot joint angles. The AnyTeleop optimizer takes ~10 ms/frame; a learned network (GeoRT) can do it at 1 kHz.
  - GR00T N1.5 fine-tuning needs ~33 GB of VRAM, so it's out. SmolVLA (`lerobot/smolvla_base`) and ACT both fit.
- Ordered the SO-ARM101 Pro (leader + follower) and the innomaker camera. Arriving 10/17.
- Human-facing camera = an AI PTZ 4K webcam. **Lesson:** "smart" camera features (auto-tracking, gesture control, auto-zoom, auto-focus) break a vision system that assumes a fixed camera. Turn them off and lock it to 1080p@60.
- Wrote the design doc: `docs/specs/2026-10-16-takeworthy-design.md`.

## 2026-10-17 · Plan 1 built (teach layer, no arm yet)
- Built 9 tasks with an implementer and an independent reviewer per task, then a final whole-branch review. Result: **81 automated tests passing**. Branch `plan1/teach-layer` on GitHub.
- **No-hardware smoke run:** camera 1920×1080 @ **59.4 fps** (MJPG); app loop **~60 fps**; camera→command latency **11.7 ms**; full loop 14.7 ms. Both are well inside the gates (≥ 20 FPS, < 150 ms). Recording on: 53 fps.
- **Bug worth a slide (camera at 5 fps):** OpenCV on Windows (DirectShow) ignored the MJPG compression setting because it was set *before* resolution and fps. The camera fell back to uncompressed YUY2, which only manages 5 fps at 1080p over USB 2. Setting MJPG **last** gave 59.4 fps. A reviewer found it by testing every setting order on the real camera. The app now reads the format back and warns if it isn't MJPG.
- **Safety bug caught in review (before the arm arrived):** the first version would have sent the arm toward a preset "ready" pose on startup, at about 300–580°/s, wherever it was resting. Fixes:
  - start from the arm's *measured* pose;
  - cap joint speed (90°/s by default, `--max-speed-dps`);
  - send mirror commands only while a hand is tracked;
  - refuse non-finite values at two layers.
  Regression tests prove the first command stays within one speed step of the real pose, including when the arm rests near a joint limit.
- **Recording hardening (final review):**
  - a single dropped serial packet no longer ends a take (hold, record NaN; exit after 15 failures in a row);
  - Windows file locks can't crash autosave;
  - hotkeys only work when the terminal has focus;
  - the app refuses to drive the arm without a sim↔real alignment file;
  - each recorded frame stores tracking status and smoother on/off, so Plan 2 can mask frames.
- **Workspace box:** x 0.222–0.443 m, y −0.335–0.332 m, z 0.02–0.25 m (base frame). It's derived from reachable poses at table height; the first version had only 5 cm of forward reach.
- **Ready pose:** joints [0, −0.6, 0.9, 0, 0, 0.6] rad puts the gripper at (0.315, −0.001, 0.155) m.
- **Course-text tie-in:** this layer is classical (geometry + IK).

## 2026-10-17 (evening) · Follower arrived: calibration + alignment attempt
- **Calibration:** the follower is on COM4 (CH343 driver). `lerobot-calibrate` with id `cta_follower`. The first attempt recorded shoulder/elbow ranges of 0–4095 (a glitch or wrap), so it was redone. Final ranges are sensible (shoulder_lift 816–3088, elbow 914–3133, etc.).
- **Read-only probe** (torque off): the calibration matches the motors and all readings are finite. Folded rest = shoulder −99.6°, elbow +96°, at the stops.
- **Directions:** the twin at the folded-rest reading matched the real arm visually, and zero in LeRobot = the calibration "middle" pose in MuJoCo.
- **Alignment tool built** (`python -m cta.apps.align_follower`): you touch the table with the limp arm and least-squares solves the joint offsets.
  - First version let the table height float, so the shoulder and wrist offsets ran to ±15° and cancelled each other. Fixed: the table is pinned to the base bottom (−2.4 mm), there's a weak prior, and an untrusted fit can't be written.
  - Tests were reading the real data folder; isolated them with conftest.py.
- **OPEN ISSUE (blocks real-arm mirroring):**
  - In stretched-out poses with the jaws on the table, the motors read shoulder ≈ +15°, elbow ≈ −7°. The twin shows the arm upright with the jaws level about 15 cm *above* the table.
  - Near-base poses agree within ~1 cm. So shoulder and/or elbow disagree with the simulator by tens of degrees in the reaching region.
  - **Unsafe direction:** the sim thinks the gripper is higher than it is, so the 2 cm table clamp would not protect the real arm.
  - Next step: a side photo of the real arm in that pose next to the twin, to identify the joint. Then either recalibrate or apply the corrected offset/sign.
  - Captures are saved in `cta-data/align_poses.npz`. The rejected maps are kept as `sim_real_map.rejected-*.json`.
  - Real-arm mirroring is on hold, and the app refuses without a map, by design. It is the *teacher* and baseline that the Plan 2 networks (ch. 6 MLP, ch. 4.4 time windows) must beat.

## 2026-10-17 (night) · Live sim mirror working ✅
- `uv run cta-mirror --robot none --mirrored --hand left` works: left/right, up/down, toward/away, tilt, roll, and pinch all drive the simulated SO-101.
- **Camera lessons:**
  - The OBSBOT was **digitally zoomed in**, which produced the blocky close-up. **Rerun remembers its view zoom per app**, so double-click a panel to reset it.
  - Only one app can hold the camera at a time (close the OBSBOT Center preview first).
  - Camera indices: 0 = OBSBOT, 1 = EOS virtual, 2 = innomaker, 4 = OBS virtual.
  - The OBSBOT image is mirrored, so `--mirrored` is needed for the left/right hand labels to be correct.
- **Bug fixed:** wrist roll thrashed because the roll difference didn't wrap at ±180°; a palm-facing-camera home pose sits right on that wrap point. Regression test added.
- **Gripper redesign (D-N):** snap open/close with a 35%/55% dead-zone, calibrated to the user's own open hand (`c`) and pinch (`p`), and bypassing the arm smoother. The proportional version was laggy and inaccurate.
- 93 tests passing.

## 2026-10-17 (night) · Sim↔real alignment solved ✅
- The "big shoulder/elbow mismatch" was **not real**. The wrist camera (innomaker, index 2, mounted on the gripper) showed the arm held *level in the air* in the "stretched" captures, which matches the motor readings exactly. To touch the table, the wrist must bend down 60–90°. Reference touch poses are rendered in `docs/assets/alignment-touch-poses.png`.
- **12 table-touch captures.** One outlier (#1, 19 mm off) was dropped.
- **Result:** shoulder_lift **+9.8°**, elbow **+4.2°**, wrist_flex **+2.1°**; uncertainty ≤ 1.3°; contact error **47 mm → 3.2 mm**.
  - The shoulder offset (~10°) is the hand-set calibration "middle" pose being off vertical.
  - Saved to `cta-data/sim_real_map.json`.
- **Deck line:** "Calibrated by touching the table 11 times: least-squares cut the sim-to-real contact error from 47 mm to 3 mm."

## 2026-10-17 (late) · First real-arm mirroring 🦾
- **No 12 V power at first:** the motors read **4.9 V** (USB only). USB powers reading but not moving. With the 12 V supply connected they read 12.0 V.
- **Speed limiter, two iterations:**
  - Ramping from the last *command* drifted away from a stalled arm.
  - Ramping from the *measured* pose crawled and shook, because each step fell inside the servo deadband.
  - **Final:** ramp from the last command, bounded to 12° ahead of the measured pose, plus a "not moving, check 12 V" warning. Regression tests cover the stall, the deadband, and the lead bound.
- **Tracking flicker:** the log alternated TRACKING/NO HAND every frame because MediaPipe's left/right label flickers on a small hand.
  - Now the app follows the same hand by wrist position, coasts through dropouts under 0.15 s (status COAST), and detects at 1280 px.
- **Result:** the real arm follows the left hand. The loop runs at ~42 fps with ~20 ms latency. Some lag remains, and the arm doesn't cover the full range of hand motion.
- **Live-tuning settings added** (defaults are Claude's starting guesses; Tray tunes them):
  - `--servo-p 24` (LeRobot default 16);
  - `--gain 1.5`;
  - `--smooth-beta 0.05` (was 0.007);
  - lead 12° (LeRobot backstop 15°).

### Tuning plan (next session)
1. Baseline: `uv run cta-mirror --robot so101 --follower-port COM4 --mirrored --hand left`
2. **Lag:** try `--servo-p 28`, then `32`. Stop when it starts to shake when your hand is still.
3. **Range:** try `--gain 1.2 / 1.5 / 2.0`, and pick the one where your comfortable hand range covers the arm's workspace.
4. **Jitter vs. lag:** try `--smooth-beta 0.02 / 0.05 / 0.1`. Also try the `s` key (smoothing off) to compare.
5. Log the chosen values in DECISIONS.md. They're deck material: "tuned servo stiffness vs. shake".

## 2026-10-18 - Webcam-only PyTorch learning infrastructure

- User confirmed no leader arm; requested milestone plan and implementation start. Working branch: codex/webcam-learning.
- Plan: docs/superpowers/plans/2026-10-18-webcam-learning.md. Collection and training guide: docs/WEBCAM_TRAINING.md.
- Runtime verified: torch 2.6.0+cu124, CUDA available, NVIDIA RTX 4070 Ti. Added Transformers (locked 4.57.6).
- Added recording audit, strict image/target pairing, tracking/finite masks, whole-take train/validation/test splits, fixed setup checks, frozen DINOv2 feature extraction, custom PyTorch regression head, checkpoint reload, CSV curves and JSON held-out metrics. Live controller unchanged.
- Actual pretrained facebook/dinov2-small downloaded and ran on CUDA: CLS features (1, 384). Custom head backward produced finite gradients on CPU.
- End-to-end extraction/training verified on three temporary generated video fixtures; temporary outputs removed automatically. These are software checks, NOT robot results.
- Verification: 232 tests passed in 39.37 seconds; 12 cover the new learning pipeline. git diff --check passed (only Windows line-ending notices).
- Real data audit: zero accepted takes and zero usable frames in the default sessions folder. No real learned-control accuracy or task success is claimed. Next action: user records 12-20 varied fixed-setup takes using the webcam teacher, then run extraction/training.

## 2026-10-18 - Erratic reach control: recorded failure and fix

- User reported uncontrolled-looking movement. Inspected both saved takes and decoded the second take's video with pose overlays; no hardware or camera was opened.
- Evidence: reach_train_02 has 1,654 frames over 67.89 seconds. Reconstructed arm selection changes 25 times. At rows 1315, 1316 and 1540, shoulder image separation collapses to 0.042, 0.047 and 0.020, despite confidence near 1.0; video shows the user close to camera with upper body cropped and false shoulder landmarks near the hand. Row 1118 shows shoulders teleporting into the torso/hand area.
- Root cause: the reach controller trusted landmark confidence and nearest-wrist arm selection every frame, allowing arm identity switches and confidently incorrect body geometry to become robot targets. No learned model is active in this live path.
- Change: ReachPipeline holds on arm identity changes, offscreen/collapsed shoulders, shoulder teleports and abrupt normalized 3D reach changes. C calibration resets the body reference. Terminal prints TRACKING HOLD with the reason. Other control modes and physical command guards remain as before.
- Tolerances are explicit conservative starting heuristics, not proof all pose errors can be detected. They may hold legitimate fast motion or large reframing; reframe with shoulders/arm visible and recalibrate.
- Offline replay with the new pipeline rejected 248 frames (195 identity, 24 shoulder jumps, 20 collapsed spans, 9 3D reach jumps). All four inspected severe rows held. Report at cta-data/reach_debug_report.json and contact sheet at cta-data/reach_debug.jpg.
- Added three regression tests; all 30 reach tests passed. Live validation still requires restarting the app and user movement. Existing recorded takes are debugging evidence, not validated clean training demonstrations.

## 2026-10-18 - Elbow display and low neutral reach mapping

- User screenshots showed no elbow marker and a low/flattened robot despite a raised human hand.
- Confirmed display bug: reach mode explicitly selected indices [shoulder, wrist] for the overlay. Elbow tracking still contributed to arm length and visibility gating internally. Changed overlay to shoulder/elbow/wrist.
- Confirmed mapping issue: legacy target = robot shoulder + robot reach * normalized human shoulder-to-wrist vector. A natural hand-below-shoulder posture can put the requested gripper below the table, clamping it to the floor. This was not neutral-pose calibrated.
- CLI now defaults to home-relative reach: first accepted body pose after C maps to calibration p0 (raised ready target); relative changes drive Cartesian position through the existing IK and guards. The library retains absolute default for compatibility with existing experiments; CLI has explicit --reach-mapping absolute fallback.
- Recordings store reach_mapping and reach_home_n; cache extraction refuses mixed references. Relative recording waits for a valid tracked home reference. Paired-mode legacy recording remains unaffected.
- Checked recent recordings: reach_train_03/04 both use legacy absolute mapping. In take 04, 7 of 243 frames with hand wrist visibly above the selected shoulder still had inferred normalized world-up below -0.1, showing additional pose inconsistency. Camera image and learned 3D pose are not interchangeable. No claim that the home offset solves all pose errors.
- New regression verifies a neutral wrist below shoulder maps to >14 cm gripper height, raising the wrist raises the target, and recalibration resets the reference. App test verifies elbow overlay and relative default.
- Live verification pending restart; no camera or physical robot operated during this diagnosis.

## 2026-10-18 - Wrist-loss hold/resume jitter correction

- User supplied a long alternating TRACKING HOLD: 3D reach jumped / TRACKING RESUMED log and screenshots showing wrist losses and loop-time spikes. No new NPZ sessions were saved for this run; the log confirms guard chatter, while screenshots alone do not isolate the latency spikes.
- Code root causes reproduced in failing tests: (1) reach could continue body-driven motion for up to one second after hand loss and change pitch to a fallback angle, (2) jump acceptance relaxed with time since last accepted frame, (3) a single good-looking frame immediately resumed motion, and (4) hand selection immediately fell back to a distant handedness-labelled detection.
- Fix: no wrist or invalid wrist orientation => exact all-joint hold. Pose and hand wrist must agree within 0.12 normalized image distance. Recovery requires three consecutive consistent observations (normalized reach delta <=0.12, shoulder delta <=0.04, gap <=0.2 seconds); geometry and identity checks still apply. A lone good frame cannot resume.
- On relative-mode recovery, rebase the human reach reference to the held target (clutch behavior). Recovery should not chase a newly inferred absolute hand location. This changes the relation to the original calibration after a loss; it is deliberate to prioritize continuity. Recalibration restores explicit home alignment.
- Relative target speed capped at 0.25 m/s; final joint steps, including solver clearance corrections, capped at 90 degrees/second with elapsed time capped at 50 ms. Table checks remain after the limiter. Hand tracker waits for position-track expiry rather than jumping to a distant hand immediately.
- Targeted verification: all 38 hand/reach tests passed, including reproduced missing-wrist movement, alternating-depth chatter, recovery and time-step bound. Full-suite result follows below. No claim of solved camera detection; live validation after restart remains required.

Implementation note: stable recovery is currently three consecutive observations, not a guarantee of correct hand detection. No camera or robot was opened in this session. Startup still reports UNCALIBRATED before calibration. A full-suite run initially caught that status regression (1 failed, 237 passed); it was corrected before the final rerun.
Final wrist-recovery verification: 238 tests passed in 45.34 seconds. git diff --check passed, with Windows line-ending notices only. Live user validation is pending restart.

## 2026-10-18 - Reach measurement fix: locked arm + hand-tracker "where" + Depth Anything V2 forward

- Spec: docs/specs/2026-10-18-reach-measurement-fix.md. The arm is locked at 'c' (nearest Pose wrist to the hand-tracker wrist, un-mirrored) and never re-picked. In-plane "where" = hand-tracker wrist minus the locked Pose shoulder, in arm lengths (1.7 x the 0.5 s median image shoulder separation). Forward = Depth Anything V2 Metric Indoor Small (palm-centre patch minus shoulder depth, 3-frame median) / (1.7 x Pose world shoulder width). Pose wrist/elbow no longer used. Hand-scale fallback when the model can't load or with --depth none.
- Removed holds made obsolete by the lock: arm identity changed, wrist disagrees with arm tracker, 3D reach jumped, 3-frame reacquisition, clutch rebase. Kept: shoulder out of view / span collapse / shoulder teleport (allowance now grows with time since the last good shoulders, so it can't hold forever), the 0.25 m/s target speed limit, 90 deg/s joint bound, clearance, all app guards.
- Replay proof (scripts/replay_reach.py, real recorded data reach_train_01..04, real model on CUDA; 'c' = first frame with hand + pose, recorded hand calibration; OLD = commit 27488e3):
  - Arm side switches after 'c': OLD 14 / 31 / 4 / 0 (49 total) -> NEW 0 / 0 / 0 / 0.
  - Frames held although hand and pose were both present: OLD 36.2% / 26.7% / 19.5% / 34.6% -> NEW 1.6% / 6.2% / 0.0% / 0.0%. TRACKING share: OLD 53.7 / 62.0 / 66.8 / 33.3% -> NEW 82.9 / 79.3 / 83.0 / 50.9%.
  - Raw target jitter (second-difference noise) p50: OLD 0.89 / 0.57 / 0.65 / 1.26 cm -> NEW 0.42 / 0.28 / 0.28 / 0.24 cm.
  - Final target (what IK gets) per-frame jump: p99 <= 1.25 cm, max 1.25 cm, 0 jumps > 3 cm in every take for BOTH OLD and NEW. This meets the acceptance (p99 <= 1.5, max <= 4) but only because the existing 0.25 m/s speed limit caps a 24 fps frame at ~1.25 cm, so it does not by itself show better measurement.
  - Smoothed target before the speed limit, consecutive frames: NEW p99 4.22 / 3.09 / 2.99 / 3.70 cm, max 5.08 / 4.29 / 4.27 / 4.78 cm (OLD 2.78 / 2.45 / 2.39 / 2.79, max 3.90 / 4.21 / 2.97 / 3.54, on the 54-67% of frames it didn't hold). Residual: monocular depth at the hand (forward axis carries most > 3 cm frames in takes 1-3; a longer 5-frame median didn't change it, so it is multi-frame depth drift, not single-frame flicker) and genuine fast motion at the 24 fps the takes were recorded at (take 4: in-plane). Re-acquisition after hand gaps now moves toward the hand's new place (smoothed up to 37 cm, glided at 0.25 m/s) because the clutch was removed.
- Depth timing: 9.2 ms/frame median in replay (518 px input, fp16, RTX 4070 Ti; 6 ms at 266 px but metric scale shifts: shoulder ~2.1-2.3 m at 266 vs ~1.6-1.7 m at 518). Live 12-14 ms (contention), run on a worker thread overlapped with MediaPipe.
- Live loop (cta-mirror --robot none --mirrored --hand left): 600-frame runs, average fps 24.4 and 22.8 with depth every frame vs 22.4 and 24.1 with --depth none; 200-frame runs 18.6-23.1. Target >= 20 fps met on the long runs; --depth-every 1 kept.
- Hand point: wrist (lm 0) chosen over palm centre: lower smoothed p99 and fewer > 3 cm frames in all four takes; palm centre has slightly lower raw jitter. Depth itself is always sampled at the palm centre (the wrist patch hit the background: hand depth jumps p99 0.9-1.4 m per frame).
- Recorder adds depth_hand, depth_shoulder, locked_side per row. Tests: 276 passed. No robot was moved; live camera runs were for fps only.
