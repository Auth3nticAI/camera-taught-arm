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
