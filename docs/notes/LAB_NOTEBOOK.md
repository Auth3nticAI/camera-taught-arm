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
