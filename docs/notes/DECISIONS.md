# Decision Log

Every engineering decision: what was asked, the options, **Tray's call**, why, and how it turned out. This log is the source material for both decks.

Format: `D-## · date · decision` → options → **choice** → why → outcome (filled in later).

---

### D-A · 2026-10-16 · Project direction
- **Options:** gesture-controlled effects (the LinkedIn HandTrack idea), touchless control, sign language → text, a rehab coach, 10 robotics-data ideas (see `reports/Webcam motion capture for robotics.md`)
- **Choice:** human motion → robot data/control (robotics), with a startup framing
- **Why:** the LinkedIn project uses hand-written rules, with no neural network and no PyTorch, so it would score B at best. Robotics data is a funded market bottleneck: Figure has paid contributors $15M, and XDOF has reported a ~$1.2B valuation on cheap teleop capture.
- **Outcome:** —

### D-B · 2026-10-16 · Which robotics build
- **Options:** (A) the hand drives the real SO-101 · (B) the fingers drive a simulated 5-finger robot hand · (C) both
- **Choice:** **A**
- **Why:** it uses the real arm, enables block picking and autonomy, and is the strongest startup story. B is a faster version of a free library (`dex-retargeting`). C doubles the work in 8 days.
- **Outcome:** —

### D-C · 2026-10-16 · Hardware
- **Choice:** Seeed SO-ARM101 Pro, assembled (leader + follower, 12V), $427.76 order including the innomaker 130° USB camera; Raspberry Pi 5 already on hand
- **Why:** it arrives in one day. The leader arm doubles as a **ground-truth sensor** for measuring webcam accuracy in degrees.
- **Outcome:** arriving 2026-10-17

### D-D · 2026-10-16 · Control mapping
- **Options:** absolute same-side · relative/clutch · mirrored
- **Choice:** **absolute, same-side** (calibrate a "home" pose, then hand box → arm workspace)
- **Why:** the most intuitive for collecting demos, and the easiest to explain on stage
- **Outcome:** —

### D-E · 2026-10-16 · Machines and demo venue
- **Choice:** develop on both the desktop (RTX 4070 Ti) and the laptop (MSI GE76, RTX 3080 Laptop GPU); train on the desktop; **demo on the laptop**
- **Why:** portability. The laptop's GPU can run the full loop live, including the pick policy.
- **Outcome:** —

### D-F · 2026-10-16 · Cameras
- **Choice:** buy a wrist camera plus a webcam to watch the human; the innomaker goes overhead
- **Why:** two-view (wrist + overhead) is the standard SO-101 policy setup, and pick policies are more reliable with it when there are only ~50 demos
- **Outcome:** —

### D-G · 2026-10-16 · Autonomy task definition
- **Choice:** pick and place into a bin; red and green blocks both always present; positions varied within ~20×20 cm
- **Why:** gives a clear pass/fail for counting success rate, and proves the arm "knows which block"
- **Outcome:** —

### D-H · 2026-10-16 · Presentation venue
- **Choice:** online (Oct 24) from the laptop, **away from home (no arm)**. The recorded video shows the real arm (mirroring + autonomous pick); the code runs live on the laptop driving the **simulated twin** from the laptop webcam.
- **Why:** the arm can't travel. The video proves the hardware works; the live sim run satisfies "run the code to show results."
- **Consequence:** new requirement R1, a no-hardware mode. The video must be recorded before leaving home.
- **Outcome:** —

### D-I · 2026-10-16 · Arm deadline and demo source
- **Constraint:** everything needing the arm (recordings + video) is done by **Mon Oct 19**; Tray is away afterward
- **Options:** leader-arm demos · **webcam-taught demos** · drop autonomy
- **Choice:** **webcam-taught demos** for the pick policy
- **Why:** proves the startup claim end to end ("a camera taught it")
- **Safeguard:** Sunday ~2 PM checkpoint where Tray decides whether mirroring is good enough or switches to the leader arm
- **Consequence:** ACT replaces SmolVLA (it trains overnight on ~50 demos); SmolVLA goes to the roadmap
- **Outcome:** —

### D-J · 2026-10-17 · Where things live
- **Options:** Hugging Face Hub · Google Drive · GitHub LFS
- **Choice:** code, docs, and notes in **GitHub** (private); recordings, datasets, and models in **Hugging Face Hub** (private)
- **Why:** no more OneDrive. Regular GitHub can't hold videos (100 MB file cap; LFS free tier is 1 GB). HF is built for datasets and models, LeRobot syncs with it natively, and it can go public later as a showcase.
- **Outcome:** —

### D-K · 2026-10-17 · Build process
- **Choice:** subagent-driven execution: a fresh implementer per task, plus a reviewer per task
- **Why:** more thorough review of each piece
- **Outcome:** —

### D-L · 2026-10-17 · Follow the course text
- **Choice:** the training code follows *Deep Learning with PyTorch* idioms (ch. 5–8, 10–13, 15); mapping in spec §8c
- **Why:** shows the course material applied, which supports the A
- **Outcome:** —

### D-09 · 2026-10-17 · Project name
- **Options:** 30-name exploration; finalists were Takeworthy, MotorKin, Deftloom, Skillwick, and Mimewire. Full report: `reports/brand-naming-2026-10-17.md`; domain evidence: `reports/naming-domain-evidence-2026-10-17.json`
- **Choice:** **Takeworthy**, pronounced TAKE-wur-thee, with the descriptor "Webcam demonstrations. LeRobot datasets."
- **Scope:** this is the **project name only**. The company/startup name is still open, and so are domain purchase and trademark clearance.
- **Why:** "a take worth training on" connects recording a demonstration with producing useful data, and it leaves room for datasets and future quality scoring. It was the only finalist with no registry record on .com, .dev, or .ai (as of 2026-10-17; not checkout-verified).
- **Watch-out:** "take" can sound like video production, so always pair the name with the robotics descriptor.
- **Outcome:** —

### D-M · 2026-10-17 · Control hand
- **Options:** right hand · left hand
- **Choice:** **left hand**: run with `--hand left --mirrored`. The OBSBOT sends a mirrored image, which is why the default "right" setting was actually tracking the left hand.
- **Why:** it tracked and felt better in the live test, and all six directions worked.
- **Outcome:** —

### D-N · 2026-10-17 · Gripper behaviour
- **Options:** snap open/close · proportional · both with a toggle
- **Choice:** **snap open/close**, with a dead-zone (closes below 35% of your pinch range, opens above 55%)
- **Why:** proportional was laggy and inaccurate live. Snap is fast and reliable for picking blocks and gives clean training demos.
- **Also:**
  - the gripper now bypasses the arm smoothing filter;
  - "open" is calibrated from your open hand when you press `c`, and "closed" from your own pinch when you press `p`.
- **Outcome:** —

### D-O · 2026-10-17 · Control mode: arm pose + hand
- **Options:** arm pose + hand · keep hand-position IK and tune it · arm pose only
- **Choice:** **arm pose + hand**. MediaPipe Pose tracks the user's shoulder and elbow, and those angles drive the robot's shoulder_pan, shoulder_lift and elbow_flex **directly, joint for joint**. The hand tracker still drives wrist pitch/roll and the snap gripper. The hand-position IK mode stays available as `--control hand`.
- **Why:** with hand-position IK, the robot's elbow didn't copy Tray's elbow ("not following me correctly"). Joint-for-joint copying matches the original goal: "show which joint I'm using so robots can emulate".
- **Known costs:**
  - the upper body must be in frame;
  - reaching toward the camera is hard to read (a camera 30–45° off to the side helps);
  - roughly 25–30 fps with both trackers running.
- **Outcome:** —

### D-P · 2026-10-17 · Arm-pose safety rules (Claude's rulings for Tray to confirm)
- **Engagement:** in arm mode, the body alone only moves the robot if your **hand was seen in the last 1.0 s**. Otherwise the robot holds. This stops passers-by or a half-visible body from driving the arm.
- **Visibility gating:** pose points MediaPipe is unsure of (visibility below 0.5, adjustable with `--pose-min-vis`) count as "no pose". If the hips are hidden behind the desk, the app falls back to the camera's vertical.
- **Sent-path table clearance:** every command actually sent to the arm is checked against the table, not just the final target. A review simulation showed intermediate commands up to 19 cm below the table without this.
- **Cost if wrong:** the arm is slightly more conservative (holds more often). Tray can relax the rules after live testing.

### D-Q · 2026-10-18 · Table safety margin
- **Options:** 0.5 cm · keep 2 cm · make it a flag
- **Choice:** **0.5 cm**, set with `--table-margin 0.005`. Internally the check uses +1 mm to cover the fast collision model's worst-case 0.76 mm error.
- **Why:** the blocks are 2.5 cm tall, and with a 2 cm margin the jaws can't reach a block's middle. The sim-to-real alignment error is about 3 mm, so 0.5 cm still keeps a buffer.
- **Outcome:** —

### D-R · 2026-10-18 · Flipped arm mapping (the robot's elbow must mimic Tray's)
- **Problem:** in a literal copy, Tray's seated pose (upper arm hanging, forearm forward) sent the robot's upper arm down to its limit and folded the elbow down, putting the jaws on the floor. Tray: "it needs to inverse my elbow."
- **Choice:** **flipped mapping is the default** (`--arm-mapping flipped`; the literal copy is `--arm-mapping direct`). The robot is Tray's arm reflected vertically: a hanging upper arm becomes the robot's standing upper arm, and a forward forearm stays forward (the robot's L pose). The elbow fold then matches the robot's natural fold.
- **Also:** when the upper arm is vertical, the **forearm's direction swings the base**, so pointing the forearm left or right pans the robot.
- **Verified:** tests check the robot's actual upper-arm and forearm directions, from the MuJoCo model, against Tray's arm reflected vertically, within 1°.
- **Outcome:** —

### D-S · 2026-10-18 · Shoulder glitch handling (Tray's engineering call)
- **Options:** stop and press `c` · pause briefly, then continue · **Tray's own design: per-frame spatial threshold**
- **Choice:**
  - A shoulder jump beyond the allowed per-frame motion is **rejected for that frame**. The arm holds its last good command and resumes on the next valid frame.
  - Claude's addition: a new position that stays consistent for **5 frames (~0.2 s)** is a real body move and becomes the new baseline automatically, with no `c` needed. This is safe because reach is measured hand-relative-to-shoulder.
- **Why:** you get the safety of rejecting glitches without the pauses. The old "press c" hold also froze forever after a real body shift.
- **Result (replay, take 02):**
  - held while visible: 18.2% → **8.1%** (original code: 26.7%);
  - 0 arm switches;
  - final per-frame target max: 1.25 cm.
- **Outcome:** —

---

## Open (waiting for Tray)
D-01 demo-day priority · D-03 label source · D-04 depth cue · D-05 smoother architecture · D-06 ACT vs SmolVLA · D-07 where the motors connect · company/startup name · D-10 confirm the label-shift plan, the smoother target, and the gate targets. Details are in `docs/specs/2026-10-16-takeworthy-design.md` §7.

### October 18 update: webcam-only learning direction

- **User instruction:** no leader arm; learn from webcam movements. Plan the next milestones and start implementation.
- **Supersedes:** D-03 leader pairing and the older design's leader-ground-truth claims. Existing paired-mode code is retained, but is excluded from the new learning workflow.
- **Implementation choice:** start with Hugging Face `facebook/dinov2-small` frozen features and a custom PyTorch regression head, trained against the working reach controller's `q_baseline` targets. This is controller distillation, not independently measured physical accuracy.
- **Reason:** reuses synchronized webcam/command recording and provides a trainable network within the October 24 deadline. Full fine-tuning, temporal modeling and live deployment follow evaluation.
- **Data constraint:** fixed camera/calibration/control setup; complete-take splits; no fabricated robot results. No recorded takes currently exist in the default data root.
- **Plan:** `docs/superpowers/plans/2026-10-18-webcam-learning.md`; commands in `docs/WEBCAM_TRAINING.md`.
