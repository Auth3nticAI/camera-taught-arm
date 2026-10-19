# Deck Notes

Running talking points for two decks. Add as you go; trim at the end.

---

**Project name: Takeworthy** — "Human skill. Captured." · descriptor: *Webcam demonstrations. LeRobot datasets.* (It's the project name only; the company name is still open.)

## Course deck (audience: professor + class · goal: an A)

**Title slide:** Takeworthy: teaching a robot arm with a webcam and neural networks.

**Format (instructor, Oct 11 post):** 7 minutes, screen share, "a few slides, run the code to show results." Target ~5 slides (~3 min) plus a live demo dashboard (~3–4 min). Oct 24 final; optional early slot Oct 21, 7:30–9:30 PM.

**Problem:** robots learn from demonstrations, but collecting them needs a second robot arm, VR gear, or a mocap rig. Can a webcam plus neural networks replace that?

**Requirement map:**
1. Problem defined: camera-taught robot arm
2. Python + PyTorch: yes
3. Own data: self-recorded hand ↔ leader-arm paired sessions, plus teaching episodes (LeRobot format)
4. Networks made or improved: retargeting network (beats the IK baseline), GRU smoother (beats One Euro), fine-tuned pick policy
5. Live demo: the arm mirrors the hand, then picks the commanded block alone

**Course-text slide:** one table mapping book chapters → project parts (see spec §8c). Ch. 6 = the retargeting MLP; ch. 4.4 = time windows for the smoother; ch. 10 = the split by take (and why it beats a per-frame split); ch. 11 = the training app + TensorBoard; ch. 12 = augmentation; ch. 15 = ONNX on the Pi 5.

**Why it's beyond a B:**
- three trained networks;
- measured ground truth from the leader arm's encoders;
- honest baselines;
- distribution shift named and tested (gripping vs. free hand).

**Results slots:** per-joint error (°), latency (ms), FPS, jitter/lag vs. One Euro, pick success % per color.

**Results slide: the network (2026-10-19, dataset v2, 12 takes):**
- Frozen DINOv2 + custom PyTorch head, trained on Tray's own webcam takes. Test error is **21.3°**, against **31.2°** for the predict-the-mean baseline (about 30% lower).
- The ch. 12 experiment, shown as `docs/figures/learning_curves.png`:
  - Dropout, weight decay, noise and augmentation cut the train/val gap from 55× to 3.4×.
  - The model chosen on validation (B) ties the baseline design on the test take.
- **Lesson to say out loud:** "Regularization stopped the memorizing, but the error didn't drop. The bottleneck is what the network sees, not how it is trained. The hand is a few pixels in a whole-body frame, so wrist roll and gripper stay hard." That points straight to the next step: a hand-crop input.
- Per-joint chart: `docs/figures/per_joint_mae.png`.

**Results slide 2: improving the network (hand-crop input, D-U):**
- The diagnosis from experiment 1 was that the hand is only a few pixels. Fix: crop it, using the hand tracker's landmarks.
- **Leave-one-take-out cross-validation** (`docs/figures/cv_inputs.png`): the hand crop beats the full frame in **11 of 11** folds.
- **Final held-out test:** **20.5°** vs 31.2° baseline (34% lower). Wrist roll error −20%.
- **Rigor talking point:** "Two models tied on validation. Instead of peeking at the test take, I cross-validated across takes to choose."
- **Honest limit:** the gripper did not improve. Its target is open/closed, so the next step is a classifier output for it.

---

## Client / investor deck (audience: robotics teams, investors · goal: start the company)

**Pain:** "Robot data gap": about 1 year of robot data exists vs. ~100,000 years of text/image data (Ken Goldberg). Teleoperation data costs ~$28–200/hour from vendors.

**Proof of market:**
- Figure's Index app: 16M uploads, $15M paid to contributors.
- XDOF: $70M Series A on a sub-$300 teleop rig.
- Mecka AI: $60M Series B (Oct 2026).

**Our wedge:** no leader arm, no headset, no glove: **a webcam teaches the robot**. Output is a standard LeRobot dataset, ready to train.

**Demo:** show it → it does it.

**Roadmap:** more tasks → a per-session quality score (buyers reportedly discard ~90% of raw footage) → edge kit (Pi 5) → crowdsourced teaching.

**Why me:** *(Tray to write)*

**Honest limits to own:** single-camera depth; gripper driven by pinch, not learned; small demo counts.
