# Deck Notes

Running talking points for two decks. Add as you go; trim at the end.

---

## Course deck (audience: professor + class · goal: an A)

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
