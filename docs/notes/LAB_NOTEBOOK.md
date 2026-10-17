# Lab Notebook

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
- Wrote the design doc: `docs/specs/2026-10-16-camera-taught-arm-design.md`.
