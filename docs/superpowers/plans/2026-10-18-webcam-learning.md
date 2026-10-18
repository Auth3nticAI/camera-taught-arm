# Webcam-to-SO-101 Learning Implementation Plan

> Execute tasks inline with superpowers:executing-plans. User explicitly requested planning followed by starting implementation.

**Goal:** Train and demonstrate a PyTorch model that imitates the working webcam reach controller by October 24, 2026, as the first learning component of the larger robotics foundation project.

**Architecture:** Keep the existing webcam/MediaPipe/reach/IK controller as teacher. Pair `human.mp4` frame i with `frames.npz:q_baseline[i]`, in simulation joint radians. Start with Hugging Face `facebook/dinov2-small` frozen visual features and a custom PyTorch regression head. Cache features once; split whole takes into train/validation/test. Physical control integration follows offline evaluation.

**Tech stack:** Existing Python 3.12, torch 2.6.0+cu124, OpenCV, Transformers, DINOv2 Small, NumPy.

**Spec:** Existing `docs/specs/2026-10-18-reach-control.md`, with the user's October 18 correction: no leader arm anywhere in the learning workflow. The older design's leader labels and accuracy claims are superseded.

## Global constraints

- Deadline October 24; October 21 is optional, not the core target.
- No leader collection or training dependencies. Preserve existing unrelated code and assets.
- Current `reach` mode is the primary collection mode. Do not silently mix `arm`, `hand`, or paired sessions.
- Exclude non-TRACKING rows, non-finite targets, incomplete takes, and mismatched video/NPZ lengths.
- Teacher targets are imitation labels, not independent physical ground truth. `q_baseline` precedes the app's final sent-command guard; keep that distinction in results.
- Test partitions contain complete unseen takes; correlated frames never cross partitions. Prefer recording blocks/days/viewpoints as separate held-out sessions; take separation alone does not establish cross-person generalization.
- No robot movement, camera acquisition, calibration changes, or deployment during offline implementation.
- Model weights, recordings and outputs live outside Git. Preserve the existing safe control path.

## Current evidence (October 18)

- Webcam reach control, recorder, twin, physical follower runtime and alignment exist.
- Runtime probe: torch 2.6.0+cu124, CUDA available, RTX 4070 Ti.
- No `sessions/*/frames*.npz` in default `C:/Users/trayd/dev/cta-data`; real training awaits user recordings.
- Transformers was absent. No learned policy or measured learning results exist yet.

## Review focus

1. Dropped tracking/NaN labels must not train a constant READY-pose predictor.
2. Missing or truncated video must fail instead of shifting image/action pairs.
3. A corrupted take must appear as rejected, never silently disappear.
4. No training statistics, early stopping or model selection may consume test takes.
5. Mirroring, calibration, smoothing and camera differences can change targets for similar images. Record them; first experiment uses one fixed setup. A single image cannot fully represent the teacher's temporal state.

## Milestones and acceptance gates

| Date | Milestone | Acceptance |
|---|---|---|
| Oct 18 | M1: collection readiness and audit | CLI reports accepted/rejected takes and usable counts; no hardware required |
| Oct 18-19 | M2: first real dataset | Target 12-20 takes, 30-60 seconds each, varied reach/height/grip, fixed camera and calibration; at least 3 usable takes required technically |
| Oct 19-20 | M3: trainable pretrained model | Download DINOv2, extract paired features, train custom head; best validation checkpoint and reproducible split saved |
| Oct 20-21 | M4: honest evaluation | Held-out per-joint MAE in degrees vs teacher, training-mean baseline, curves, latency; inspect failures; do not promise improvement |
| Oct 21-22 | M5: measured improvement | Compare frozen encoder with last-block fine-tuning OR causal GRU; choose based on measured failure, same split and data budget |
| Oct 22-23 | M6: demo integration | Explicit learned-mode opt-in; existing tracking gates/limits/clearance remain; webcam-to-twin runs on laptop; physical video recorded while arm available |
| Oct 23-24 | M7: presentation | Rehearsed seven-minute deck/demo if existing course notes remain current, local assets, recorded backup, measured results and limitations |

Autonomous SmolVLA tasks and humanoid transfer remain later milestones, not prerequisites for this submission. Hardware availability after October 19 is unconfirmed; older project notes say travel then, so collect arm footage early where possible.

## Task 1: audit and video-aligned data

Files: `src/cta/learning_data.py`, `tests/test_learning_data.py`.
Interfaces: `audit_sessions(root: Path, control='reach', stride=6) -> dict`; `read_take(path: Path, control='reach', stride=6) -> (dict, np.ndarray)`; `iter_rgb(path: Path, rows: np.ndarray)`.

- [x] Write tests for TRACKING/finite masks, mode rejection, empty root, complete-take splitting and actual video alignment/truncation.
- [x] Run tests red, implement validation and deterministic `split_takes(names, seed=42)`, then run green.
- [x] Expose `python -m cta.learning audit` with machine-readable JSON and nonzero exit for no usable data.

## Task 2: frozen pretrained encoder and head training

Files: `src/cta/learning.py`, `tests/test_learning.py`, dependency lock.
Interfaces: `JointHead(input_dim: int)`, `train_head(takes: dict, output: Path, epochs=100, seed=42) -> dict`; module CLI `extract`, `train`, `smoke-model`.

- [x] Test that training decreases a controlled regression loss; whole-take splits are disjoint and mean baseline uses train only; checkpoints reload for evaluation.
- [x] Cache frozen DINOv2 CLS embeddings with targets and source identities; retain original image proportions by letterboxing to 224 square, never horizontal-flip camera labels.
- [x] Fit feature normalization on train only, optimize radians MSE with AdamW, select by validation only, report held-out MAE degrees after selection.
- [x] Save configuration, source metadata, splits, best checkpoint, CSV learning curve and JSON metrics. Refuse output overwrite.
- [x] Test real pretrained model forward and action-head backward smoke on CUDA/CPU respectively, clearly separate from robotics evaluation. End-to-end fixture extraction and training also verified.

## Task 3: collect and run first real experiment (requires user motion)

- [ ] Record fixed-camera, same-control/calibration takes with existing `cta-mirror --record reach_train`; user performs motions and uses C/P/R controls.
- [ ] Audit, inspect coverage, reserve held-out takes, extract, train and inspect actual results. Do not invent data or performance.

## Task 4: improvement and live integration (after offline results)

- [ ] Pick an improvement from observed errors, write model and safety-integration tests first.
- [ ] Add explicit learned controller option with checkpoint/calibration compatibility checks; route predictions through all existing hold, speed, joint, clearance and follower guards.
- [ ] Benchmark latency on target laptop and simulator, then validate physical movements with user present.
- [ ] Prepare results plots, physical video and slides; update decision log and notebook with measured evidence.

## Execution record

User authorized plan-and-start; implementation proceeds inline without another approval round. This turn delivers Tasks 1-2 as offline infrastructure. Real dataset collection requires the user in front of the camera; later milestones are conditional on recorded data and evaluation evidence. No synthetic test result will be presented as robot performance.
