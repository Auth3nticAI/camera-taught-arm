# Webcam-to-SO-101 class project

The project now has an offline DINOv2 feature extractor and a custom PyTorch joint-regression head. The existing reach/IK controller remains the teacher and the live controller. No leader arm is used. The learned head is not yet connected to the robot.

Milestone plan: [October 18-24](superpowers/plans/2026-10-18-webcam-learning.md).

## 1. Record the real data

In the repository terminal, close other apps that hold the camera, and run:

```powershell
uv run cta-mirror --robot none --control reach --hand left --record reach_train
```

This records real webcam motion with the simulator teacher; physical-arm movement is not needed to train the first network. `--hand left` matches Tray's setup. **Do not pass `--mirrored`:** the OBSBOT image is NOT mirrored (verified 2026-10-18: the "EMERGE" chair text reads normally). The flag swaps left/right.

For actual follower demonstrations, use your already validated hardware setup:

```powershell
uv run cta-mirror --robot so101 --follower-port COM4 --control reach --hand left --record reach_train
```

COM4 is the previously logged follower port; use the current correct port if it changed. Keep the normal alignment and control guards enabled.

1. Click the terminal for keyboard controls. Hold the hand at home and press **C**. Calibrate your pinch with **P**.
2. Press **R** to start a take, perform movements, press **R** to stop. Aim for 12-20 takes of 30-60 seconds. Three takes is only the technical minimum.
3. Include reaching forward/back, left/right, high/low, wrist tilt/rotation and open/close grip. Include still holds as well as moving segments. Mix these within each take so each partition has coverage.
4. Keep the camera, zoom, mirror setting, hand, calibration and gains fixed throughout this experiment. Do not press C/P or toggle smoothing between takes. Calibration changes cause extraction to reject mixing those takes.
5. Press **Q** after stopping the last recording. Sessions are saved in `CTA_DATA_DIR/sessions` (default `C:/Users/trayd/dev/cta-data/sessions`).

The target is `q_baseline`: the classical pipeline's joint command in simulation radians, before the app's final sent-command guard. Follower observations are not training targets. Bad tracking, calibration/hold rows and NaN labels are excluded. The first model learns the teacher, not independently verified physical intent or autonomous task completion.

## 2. Check the data

```powershell
uv run python -m cta.learning audit
```

The audit decodes video to verify correspondence with NPZ rows. It reports each accepted/rejected take and the usable frames after stride-6 sampling (roughly 10 Hz if recording at 60 FPS). Empty data or rejected takes return exit code 1. Fix or move incompatible/incomplete takes before extracting. Use `--sessions PATH` for a different session root. No camera or robot is opened by the learning commands.

## 3. Extract pretrained features and train

```powershell
uv run python -m cta.learning extract --output C:/Users/trayd/dev/cta-data/learning/reach_dino_v1
uv run python -m cta.learning train --cache C:/Users/trayd/dev/cta-data/learning/reach_dino_v1 --output C:/Users/trayd/dev/cta-data/learning/reach_head_v1 --epochs 100
```

Run `uv sync` on another machine first. Both commands refuse to overwrite outputs. Choose a fresh versioned directory for each experiment. Extraction downloads `facebook/dinov2-small` from Hugging Face the first time, uses CUDA if available, letterboxes the complete image to 224 square and caches frozen CLS features. The training command trains the custom head on CPU; the expensive encoder runs only during extraction. This is frozen-backbone transfer learning; full encoder fine-tuning is a later experiment.

Outputs:

- `best.pt`: best validation checkpoint, feature normalization, target units, seed and take split.
- `metrics.json`: held-out per-joint imitation MAE in degrees, training-mean baseline and sample counts.
- `history.csv`: training/validation loss per epoch, in radians squared.
- `source_manifest.json`: encoder identity, source metadata and NPZ fingerprints.

Take names are deterministically split at seed 42, approximately 70/15/15, with at least one take in each partition. Statistics come only from training data. Validation selects the checkpoint; test error is calculated afterward. Do not use repeated test results to choose architecture or hyperparameters; keep a fresh final recording block for final evaluation. Separate takes from the same session still do not prove cross-camera, cross-day or cross-person generalization.

## 4. Verify the software independently of robot performance

```powershell
uv run python -m cta.learning smoke-model
uv run pytest -q
```

The smoke command checks pretrained encoder inference and head gradients using a blank image. Unit tests use controlled fixtures to check training and alignment. Neither is evidence of successful learned robot control.

## 5. Next milestones

After real training, inspect imitation errors and learning curves. A single image cannot resolve every history-dependent output of a smoothed controller. If errors reflect temporal ambiguity, compare a causal sequence head. If visual features miss arm position, compare spatial patch features or selective encoder fine-tuning.

Only then add an explicit learned mode to the simulator, benchmark laptop latency, and retain all tracking, joint, speed and clearance guards before physical integration. The current `cta-mirror` still uses the classical teacher. SmolVLA autonomy and humanoid transfer remain on the longer-term roadmap.

## Tracking troubleshooting (October 18)

A readable take is not necessarily a good demonstration. The first two recorded takes contain pose glitches; keep them as diagnostic evidence and record clean takes after validating control. Close/restart the mirror app to load the tracking fix. First test without recording:

```powershell
uv run cta-mirror --robot none --control reach --hand left
```

Keep both shoulders, the tracked elbow and the hand visible; sitting farther back avoids cropped-body hallucinations. Press C at your comfortable home pose, then P for pinch. Reach mode uses shoulders and arm pose, so a hand-only closeup cannot reliably drive it. The new checks hold the command on arm identity switches or implausible shoulder/reach jumps and print TRACKING HOLD with the reason. Return to the previous view, or reframe and press C. The tolerances are initial conservative heuristics and need live validation. Resume data collection only after the movement matches your intent.

## Home-relative reach correction (October 18)

Reach mode now defaults to `--reach-mapping relative`: the first valid body pose after C becomes the human home reference, mapped to the robot's raised ready position (~15.5 cm gripper height). Raise/lower or move your arm forward/back/sideways relative to that pose to move the gripper. This preserves the robot's own IK; it does not copy human elbow angles.

Restart the app, hold a comfortable bent-arm neutral position with your full arm visible, then press C and P. The overlay now includes shoulder, elbow and wrist. `--reach-mapping absolute` retains the previous experimental mapping. New recordings save the mapping and reach home reference so training cannot silently mix them with older takes. Training collection waits until a valid home pose has been observed.

## Wrist-loss recovery behavior

Reach mode now holds every joint immediately when the wrist disappears or disagrees with the pose wrist. It requires three consistent observations before resuming. In relative mode it reanchors the recovered human position to the held robot target, so recovering a lost track should not pull the robot toward a different target. C resets the original home relationship. Relative movement is capped at 0.25 m/s and 90 degrees/second; this can make very fast gestures feel slower. Do not interpret a held robot as proof the camera is accurately tracking the wrist.
