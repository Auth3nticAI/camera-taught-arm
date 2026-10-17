# Camera-Taught Robot Arm — Design

*Working name; naming is Tray's call. Written 2026-10-16. Owner and decision-maker: Tray Branch. Implementation: Claude Code.*

## 1. Intent

A person teaches a low-cost robot arm by showing it a task with their hand in front of a camera, and the robot then performs the task on its own. Today, robot demonstrations are mostly collected with a second "leader" arm, VR headsets, gloves, or mocap rigs. This project shows that a webcam plus trained neural networks can replace them.

The project serves two audiences and two outcomes:

- **Objective (course):** an A on the PyTorch class project.
  - Due **2026-10-24**. Early demo slot **2026-10-21** if submitted earlier.
  - Course requirements: Python + PyTorch, own data, make/modify/improve a neural network, runs live, demoed and presented to the class.
  - A downloaded project applied to your own problem scores B; an A needs more work.
- **Goal (startup):** the first proof point of a company: "anyone with a camera can teach a robot."
  - Research context is in `reports/Webcam motion capture for robotics.md`.
  - Key market facts: the robot-data bottleneck; vendors charge about $28–200 per hour for teleoperation data; cheap-capture companies such as XDOF (reported ~$1.2B valuation talks) are raising large rounds.

Tray makes the engineering decisions and wants to learn the material. Every decision is logged (see §8) because it feeds two decks: one for the course and one for clients and investors.

## 2. Goals, non-goals, anti-goals

### Goals

| # | Goal | Proves |
|---|------|--------|
| G1 | **Live mirroring:** Tray's hand moves → the real SO-101 follower moves. Hand position and tilt drive the arm joints; a thumb–index pinch drives the gripper. A MuJoCo SO-101 "twin" mirrors the same commands on screen. | The wow moment; the teaching interface |
| G2 | **Learned retargeting network** (PyTorch) maps hand features → 6 joint targets. Benchmarked against a non-learned baseline on accuracy (vs. ground truth) and latency. | "Make/improve a network" (course A) |
| G3 | **Learned temporal smoother** (small GRU or TCN), benchmarked against the One Euro filter on jitter *and* lag. | Second network plus a measurable improvement |
| G4 | **Ground-truth accuracy:** per-joint error in degrees, measured against the SO-101 **leader arm's** motor encoders. | Honest numbers for both decks |
| G5 | **Dataset export:** every teaching session is saved as a LeRobotDataset (joint states, actions, robot-camera video). | The data product; feeds G6 |
| G6 | **Do it alone:** on command ("pick up the red block" / "green block"), the arm picks the right block using a policy fine-tuned on demos that were collected through G1. | Startup claim: teach-by-showing works |
| G7 | **Notes:** a decision log, lab notebook, and deck notes kept throughout. | Both decks |

### Stretch goals (only after the G1–G6 gates pass)

- **S1:** Raspberry Pi 5 runs the arm and robot camera locally (edge AI), with the PC used only for training.
- **S2:** "Find X": an open-vocabulary detector (OWLv2 or Grounding DINO) locates a named object in the robot camera, and the arm points at or reaches toward it.
- **S3:** A per-session quality score that flags jittery or badly tracked demos before they enter the dataset.

### Non-goals

- Finger-level dexterous retargeting. The SO-101 has a single gripper, not fingers.
- Full-body tracking; clinical or biomechanics-grade accuracy claims.
- GR00T or other large foundation-model fine-tuning. GR00T N1.5 needs ~33 GB of VRAM; the RTX 4070 Ti has 12 GB.
- A product UI, accounts, cloud hosting, or a crowdsourcing platform. These belong on the startup roadmap slide, not in this build.

### Anti-goals (failure even if it "works")

- It reads as **a downloaded project with tweaks**, which is the B trap. The trained networks, the comparisons, and the measured results must clearly be Tray's own work.
- **It only works on video.** It must run live in the room.
- **There is no fallback.** A single hardware failure must not sink the demo; the sim twin and recorded results are the backstop.
- **Claude makes engineering decisions silently.** Every architecture, model, data, metric, or scope choice goes to Tray and into the decision log.
- **Accuracy numbers get inflated.** Report the measured numbers as they are, including weaknesses such as depth.

## 3. Constraints

- **Time:** solo. Today is 2026-10-16.
  - **Oct 17: final exam, 4:20–5:50 PM.** It takes priority, and the arm arrives the same day.
  - Oct 21, 7:30–9:30 PM: optional early presentation slot.
  - **Oct 24: final presentation.**
  - Grades are due Oct 30.
- **Presentation format (from the instructor):** **7 minutes per student**, via screen share: "explain what you did with a few slides, run the code to show results."
  - Budget about 3 minutes of slides and 3–4 minutes of demo.
  - The demo must be a **single dashboard window** showing the human camera, a camera view of the physical arm, the sim twin, and live metrics, because the audience sees only the shared screen.
  - A recorded backup video of the full demo is required.
  - **Venue (Tray decided): online, from the laptop, away from home — the arm is not available on Oct 24.**
  - **Demo format (Tray decided):**
    - A **recorded video** shows the physical arm: live mirroring, then the autonomous pick. It must be recorded at home before Tray leaves.
    - Then Tray **runs the code live on the laptop**: laptop webcam → trained networks → MuJoCo SO-101 twin, with live joint angles, FPS, and smoother on/off.
    - The pick policy appears only in the video, plus its logged success rate.
  - **Requirement R1, no-hardware mode:** one command runs the full mirroring pipeline (camera → retargeting net → smoother → twin + metrics) with no arm, no leader, and no robot cameras connected. The trained models and configs must already be on the laptop. Rehearse it on the laptop before leaving home.
- **Compute (two machines; Tray's decision):**
  - **Desktop:** i9-13900KF, RTX 4070 Ti (12 GB), 32 GB RAM, Windows 11, Python 3.11.3, torch 2.6.0+cu124 (CUDA working). The home dev and training machine.
  - **Laptop:** MSI Raider GE76 12UH, i7-12700H, RTX 3080 Laptop GPU (VRAM to be confirmed), 64 GB RAM, Windows 11 Pro. The travel dev machine and **the demo machine**.
  - Both machines must run the same code. The environment must be reproducible (pinned dependencies), and code, models, and datasets must move between them without hand-copying. The mechanism is the builder's call; virtual environments must not live in OneDrive-synced folders.
- **Hardware (arriving 2026-10-17):**
  - Seeed SO-ARM101 Pro, assembled: a **leader + follower** pair with 12V Feetech STS3215 servos.
  - innomaker 1080p USB camera with a 130° wide-angle lens.
  - A Raspberry Pi 5 is already on hand.
  - **Human-facing camera (Tray's):** an AI PTZ 4K webcam (4K@30 / 1080p@60, auto-tracking, gesture control, 4x digital zoom, auto-focus, HDR, sleep mode, companion app). For this system it must behave as a **fixed camera**:
    - auto-tracking **off**, gesture control **off**, zoom **locked at 1x**, auto-focus off or fixed, HDR off (test), sleep **off**;
    - run at **1080p@60**;
    - save the settings as a preset in its app, and check that they survive a replug. The startup check should warn if the frame size or FPS is wrong.
    - **Reason:** absolute mapping and the hand-scale depth cue both assume the camera's view and zoom never change.
  - **Recommendation (Claude; Tray to decide):** bring this camera when traveling and use it for the live Oct 24 run, so the camera and viewpoint match the training data. The laptop's built-in webcam plus re-calibration is the fallback.
  - **To buy (Tray decided):** a small USB **wrist camera** for the gripper (~$20–30) and a ~$20 webcam to watch the human at the desktop. The laptop's built-in webcam can watch the human when traveling, but this needs checking.
- **Course rules:** PyTorch; own data; a network made or modified; live demo.
- **Licensing:** for a startup, prefer self-collected data and permissively licensed models. FreiHAND, Human3.6M, EgoDex, and MANO/SMPL are non-commercial and stay out of the product path.
- **Platform risk:** LeRobot targets Linux and macOS first, so native Windows support for Feetech motor control must be verified on day 1. Fallbacks are the Pi 5 (Linux) driving the motors, or WSL with usbipd.

## 4. System

```
 HUMAN CAMERA ──► MediaPipe Hands (21 landmarks, 3D, wrist-relative)
                      │   + depth cue (hand scale and/or Depth-Anything-V2-Small)
                      ▼
               Hand features: wrist position (x, y, depth), palm orientation, pinch distance
                      ▼
          ┌── RETARGETING NET (PyTorch, G2) ──► 6 joint targets ──┐
          │   baseline: geometric mapping + IK                     │
          └────────────────────────────────────────────────────────┘
                      ▼
               SMOOTHER NET (GRU/TCN, G3)    baseline: One Euro
                      ▼
          clamp to joint limits ──► SO-101 FOLLOWER (Feetech bus)  +  MuJoCo twin on screen
                      │
 ROBOT CAMERA (innomaker, overhead/side) ─┐
                      ▼                   ▼
               LeRobotDataset recorder (states, actions, video, task text)   (G5)
                      ▼
               POLICY fine-tune (SmolVLA or ACT) on the GPU  ──► autonomous pick on command  (G6)

 LEADER ARM: ground-truth joint angles for training labels and evaluation (G2, G4)
```

### 4.1 Teach layer (G1–G5)

- **Control mapping (Tray decided): absolute, same-side.**
  - A one-time calibration at session start: Tray holds the hand at a "home" pose.
  - After that, the hand's position inside a box in front of the camera maps to the gripper's position in the arm's workspace, like a puppet facing the same way. Hand tilt maps to wrist pitch and roll.
  - The box size, scale, and workspace limits are the builder's call and get tuned live.
  - Targets are Cartesian (gripper pose); the baseline converts them to joints with IK on the SO-101 model.
- **Input:** MediaPipe Hands on the human-facing camera.
  - MediaPipe's hand coordinates are wrist-relative and its depth is not metric, so absolute hand distance needs a depth cue (D-04).
  - Wide-angle lenses distort the image; a checkerboard calibration may be needed (D-02).
- **Retargeting network:** a small network that maps hand features to the SO-101's 5 arm joints plus the gripper. Training labels come from one of two sources (D-03):
  - **(a) Distillation:** a slow optimizer/IK solver on the SO-101 model produces joint labels from hand poses. The network learns to reproduce it faster and more smoothly.
  - **(b) Leader-arm pairing:** Tray holds the leader arm's handle while the camera watches the hand, and the leader's encoders give the true joint angles. The network learns the camera → joint mapping directly from real data.
- **Baseline = teacher (Claude's proposal; fixes an inconsistency).**
  - The non-learned baseline is the geometric mapping + IK pipeline above. It is also the "slow optimizer" whose outputs are the distillation labels in option (a).
  - G2's comparison: the network matches or beats the baseline's error against leader ground truth while running faster and producing smoother output.
- **Label shift (Claude's proposal; Tray to confirm).** In leader-paired sessions the hand grips the leader's handle; in live use it is open and pinching. So:
  - the network learns the **5 arm joints** from paired data, plus distillation labels on free-hand sessions;
  - accuracy is reported on both paired and free-hand held-out sets;
  - the **gripper** is driven by thumb–index pinch distance through a calibrated threshold mapping, not the network;
  - the decks state this caveat.
- **Smoother:** a causal GRU/TCN over a short window. It must run causally (no peeking at future frames) because control is live. Judged on both residual jitter and lag; One Euro is known to cut real motion along with the noise.
  - **Training target (Claude's proposal):** input = the retargeter's raw per-frame joint outputs; target = the leader-encoder trajectory from paired sessions.
  - If the retargeter ends up taking a time window itself, Tray decides whether to merge the two into one network.
- **Robot I/O:** LeRobot's SO-101 follower driver; the MuJoCo SO-101 model (via `mujoco` and the MuJoCo Menagerie) posed every frame for the twin.
- **Recording:** LeRobotDataset v3. Call `finalize()` at the end of every session, or the Parquet files are corrupt.

### 4.2 Do-it-alone layer (G6)

- **Task (Tray decided): pick and place in a bin.**
  - Red and green blocks are both always on the table, so the color choice matters.
  - Positions vary within a ~20×20 cm area.
  - Success = the commanded block is grasped, lifted, and dropped in the bin.
  - **Cameras:** wrist camera + innomaker overhead/front, the standard two-view SO-101 setup.
- **Data (Tray decided): webcam-taught.** ~50 episodes total (both colors, roughly balanced), recorded through Tray's own hand-mirroring pipeline on Sun Oct 18. This is the startup claim in action.
  - The Sunday ~2 PM checkpoint (§6) is Tray's decision point to switch to the leader arm if mirroring isn't good enough.
  - The deck states which method produced the training data.
- **"On command":** with ACT this means a color flag set by key or menu; with SmolVLA it means typed text. The deck uses "pick up the red block" language only if SmolVLA is the policy that runs.
- **Policy:** either fine-tune `lerobot/smolvla_base`, which takes typed language commands, or train LeRobot ACT, which has no language input and so needs a color flag or one policy per color (D-06).
  - The SmolVLA fine-tuning recipe uses batch 64; reduce the batch size for 12 GB of VRAM.
  - Community SO-101 SmolVLA checkpoints exist as references. One reports 89% grab success, but with 340 demos.
- **Evaluation:** success rate over N trials per color, with block positions varied.

### 4.3 Edge layer (S1)

The Pi 5 runs motor control and the robot camera, plus a small detector (e.g., a YOLO11n fine-tuned on the blocks). Policy inference stays on the PC unless measurements show the Pi can keep up.

## 5. How success is measured

| Metric | Applies to | How |
|--------|-----------|-----|
| Per-joint MAE (°) vs. leader encoders | G2, G4 | Held-out recording sessions |
| End-to-end latency (ms), camera → motor command | G1, G2 | Timestamps per frame |
| Throughput (FPS) | G1 | Live counter on screen |
| Jitter (e.g., mean absolute joint acceleration while the hand is still) | G3 | Hold-still segments |
| Lag (ms, cross-correlation vs. ground truth) | G3 | Moving segments |
| Pick success rate (%) per color | G6 | N ≥ 10 trials each, logged |

Every number reported in either deck comes from these logged measurements.

**Gate targets (Claude's proposal; Tray to confirm or edit).** Missing a target triggers a decision by Tray; it does not mean failure.

| Measure | Target |
|---|---|
| Live loop | ≥ 20 FPS on the laptop; camera → motor command < 150 ms |
| Retargeting | Per-joint MAE ≤ 10° on the big joints (shoulder pan/lift, elbow); wrist joints reported as measured |
| Smoother | ≥ 50% jitter reduction vs. raw, with lag ≤ One Euro's at a matched jitter level |
| Pick policy | ≥ 60% success per color over 10 trials |

## 6. Schedule and gates

**Hard deadline (Tray): everything that needs the arm is done by Monday, Oct 19.** Tray is away from home after that. Arm-dependent work = all recordings plus the video. Everything else (retraining, benchmarks, dashboard, R1, slides) can be done on the laptop Tue–Fri from recorded data.

| Date | Milestone | Gate |
|------|-----------|------|
| **Sat Oct 17** | **Exam prep first; final exam 4:20–5:50 PM.** Arm arrives. Evening: unbox, install on the desktop (LeRobot, MuJoCo, MediaPipe), calibrate motors, leader → follower teleop | Arm moves under software control |
| **Sun Oct 18 AM** | Baseline mirroring (geometric + IK) on the real arm + twin; leader-paired recordings (~30 min) + free-hand sessions; train retargeting net v1 | Mirroring runs live |
| **Sun Oct 18 ~2 PM: checkpoint** | **Tray decides:** is webcam mirroring good enough to record pick demos? | If not: Tray chooses to switch the demos to the leader arm, or to proceed anyway |
| **Sun Oct 18 PM** | Record ~50 **webcam-taught** pick episodes (both colors present); finalize the dataset; **train ACT overnight** on the desktop GPU | Dataset finalized and loads |
| **Mon Oct 19** | Policy success trials (≥10 per color, logged); **record the video** (mirroring + autonomous pick); copy all data and models to the laptop; test R1 on the laptop | Video recorded; R1 runs on the laptop |
| Tue–Thu Oct 20–22 | (away, laptop) Retrain and improve the retargeting net + smoother on recorded data; benchmarks vs. IK and One Euro; dashboard; slides; rehearse to 7 minutes. **Optional early presentation Wed Oct 21, 7:30–9:30 PM** | Full run fits in 7 minutes |
| Sat Oct 24 | Final 7-minute online presentation: video + live R1 run | — |

- **Policy (Claude's proposal, forced by the timeline):** ACT, trained overnight. SmolVLA moves to the roadmap slide.
- If the policy fails Monday's trials, the video shows the attempts plus the logged success rate, honestly, and autonomy becomes "in progress." Tray decides how to frame it.

## 7. Engineering decisions

### Made by Tray

- Option A: the hand drives the real SO-101 (not a simulated dexterous hand, not both).
- Hardware: SO-ARM101 Pro assembled pair, innomaker USB camera, Raspberry Pi 5.
- Scope: Teach layer → Do-it-alone layer → Edge layer, with the startup framing and the course-A objective.
- Process: Claude writes the code; Tray makes the engineering decisions; everything is logged.
- Control mapping: absolute, same-side (§4.1).
- Machines: develop on both; train on the desktop; **demo on the laptop** (D-08 resolved).
- Buy a wrist camera, plus a webcam to watch the human (D-02 resolved).
- Autonomy task: pick and place in a bin, both colors present, positions varied (§4.2).

### Waiting for Tray (Claude's recommendation in brackets)

| ID | Decision | Options | [Rec] |
|----|----------|---------|-------|
| D-01 | Demo-day priority if only one thing can be flawless | Live mirroring / autonomous pick | [Mirroring: it's the course core and the startup interface] |
| D-03 | Retargeting label source | (a) optimizer distillation / (b) leader-arm pairing / both | [Both: (b) for training *and* ground truth, (a) as the baseline it must beat] |
| D-04 | Hand-depth cue | Hand-scale heuristic / Depth-Anything-V2-Small / learned regressor | [Start with hand scale; add Depth Anything if error is too high] |
| D-05 | Smoother architecture | GRU / TCN / tiny transformer | [GRU: simplest causal option] |
| D-06 | Autonomy policy | SmolVLA fine-tune (language) / ACT (proven, no language) | [Start ACT as the safe path; try SmolVLA if time allows. Language is the stronger pitch] |
| D-07 | Where the motors connect | Windows PC directly / Pi 5 / WSL | [Windows first; Pi 5 if the drivers fight] |
| D-10 | Confirm the label-shift plan (§4.1), the smoother target (§4.1), and the gate targets (§5) | Accept / edit | [Accept as written] |
| D-09 | Project and brand name | — | [Tray] |

## 8. Notes system (for both decks)

All notes live in `docs/notes/` in the project folder:

- `DECISIONS.md`: one entry per decision: date, question, options, Tray's choice, why, and the outcome later.
- `LAB_NOTEBOOK.md`: dated entries covering what was tried, what broke, what was learned, the numbers, and screenshot paths.
- `DECK_NOTES.md`: two running sections. **Course deck:** problem, networks, data, results, the "beyond B" work. **Client deck:** pain point, demo, cost vs. teleop rigs, data product, roadmap, why Tray.

Claude adds entries as work happens. Tray edits freely.

## 8b. Storage (Tray decided)

- Code, docs, and notes live in a **GitHub** private repo. No OneDrive.
- Recordings, datasets, and trained models live in **Hugging Face Hub** private repos: datasets in `cta-sessions`, models in `cta-models`. These sync to the laptop, and can be made public later as a portfolio showcase.

## 8c. Course-text alignment (*Deep Learning with PyTorch*, Stevens/Antiga/Viehmann)

| Chapter | Applied as |
|---|---|
| 2 Pretrained networks | Pretrained ResNet-18 image encoder inside ACT; MediaPipe as a pretrained perception model |
| 3 Tensors (GPU, NumPy interop, serialization) | Landmark tensors (N, 21, 3); `torch.from_numpy` from the numpy core; GPU batches |
| 4.4 Time series | Sliding windows of joint angles as the GRU smoother's input |
| 5 Mechanics of learning | Input standardization from training-set statistics; train/val; overfitting checks |
| 6 Fitting with a neural network | Retargeting MLP (`nn.Sequential`, MSE regression) |
| 7 Dataset / DataLoader | `SessionDataset` over recorded sessions |
| 8 nn.Module, saving, GPU, width/depth/regularization | Model classes, `state_dict` checkpoints, a width/depth/dropout comparison table |
| 10 Train/val split | Held out **by take**, not by frame, because adjacent frames leak (an improvement on the book's per-sample stride) |
| 11 Training app + TensorBoard | `TrainingApp`-style scripts (`initModel`, `initOptimizer`, `computeBatchLoss`, `logMetrics`); TensorBoard curves for the slides |
| 12 Metrics, overfitting, augmentation, balance | Landmark augmentation (rotation/scale/noise); noise-injected pairs for the smoother; balanced red/green episodes |
| 13 Second model, Adam | Retargeter + smoother as two models; Adam |
| 15 Deployment (tracing, ONNX, quantization) | Pi 5 edge layer via ONNX Runtime; the book notes it is "quite a bit faster on the Raspberry Pi than running PyTorch directly" |

Plan 2 and Plan 3 implement these idioms. The course deck includes a slide mapping chapters to project parts.

## 9. Risks and fallbacks

| Risk | Fallback |
|------|----------|
| Feetech/LeRobot motor control on native Windows | Pi 5 (Linux) or WSL with usbipd |
| Hand depth from one camera is unreliable | Depth cue (D-04); report the error honestly; limit the reach workspace |
| Webcam-taught demos too noisy for the policy | Collect with the leader arm instead; still frame webcam teaching as the roadmap |
| Policy fails live | Show a recorded success clip plus the logged success rate; live demo limited to G1–G4 |
| PTZ camera's auto-tracking, gesture, or zoom features move or rescale the view mid-session | Disable them (§3); startup check on frame size and FPS; never touch the zoom after calibration |
| Classroom lighting or camera differences | Rehearse in the room if possible; fixed camera mount; recorded backup |
| A new MediaPipe release breaks things (1.1.1 was released 2026-10-14) | Pin known-good versions on day 1 |

## 10. Left to the builder

Code structure, module and file layout, logging format, the exact feature-vector layout, hyperparameter sweeps, plotting, and test scaffolding. Anything in this list that turns out to change behavior, metrics, or scope goes to Tray.
