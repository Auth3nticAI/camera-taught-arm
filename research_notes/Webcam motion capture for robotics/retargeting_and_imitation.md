# Learning Robot Motion from Human Video / Motion Capture: Retargeting, Camera Teleoperation, and Imitation Learning

Researched 2026-10-16. Roughly 16 tool calls. Primary sources are arXiv, PMLR, GitHub READMEs, and HF docs. Where only a secondary summary was available, that is flagged.

## Q1. Notable systems that turn human video or camera-tracked hands/bodies into robot actions

### Takeaway
Since 2023 the field has settled on one pipeline: an off-the-shelf pose estimator (MediaPipe for webcam hands, HaMeR for higher-quality monocular hand mesh, SMPL/SMPL-X for bodies) feeds a retargeter, which feeds either a robot controller (teleoperation) or a dataset for imitation learning. Plain RGB is enough for finger joints. Wrist/body 3D position from monocular RGB is the weak point, so many systems add depth, VR headsets (Vision Pro), Aria glasses, or mocap gloves.

### Cited Findings

**Webcam / monocular-RGB capable systems (most relevant to a webcam-only student)**
- **AnyTeleop (RSS 2023, Qin et al., UCSD).** Input is one or more RGB or RGB-D cameras, with no extrinsic calibration needed. It uses MediaPipe for 21 hand keypoints (3D in the wrist frame plus 2D in the image). With RGB only, wrist position comes from a network predicting weak-perspective scale (adapted from FrankMocap). With RGB-D, it uses depth lookup plus PnP. Output is arm plus dexterous hand across many robots and simulators. — [AnyTeleop arXiv HTML](https://arxiv.org/html/2307.04577v3)
- The open-source code is the `dex-retargeting` library (`pip install dex_retargeting`, dexsuite GitHub), which is "derived from the AnyTeleop project." It includes a human-hand-video teleop example (`example/vector_retargeting`) and a dataset post-processing example (`example/position_retargeting`). Robot hand URDFs come from `dex-urdf`. — [dex-retargeting README](https://raw.githubusercontent.com/dexsuite/dex-retargeting/HEAD/README.md); [PyPI](https://pypi.org/project/dex-retargeting)
- A third-party paper (TelePreview, Dec 2024) reported difficulty reproducing AnyTeleop, OpenTeach, and OpenTelevision on a LEAP Hand setup. — [TelePreview arXiv](https://arxiv.org/pdf/2412.13548) (per search snippet)
- **H2O (2024, CMU/LeCAR).** A "real-time whole-body teleoperation system using just an RGB camera" for a humanoid. It is RL-based. — [TechXplore on H2O](https://techxplore.com/news/2024-04-scalable-learningbased-framework-teleoperation-humanoid.amp)
- **OmniH2O (CoRL 2024).** It uses kinematic pose as a "universal control interface," with VR headset, verbal instruction, or RGB camera input, for a full-size humanoid with dexterous hands. It also learns autonomy from teleoperated demos. — [OmniH2O PMLR](https://proceedings.mlr.press/v270/he25b.html); [arXiv 2406.08858](https://arxiv.org/html/2406.08858v1). A secondary AI summary says it uses Unitree H1 and AMASS-based training. That is unverified from the primary text: [Moonlight review](https://themoonlight.io/review/omnih2o-universal-and-dexterous-human-to-humanoid-whole-body-teleoperation-and-learning)
- **HumanPlus (CoRL 2024, Stanford, Fu/Zhao/Wu/Wetzstein/Finn).** A single RGB camera estimates body and hand poses. HaMeR handles hands. A low-level RL policy trained in sim on 40 hours of human motion data tracks the poses in real time on a 33-DoF Unitree-H1-based humanoid (19-DoF body, two 6-DoF hands, two 1-DoF wrists). Shadowing collects demos, and behavior cloning reaches 60–100% success with up to 40 demos. The code is reported to be open source. — [HumanPlus arXiv](https://arxiv.org/pdf/2406.10454); [PMLR](https://proceedings.mlr.press/v270/fu25a.html); [CircuitDigest](https://circuitdigest.com/news/human-plus-the-open-source-humanoid-robot-shadowing-human-motions)
- **Phantom (CoRL 2025, Lepert/Fang/Bohg, Stanford).** It trains policies from human videos only, with no robot data. HaMeR hand pose provides action labels. The human arm is inpainted out and a rendered robot is overlaid. It trains Diffusion Policy and deploys zero-shot on Franka and Kinova at up to 92% success (deformable manipulation, sweeping, insertion). It compares against EgoMimic-style arm masking. — [Phantom arXiv 2503.00779](https://arxiv.org/pdf/2503.00779); [PMLR](https://proceedings.mlr.press/v305/lepert25a.html)
- **DexMV (ECCV 2022, Qin et al.).** It records human video, runs hand/object pose estimation, then retargets the motion into robot-hand demonstrations for imitation learning or RL in sim. The code is open (`yzqin/dexmv-sim` holds the env and retargeting; `dexmv-learn` holds the learning). — [GitHub dexmv-sim](https://github.com/yzqin/dexmv-sim); [arXiv](https://arxiv.org/html/2108.05877v5)

**Systems needing extra hardware (VR, glasses, gloves, exoskeleton)**
- **Open-TeleVision (CoRL 2024, Cheng/Li/Yang/Yang/Xiaolong Wang).** Uses Apple Vision Pro (WebXR) hand and head tracking with a ZED stereo camera for immersive feedback. Trained ACT-based imitation policies on 4 tasks across 2 humanoids. Code: github.com/OpenTeleVision/TeleVision (a mirror lists Apache-2.0; unverified). — [arXiv 2407.01512](https://arxiv.org/abs/2407.01512v2); [GitHub](https://github.com/OpenTeleVision/TeleVision)
- **Bunny-VisionPro (2024, UCSD).** Real-time bimanual dexterous teleoperation from Vision Pro. Its retargeting supports loop-joint hands (e.g., Ability hand). Code builds on VisionProTeleop, dex-retargeting, and sim-web-visualizer. — [arXiv 2407.03162](https://arxiv.org/pdf/2407.03162); [GitHub](https://github.com/Dingry/BunnyVisionPro)
- **EgoMimic (2024, Georgia Tech + Meta).** Uses Project Aria glasses (75 g) for egocentric video plus 3D hand tracking, and co-trains one policy on human and robot data. Human data improves performance by 34–228% over robot-only data. A third-party "400%" claim conflicts with the paper. — [EgoMimic arXiv 2410.24221](https://arxiv.org/html/2410.24221v1); [Meta AI blog](https://ai.meta.com/blog/egomimic-project-aria-georgia-tech-ego4d-robotics-embodied-ai/)
- **DexCap (2024, Stanford).** A portable mocap system using SLAM plus electromagnetic-field gloves for occlusion-robust wrist and finger tracking, along with 3D environment observation. It retargets to the LEAP hand by dropping the pinky and running IK to match fingertip positions. — [DexCap site](https://dex-cap.github.io/); [arXiv 2403.07788](https://arxiv.org/html/2403.07788v1)
- **ACE (CoRL 2024, UCSD).** A low-cost cross-platform "visual exoskeleton." A hand-facing camera gives 3D finger pose and the exoskeleton gives wrist pose. Works with humanoid hands, arm-hand, arm-gripper, and quadruped-gripper setups. **ACE-F** (Nov 2025) adds a foldable design and force feedback, and notes that IK-based Cartesian control generalizes across embodiments while joint-copying leader arms do not. — [ACE arXiv 2408.11805](https://arxiv.org/html/2408.11805v1); [PMLR](https://proceedings.mlr.press/v270/yang25d.html); [ACE-F arXiv 2511.20887](https://arxiv.org/pdf/2511.20887)

**Body retargeting tooling**
- **GMR – General Motion Retargeting ("Retargeting Matters," Stanford, Oct 2025).** Real-time, MIT-licensed. Retargets human motion (SMPL-X from AMASS/OMOMO, BVH from LAFAN1, FBX per a mirror) to many humanoids. Designed for whole-body teleop (used in TWIST) and RL tracking. It addresses foot sliding, self-penetration, and physically infeasible motion. First release 2025-08-04. — [arXiv 2510.02252](https://arxiv.org/html/2510.02252v1); GitHub mirrors e.g. [ryanrudes/GMR](https://github.com/ryanrudes/GMR)

**Curated list**
- The awesome list of robot learning from human videos (IRMVLab) indexes MimicPlay (CoRL 2023), Vid2Robot (RSS 2024, arXiv 2403.12943), and others, with code links. — [awesome-robot-learning-from-human-videos](https://github.com/IRMVLab/awesome-robot-learning-from-human-videos)

### Inferences
- Summary table. Rows marked * were not verified in this session's sources.

| System | Year | Input sensor | What's retargeted | Output robot | Open code |
|---|---|---|---|---|---|
| AnyTeleop / dex-retargeting | 2023 | RGB or RGB-D webcam(s) | 21 hand keypoints + wrist pose | Many arms + dex hands | Yes (dex-retargeting) |
| DexMV | 2022 | Multi-cam video | Hand + object pose | Sim dex hand | Yes |
| H2O / OmniH2O | 2024 | RGB cam / VR | Full body (+hands in Omni) | Humanoid | Not verified here |
| HumanPlus | 2024 | Single RGB | Body + hands (HaMeR) | Unitree H1 33-DoF | Reported yes |
| Phantom | 2025 | RGB(-D) human video | Hand pose -> EE pose + gripper | Franka, Kinova | Not verified here |
| EgoMimic | 2024 | Aria glasses | 3D hand pose | Bimanual arms | Reported open source (Meta blog) |
| DexCap | 2024 | EM gloves + SLAM | Wrist + fingertips | LEAP hand | Not verified here |
| Open-TeleVision | 2024 | Vision Pro + ZED | Head, wrists, hands | 2 humanoids | Yes |
| Bunny-VisionPro | 2024 | Vision Pro | Bimanual hands + wrists | Arms + dex hands | Yes |
| ACE | 2024 | Hand cam + exoskeleton | Fingers + wrist | Multi-platform | Not verified here |
| GMR | 2025 | Mocap files (SMPL-X/BVH) | Full body | Many humanoids | Yes (MIT) |
| MimicPlay* / Vid2Robot* | 2023/2024 | Human play video / video prompt | Latent plans / video-conditioned policy | Arms | Listed in awesome list |

- The webcam-only reachable stack is MediaPipe Hands/Pose (or HaMeR on GPU) -> dex-retargeting or a custom IK -> a simulated robot (SAPIEN/MuJoCo/PyBullet). This is exactly what AnyTeleop's RGB-only mode does.
- Industry is moving toward egocentric human data at scale (EgoMimic with Meta Aria, Phantom-style "robot-free" data). This supports a "human video as robot data" framing for a student project.

### Gaps
- I did not fetch primary pages for MimicPlay (Stanford/NVIDIA, CoRL 2023) or Vid2Robot (Google DeepMind, RSS 2024), so their sensor and output details aren't verified here.
- I did not confirm the code release status of H2O/OmniH2O, Phantom, DexCap, or ACE in this session.
- Company efforts (Tesla Optimus video learning, Figure, 1X, Physical Intelligence human-video data) were not researched. I found no primary sources in this pass.

## Q2. How joint retargeting works technically, and its failure modes

### Takeaway
There are three families. (1) Direct kinematic mapping: copy joint angles, or IK to match fingertip/wrist positions (DexCap, ACE-F's IK point). (2) Per-frame optimization: minimize keypoint-vector error with a scale factor and a smoothness term under joint limits (AnyTeleop/dex-retargeting; GMR for bodies). (3) Learned networks that amortize the optimization or learn a cross-embodiment latent space, which give 1 kHz inference and no test-time solve (GeoRT, IKMR, ImitationNet). The main failure modes are monocular depth/scale error, self-occlusion, fast-motion tracking loss, morphology mismatch, and physically infeasible output (foot sliding, self-penetration).

### Cited Findings
- **AnyTeleop vector retargeting objective:** `min_{q_t} Σ_i ||α·v_i^t − f_i(q_t)||² + β||q_t − q_{t−1}||²` subject to `q_l ≤ q_t ≤ q_u`. Here v_i are human keypoint vectors (e.g., wrist-to-fingertip), f_i is the corresponding robot keypoint vector from forward kinematics, α scales for hand size, and β gives temporal smoothness. Non-humanoid hands (DClaw) need a manual keypoint correspondence. — [AnyTeleop](https://arxiv.org/html/2307.04577v3)
- dex-retargeting implements several optimizers: vector retargeting, `PositionOptimizer` (from "From One Hand to Multiple Hands"), and `DexPilotOptimizer` (from DexPilot). It uses pinocchio for kinematics. — [dex-retargeting README](https://raw.githubusercontent.com/dexsuite/dex-retargeting/HEAD/README.md)
- AnyTeleop runtime: hand detection takes about 26–35 ms per frame (capped at 25 Hz) and retargeting about 9–10 ms. Motion generation targets 120 Hz. — [AnyTeleop](https://arxiv.org/html/2307.04577v3)
- **Kinematic/IK mapping:** DexCap drops the pinky and solves IK to match fingertip positions on the 4-finger LEAP hand. — [DexCap arXiv](https://arxiv.org/html/2403.07788v1). ACE-F notes that IK Cartesian control generalizes across embodiments, while direct joint-copying needs hardware rebuilt per robot. — [ACE-F](https://arxiv.org/pdf/2511.20887)
- **Learned retargeting:**
  - **GeoRT (Geometric Retargeting, Mar 2025).** A neural hand retargeter trained unsupervised, with no paired human-robot data. It maps human fingertip keypoints to robot hand at 1 kHz with no test-time optimization. Its losses cover motion fidelity, C-space coverage, uniform response ("flatness"), pinch correspondence, and self-collision. Project page zhaohengyin.github.io/geort. Code release not confirmed. — [arXiv 2503.07541](https://arxiv.org/abs/2503.07541)
  - **IKMR (Sep 2025).** A learned kinodynamic retargeter for humanoids: a pretrained motion-topology representation plus a dual encoder-decoder that learns a motion domain mapping. — [arXiv 2509.15443](https://arxiv.org/html/2509.15443v1)
  - **ImitationNet (2023).** Human-to-robot pose retargeting without paired data, via a shared latent space. — [arXiv 2309.05310](https://arxiv.org/html/2309.05310v3)
  - **Learning to Transfer Human Hand Skills (Jan 2025).** Learns a joint motion manifold over human hand, robot hand, and object motion. — [arXiv 2501.04169](https://arxiv.org/abs/2501.04169v1)
  - **UMR (arXiv 2609.02134, appears to be Sep 2026).** Unified humanoid retargeting via learned point-cloud correspondence. Date not verified. — [arXiv](https://arxiv.org/pdf/2609.02134)
- Newer non-learned hand retargeters keep appearing, e.g. "Smooth Operator" (sampling-based, 2026) and "DexTwist" (2026). — [Smooth Operator](https://arxiv.org/pdf/2607.07491); [DexTwist](https://arxiv.org/pdf/2605.12182) (titles from search results only; not read)
- **Failure modes (documented):**
  - **Depth / monocular ambiguity.** With AnyTeleop's piano task, single RGB gave 28.1% error vs 21.8% for single RGB-D and 12.5% for two RGB-D cameras. RGB-only wrist position error is explicitly larger. — [AnyTeleop](https://arxiv.org/html/2307.04577v3)
  - **Self-occlusion.** Pose is "unreliable when the hand is in self-occlusion" (e.g., palm perpendicular to the camera). — [AnyTeleop](https://arxiv.org/html/2307.04577v3). DexCap uses EM gloves specifically for occlusion robustness. — [DexCap](https://dex-cap.github.io/)
  - **Fast motion.** Tracking is lost and must pause and re-detect. — [AnyTeleop](https://arxiv.org/html/2307.04577v3)
  - **Jitter.** AnyTeleop handles it only with the β smoothness term and does not quantify it. — [AnyTeleop](https://arxiv.org/html/2307.04577v3)
  - **Embodiment/morphology gap.** Different finger counts (pinky dropped) and non-humanoid hands need manual mapping. — [DexCap](https://arxiv.org/html/2403.07788v1); [AnyTeleop](https://arxiv.org/html/2307.04577v3)
  - **Physical infeasibility in body retargeting.** Foot sliding, self-penetration, and infeasible motion hurt downstream RL tracking. This is the motivation for GMR. — [GMR arXiv](https://arxiv.org/html/2510.02252v1)
  - **Visual embodiment gap for policies.** The human arm/hand in images differs from the robot's, so Phantom inpaints the human and renders the robot, and EgoMimic masks the arm. Phantom's ablations show that image-edit quality matters. — [Phantom](https://arxiv.org/pdf/2503.00779)
  - **Reproducibility across hardware.** — [TelePreview](https://arxiv.org/pdf/2412.13548)

### Inferences
- Optimization retargeting (AnyTeleop) costs about 10 ms per frame on CPU and is the natural "teacher." A student can generate (MediaPipe landmarks -> optimized joint angles) pairs offline with dex-retargeting, then train a small MLP to amortize it, as GeoRT does with its geometric losses. This is a well-scoped "learned retargeting" project.
- Scale mismatch is handled by a single α in AnyTeleop. Per-user hand-size calibration is a simple, under-explored improvement.

### Gaps
- I didn't find a quantitative jitter benchmark for MediaPipe-based teleop.
- I didn't verify GeoRT's code release or which robot hands it was tested on.

## Q3. What data format robot learning pipelines expect

### Takeaway
Policies (ACT, Diffusion Policy, etc.) consume time-indexed pairs of (observation: images plus proprioceptive state vector) and (action: target joint positions, or end-effector pose deltas plus gripper) at a fixed FPS. LeRobotDataset v3.0 is the de facto open format: Parquet for state/action, MP4 for cameras, and JSON/Parquet metadata.

### Cited Findings
- LeRobotDataset v3.0 is in `lerobot >= 0.4.0`. It has three pillars: Parquet for low-dimensional signals (states, actions, timestamps), MP4 per camera, and metadata (schema, FPS, normalization stats, episode offsets). Many episodes are stored per file, and episode boundaries come from metadata. — [LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- Layout: `meta/info.json` (features, shapes, dtypes, FPS, path templates), `meta/stats.json` (mean/std/min/max for normalization), `meta/tasks.jsonl` (natural-language tasks), `meta/episodes/` (chunked Parquet), `data/` (Parquet shards), `videos/` (MP4 shards). — [LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- A loaded sample is a dict of torch tensors: `observation.state`, `action`, `observation.images.<cam>` [C,H,W], `timestamp`. `delta_timestamps` returns temporal windows. It works with `torch.utils.data.DataLoader`. — [LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- Creation API: `LeRobotDataset.create(...)`, then `dataset.add_frame(frame)` per frame, `dataset.save_episode()` per episode, then `dataset.finalize()` (required, or the Parquet files are corrupt), then `push_to_hub()`. — [LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- The meaning of state and action is defined per dataset. Example (a LIBERO conversion): state is absolute EE pose (axis-angle converted to quaternion); action is delta position plus delta quaternion plus absolute gripper. — [HF dataset GT-111/libero_v3_eef](https://huggingface.co/datasets/GT-111/libero_v3_eef)
- The stock recording flow pairs a leader-arm teleop with a follower (e.g., SO-101) and OpenCV cameras via `lerobot-record`. — [LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- Phantom converts hand pose into robot action labels (EE pose plus gripper) for Diffusion Policy. — [Phantom](https://arxiv.org/pdf/2503.00779)

### Inferences
- A webcam project could output a valid LeRobotDataset: `observation.images.webcam`, `observation.state` = retargeted robot joint angles at t, and `action` = joint angles at t+1 (or EE pose delta). This makes the output "robot-usable" even without hardware, and it is a concrete deliverable to demo.
- A low-cost SO-100/SO-101 arm (6-DoF incl. gripper) is the LeRobot reference robot. Simulating it is a natural webcam-to-robot target if no hardware is available.

### Gaps
- I did not verify the exact SO-101 joint naming in `info.json`, or whether LeRobot ships an official human-hand-video-to-dataset converter.

## Q4. Open problems a student could attack with a small PyTorch net in a few days

### Takeaway
The most tractable, demo-able projects sit in the gap between noisy monocular landmarks and clean, feasible robot joint commands. Four candidates: (a) amortized/learned retargeting (an MLP from 21 MediaPipe landmarks to robot hand joints, supervised by dex-retargeting); (b) temporal denoising/jitter filtering (a small GRU/1D-conv vs One-Euro/Kalman baselines); (c) monocular depth/wrist-pose correction; (d) action segmentation or gesture-to-primitive classification. All run live on a webcam with a simulated robot.

### Cited Findings
- Learned retargeting is an active 2025–2026 research topic: GeoRT reports 1 kHz unsupervised neural retargeting, and there are IKMR, UMR, and "Learning to Transfer Human Hand Skills." — [GeoRT](https://arxiv.org/abs/2503.07541); [IKMR](https://arxiv.org/html/2509.15443v1); [UMR](https://arxiv.org/pdf/2609.02134); [2501.04169](https://arxiv.org/abs/2501.04169v1)
- Optimization retargeting costs about 9–10 ms per frame and hand detection about 26–35 ms per frame. Speed and smoothness are bottlenecks for 120 Hz control. — [AnyTeleop](https://arxiv.org/html/2307.04577v3)
- RGB-only wrist depth is notably worse than RGB-D (28.1% vs 21.8% error in one task), so monocular 3D wrist estimation is an open weakness. — [AnyTeleop](https://arxiv.org/html/2307.04577v3)
- Jitter is handled only by a smoothness penalty, with no learned filtering, in AnyTeleop. — [AnyTeleop](https://arxiv.org/html/2307.04577v3)
- Self-collision and pinch correspondence are explicit loss terms in GeoRT. These are easy-to-implement differentiable losses for a student MLP. — [GeoRT](https://arxiv.org/abs/2503.07541)
- Unpaired human-to-robot mapping (no paired data) is an open direction (ImitationNet, GeoRT). — [ImitationNet](https://arxiv.org/html/2309.05310v3)
- Physical feasibility of retargeted body motion (foot sliding, penetration) remains a problem even in 2025. — [GMR](https://arxiv.org/html/2510.02252v1)

### Inferences (candidate project ideas, ranked for 8 days / webcam only)
1. **"Neural dex-retargeting" (best fit).** Record about 20–60 minutes of webcam MediaPipe hand landmarks, or use public hand datasets. Label each frame with dex-retargeting's optimizer for a hand URDF (e.g., Allegro/LEAP/Inspire from dex-urdf). Train an MLP (63 -> 256 -> 256 -> 16 joints) with joint-MSE plus a forward-kinematics fingertip loss (optionally a pinch loss, GeoRT-style). Measure speed against the optimizer, and accuracy and smoothness against ground truth. Demo it live: webcam to robot hand rendered in PyBullet/MuJoCo/SAPIEN.
2. **Learned temporal filter.** Train a causal GRU/TCN to denoise landmark or joint sequences. Supervise with smoothed offline trajectories or synthetic noise injection, and compare against a One-Euro filter. Metrics: jerk vs lag. This can be combined with #1.
3. **Monocular wrist-depth regressor.** Predict wrist z or palm scale from 2D landmarks plus bounding box. Train on a dataset with 3D labels (e.g., FreiHAND/InterHand-style) to improve RGB-only teleop. This is a riskier data requirement.
4. **Gesture/action segmentation to robot primitives.** Classify landmark windows into grasp/release/move phases to drive the gripper state. This is the "action" field of a LeRobot dataset.
5. **Webcam to LeRobotDataset exporter** as the glue for any of the above. Write episodes with `add_frame`/`save_episode`/`finalize`, then optionally train a tiny ACT/Diffusion policy in sim. This is probably too heavy for 8 days, except as a stretch goal.
- Key risks: MediaPipe 3D landmarks are wrist-relative and scale-normalized, so absolute arm position from a webcam is hard. Scope the project to hand/finger retargeting, or to wrist orientation plus coarse position.

### Gaps
- I didn't find published benchmarks comparing learned vs optimization retargeting on MediaPipe webcam input specifically. That makes the comparison a genuine (small) contribution, but I couldn't confirm novelty exhaustively.
- I didn't verify which public datasets give paired monocular RGB with accurate 3D hand joints under a permissive license, beyond knowing they exist.
