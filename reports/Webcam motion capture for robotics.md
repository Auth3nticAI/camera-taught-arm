# Teach a Robot Hand From Your Webcam

The strongest project for an 8-day, laptop-only PyTorch deadline is **"neural retargeting"**. MediaPipe reads your hand from the webcam, a small PyTorch network you train turns its 21 landmarks into joint angles for a real robot hand model (LEAP, Allegro or Shadow), and a MuJoCo simulation mirrors your fingers live in front of the class. The network learns to imitate a slow optimization-based retargeter, the published AnyTeleop method that underlies the open `dex-retargeting` library. That gives you a clean before-and-after story (speed, smoothness, accuracy) and puts you on a research thread that is active right now (GeoRT, IKMR, ImitationNet). The second-best option is a webcam-to-arm mirror that drives the $100–$230 SO-101 arm in simulation and writes the result as a LeRobot dataset. The third is a learned temporal denoiser, which is the safest option and can be added to either of the first two. All ten ideas below share one honest constraint. A single webcam measures big joints such as the elbow and knee to within a few degrees, but finger angles to only about 11–22°, and its depth output is not metric. The project should therefore be framed around **retargeting and smoothing that robots can use**, not around clinical-grade measurement. The startup case is real, because human-motion data for robots became a venture category in 2026. Mecka AI raised $60M from Sequoia, XDOF is reportedly raising at $1.2B, and Figure has committed over $1B to data and compute. The open opening for a solo builder is the software layer: getting from video to trajectory, checking quality, and converting formats. Raw collection volume is already taken.

## Every idea runs on one CPU-friendly pipeline

All ten projects share the same backbone, and that is what makes 8 days enough. **MediaPipe 1.1.1 shipped on Oct 14, 2026 with a native `win_amd64` wheel for Python 3.10–3.13** ([PyPI mediapipe](https://pypi.org/project/mediapipe/#files)). Its Hand Landmarker runs in about **17 ms per frame on a phone CPU**, and in video or live-stream mode it skips palm detection while tracking holds ([Google MediaPipe](https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker)). For the robot side, **MuJoCo Menagerie now installs with pip (`pip install mujoco-menagerie`)**. It includes ready models of the LEAP Hand (16 DoF), Allegro (16 DoF), Shadow Hand (24 DoF), the SO-101 arm (6 DoF), Franka Panda and the Unitree G1 humanoid (29 DoF) ([MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)). For a mirror demo you only need to set joint positions and recompute poses (`qpos` and `mj_forward`, no physics stepping), and that runs comfortably in real time on a CPU. Rerun can show the robot model (loaded from its URDF file) and live joint-angle plots in one window, and it is the same viewer LeRobot uses ([Rerun URDF how-to](https://rerun.io/docs/howto/urdf); [LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)).

The other simulators are worse fits on a Windows laptop. **PyBullet has no Windows wheels on PyPI**, so pip falls back to an 80 MB source build ([PyPI pybullet](https://pypi.org/project/pybullet/#files)). **Isaac Sim needs at least an RTX 4080 with 16 GB of VRAM** ([Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html)). Genesis officially supports CPU simulation on Windows, but it is newer and heavier ([Genesis docs](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)).

The PyTorch component is cheap at inference time because the inputs are landmark vectors, not images. That means about 63 numbers per frame for a hand and 99 for a body. A small MLP or GRU adds well under a millisecond, so the frame budget is set by MediaPipe. For the "output robots can use" requirement, **LeRobotDataset v3** is the de facto open format: Parquet files for joint states and actions, MP4 for camera video, and metadata files, all loadable as PyTorch tensors. You create it with `create`, call `add_frame` for each frame and `save_episode` for each episode, and must call `finalize()` at the end, or the Parquet files are corrupt ([LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)).

The field has converged on this exact architecture. **AnyTeleop (RSS 2023) runs MediaPipe on plain RGB cameras, retargets 21 hand keypoints, and drives many arms and dexterous hands** ([AnyTeleop](https://arxiv.org/html/2307.04577v3)). HumanPlus (CoRL 2024) uses a single RGB camera to puppet a 33-DoF humanoid ([HumanPlus](https://arxiv.org/pdf/2406.10454)). Phantom (CoRL 2025) trains robot policies from human video alone and reaches up to 92% success on Franka and Kinova arms ([Phantom](https://arxiv.org/pdf/2503.00779)). A webcam project is therefore a faithful miniature of state-of-the-art systems, not a toy.

## Ten projects, from safest to most ambitious

The table compares the ten ideas, and the subsections below give the details. Difficulty is rated for one student over 8 days with a laptop CPU, plus optional free Colab GPU time.

| # | Project | Network trained | Data source | Difficulty (8 days) | Wow |
|---|---|---|---|---|---|
| 1 | Neural hand retargeting | MLP: landmarks → robot-hand joints | Self-recorded webcam, labeled by `dex-retargeting` | Medium | Very high |
| 2 | Arm mirror to SO-101 + dataset export | MLP neural IK: pose → 6 arm joints | Synthetic pairs from MuJoCo forward kinematics | Medium | High |
| 3 | Learned jitter filter | Causal GRU/TCN denoiser | Self-recorded sequences + synthetic noise | Easy–Medium | Medium (high with side-by-side) |
| 4 | "Which joint is moving" detector | Multi-label GRU classifier | Self-labeled webcam clips | Easy | Medium–High |
| 5 | Gesture-to-gripper primitives | GRU/1D-conv phase classifier | Self-recorded labeled gestures | Easy–Medium | High |
| 6 | Distilled finger-angle accuracy | MLP: MediaPipe → MANO-style angles | Offline WiLoR/HaMeR pseudo-labels (Colab GPU) | Medium–Hard | Medium |
| 7 | Demonstration quality scorer | Clip-quality classifier/regressor | Self-recorded good/bad clips | Medium | Medium |
| 8 | Monocular wrist-depth regressor | MLP: 2D landmarks + box → wrist depth | FreiHAND 3D labels | Hard | High if it works |
| 9 | Upper-body humanoid shadowing | 2D→3D lifter (Martinez MLP) | AMASS-synthesized or Human3.6M pairs | Hard | Very high |
| 10 | Watch-then-imitate behavior cloning | Tiny behavior-cloning policy | Webcam demos exported to LeRobotDataset | Hard | Very high |

### 1. Neural hand retargeting: your fingers drive a LEAP Hand in real time

**Pitch:** a PyTorch network replaces a slow optimizer and turns webcam hand landmarks into dexterous-robot joint angles in microseconds. The labels come from the established retargeter. AnyTeleop's vector retargeting solves `min Σ‖α·v_i − f_i(q)‖² + β‖q_t − q_{t−1}‖²` within joint limits. Here `v_i` are human keypoint vectors (for example, wrist to fingertip), `f_i(q)` are the matching robot vectors computed from the robot's joint angles `q`, α corrects for hand size, and the β term keeps motion smooth ([AnyTeleop](https://arxiv.org/html/2307.04577v3)). The open implementation is `pip install dex_retargeting` (MIT license, about 1.3k stars), which ships vector, position and DexPilot optimizers ([dex-retargeting](https://github.com/dexsuite/dex-retargeting)).

**Network and data:** record 20–60 minutes of your own hand through MediaPipe, label every frame offline with the optimizer, and train an MLP that maps 63 inputs through two hidden layers of 256 to 16 joint outputs. The loss is joint MSE plus a fingertip-position term computed from the robot's forward kinematics. **GeoRT (2025) shows the payoff: a neural hand retargeter running at 1 kHz with no test-time optimization**, using geometric losses (pinch correspondence, self-collision) that a student can copy ([GeoRT](https://arxiv.org/abs/2503.07541)). AnyTeleop's optimizer costs **about 9–10 ms per frame** on top of 26–35 ms of hand detection ([AnyTeleop](https://arxiv.org/html/2307.04577v3)), so you can report a concrete speedup. No published benchmark compares learned and optimized retargeting on MediaPipe webcam input specifically, which makes the comparison a small but genuine contribution.

**Live demo:** the webcam feed with the skeleton overlaid sits beside a MuJoCo LEAP Hand that copies your pinches, fists and finger counting, next to a live plot of the 16 joint angles. A toggle lets the class watch optimizer and network outputs side by side.

**Feasibility: Medium.** The pipeline exists. The two pieces of work are matching joint names between the hand models `dex-retargeting` uses (from its dex-urdf collection) and the Menagerie MuJoCo models, and skipping SAPIEN, the renderer in `dex-retargeting`'s examples, whose Windows support is unverified. If the optimizer fights you, fall back to computing finger-flexion angles directly from landmark vectors and using those as labels.

**Startup angle:** "video to robot-hand trajectory" is exactly the software layer companies are buying. **Agile Robots announced in Oct 2026 that it turns ordinary worker videos into trajectories without teleoperation rigs** ([Crypto Briefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/)). Apple's EgoDex, the best open hand-pose dataset, is licensed CC-BY-NC-ND and cannot be used commercially ([EgoDex](https://github.com/apple/ml-egodex)), which leaves room for commercially usable hand-retargeting pipelines.

### 2. Webcam arm mirror to SO-101, saved as a LeRobot dataset

**Pitch:** raise your arm and a simulated SO-101, the standard cheap LeRobot arm, copies your shoulder, elbow, wrist and gripper. Every second is recorded as a robot-ready dataset.

**Network and data:** train a **neural inverse-kinematics (IK) MLP** on synthetic data you generate yourself. Sample random SO-101 joint configurations in MuJoCo, compute the resulting elbow, wrist and gripper positions with forward kinematics, normalize them the way you will normalize MediaPipe Pose landmarks 11–16 (shoulders, elbows, wrists), and learn the inverse mapping. The gripper opening comes from the distance between thumb and index finger. Generating data is unlimited and free, and nothing needs a license. The demo then writes `observation.state`, `action`, `observation.images.webcam` and the landmarks into a LeRobotDataset ([LeRobot v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)).

**Live demo:** you wave, point and "grab". The MuJoCo SO-101 follows, and at the end you open the recorded dataset in the Rerun viewer or push it to the Hugging Face Hub live.

**Feasibility: Medium.** No polished open-source webcam-to-MuJoCo arm repo exists, so you write the arm mapping yourself. The research notes did not confirm that LeRobot installs on native Windows. If it fails, the fallback is HDF5 files in the robomimic layout (`/data/demo_N/obs`, `/data/demo_N/actions`) written with h5py ([robosuite demonstrations](https://robosuite.ai/docs/algorithms/demonstrations.html)).

**Startup angle:** the hardware next step is cheap and credible. Hugging Face launched the SO-101 "starting at $100" ([TechCrunch](https://techcrunch.com/?p=2999850)), and a do-it-yourself follower-plus-leader set runs about $230 ([roboticscenter.ai](https://roboticscenter.ai/hardware/so-101)). Your webcam replaces the $100+ leader arm that LeRobot's standard teleop flow needs. XDOF built a company on GELLO, a teleop device whose parts reportedly cost under $300, and is reportedly raising at **about $1.2B with annualized revenue near $50M** ([Dealroom](https://dealroom.co/news/149061-xdof-nears-1-2b-series-b-just-months-after-70m-series-a/)).

### 3. Learned jitter filter that beats One Euro

**Pitch:** a small causal network makes webcam joint angles smooth enough to send to a motor without adding visible lag.

**Network and data:** a GRU or temporal convolution (TCN) over a sliding window of 16–32 frames, trained on your own recordings. Targets come from offline smoothing that can look ahead in time, or you add synthetic noise to clean sequences and train the network to remove it. SmoothNet (ECCV 2022) is the published precedent: a temporal-only network that plugs onto 11 pose backbones ([SmoothNet](https://github.com/cure-lab/SmoothNet)). The baseline problem is real. In a 2026 study, **a One Euro filter suppressed jitter but cut real motion roughly in half (path-length ratio 0.484)**, so the authors switched to an exponential moving average ([MDPI Sensors 2026](https://www.mdpi.com/1424-8220/26/12/3730)). AnyTeleop handles jitter only with its β smoothness term and never measures it ([AnyTeleop](https://arxiv.org/html/2307.04577v3)).

**Live demo:** three robot hands or arms side by side, driven by raw MediaPipe, One Euro and your network, with live jitter-versus-lag plots.

**Feasibility: Easy–Medium.** The model is tiny and the metrics (acceleration error, lag in ms) are simple to compute. It works best as a module inside idea 1 or 2.

**Startup angle:** buyers reportedly discard about 90% of bulk footage, and consumer devices show "jittery sensor sync and pose drift" ([Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/), a vendor-authored source). Cleaning up trajectories is part of the curation layer where value is moving.

### 4. "Which joint is moving": a live per-joint motion heatmap

**Pitch:** the skeleton lights up joint by joint as each one moves, showing its angle and angular velocity, which directly answers "which joints are moving."

**Network and data:** a multi-label GRU classifier over landmark windows that outputs moving or still for each joint. Train it on your own clips, labeled by hand or semi-automatically. The baseline is a threshold on the derivative of the angle. Threshold-based onset detection is known to fire late ([Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/neuro.20.002.2009/pdf)), and differentiating noisy markerless angles amplifies noise. MediaPipe-derived angular velocity agreed with markers at only **R = 0.11 at the ankle** in one study ([OMU avesis](https://avesis.omu.edu.tr/yayin/5e4cb644-c6e6-4ee0-9ee9-3a601786df6d/markerless-joint-angle-estimation-using-mediapipe-with-a-rapid-setup-for-joint-moment-calculation)). Report per-joint F1, onset latency and false triggers while you hold still.

**Live demo:** you wiggle one finger, then bend an elbow, and only those joints glow. A sidebar shows live angles and the robot copies only the active joints.

**Feasibility: Easy.** No standard benchmark exists for this, so you label your own clips. That is extra work, but it also makes your evaluation original.

**Startup angle:** segmenting demonstrations into subtasks costs **$10–16 per data-hour** to annotate ([DexSet](https://dexset.ai/blogs/robot-training-data-costs-pricing-complete-2026/), vendor pricing). Automating it is directly monetizable. The same signal also serves ergonomics: TuMeke raised a $10M Series A for phone-based worker injury-risk scoring ([Intel Capital](https://www.intelcapital.com/tumeke-raises-10m-in-series-a-funding-led-by-intel-capital/)).

### 5. Gesture-to-primitive control: pinch to grasp, open to release

**Pitch:** the network recognizes grasp, hold, release and move phases from your hand and commands a simulated gripper to pick up a cube.

**Network and data:** a GRU or 1D-convolution classifier over landmark windows, trained on a few hundred labeled gesture clips you record yourself. Its output fills the gripper channel of the robot's action. This is the action field of a LeRobot dataset, and Phantom likewise converts hand pose into end-effector pose plus gripper labels ([Phantom](https://arxiv.org/pdf/2503.00779)).

**Live demo:** a Franka or SO-101 in MuJoCo, with the arm following your wrist and the gripper triggered by your gestures, picking up and placing a block. This needs physics stepping, unlike the pure mirror demos.

**Feasibility: Easy–Medium.** Classification is easy. Making the physics grasp in MuJoCo work reliably is the risk.

**Startup angle:** commercial vendors already demo gesture-controlled arms (AgileX PiPER, myCobot 320) ([Hackster PiPER](https://www.hackster.io/agilexrobotics/control-piper-arm-with-hand-gestures-7c2779)). A PyPI package that drives a 4-DOF servo arm from MediaPipe already exists ([robot-visionaries](https://pypi.org/project/robot-visionaries/)). That proves demand but makes this a crowded, low-moat position.

### 6. Distilled finger-angle accuracy: making MediaPipe act like HaMeR

**Pitch:** a small network gets accuracy close to heavy GPU models out of a CPU-only webcam stream.

**Network and data:** run WiLoR or HaMeR offline on recorded webcam clips (Colab GPU) to get pseudo ground-truth joint angles. These models output parameters of MANO, a standard hand model, so each finger joint's rotation comes out directly. Then train an MLP from MediaPipe landmarks to those angles. The accuracy gap is large. **WiLoR reaches 5.5 mm and HaMeR 6.0 mm alignment-corrected joint error (PA-MPJPE) on the FreiHAND benchmark** ([WiLoR](https://arxiv.org/html/2409.12259)), but HaMeR takes about 60 ms per frame on a modern GPU ([EmergentMind](https://www.emergentmind.com/topics/hamer)). MediaPipe finger angles showed **22.5° RMSE against a Vicon marker system** in dynamic tasks ([PMC12473350](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12473350/)), and 10.9° in a study using a different protocol ([PMC11858933](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11858933/)). There is strong precedent for learned corrections: a small ML model raised thumb-abduction correlation with a goniometer **from r = 0.84 to 0.94–0.99** ([PMC11540810](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11540810/)), and LightGBM on MediaPipe landmarks reached **R² = 0.988** for shoulder abduction ([JBPE](https://jbpe.sums.ac.ir/article_51738.html)).

**Live demo:** angle readouts from raw MediaPipe and the distilled network shown side by side, with a recorded WiLoR reference clip for comparison.

**Feasibility: Medium–Hard.** Getting HaMeR or WiLoR running on Colab can eat two days. Their outputs and the MANO model files carry non-commercial terms.

**Startup angle:** accurate monocular extraction of 3D hand and arm motion is one of the identified underserved openings ([Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)). The licensing kills direct commercial use of the labels, though.

### 7. Demonstration quality scorer for robot-training clips

**Pitch:** a live "traffic light" that rates whether the clip you are recording is usable robot-training data, based on hand visibility, occlusion, tracking confidence, blur and jitter.

**Network and data:** a classifier or regressor over features from each landmark window (MediaPipe confidences, landmark velocity statistics, detection dropouts). Train it on clips you record and label good or bad, including deliberately occluded, too-fast and badly lit ones.

**Live demo:** the indicator turns red when you turn your palm sideways or move too fast, and the system automatically trims and keeps only the green segments in the exported dataset.

**Feasibility: Medium.** It is technically easy, but the demo is less dramatic than a moving robot. Pair it with idea 1 or 2 for visual punch.

**Startup angle:** this is the clearest wedge in the market notes. One buyer said "in any given hour of footage, we are lucky to find 2 or 3 segments that are suitable", and curated data commands a premium ([Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/), vendor-authored). Supply is flooding in. **Mecka AI raised a $60M Series B from Sequoia** to pay people to record tasks ([TechCrunch](https://techcrunch.com/2026/10/07/robot-data-startup-mecka-ai-nabs-60m-from-sequoia/)), and **Figure's Index app logged 16M video uploads and $15M paid to contributors** ([Humanoids Daily](https://www.humanoidsdaily.com/news/figure-ai-unveils-index-crowdsourcing-real-world-human-video-to-train-helix)). More raw footage makes filtering more valuable.

### 8. Monocular wrist-depth regressor for true 3D reaching

**Pitch:** recover how far your hand is from the camera, so the robot arm reaches forward and back, not only up, down and sideways.

**Network and data:** an MLP from 2D landmarks plus bounding-box size to wrist depth or palm scale, trained on FreiHAND, which has about 32.5K training frames with 3D labels and is licensed for non-commercial research ([Hamba](https://arxiv.org/pdf/2407.09646)). This attacks the most documented weakness. **AnyTeleop's piano task had 28.1% error with RGB alone, against 21.8% with RGB-D** ([AnyTeleop](https://arxiv.org/html/2307.04577v3)), and MediaPipe's hand depth "is not a metric measurement in the physical camera coordinate system" ([MDPI Sensors 2026](https://www.mdpi.com/1424-8220/26/12/3730)).

**Live demo:** push your hand toward the webcam and the simulated arm extends, with a depth readout checked against a tape measure.

**Feasibility: Hard.** The dataset download, mapping between camera models, and generalizing to your own webcam's lens are all risky in 8 days.

**Startup angle:** removing the need for a depth camera or VR headset widens the pool of contributors to anyone with a laptop. That is the same direction as Figure Index's smartphone-first capture.

### 9. Upper-body humanoid shadowing on a Unitree G1

**Pitch:** stand in front of the webcam and a simulated 29-DoF Unitree G1 humanoid copies your arms and torso.

**Network and data:** a Martinez-style MLP that "lifts" 2D keypoints into 3D. The original simple baseline improved on the prior Human3.6M state of the art by about 30% ([Martinez et al.](https://arxiv.org/pdf/1705.03098)). You train it on 2D/3D pairs synthesized by projecting AMASS motion-capture data through random cameras ([AMASS-to-3DHPE](https://github.com/goldbricklemon/AMASS-to-3DHPE)), or on Human3.6M, which needs academic registration with an unknown approval time ([Human3.6M EULA](https://vision.imar.ro/human3.6m/eula.php)). For large body joints, MediaPipe already performs close to a Kinect V2 against optical motion capture ([PMC9824231](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9824231/)). The best video lifter in one benchmark against IMU-based joint angles reached **9.27° RMSE** ([arXiv 2510.02264](https://arxiv.org/abs/2510.02264)).

**Live demo:** the most theatrical option, with a full humanoid copying you, waving, raising its arms and turning.

**Feasibility: Hard.** Data access and the retargeting from human to G1 joints are both nontrivial. Body retargeting also suffers from foot sliding and self-penetration, which is the reason GMR (general motion retargeting) exists ([GMR](https://arxiv.org/html/2510.02252v1)). Keep the legs fixed.

**Startup angle:** this is HumanPlus and H2O territory, single-RGB humanoid teleoperation ([TechXplore H2O](https://techxplore.com/news/2024-04-scalable-learningbased-framework-teleoperation-humanoid.amp)). It makes great portfolio material, but well-funded labs dominate.

### 10. Watch-then-imitate: a robot policy trained only on your webcam demos

**Pitch:** demonstrate a reach-to-target motion five dozen times on camera. A trained policy then performs it by itself in simulation.

**Network and data:** use idea 2's pipeline to record retargeted demonstrations as a LeRobotDataset, then train a small behavior-cloning MLP, or a lightweight version of ACT (Action Chunking with Transformers), that maps target position and joint state to the next joint action. HumanPlus reached **60–100% success with up to 40 demonstrations** collected by shadowing ([HumanPlus](https://arxiv.org/pdf/2406.10454)), and Phantom showed that policies trained only on human video transfer to real arms ([Phantom](https://arxiv.org/pdf/2503.00779)).

**Live demo:** you place a virtual target, step away from the camera, and the robot reaches it by itself.

**Feasibility: Hard.** It stacks ideas 2 and 5 plus policy training. Treat it as a stretch goal.

**Startup angle:** this is the industry thesis in miniature. Figure says Helix learned to navigate cluttered spaces from egocentric human video with **zero robot demonstrations** ([Figure](https://www.figure.ai/news/project-go-big)). EgoMimic found that co-training on human data improved performance by **34–228%** over robot-only data ([EgoMimic](https://arxiv.org/html/2410.24221v1)).

## Rank neural retargeting first, the arm mirror second, the denoiser third

**First choice: idea 1, neural hand retargeting.** It meets every course requirement without stretching. The PyTorch network is real and trained on data you generate, the before-and-after comparison is quantitative (optimizer milliseconds against network microseconds, plus joint error and smoothness), and dexterous fingers mirroring yours live is the most visceral demo on the list. The research framing is current: amortized retargeting runs from AnyTeleop's optimizer through GeoRT's 1 kHz network. A plausible 8-day plan is as follows. Days 1–2 get MediaPipe, MuJoCo and the LEAP model running with direct geometric angles, so a fallback demo exists from day 2. Days 3–4 record data and generate optimizer labels. Days 5–6 train and evaluate the network. Day 7 adds smoothing, clamping to joint limits, and the side-by-side toggle. Day 8 is for rehearsal.

**Second choice: idea 2, the SO-101 arm mirror with neural IK and LeRobot export.** It is the best fit for a startup or portfolio story, because it outputs a dataset in the format robot-learning pipelines actually consume, and it points to a $100–$230 hardware upgrade. The neural-IK training data is unlimited and free of licenses. It ranks second only because arm depth from a monocular camera is weak and the LeRobot install on Windows is unverified.

**Third choice: idea 3, the learned jitter filter.** It is the lowest-risk way to satisfy "train and improve a network", and it slots into either of the first two as a second contribution. The One Euro filter's documented over-smoothing gives you a credible baseline to beat. If time runs short, combining 1 and 3 ("learned retargeting plus learned smoothing") is a coherent single project. Idea 4's joint heatmap can be added as a cheap visual overlay that directly answers the brief's "which joints are moving" requirement.

## Seven risks that can sink the demo

**Depth from MediaPipe is not metric.** Image-space z "is not recommended", and only world landmarks are in meters ([BlazePose wrapper docs](https://github.com/ajinkyapuar/blazeposebarracuda)). Hand depth is relative ([MDPI Sensors 2026](https://www.mdpi.com/1424-8220/26/12/3730)). Prefer angles computed in the image plane, aim the camera side-on at the joints you care about, and do not promise forward-and-back reach unless you build idea 8. **Finger accuracy is modest and depends on viewpoint.** One pilot found a 4.22° error from a side view that got worse from the front ([DOAJ Sensors 2025](https://doaj.org/article/8c821edb8b654ff4ad3526ddab90bd0a)). Self-occlusion, such as a palm turned perpendicular to the camera, and fast motion break tracking ([AnyTeleop](https://arxiv.org/html/2307.04577v3)). Rehearse poses that face the camera, and test under the classroom's lighting.

**Windows tooling is the biggest schedule risk.** Avoid PyBullet (no Windows wheels) and Isaac Sim (RTX 4080 minimum). MuJoCo's Windows wheel for version 3.16.0 was not confirmed on its truncated PyPI page, so run `pip install mujoco mujoco-menagerie mediapipe torch` on day 1. MediaPipe 1.1.1 is only two days old, so pin a known-good version once it works. `dex-retargeting` v0.5.0 needs NumPy 2, and you should pin `<0.5.0` for NumPy 1 ([dex-retargeting](https://github.com/dexsuite/dex-retargeting)). SAPIEN, `somehand` and LeRobot are all unverified on native Windows, and LeRobot video encoding needs ffmpeg or torchcodec. Check out only the model you need from Menagerie, because the full repository is over 2 GB ([wikidocs](https://wikidocs.net/431647)).

**Laptop frame rate is unbenchmarked.** No published x86 laptop numbers exist for MediaPipe Hand or Pose. Measure on day 1, capture at 640×480, and drop stale frames rather than queuing them. **Licenses limit the startup path.** FreiHAND, Human3.6M, EgoDex and the MANO/SMPL model files are research-only or non-commercial ([Human3.6M EULA](https://vision.imar.ro/human3.6m/eula.php); [EgoDex](https://github.com/apple/ml-egodex)), so self-collected and synthetic data are the only commercially clean training sources. **Consent and privacy matter if you go to market.** India's IT ministry (MeitY) is examining consent practices at egocentric-data startups ([TechCrunch](https://techcrunch.com/2026/05/26/human-archive-taps-into-indias-services-startups-to-collect-data-for-physical-ai/)).

## Conclusion

The bottleneck has moved. Off-the-shelf pose estimation is solved enough for big joints, and data collection is a well-funded race that a solo founder cannot win. What remains scarce is the layer between a noisy landmark and a feasible, smooth robot command: retargeting, temporal cleanup, quality scoring and standard-format export. A small PyTorch network trained on landmark vectors is exactly the right tool for that layer, and it runs on a laptop CPU. Your project's credibility comes less from claiming accurate angles than from honestly measuring the gap between raw MediaPipe, a classical baseline and your network.

For portfolio or startup value, the most defensible move is to build ideas 1 and 2 on self-collected and synthetic data only, export to LeRobotDataset, and treat quality scoring (idea 7) as the eventual product. That keeps you clear of non-commercial dataset licenses. It also puts you on the curation side of a market where vendors report that buyers discard most raw footage, and where per-hour prices for undifferentiated video are already falling toward $2–$5 ([Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)).
