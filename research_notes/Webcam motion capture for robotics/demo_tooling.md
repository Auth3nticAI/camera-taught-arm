# Demo Tooling: Webcam -> MediaPipe -> Simulated/Cheap Robot Mirroring (Windows, CPU, 8 days)

Research date: 2026-10-16. Some star counts and dates come from GitHub pages that do not show commit dates; these are flagged.

## Simulators and viewers: which run on Windows without a big GPU, and which are pip-installable?

### Takeaway
MuJoCo (with the `mujoco-menagerie` pip package for models) is the strongest CPU-friendly, pip-installable choice on Windows. It ships ready-made SO-101, LEAP Hand, Shadow Hand, Allegro, Franka and Unitree G1/H1 models. Genesis also officially supports Windows CPU. PyBullet has no Windows wheels on PyPI, so it is risky on Windows. Isaac Sim is out without an RTX 4080-class GPU. For joint-angle visualization, Rerun (it has a URDF loader and logs time series) or a browser three.js URDF viewer are light and easy to add.

### Cited Findings
- **MuJoCo Menagerie** includes these models: Shadow Hand E3M5 (24 DoF, Apache-2.0), Allegro Hand V3 (16 DoF, BSD-2), LEAP Hand (16 DoF, MIT), Shadow DEX-EE, Robotiq 2F-85, Franka Panda/FR3, The Robot Studio SO101 (6 DoF, Apache-2.0), "Standard Open Arm-100 5DOF" (the SO-100 lineage), Unitree H1 (19 DoF) and Unitree G1 (29 DoF). There is no Koch model. — [MuJoCo Menagerie GitHub](https://github.com/google-deepmind/mujoco_menagerie)
- Menagerie now has a Python package: `pip install mujoco-menagerie`, then `mm.load('<model>')` downloads a model on first use into a per-user cache. Release 2026.9.0 pins every model. Models are also available through the third-party `robot_descriptions` package, or you can clone the repo and open `scene.xml` in the MuJoCo viewer. — [MuJoCo Menagerie GitHub](https://github.com/google-deepmind/mujoco_menagerie)
- The full Menagerie repo is over 2 GB, so one tutorial sparse-checks out only `robotstudio_so101`. That model is about 27 MB, and the tutorial was tested on MuJoCo 3.13. — [wikidocs SO-101 MuJoCo tutorial](https://wikidocs.net/431647); [wikidocs page 2](https://wikidocs.net/431648)
- MuJoCo's latest PyPI version is 3.16.0. The fetched page was truncated, so I did not confirm the wheel list. — [PyPI mujoco](https://pypi.org/project/mujoco/#files)
- **PyBullet**: the latest version is 3.2.7 (Jan 30, 2025). PyPI lists only Linux x86-64 wheels for CPython 3.6–3.11, with **no win_amd64 wheels**. On Windows, pip falls back to an 80.5 MB source build. — [PyPI pybullet files](https://pypi.org/project/pybullet/#files)
- Windows users report that PyBullet needs Visual C++ Build Tools, that builds fail on Python 3.12, and that conda-forge is the workaround. These reports are unverified forum posts. — [python-forum.de thread](https://www.python-forum.de/viewtopic.php?p=424375); [Medium install guide](https://deepakjogi.medium.com/how-to-install-pybullet-physics-simulation-in-windows-e1f16baa26f6)
- The bullet3 GitHub issue tracker is closed because it was "flooded with support questions". That suggests limited maintenance. — [bullet3 fork README mirror](https://github.com/DVLP/bullet3)
- **Genesis**: install PyTorch, then run `pip install genesis-world`. It supports Python 3.10–3.13 on Linux, macOS and Windows. The support matrix lists CPU simulation as supported on Windows, and the CPU backend works without a GPU. — [Genesis installation docs](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)
- **Isaac Sim 5.0** minimum on Windows 10/11: GeForce RTX 4080 with 16 GB VRAM, 32 GB RAM, 4 cores and a 50 GB SSD. GPUs without RT cores are unsupported. — [Isaac Sim 5.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html)
- One user reports that Isaac Sim 5.x/4.5 will not run on a Quadro RTX 3000 with 6 GB. — [NVIDIA forum](https://forums.developer.nvidia.com/t/isaac-sim-5-x-and-4-5-wont-run-on-quadro-rtx-3000-6-gb-vram-recommended-path-for-internship/372141)
- **Rerun**: it has a built-in URDF data-loader. Logging a URDF file creates mesh entities and joint transforms, and you animate joints by logging `Transform3D`s computed with `UrdfJoint.compute_transform(angle)`, with the angle in radians for revolute joints. Versions from 0.28 use named frame IDs, while older versions used implicit per-joint entity paths. — [Rerun URDF how-to](https://rerun.io/docs/howto/urdf); [Rerun animated_urdf example](https://rerun.io/examples/robotics/animated_urdf); [Rerun URDF API ref](https://ref.rerun.io/docs/python/dev/urdf/)
- Rerun is the visualizer LeRobot uses: `lerobot-record --display_data=true` streams data live. — [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- **three.js urdf-loaders** (gkjohnson): URDF loaders for three.js and Unity. Apache-2.0, about 819 stars, 975 commits. There is a live browser demo. — [gkjohnson/urdf-loaders](https://github.com/gkjohnson/urdf-loaders); [live demo](https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/)
- **SAPIEN** is the renderer used in dex-retargeting's examples. — [dex-retargeting GitHub](https://github.com/dexsuite/dex-retargeting)

### Inferences
- The recommended Windows CPU stack is `pip install mediapipe mujoco mujoco-menagerie rerun-sdk torch`. Use MuJoCo's passive viewer, or just use `mj_kinematics` to set `qpos` and render. For a mirror demo, kinematic playback is enough, so you don't need physics stepping. Kinematic playback (setting qpos and calling mj_forward) is trivially real-time on CPU.
- MuJoCo has historically shipped win_amd64 wheels, but this was not verified on the truncated PyPI page. Run `pip install mujoco` on day 1 to confirm.
- Avoid PyBullet on Windows unless you use conda. Avoid Isaac Sim/Lab entirely without a high-end NVIDIA GPU. List them as "GPU-dependent future work".
- Genesis works, but it is heavier and newer. MuJoCo plus Menagerie is the lower-risk path in an 8-day window.
- For the joint-angle plot, Rerun can show the 3D URDF and time-series panels in one window. A cheaper alternative is OpenCV/matplotlib overlays drawn on the webcam frame.

### Gaps
- No source confirmed whether Open3D can animate URDFs in real time. It was not researched in depth, but it is likely unnecessary given Rerun.
- I did not confirm Rerun's scalar time-series logging API for joint-angle plots. It probably works through `rr.Scalars`/`rr.Scalar`, but check the current Rerun docs.
- Genesis real-time CPU performance on laptops was not quantified.
- I did not confirm whether SAPIEN, which dex-retargeting's examples need, installs on Windows.

## Cheap real hardware as a next step

### Takeaway
The SO-101 (LeRobot) is the standard cheap arm. It costs roughly $100–$230 DIY (follower, or leader plus follower), and several hundred dollars assembled. The Koch v1.1 is about $199 per arm with Dynamixel servos. The LEAP Hand costs about $2,000 to build, which is too expensive for a student demo but citable as a "real dexterous hand" next step. The SO-101 and LEAP Hand both have MuJoCo models, so a sim-to-real story is natural.

### Cited Findings
- At launch, Hugging Face said the SO-101 starts at $100. Assembled units and US tariffs push supplier prices to about $100–$500. — [TechCrunch](https://techcrunch.com/?p=2999850)
- DIY estimates: the follower alone is about $122, and the follower plus leader set is about $230. Another guide estimates about $150. — [roboticscenter.ai SO-101](https://roboticscenter.ai/hardware/so-101)
- An Australian retailer lists the SO-ARM101 kit without printed parts at A$573.25 and an assembled Pro at A$714.19. Both were out of stock. — [Pakronics kit](https://www.pakronics.com.au/products/new-so-arm101-low-cost-ai-arm-servo-motor-kit-for-lerobot-without-3d-printed-parts-ss114993666); [Pakronics Pro](https://pakronics.com.au/products/so-arm101-low-cost-ai-arm-servo-motor-kit-pro-for-lerobot-assembled-version-ss100046482)
- LeRobot natively records SO-101 leader/follower teleop to a Hub dataset (`--robot.type=so101_follower --teleop.type=so101_leader`). — [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- Koch v1.1 uses Dynamixel XL430-W250 and XL330-M288 servos through a U2D2 USB adapter, with `pip install lerobot`. A community BOM totals about $199 in US pricing. Teleop needs both a leader and a follower arm, and you need a 3D printer for the parts. — [Mintlify LeRobot Koch](https://www.mintlify.com/huggingface/lerobot/robots/koch); [jess-moss/koch-v1-1](https://github.com/jess-moss/koch-v1-1); [OpenELAB buying guide](https://openelab.io/it/blogs/learn/lerobot-hardware-buying-guide-so-arm101-vs-koch-v1-1-vs-openarm-vs-lekiwi-vs-xlerobot)
- ROBOTIS sells a Koch v1.1 leader product. — [robotis.us](https://robotis.us/koch-v1-1-low-cost-robot-arm-leader/)
- LEAP Hand (CMU, RSS 2023): about $2,000 in off-the-shelf parts and about 4 hours to assemble. The URDF, CAD and simulation are public. The paper says it outperforms the $16K+ Allegro Hand at 1/8 of the cost. — [LEAP Hand paper (arXiv)](https://ar5iv.arxiv.org/html/2309.06440); [RSS proceedings](https://www.roboticsproceedings.org/rss19/p089.html)
- A vendor-style guide lists LEAP v1 at about $2,066 and v2 at about $4,550. These prices are unverified. — [roboticscenter.ai hand comparison](https://roboticscenter.ai/learn/dexterous-robot-hand-comparison-2026)
- Webcam-plus-MediaPipe control of a cheap physical 4-DOF servo arm already exists as a PyPI package (`robot-visionaries`). — [PyPI robot-visionaries](https://pypi.org/project/robot-visionaries/)

### Inferences
- In the presentation, the SO-101 is the natural next step: about $120–$230, the same MuJoCo model as the sim, and LeRobot integration. Servo-based 3D-printed hands and InMoov are hobby options, but they have no standard sim model.

### Gaps
- I found no sourced current prices for InMoov or generic servo-driven 3D-printed hands.
- I found no official statement on LeRobot's Windows support for serial-port robots. The docs show macOS/Linux port examples only.

## Existing open-source projects (webcam -> MediaPipe -> robot hand/arm in sim)

### Takeaway
dex-retargeting (MIT, about 1.3k stars) is the canonical hand-retargeting library from AnyTeleop. It maps MediaPipe hand keypoints to robot-hand joint angles for many hands, and its examples render in SAPIEN. somehand (Apache-2.0, 77 stars) is a newer CLI that does webcam -> MediaPipe -> MuJoCo for 20 commercial hands. For arms, I found no polished MuJoCo/PyBullet webcam repo, so the student would write the arm mapping (shoulder/elbow/wrist angles from MediaPipe Pose, or IK) themselves.

### Cited Findings
- **dex-retargeting** (dexsuite): "various retargeting optimizers to translate human hand motion to robot hand motion". MIT license, about 1.3k stars, 150 forks, 104 commits. Install the core with `pip install dex_retargeting`; for examples, run `pip install -e ".[example]"` from a clone. v0.5.0 added NumPy 2 support; use `<0.5.0` for NumPy 1. Hand models come from the dex-urdf project. Examples cover retargeting from human video (teleop, as in AnyTeleop) and from hand-object datasets (DexMV). SAPIEN is used for rendering. The page shows no commit dates or Windows notes. — [dex-retargeting GitHub](https://github.com/dexsuite/dex-retargeting); [PyPI](https://pypi.org/project/dex-retargeting)
- Star counts conflict across aggregators: about 925 on one and about 1,120 in a July 2026 blog, against about 1.3k on GitHub at fetch time. — [gittrend](https://gittrend.io/repo/dexsuite/dex-retargeting); [BrightCoding blog](https://www.blog.brightcoding.dev/2026/07/19/dexsuitedex-retargeting-hand-motion-translation-for-robot-control)
- dex-retargeting came out of AnyTeleop, a vision-based dexterous arm-hand teleoperation system presented at RSS 2023. MediaPipe works with NumPy 2 despite pinning warnings. — [dex-retargeting search summary / PyPI](https://pypi.org/project/dex-retargeting)
- **somehand** (BotRunner64): "Universal dexterous-hand retargeting with MediaPipe, MuJoCo, and YAML-configured robot hands". It takes input from a webcam, video, PICO Bridge, hc_mocap UDP or recordings. It has a CLI and a Python API, and supports 20 hand models (LinkerHand, Unitree Dex5, Inspire, BrainCo Revo2, Sharpa Wave and others). Apache-2.0, 77 stars, 110 commits. It provides a `somehand webcam` command, and on macOS it needs `mjpython`. The page shows no last-commit date and no Windows notes. — [somehand GitHub](https://github.com/BotRunner64/somehand); [mirror](https://proxy-ga.blitzz.co/proxy/123456/github.com/BotRunner64/somehand)
- **FrankaTeleop** (gjcliff): a ROS 2 package that uses MediaPipe hand tracking plus MoveIt Servo to move a Franka Panda. It needs ROS, so it is Linux-oriented. — [FrankaTeleop mirror](https://kompas.wasmer.app/https_github_com/gjcliff/FrankaTeleop)
- An SBR/LARS 2025 paper teleoperates a simulated Panda in Gazebo from single-RGB MediaPipe body tracking. — [SOL SBC paper](https://sol.sbc.org.br/index.php/sbrlars/article/view/39213)
- Commercial demos: AgileX PiPER and myCobot 320 have hand-gesture-controlled arms. — [Open Robotics discourse](https://discourse.openrobotics.org/t/agilex-piper-teleoperation-with-hand-gesture-recognition/49994); [Hackster PiPER](https://www.hackster.io/agilexrobotics/control-piper-arm-with-hand-gestures-7c2779); [Hackster myCobot](https://hackster.io/gary26/hand-gestures-as-the-remote-controlling-mycobot-320-5dd749)

### Inferences
- Fastest hand path: MediaPipe 21 landmarks -> dex-retargeting optimizer (vector or position retargeting) with the LEAP/Allegro/Shadow config -> write qpos into the matching Menagerie MuJoCo model. This means skipping SAPIEN on Windows. Joint names may need mapping between dex-urdf and Menagerie MJCF. A simpler fallback computes finger flexion angles directly from landmark vectors.
- Arm path: compute shoulder, elbow and wrist angles from MediaPipe Pose landmarks (11–16), then map them to the SO-101's 6 joints. The gripper opening comes from the hand thumb-index distance.
- A PyTorch component could be a small MLP that learns landmarks -> joint angles, trained on retargeting-optimizer outputs (distilling a slow optimizer into a fast net), or a gesture classifier.

### Gaps
- I could not verify last-commit dates for dex-retargeting or somehand; the GitHub pages did not show them.
- I could not verify Windows compatibility of dex-retargeting's examples (SAPIEN) or of somehand.
- I did not find LeRobot examples that use a webcam/MediaPipe as a teleop device. Its teleop types shown are leader arms.

## Real-time pipeline concerns (latency, smoothing, CPU FPS)

### Takeaway
MediaPipe Hand Landmarker runs in about 17 ms per frame on a Pixel 6 CPU, using tracking mode that skips palm detection when tracking holds. On a laptop CPU, 20–30 FPS for MediaPipe plus a small PyTorch model plus kinematic MuJoCo rendering is plausible, though I found no laptop benchmark to confirm it. Use a One Euro filter per landmark coordinate (or per joint angle) to remove jitter without adding much lag.

### Cited Findings
- Official Hand Landmarker (full) latency is about 17.12 ms on CPU and 12.27 ms on GPU, averaged on a Pixel 6. In video/live-stream mode it reuses the previous frame's hand box and reruns palm detection only when tracking fails. — [Google MediaPipe Hand Landmarker guide](https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker)
- MediaPipe 1.1.1 was released Oct 14, 2026. It has a `py3-none-win_amd64` wheel (22.8 MB) and lists Python 3.10–3.13 classifiers. — [PyPI mediapipe](https://pypi.org/project/mediapipe/#files)
- One Euro filter: an adaptive low-pass filter that uses a low cutoff at low speed (less jitter) and raises the cutoff at high speed (less lag). It has two tunable parameters. A pure-Python `OneEuroFilter` package on PyPI takes `freq`, `mincutoff`, `beta` and `dcutoff`, and you call it with a value and a timestamp. — [PyPI OneEuroFilter](https://pypi.org/project/oneeurofilter)
- Tuning: a lower min_cutoff removes more jitter, and a higher beta reacts faster. One community example for pose landmarks uses min_cutoff=1.0, beta=0.007. — [needle.tools OneEuroFilter docs](https://engine.needle.tools/docs/api/@needle-tools/engine/3.44.1/classes/engine_api.OneEuroFilter.html); [mediapipe-pose-smooth](https://github.com/yousufkalim/mediapipe-pose-smooth)
- For offline use, a Butterworth low-pass at 6–10 Hz is suggested; for real time, One Euro. Raising MediaPipe tracking confidence also reduces jitter. These come from secondary skill-guide sources. — [agentskills mediapipe-pose-detection](https://agentskills.so/skills/feniix-kinemotion-mediapipe-pose-detection)
- A SIGGRAPH/SCA 2026 poster combines One Euro with confidence weighting to reduce monocular pose jitter. — [Bournemouth SCA2026 poster](https://eprints.bournemouth.ac.uk/42215/1/VA_SCA2026_Poster.pdf)

### Inferences
- Frame budget at 30 FPS is 33 ms: MediaPipe about 15–30 ms on a laptop CPU, a tiny MLP under 1 ms, MuJoCo mj_forward plus viewer sync a few ms. Run capture and inference in one thread and the viewer in its own, or use the MuJoCo passive viewer, and drop frames rather than queue them so latency stays bounded.
- Filter the joint angles (after retargeting) instead of, or as well as, the raw landmarks. Fewer signals to filter, and the joint-angle output is what the demo shows. Also clamp the output to the joint limits from the MJCF.
- Lower the webcam resolution (640x480) and use `running_mode=LIVE_STREAM` or VIDEO to get tracking-mode speed.
- A Kalman filter is a valid alternative, but One Euro is simpler to tune within 8 days.

### Gaps
- I found no published x86 laptop CPU benchmarks for MediaPipe Hand/Pose; measure on the target laptop.
- I found no measured end-to-end latency numbers for dex-retargeting's optimizer; AnyTeleop reports may contain them but were not fetched.

## Recording and exporting a dataset in a robotics-standard format

### Takeaway
Use LeRobotDataset v3: Parquet for states and actions, MP4 for camera frames, and JSON/Parquet metadata. Build it programmatically with `LeRobotDataset.create(...)`, `add_frame`, `save_episode`, `finalize()` and `push_to_hub()` to publish on the Hugging Face Hub. The alternative is robomimic/ACT-style HDF5 (`/data/demo_N/obs/...`, `/data/demo_N/actions`), which is simple to write with h5py.

### Cited Findings
- LeRobotDataset v3 ships in `lerobot >= 0.4.0`. It stores many episodes per Parquet or MP4 file (v2 used one per episode), and episode boundaries live in metadata. The layout has `meta/info.json` (features, shapes, dtypes, fps, path templates), `meta/stats.json` (normalization), `meta/tasks.jsonl`, `meta/episodes/` (chunked Parquet), `data/` (Parquet shards) and `videos/` (MP4 per camera). — [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- Programmatic creation follows this pattern: `dataset = LeRobotDataset.create(...)`, then `dataset.add_frame(frame)` for each frame and `dataset.save_episode()` for each episode, then `dataset.finalize()` and `dataset.push_to_hub()`. You **must** call `finalize()`, or the Parquet files are corrupt (PR #1903). — [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- Loading returns dicts of PyTorch tensors (`observation.state`, `action`, `observation.images.<cam>`, `timestamp`) that are compatible with `torch.utils.data.DataLoader`. `delta_timestamps` gives temporal windows, and `StreamingLeRobotDataset` streams from the Hub. — [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- File naming is `data/chunk-000/file-000.parquet` and `videos/<camera>/chunk-000/file-000.mp4`. A v2.1->v3 converter exists. — [LeRobot porting datasets v3](https://huggingface.co/docs/lerobot/porting_datasets_v3)
- robomimic-style HDF5 is laid out as `/data/demo_0/obs/{camera, joint_positions, ...}` and `/data/demo_0/actions`, with a `/mask` group for train/val splits. The `data` group carries `total` and `env_args` (JSON) attributes. These sources are secondary, not the upstream spec. — [claru.ai HDF5 guide](https://claru.ai/formats/hdf5); [robopal docs](https://robopal.readthedocs.io/zh/latest/tutorials/imitation_learning.html); [robosuite demonstrations](https://robosuite.ai/docs/algorithms/demonstrations.html)
- Community conversions from robomimic to LeRobot v3 exist on the Hub, for example Lift PH with action dim 7 and state dim 53. — [ViaCatalyst/robomimic-lift-ph-lerobot-v3](https://huggingface.co/datasets/ViaCatalyst/robomimic-lift-ph-lerobot-v3)

### Inferences
- Suggested feature schema for the demo: `observation.state` = robot joint angles (float32[N]); `observation.landmarks` = MediaPipe 21x3 (or 33x3 for pose) flattened; `action` = target joint angles; `observation.images.webcam` = the webcam video. Plus `fps=30` and a task string such as "mirror right hand". This makes the "human video -> robot action" dataset directly loadable for PyTorch training and viewable in the Hub dataset visualizer.
- If installing LeRobot on Windows is painful, write HDF5 with h5py in robomimic layout as a fallback. It needs no dependencies beyond h5py and numpy, and can be converted to LeRobot later.

### Gaps
- I did not verify the exact `LeRobotDataset.create` signature (`repo_id`, `fps`, `features`, `use_videos`, ...) in the current source. Check the lerobot code or examples before implementing.
- I did not verify that LeRobot installs on native Windows; video encoding needs ffmpeg/torchcodec.
- I did not fetch ACT's own HDF5 schema (ALOHA style: `/observations/qpos`, `/observations/images/<cam>`, `/action`).
