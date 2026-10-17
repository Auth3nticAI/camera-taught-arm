# Joint-Angle Accuracy from a Single RGB Webcam: MediaPipe Limits, Better Neural Models, Datasets, and Student-Trainable Architectures

Research date: 2026-10-16. About 16 search and fetch calls. Where a number comes from a secondary aggregator, it is flagged as such. Statements without a source appear under Inferences or Gaps.

## Q1. MediaPipe Hands/Pose/BlazePose accuracy numbers and known weaknesses; validation against mocap and goniometers

### Takeaway
For large body joints (hip, knee, shoulder, elbow) seen from a good camera angle, MediaPipe/BlazePose joint angles come within a few degrees of goniometers and correlate highly (r about 0.93-0.98) with marker-based mocap. Accuracy drops at extreme flexion and extension and for out-of-plane motion. Finger joint angles are much less reliable: RMSE ranges from about 4 deg (one joint, best camera view, static task) to about 11-22 deg (dynamic multi-joint tasks against Vicon). Both the image-space z (depth) and the hand depth output are relative rather than metric, and they jitter.

### Cited Findings
**BlazePose (body) published accuracy**
- The BlazePose GHUM Holistic paper reports 3D mean absolute error on its yoga evaluation set of 36 mm (Heavy), 39 mm (Full), and 45 mm (Lite). The metric is MAE rather than standard MPJPE, measured on 17 COCO keypoints with a single person 2-4 m from the camera. — [BlazePose GHUM Holistic, arXiv 2206.11678](https://arxiv.org/pdf/2206.11678)
- The authors acknowledge depth ambiguity: fitting 3D to 2D annotations gives several plausible poses, and adding annotator depth-order hints reduced depth-ordering errors from 25% to 3%. — [arXiv 2206.11678](https://ar5iv.labs.arxiv.org/html/2206.11678)
- Practical guidance from wrapper documentation: the image-space z value "is not recommended". Use world landmarks instead, which are metric coordinates in meters with the origin at the hip center. — [BlazePose Barracuda repo](https://github.com/ajinkyapuar/blazeposebarracuda)

**Body joint angles vs. gold standards**
- MediaPipe with a single RGB camera, compared against Qualisys optical mocap, performed close to a Kinect V2 RGB-D sensor in absolute error, RMS, and correlation. Its error fell within the error range of short- and long-arm goniometers. — [Sensors 2023, PMC9824231](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9824231/)
- Running gait (one runner, 6.5 m/s): correlation with marker-based data was r = 0.968 at the hip, 0.983 at the knee, and 0.928 at the ankle. Markerless methods were limited at predicting maximum flexion and extension angles. — [search summary of WKU IJESAB abstract](https://digitalcommons.wku.edu/ijesab/vol14/iss2/179)
- A telerehabilitation study using limits of agreement concluded that MediaPipe is reliable and valid for the shoulder and knee movements it studied. — [DOAJ, Sensors Jul 2023](https://doaj.org/article/a043a0af3c0147baa2bdb4ab6ad1093b)
- A shoulder abduction study fed MediaPipe landmarks into a LightGBM model trained on goniometer labels and reached R^2 = 0.988 and MAPE 1.539%. This is a learned correction on top of MediaPipe, not raw MediaPipe output, and it is the same idea as the student project: learn landmarks to angle. — [JBPE](https://jbpe.sums.ac.ir/article_51738.html)
- Sit-to-stand: angular velocities computed from MediaPipe joint angles agreed weakly with marker-based data at the ankle (R = 0.11). — [OMU avesis record, "Markerless joint angle estimation using MediaPipe..."](https://avesis.omu.edu.tr/yayin/5e4cb644-c6e6-4ee0-9ee9-3a601786df6d/markerless-joint-angle-estimation-using-mediapipe-with-a-rapid-setup-for-joint-moment-calculation)
- A 2026 JMIR preprint validating BlazePose Heavy knee angles (concurrent validity plus test-retest) was still under review, and its results were not visible. — [JMIR preprint 102399](https://preprints.jmir.org/preprint/102399)

**Hand / finger joint angles**
- Against a Vicon marker-based system in dynamic tasks: Leap Motion Controller 2 had mean RMSE 14.8 deg (r = 0.58), and MediaPipe had mean RMSE 22.5 deg (r = 0.45). — [Evaluation of Commercial Camera-Based Solutions for Tracking Hand Kinematics, PMC12473350](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12473350/)
- An OpenSim-based comparison (n = 15) found the reverse ranking: the MediaPipe setup had RMSE 10.9 deg and Leap Motion 14.7 deg. The two studies used different protocols, so their results conflict. — [Optimisation and Comparison of Markerless and Marker-Based Motion Capture for Hand and Finger, PMC11858933](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11858933/)
- PIP joint during computer-mouse use against an electrogoniometer: MediaPipe from a lateral view had RMSE 4.22 deg, with 92.7% of measurements within 5% error. The frontal view was worse, so camera view strongly affects results. The study was a small pilot (about 5 participants). — [Sensors Feb 2025 via DOAJ](https://doaj.org/article/8c821edb8b654ff4ad3526ddab90bd0a)
- Thumb radial abduction against a goniometer: raw MediaPipe Hands had r = 0.84. Adding an ML regressor raised it to r = 0.94-0.99 (n = 10). — [PMC11540810](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11540810/); [Cureus PDF](https://assets.cureus.com/uploads/original_article/pdf/301169/20241008-605168-phpok1.pdf)
- Hand depth is not metric: "RGB-based hand landmark detectors provide stable semantic 2D landmarks, but their depth output is not a metric measurement in the physical camera coordinate system." The authors fused MediaPipe with ZED2 stereo depth to fix this. — [MDPI Sensors 26(12):3730, 2026](https://www.mdpi.com/1424-8220/26/12/3730)
- Jitter: in the same paper, a One Euro filter smoothed strongly but attenuated real motion (path-length ratio 0.484), so the authors chose an EMA with alpha = 0.7. — [MDPI Sensors 2026](https://www.mdpi.com/1424-8220/26/12/3730)
- MediaPipe hand position and rotation show jitter. A moving average works for position but poorly for rotation. — [Karsanusa journal](https://journalng.uwks.ac.id/karsanusa/article/download/644/373/3075)
- Snap's Spectacles SDK filters palm and wrist landmarks separately because of extra jitter, and smooths the hand's mean z on its own. — [Snap Lens Studio docs](https://developers.snap.com/lens-studio/api/lens-scripting/classes/Packages_SpectaclesInteractionKit_Core_Interactor_raycastAlgorithms_RaycastBase.default.html)

### Inferences
- A realistic demo claim is about 5-10 deg error for elbow and knee flexion in-plane with a side-on camera. Finger flexion is likely 10-20 deg or worse unless the hand faces the camera side-on and nothing is occluded.
- Angles computed from 2D landmarks in a plane parallel to the camera are often more trustworthy than angles that rely on MediaPipe z.
- Several studies show that a small learned regressor on top of MediaPipe landmarks improves angle accuracy (LightGBM shoulder, ML thumb). That gives direct published precedent for a "train a small net on landmarks" PyTorch project.

### Gaps
- Google's official model card error numbers for MediaPipe Hands 3D (for example, MPJPE on FreiHAND) were not retrieved.
- No dedicated per-axis z-error study of BlazePose world landmarks was found.
- No validation study reporting finger self-occlusion error broken down by occlusion level was found.

## Q2. Models that do better than MediaPipe (HaMeR, WiLoR, FrankMocap, MotionBERT, VideoPose3D, 4DHumans/HMR2.0, RTMPose, MANO, SMPL-X): size, speed, laptop real-time

### Takeaway
For hands, WiLoR (5.5 mm PA-MPJPE on FreiHAND) and HaMeR (6.0 mm) are the accuracy state of the art. Both use a ViT-Large-class backbone and need a GPU; HaMeR takes about 60 ms per frame on a modern GPU. Lighter mesh models (FastViT baseline, ReJSHand, MobRecon) reach 5.7-6.3 mm at about 70 fps on a GPU. For body, video lifters (MotionBERT about 37-39 mm MPJPE on Human3.6M, MotionAGFormer) beat BlazePose and run on 2D keypoints, so they are cheap. RTMPose is the fastest high-accuracy 2D detector on CPU. On a laptop CPU, real-time is realistic only for MediaPipe, RTMPose, and small lifters, not for ViT-L or ViT-H mesh models.

### Cited Findings
**Hands**
- WiLoR, FreiHAND: PA-MPJPE 5.5 mm, PA-MPVPE 5.1, F@5 0.825, F@15 0.993. Comparison results: HaMeR 6.0/5.7/0.785/0.990, MobRecon 5.7/5.8, Mesh Graphormer 5.9/6.0, METRO 6.5/6.3. — [WiLoR arXiv 2409.12259](https://arxiv.org/html/2409.12259)
- WiLoR, HO3D v2: PA-MPJPE 7.5 mm, compared with 7.7 for HaMeR. — [WiLoR](https://arxiv.org/html/2409.12259)
- WiLoR uses a ViT-Large backbone with ViTPose weights (hidden dim 1280). Its detector runs at 138 fps (M) or 175 fps (S) on an RTX 4090. End-to-end reconstruction FPS is not reported. It introduces the WHIM dataset (over 2M in-the-wild images from more than 1,400 YouTube videos) and trains on 4.2M images from 14 datasets. The paper is licensed CC BY-NC-ND. — [WiLoR](https://arxiv.org/html/2409.12259)
- HaMeR takes about 60 ms per frame on a single modern GPU ("near-real-time"), with FreiHAND PA-MPJPE 6.0 and PA-MPVPE 5.7. This comes from a secondary summary. — [EmergentMind HaMeR](https://www.emergentmind.com/topics/hamer)
- Lightweight hand mesh models: the "Simple Baseline" with FastViT-MA36 reaches 5.7 mm PA-MPJPE and 6.0 PA-MPVPE on FreiHAND at up to about 70 fps — [arXiv 2403.01813](https://arxiv.org/html/2403.01813v1). ReJSHand reaches 6.3 mm at 72 fps — [arXiv 2503.05995](https://arxiv.org/pdf/2503.05995). M3DHMR (ResNet34) reaches 6.5 mm and is reported as best among real-time camera-space methods — [arXiv 2505.20058](https://arxiv.org/pdf/2505.20058)

**Body**
- MotionBERT (DSTformer, a dual-stream spatial and temporal transformer) reaches 39.2 mm MPJPE on H36M trained from scratch and 37.2 mm after fine-tuning. The full checkpoint is 162 MB and MotionBERT-Lite is 61 MB. It is pretrained to recover 3D from noisy, masked 2D sequences, which is relevant because webcam 2D is noisy. — [MotionBERT GitHub](https://github.com/walter0807/motionbert); [paper arXiv 2210.06551](https://arxiv.org/pdf/2210.06551); [MMPose MotionBERT README](https://cdn05042023.gitlink.org.cn/OpenMMLab/mmpose/src/branch/main/configs/body_3d_keypoint/motionbert/README.md)
- An independent 2025 benchmark against IMU plus OpenSim joint angles (VIDIMU dataset, 13 daily activities, healthy adults) compared MotionAGFormer, MotionBERT, MMPose lifting, and NVIDIA BodyTrack. MotionAGFormer was best, with RMSE 9.27 +/- 4.80 deg, MAE 7.86 deg, r = 0.86, and R^2 = 0.67. The authors call video-based kinematics "clinically promising" but less precise than IMUs in some cases. — [arXiv 2510.02264](https://arxiv.org/abs/2510.02264)
- LiftFormer (attention lifter, 9.5M params) reaches 44.8 mm MPJPE on H36M from detected 2D and 31.9 mm from ground-truth 2D. — [arXiv 2009.00348](https://export.arxiv.org/abs/2009.00348)
- Martinez et al. "simple baseline" MLP lifter (ICCV 2017): a plain feedforward 2D-to-3D network that beat the prior state of the art on H36M by about 30%. — [arXiv 1705.03098](https://arxiv.org/pdf/1705.03098)
- HMR 2.0 / 4DHumans is a fully transformer-based mesh recovery model (ViT encoder). An aggregator lists 3DPW MPJPE 69.8, PA-MPJPE 44.4, and MPVPE 82.2 (unverified against the paper). — [arXiv 2305.20091](https://ar5iv.labs.arxiv.org/html/2305.20091); [EmergentMind HMR-ViT](https://www.emergentmind.com/topics/hmr-vit)
- RTMPose-m reaches 75.8% AP on COCO at 90+ fps on an Intel i7-11700 CPU. RTMPose-l reaches 67.0% AP on COCO-WholeBody. RTMPose-m hand model (COCO-WholeBody-Hand): PCK@0.2 0.815, EPE 4.51 px. — [MMPose RTMPose project](https://cdn05042023.gitlink.org.cn/OpenMMLab/mmpose/src/branch/main/projects/rtmpose); [hand README](https://cdn05042023.gitlink.org.cn/OpenMMLab/mmpose/src/branch/main/configs/hand_2d_keypoint/rtmpose/README.md)
- RTMW-l reaches 70.2 mAP on COCO-WholeBody, and the paper also covers 3D whole-body. — [RTMW arXiv 2407.08634](https://arxiv.org/html/2407.08634v1)

### Inferences
- Laptop real-time ranking, from most to least practical: MediaPipe (CPU real-time) > RTMPose ONNX (CPU real-time) > MediaPipe/RTMPose plus a small lifter or smoother (CPU OK) > lightweight hand mesh (needs a GPU for 30 fps) > HaMeR/WiLoR/HMR2.0 (ViT-L/H; likely under 10 fps without a strong discrete GPU). The last tier is inferred from the 60 ms/frame GPU figure.
- HaMeR and WiLoR output MANO parameters, so finger joint angles come directly from MANO pose (axis-angle per joint). This avoids computing angles from noisy 3D points, which is a real advantage for an angle-focused project, but they are too heavy for a live demo on a laptop CPU.
- A good use of the heavy models is offline: run WiLoR or HaMeR on recorded webcam clips to generate pseudo-ground-truth angles, then train a small net on MediaPipe landmarks to match them. This is a form of distillation.

### Gaps
- FrankMocap speed and accuracy numbers were not retrieved. It is older (2021) and superseded by HaMeR and HMR2.0.
- VideoPose3D numbers were not retrieved in this session. From memory it is a TCN lifter at about 46.8 mm H36M MPJPE with a 243-frame receptive field; verify before citing.
- Parameter counts for MotionBERT and HaMeR were not confirmed.
- No SMPL-X-specific model numbers (for example, OSX or SMPLer-X) were gathered.

## Q3. Datasets with ground truth: sizes, licenses, ease of use

### Takeaway
FreiHAND (about 33K train and 4K eval, MANO labels, non-commercial) is the easiest hand dataset for a few-day project. InterHand2.6M and DexYCB are large (2.6M and 582K frames) and heavy to download. Human3.6M needs academic registration and forbids redistribution. AMASS provides SMPL joint rotations, which are true joint angles, and can be rendered into synthetic 2D/3D pairs. Hand datasets give 3D joints plus MANO pose. True clinically measured joint angles (goniometer) are rare in public data.

### Cited Findings
- FreiHAND: 32,560 training frames, about 130K with background augmentation, and about 4K evaluation images. Research and non-commercial use only. — [Hamba arXiv 2407.09646](https://arxiv.org/pdf/2407.09646); [DexYCB comparison table](https://arxiv.org/pdf/2104.04631)
- InterHand2.6M: 2.6M frames of single and interacting hands, images downscaled to 512x334 for privacy. — [arXiv 2008.09309](https://arxiv.org/pdf/2008.09309.pdf). One catalog lists about 366K train, 110K val, and 261K test samples for the released version — [TIB](https://service.tib.eu/ldmservice/group/hand-object-interaction)
- HO-3D: about 78K frames at 640x480, markerless hand-object with YCB objects, v2 (2020) and v3 (2022). — [DexYCB paper](https://arxiv.org/pdf/2104.04631); [Hamba](https://arxiv.org/pdf/2407.09646)
- DexYCB: about 582K RGB-D frames, 10 subjects, hand grasping YCB objects. — [arXiv 2104.04631](https://arxiv.org/pdf/2104.04631)
- Human3.6M: free for academic use only, requested from an academic email. The license is non-transferable and forbids sublicensing, and commercial use needs a separate license. — [Human3.6M EULA](https://vision.imar.ro/human3.6m/eula.php)
- AMASS unifies many optical marker-based mocap datasets into one SMPL-parameterized database and has its own collection licence. — [GenMotion dataset docs](https://genmotion.readthedocs.io/en/latest/dataset.html). A tool exists to convert AMASS to 3D HPE training data — [AMASS-to-3DHPE](https://github.com/goldbricklemon/AMASS-to-3DHPE)
- VIDIMU: video plus IMU dataset of 13 daily activities, used to benchmark video joint angles against IMU and OpenSim. — [arXiv 2510.02264](https://arxiv.org/abs/2510.02264)

### Inferences
- The fastest data path for an 8-day project is to self-collect. The student records their own webcam sessions, runs MediaPipe for input landmarks, and gets labels from (a) a heavy offline model (WiLoR/HaMeR MANO angles, or MotionBERT 3D) or (b) a phone goniometer app for a few calibration poses. This avoids license and download delays.
- For a body 2D-to-3D lifter, the pre-extracted Human3.6M 2D/3D .npz files used by VideoPose3D and MotionBERT are the standard route, but registration can take days. AMASS is a fallback because 2D can be synthesized by projecting SMPL joints with random cameras.
- MANO and SMPL model files themselves require registration with non-commercial terms.

### Gaps
- Exact licenses for HO-3D, DexYCB, InterHand2.6M (I believe CC BY-NC 4.0, but not verified), and AMASS text were not retrieved.
- Human3.6M approval turnaround time is unknown.

## Q4. Small trainable PyTorch architectures (landmarks to angles, 2D-to-3D lifting, temporal smoothing): training time and accuracy

### Takeaway
Proven small options exist. A Martinez-style residual MLP lifter (about 4M params) and LiftFormer-class transformers (9.5M) reach roughly 45-63 mm on H36M. SmoothNet is a temporal-only fully connected refiner that plugs onto any estimator. MotionBERT-Lite (61 MB) can be fine-tuned. Published work shows small regressors on MediaPipe landmarks sharply improve angle accuracy. All of these take landmark vectors (about 42-99 floats per frame), so they train in minutes to a few hours on a laptop CPU or GPU.

### Cited Findings
- Martinez simple baseline: a feedforward 2D-to-3D MLP that improved on the prior H36M state of the art by about 30%. An aggregator lists 62.9 mm, with other protocol entries of 45.5 and 47.7, so verify against the paper's tables. — [arXiv 1705.03098](https://arxiv.org/pdf/1705.03098); [aggregator](https://www.wizwand.com/paper/68fa638c1e209131ce9ff79d)
- LiftFormer: 9.5M params, 44.8 mm (detected 2D). — [arXiv 2009.00348](https://export.arxiv.org/abs/2009.00348)
- SmoothNet (ECCV 2022) is a temporal-only, motion-aware fully connected network that models each joint's long-range temporal relations, ignoring noisy cross-joint correlations. It plugs onto 11 backbones across 5 datasets for 2D, 3D, and mesh. Pretrained checkpoints exist (AIST-VIBE-3D, 3DPW-SPIN-3D, H36M-FCN-3D), and it is included in MMPose 0.25+. — [ECCV paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650615.pdf); [GitHub](https://github.com/cure-lab/SmoothNet)
- MotionBERT-Lite checkpoint (61 MB), with pretraining on noisy and masked 2D. — [GitHub](https://github.com/walter0807/motionbert)
- MotionAGFormer (transformer plus GCNFormer) had the best joint-angle RMSE (9.27 deg) against IMUs in the VIDIMU benchmark. — [arXiv 2510.02264](https://arxiv.org/abs/2510.02264); [MotionAGFormer arXiv 2310.16288](https://ar5iv.labs.arxiv.org/html/2310.16288)
- Landmarks to angle regression on MediaPipe: LightGBM reached R^2 0.988 for shoulder abduction against a goniometer — [JBPE](https://jbpe.sums.ac.ir/article_51738.html). An ML model raised thumb abduction correlation from r = 0.84 to 0.94-0.99 — [PMC11540810](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11540810/)
- Uplift-and-upsample reaches 45.0 mm on H36M with 12x faster inference. — [search result referencing efficient lifting](https://arxiv.org/pdf/2210.06110v1)

### Inferences
These are suggestions for the student and are not sourced.
- Option A, landmark to joint-angle regressor: an MLP or GRU over a 21x3 window of MediaPipe hand landmarks outputs 15-20 finger angles, with labels from offline WiLoR/HaMeR MANO pose. Data comes from self-recorded sessions (about 30-60 minutes, roughly 100K frames). Training would likely take under 1 hour on a laptop GPU, and a CPU is feasible.
- Option B, temporal denoiser: a 1D TCN, GRU, or SmoothNet-style FC over a sliding window of about 16-32 frames, trained to map noisy MediaPipe sequences to smooth targets (offline heavy-model outputs, or synthetic noise added to clean AMASS or self-filtered data). Report jitter (acceleration error) together with lag.
- Option C, body 2D-to-3D lifter: a Martinez residual MLP or a small spatial-temporal GCN on MediaPipe or COCO 2D, trained on H36M or AMASS-synthesized pairs. Expect tens of minutes to a few hours of training.
- All three run at well over 30 fps on CPU at inference, so the live demo is safe. Compare against two baselines: raw MediaPipe geometric angles, and a One Euro or EMA filter.
- ST-GCN-style graph convolution suits joint-graph inputs, but for 21-33 joints an MLP or GRU is usually adequate and simpler to debug in 8 days.

### Gaps
- No source reported concrete wall-clock training times for these lifters on laptop hardware. The estimates above are inferences.
- No published paper was found that trains a MediaPipe-to-MANO-angle distillation network specifically.

## Q5. Quantifying "which joint is moving" (angular velocity, activity detection) and evaluating it

### Takeaway
The common approach is to differentiate filtered joint angles to get angular velocity, then threshold each joint (often relative to peak velocity, for example a fraction or a standard deviation below peak) to label movement onset and offset. Researchers evaluate onset and offset timing against marker-based or expert-labelled ground truth, reporting bias and limits of agreement. Velocity thresholds are known to bias onset late, and derivative signals amplify markerless noise. Velocity agreement can be poor at small, distal joints (ankle R = 0.11).

### Cited Findings
- Velocity-threshold onset detection gives positive (late) bias in onset time. Alternatives include regression-based methods and comparison against expert manual labels. — [Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/neuro.20.002.2009/pdf)
- A markerless 3D reaching validation segmented movements using marker-based wrist tangential velocity, with start and end where velocity crossed 1 SD below mean peak velocity, then applied those windows to markerless data for comparison. — [bioRxiv 2023.03.16.532711](https://www.biorxiv.org/content/10.1101/2023.03.16.532711.full.pdf)
- MediaPipe-derived angular velocity agreed weakly with markers at the ankle during sit-to-stand (R = 0.11). — [OMU avesis](https://avesis.omu.edu.tr/yayin/5e4cb644-c6e6-4ee0-9ee9-3a601786df6d/markerless-joint-angle-estimation-using-mediapipe-with-a-rapid-setup-for-joint-moment-calculation)
- Velocity-based pose metrics such as MPJVE (mean per-joint velocity error) quantify temporal smoothness and accuracy. — [3D pose review arXiv 2010.06449](https://arxiv.org/pdf/2010.06449)
- Pipelines using markerless pose, then time-series segmentation, then event detection have been used for naturalistic hand movement mining. — [arXiv 2001.08349](https://arxiv.org/pdf/2001.08349)
- Joint-angle validation metrics in practice: RMSE, MAE, Pearson r, and R^2 per joint — [arXiv 2510.02264](https://arxiv.org/abs/2510.02264). Limits of agreement are also used — [Sensors 2023 via DOAJ](https://doaj.org/article/a043a0af3c0147baa2bdb4ab6ad1093b)

### Inferences
- For the demo, compute per-joint angular velocity from smoothed angles (for example with a Savitzky-Golay derivative or a learned smoother), then label a joint "moving" when |omega| > k x (noise floor measured at rest), for example k about 3, with hysteresis and a minimum duration. Choose k from a recorded still-hand clip.
- Evaluate on a self-recorded clip with hand-labelled per-joint moving or still segments. Report per-joint precision, recall, and F1, onset latency (ms), and false-trigger rate at rest. Compare raw MediaPipe, One Euro, and the trained neural smoother. This gives the "trained network improves X" result the course requires.
- A learned alternative: a small GRU or TCN classifier that outputs per-joint moving or still from a landmark window. This is multi-label classification and also satisfies the train-a-network requirement.

### Gaps
- No study was found that directly validates per-joint angular-velocity activity detection from MediaPipe for fingers.
- No standard benchmark dataset exists for "which joint is moving" labels. Self-labelling is needed.
