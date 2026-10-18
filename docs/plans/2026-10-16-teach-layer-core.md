# Takeworthy — Teach Layer Core Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the live loop: camera → hand features → absolute same-side mapping → IK baseline → MuJoCo twin and the real SO-101. Add session recording (paired with the leader arm, or free-hand). It must run with no hardware tonight (Fri Oct 16), on the real arm Saturday night, and record training data Sunday morning.

**Architecture:** Pure-numpy core modules (features, mapping, filters, units, kinematics) are unit-tested. Thin I/O wrappers sit around them: the camera, MediaPipe, LeRobot motors, the MuJoCo twin, and Rerun. One CLI app (`cta-mirror`) wires them together. Its `--robot none` mode is design requirement R1. The IK baseline built here is also the "teacher" for Plan 2's learned retargeting network.

**Tech Stack:** Python 3.12 (uv), torch 2.6.0+cu124, lerobot[feetech] 0.4.4, mediapipe 1.1.1 (Tasks API), mujoco 3.16.0, mujoco-menagerie 2026.10.3 (`robotstudio_so101`), rerun-sdk 0.26.2, pytest.

**Spec:** `docs/specs/2026-10-16-takeworthy-design.md` (moves into the repo in Task 1)

**Later plans (written when needed):**
- Plan 2: learned retargeting network + GRU smoother + benchmarks (v1 trains Sunday AM from Task 9's sessions).
- Plan 3: policy episodes through mirroring → LeRobotDataset → ACT overnight → trials → video.
- R1 polish (dashboard, replay fallback) folds into Plan 2.

## Global Constraints

- **Python 3.12.** The system Python 3.11.3 breaks the mujoco-menagerie downloader (`tarfile.extractall(filter=...)` needs ≥ 3.11.4).
- **Exact pins:**
  - `lerobot[feetech]==0.4.4`, `mediapipe==1.1.1`, `mujoco==3.16.0`, `mujoco-menagerie==2026.10.3`
  - `opencv-contrib-python==4.12.0.88`: matches lerobot's `opencv-python-headless==4.12.0.88`, because both provide `cv2`
  - torch from the cu124 index (`torch==2.6.0+cu124`)
- **No `cv2.imshow` anywhere.** The headless OpenCV may shadow the GUI build. All display goes through Rerun (the single dashboard window the spec requires) and the MuJoCo renderer.
- **Locations (Tray decided: no OneDrive):**
  - Code, docs, and notes: GitHub private repo `camera-taught-arm` (gh is logged in as `Auth3nticAI`), cloned at `C:\Users\trayd\dev\camera-taught-arm` with `.venv` inside it. The laptop clones the same repo.
  - Data and models: local `CTA_DATA_DIR` (default `C:\Users\trayd\dev\cta-data`, never committed), synced through **Hugging Face Hub private repos**:
    - dataset repo `<hf_user>/cta-sessions` (sessions + LeRobot datasets)
    - model repo `<hf_user>/cta-models` (checkpoints, calibration JSONs)
    - via `scripts/sync_data.py {push,pull}` (uses `huggingface_hub`, already a lerobot dependency)
  - Tray runs `uv run hf auth login` once per machine.
- **Never commit `Deep_Learning_with_PyTorch.pdf`** (licensed copy) to the repo.
- **Training code follows the course text** (*Deep Learning with PyTorch*, Stevens/Antiga/Viehmann): see the spec's "Course-text alignment" section. Applies to Plan 2/3; add `tensorboard` to dependencies now.
- **Joint names**, identical in LeRobot and MuJoCo, in this order everywhere: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- **Units:**
  - MuJoCo uses radians. LeRobot (`use_degrees=True`, the default) uses degrees for joints 1–5 and 0–100 for the gripper.
  - Inside `cta`, joint vectors are **radians, shape (6,)**, until `units.sim_to_real`.
  - Sim joint limits (rad): pan ±1.92, lift ±1.745, elbow ±1.69, wrist_flex ±1.658, wrist_roll −2.744…2.841, gripper −0.175…1.745.
- **Human camera:** a PTZ webcam at 1080p@60 with tracking, gesture control, and zoom off (spec §3). MediaPipe input is downscaled to 960×540.
- **Control mapping (Tray decided):** absolute, same-side, with a calibrated home pose.
- **Gripper** is driven by pinch distance, not by a network. The depth cue is hand scale first (D-04 recommendation).
- **Gate targets** (spec §5): ≥ 20 FPS; camera → command < 150 ms.
- **Process:**
  - Tray makes the engineering decisions. Any choice that changes behavior, metrics, or scope stops and asks.
  - After each milestone task, append an entry to `docs/notes/LAB_NOTEBOOK.md`. Log any decision in `docs/notes/DECISIONS.md`.

## Review Focus

1. **The hand leaves the frame or detection drops.** The arm and twin must **hold the last command** (never jump to zero or home), and the dashboard shows "NO HAND". Test: `test_step_holds_last_command_when_no_hand` (Task 7).
2. **Two hands are in view, or handedness is flipped.** MediaPipe assumes a mirrored selfie image, so on our *unmirrored* frames its "Left" label is the user's right hand. Select by configured hand with that flip applied. Test: `test_select_hand_flips_handedness_for_unmirrored` (Task 6).
3. **The PTZ preset was lost** (wrong resolution or FPS after a replug). The startup check returns warnings and the app prints them prominently. Test: `test_camera_check_flags_wrong_mode` (Task 6).
4. **The target is out of reach.** IK returns finite joints within limits, and the app limits per-frame joint change, so there are no NaNs or lunges. Tests: `test_ik_unreachable_target_stays_finite_and_in_limits` (Task 2), `test_step_rate_limits_joint_jumps` (Task 7).
5. **The USB/serial link dies mid-session, or the app crashes.** The recorder must keep everything up to the failure: autosave every 30 s plus a save in `finally`. Test: `test_recorder_saves_on_exception` (Task 9).

---

### Task 1: Repo, environment, and smoke test

**Files:**
- Create: `C:\Users\trayd\dev\camera-taught-arm\pyproject.toml`, `.gitignore`, `README.md`, `src/cta/__init__.py`, `src/cta/paths.py`, `scripts/download_models.py`, `scripts/sync_data.py`, `tests/test_env.py`
- Move into the repo: `docs/` (specs, plans, notes), `reports/`, and `research_notes/` from `C:\Users\trayd\OneDrive\Desktop\Pytorch Midterm\`. Leave `MOVED.md` there pointing to the GitHub repo. Do **not** move the book PDF or the midterm files.

**Interfaces:**
- Produces:
  - the installable package `cta` (src layout)
  - `cta.paths.data_dir() -> pathlib.Path`: reads `CTA_DATA_DIR`, default `C:\Users\trayd\dev\cta-data`, created if missing
  - `cta.paths.models_dir() -> Path` (= `data_dir()/"models"`)
  - the model file `models_dir()/"hand_landmarker.task"`
  - `scripts/sync_data.py push|pull [--what sessions|models|all]`: `huggingface_hub.upload_folder` / `snapshot_download` against `<hf_user>/cta-sessions` (repo_type dataset) and `<hf_user>/cta-models`; creates them private if missing; `hf_user` comes from `HfApi().whoami()`.

- [ ] **Step 1: Create the repo and the uv project.** `pyproject.toml` declares `requires-python = "==3.12.*"`, the Global Constraints pins, `[tool.uv] index` for torch cu124 (`https://download.pytorch.org/whl/cu124`, explicit, torch/torchvision only), and the script entry point `cta-mirror = "cta.apps.mirror:main"`. Run `uv sync`. Expected: success.
- [ ] **Step 2: Write the failing test** `tests/test_env.py`:
```python
def test_imports_and_cuda():
    import torch, mujoco, mediapipe, lerobot, rerun  # noqa
    assert torch.cuda.is_available()

def test_so101_model_loads():
    import mujoco_menagerie as mm
    m = mm.load("robotstudio_so101")
    assert m.nu == 6

def test_hand_model_present():
    from cta.paths import models_dir
    assert (models_dir() / "hand_landmarker.task").stat().st_size > 1_000_000
```
- [ ] **Step 3: Run it.** `uv run pytest tests/test_env.py -v`. Expected: `test_hand_model_present` FAILS (no paths module or model).
- [ ] **Step 4: Implement `cta/paths.py` and `scripts/download_models.py`.** The script downloads `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task` into `models_dir()`. Run it.
- [ ] **Step 5: Run the tests.** `uv run pytest -v`. Expected: 3 PASS. Also run `uv run python -c "import cv2; print(cv2.__version__)"` and expect `4.12.0`.
- [ ] **Step 6: HF login check.** Tray runs `uv run hf auth login`, pasting a write token from huggingface.co/settings/tokens. Then `uv run python scripts/sync_data.py push --what models` uploads `hand_landmarker.task`. Expected: the private repo `<hf_user>/cta-models` exists. If Tray hasn't logged in yet, skip this and remind Tray; it isn't blocking tonight.
- [ ] **Step 7: Commit, create the private GitHub repo, and push.** `git init`; `gh repo create camera-taught-arm --private --source . --push`. Commit message: `chore: bootstrap camera-taught-arm env`.

---

### Task 2: Kinematics (FK, IK) on the MuJoCo SO-101

**Files:**
- Create: `src/cta/config.py`, `src/cta/kinematics.py`, `tests/test_kinematics.py`

**Interfaces:**
- Produces:
  - `config.JOINTS: tuple[str, ...]` (the 6 names)
  - `config.SIM_LIMITS: np.ndarray` (6, 2) rad, read from the model
  - `config.READY_Q: np.ndarray` (6,) rad, a reachable pose with the gripper in front of the base
  - `config.APPROACH_AXIS: int`: the column of the `gripperframe` site's `xmat` that points from the wrist to the jaw tips
  - `class Kinematics`:
    - `__init__(self)`
    - `fk(self, q: np.ndarray) -> tuple[np.ndarray, float]`: gripper site position (3,) in the base frame, plus the approach pitch in rad (positive = pointing up)
    - `ik(self, target_pos: np.ndarray, target_pitch: float, q_init: np.ndarray, wrist_roll: float, gripper: float) -> np.ndarray`: (6,) rad, clamped to `SIM_LIMITS`

- [ ] **Step 1: Write the failing tests.**
```python
def test_fk_ready_pose_in_front_of_base():
    k = Kinematics(); p, _ = k.fk(READY_Q)
    assert p[2] > 0.02 and np.linalg.norm(p[:2]) > 0.10

def test_ik_roundtrip_under_5mm():
    k = Kinematics(); rng = np.random.default_rng(0)
    for _ in range(20):
        q = READY_Q + rng.uniform(-0.4, 0.4, 6); q = np.clip(q, *SIM_LIMITS.T)
        p, pitch = k.fk(q)
        q2 = k.ik(p, pitch, READY_Q, q[4], q[5])
        p2, _ = k.fk(q2)
        assert np.linalg.norm(p2 - p) < 0.005

def test_ik_unreachable_target_stays_finite_and_in_limits():
    k = Kinematics()
    q = k.ik(np.array([2.0, 0.0, 2.0]), 0.0, READY_Q, 0.0, 0.5)
    assert np.all(np.isfinite(q))
    assert np.all(q >= SIM_LIMITS[:, 0]) and np.all(q <= SIM_LIMITS[:, 1])
```
- [ ] **Step 2: Run them.** `uv run pytest tests/test_kinematics.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
  - Load `mm.load("robotstudio_so101")` once.
  - `fk` sets qpos, calls `mujoco.mj_kinematics`, and reads `site_xpos`/`site_xmat` for `gripperframe` relative to the `baseframe` site.
  - Pick `APPROACH_AXIS` empirically: print `xmat` at two poses and choose the column that changes as expected when the wrist flexes. Pick `READY_Q` by eye in the MuJoCo viewer (`python -m mujoco.viewer`).
  - `ik`: damped least squares over joints `[pan, lift, elbow, wrist_flex]`.
    - Residual: `[target_pos − p (3), 0.05 m/rad · (target_pitch − pitch)]`.
    - **Numerical Jacobian** (finite differences, ε = 1e-4), λ = 0.05, ≤ 15 iterations, warm start from `q_init`.
    - Clamp to limits every iteration.
    - Joints 4 and 5 are set directly from `wrist_roll` and `gripper`.
- [ ] **Step 4: Run them.** Expected: 3 PASS. Also time it: `ik` takes under 2 ms on average (`python -m timeit`). If it's slower, reduce the iteration count.
- [ ] **Step 5: Commit.** `feat: SO-101 FK and DLS IK on MuJoCo model`

---

### Task 3: Hand features (pure numpy)

**Files:**
- Create: `src/cta/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: MediaPipe indices (wrist 0, thumb tip 4, index MCP 5, index tip 8, middle MCP 9, pinky MCP 17).
- Produces:
  - `@dataclass HandFeatures`: `u: float`, `v: float` (wrist, normalized image coordinates), `scale: float` (mean image distance from the wrist to MCPs 5, 9, 17, as a fraction of image width), `pitch: float` (rad, wrist → middle-MCP vector vs. the camera's horizontal plane, from world landmarks; up = positive), `roll: float` (rad, the index-MCP → pinky-MCP vector's angle around the hand axis), `pinch: float` (thumb tip–index tip distance / palm size, world landmarks)
  - `compute_features(lm_img: np.ndarray, lm_world: np.ndarray, aspect: float) -> HandFeatures`: `lm_img` is (21, 3), normalized; `lm_world` is (21, 3) in meters; `aspect` = width/height

- [ ] **Step 1: Write the failing tests** using a synthetic flat hand built in the test: points on a plane, with the wrist at the origin and MCPs 8 cm away.
```python
def test_scale_doubles_when_hand_twice_as_big(): ...   # scale(2x landmarks about wrist) ≈ 2 * scale
def test_pinch_small_when_tips_touch(): ...            # thumb tip == index tip → pinch < 0.05
def test_pitch_positive_when_fingers_point_up(): ...   # rotate the world landmarks +30° about the camera x-axis → pitch ≈ +0.52 ± 0.05
def test_roll_changes_sign_with_hand_rotation(): ...   # ±30° about the hand axis → roll sign flips
```
- [ ] **Step 2: Run them.** `uv run pytest tests/test_features.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `compute_features`.** Correct x for aspect before measuring image distances (`dx * aspect`). Palm size = the mean world distance from the wrist to MCPs 5, 9, 17.
- [ ] **Step 4: Run them.** Expected: 4 PASS.
- [ ] **Step 5: Commit.** `feat: hand feature extraction`

---

### Task 4: Calibration and absolute same-side mapping

**Files:**
- Create: `src/cta/mapping.py`, `tests/test_mapping.py`

**Interfaces:**
- Consumes: `HandFeatures` (Task 3); `Kinematics`, `READY_Q` (Task 2).
- Produces:
  - `@dataclass Calibration`: `u0, v0, s0, pitch0, roll0: float`; `pinch_closed: float = 0.25`, `pinch_open: float = 0.9`; `p0: np.ndarray` (3,), the ready-pose gripper position; `pitch_home: float`
  - `@dataclass MapGains`: `k_lat: float = 0.30`, `k_vert: float = 0.30`, `k_fwd: float = 0.25` (meters); `box_min: np.ndarray`, `box_max: np.ndarray` (workspace clamp in the base frame, set from FK samples of reachable poses)
  - `calibrate(f: HandFeatures, kin: Kinematics) -> Calibration`
  - `map_to_target(f: HandFeatures, cal: Calibration, g: MapGains) -> tuple[np.ndarray, float, float, float]`: returns (target_pos (3,), target_pitch, wrist_roll rad, gripper rad)
  - `save_calibration(cal, path)` and `load_calibration(path) -> Calibration` (JSON)

The mapping is a spec-level decision (absolute, same-side), so here is the exact formula. The camera faces the user and the robot faces the same way the user faces.

```
y   = p0.y + k_lat  * (u - u0)            # user moves to their left → image u increases → robot +y (its left)
z   = p0.z - k_vert * (v - v0)            # hand up → v decreases → robot up
x   = p0.x + k_fwd  * (1 - s0 / scale)    # hand toward camera → scale grows → robot forward
pos = clip([x, y, z], box_min, box_max)
target_pitch = pitch_home + (pitch - pitch0)
wrist_roll   = roll - roll0
gripper      = lerp(SIM_LIMITS.gripper, clip((pinch - pinch_closed) / (pinch_open - pinch_closed), 0, 1))
```

**Assumption to verify:** MuJoCo base frame is +x forward, +y left, +z up. Confirm in Task 2's viewer. If the axes differ, fix the signs here and note it in the LAB_NOTEBOOK.

- [ ] **Step 1: Write the failing tests.**
```python
def test_home_features_map_to_p0(): ...                 # calibrate(f) then map(f) → pos ≈ p0, gripper per pinch
def test_user_left_moves_robot_plus_y(): ...            # u += 0.1 → pos.y increases by 0.03
def test_hand_closer_moves_robot_forward(): ...         # scale *= 1.25 → pos.x increases
def test_pinch_closed_gives_min_gripper(): ...          # pinch = 0.1 → gripper == SIM_LIMITS[5, 0]
def test_target_clipped_to_box(): ...                   # u += 5 → pos within [box_min, box_max]
def test_calibration_json_roundtrip(tmp_path): ...
```
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Use the formula above, with defaults from the Interfaces block.
- [ ] **Step 4: Run them.** Expected: 6 PASS.
- [ ] **Step 5: Commit.** `feat: absolute same-side hand→workspace mapping with calibration`

---

### Task 5: One Euro filter and sim↔real unit conversion

**Files:**
- Create: `src/cta/filters.py`, `src/cta/units.py`, `tests/test_filters.py`, `tests/test_units.py`

**Interfaces:**
- Produces:
  - `class OneEuro`: `__init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0)`; `__call__(self, x: np.ndarray, t: float) -> np.ndarray` (vector-valued, independent per element); `reset()`
  - `units.SimRealMap`: `sign: np.ndarray` (5,), default ones; `offset_deg: np.ndarray` (5,), default zeros; `load/save` as JSON at `data_dir()/"sim_real_map.json"`
  - `sim_to_real(q: np.ndarray, m: SimRealMap) -> dict[str, float]`: LeRobot action dict `{"shoulder_pan.pos": deg, ..., "gripper.pos": 0–100}`
  - `real_to_sim(obs: dict[str, float], m: SimRealMap) -> np.ndarray`: (6,) rad

Formulas: `deg_j = sign_j * degrees(q_j) + offset_j` for j < 5. `gripper_pos = 100 * (q5 − lo) / (hi − lo)`, using the sim gripper limits.

- [ ] **Step 1: Write the failing tests.**
```python
def test_one_euro_passes_constant(): ...                # constant input → output == input after a few samples
def test_one_euro_reduces_noise(): ...                  # std(output) < 0.5 * std(input) for white noise around a constant
def test_sim_real_roundtrip(): ...                      # real_to_sim(sim_to_real(q)) ≈ q for random q within limits
def test_gripper_limits_map_to_0_and_100(): ...
def test_sign_flip_applies(): ...                       # sign[1] = -1 → shoulder_lift degrees negated
```
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Use the standard One Euro (Casiez et al.) with α computed from dt.
- [ ] **Step 4: Run them.** Expected: 5 PASS.
- [ ] **Step 5: Commit.** `feat: One Euro filter and sim/real joint conversion`

---

### Task 6: Camera and MediaPipe hand wrappers

**Files:**
- Create: `src/cta/camera.py`, `src/cta/hands.py`, `tests/test_camera.py`, `tests/test_hands.py`

**Interfaces:**
- Produces:
  - `@dataclass CameraMode`: `width: int`, `height: int`, `fps: float`
  - `camera_check(actual: CameraMode, wanted: CameraMode = CameraMode(1920, 1080, 60)) -> list[str]`: warnings, empty if OK; FPS must be ≥ 0.85 × wanted
  - `class Camera`:
    - `__init__(self, index: int, wanted: CameraMode)`: opens with `cv2.CAP_DSHOW`, MJPG FOURCC, and sets the size and fps
    - `read(self) -> tuple[np.ndarray | None, float]`: RGB frame and `time.perf_counter()`
    - `measure_mode(self, n: int = 60) -> CameraMode`: the measured fps
    - `close()`
  - `@dataclass HandObservation`: `lm_img: np.ndarray` (21, 3), `lm_world: np.ndarray` (21, 3), `side: str` (the user's actual hand: "right" or "left"), `score: float`
  - `select_hand(handedness_labels: list[str], scores: list[float], want: str, mirrored_input: bool) -> int | None`: index of the wanted hand
  - `class HandTracker`:
    - `__init__(self, model_path: Path, want: str = "right", mirrored_input: bool = False, infer_width: int = 960)`
    - `detect(self, rgb: np.ndarray, t: float) -> HandObservation | None`: Tasks API `HandLandmarker`, `RunningMode.VIDEO`, `num_hands=2`, timestamps in increasing ms

- [ ] **Step 1: Write the failing tests.**
```python
def test_camera_check_ok(): assert camera_check(CameraMode(1920, 1080, 59.0)) == []
def test_camera_check_flags_wrong_mode():
    w = camera_check(CameraMode(1280, 720, 30.0)); assert len(w) == 2
def test_select_hand_flips_handedness_for_unmirrored():
    # MediaPipe labels assume a mirrored selfie; on an unmirrored frame its "Left" is the user's right
    assert select_hand(["Left", "Right"], [0.9, 0.9], want="right", mirrored_input=False) == 0
    assert select_hand(["Left", "Right"], [0.9, 0.9], want="right", mirrored_input=True) == 1
def test_select_hand_none_when_absent(): assert select_hand(["Left"], [0.9], "left", False) is None
def test_tracker_returns_none_on_blank_image():
    t = HandTracker(models_dir() / "hand_landmarker.task")
    assert t.detect(np.zeros((1080, 1920, 3), np.uint8), 0.0) is None
```
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Downscale to `infer_width` before detection (normalized landmarks are unaffected). Read the Tasks API names from `mediapipe.tasks.python.vision` in 1.1.1. If they differ from 0.10.x (`HandLandmarker`, `HandLandmarkerOptions`, `RunningMode`), adapt and note it in the LAB_NOTEBOOK.
- [ ] **Step 4: Run them.** Expected: 5 PASS.
- [ ] **Step 5: Manual check with the PTZ camera.** Run `uv run python -m cta.camera --index 0`, a `__main__` that prints `measure_mode()` and the warnings. Expected: about 1920×1080 at ~60 fps with no warnings. If it isn't, Tray fixes the camera's app preset (tracking, gesture, and zoom off) and reruns. Record the result in the LAB_NOTEBOOK.
- [ ] **Step 6: Commit.** `feat: camera with mode check and MediaPipe hand tracker`

---

### Task 7: Twin, dashboard, and `cta-mirror --robot none` (tonight's milestone)

**Files:**
- Create: `src/cta/twin.py`, `src/cta/dashboard.py`, `src/cta/pipeline.py`, `src/cta/apps/__init__.py`, `src/cta/apps/mirror.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces:
  - `class Twin`: `__init__(self, width: int = 640, height: int = 480)`; `render(self, q: np.ndarray) -> np.ndarray` (RGB, offscreen via `mujoco.Renderer`, scene entry `"scene"`)
  - `class Dashboard`: `__init__(self, app_id: str = "cta")` (`rr.init(spawn=True)`); `log_frame(self, rgb, obs: HandObservation | None, twin_rgb, q_cmd: np.ndarray, fps: float, latency_ms: float, status: str)`. Uses `rr.Image`, `rr.Points2D` for landmarks, and `rr.Scalars` under `joints/<name>` in degrees.
  - `class MirrorPipeline`: `__init__(self, kin, cal: Calibration | None, gains: MapGains, smoother: OneEuro | None, max_step_rad: float = 0.15)`; `step(self, obs: HandObservation | None, t: float) -> tuple[np.ndarray, str]` (q_cmd (6,) rad and a status in `{"NO HAND", "UNCALIBRATED", "TRACKING"}`); `calibrate(self, obs)`. It starts from `READY_Q`.
  - CLI `cta-mirror --robot {none,so101} --camera 0 --hand right [--calib PATH]`. Keys, via `pynput` (already a lerobot dependency): `c` = calibrate now, `s` = toggle smoother, `q` = quit.

- [ ] **Step 1: Write the failing tests** with a fake `HandObservation` built from Task 3's synthetic hand.
```python
def test_step_uncalibrated_holds_ready(): ...           # no calibration → q == READY_Q, status "UNCALIBRATED"
def test_step_holds_last_command_when_no_hand():
    # after a TRACKING step, obs=None → identical q_cmd, status "NO HAND"
def test_step_rate_limits_joint_jumps():
    # calibrate, then an obs with u += 0.5 → max |Δq| per step <= max_step_rad
def test_step_outputs_within_limits(): ...              # 100 random plausible obs → all q within SIM_LIMITS
```
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** the pipeline, twin, dashboard, and app loop. The loop: camera.read → tracker.detect → pipeline.step → twin.render → dashboard. Show fps (EMA) and latency (frame timestamp → command ready). On start, print `camera_check` warnings in a banner.
- [ ] **Step 4: Run them.** `uv run pytest -v`. Expected: all PASS.
- [ ] **Step 5: Live check (Tray). Milestone.** Run `uv run cta-mirror --robot none`, then press `c` with the hand at home.
  - Expected: the twin follows the hand in Rerun. Left/right, up/down, toward/away, tilt, and pinch all move the right joints in the right direction.
  - Gate: FPS ≥ 20 and latency < 150 ms.
  - Any wrong direction → fix the sign in `mapping.py`, re-run the Task 4 tests, and note it.
  - Tune `MapGains` live. Gain values are the builder's call; tell Tray the final values.
- [ ] **Step 6: Notes and commit.** Add a LAB_NOTEBOOK entry with FPS, latency, gain values, and a screenshot path. Commit: `feat: no-hardware mirror app (R1 baseline) with twin and Rerun dashboard`.

---

### Task 8: Real arm I/O (Saturday evening, after the exam)

**Files:**
- Create: `src/cta/robot_io.py`, `src/cta/apps/sim_real_check.py`, `tests/test_robot_io.py`
- Modify: `src/cta/apps/mirror.py` (add the `--robot so101 --follower-port COMx --leader-port COMy --mode {mirror,paired}` path)

**Interfaces:**
- Consumes: `units.sim_to_real`/`real_to_sim`/`SimRealMap` (Task 5); `MirrorPipeline` (Task 7).
- Produces:
  - `class ArmIO(Protocol)`: `connect()`, `send(q: np.ndarray) -> None`, `read() -> np.ndarray | None` (rad), `close()`
  - `NullArm(ArmIO)`: send is a no-op; read returns the last sent value
  - `FollowerArm(ArmIO)`: wraps `SO101Follower(SO101FollowerConfig(port=..., id="cta_follower", max_relative_target=10.0))`
  - `LeaderArm`: `connect()`, `read() -> np.ndarray` (rad, via `SO101Leader(SO101LeaderConfig(port=..., id="cta_leader")).get_action()`), `close()`
  - `clamp_margin(q: np.ndarray, margin_rad: float = 0.087) -> np.ndarray`: keeps commands 5° inside `SIM_LIMITS`
  - In `--mode paired`, the follower is commanded from the leader (true teleop) while the camera pipeline runs in parallel, so its q is logged but not sent.

- [ ] **Step 1: Hardware bring-up with LeRobot CLIs (Tray, no code).** These are the LeRobot 0.4.4 entry points; confirm the exact flag names with `--help`.
  - `uv run lerobot-find-port` for each arm.
  - `uv run lerobot-setup-motors` only if the kit's motor IDs aren't set (the assembled kit should already have them).
  - `uv run lerobot-calibrate --robot.type=so101_follower --robot.port=COMx --robot.id=cta_follower`
  - `uv run lerobot-calibrate --teleop.type=so101_leader --teleop.port=COMy --teleop.id=cta_leader`
  - `uv run lerobot-teleoperate --robot.type=so101_follower --robot.port=COMx --robot.id=cta_follower --teleop.type=so101_leader --teleop.port=COMy --teleop.id=cta_leader`
  - **Expected: moving the leader moves the follower (the spec's Oct 17 gate).** If native Windows serial fails, stop and bring D-07 to Tray (Pi 5 or WSL with usbipd).
- [ ] **Step 2: Write the failing tests.**
```python
def test_clamp_margin_keeps_inside(): ...               # q at the limits → result 0.087 rad inside
def test_null_arm_read_returns_last_sent(): ...
```
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement** `robot_io.py` and the mirror `so101` path. Commands go `clamp_margin` → `sim_to_real` → `send_action`.
- [ ] **Step 5: Run them.** Expected: PASS.
- [ ] **Step 6: Sim↔real alignment (Tray + tool).** `uv run python -m cta.apps.sim_real_check --leader-port COMy` shows the twin driven by `real_to_sim(leader)`. Tray moves the leader joint by joint; for any joint where the twin moves the opposite way or sits offset, the tool's prompts set `sign`/`offset_deg` and save `sim_real_map.json`. Expected: the twin visually matches the leader at 3 poses. Note the values.
- [ ] **Step 7: Live check.** Run `uv run cta-mirror --robot so101 --follower-port COMx`. Press `c`, then move slowly. Expected: the real arm mirrors the hand with no lunges, and it holds still when the hand leaves the frame. Hand on the e-stop habit: power switch within reach.
- [ ] **Step 8: Notes and commit.** LAB_NOTEBOOK entry: the ports, sim_real_map values, observed latency, and any Windows issues. Commit: `feat: SO-101 follower/leader I/O and sim-real alignment`.

---

### Task 9: Session recorder for paired and free-hand data (Sunday AM)

**Files:**
- Create: `src/cta/recorder.py`, `tests/test_recorder.py`
- Modify: `src/cta/apps/mirror.py` (add `--record NAME`; `r` key starts/stops a take)

**Interfaces:**
- Consumes: `HandObservation`, `HandFeatures`, `MirrorPipeline` outputs, `LeaderArm.read`, `FollowerArm.read`.
- Produces (what Plan 2 trains on):
  - `class SessionRecorder`:
    - `__init__(self, root: Path, name: str, meta: dict, autosave_s: float = 30.0, video_size: tuple[int, int] = (1280, 720))`
    - `add(self, t: float, obs: HandObservation | None, feats: HandFeatures | None, q_baseline: np.ndarray, q_leader: np.ndarray | None, q_follower: np.ndarray | None, rgb: np.ndarray)`
    - `save()`; `close()`; usable as a context manager that saves in `__exit__`, even on an exception
  - After each recording block, push with `scripts/sync_data.py push --what sessions`, so the laptop can pull everything before the trip.
  - On-disk layout: `data_dir()/"sessions"/<name>/`
    - `frames.npz` with keys `t (N,)`, `hand_present (N,) bool`, `lm_img (N,21,3)`, `lm_world (N,21,3)`, `feats (N,6)` [u, v, scale, pitch, roll, pinch], `q_baseline (N,6)`, `q_leader (N,6)`, `q_follower (N,6)` (rad; NaN where absent)
    - `human.mp4` (frame i ↔ row i)
    - `meta.json` (mode paired/free, camera mode, calibration, gains, sim_real_map, git commit)
  - `load_session(path: Path) -> dict[str, np.ndarray]`

- [ ] **Step 1: Write the failing tests.**
```python
def test_roundtrip_with_missing_leader(tmp_path): ...   # add 10 frames, leader None → q_leader all NaN, shapes OK
def test_no_hand_rows_are_nan(tmp_path): ...            # obs None → hand_present False, lm_img NaN
def test_recorder_saves_on_exception(tmp_path):
    # with SessionRecorder(...) as r: add 5 frames; raise → frames.npz exists with N == 5
def test_autosave_writes_partial(tmp_path): ...         # autosave_s=0 → file exists after the first add
```
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Write the video with `cv2.VideoWriter` (mp4v) at `video_size`. Autosave rewrites `frames.npz` atomically (temp file + `os.replace`).
- [ ] **Step 4: Run them.** Expected: 4 PASS.
- [ ] **Step 5: Record the training data (Tray). Sunday AM.**
  - **Paired:** `uv run cta-mirror --robot so101 --mode paired --record paired_01 ...`. Tray moves the leader by its handle through the whole workspace while facing the camera, in 5 × 5-minute takes covering slow, fast, reach, tilt, and roll.
  - **Free-hand:** `--mode mirror --record free_01`, 3 × 3-minute takes.
  - **Expected:** `load_session` on each take shows `hand_present` ≥ 90%, and `q_leader` is non-NaN for paired takes.
- [ ] **Step 6: Notes and commit.** LAB_NOTEBOOK entry: take list, durations, hand-present %, and anything odd. Commit: `feat: session recorder for paired and free-hand data`.

---

## After this plan

- Sunday AM, Plan 2 trains retargeting network v1 on `sessions/paired_*` (+ distillation labels from `q_baseline` on `free_*`).
- Sunday ~2 PM: **Tray's checkpoint** (spec §6) uses Task 7/8's live mirroring quality.
- Plan 3 then records the policy episodes.
