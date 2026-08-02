# D1-01 CoppeliaSim RGB-D Simulation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` or `superpowers:subagent-driven-development`, and execute this plan Task by Task. Every behavior change follows RED → observed expected failure → minimal GREEN → regression → independent commit.

**Goal:** Deliver an isolated CoppeliaSim RGB-D simulation foundation that captures aligned BGR and metric-depth arrays from one vision sensor, derives validated intrinsics, emits safe previews and evidence, and passes a real opt-in CoppeliaSim probe.

**Architecture:** Keep the existing `vision_platform/rgbd/**` package pure. A new `vision_platform/rgbd_sim/**` adapter reuses the existing CoppeliaSim resolver and validated image-conversion conventions, but owns one explicit RGB-D transaction so it can read live sensor parameters and normalize a declared source-depth model before constructing the existing RGB-D contracts. It binds each capture to sensor/scene metadata and exposes no raw simulator handle. A new independent scene provides deterministic depth layers. Development probe tooling saves previews and JSON outside Git, but no formal experiment, student SDK, UI, CLI catalog or robot motion is added in this branch.

**Tech Stack:** Python 3.11, NumPy, OpenCV, pytest, CoppeliaSim ZeroMQ Remote API, PowerShell, Git/GitHub.

---

## 0. Required reading and non-negotiable rules

Read completely before any implementation:

- `docs/superpowers/specs/2026-08-02-v1-08-d1-01-parallel-coordination-design.md`
- `docs/superpowers/plans/2026-08-02-d1-01-coppeliasim-rgbd-foundation-plan.md`
- `docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md`
- `docs/superpowers/plans/2026-08-02-d1-rgbd-kernel-plan.md`

Rules:

- Work only on `codex/v2-2-d1-01-rgbd-sim-foundation` in a dedicated worktree from the exact gated `origin/main` in Task 1.
- Do not start from the old D1 branch or V1-08 feature branch. Do not merge/cherry-pick the V1-08 branch.
- This branch owns only `vision_platform/rgbd_sim/**`, `tests/test_rgbd_sim/**`, `simulation/rgbd_lab/**`, `tools/rgbd_lab/**`, its validation report and append-only retained/line-ending entries.
- Treat `vision_platform/rgbd/**` and `vision_platform/cameras/coppeliasim.py` as read-only prerequisites. If a defect is found, add a focused failing regression and report an integration recommendation; do not silently cross the ownership boundary.
- Do not modify `vision_platform/student/**`, `vision_platform/experiments/**`, UI, formal catalog/course CLI, `tools/vision_lab/**` PowerShell, V1-08 files, existing formal scenes, URDF, STL, meshes or robot assets. New dedicated build/probe PowerShell under owned `tools/rgbd_lab/**` is explicitly allowed.
- No robot motion, suction, formal D1-01 course registration, student API, PyQt or hardware integration is in scope.
- Use `apply_patch` for source/text edits. Generate the new `.ttt` only with a committed reproducible tool and a dedicated CoppeliaSim process/port.
- RGB is BGR `uint8 HxWx3`; CoppeliaSim source depth is aligned `float32 HxW` in metres; only explicitly validated/normalized optical-axis Z may enter the existing `RgbdFrame.depth_m`; arrays exposed to pure code are copied, C-contiguous and immutable.
- One capture means the sensor is configured with `explicit_handling=true`, the adapter verifies `getExplicitHandling(sensor)==1`, and exactly one `handleVisionSensor(sensor)` occurs before RGB/depth reads. Non-explicit sensors fail closed. RGB and depth must come from that same cycle and sensor path.
- Do not claim normalized buffer values are metres. Use the current CoppeliaSim metric-depth option, record whether its source model is `optical_z` or `ray_range`, and prove that model online with center/off-axis known-plane geometry before producing optical-Z depth.
- Add every new formal file to `RETAINED_FILES.txt` in the same Task. Append only.
- A skipped online test is not PASS. Keep teaching `PENDING_HUMAN_ACCEPTANCE` and hardware `PENDING_HARDWARE`.
- Record every RED/GREEN command in the validation report; stop only for a real blocker or scope expansion.

Use the worktree-local interpreter:

```powershell
.\.venv-vision\Scripts\python.exe
```

## Task 1: Gate the published baseline and create the isolated worktree

**Files:**

- Verify: `docs/superpowers/specs/2026-08-02-v1-08-d1-01-parallel-coordination-design.md`
- Verify: `docs/superpowers/plans/2026-08-02-d1-01-coppeliasim-rgbd-foundation-plan.md`
- Verify: `vision_platform/cameras/coppeliasim.py`
- Verify: `vision_platform/rgbd/models.py`
- Verify: `tests/test_vision_platform/test_coppeliasim_camera.py`
- Verify: `tests/test_rgbd/test_pipeline_contract.py`

- [ ] **Step 1: Fetch and compare the exact base SHA with the dispatched Prompt**

```powershell
git fetch origin --prune
git status --short --branch
git rev-parse origin/main
```

- [ ] **Step 2: Prove prerequisites are present on that exact commit**

```powershell
git cat-file -e origin/main:docs/superpowers/specs/2026-08-02-v1-08-d1-01-parallel-coordination-design.md
git cat-file -e origin/main:docs/superpowers/plans/2026-08-02-d1-01-coppeliasim-rgbd-foundation-plan.md
git cat-file -e origin/main:vision_platform/cameras/coppeliasim.py
git cat-file -e origin/main:vision_platform/rgbd/models.py
git cat-file -e origin/main:tests/test_vision_platform/test_coppeliasim_camera.py
git cat-file -e origin/main:tests/test_rgbd/test_pipeline_contract.py
```

Expected: all exit 0. Otherwise report `BASELINE_GATE_NOT_READY` and stop; do not substitute a candidate branch.

- [ ] **Step 3: Create the dedicated worktree**

```powershell
git worktree add `
  C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-01-rgbd-sim-foundation `
  -b codex/v2-2-d1-01-rgbd-sim-foundation origin/main
Set-Location C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-01-rgbd-sim-foundation
git rev-parse HEAD
git status --short --branch
```

- [ ] **Step 4: Establish and verify the untracked worktree-local environment junction**

Git worktrees do not carry the ignored virtual environment. Link only to the already validated shared environment:

```powershell
$sharedVisionEnv = 'C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\main-release\.venv-vision'
$sharedPython = Join-Path $sharedVisionEnv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $sharedPython)) { throw 'ENVIRONMENT_GATE_NOT_READY: shared .venv-vision is missing' }
if (-not (Test-Path -LiteralPath '.\.venv-vision')) {
  New-Item -ItemType Junction -Path '.\.venv-vision' -Target $sharedVisionEnv | Out-Null
}
if (-not (Test-Path -LiteralPath '.\.venv-vision\Scripts\python.exe')) { throw 'ENVIRONMENT_GATE_NOT_READY: worktree junction is invalid' }
.\.venv-vision\Scripts\python.exe -c "import cv2, numpy, pytest; print('VISION_ENV_READY')"
```

Expected: `VISION_ENV_READY`. If an existing `.venv-vision` path is invalid, stop; do not delete or overwrite it automatically.

- [ ] **Step 5: Run full static baseline**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: zero failures; record skips separately.

- [ ] **Step 6: Record the gate without an empty commit**

Record exact SHA, interpreter, worktree and test result for the Task 9 validation report.

## Task 2: Define immutable simulator capture metadata and intrinsics derivation

**Files:**

- Create: `vision_platform/rgbd_sim/__init__.py`
- Create: `vision_platform/rgbd_sim/errors.py`
- Create: `vision_platform/rgbd_sim/models.py`
- Create: `vision_platform/rgbd_sim/intrinsics.py`
- Create: `tests/test_rgbd_sim/__init__.py`
- Create: `tests/test_rgbd_sim/test_models.py`
- Create: `tests/test_rgbd_sim/test_intrinsics.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write strict RED tests**

Define `RgbdSensorMetadata`, `RgbdSourceCapture` and `RgbdSimCapture`. `RgbdSourceCapture` retains immutable BGR plus immutable raw `source_depth_m` and records `expected_source_depth_model` only as an untrusted scene expectation. `RgbdSimCapture` may exist only after an observation result matches that expectation; it contains `observed_source_depth_model in {optical_z, ray_range}`, `output_depth_model == optical_z` and a validated `RgbdFrame`. Test exact strings/paths, positive finite near/far with `near < far`, valid resolution, perspective angle `(0, pi)`, nonnegative integer sequence ID, finite timestamp, 64-char lowercase scene SHA, required conversion/unit conventions, matching dimensions, source immutability and no simulator object/handle in public serialization.

Use this exact pinhole contract for perspective sensors:

```python
def expected_intrinsics(width: int, height: int, angle: float) -> tuple[float, float]:
    if width >= height:
        fx = width / (2.0 * math.tan(angle / 2.0))
        fy = fx
    else:
        fy = height / (2.0 * math.tan(angle / 2.0))
        fx = fy
    return fx, fy
```

Principal point is `((width - 1) / 2, (height - 1) / 2)` under the existing `integer_center_top_left_zero` convention. Cover 640x480, 480x640 and 512x512 plus booleans, zero, NaN/Inf and illegal angle.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_models.py tests/test_rgbd_sim/test_intrinsics.py
```

- [ ] **Step 3: Implement frozen strict contracts**

Use existing `CameraIntrinsics` and `RgbdFrame`; do not duplicate or loosen them. `source_capture_to_dict()` and `capture_to_dict()` return JSON-native scalar metadata/sample summaries only, not full arrays.

- [ ] **Step 4: Run GREEN and pure RGB-D regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd_sim/test_models.py `
  tests/test_rgbd_sim/test_intrinsics.py `
  tests/test_rgbd/test_models.py `
  tests/test_rgbd/test_geometry.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/rgbd_sim tests/test_rgbd_sim/test_models.py tests/test_rgbd_sim/test_intrinsics.py tests/test_rgbd_sim/__init__.py RETAINED_FILES.txt
git commit -m "feat(rgbd-sim): define capture and intrinsics contracts"
```

## Task 3: Implement one-cycle CoppeliaSim RGB-D acquisition

**Files:**

- Create: `vision_platform/rgbd_sim/capture.py`
- Create: `vision_platform/rgbd_sim/depth_model.py`
- Create: `tests/test_rgbd_sim/test_capture.py`
- Create: `tests/test_rgbd_sim/test_depth_model.py`
- Modify: `vision_platform/rgbd_sim/__init__.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write fake-simulator RED tests**

The fake records call order. Require open/resolve, exact sensor path, `getExplicitHandling(sensor)==1`, exactly one `handleVisionSensor(sensor)`, then one RGB and one `getVisionSensorDepth(sensor, 1)` call, perspective/resolution/RGB-ignore/depth-ignore/near/far parameter reads, scene path/hash injection, shared resolution, bottom-up vertical flip, RGB-to-BGR, little-endian float32 source depth, explicit source-depth model, immutable optical-Z output and monotonic sequence.

```python
def test_source_capture_uses_one_handling_cycle_and_metric_depth_option():
    sim = FakeSim(explicit=True)
    source = CoppeliaRgbdCapture(
        sim=sim,
        sensor_path="/RgbdLab/CameraRig/RgbdSensor",
        scene_binding=_binding(expected_source_depth_model="optical_z"),
    ).read_source()
    assert sim.calls.count(("handleVisionSensor", sim.sensor)) == 1
    assert ("getVisionSensorDepth", sim.sensor, 1) in sim.calls
    assert source.image_bgr.dtype == np.uint8
    assert source.source_depth_m.dtype == np.float32
    assert source.image_bgr.shape[:2] == source.source_depth_m.shape
```

Add exact observation/normalization tests. `observe_source_depth_model(source, anchor_contract)` compares raw center/off-axis samples with known planar optical Z/range expectations and returns one unambiguous observed model or a structured failure; the scene declaration is never used as the classifier answer. After observation matches the declared expectation, `normalize_source_capture(source, observed_model)` uses copied identity for `optical_z` or `z = range / sqrt(1 + x_n**2 + y_n**2)` for `ray_range`; zero remains zero; inputs stay unchanged. Reject ambiguous/tolerance-exceeding anchors, declaration mismatch, bad model, shape, dtype, NaN/Inf and negative values. Also reject empty/short/long buffers, bad/changed resolution, `getExplicitHandling != 1`, zero or two handling calls, non-perspective sensor, ignored RGB/depth flags, missing sensor, invalid parameter types and color-depth resolution mismatch. Test close idempotency and resolver ownership.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_capture.py tests/test_rgbd_sim/test_depth_model.py
```

- [ ] **Step 3: Implement the adapter without changing existing camera code**

Composition may use shared conversion helpers only inside the new package. Read `visionintparam_perspective_operation`, resolution, `visionintparam_rgbignored`, `visionintparam_depthignored`, `visionfloatparam_perspective_angle`, `visionfloatparam_near_clipping` and `visionfloatparam_far_clipping`. `read_source()` must return raw immutable metric source depth; only `observe_source_depth_model` followed by `normalize_source_capture` may construct `RgbdFrame`. Return no raw `sim`, client or sensor handle in either public capture.

- [ ] **Step 4: Run GREEN and camera/pure-kernel regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd_sim/test_capture.py `
  tests/test_rgbd_sim/test_depth_model.py `
  tests/test_vision_platform/test_coppeliasim_camera.py `
  tests/test_rgbd/test_pipeline_contract.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/rgbd_sim/capture.py vision_platform/rgbd_sim/depth_model.py vision_platform/rgbd_sim/__init__.py tests/test_rgbd_sim/test_capture.py tests/test_rgbd_sim/test_depth_model.py RETAINED_FILES.txt
git commit -m "feat(rgbd-sim): capture aligned metric RGB-D frames"
```

## Task 4: Build the independent deterministic RGB-D scene

**Files:**

- Create: `tools/rgbd_lab/build_rgbd_scene.py`
- Create: `tools/rgbd_lab/build_rgbd_scene.ps1`
- Create: `simulation/rgbd_lab/scene_spec.json`
- Create: `simulation/rgbd_lab/BL23_rgbd_lab.ttt`
- Create: `simulation/rgbd_lab/scene_manifest.json`
- Create: `simulation/rgbd_lab/profiles.json`
- Create: `tests/test_rgbd_sim/test_scene_contract.py`
- Create: `tests/test_rgbd_sim/test_scene_builder.py`
- Modify: `.gitattributes`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write static scene RED tests**

Require these exact paths:

```python
REQUIRED = {
    "/RgbdLab",
    "/RgbdLab/CameraRig/RgbdSensor",
    "/RgbdLab/ReferencePlane",
    "/RgbdLab/Targets/near_block",
    "/RgbdLab/Targets/far_block",
    "/RgbdLab/Targets/step_low",
    "/RgbdLab/Targets/step_high",
    "/RgbdLab/ProbeAnchors/center",
    "/RgbdLab/ProbeAnchors/off_axis_left",
    "/RgbdLab/ProbeAnchors/off_axis_right",
}
```

`scene_spec.json` must be a strict standalone input to the dedicated builder. Manifest must bind template and new scene SHA, fixed square `256x256` resolution, 60-degree maximum opening angle, `explicit_handling=true`, perspective mode, RGB/depth enabled flags, near/far, expected source-depth model, sensor pose, object surface geometry, center/off-axis anchors, fixed validation ROIs and expected relative depth order. Fake-builder tests prove the Python builder accepts only this spec and port `23009`; the PowerShell wrapper uses owned-process PID/path/start-time cleanup on the same fixed port. Assert protected existing scenes unchanged and no tracked runtime artifacts.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_scene_builder.py tests/test_rgbd_sim/test_scene_contract.py
```

- [ ] **Step 3: Implement the deterministic builder and generate only the new scene**

The Python builder reads only `simulation/rgbd_lab/scene_spec.json`, connects only to port `23009`, starts from the approved template read-only, creates `/RgbdLab` content and the RGB-D sensor, saves to the new path and computes canonical LF JSON hashes after final save. The PowerShell wrapper composes the existing launcher/identity-checked cleanup helpers and cleans up only a process it started. Builder rerun must not alter unrelated tracked files.

Run the actual unattended build:

```powershell
& .\tools\rgbd_lab\build_rgbd_scene.ps1 -Port 23009 -Hidden
```

- [ ] **Step 4: Run GREEN plus protected-asset regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd_sim/test_scene_contract.py `
  tests/test_rgbd_sim/test_scene_builder.py `
  tests/test_simulation/test_v1_07_scene_contract.py `
  tests/test_acceptance/test_delivery_contract.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add tools/rgbd_lab/build_rgbd_scene.py tools/rgbd_lab/build_rgbd_scene.ps1 simulation/rgbd_lab tests/test_rgbd_sim/test_scene_builder.py tests/test_rgbd_sim/test_scene_contract.py .gitattributes RETAINED_FILES.txt
git commit -m "feat(rgbd-sim): add independent RGB-D lab scene"
```

## Task 5: Add safe RGB and depth preview generation

**Files:**

- Create: `vision_platform/rgbd_sim/preview.py`
- Create: `tests/test_rgbd_sim/test_preview.py`
- Modify: `vision_platform/rgbd_sim/__init__.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write preview RED tests**

Require copied BGR preview and deterministic `uint8 HxWx3` depth colorization. Zero depth uses an explicit invalid color; valid scaling uses bounded fixed or robust percentiles without modifying source; one valid value, all zero, narrow range and extreme finite values are handled. Reject NaN/Inf/negative/wrong dtype and never expose raw arrays through JSON.

```python
def test_zero_depth_has_explicit_invalid_colour_and_input_is_unchanged():
    depth = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    before = depth.copy()
    preview = colorize_depth(depth, minimum_m=0.2, maximum_m=1.2)
    assert tuple(preview[0, 0]) == (255, 0, 255)
    np.testing.assert_array_equal(depth, before)
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_preview.py
```

- [ ] **Step 3: Implement pure preview functions**

Use OpenCV only for color map/PNG encoding at the tool boundary. Package functions return arrays and numeric summaries, not files.

- [ ] **Step 4: Run GREEN and performance sanity**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_preview.py tests/test_rgbd/test_performance.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/rgbd_sim/preview.py vision_platform/rgbd_sim/__init__.py tests/test_rgbd_sim/test_preview.py RETAINED_FILES.txt
git commit -m "feat(rgbd-sim): render deterministic depth previews"
```

## Task 6: Add scene binding, ROI evidence and JSON-native probe summaries

**Files:**

- Create: `vision_platform/rgbd_sim/scene_binding.py`
- Create: `vision_platform/rgbd_sim/probe.py`
- Create: `tests/test_rgbd_sim/test_scene_binding.py`
- Create: `tests/test_rgbd_sim/test_probe.py`
- Modify: `vision_platform/rgbd_sim/__init__.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: RED-test manifest security and evidence semantics**

Reject path escape, symlink, wrong scene hash, unknown/missing/overlapping/out-of-bounds ROI, non-finite expectations and extra keys. Probe summaries contain exact scene/sensor IDs, declared/observed source-depth model, output model, intrinsics, sequence, min/median/max valid depth, invalid count and per-ROI median/count. Validate `near_block < far_block` and `step_high < step_low` by configured positive margins. Center/off-axis plane samples must distinguish `optical_z` from `ray_range` within configured simulation tolerance and match the manifest declaration; do not overclaim real-camera accuracy.

```python
def test_probe_proves_configured_virtual_depth_order():
    report = build_probe_report(_capture(), _binding())
    assert report.status == "PASS"
    assert report.observed_source_depth_model == report.expected_source_depth_model
    assert report.output_depth_model == "optical_z"
    assert report.rois["near_block"].median_depth_m < report.rois["far_block"].median_depth_m
    json.dumps(probe_report_to_dict(report), allow_nan=False)
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_scene_binding.py tests/test_rgbd_sim/test_probe.py
```

- [ ] **Step 3: Implement strict loader and in-memory probe**

Only `scene_binding.py` reads the manifest. Copy data into frozen models; no arbitrary callbacks, code evaluation or dynamic imports.

- [ ] **Step 4: Run GREEN and serialization regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd_sim/test_scene_binding.py `
  tests/test_rgbd_sim/test_probe.py `
  tests/test_rgbd/test_serialization.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/rgbd_sim/scene_binding.py vision_platform/rgbd_sim/probe.py vision_platform/rgbd_sim/__init__.py tests/test_rgbd_sim/test_scene_binding.py tests/test_rgbd_sim/test_probe.py RETAINED_FILES.txt
git commit -m "feat(rgbd-sim): bind captures to scene evidence"
```

## Task 7: Add a bounded development probe CLI and artifact policy

**Files:**

- Create: `tools/rgbd_lab/run_rgbd_probe.py`
- Create: `tests/test_rgbd_sim/test_probe_cli.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write CLI RED tests**

Allow only fixed/validated options: scene path under the repository, known sensor path default, port range, timeout range and explicit output directory outside tracked roots. Reject path escape, tracked output roots, booleans/non-numbers, unknown options and overwrite without explicit flag. On success write `rgb.png`, `depth.png`, `report.json`; on failure write a bounded JSON error and return nonzero. Never kill unrelated processes.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_probe_cli.py
```

- [ ] **Step 3: Implement the thin CLI**

The CLI composes binding, capture, preview and probe only. It does not join the formal experiment CLI or student runner. JSON uses UTF-8, sorted keys and `allow_nan=False`.

- [ ] **Step 4: Run GREEN and verify no runtime artifacts are tracked**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_probe_cli.py
git ls-files | rg "(^|/)(artifacts|evidence|logs|__pycache__|\.pytest_cache)(/|$)"
git diff --check
```

Expected: test passes; tracked-runtime search returns no newly introduced artifact path.

- [ ] **Step 5: Commit**

```powershell
git add tools/rgbd_lab/run_rgbd_probe.py tests/test_rgbd_sim/test_probe_cli.py RETAINED_FILES.txt
git commit -m "feat(rgbd-sim): add bounded RGB-D probe tool"
```

## Task 8: Add real opt-in CoppeliaSim acceptance

**Files:**

- Create: `tests/test_rgbd_sim/conftest.py`
- Create: `tests/test_rgbd_sim/test_coppeliasim_rgbd_online.py`
- Create: `tests/test_rgbd_sim/coppeliasim_process.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write the opt-in online acceptance contract**

Because `tests/test_acceptance/conftest.py` does not apply to this sibling test directory, add a local `tests/test_rgbd_sim/conftest.py` that registers the `coppeliasim` marker and skips marked tests unless `coppeliasim` is present in `request.config.getoption("-m")`; its skip message must explicitly say a skip is not an online PASS. The test itself is marked `@pytest.mark.coppeliasim`, launches one owned CoppeliaSim process on fixed D1-01 port `23009`, rejects any unexpected listener, loads the exact scene, validates scene hash and required paths, reads back `getExplicitHandling==1`, perspective/resolution/RGB-depth flags/angle/near/far, acquires at least three captures, and proves exactly one explicit handle call per capture, same color/source-depth dimensions, finite nonnegative metres, center/off-axis source-model agreement, explicit optical-Z output, stable intrinsics, monotonic sequence, bounded frame-to-frame ROI median variation and configured depth order. It saves probe artifacts to a temporary/evidence path and terminates only its process.

Add online failure checks for wrong sensor path and tampered manifest. Do not add robot movement.

- [ ] **Step 2: Confirm explicit skip without the environment gate**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_coppeliasim_rgbd_online.py
```

Expected: explicit skip. Record it as skip, not PASS.

- [ ] **Step 3: Implement process ownership and online test**

Use exact PID/port ownership, bounded startup/teardown timeouts and `finally` cleanup. Reuse patterns by copying the minimum into this owned test helper; do not edit shared V1 process code.

- [ ] **Step 4: Run the actual online acceptance**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q -s -m coppeliasim tests/test_rgbd_sim/test_coppeliasim_rgbd_online.py
```

Expected: test runs, does not skip, and passes. If the installed API contradicts the declared source-depth, metric-depth or perspective-angle semantics, stop, document exact evidence and correct the new profile/adapter/tests; do not silently reinterpret normalized depth or send an unverified source model into `RgbdFrame`.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_rgbd_sim/conftest.py tests/test_rgbd_sim/test_coppeliasim_rgbd_online.py tests/test_rgbd_sim/coppeliasim_process.py RETAINED_FILES.txt
git commit -m "test(rgbd-sim): verify online metric RGB-D capture"
```

## Task 9: Release audit, independent review and validation report

**Files:**

- Create: `docs/validation/d1-01-rgbd-sim-foundation-validation.md`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Run focused and full static regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd_sim `
  tests/test_rgbd `
  tests/test_vision_platform/test_coppeliasim_camera.py `
  tests/test_acceptance/test_delivery_contract.py
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
```

- [ ] **Step 2: Re-run the actual online acceptance**

Use Task 8's exact opt-in command. Record CoppeliaSim version, port, scene hash, pass count, artifact path and cleanup result. Do not call a non-enabled skip PASS.

- [ ] **Step 3: Run independent review and fix P0/P1/P2**

Review metric-depth and source-model semantics, optical-Z normalization, one-cycle alignment, buffer validation, intrinsics convention, manifest/path/hash security, process ownership, protected assets, JSON-native evidence, input immutability, no shared-file violations and claim boundaries. Each functional fix starts with a failing regression and gets a separate commit.

- [ ] **Step 4: Write validation report and ownership audit**

Report exact base/tip SHA, Task commits, RED/GREEN evidence, focused/full pass-skip counts, actual online result, declared/observed source model, optical-Z normalization result, depth-order summary, protected hashes, retained paths, review fixes and limitations. State explicitly:

```text
Validated: CoppeliaSim RGB-D simulation foundation.
Not yet validated: formal D1-01 student course, D1-02/D1-03, robot motion, real depth camera, hardware, teaching effectiveness.
Hardware: PENDING_HARDWARE.
Teaching: PENDING_HUMAN_ACCEPTANCE.
```

- [ ] **Step 5: Commit the report**

```powershell
git add docs/validation/d1-01-rgbd-sim-foundation-validation.md RETAINED_FILES.txt
git commit -m "docs(rgbd-sim): record D1-01 foundation validation"
```

## Task 10: Final clean-room verification and push

**Files:**

- Verify only; modify only when a newly failing regression requires a focused fix.

- [ ] **Step 1: Prove branch ownership**

```powershell
git diff --name-only origin/main...HEAD
```

Expected: only the ownership paths listed in section 0 plus `.gitattributes` and `RETAINED_FILES.txt`. Any V1, student, experiment catalog, UI, `.gitignore` or existing scene change must be removed or returned to its owner.

- [ ] **Step 2: Re-run final static gates**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim tests/test_rgbd tests/test_acceptance/test_delivery_contract.py
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
git status --short --branch
```

- [ ] **Step 3: Verify formal files are retained and runtime artifacts are not**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd_sim/test_scene_contract.py tests/test_rgbd_sim/test_probe_cli.py
git ls-files
```

- [ ] **Step 4: Push only the feature branch**

```powershell
git push -u origin codex/v2-2-d1-01-rgbd-sim-foundation
git rev-parse HEAD
git rev-parse origin/codex/v2-2-d1-01-rgbd-sim-foundation
git status --short --branch
```

Expected: local/remote hashes identical and worktree clean.

- [ ] **Step 5: Handoff without merging main**

Report branch/tip, exact baseline, changed-file ownership audit, test results, actual online result, review fixes and remaining PENDING states. Do not merge `main`; integration waits until V1-08 is independently accepted and merged first.
