# V2.2 V1-08 OCR Sorting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` or `superpowers:subagent-driven-development`, and execute this plan Task by Task. Every behavior change follows RED → observed expected failure → minimal GREEN → regression → independent commit.

**Goal:** Deliver V1-08 as a complete CoppeliaSim teaching experiment that trains a deterministic OCR classifier from hash-bound original glyphs, recognizes four printed part identifiers, freezes a host-owned whitelist sort plan, and safely sorts all parts with same-run evidence.

**Architecture:** The host owns assets, model training, fixed ROIs, identifier-to-route mapping and the complete sort plan. The student first calls one no-argument OCR command, then may call a second host-action command whose only parameter is an approved `entry_id`; the host executes the complete finite trajectory and the student never submits coordinates. A dedicated guard blocks all motion until the four-entry plan is complete and allows each approved entry exactly once. The scene, OCR assets and evidence are independent of previous formal labs.

**Tech Stack:** Python 3.11, NumPy, OpenCV, PyQt5, pytest, CoppeliaSim ZeroMQ Remote API, PowerShell, Git/GitHub.

---

## 0. Required reading and non-negotiable rules

Read completely before any implementation:

- `docs/superpowers/specs/2026-08-02-v1-08-d1-01-parallel-coordination-design.md`
- `docs/superpowers/plans/2026-08-02-v1-08-ocr-sorting-plan.md`
- `docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md`
- `docs/superpowers/plans/2026-08-02-v1-07-code-routing-plan.md`
- `docs/superpowers/specs/2026-08-02-v1-07-v1-09-vision2d-algorithm-kernels-design.md`

Rules:

- Work only on `codex/v2-2-v1-08-ocr-sorting` in a new dedicated worktree from the exact gated `origin/main` in Task 1.
- Do not start from V1-07, D1 or another candidate feature branch. Do not merge/cherry-pick the D1-01 branch.
- Do not modify `vision_platform/rgbd/**`, `vision_platform/rgbd_sim/**`, `tests/test_rgbd/**`, `tests/test_rgbd_sim/**`, `simulation/rgbd_lab/**` or `tools/rgbd_lab/**`.
- Do not modify an existing formal `.ttt`, URDF, STL, mesh or robot asset. Generate new PNG and `.ttt` binaries only through committed deterministic tools.
- Reuse `vision_platform/vision2d/ocr.py`; do not weaken its validation or replace it with Tesseract, a network service, an external model or a system font.
- The student recognition command `vision2d.ocr_sorting` has no arguments. The only action command is `vision2d.ocr_sort_entry` with exact schema `{"entry_id": <approved id>}`; it accepts no path, ROI, threshold, method, identifier, route, coordinate, payload or command string.
- OCR text is inert data and only an exact whitelist key. Never evaluate it or pass it to Python, Lua, shell, path APIs, arbitrary CoppeliaSim calls or raw robot commands.
- No motion before the complete four-entry plan is atomically validated and activated.
- V1-08 does not publish raw `robot.*` or `tool.*` capabilities. Direct student requests for `robot.move_world`, `robot.home`, `robot.pose`, `tool.on` or `tool.off` fail closed in every V1-08 state; only the runner's private OCR-entry executor may touch devices.
- Use `apply_patch` for text/source edits. Add every new formal file to `RETAINED_FILES.txt` in the same Task; append only.
- A skipped online test is not PASS. Keep teaching `PENDING_HUMAN_ACCEPTANCE` and hardware `PENDING_HARDWARE`.
- Record each RED/GREEN command and result in the validation report; stop only for a real blocker or scope expansion.

Use the worktree-local interpreter:

```powershell
.\.venv-vision\Scripts\python.exe
```

## Task 1: Gate the published baseline and create the isolated worktree

**Files:**

- Verify: `docs/superpowers/specs/2026-08-02-v1-08-d1-01-parallel-coordination-design.md`
- Verify: `docs/superpowers/plans/2026-08-02-v1-08-ocr-sorting-plan.md`
- Verify: `vision_platform/vision2d/ocr.py`
- Verify: `tests/test_vision2d/test_ocr.py`
- Verify: `config/experiments/V1-07.json`
- Verify: `vision_platform/rgbd/__init__.py`

- [ ] **Step 1: Fetch and record the exact published base**

```powershell
git fetch origin --prune
git status --short --branch
git rev-parse origin/main
```

Expected: source checkout clean. Compare the SHA with the exact value in the dispatched short Prompt.

- [ ] **Step 2: Prove all prerequisite artifacts exist on that exact commit**

```powershell
git cat-file -e origin/main:docs/superpowers/specs/2026-08-02-v1-08-d1-01-parallel-coordination-design.md
git cat-file -e origin/main:docs/superpowers/plans/2026-08-02-v1-08-ocr-sorting-plan.md
git cat-file -e origin/main:vision_platform/vision2d/ocr.py
git cat-file -e origin/main:tests/test_vision2d/test_ocr.py
git cat-file -e origin/main:config/experiments/V1-07.json
git cat-file -e origin/main:vision_platform/rgbd/__init__.py
```

Expected: every command exits 0. Otherwise report `BASELINE_GATE_NOT_READY` and stop.

- [ ] **Step 3: Create the branch/worktree only after the gate passes**

```powershell
git worktree add `
  C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-08-ocr-sorting `
  -b codex/v2-2-v1-08-ocr-sorting origin/main
Set-Location C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-08-ocr-sorting
git rev-parse HEAD
git status --short --branch
```

- [ ] **Step 4: Establish and verify the untracked worktree-local environment junction**

Git worktrees do not carry the ignored virtual environment. Link only to the already validated shared environment; never create an empty environment or judge the baseline with another Python:

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

Expected: `VISION_ENV_READY`. If an existing `.venv-vision` path lacks the interpreter, stop; do not delete or overwrite it automatically.

- [ ] **Step 5: Run the full static baseline**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: zero failures. Record pass and skip counts separately.

- [ ] **Step 6: Record the gate without an empty commit**

Start `docs/validation/v1-08-ocr-sorting-validation.md` locally only when Task 2 adds it formally. Record base SHA, worktree, interpreter and baseline result.

## Task 2: Define the host-owned OCR asset and sort-plan contracts

**Files:**

- Create: `vision_platform/experiments/ocr_sorting.py`
- Create: `tests/test_experiments/test_ocr_sorting.py`
- Modify: `vision_platform/experiments/__init__.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write strict RED tests**

The successful contract must use exactly these identifiers and routes:

```python
ROUTES = (
    {"entry_id": "entry_a", "part_id": "part_a", "identifier": "A1", "route_id": "route_alpha"},
    {"entry_id": "entry_b", "part_id": "part_b", "identifier": "A2", "route_id": "route_alpha"},
    {"entry_id": "entry_c", "part_id": "part_c", "identifier": "B1", "route_id": "route_beta"},
    {"entry_id": "entry_d", "part_id": "part_d", "identifier": "B2", "route_id": "route_beta"},
)
```

Test exact schemas, deterministic entry ordering, JSON-native serialization, finite built-in numbers, complete expected set, unique identifier/part/entry/drop slot, route/part binding, model report threshold, exact two-character PASS result, per-character confidence, ROI bounds, source non-mutation and rejection of extra fields, booleans, NaN/Inf, missing/duplicate/unknown text, low confidence and mismatched scene sets.

```python
def test_build_plan_requires_all_four_results_before_activation():
    plan = build_ocr_sort_plan(_config(), _results(), scene_part_ids={"part_a", "part_b", "part_c", "part_d"})
    assert [entry.identifier for entry in plan.entries] == ["A1", "A2", "B1", "B2"]
    assert [entry.route_id for entry in plan.entries] == ["route_alpha", "route_alpha", "route_beta", "route_beta"]
    assert ocr_sort_plan_to_dict(plan)["status"] == "PASS"


@pytest.mark.parametrize("mutator", [
    lambda results: results[:-1],
    lambda results: (*results[:-1], replace(results[-1], text="A1")),
    lambda results: (*results[:-1], replace(results[-1], text="Z9")),
])
def test_incomplete_duplicate_or_unknown_identifier_fails_closed(mutator):
    with pytest.raises(OcrSortError):
        build_ocr_sort_plan(_config(), mutator(_results()), scene_part_ids={"part_a", "part_b", "part_c", "part_d"})
```

- [ ] **Step 2: Run RED and confirm the missing-module failure**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_ocr_sorting.py
```

- [ ] **Step 3: Implement frozen dataclasses and pure validation**

Implement `OcrAssetSpec`, `OcrRouteSpec`, `ApprovedOcrSortEntry`, `OcrSortPlan`, `OcrSortError`, `validate_training_report`, `build_ocr_sort_plan` and `ocr_sort_plan_to_dict`. No file, camera, simulator or robot access belongs in this module.

- [ ] **Step 4: Run GREEN and related regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_ocr_sorting.py `
  tests/test_vision2d/test_ocr.py `
  tests/test_experiments/test_code_routing.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/experiments/ocr_sorting.py vision_platform/experiments/__init__.py tests/test_experiments/test_ocr_sorting.py RETAINED_FILES.txt
git commit -m "feat(v1-08): define OCR sorting plan contract"
```

## Task 3: Generate and verify original OCR assets

**Files:**

- Create: `tools/vision_lab/generate_ocr_assets.py`
- Create: `simulation/vision_ocr_sorting_lab/training/**`
- Create: `simulation/vision_ocr_sorting_lab/labels/**`
- Create: `simulation/vision_ocr_sorting_lab/ocr_assets_manifest.json`
- Create: `tests/test_simulation/test_v1_08_ocr_assets.py`
- Modify: `.gitattributes`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write asset RED tests**

Assert the manifest has exact keys, glyph set `A B 1 2`, at least eight training variants per glyph, exactly four label images, no path traversal/symlink, unique normalized paths, PNG decode success, fixed dimensions, exact SHA-256 and a regeneration byte-for-byte check in a temporary directory. Assert no system font, Tesseract, network or external model import is used.

```python
def test_manifest_binds_original_glyphs_and_scene_labels():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["alphabet"] == ["1", "2", "A", "B"]
    assert {item["identifier"] for item in manifest["labels"]} == {"A1", "A2", "B1", "B2"}
    assert all(_sha256(ROOT / item["path"]) == item["sha256"] for item in _all_assets(manifest))
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_simulation/test_v1_08_ocr_assets.py
```

- [ ] **Step 3: Implement the deterministic 5x7 bitmap generator**

Use a literal repository-owned bitmap table, seed `20260802`, bounded translation/scale/line-width variations and OpenCV PNG encoding. Generate held-out-capable variants without duplicating bytes. Manifest JSON is canonical UTF-8/LF with sorted paths and SHA-256.

- [ ] **Step 4: Regenerate, run GREEN and prove OCR fitness**

```powershell
.\.venv-vision\Scripts\python.exe tools/vision_lab/generate_ocr_assets.py --output simulation/vision_ocr_sorting_lab
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_simulation/test_v1_08_ocr_assets.py `
  tests/test_vision2d/test_ocr.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add tools/vision_lab/generate_ocr_assets.py simulation/vision_ocr_sorting_lab tests/test_simulation/test_v1_08_ocr_assets.py .gitattributes RETAINED_FILES.txt
git commit -m "feat(v1-08): add deterministic OCR teaching assets"
```

## Task 4: Build the independent V1-08 scene and static contract

**Files:**

- Modify: `simulation/training_scenes/build_scene.py`
- Create: `simulation/vision_ocr_sorting_lab/scene_spec.json`
- Create: `simulation/vision_ocr_sorting_lab/BL23_vision_ocr_sorting_lab.ttt`
- Create: `simulation/vision_ocr_sorting_lab/scene_manifest.json`
- Create: `simulation/vision_ocr_sorting_lab/profiles.json`
- Create: `tools/vision_lab/build_v1_08_scene.py`
- Create: `tools/vision_lab/build_v1_08_scene.ps1`
- Create: `tests/test_simulation/test_v1_08_scene_builder.py`
- Create: `tests/test_simulation/test_v1_08_scene_contract.py`
- Modify: `.gitattributes`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Add static scene RED tests**

Require a strict `scene_spec.json` registered by the existing formal builder, unique scene path/hash, protected-scene hashes unchanged, exact object paths, four pickables, four CodeFace objects, two routes with two fixed slots each, fixed 1024x1024 profile, camera/lighting paths, calibration matrix, initial positions, workspace/safe Z, reset contract, asset manifest hash and no mutation of the template. Fake-builder tests must prove the Python wrapper supplies exactly the V1-08 spec and dedicated port `23008`, while the PowerShell wrapper launches only its owned process, enforces port `23008`, invokes the Python wrapper, and performs identity-checked cleanup in `finally` without accepting arbitrary code or output paths.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_simulation/test_v1_08_scene_builder.py tests/test_simulation/test_v1_08_scene_contract.py
```

- [ ] **Step 3: Extend the builder and generate only the new scene family**

Register the new strict spec without changing existing spec semantics. The Python wrapper calls `build_scene` with only `simulation/vision_ocr_sorting_lab/scene_spec.json` and port `23008`. The PowerShell wrapper composes existing `launch_coppeliasim.ps1` and `process_ownership.ps1`, records PID/path/start time, calls the Python wrapper, writes only the new output and cleans up only a process it started. Build under `/VisionOcrSortingLab`; bind `part_a..part_d`, `route_alpha`, `route_beta`, label textures and exact route slots. Never overwrite source scenes.

Run the actual build with the committed wrapper:

```powershell
& .\tools\vision_lab\build_v1_08_scene.ps1 -Port 23008 -Hidden
```

- [ ] **Step 4: Run GREEN plus protected-asset regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_simulation/test_v1_08_scene_contract.py `
  tests/test_simulation/test_v1_08_scene_builder.py `
  tests/test_simulation/test_v1_07_scene_contract.py `
  tests/test_acceptance/test_delivery_contract.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add simulation/training_scenes/build_scene.py simulation/vision_ocr_sorting_lab tools/vision_lab/build_v1_08_scene.py tools/vision_lab/build_v1_08_scene.ps1 tests/test_simulation/test_v1_08_scene_builder.py tests/test_simulation/test_v1_08_scene_contract.py .gitattributes RETAINED_FILES.txt
git commit -m "feat(v1-08): build independent OCR sorting scene"
```

## Task 5: Add hash-bound training loader and online OCR service

**Files:**

- Create: `vision_platform/experiments/ocr_assets.py`
- Create: `vision_platform/experiments/ocr_service.py`
- Create: `tests/test_experiments/test_ocr_assets.py`
- Create: `tests/test_experiments/test_ocr_service.py`
- Modify: `vision_platform/experiments/__init__.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: RED-test path and model boundaries**

Cover containment, regular-file only, no symlink, exact bytes/hash/shape/dtype, fixed seed/method/fraction, minimum held-out accuracy, one model per session, immutable copied ROIs, exact four ROI results, deterministic order, annotated copy, source non-mutation and failures before any plan activation.

```python
def test_service_trains_once_and_returns_four_whitelisted_results(tmp_path):
    service = OcrSortingService.from_manifest(_manifest_path(), minimum_accuracy=0.95)
    output = service.analyze(_captured_frame(), _fixed_rois())
    assert service.training_report.held_out_accuracy >= 0.95
    assert [result.text for result in output.results] == ["A1", "A2", "B1", "B2"]
    assert output.plan.status == "PASS"
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_ocr_assets.py tests/test_experiments/test_ocr_service.py
```

- [ ] **Step 3: Implement minimal loader/service**

The loader is the only file reader. The service passes in-memory BGR glyphs to the existing OCR kernel and returns a frozen host-side result. Do not serialize or expose the OpenCV classifier object.

- [ ] **Step 4: Run GREEN and OCR regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_ocr_assets.py `
  tests/test_experiments/test_ocr_service.py `
  tests/test_vision2d/test_ocr.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/experiments/ocr_assets.py vision_platform/experiments/ocr_service.py vision_platform/experiments/__init__.py tests/test_experiments/test_ocr_assets.py tests/test_experiments/test_ocr_service.py RETAINED_FILES.txt
git commit -m "feat(v1-08): add controlled OCR training service"
```

## Task 6: Add the atomic sort guard and cleanup rules

**Files:**

- Create: `vision_platform/student/ocr_sort_guard.py`
- Create: `tests/test_student_programs/test_v1_08_sort_guard.py`
- Modify: `vision_platform/student/runner.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write state-machine RED tests**

Test `EMPTY -> ACTIVE -> ENTRY_ACTIVE -> ENTRY_AWAITING_PROBE -> COMPLETED/INVALIDATED`; no action plan in EMPTY; only approved unconsumed entry IDs; `begin_entry(entry_id)` returns a frozen ordered tuple of typed move/tool actions; `confirm_action(action_id)` advances exactly one step and the final action enters `ENTRY_AWAITING_PROBE`; `confirm_entry_probe(entry_id, evidence_ref)` accepts only the active entry plus a bounded same-run evidence reference and atomically marks it consumed; probe failure calls `fail_entry` and invalidates the plan. Duplicate/unknown entry, out-of-order/duplicate confirmation, probe-before-actions and concurrent entry are rejected; stop/reset invalidates; failure retains the first error. The guard is pure state: it must not import, store or call robot/tool/application/gateway objects.

```python
def test_motion_is_impossible_before_atomic_plan_activation():
    guard = OcrSortGuard()
    with pytest.raises(OcrSortGuardError, match="OCR_SORT_PLAN_NOT_ACTIVE"):
        guard.begin_entry("entry_a")


def test_guard_returns_frozen_actions_without_device_access():
    guard = _active_guard()
    actions = guard.begin_entry("entry_a")
    assert [action.kind for action in actions] == [
        "move_safe_pick", "move_pick", "tool_on", "move_safe_pick",
        "move_safe_drop", "move_drop", "tool_off", "move_safe_drop",
    ]
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_student_programs/test_v1_08_sort_guard.py
```

- [ ] **Step 3: Implement the dedicated guard**

Keep the V1-07 guard unchanged. Store only frozen plan state, active action IDs, consumed IDs, bounded evidence references and terminal status. `begin_entry` receives a strict approved ID internally and derives typed actions from the frozen plan; `confirm_action`, `confirm_entry_probe` and `fail_entry` only advance state. Never accept caller coordinates and never pass robot/tool/sim objects into this module. Runner integration and per-step safety belong to Task 7.

- [ ] **Step 4: Run GREEN and runner safety regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_v1_08_sort_guard.py `
  tests/test_student_programs/test_v1_07_route_guard.py `
  tests/test_student_programs/test_runner.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/student/ocr_sort_guard.py vision_platform/student/runner.py tests/test_student_programs/test_v1_08_sort_guard.py RETAINED_FILES.txt
git commit -m "feat(v1-08): guard atomic OCR sort execution"
```

## Task 7: Integrate capability, protocol, SDK and experiment gateway

**Files:**

- Modify: `vision_platform/experiments/capabilities.py`
- Modify: `vision_platform/student/protocol.py`
- Modify: `vision_platform/student/sdk.py`
- Modify: `vision_platform/student/experiment_gateway.py`
- Modify: `vision_platform/student/runner.py`
- Create: `tests/test_student_programs/test_v1_08_protocol_sdk.py`
- Create: `tests/test_student_programs/test_v1_08_gateway.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: RED-test both finite public contracts**

Add capability `vision2d.ocr_sorting`, no-argument command `vision2d.ocr_sorting`, and action command `vision2d.ocr_sort_entry`. The capability checker verifies host sim/camera/robot/tool readiness internally, but V1-08's published experiment capabilities exclude every `robot.*` and `tool.*` command. Recognition rejects all request params and all unknown/extra/non-native response fields. Action accepts exactly one object field `entry_id`, requires a bounded ASCII ID already present and unconsumed in the active plan, and rejects missing/extra fields, booleans, numbers, unknown IDs and duplicate consumption before any motion. Validate schema version, training report, snapshot/bundle references, exactly four results, character boxes/confidences, plan entries and status. Test deep-copy isolation. Add runner tests proving each generated action passes through existing motion validation, `_raise_if_stopping`, pause/continue/single-step, pose/event updates and evidence; stop between any two actions prevents the next device call; cleanup attempts tool-off without replacing the primary error. Parameterize direct raw `robot.move_world`, `robot.home`, `robot.pose`, `tool.on` and `tool.off` requests across EMPTY, ACTIVE, ENTRY_ACTIVE, COMPLETED and INVALIDATED, asserting the same fake robot/tool instances receive zero calls.

```python
def test_student_command_is_no_argument_and_json_native():
    result = ctx.vision2d.ocr_sorting()
    assert result.training.held_out_accuracy >= 0.95
    assert tuple(item.identifier for item in result.entries) == ("A1", "A2", "B1", "B2")
    json.dumps(result.to_dict(), allow_nan=False)


def test_student_selects_only_an_approved_entry_id():
    ctx.vision2d.ocr_sorting()
    receipt = ctx.vision2d.sort_ocr_entry("entry_c")
    assert receipt.entry_id == "entry_c"
    assert receipt.status == "COMPLETED"


def test_runner_never_touches_devices_before_plan_activation():
    robot = _fake_robot()
    tool = _fake_tool()
    runner = _runner(robot=robot, tool=tool, ocr_sort_guard=OcrSortGuard())
    with pytest.raises(RuntimeError, match="OCR_SORT_PLAN_NOT_ACTIVE"):
        runner.dispatch({"command": "vision2d.ocr_sort_entry", "args": {"entry_id": "entry_a"}})
    assert robot.moves == []
    assert tool.events == []


@pytest.mark.parametrize("state", ["EMPTY", "ACTIVE", "ENTRY_ACTIVE", "COMPLETED", "INVALIDATED"])
@pytest.mark.parametrize("command", ["robot.move_world", "robot.home", "robot.pose", "tool.on", "tool.off"])
def test_v1_08_never_exposes_raw_device_commands(state, command):
    robot = _fake_robot()
    tool = _fake_tool()
    runner = _v1_08_runner_in_state(state, robot=robot, tool=tool)
    with pytest.raises(RuntimeError, match="COMMAND_NOT_ALLOWED"):
        runner.dispatch(_well_formed_raw_request(command))
    assert robot.events == []
    assert tool.events == []
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_student_programs/test_v1_08_protocol_sdk.py tests/test_student_programs/test_v1_08_gateway.py
```

- [ ] **Step 3: Implement minimum integration**

Gateway handles only the recognition/planning half: it captures the fixed profile once, verifies manifest/scene bindings, trains/analyses host-side, saves raw/annotated evidence, activates the runner-owned guard through a narrow callback and returns only a copied public payload. Runner owns `_command_ocr_sort_entry`: it validates the strict ID, requests frozen typed actions from the guard, and executes each action through private narrow device helpers that reuse the same motion guard, stop/pause/single-step, pose/event and evidence logic as normal commands without redispatching through the public protocol. Before every action it enforces stop, pause and single-step; after every successful action it records evidence and confirms exactly that action to the guard. After the last action, runner invokes a read-only per-entry scene probe and verifies the same run ID, scene hash, part ID, route ID and configured slot before calling `confirm_entry_probe`; only then may it return `COMPLETED` and mark the entry consumed. On action/probe failure it invalidates the entry/plan and performs bounded tool-off/safe-home cleanup without masking the first error. The V1-08 dispatcher rejects public raw robot/tool commands in every state. Gateway and guard never receive raw robot/tool/sim objects; no coordinate appears in the public action request, and OCR text is never executed as a command.

- [ ] **Step 4: Run GREEN and V1-06/V1-07 regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_v1_08_protocol_sdk.py `
  tests/test_student_programs/test_v1_08_gateway.py `
  tests/test_student_programs/test_v1_06_gateway.py `
  tests/test_student_programs/test_v1_07_gateway.py `
  tests/test_student_programs/test_sdk.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/experiments/capabilities.py vision_platform/student/protocol.py vision_platform/student/sdk.py vision_platform/student/experiment_gateway.py vision_platform/student/runner.py tests/test_student_programs/test_v1_08_protocol_sdk.py tests/test_student_programs/test_v1_08_gateway.py RETAINED_FILES.txt
git commit -m "feat(v1-08): expose controlled OCR sorting command"
```

## Task 8: Publish V1-08 definition, template, guide and static material contract

**Files:**

- Create: `config/experiments/V1-08.json`
- Modify: `config/experiments/catalog.json`
- Create: `student_programs/templates/v1_08_ocr_sorting.py`
- Create: `docs/experiments/V1-08.md`
- Create: `tests/test_vision_quality/test_v1_08_materials.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write material RED tests**

Require exact paths, capabilities, workspace, fixed profile, four routes, training manifest/hash, acceptance fields, `PENDING_HARDWARE`, guide/template cross-links and ordered catalog entry. Assert capabilities include `vision2d.ocr_sorting` but no `robot.*` or `tool.*`. Assert the template calls `ocr_sorting()` first, validates all four entries, then calls only `sort_ocr_entry(entry_id)` for its chosen order; it must not submit world coordinates or directly drive robot/tool methods. The host action handles tool-off and final home requirements.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision_quality/test_v1_08_materials.py
```

- [ ] **Step 3: Add complete Chinese teaching material**

The guide covers objectives, train/test split, character segmentation, confidence, whitelist plan, failure analysis, safe reset, evidence interpretation and questions. Do not claim human teaching acceptance.

- [ ] **Step 4: Run GREEN and catalog regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_quality/test_v1_08_materials.py `
  tests/test_vision_quality/test_v1_07_materials.py `
  tests/test_experiments/test_catalog.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add config/experiments/V1-08.json config/experiments/catalog.json student_programs/templates/v1_08_ocr_sorting.py docs/experiments/V1-08.md tests/test_vision_quality/test_v1_08_materials.py RETAINED_FILES.txt
git commit -m "feat(v1-08): publish OCR sorting curriculum"
```

## Task 9: Add scene probe, CLI, PowerShell and same-run evidence

**Files:**

- Modify: `vision_platform/experiments/probes.py`
- Modify: `vision_platform/experiments/cli.py`
- Modify: `tools/vision_lab/run_experiment.ps1`
- Create: `tests/test_experiments/test_v1_08_cli.py`
- Create: `tests/test_experiments/test_v1_08_probe.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: RED-test final occupancy and launcher contracts**

Add two probe layers. Runner's per-entry read-only probe verifies one part in its configured route slot with matching run ID, scene hash, part/route/slot and produces the bounded evidence reference consumed by `confirm_entry_probe`. The final probe verifies the exact part set, all configured route slots, no duplicate occupancy, robot home, suction off, same run ID/snapshot/scene hash, four per-entry evidence references and all four entries consumed. CLI/PowerShell select V1-08 without arbitrary command execution and propagate stop/reset/exit status.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_v1_08_cli.py tests/test_experiments/test_v1_08_probe.py
```

- [ ] **Step 3: Implement minimal probe and launch integration**

Reuse existing process ownership and evidence bundle patterns. Do not weaken other experiment probes.

- [ ] **Step 4: Run GREEN and earlier CLI/probe regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_v1_08_cli.py `
  tests/test_experiments/test_v1_08_probe.py `
  tests/test_experiments/test_v1_07_cli.py `
  tests/test_experiments/test_probes.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/experiments/probes.py vision_platform/experiments/cli.py tools/vision_lab/run_experiment.ps1 tests/test_experiments/test_v1_08_cli.py tests/test_experiments/test_v1_08_probe.py RETAINED_FILES.txt
git commit -m "feat(v1-08): add launcher and occupancy evidence"
```

## Task 10: Add the read-only PyQt teaching result panel

**Files:**

- Modify: `vision_platform/ui/experiment_catalog_panel.py`
- Modify: `vision_platform/ui/vision_result_panel.py`
- Create: `tests/test_vision_platform/test_v1_08_result_panel.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write offscreen UI RED tests**

Require V1-08 catalog label, raw/annotated image references, training accuracy, four identifiers/confidences/routes, motion status and final occupancy. All fields are read-only; no rich text, clickable path, edit, eval or execute path. Test long errors, missing evidence, 100% and 125% scaling geometry.

- [ ] **Step 2: Confirm RED**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision_platform/test_v1_08_result_panel.py
```

- [ ] **Step 3: Implement the smallest renderer extension**

Preserve all existing panels and data flow. Render text as plain text and evidence paths as labels only.

- [ ] **Step 4: Run GREEN plus existing UI regressions**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_platform/test_v1_08_result_panel.py `
  tests/test_vision_platform/test_v1_07_result_panel.py `
  tests/test_vision_platform/test_experiment_catalog_panel.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add vision_platform/ui/experiment_catalog_panel.py vision_platform/ui/vision_result_panel.py tests/test_vision_platform/test_v1_08_result_panel.py RETAINED_FILES.txt
git commit -m "feat(v1-08): show read-only OCR sorting results"
```

## Task 11: Add real CoppeliaSim online acceptance and release contract

**Files:**

- Create: `tests/test_acceptance/test_coppeliasim_v1_08.py`
- Modify: `tests/test_acceptance/test_delivery_contract.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write opt-in acceptance and release RED tests**

Online acceptance must launch an owned CoppeliaSim process on fixed V1-08 port `23008`, reject any unexpected listener, load the exact scene, reset, train, capture, recognize four IDs, prove zero motion before plan, execute four transfers, verify occupancy/home/tool-off/same-run evidence, terminate only owned processes and preserve artifacts outside Git. Add fail-closed cases for tampered asset hash and incomplete recognition before motion.

- [ ] **Step 2: Confirm static RED and explicit online skip**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_acceptance/test_delivery_contract.py tests/test_acceptance/test_coppeliasim_v1_08.py
```

Expected before enabling online: delivery RED until retained set is added; online test is an explicit skip, not PASS.

- [ ] **Step 3: Complete the delivery whitelist and online wrapper**

Add the complete V1-08 required path set. Reuse proven process ownership and cleanup helpers; never kill unrelated CoppeliaSim/Python processes.

- [ ] **Step 4: Run static GREEN and actual opt-in online acceptance**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_acceptance/test_delivery_contract.py `
  tests/test_acceptance/test_coppeliasim_v1_08.py

.\.venv-vision\Scripts\python.exe -m pytest -q -s -m coppeliasim tests/test_acceptance/test_coppeliasim_v1_08.py
```

Expected: the second command runs rather than skips and passes. If CoppeliaSim is unavailable, report `PENDING_SIMULATION` and do not claim completion.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_acceptance/test_coppeliasim_v1_08.py tests/test_acceptance/test_delivery_contract.py RETAINED_FILES.txt
git commit -m "test(v1-08): verify online OCR sorting delivery"
```

## Task 12: Full regression, independent review, validation report and push

**Files:**

- Create: `docs/validation/v1-08-ocr-sorting-validation.md`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Run all focused and full static tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision2d/test_ocr.py `
  tests/test_experiments/test_ocr_sorting.py `
  tests/test_experiments/test_ocr_assets.py `
  tests/test_experiments/test_ocr_service.py `
  tests/test_student_programs/test_v1_08_sort_guard.py `
  tests/test_student_programs/test_v1_08_protocol_sdk.py `
  tests/test_student_programs/test_v1_08_gateway.py `
  tests/test_simulation/test_v1_08_ocr_assets.py `
  tests/test_simulation/test_v1_08_scene_contract.py `
  tests/test_vision_quality/test_v1_08_materials.py `
  tests/test_experiments/test_v1_08_cli.py `
  tests/test_experiments/test_v1_08_probe.py `
  tests/test_vision_platform/test_v1_08_result_panel.py `
  tests/test_acceptance/test_delivery_contract.py
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
```

- [ ] **Step 2: Re-run online acceptance and perform human UI checks**

Run the exact opt-in command from Task 11. Capture 100% and 125% UI screenshots for human review; record `PENDING_HUMAN_ACCEPTANCE` unless the user explicitly accepts them.

- [ ] **Step 3: Run independent review and fix all P0/P1/P2 findings**

Review security, path/hash binding, no-motion-before-plan, guard reset, process ownership, protected assets, JSON-native protocol, UI read-only behavior and claim boundaries. Each functional fix gets a failing regression before implementation and a separate commit.

- [ ] **Step 4: Write the validation report and final release audit**

The report contains exact base/tip SHA, Task commits, RED/GREEN evidence, focused/full pass-skip counts, actual online command/result, UI evidence status, protected hashes, retained-file audit, review findings/fixes and remaining `PENDING_*` items.

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_acceptance/test_delivery_contract.py
git diff --check
git status --short --branch
```

- [ ] **Step 5: Commit report and push only this branch**

```powershell
git add docs/validation/v1-08-ocr-sorting-validation.md RETAINED_FILES.txt
git commit -m "docs(v1-08): record final validation evidence"
git push -u origin codex/v2-2-v1-08-ocr-sorting
git status --short --branch
```

Expected handoff: clean worktree, local/remote tip hashes identical, no main merge, and claims limited to the validated simulation/software layers.
