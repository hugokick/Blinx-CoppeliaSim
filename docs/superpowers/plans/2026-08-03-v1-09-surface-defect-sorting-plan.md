# V2.2 V1-09 Surface Defect Sorting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and execute this plan Task by Task. Every behavior change follows RED → observed expected failure → minimal GREEN → regression → small commit. Worker self-tests are evidence inputs only; the coordinating task owns planning, dispatch, independent review, acceptance, and integration decisions.

**Goal:** Deliver V1-09 as a complete CoppeliaSim teaching experiment that detects one qualified part and all five approved surface-defect classes from one fixed pre-motion frame, freezes a six-entry host-owned sort plan, and safely places each part into its dedicated slot with same-run evidence.

**Architecture:** A hash-bound asset loader and fixed-ROI service call the existing pure `detect_surface_defects()` kernel six times against one reference crop. A pure adapter accepts only the approved result semantics and creates an immutable `DefectSortPlan` when the qualified/missing/hole/foreign/broken/dimension set is complete. A separate pure guard controls plan state, while the runner alone performs private per-device robot/tool actions. Students request analysis once and select only an unconsumed `entry_id`; they cannot submit images, thresholds, routes, coordinates, speed, raw robot commands, or tool commands.

**Tech Stack:** Python 3.11, NumPy, OpenCV, PyQt5, pytest, CoppeliaSim ZeroMQ Remote API, PowerShell, Git/GitHub.

---

## 0. Required reading, authority, and non-negotiable rules

Read completely:

- `docs/superpowers/specs/2026-08-03-v1-09-surface-defect-sorting-design.md`
- `docs/superpowers/plans/2026-08-03-v1-09-surface-defect-sorting-plan.md`
- `docs/superpowers/specs/2026-08-02-v1-07-v1-09-vision2d-algorithm-kernels-design.md`
- `docs/superpowers/plans/2026-08-02-v1-07-v1-09-vision2d-algorithm-kernels-plan.md`
- `docs/superpowers/plans/2026-08-02-v1-08-ocr-sorting-plan.md`

Rules:

- The coordinating task owns architecture, task boundaries, prompts, worktrees, monitoring, review, acceptance, integration order, and the final gate. Developer tasks only implement assigned files and return self-test evidence.
- The design-time `origin/main` is `1c11302811a070d62c4d440a012a0399027a1683`. It is dispatchable only if a fresh fetch still returns that exact SHA. A change requires a coordinator refresh before any worktree is created.
- The D1-01 candidate is tracked separately. No developer may merge/cherry-pick it, push `main`, or tag. Final integration order is D1-01 first and V1-09 second, each after separate user authorization.
- Main branch/worktree: `codex/v2-2-v1-09-surface-defect-sorting` at `C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-09-surface-defect-sorting`.
- Optional asset-only branch/worktree: `codex/v2-2-v1-09-defect-assets` at `C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-09-defect-assets`.
- Only the main owner edits shared catalog, protocol, SDK, gateway, runner, UI, CLI, delivery, retention, attributes, or generic scene-builder files.
- Keep `vision_platform/vision2d/defect_detection.py` and its tests read-only. Keep all V1-07/V1-08, RGB-D/D1, URDF, STL, mesh, robot, and existing formal `.ttt` assets read-only.
- The only authorized new scene is `simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt`.
- Runtime code never imports or reads `acceptance_ground_truth.json`.
- V1-08 remains fixed to `23008`; V1-09 is fixed to `23010`; other port behavior remains unchanged.
- Skipped CoppeliaSim tests are not PASS. Real camera/robot/emergency-stop/pneumatic/grasping stay `PENDING_HARDWARE`; teaching acceptance stays `PENDING_HUMAN_ACCEPTANCE`.
- Use `apply_patch` for text/source edits, canonical UTF-8/LF JSON with sorted keys and no NaN, and small commits.

Use `.\.venv-vision\Scripts\python.exe` in each implementation worktree.

## Task 1: Coordinator gate and isolated worktree provisioning

**Owner:** Coordinating task only.

**Files:** Verify the approved spec/plan, defect kernel/test, generic scene builder, and process-ownership helper.

- [ ] **Step 1: Fetch and prove the live base**

~~~powershell
Set-Location 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean'
git fetch origin --prune
$liveBase = git rev-parse origin/main
if ($liveBase -ne '1c11302811a070d62c4d440a012a0399027a1683') { throw "BASELINE_CHANGED: plan=1c11302811a070d62c4d440a012a0399027a1683 live=$liveBase" }
git status --short --branch
~~~

- [ ] **Step 2: Prove prerequisites exist**

~~~powershell
git cat-file -e origin/main:vision_platform/vision2d/defect_detection.py
git cat-file -e origin/main:tests/test_vision2d/test_defect_detection.py
git cat-file -e origin/main:simulation/training_scenes/build_scene.py
git cat-file -e origin/main:tools/vision_lab/process_ownership.ps1
git cat-file -e origin/main:vision_platform/experiments/ocr_sorting.py
git cat-file -e origin/main:vision_platform/student/ocr_sort_guard.py
~~~

Every command must exit 0 or the gate is `BASELINE_GATE_NOT_READY`.

- [ ] **Step 3: Record protected scene hashes before work**

~~~powershell
Get-ChildItem simulation -Recurse -Filter *.ttt | Where-Object FullName -NotLike '*vision_defect_sorting_lab*' | Get-FileHash -Algorithm SHA256 | Sort-Object Path | ConvertTo-Json -Depth 4
~~~

- [ ] **Step 4: Create the main worktree at the exact SHA**

~~~powershell
git worktree add 'C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-09-surface-defect-sorting' -b codex/v2-2-v1-09-surface-defect-sorting 1c11302811a070d62c4d440a012a0399027a1683
~~~

- [ ] **Step 5: Create the asset worktree only when Task 3 is dispatched**

~~~powershell
git worktree add 'C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-09-defect-assets' -b codex/v2-2-v1-09-defect-assets 1c11302811a070d62c4d440a012a0399027a1683
~~~

- [ ] **Step 6: Link the validated environment and run baseline**

~~~powershell
$sharedVisionEnv = 'C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\main-release\.venv-vision'
if (-not (Test-Path (Join-Path $sharedVisionEnv 'Scripts\python.exe'))) { throw 'ENVIRONMENT_GATE_NOT_READY' }
if (-not (Test-Path '.\.venv-vision')) { New-Item -ItemType Junction -Path '.\.venv-vision' -Target $sharedVisionEnv | Out-Null }
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
~~~

Expected: zero failures. Record passed and skipped counts separately.

## Task 2: Define pure decision, plan, and receipt contracts

**Owner:** Main implementation owner.

**Files:**

- Create: `vision_platform/experiments/defect_sorting.py`
- Create: `tests/test_experiments/test_defect_sorting.py`
- Modify: `vision_platform/experiments/__init__.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write strict RED tests for the fixed batch**

~~~python
EXPECTED = (
    ("entry_a", "part_a", "qualified", "route_qualified", "slot_qualified"),
    ("entry_b", "part_b", "missing", "route_missing", "slot_missing"),
    ("entry_c", "part_c", "hole", "route_hole", "slot_hole"),
    ("entry_d", "part_d", "foreign", "route_foreign", "slot_foreign"),
    ("entry_e", "part_e", "broken", "route_broken", "slot_broken"),
    ("entry_f", "part_f", "dimension", "route_dimension", "slot_dimension"),
)

def test_complete_batch_freezes_all_six_decisions():
    plan = build_defect_sort_plan(
        _config(), _observations(),
        run_id="run-001", frame_id="frame-001",
        scene_sha256="1" * 64,
        config_sha256="2" * 64,
        asset_manifest_sha256="3" * 64,
    )
    assert tuple((e.entry_id, e.part_id, e.decision, e.slot_id) for e in plan.entries) == tuple(
        (a, b, c, e) for a, b, c, _d, e in EXPECTED
    )
    assert defect_sort_plan_to_dict(plan)["plan_id"] == plan.plan_id
~~~

Also test exact schemas, deterministic order, source non-mutation, deep tuple copies, finite built-in numbers, digest format, ROI/workspace bounds, unique entry/part/route/slot, one qualified decision, all five defects once, and canonical JSON.

- [ ] **Step 2: Write every fail-closed decision test**

~~~python
def test_candidate_empty_is_not_missing():
    result = _result(status="PARTIAL", failure_code="CANDIDATE_EMPTY", defect_types=("missing",))
    with pytest.raises(DefectSortError, match="DEFECT_SORT_ANALYSIS_REJECTED"):
        classify_defect_result(result)

def test_multiple_types_are_ambiguous():
    result = _result(status="PARTIAL", failure_code="DEFECTS_FOUND", defect_types=("missing", "hole"))
    with pytest.raises(DefectSortError, match="DEFECT_SORT_DECISION_AMBIGUOUS"):
        classify_defect_result(result)
~~~

Cover `PASS`/no finding → qualified; `PARTIAL`/`DEFECTS_FOUND`/one type → that type; `NO_TARGETS`, `REJECTED`, other failure codes, unknown types, incomplete IDs, duplicates, and incomplete class coverage → stable errors.

- [ ] **Step 3: Run RED**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_defect_sorting.py
~~~

Expected: missing-module collection failure.

- [ ] **Step 4: Implement immutable contracts**

Implement `DefectObservation`, `ApprovedDefectSortEntry`, `DefectSortPlan`, `DefectSortReceipt`, `DefectSortError`, `classify_defect_result()`, `build_defect_sort_plan()`, and canonical dict serializers.

~~~python
APPROVED_DEFECTS = frozenset({"missing", "hole", "foreign", "broken", "dimension"})

def classify_defect_result(result: DefectResult) -> str:
    if result.status == "PASS" and result.failure_code is None and not result.defects:
        return "qualified"
    if result.status != "PARTIAL" or result.failure_code != "DEFECTS_FOUND" or not result.defects:
        raise DefectSortError("DEFECT_SORT_ANALYSIS_REJECTED", "kernel result cannot activate a route")
    kinds = {finding.defect_type for finding in result.defects}
    if not kinds <= APPROVED_DEFECTS:
        raise DefectSortError("DEFECT_SORT_ANALYSIS_REJECTED", "unknown defect type")
    if len(kinds) != 1:
        raise DefectSortError("DEFECT_SORT_DECISION_AMBIGUOUS", "multiple defect classes")
    return next(iter(kinds))
~~~

- [ ] **Step 5: Run GREEN and kernel regression**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_defect_sorting.py tests/test_vision2d/test_defect_detection.py
git diff --check
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add vision_platform/experiments/defect_sorting.py vision_platform/experiments/__init__.py tests/test_experiments/test_defect_sorting.py RETAINED_FILES.txt
git commit -m "feat(v1-09): define defect sorting contracts"
~~~

## Task 3: Generate deterministic original defect assets

**Owner:** Optional asset-only owner; the only independent parallel implementation Task.

**Allowlist:**

- `tools/vision_lab/generate_v1_09_defect_assets.py`
- seven PNGs under `simulation/vision_defect_sorting_lab/assets/`
- `simulation/vision_defect_sorting_lab/defect_assets_manifest.json`
- `tests/test_simulation/test_v1_09_defect_assets.py`

Everything else, especially `.ttt`, scene builder/spec/manifest, retention/attributes, SDK/gateway/runner/UI/catalog/CLI/delivery, is denied.

- [ ] **Step 1: Write RED identity tests**

~~~python
EXPECTED_ASSETS = {
    "reference": "assets/reference.png",
    "entry_a": "assets/candidate_a.png",
    "entry_b": "assets/candidate_b.png",
    "entry_c": "assets/candidate_c.png",
    "entry_d": "assets/candidate_d.png",
    "entry_e": "assets/candidate_e.png",
    "entry_f": "assets/candidate_f.png",
}

def test_manifest_binds_project_original_pngs():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["scene_id"] == "V1-09"
    assert payload["seed"] == 20260803
    assert payload["source"] == "project-original-generated"
    assert {item["asset_id"]: Path(item["path"]).name for item in payload["assets"]} == {
        key: Path(value).name for key, value in EXPECTED_ASSETS.items()
    }
~~~

Reject absolute/parent paths, symlinks, duplicates, non-PNG, wrong 256x256 size, wrong SHA, extra assets, and noncanonical JSON.

- [ ] **Step 2: Run RED**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_simulation/test_v1_09_defect_assets.py
~~~

- [ ] **Step 3: Implement literal OpenCV/NumPy geometry**

Use local `numpy.random.default_rng(20260803)` only. Produce one common base subject; qualified is unchanged; missing removes an edge region; hole adds an enclosed void; foreign adds a separated component; broken splits the subject into two substantial components; dimension changes one principal dimension beyond `0.08`. Do not use fonts, network, external images/models, timestamps, or machine paths.

- [ ] **Step 4: Emit canonical real hashes and provenance**

Every record has exact keys `asset_id`, `generator`, `generator_version`, `path`, `purpose`, `sha256`, `size_px`, and `source`.

- [ ] **Step 5: Prove regeneration and kernel semantics**

~~~powershell
.\.venv-vision\Scripts\python.exe tools/vision_lab/generate_v1_09_defect_assets.py --check
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_simulation/test_v1_09_defect_assets.py tests/test_vision2d/test_defect_detection.py
git diff --check
~~~

The real kernel must classify A as `PASS` and B–F as exactly missing/hole/foreign/broken/dimension under frozen defaults.

- [ ] **Step 6: Scope-check, commit, and push**

~~~powershell
git diff --name-only origin/main...HEAD
git add tools/vision_lab/generate_v1_09_defect_assets.py simulation/vision_defect_sorting_lab/assets simulation/vision_defect_sorting_lab/defect_assets_manifest.json tests/test_simulation/test_v1_09_defect_assets.py
git commit -m "feat(v1-09): generate original defect assets"
git push -u origin codex/v2-2-v1-09-defect-assets
~~~

The coordinator independently reviews and integrates the accepted asset commit into the feature branch. The asset owner does not integrate it.

## Task 4: Define the independent scene and offline builder contract

**Owner:** Main implementation owner. Steps 1–4 may prepare the offline contract while Task 3 runs; asset-dependent GREEN, retention, and commit wait until the coordinator integrates the accepted asset commit.

**Files:**

- Create: `simulation/vision_defect_sorting_lab/scene_spec.json`
- Create: `simulation/vision_defect_sorting_lab/profiles.json`
- Create: `simulation/vision_defect_sorting_lab/acceptance_ground_truth.json`
- Create: `tools/vision_lab/build_v1_09_scene.py` and `build_v1_09_scene.ps1`
- Create: `tests/test_simulation/test_v1_09_scene_contract.py` and `test_v1_09_scene_builder.py`
- Modify: `simulation/training_scenes/build_scene.py`, `.gitattributes`, `RETAINED_FILES.txt`

- [ ] **Step 1: Write RED static-contract tests**

Assert the only output is the authorized new `.ttt`; root/camera paths are `/VisionDefectSortingLab` and `/VisionDefectSortingLab/CameraRig/Camera`; profile is `standard` at `1024x1024`; part paths are `part_a` through `part_f` with `InspectionFace`; slots are qualified/missing/hole/foreign/broken/dimension; all seven ROIs are positive, in-bounds, and non-overlapping; port is `23010`; manifest/assets are hash-bound; ground truth is absent from runtime bindings; protected scene hashes are unchanged.

- [ ] **Step 2: Freeze concrete geometry and image ROIs**

| Entry | Pick XYZ mm | Decision | Drop XYZ mm |
|---|---:|---|---:|
| entry_a | [32, -58, 18] | qualified | [112, -78, 22] |
| entry_b | [60, -58, 18] | missing | [126, -78, 22] |
| entry_c | [88, -58, 18] | hole | [140, -78, 22] |
| entry_d | [32, 10, 18] | foreign | [112, 78, 22] |
| entry_e | [60, 10, 18] | broken | [126, 78, 22] |
| entry_f | [88, 10, 18] | dimension | [140, 78, 22] |

Workspace is `x=[20,155]`, `y=[-95,95]`, `z=[10,140]`, `safe_z_mm=110`, `speed_mm_s=15`.

~~~json
{
  "reference": [416, 56, 192, 192],
  "entry_a": [80, 320, 192, 192],
  "entry_b": [416, 320, 192, 192],
  "entry_c": [752, 320, 192, 192],
  "entry_d": [80, 648, 192, 192],
  "entry_e": [416, 648, 192, 192],
  "entry_f": [752, 648, 192, 192]
}
~~~

The builder may tune camera pose to render these bound ROIs, but may not change the ROI/part/slot contract without a coordinator-approved plan revision.

- [ ] **Step 3: Run RED**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_simulation/test_v1_09_scene_contract.py tests/test_simulation/test_v1_09_scene_builder.py
~~~

- [ ] **Step 4: Implement only the narrow V1-09 builder path**

Extend the generic builder with one additive formal-scene record and V1-09 validator. Keep V1-07/V1-08 branches unchanged.

~~~python
def main() -> int:
    manifest = build_scene(
        spec_path=PROJECT_ROOT / "simulation" / "vision_defect_sorting_lab" / "scene_spec.json",
        host="127.0.0.1",
        port=23010,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0
~~~

Use identical part geometry/mass/grasp height, change only generated inspection textures, keep the reference non-pickable, and show all six labelled slots together.

- [ ] **Step 5: Run offline GREEN and older-scene regressions**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_simulation/test_v1_09_scene_contract.py tests/test_simulation/test_v1_09_scene_builder.py tests/test_simulation/test_v1_08_scene_builder.py tests/test_simulation/test_v1_08_scene_contract.py tests/test_acceptance/test_coppeliasim_readiness.py
git diff --check
~~~

No online scene generation occurs in this Task.

- [ ] **Step 6: Register retention rules and commit**

Append only the V1-09 PNG/`.ttt` binary rules and new formal paths; do not reorder existing entries.

~~~powershell
git add simulation/vision_defect_sorting_lab/scene_spec.json simulation/vision_defect_sorting_lab/profiles.json simulation/vision_defect_sorting_lab/acceptance_ground_truth.json tools/vision_lab/build_v1_09_scene.py tools/vision_lab/build_v1_09_scene.ps1 simulation/training_scenes/build_scene.py tests/test_simulation/test_v1_09_scene_contract.py tests/test_simulation/test_v1_09_scene_builder.py .gitattributes RETAINED_FILES.txt
git commit -m "feat(v1-09): define defect sorting scene contract"
~~~

## Task 5: Add hash-bound loading, fixed-frame analysis, and teaching visualization

**Owner:** Main implementation owner.

**Files:**

- Create: `vision_platform/experiments/defect_assets.py`, `defect_service.py`, `defect_visualization.py`
- Create: `tests/test_experiments/test_defect_assets.py`, `test_defect_service.py`, `test_defect_visualization.py`
- Modify: `vision_platform/experiments/__init__.py`, `RETAINED_FILES.txt`

- [ ] **Step 1: Write loader RED tests**

Test exact manifest keys, rooted resolution, no symlink/path escape, fixed `256x256 uint8` decode, unique records, lowercase SHA, scene `V1-09`, seed `20260803`, immutable arrays, and byte tamper raising `DEFECT_SORT_ASSET_INVALID` before detector invocation.

- [ ] **Step 2: Write fixed-frame service RED tests**

~~~python
result = DefectSortingService.from_manifest(MANIFEST).analyze(
    frame,
    reference_roi=(416, 56, 192, 192),
    candidate_rois=FIXED_CANDIDATE_ROIS,
    run_id="run-001",
    frame_id="frame-001",
    scene_sha256="1" * 64,
)
assert [entry.decision for entry in result.plan.entries] == [
    "qualified", "missing", "hole", "foreign", "broken", "dimension"
]
assert result.config == DefectConfig()
assert result.reference_crop.flags.writeable is False
~~~

Also assert one shared pre-motion frame, six detector calls, exact resolution/profile, non-overlap, source non-mutation, all-or-nothing activation, no second analysis in one run, and all Task 2 rejection codes.

- [ ] **Step 3: Write visualization RED tests**

`render_defect_teaching_layers()` returns an annotated BGR frame, bbox-only uint8 finding mask, and stable legend. It must not mutate a `DefectResult` or affect the decision. The public label is `缺陷区域示意`, never segmentation ground truth.

- [ ] **Step 4: Run RED**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_defect_assets.py tests/test_experiments/test_defect_service.py tests/test_experiments/test_defect_visualization.py
~~~

- [ ] **Step 5: Implement the minimal loader/service/renderer**

~~~python
FORMAL_DEFECT_CONFIG = DefectConfig(
    missing_ratio=0.01,
    hole_ratio=0.005,
    foreign_ratio=0.002,
    dimension_ratio=0.08,
    min_component_ratio=0.002,
    morphology_kernel_size=3,
    max_alignment_shift_px=8.0,
    min_contrast=4.0,
)
~~~

Accept in-memory images only. Never infer from filenames/IDs, import ground truth, expose coordinates, or access devices.

- [ ] **Step 6: Run GREEN, scan runtime dependencies, and commit**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_defect_assets.py tests/test_experiments/test_defect_sorting.py tests/test_experiments/test_defect_service.py tests/test_experiments/test_defect_visualization.py tests/test_vision2d/test_defect_detection.py
rg -n "acceptance_ground_truth|candidate_[a-f]\.png" vision_platform/experiments/defect_*.py
git diff --check
git add vision_platform/experiments/defect_assets.py vision_platform/experiments/defect_service.py vision_platform/experiments/defect_visualization.py vision_platform/experiments/__init__.py tests/test_experiments/test_defect_assets.py tests/test_experiments/test_defect_service.py tests/test_experiments/test_defect_visualization.py RETAINED_FILES.txt
git commit -m "feat(v1-09): analyze hash-bound defect batch"
~~~

The `rg` scan must return no runtime ground-truth or candidate-filename dependency.

## Task 6: Add the pure atomic defect-sort guard

**Owner:** Main implementation owner.

**Files:** Create `vision_platform/student/defect_sort_guard.py` and `tests/test_student_programs/test_v1_09_defect_sort_guard.py`; modify `RETAINED_FILES.txt`.

- [ ] **Step 1: Write RED state tests**

~~~python
class DefectSortState(str, Enum):
    EMPTY = "EMPTY"
    ANALYZING = "ANALYZING"
    ACTIVE = "ACTIVE"
    ENTRY_ACTIVE = "ENTRY_ACTIVE"
    AWAITING_PROBE = "AWAITING_PROBE"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
~~~

Test `begin_analysis()`, atomic `activate(plan)`, analysis failure with zero device calls, one unconsumed entry, ordered confirmation, mandatory post-probe, six-entry completion, any nonterminal stop/failure → `FAILED`, and controlled reset → `EMPTY`.

- [ ] **Step 2: Write immutable action/probe RED tests**

~~~python
ACTION_KINDS = (
    "move_safe_pick", "move_pick", "tool_on", "move_safe_pick",
    "move_safe_drop", "move_drop", "tool_off", "move_safe_drop",
)
~~~

Pre/post proof binds exact `run_id`, `scene_sha256`, `frame_id`, `plan_id`, `entry_id`, `part_id`, `slot_id`, and `evidence_id`. Reject missing/extra keys, replay, mismatch, mutable nested data, NaN/Inf, and oversized text.

- [ ] **Step 3: Run RED**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_student_programs/test_v1_09_defect_sort_guard.py
~~~

- [ ] **Step 4: Implement without runtime/device imports**

The guard deep-freezes the approved plan and returns typed `DefectSortAction` objects. It imports no Qt, subprocess, CoppeliaSim, gateway, runner, robot, or tool module.

- [ ] **Step 5: Run GREEN and V1-08 guard regression**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_student_programs/test_v1_09_defect_sort_guard.py tests/test_student_programs/test_v1_08_sort_guard.py tests/test_experiments/test_defect_sorting.py
git diff --check
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add vision_platform/student/defect_sort_guard.py tests/test_student_programs/test_v1_09_defect_sort_guard.py RETAINED_FILES.txt
git commit -m "feat(v1-09): guard defect sort execution"
~~~

## Task 7: Integrate capability, protocol, SDK, gateway, and private runner

**Owner:** Main implementation owner; no other endpoint edits these shared files.

**Files:**

- Modify: `vision_platform/experiments/capabilities.py`
- Modify: `vision_platform/student/protocol.py`, `sdk.py`, `experiment_gateway.py`, `runner.py`
- Create: `tests/test_student_programs/test_v1_09_protocol_sdk.py`, `test_v1_09_gateway.py`, `test_v1_09_runner.py`
- Modify: `tests/test_student_programs/test_protocol.py`, `RETAINED_FILES.txt`

- [ ] **Step 1: Write protocol/SDK RED tests**

~~~python
CommandMessage("000001", "vision2d.surface_defects", {})
CommandMessage("000002", "vision2d.defect_sort_entry", {"entry_id": "entry_a"})
analysis = ctx.vision2d.surface_defects()
receipt = ctx.vision2d.defect_sort_entry(analysis.entries[0].entry_id)
~~~

Reject extra/missing args, non-ASCII/overlong IDs, nested inputs, threshold/ROI/path/route/slot/coordinate/speed keys, and non-JSON-native responses. DTOs expose decisions/findings/digests/status, not coordinates, paths, devices, or actions.

- [ ] **Step 2: Write gateway RED tests**

Analysis requires the four approved capabilities, standard 1024x1024 profile, one capture, bound hashes, one service call, full-plan guard activation, raw/reference/six candidates/annotated/mask evidence, zero device calls on failure, no second analysis before reset, and no ground-truth read. Direct gateway entry dispatch raises `DEFECT_SORT_ENTRY_RUNNER_ONLY`.

- [ ] **Step 3: Write private-runner RED tests**

For V1-09, raw robot/tool commands fail with zero calls. The private entry executor accepts only `entry_id`, probes before motion and after actions, checks stop/pause/single-step/workspace/speed/guard before every actual move/tool call, permits at most one device call per step, does not consume on pause, requires post-probe before receipt, retains the first error, attempts tool-off and safe home, invalidates on stop, and proves final home/tool-off/six slots.

- [ ] **Step 4: Run RED**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_student_programs/test_v1_09_protocol_sdk.py tests/test_student_programs/test_v1_09_gateway.py tests/test_student_programs/test_v1_09_runner.py tests/test_student_programs/test_protocol.py
~~~

- [ ] **Step 5: Implement narrow API and private same-run context**

~~~python
self._defect_evidence = {
    "plan": plan,
    "run_id": run_id,
    "frame_id": frame_id,
    "snapshot_id": snapshot_id,
    "scene_sha256": self.context.scene_sha256,
    "config_sha256": output.config_sha256,
    "asset_manifest_sha256": output.asset_manifest_sha256,
}
~~~

Identify V1-09 by `vision2d.surface_defects` capability, block raw commands before generic dispatch, call devices only from the private runner path, and never recursively dispatch raw public commands.

- [ ] **Step 6: Run GREEN and commit in two increments**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_student_programs/test_v1_09_protocol_sdk.py tests/test_student_programs/test_v1_09_gateway.py tests/test_student_programs/test_protocol.py tests/test_student_programs/test_v1_08_protocol_sdk.py tests/test_student_programs/test_v1_08_gateway.py
git add vision_platform/experiments/capabilities.py vision_platform/student/protocol.py vision_platform/student/sdk.py vision_platform/student/experiment_gateway.py tests/test_student_programs/test_v1_09_protocol_sdk.py tests/test_student_programs/test_v1_09_gateway.py tests/test_student_programs/test_protocol.py RETAINED_FILES.txt
git commit -m "feat(v1-09): expose controlled defect analysis"
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_student_programs/test_v1_09_runner.py tests/test_student_programs/test_v1_09_defect_sort_guard.py tests/test_student_programs/test_v1_08_gateway.py
git add vision_platform/student/runner.py tests/test_student_programs/test_v1_09_runner.py RETAINED_FILES.txt
git commit -m "feat(v1-09): execute private defect sort actions"
~~~

## Task 8: Publish the formal definition, template, guide, and material contract

**Owner:** Main implementation owner.

**Files:**

- Create: `config/experiments/V1-09.json`, `student_programs/templates/v1_09_surface_defects.py`, `docs/experiments/V1-09.md`
- Create: `tests/test_vision_quality/test_v1_09_materials.py` and `tests/test_student_programs/test_v1_09_program.py`
- Modify: `config/experiments/catalog.json`
- Modify only if required: `vision_platform/experiments/catalog.py`
- Modify: `vision_platform/experiments/capabilities.py`, `vision_platform/ui/experiment_catalog_panel.py`
- Modify related catalog/CLI/panel tests and `RETAINED_FILES.txt`

- [ ] **Step 1: Write RED catalog/material tests**

Assert V1-09 resolves only the new scene/manifest/template/guide; capabilities equal `("camera.rgb", "camera.profile", "lighting.profile", "vision2d.surface_defects")`; root/camera paths are V1-09-specific; fixed config/ROIs and six neutral IDs are bound; hardware/human gates stay pending.

- [ ] **Step 2: Write exact template RED test**

~~~python
def main(ctx):
    analysis = ctx.vision2d.surface_defects()
    ctx.log("缺陷计划已冻结", plan_id=analysis.plan_id, entries=len(analysis.entries))
    for entry in analysis.entries:
        receipt = ctx.vision2d.defect_sort_entry(entry.entry_id)
        ctx.log("条目完成", entry_id=receipt.entry_id, decision=receipt.decision, status=receipt.status)
~~~

The template never calls raw robot/tool APIs, reads files, submits thresholds, inspects filenames, or infers a route.

- [ ] **Step 3: Run RED**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision_quality/test_v1_09_materials.py tests/test_student_programs/test_v1_09_program.py tests/test_experiments/test_formal_catalog.py tests/test_vision_platform/test_experiment_catalog_panel.py
~~~

- [ ] **Step 4: Implement definition, guide, template, and additive catalog entry**

The guide explains seven ROIs, five defects, frozen thresholds, failure semantics, one-frame constraint, private motion, bbox-only mask, same-run evidence, pause/step/stop/reset, and pending hardware/human acceptance.

- [ ] **Step 5: Run GREEN and curriculum regressions**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision_quality/test_v1_09_materials.py tests/test_student_programs/test_v1_09_program.py tests/test_experiments/test_formal_catalog.py tests/test_experiments/test_cli.py tests/test_vision_platform/test_experiment_catalog_panel.py tests/test_vision2d/test_guides.py tests/test_acceptance/test_experiment_guides.py
git diff --check
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add config/experiments/V1-09.json config/experiments/catalog.json student_programs/templates/v1_09_surface_defects.py docs/experiments/V1-09.md vision_platform/experiments/catalog.py vision_platform/experiments/capabilities.py vision_platform/ui/experiment_catalog_panel.py tests/test_vision_quality/test_v1_09_materials.py tests/test_student_programs/test_v1_09_program.py tests/test_experiments/test_formal_catalog.py tests/test_experiments/test_cli.py tests/test_vision_platform/test_experiment_catalog_panel.py RETAINED_FILES.txt
git commit -m "feat(v1-09): publish surface defect sorting lab"
~~~

If `catalog.py` needs no change, omit it from staging.

## Task 9: Add probes, CLI, PowerShell, and same-run evidence

**Owner:** Main implementation owner.

**Files:**

- Modify: `vision_platform/experiments/probes.py`, `vision_platform/cli.py`, `tools/vision_lab/run_experiment.ps1`
- Create: `tools/vision_lab/run_v1_09_surface_defects.ps1`
- Create: `tests/test_experiments/test_v1_09_probe.py`, `test_v1_09_cli.py`
- Create: `tests/test_student_programs/test_v1_09_evidence.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write RED pre/post/final probe tests**

`probe_defect_entry_pre` proves the chosen part is at its start and its slot contract is valid. `probe_defect_entry_post` proves the part is in its bound slot and all others are at their own start/confirmed slot. `probe_defect_final` proves six exact matches, home, tool-off, six same-run proofs, one frame/plan, and no extra movement. Swapping two parts must fail even when all slots are occupied.

- [ ] **Step 2: Write RED CLI/port/process tests**

~~~python
assert _resolve_experiment_port("V1-08", None) == 23008
assert _resolve_experiment_port("V1-09", None) == 23010
assert _resolve_experiment_port("V1-09", 23010) == 23010
with pytest.raises(ValueError):
    _resolve_experiment_port("V1-09", 23000)
~~~

The wrapper must use `launch_coppeliasim.ps1`, `process_ownership.ps1`, `Stop-ExactOwnedProcess`, exact PID/path/start time, and never process-name kills.

- [ ] **Step 3: Write same-run evidence RED tests**

Bind experiment/run/frame/scene/config/asset/plan/entry hashes; raw/reference/six candidate/annotated/mask hashes; detector result/decision; pre/post probes; device counts; final slots/home/tool; process identity; port cleanup. Any run/scene/frame/plan/entry replay raises `DEFECT_SORT_RUN_MISMATCH`.

- [ ] **Step 4: Run RED**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_v1_09_probe.py tests/test_experiments/test_v1_09_cli.py tests/test_student_programs/test_v1_09_evidence.py
~~~

- [ ] **Step 5: Implement narrow V1-09 branches**

Keep V1-08 probes and `23008` unchanged. Evidence uses canonical JSON, relative paths, bounded errors, and `PENDING_HARDWARE`.

- [ ] **Step 6: Run GREEN and commit**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_v1_09_probe.py tests/test_experiments/test_v1_09_cli.py tests/test_student_programs/test_v1_09_evidence.py tests/test_experiments/test_v1_08_probe.py tests/test_experiments/test_v1_08_cli.py tests/test_experiments/test_cli.py
git diff --check
git add vision_platform/experiments/probes.py vision_platform/cli.py tools/vision_lab/run_experiment.ps1 tools/vision_lab/run_v1_09_surface_defects.ps1 tests/test_experiments/test_v1_09_probe.py tests/test_experiments/test_v1_09_cli.py tests/test_student_programs/test_v1_09_evidence.py RETAINED_FILES.txt
git commit -m "feat(v1-09): record controlled defect evidence"
~~~

## Task 10: Add the read-only PyQt teaching result page

**Owner:** Main implementation owner.

**Files:** Modify `vision_platform/ui/vision_result_panel.py`, `experiment_catalog_panel.py`, `tests/test_vision_platform/test_vision_result_panel.py`, and `test_experiment_catalog_panel.py`.

- [ ] **Step 1: Write RED rendering tests**

A complete bundle displays reference, selected candidate, annotated image, `缺陷区域示意`, decision/type, finding bbox/area/metric/threshold/confidence, thresholds, plan/slot progress, selectable digests, and both pending notices. Tampered/malformed/missing/path-escaping/nonfinite bundles show bounded error and no executable control.

- [ ] **Step 2: Write RED state tests**

Cover complete, paused, single-step, stopped, reset, and failed states. No manual robot/tool button, coordinate/threshold editor, route selector, or path input is allowed.

- [ ] **Step 3: Run RED**

~~~powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision_platform/test_vision_result_panel.py tests/test_vision_platform/test_experiment_catalog_panel.py
~~~

- [ ] **Step 4: Implement an additive `_show_defect_sorting()` renderer**

Reuse verified layer loading and selection. Do not reinterpret OCR fields or create another 3D renderer; CoppeliaSim is authoritative for motion.

- [ ] **Step 5: Run GREEN and UI regressions**

~~~powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision_platform/test_vision_result_panel.py tests/test_vision_platform/test_experiment_catalog_panel.py tests/test_vision_platform/test_simui_panel.py
git diff --check
~~~

- [ ] **Step 6: Commit**

~~~powershell
git add vision_platform/ui/vision_result_panel.py vision_platform/ui/experiment_catalog_panel.py tests/test_vision_platform/test_vision_result_panel.py tests/test_vision_platform/test_experiment_catalog_panel.py
git commit -m "feat(v1-09): explain defect sorting evidence"
~~~

## Task 11: Build the only authorized scene and run real CoppeliaSim acceptance

**Owner:** Main implementation owner generates/self-tests; coordinator independently verifies.

**Files:**

- Create: `simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt` and `scene_manifest.json`
- Create: `tests/test_acceptance/test_coppeliasim_v1_09.py`
- Modify: `tests/test_acceptance/test_coppeliasim_readiness.py`, `test_delivery_contract.py`, `RETAINED_FILES.txt`

- [ ] **Step 1: Write RED online/release tests before generation**

Prove the independent scene, asset tamper failure, zero motion on rejected/ambiguous/incomplete analysis, exact owned `23010` process, one-frame six-decision run, no public raw command, six correct slots, home/tool-off, clean listener/process, and JUnit tests > 0 with zero skips/failures/errors.

- [ ] **Step 2: Hash protected scenes immediately before build**

~~~powershell
$protectedBefore = Get-ChildItem simulation -Recurse -Filter *.ttt | Where-Object FullName -NotLike '*vision_defect_sorting_lab*' | Get-FileHash -Algorithm SHA256 | Sort-Object Path
~~~

- [ ] **Step 3: Prove port free and build with owned scope**

~~~powershell
Get-NetTCPConnection -State Listen -LocalPort 23010 -ErrorAction SilentlyContinue
.\tools\vision_lab\build_v1_09_scene.ps1 -Port 23010
~~~

Expected: no listener before launch; only the new V1-09 scene/manifest is saved.

- [ ] **Step 4: Prove protected hashes unchanged**

~~~powershell
$protectedAfter = Get-ChildItem simulation -Recurse -Filter *.ttt | Where-Object FullName -NotLike '*vision_defect_sorting_lab*' | Get-FileHash -Algorithm SHA256 | Sort-Object Path
Compare-Object $protectedBefore $protectedAfter -Property Path,Hash
~~~

Expected: no output. Any difference is a hard blocker; do not overwrite an existing scene.

- [ ] **Step 5: Run explicit online acceptance**

~~~powershell
.\tools\vision_lab\run_v1_09_surface_defects.ps1 -OutputDir artifacts\vision_lab\v1-09-online -Port 23010
~~~

Required: tests > 0, zero skipped/failures/errors; final `6/6`; one qualified plus five defect types; home true; tool-off true; port clean. Hardware/teaching remain pending.

- [ ] **Step 6: Delivery checks and scene commit**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_acceptance/test_coppeliasim_v1_09.py -m "not coppeliasim"
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_acceptance/test_coppeliasim_readiness.py tests/test_acceptance/test_delivery_contract.py
git diff --check
git add simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt simulation/vision_defect_sorting_lab/scene_manifest.json tests/test_acceptance/test_coppeliasim_v1_09.py tests/test_acceptance/test_coppeliasim_readiness.py tests/test_acceptance/test_delivery_contract.py RETAINED_FILES.txt
git commit -m "feat(v1-09): deliver defect sorting scene"
~~~

## Task 12: Full regression, worker evidence, coordinator review, and publication

**Owner:** Developer records evidence; coordinator decides acceptance.

**Files:** Create `docs/validation/v1-09-surface-defect-sorting-validation.md`; modify `RETAINED_FILES.txt`.

- [ ] **Step 1: Run the focused V1-09 suite**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_defect_assets.py tests/test_experiments/test_defect_sorting.py tests/test_experiments/test_defect_service.py tests/test_experiments/test_defect_visualization.py tests/test_experiments/test_v1_09_probe.py tests/test_experiments/test_v1_09_cli.py tests/test_student_programs/test_v1_09_defect_sort_guard.py tests/test_student_programs/test_v1_09_protocol_sdk.py tests/test_student_programs/test_v1_09_gateway.py tests/test_student_programs/test_v1_09_runner.py tests/test_student_programs/test_v1_09_evidence.py tests/test_student_programs/test_v1_09_program.py tests/test_simulation/test_v1_09_defect_assets.py tests/test_simulation/test_v1_09_scene_contract.py tests/test_simulation/test_v1_09_scene_builder.py tests/test_vision_quality/test_v1_09_materials.py tests/test_acceptance/test_coppeliasim_v1_09.py -m "not coppeliasim"
~~~

- [ ] **Step 2: Run protected, V1-08, delivery, and full-static regressions**

~~~powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d/test_defect_detection.py
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_ocr_assets.py tests/test_experiments/test_ocr_service.py tests/test_experiments/test_ocr_sorting.py tests/test_experiments/test_v1_08_probe.py tests/test_student_programs/test_v1_08_sort_guard.py tests/test_student_programs/test_v1_08_protocol_sdk.py tests/test_student_programs/test_v1_08_gateway.py tests/test_simulation/test_v1_08_scene_contract.py tests/test_simulation/test_v1_08_scene_builder.py
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_acceptance/test_delivery_contract.py
.\.venv-vision\Scripts\python.exe -m pytest -q
~~~

All commands require zero failures; record passed and skipped separately.

- [ ] **Step 3: Re-run online acceptance and cleanup proof**

~~~powershell
.\tools\vision_lab\run_v1_09_surface_defects.ps1 -OutputDir artifacts\vision_lab\v1-09-release -Port 23010
Get-NetTCPConnection -State Listen -LocalPort 23010 -ErrorAction SilentlyContinue
~~~

Expected: online tests > 0 with zero skips; no final listener.

- [ ] **Step 4: Write a bounded validation report**

Include exact base/branch/local+remote tip/worktree/interpreter; changed files and protected hashes; per-Task commits; every command/count/time; scene/asset/config/plan/evidence digests; process/listener cleanup; worker self-review; and explicit pending hardware/human items. Do not claim real Hikvision/robot/emergency-stop/pneumatic/grasp/teaching acceptance.

- [ ] **Step 5: Scan, commit report, and push feature branch**

~~~powershell
rg --pcre2 -n "TO[D]O|T[B]D|FIX[M]E|PENDING_(?!HARDWARE|HUMAN_ACCEPTANCE)" docs/validation/v1-09-surface-defect-sorting-validation.md docs/experiments/V1-09.md student_programs/templates/v1_09_surface_defects.py
git diff --check
git add docs/validation/v1-09-surface-defect-sorting-validation.md RETAINED_FILES.txt
git commit -m "docs(v1-09): record defect sorting validation"
git push -u origin codex/v2-2-v1-09-surface-defect-sorting
~~~

- [ ] **Step 6: Return evidence; do not self-accept**

~~~powershell
git fetch origin --prune
git rev-parse HEAD
git rev-parse origin/codex/v2-2-v1-09-surface-defect-sorting
git merge-base --is-ancestor 1c11302811a070d62c4d440a012a0399027a1683 HEAD
git status --short --branch
git log --oneline 1c11302811a070d62c4d440a012a0399027a1683..HEAD
git diff --name-status 1c11302811a070d62c4d440a012a0399027a1683..HEAD
~~~

The coordinator reruns required tests, performs read-only review, classifies P0/P1/P2, and alone decides whether V1-09 may enter integration review.

## 13. Design-to-plan coverage gate

| Approved design requirement | Tasks |
|---|---|
| One qualified plus all five defect classes | 2, 3, 4, 5, 11 |
| Only one new formal `.ttt`; existing assets protected | 1, 4, 11, 12 |
| Frozen defaults and unique formal decision semantics | 2, 5 |
| One pre-motion 1024x1024 frame and seven fixed ROIs | 4, 5, 7 |
| Host-owned complete plan; no ground-truth routing | 2, 5, 7 |
| Strict public API; no raw robot/tool commands | 7, 8 |
| Pure guard and device-level pause/step/stop | 6, 7 |
| Pre/post/final same-run probes | 6, 7, 9, 11 |
| Bbox-only mask and read-only PyQt view | 5, 10 |
| Fixed 23010 port and exact process ownership | 9, 11, 12 |
| Static, delivery, real CoppeliaSim, UI, pending gates | 10, 11, 12 |
| Non-overlapping optional asset work | 1, 3, 4, 12 |

## 14. Coordinator dispatch sequence and short Prompts

These Prompts are valid only while a fresh fetch resolves `origin/main` to `1c11302811a070d62c4d440a012a0399027a1683`. If it changes, the coordinator updates the gate before dispatch.

### Prompt A — main implementation owner, first tranche

> 在已由协调窗口创建的 `C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-09-surface-defect-sorting` / `codex/v2-2-v1-09-surface-defect-sorting` 上实施已批准计划。精确基线必须为 `1c11302811a070d62c4d440a012a0399027a1683`；先 fresh fetch 并核验，否则停止。完整阅读 V1-09 设计和计划，仅执行 Task 2、Task 6，以及 Task 4 中不依赖资产集成的 RED/场景合同/构建器工作。严格 RED→GREEN→回归→小提交。不得规划、派发、审核或验收；不得修改缺陷内核、已有 `.ttt`、URDF/STL/机器人资产、RGB-D/D1、V1-07/V1-08 私有语义；不得 merge/push main 或打 tag。只返回逐 Task commit、changed files、测试原始计数、工作树状态和阻塞。

### Prompt B — optional asset-only owner

> 在已由协调窗口创建的 `C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-09-defect-assets` / `codex/v2-2-v1-09-defect-assets` 上只执行 Task 3。精确基线必须为 `1c11302811a070d62c4d440a012a0399027a1683`；先 fresh fetch 并核验，否则停止。唯一允许范围是生成器、7 张原创 PNG、`defect_assets_manifest.json` 和纯资产测试；禁止 `.ttt`、scene builder/spec/manifest、`RETAINED_FILES.txt`、`.gitattributes`、SDK/gateway/runner/UI/catalog/CLI/delivery 及已有资产。固定 seed `20260803`，先 RED，再生成，证明字节再生和真实内核六类语义，单一 commit 后推送该分支。不得集成、审核或验收；返回精确 SHA、allowlist diff 和测试计数。

### Prompt C — main owner continuation after coordinator asset integration

> 协调窗口已独立审核并把资产 commit 集成到原 V1-09 实施分支。继续同一工作树/分支，依次完成 Task 4 剩余步骤及 Task 5、7、8、9、10、11、12；不得重写已验收资产。每 Task 执行 RED→GREEN→相关回归→小提交；共享文件仍由你单独拥有。只新建授权的 `simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt`，构建前后证明已有场景哈希不变。真实 CoppeliaSim 只用 `23010` 且零 skipped；硬件/教学仍为 `PENDING_HARDWARE` / `PENDING_HUMAN_ACCEPTANCE`。完成后推送 feature branch并提交完整 Git、测试、在线、进程清理和 PENDING 证据；不得自行宣告验收或合入 main。

## 15. Final implementation gate

Implementation may be dispatched only after:

1. the user approves this written plan;
2. the coordinator performs a fresh `origin/main` fetch and provisions exact-base worktrees;
3. file ownership is recorded with no high-risk overlap;
4. D1-01 remains separately tracked;
5. no developer endpoint receives planning, monitoring, review, acceptance, or integration authority.

This plan itself does not authorize implementation, scene generation, `main` merge/push, or a tag.
