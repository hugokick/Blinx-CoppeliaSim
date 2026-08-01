# V2.2-C1 Vision2D Curriculum Integration Implementation Plan

> **Execution requirement:** Follow this plan with the `executing-plans`,
> `test-driven-development`, and `verification-before-completion` skills. Every
> behavioral Task must show a genuine RED before production code changes.

**Goal:** Integrate the tested `vision2d` algorithm kernel with the controlled
student runner and the V1 vision-quality platform, then deliver formal V1-02
through V1-05 experiments with reproducible CoppeliaSim and UI evidence.

**Architecture:** Student code calls a no-argument allowlisted
`vision2d.analyze` command. The host validates experiment-owned configuration,
captures a `standard` CoppeliaSim frame, applies a full-coordinate ROI mask,
runs the pure algorithm kernel, and records a five-layer `VisionResultBundle`.
V1-02 is completed as the first vertical slice; V1-03 through V1-05 reuse the
same adapter and evidence contract.

**Tech stack:** Python 3.11, NumPy, OpenCV, PyQt5, pytest, CoppeliaSim ZeroMQ
Remote API, PowerShell, Git/GitHub.

**Design:**
`docs/superpowers/specs/2026-08-02-vision2d-curriculum-integration-design.md`

## 0. Non-negotiable execution rules

- Work only in
  `C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-vision-curriculum-integration`.
- Stay on `codex/v2-2-vision-curriculum-integration`; never implement on
  `main`.
- Preserve the already committed checkout-stability fix `04c6cf5` and design
  commit `399efa3`.
- Merge the algorithm branch; do not copy files manually or squash away its
  history.
- Use `apply_patch` for source, tests, Markdown, JSON and PowerShell edits.
- For every behavior change: add one focused failing test, run it and record
  the expected failure, implement the minimum change, rerun the focused test,
  then run the listed regression.
- Keep one implementation concern per commit. Never use `--no-verify`.
- Do not modify either formal `.ttt`, robot assets, URDF, STL, or
  `vision_platform/recognition/color_shape.py`.
- Add every new formal file to `RETAINED_FILES.txt` in the same Task that
  creates it.
- A skipped CoppeliaSim or UI evidence test is not a PASS. Online acceptance
  requires zero skips in the explicitly enabled JUnit report.
- Keep hardware as `PENDING_HARDWARE` and teaching effect as
  `PENDING_HUMAN_ACCEPTANCE`.
- User approval covers the recommended in-scope engineering choices and
  unattended execution. Stop only for a genuine blocker or a required scope
  expansion.

Use this interpreter for all Python commands:

```powershell
.\.venv-vision\Scripts\python.exe
```

## Task 1: Merge and revalidate the pure algorithm kernel

**Files:**

- Merge: `codex/v2-2-vision2d-algorithm-kernel` at `f5bef91`
- Resolve: `RETAINED_FILES.txt`
- Verify: `vision_platform/vision2d/**`
- Verify: `tests/test_vision2d/**`
- Verify: `docs/experiments/V1-02.md` through `V1-05.md`

- [ ] **Step 1: Confirm clean integration state and exact tips**

```powershell
git status --short --branch
git rev-parse HEAD
git rev-parse codex/v2-2-vision2d-algorithm-kernel
```

Expected: clean branch containing this plan commit; design commit `399efa3` is
an ancestor; algorithm tip is `f5bef91`.

- [ ] **Step 2: Merge with history**

```powershell
git merge --no-ff codex/v2-2-vision2d-algorithm-kernel
```

Expected: only `RETAINED_FILES.txt` may conflict. Resolve it as the union of
both sorted logical file sets. Preserve `.gitattributes`, V1-01, the new design,
all algorithm sources/tests/guides, and both earlier specs/plans.

If Git pauses for that conflict, stage the resolved retained list and complete
the merge without changing its generated message:

```powershell
git add -- RETAINED_FILES.txt
git commit --no-edit
```

- [ ] **Step 3: Prove the merge contains the exact algorithm tip**

```powershell
git merge-base --is-ancestor f5bef91 HEAD
git status --short
```

Expected: ancestor check exits 0 and no conflict markers remain.

- [ ] **Step 4: Re-run algorithm and integration baselines**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: algorithm tests and full static suite pass. Existing explicit online
skips remain skips and are reported, not counted as online PASS.

- [ ] **Step 5: Commit only if conflict resolution required an additional commit**

Record the resulting merge hash in the implementation log; do not create an
empty follow-up commit.

## Task 2: Build the strict curriculum configuration adapter

**Files:**

- Create: `vision_platform/vision2d/curriculum.py`
- Modify: `vision_platform/vision2d/__init__.py`
- Create: `tests/test_vision2d/test_curriculum.py`
- Modify: `tests/test_vision2d/test_delivery.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write failing exact-schema and ROI tests**

Cover:

- exact `vision2d` field set;
- `profile_id=standard` and 512×512 published resolution;
- finite numeric bounds and four-coordinate ROI semantics;
- optional two-value positive finite `pixel_scale_mm`;
- rejection of booleans, NaN, infinity, extra fields and invalid ROI;
- full-size ROI masking with pixels outside zero and inside unchanged;
- source image remains unchanged and result image owns independent bytes;
- conversion to `Vision2DConfig` with/without `PixelScale`.

Run:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d/test_curriculum.py
```

Expected RED: import or missing API failure.

- [ ] **Step 2: Implement immutable normalized configuration**

Create a frozen `CurriculumVision2DConfig` and public helpers:

```python
parse_curriculum_config(public_parameters, *, image_size)
mask_to_curriculum_roi(image_bgr, config)
```

The parser reads only `public_parameters["vision2d"]`, accepts no aliases or
fallbacks, copies JSON values, and raises `Vision2DConfigError` with stable code
`VISION2D_CONFIG_INVALID`.

- [ ] **Step 3: Run focused GREEN and algorithm regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision2d/test_curriculum.py `
  tests/test_vision2d
```

- [ ] **Step 4: Register new formal files and commit**

```powershell
git diff --check
git add -- vision_platform/vision2d tests/test_vision2d RETAINED_FILES.txt
git commit -m "feat(vision2d): add curriculum analysis configuration"
```

## Task 3: Add the allowlisted command and strict student SDK result

**Files:**

- Modify: `vision_platform/student/protocol.py`
- Modify: `vision_platform/student/sdk.py`
- Modify: `vision_platform/student/__init__.py`
- Modify: `tests/test_student_programs/test_protocol.py`
- Modify: `tests/test_student_programs/test_sdk.py`

- [ ] **Step 1: Write protocol RED tests**

Require `vision2d.analyze` in `ALLOWED_COMMANDS`, verify its command message is
accepted, and confirm plausible dangerous variants such as
`vision2d.analyze_file`, `vision2d.configure` and raw OpenCV commands remain
rejected.

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_protocol.py
```

Expected RED: command not allowed.

- [ ] **Step 2: Add only the exact command**

Update the immutable allowlist without changing schema version 1 or any existing
command spelling.

- [ ] **Step 3: Write SDK RED tests**

Define the exact successful response contract and test:

- `ctx.vision2d.analyze()` sends an empty argument object;
- returned object is frozen/read-only at its public boundary;
- snapshot/bundle/profile/status/image size/targets/rejections are exposed;
- result JSON is copied and contains only finite JSON-native values;
- missing/extra top-level fields, invalid IDs, invalid status, invalid size,
  non-list targets, non-mapping target items, non-finite values and mutable
  aliasing are rejected as `PROTOCOL_RESPONSE_INVALID`.

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_sdk.py -k vision2d
```

Expected RED: `StudentContext` has no `vision2d` property.

- [ ] **Step 4: Implement `StudentVision2D` and result model**

Do not expose NumPy images or accept parameters. Reuse `_copy_json_native` and
add the minimum strict normalization needed for the exact response.

- [ ] **Step 5: Run protocol/SDK regression and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_protocol.py `
  tests/test_student_programs/test_sdk.py
git diff --check
git add -- vision_platform/student tests/test_student_programs
git commit -m "feat(student): expose controlled vision2d analysis"
```

## Task 4: Execute analysis in the experiment gateway and record evidence

**Files:**

- Modify: `vision_platform/experiments/capabilities.py`
- Modify: `vision_platform/student/experiment_gateway.py`
- Modify: `vision_platform/student/runner.py`
- Modify: `tests/test_experiments/test_capabilities.py`
- Modify: `tests/test_student_programs/test_experiment_gateway.py`
- Modify: `tests/test_student_programs/test_runner.py`

- [ ] **Step 1: Write capability RED tests**

Require `vision2d.analysis` to be known only when a CoppeliaSim RGB camera and
the paired visual profile capabilities are available. Verify replay, Hikvision,
missing camera, missing sim and incomplete profile pairs are not accepted as a
formal online analysis context.

- [ ] **Step 2: Write gateway RED tests**

Use deterministic fake frames and real algorithm code to require:

- exact empty args;
- required capability and experiment binding;
- strict published configuration;
- current profile and resolution match;
- one camera read and one raw snapshot record;
- ROI excludes a colored distractor outside the board;
- algorithm result contains the expected three synthetic targets;
- result bundle layers are exactly `raw`, `roi-input`, `foreground-mask`,
  `cleaned-mask`, `annotated` in that order;
- masks are converted to BGR without changing their values;
- raw evidence is reused, not duplicated;
- bundle profile contains both camera and normalized curriculum config;
- response matches the SDK contract;
- invalid config, mismatched profile, algorithm error and evidence error fail
  closed.

Run focused RED:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_capabilities.py -k vision2d `
  tests/test_student_programs/test_experiment_gateway.py -k vision2d
```

- [ ] **Step 3: Refactor capture internals without changing public capture**

Extract one private raw-frame capture path that returns validated image,
metadata, encoded PNG and evidence record. Keep `camera.capture` output and its
single-layer V1-01 bundle byte-for-byte compatible at the contract level.

- [ ] **Step 4: Implement `vision2d.analyze` gateway dispatch**

Call the curriculum adapter and algorithm in the host process. Translate
configuration failures to `VISION2D_CONFIG_INVALID`, profile mismatches to
`VISION2D_PROFILE_MISMATCH`, and unexpected algorithm failures to
`VISION2D_ANALYSIS_FAILED` while preserving the cause.

- [ ] **Step 5: Route the command through the runner**

Add `vision2d.analyze` to the experiment-gateway dispatch set. The response is
JSON-safe, so it must not use the special raw-byte `camera.capture` response
path.

- [ ] **Step 6: Run GREEN plus compatibility regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_capabilities.py `
  tests/test_student_programs/test_experiment_gateway.py `
  tests/test_student_programs/test_runner.py `
  tests/test_vision_quality/test_results.py `
  tests/test_vision_quality/test_evidence.py `
  tests/test_vision_quality/test_v1_01_materials.py
```

- [ ] **Step 7: Commit the host integration**

```powershell
git diff --check
git add -- vision_platform/experiments vision_platform/student `
  tests/test_experiments tests/test_student_programs
git commit -m "feat(student): run vision2d analysis through gateway"
```

## Task 5: Deliver the V1-02 vertical slice

**Files:**

- Create: `config/experiments/V1-02.json`
- Modify: `config/experiments/catalog.json`
- Modify: `docs/experiments/V1-02.md`
- Create: `student_programs/templates/v1_02_size_measurement.py`
- Create: `tests/test_vision_quality/test_v1_02_materials.py`
- Modify: `tests/test_experiments/test_formal_catalog.py`
- Modify: `tests/test_experiments/test_student_templates.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write V1-02 materials RED tests**

Require exact definition fields, formal scene/manifest reuse, capabilities,
strict `vision2d` configuration, explicit pixel scale, three expected shapes,
size focus, automated/human checks, `PENDING_HARDWARE`, guide sections and a
template that uses only public SDK calls.

The real-public-context template test must expect this command sequence:

```text
experiment.info
camera.profile.apply standard
vision2d.analyze
context.log ...
context.checkpoint ...
camera.profile.reset
```

Run and observe missing-file RED.

- [ ] **Step 2: Create the strict V1-02 definition**

Use `version=2.2.0`, the existing vision-quality scene and manifest, capabilities
`camera.rgb`, paired profiles, `vision2d.analysis`, `experiment.info`, and
`scene.probe`. Publish the approved ROI and explicit simulated scale.

- [ ] **Step 3: Extend the guide into a formal curriculum document**

Preserve the algorithm guide's valid explanations. Add unified entry points,
safety boundary, exact student steps, expected five-layer evidence, automated
checks, human checks, error interpretation and hardware/teaching boundaries.

- [ ] **Step 4: Implement the public-SDK-only template**

Read experiment info, validate the focus/profile expectations, apply standard,
analyze once, log each target's pixel and millimetre dimensions, checkpoint the
result, and reset profile in `finally` without hiding the primary exception.

- [ ] **Step 5: Register V1-02 and retained paths**

Append `V1-02.json` after V1-01. Update formal catalog and delivery expectations
without changing any R1/V1-01 contract.

- [ ] **Step 6: Run GREEN and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_quality/test_v1_02_materials.py `
  tests/test_experiments/test_formal_catalog.py `
  tests/test_experiments/test_student_templates.py
git diff --check
git add -- config/experiments docs/experiments/V1-02.md `
  student_programs/templates/v1_02_size_measurement.py `
  tests/test_vision_quality/test_v1_02_materials.py `
  tests/test_experiments RETAINED_FILES.txt
git commit -m "feat(curriculum): add V1-02 size measurement lab"
```

## Task 6: Prove V1-02 in the formal CoppeliaSim scene

**Files:**

- Create: `tests/test_acceptance/test_coppeliasim_v1_02.py`
- Modify: `tools/vision_lab/run_vision_quality_acceptance.ps1`
- Modify: `tests/test_acceptance/test_powershell_process_ownership.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write the explicitly enabled online RED test**

Run V1-02 through `vision_platform.cli experiment-run`. Require PASS,
`PENDING_HARDWARE`, initial/final probe PASS, standard final profile, exactly one
analysis command, three detected targets, non-null finite millimetre dimensions,
five exact layers, valid bundle hashes and paths, and no raw-only duplicate
bundle.

Before implementation is complete, run it only against an owned CoppeliaSim
process and confirm the expected contract failure. A connection failure is not
an acceptable RED.

- [ ] **Step 2: Extend the owned-process acceptance wrapper**

Add the V1-02 online test and experiment run to the existing wrapper. Update
the exact JUnit test count, timeouts, JSON summary, cleanup assertions and
process ownership tests. Do not remove V1-01 coverage.

- [ ] **Step 3: Run owned online GREEN**

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\vision_lab\run_vision_quality_acceptance.ps1 `
  -OutputDir artifacts\vision_lab\v2-2-c1-v1-02
```

Expected: wrapper PASS; JUnit has the exact enabled tests and zero skips,
failures or errors; V1-01 and V1-02 experiment runs PASS; port 23005 is free
after cleanup.

- [ ] **Step 4: Commit acceptance infrastructure**

```powershell
git diff --check
git add -- tests/test_acceptance `
  tools/vision_lab/run_vision_quality_acceptance.ps1 RETAINED_FILES.txt
git commit -m "test(acceptance): verify V1-02 live analysis"
```

## Task 7: Add V1-03 positioning and angle measurement

**Files:**

- Create: `config/experiments/V1-03.json`
- Modify: `config/experiments/catalog.json`
- Modify: `docs/experiments/V1-03.md`
- Create: `student_programs/templates/v1_03_pose_measurement.py`
- Create: `tests/test_vision_quality/test_v1_03_materials.py`
- Modify: shared catalog/template tests and `RETAINED_FILES.txt`

- [ ] **Step 1: Write materials and template RED tests**

Require pose focus, `pixel_scale_mm=null`, center/rotated-box/angle fields,
circle angle null, circle undefined flag, and preservation of square ambiguity.

- [ ] **Step 2: Add definition, formal guide and public template**

The template applies standard, analyzes once, logs center and angle, treats
circle `None` as defined behavior rather than an error, checkpoints, and resets.

- [ ] **Step 3: Register, run GREEN and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_quality/test_v1_03_materials.py `
  tests/test_experiments/test_formal_catalog.py `
  tests/test_experiments/test_student_templates.py
git diff --check
git add -- config/experiments docs/experiments/V1-03.md `
  student_programs/templates/v1_03_pose_measurement.py `
  tests/test_vision_quality/test_v1_03_materials.py `
  tests/test_experiments RETAINED_FILES.txt
git commit -m "feat(curriculum): add V1-03 pose measurement lab"
```

## Task 8: Add V1-04 perimeter and area measurement

**Files:**

- Create: `config/experiments/V1-04.json`
- Modify: `config/experiments/catalog.json`
- Modify: `docs/experiments/V1-04.md`
- Create: `student_programs/templates/v1_04_geometry_measurement.py`
- Create: `tests/test_vision_quality/test_v1_04_materials.py`
- Modify: shared catalog/template tests and `RETAINED_FILES.txt`

- [ ] **Step 1: Write materials and template RED tests**

Require geometry focus, explicit simulated scale, pixel/mm perimeter and area,
circularity, aspect ratio, finite results, evidence explanation and no accuracy
certification language.

- [ ] **Step 2: Add definition, formal guide and public template**

The template logs pixel and simulated physical geometry for all detections,
checkpoints once, and restores standard in `finally`.

- [ ] **Step 3: Register, run GREEN and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_quality/test_v1_04_materials.py `
  tests/test_experiments/test_formal_catalog.py `
  tests/test_experiments/test_student_templates.py
git diff --check
git add -- config/experiments docs/experiments/V1-04.md `
  student_programs/templates/v1_04_geometry_measurement.py `
  tests/test_vision_quality/test_v1_04_materials.py `
  tests/test_experiments RETAINED_FILES.txt
git commit -m "feat(curriculum): add V1-04 geometry measurement lab"
```

## Task 9: Add V1-05 color, shape and contour recognition

**Files:**

- Create: `config/experiments/V1-05.json`
- Modify: `config/experiments/catalog.json`
- Modify: `docs/experiments/V1-05.md`
- Create: `student_programs/templates/v1_05_color_shape.py`
- Create: `tests/test_vision_quality/test_v1_05_materials.py`
- Modify: shared catalog/template tests and `RETAINED_FILES.txt`

- [ ] **Step 1: Write materials and template RED tests**

Require appearance focus, no physical scale, expected color/shape sets,
color/shape/vertex/contour/circularity fields, explicit preservation of
`unknown`, and no dependency on the legacy `recognition/color_shape.py` path.

- [ ] **Step 2: Add definition, formal guide and public template**

The template logs recognized and unknown labels without coercion, checkpoints
once, and restores the profile in `finally`.

- [ ] **Step 3: Register, run GREEN and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_quality/test_v1_05_materials.py `
  tests/test_experiments/test_formal_catalog.py `
  tests/test_experiments/test_student_templates.py
git diff --check
git add -- config/experiments docs/experiments/V1-05.md `
  student_programs/templates/v1_05_color_shape.py `
  tests/test_vision_quality/test_v1_05_materials.py `
  tests/test_experiments RETAINED_FILES.txt
git commit -m "feat(curriculum): add V1-05 color shape lab"
```

## Task 10: Publish all shared entry points and read-only UI behavior

**Files:**

- Modify: `tools/vision_lab/run_experiment.ps1`
- Modify: `tests/test_acceptance/test_powershell_process_ownership.py`
- Modify: `tests/test_experiments/test_cli.py`
- Modify: `tests/test_vision_platform/test_experiment_catalog_panel.py`
- Modify: `tests/test_vision_platform/test_vision_result_panel.py`
- Modify: `tests/test_vision_platform/test_pyqt_smoke.py`

- [ ] **Step 1: Write shared-entry RED tests**

Require catalog/CLI/PyQt order through V1-05, `experiment-show` for all four,
PowerShell validation for all IDs, and unchanged rejection of unknown IDs.

- [ ] **Step 2: Write multi-layer UI RED tests**

Load a five-layer recorded bundle. Require stable layer order, switching among
raw/masks/annotated, readable structured result, profile plus curriculum config,
`PENDING_HARDWARE`, and “not a course grade” boundary.

- [ ] **Step 3: Make the minimum shared-entry changes**

Extend only the existing experiment ID list and any label formatting needed for
the nested profile. Do not create V1-specific duplicate windows or launchers.

- [ ] **Step 4: Run UI/CLI/PowerShell regression and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_cli.py `
  tests/test_vision_platform/test_experiment_catalog_panel.py `
  tests/test_vision_platform/test_vision_result_panel.py `
  tests/test_vision_platform/test_pyqt_smoke.py `
  tests/test_acceptance/test_powershell_process_ownership.py
git diff --check
git add -- tools/vision_lab/run_experiment.ps1 `
  tests/test_acceptance/test_powershell_process_ownership.py `
  tests/test_experiments/test_cli.py tests/test_vision_platform
git commit -m "feat(ui): publish V1-02 through V1-05 entry points"
```

## Task 11: Run all four online experiments and capture UI evidence

**Files:**

- Create: `tests/test_acceptance/test_coppeliasim_v1_03_to_v1_05.py`
- Modify: `tools/vision_lab/run_vision_quality_acceptance.ps1`
- Modify: `tests/test_vision_platform/test_pyqt_smoke.py`
- Modify: delivery/retained tests as required

- [ ] **Step 1: Write online RED tests for V1-03～V1-05**

Reuse a helper that runs each formal template and loads its actual bundle.
Require common command/layer/hash/probe contracts plus topic-specific fields:

- V1-03: finite centers/boxes, circle angle null and other usable angles;
- V1-04: positive pixel and simulated physical perimeter/area;
- V1-05: three expected shapes and published color labels, contours non-empty.

- [ ] **Step 2: Extend the owned-process wrapper to all five V1 experiments**

Run the scene contract plus V1-01 through V1-05 online tests with exact JUnit
count and zero skip. Run four new formal `experiment-run` commands into separate
subdirectories. Preserve bounded timeouts and exact process cleanup.

- [ ] **Step 3: Execute online acceptance**

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\vision_lab\run_vision_quality_acceptance.ps1 `
  -OutputDir artifacts\vision_lab\v2-2-c1-v1-02-to-v1-05
```

If real-frame values reveal an ROI defect, add a failing contract test, adjust
only the published ROI consistently across the four definitions, rerun all
static material tests, and rerun online acceptance. Do not alter `.ttt` or relax
algorithm unit tolerances.

- [ ] **Step 4: Capture 100% and 125% PyQt screenshots from a real bundle**

Use the existing environment-variable screenshot harness, generalized from its
V1-01 name if necessary. Capture at least the annotated layer and one mask from
the online evidence. Inspect both images visually for labels, result text,
layer selector and boundary text.

- [ ] **Step 5: Commit online and UI acceptance**

```powershell
git diff --check
git add -- tests/test_acceptance tests/test_vision_platform `
  tools/vision_lab/run_vision_quality_acceptance.ps1 RETAINED_FILES.txt
git commit -m "test(acceptance): verify V1-02 through V1-05 online"
```

Do not add generated `artifacts/**` or screenshots to Git unless an existing
release contract explicitly requires those paths.

## Task 12: Release contract, full regression, boundary audit and push

**Files:**

- Modify: `tests/test_acceptance/test_delivery_contract.py`
- Modify: `tests/test_vision_quality/test_delivery.py`
- Modify: `README.md` if the existing user-facing catalog summary requires it
- Modify: `docs/视觉仿真实训平台使用说明.md`
- Modify: `docs/视觉仿真实训平台自动验收报告.md`
- Modify: this plan's checkboxes as Tasks are completed
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write release-contract RED tests**

Require every new source, test, definition, guide, template, plan and spec in
the delivery whitelist. Require no duplicate retained entries and no missing
paths. Check all formal V1 definitions remain `PENDING_HARDWARE` and contain no
grading fields.

- [ ] **Step 2: Update delivery documents with executed evidence only**

Record exact static, online JUnit, experiment-run and UI results. Mark true
hardware and human teaching acceptance pending. Do not copy old V1-01 counts as
new evidence.

- [ ] **Step 3: Run focused release tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision2d `
  tests/test_vision_quality `
  tests/test_student_programs `
  tests/test_experiments `
  tests/test_acceptance/test_delivery_contract.py
```

- [ ] **Step 4: Run fresh full static regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q
```

Record total passed/skipped counts. List every skipped online/hardware reason;
do not call skips PASS.

- [ ] **Step 5: Re-run final owned online acceptance**

Use a new output directory:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\vision_lab\run_vision_quality_acceptance.ps1 `
  -OutputDir artifacts\vision_lab\v2-2-c1-final
```

Inspect `acceptance-summary.json` and JUnit XML. Require overall PASS, exact
test count, zero skips/failures/errors, all experiment summaries PASS, hardware
pending, teaching pending, and free port after cleanup.

- [ ] **Step 6: Audit protected files against the design baseline**

Compare `399efa3` to HEAD for:

```text
simulation/vision_quality_lab/BL23_vision_quality_lab.ttt
simulation/vision_lab/BL23_vision_lab.ttt
simulation/vision_lab/robot_assets/**
vision_platform/recognition/color_shape.py
**/*.urdf
**/*.stl
```

Expected: zero changes.

- [ ] **Step 7: Audit whitespace, branch and worktree**

```powershell
git diff --check
git status --short --branch
git log --oneline --decorate -15
```

- [ ] **Step 8: Commit final delivery evidence**

```powershell
git add -- README.md RETAINED_FILES.txt docs tests/test_acceptance `
  tests/test_vision_quality
git commit -m "docs: complete V1-02 through V1-05 delivery"
```

If a listed file has no change, do not stage or fabricate a change.

- [ ] **Step 9: Push the integration branch**

```powershell
git push -u origin codex/v2-2-vision-curriculum-integration
```

- [ ] **Step 10: Verify remote equality**

```powershell
git fetch origin
git rev-parse HEAD
git rev-parse origin/codex/v2-2-vision-curriculum-integration
git status --short --branch
```

Expected: local and remote hashes are identical and the worktree is clean.

## Final handoff contents

The final response must include:

- integration branch and exact remote commit;
- merge commit proving `f5bef91` ancestry;
- per-Task commit list;
- changed-file groups and protected-file audit;
- algorithm, focused, full-static and explicit online results separately;
- online evidence and UI screenshot paths;
- retained/delivery contract counts;
- `PENDING_HARDWARE` and `PENDING_HUMAN_ACCEPTANCE` boundaries;
- recommended next step: independent review, then merge into `main` only after
  review approval.
