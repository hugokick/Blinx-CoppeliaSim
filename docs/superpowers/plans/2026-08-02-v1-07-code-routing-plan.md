# V2.2 V1-07 Code Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver V1-07 as a complete CoppeliaSim teaching loop that recognizes QR and EAN-13 symbols, forms a host-owned whitelist route plan, and safely moves four simulated parts into their assigned bins with reproducible evidence.

**Architecture:** A no-argument student SDK command asks the host to capture one fixed-profile frame, run the existing pure code-recognition kernel, map decoded payloads through the experiment definition and scene manifest, and atomically activate a frozen route plan. A route guard then permits only the finite pick-and-place trajectory for those approved entries while preserving the existing workspace, speed, pause, stop, reset, cleanup, and evidence controls. The new lab uses its own scene and original generated code assets; no existing formal scene or robot asset is changed.

**Tech Stack:** Python 3.11, NumPy, OpenCV, PyQt5, pytest, CoppeliaSim ZeroMQ Remote API, PowerShell, Git/GitHub.

---

## 0. Required design and non-negotiable rules

Read completely before implementation:

- `docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md`
- `docs/superpowers/plans/2026-08-02-v1-07-code-routing-plan.md`

Execution rules:

- Work only on branch `codex/v2-2-v1-07-code-routing` in a dedicated worktree created from the gated `origin/main` described in Task 1.
- Do not begin implementation if Task 1 does not pass exactly. Do not start from either candidate feature branch.
- Use `apply_patch` for source, tests, JSON, Markdown, and PowerShell. Generate PNG and `.ttt` files only through committed reproducible tools.
- For every behavior change, add one focused failing test, run it and confirm the expected contract failure, implement the minimum change, run focused GREEN, run the listed regression, then make the Task commit.
- Do not merge `main`, cherry-pick the D1 branch, or modify `vision_platform/rgbd/**` and `tests/test_rgbd/**`.
- Never modify existing formal `.ttt` files, URDF, STL, meshes, robot assets, or the existing `simulation/vision_quality_lab/**` scene family.
- The only allowed `vision_platform/vision2d/**` changes are package exports or the smallest adapter needed to consume the already-integrated `code_recognition.py` contract. Do not rewrite the recognition algorithm in this branch.
- Code payload is inert data used only as an exact whitelist key. Never evaluate it or pass it to Python, Lua, a shell, a path API, CoppeliaSim, or robot commands.
- The student command accepts no ROI, threshold, code type, file path, payload, coordinates, or command string. Motion is forbidden until all four readings and the entire route plan pass validation.
- Add every new formal file to `RETAINED_FILES.txt` in the same Task. Append only; do not remove, reorder, or deduplicate unrelated entries.
- A skipped online test is not PASS. Hardware remains `PENDING_HARDWARE`; teaching effectiveness remains `PENDING_HUMAN_ACCEPTANCE`.
- Stop only for a real blocker or scope expansion. Record RED and GREEN commands in the execution report as work proceeds.

Use the worktree-local interpreter for Python commands:

```powershell
.\.venv-vision\Scripts\python.exe
```

## Task 1: Establish the exact integrated baseline and isolated worktree

**Files:**

- Verify: `docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md`
- Verify: `docs/superpowers/plans/2026-08-02-v1-07-code-routing-plan.md`
- Verify: `config/experiments/V1-06.json`
- Verify: `vision_platform/vision2d/template_matching.py`
- Verify: `vision_platform/vision2d/code_recognition.py`
- Verify: `vision_platform/vision2d/ocr.py`
- Verify: `vision_platform/vision2d/defect_detection.py`
- Verify: `tests/test_vision2d/test_defect_detection.py`

- [ ] **Step 1: Fetch and prove that the source repository is clean**

```powershell
git fetch origin --prune
git status --short --branch
git rev-parse origin/main
```

Expected: the current coordination or source checkout has no uncommitted implementation files. Record the exact `origin/main` SHA in the execution report.

- [ ] **Step 2: Verify all integration artifacts are present on `origin/main`**

```powershell
git cat-file -e origin/main:config/experiments/V1-06.json
git cat-file -e origin/main:vision_platform/vision2d/template_matching.py
git cat-file -e origin/main:vision_platform/vision2d/code_recognition.py
git cat-file -e origin/main:vision_platform/vision2d/ocr.py
git cat-file -e origin/main:vision_platform/vision2d/defect_detection.py
git cat-file -e origin/main:docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md
git cat-file -e origin/main:docs/superpowers/plans/2026-08-02-v1-07-code-routing-plan.md
```

Expected: every command exits 0. If any path is absent, stop and report `BASELINE_GATE_NOT_READY`; do not substitute a local candidate branch.

- [ ] **Step 3: Verify the final multi-component defect regressions semantically**

```powershell
git show origin/main:tests/test_vision2d/test_defect_detection.py
```

Confirm the file contains both behaviors: an identical multi-component reference/candidate is not `broken`; when one component of a multi-component reference is split, only `broken` is reported. Then run:

```powershell
git worktree add `
  C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-code-routing `
  -b codex/v2-2-v1-07-code-routing origin/main
Set-Location C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-code-routing
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision2d/test_defect_detection.py `
  tests/test_vision2d/test_code_recognition.py `
  tests/test_vision2d/test_template_matching.py
```

Expected: all selected tests pass with zero failures. Missing worktree-local Python is an environment setup issue; restore the approved local environment or junction before judging the baseline.

- [ ] **Step 4: Run the complete static baseline**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
git status --short --branch
```

Expected: zero failures and a clean branch. Report skips separately; do not call them online PASS.

- [ ] **Step 5: Record the baseline without creating a commit**

Record the exact base SHA, test command, pass/fail/skip outcome, and worktree path in the implementation report. Do not create an empty baseline commit.

## Task 2: Add the immutable code-route planning contract

**Files:**

- Create: `vision_platform/experiments/code_routing.py`
- Create: `tests/test_experiments/test_code_routing.py`
- Modify: `vision_platform/experiments/__init__.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write strict RED tests for route configuration and full-plan validation**

Cover exact-schema parsing, finite built-in numbers, exact QR/EAN-13 payload matching, unique part IDs, unique drop slots, nonempty route IDs, decoded-only readings, confidence threshold, bounded pixel centers, affine pixel-to-world mapping, fixed pick Z, fixed safe Z, source input non-mutation, deterministic ordering, JSON-native output, and rejection of extra fields, duplicate/unknown/unreadable codes, non-finite values, booleans, or mismatched scene part sets. Task 4 binds every route ID/drop pair to the hash-bound scene slots before capture.

Create the test file with this successful contract plus the explicit failure table:

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from vision_platform.experiments.code_routing import (
    CodeRouteError,
    build_code_route_plan,
    code_route_plan_to_dict,
)
from vision_platform.vision2d.code_recognition import CodeReading, CodeRecognitionResult


def _reading(code_type: str, payload: str, center: tuple[float, float]) -> CodeReading:
    u, v = center
    return CodeReading(
        code_type=code_type,
        data=payload,
        polygon_px=((u - 2, v - 2), (u + 2, v - 2), (u + 2, v + 2), (u - 2, v + 2)),
        bbox_px=(int(u - 2), int(v - 2), 4, 4),
        center_px=center,
        confidence=0.98,
        decoded=True,
    )


def _recognition() -> CodeRecognitionResult:
    return CodeRecognitionResult(
        status="PASS",
        readings=(
            _reading("qr", "V1-07-B", (80.0, 45.0)),
            _reading("ean13", "6901234567892", (40.0, 95.0)),
            _reading("qr", "V1-07-A", (40.0, 45.0)),
            _reading("ean13", "6901234567809", (80.0, 95.0)),
        ),
        image_size=(512, 512),
        failure_code=None,
        detector_order=("qr", "ean13"),
        processing_ms=4.0,
    )


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "expected_count": 4,
        "confidence_min": 0.90,
        "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, -90.0]],
        "pick_z_mm": 18.0,
        "safe_z_mm": 110.0,
        "speed_mm_s": 15.0,
        "routes": [
            {"entry_id": "entry_a", "part_id": "part_a", "code_type": "qr", "payload": "V1-07-A", "route_id": "route_red", "drop_xyz_mm": [116.0, -60.0, 22.0]},
            {"entry_id": "entry_b", "part_id": "part_b", "code_type": "qr", "payload": "V1-07-B", "route_id": "route_blue", "drop_xyz_mm": [116.0, 60.0, 22.0]},
            {"entry_id": "entry_c", "part_id": "part_c", "code_type": "ean13", "payload": "6901234567892", "route_id": "route_red", "drop_xyz_mm": [128.0, -60.0, 22.0]},
            {"entry_id": "entry_d", "part_id": "part_d", "code_type": "ean13", "payload": "6901234567809", "route_id": "route_blue", "drop_xyz_mm": [128.0, 60.0, 22.0]},
        ],
    }


WORKSPACE = {"x_mm": [20.0, 140.0], "y_mm": [-90.0, 90.0], "z_mm": [10.0, 140.0], "safe_z_mm": 110.0}


def test_builds_deterministic_complete_route_plan() -> None:
    config = _config()
    untouched = deepcopy(config)
    plan = build_code_route_plan(
        _recognition(),
        routing_config=config,
        workspace=WORKSPACE,
        scene_part_ids={"part_a", "part_b", "part_c", "part_d"},
        image_size=(512, 512),
    )
    assert config == untouched
    assert tuple(entry.entry_id for entry in plan.entries) == (
        "entry_a", "entry_b", "entry_c", "entry_d"
    )
    assert plan.entries[0].pick_xyz_mm == pytest.approx((40.0, -45.0, 18.0))
    assert plan.entries[2].drop_xyz_mm == (128.0, -60.0, 22.0)
    assert len(plan.plan_id) == 64
    assert code_route_plan_to_dict(plan)["status"] == "PASS"


def test_plan_id_binds_the_approved_safety_parameters() -> None:
    first = build_code_route_plan(
        _recognition(), routing_config=_config(), workspace=WORKSPACE,
        scene_part_ids={"part_a", "part_b", "part_c", "part_d"},
        image_size=(512, 512),
    )
    changed = _config()
    changed["speed_mm_s"] = 14.0
    second = build_code_route_plan(
        _recognition(), routing_config=changed, workspace=WORKSPACE,
        scene_part_ids={"part_a", "part_b", "part_c", "part_d"},
        image_size=(512, 512),
    )
    assert first.plan_id != second.plan_id


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda config, recognition: config.update({"extra": True}), "CODE_ROUTE_CONFIG_INVALID"),
        (lambda config, recognition: config.update({"confidence_min": float("nan")}), "CODE_ROUTE_CONFIG_INVALID"),
        (lambda config, recognition: config["routes"].append(deepcopy(config["routes"][0])), "CODE_ROUTE_CONFIG_INVALID"),
    ],
)
def test_route_plan_fails_closed(mutation, error_code: str) -> None:
    config = _config()
    recognition = _recognition()
    mutation(config, recognition)
    with pytest.raises(CodeRouteError) as captured:
        build_code_route_plan(
            recognition,
            routing_config=config,
            workspace=WORKSPACE,
            scene_part_ids={"part_a", "part_b", "part_c", "part_d"},
            image_size=(512, 512),
        )
    assert captured.value.code == error_code


@pytest.mark.parametrize(
    "case",
    ["unknown", "undecoded", "low_confidence", "duplicate", "wrong_size", "out_of_bounds"],
)
def test_recognition_set_must_be_complete_and_approved(case: str) -> None:
    recognition = _recognition()
    readings = list(recognition.readings)
    if case == "unknown":
        readings[0] = replace(readings[0], data="UNKNOWN")
    elif case == "undecoded":
        readings[0] = replace(readings[0], data=None, decoded=False, failure_code="QR_DECODE_FAILED")
    elif case == "low_confidence":
        readings[0] = replace(readings[0], confidence=0.2)
    elif case == "duplicate":
        readings[1] = replace(readings[1], data=readings[0].data, code_type=readings[0].code_type)
    elif case == "out_of_bounds":
        readings[0] = replace(readings[0], center_px=(700.0, 45.0))
    recognition = replace(
        recognition,
        readings=tuple(readings),
        image_size=(320, 240) if case == "wrong_size" else recognition.image_size,
    )
    with pytest.raises(CodeRouteError) as captured:
        build_code_route_plan(
            recognition,
            routing_config=_config(),
            workspace=WORKSPACE,
            scene_part_ids={"part_a", "part_b", "part_c", "part_d"},
            image_size=(512, 512),
        )
    assert captured.value.code == "CODE_ROUTE_RECOGNITION_INVALID"


@pytest.mark.parametrize("case", ["boolean", "duplicate_part", "duplicate_drop", "scene_mismatch"])
def test_config_and_scene_binding_fail_closed(case: str) -> None:
    config = _config()
    scene_parts = {"part_a", "part_b", "part_c", "part_d"}
    if case == "boolean":
        config["pick_z_mm"] = True
    elif case == "duplicate_part":
        config["routes"][1]["part_id"] = config["routes"][0]["part_id"]
    elif case == "duplicate_drop":
        config["routes"][1]["drop_xyz_mm"] = config["routes"][0]["drop_xyz_mm"]
    else:
        scene_parts.remove("part_d")
    with pytest.raises(CodeRouteError) as captured:
        build_code_route_plan(
            _recognition(),
            routing_config=config,
            workspace=WORKSPACE,
            scene_part_ids=scene_parts,
            image_size=(512, 512),
        )
    assert captured.value.code == "CODE_ROUTE_CONFIG_INVALID"
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_experiments/test_code_routing.py
```

Expected RED: import failure for `vision_platform.experiments.code_routing`.

- [ ] **Step 2: Implement frozen route models and stable errors**

Use these public signatures and keep all validation in the host process:

```python
@dataclass(frozen=True)
class ApprovedCodeRoute:
    entry_id: str
    part_id: str
    code_type: str
    payload: str
    route_id: str
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        for name in ("entry_id", "part_id", "payload", "route_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value) > 80:
                raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", f"{name} is invalid")
        if self.code_type not in {"qr", "ean13"}:
            raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "code_type is invalid")
        for name in ("pick_xyz_mm", "drop_xyz_mm"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != 3
                or any(
                    type(component) not in {int, float}
                    or not math.isfinite(float(component))
                    for component in value
                )
            ):
                raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", f"{name} is invalid")
            object.__setattr__(self, name, tuple(float(component) for component in value))
        if (
            type(self.confidence) not in {int, float}
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "confidence is invalid")
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class CodeRoutePlan:
    plan_id: str
    entries: tuple[ApprovedCodeRoute, ...]
    safe_z_mm: float
    speed_mm_s: float
    status: str = "PASS"

    def __post_init__(self) -> None:
        if (
            type(self.plan_id) is not str
            or len(self.plan_id) != 64
            or any(character not in "0123456789abcdef" for character in self.plan_id)
            or not isinstance(self.entries, tuple)
            or len(self.entries) != 4
            or any(not isinstance(entry, ApprovedCodeRoute) for entry in self.entries)
            or len({entry.entry_id for entry in self.entries}) != 4
            or len({entry.part_id for entry in self.entries}) != 4
            or len({(entry.code_type, entry.payload) for entry in self.entries}) != 4
            or len({entry.drop_xyz_mm for entry in self.entries}) != 4
            or self.status != "PASS"
            or type(self.safe_z_mm) not in {int, float}
            or type(self.speed_mm_s) not in {int, float}
            or not math.isfinite(float(self.safe_z_mm))
            or not math.isfinite(float(self.speed_mm_s))
            or float(self.speed_mm_s) <= 0.0
            or float(self.safe_z_mm) <= max(
                coordinate
                for entry in self.entries
                for coordinate in (entry.pick_xyz_mm[2], entry.drop_xyz_mm[2])
            )
        ):
            raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "plan fields are invalid")
        object.__setattr__(self, "safe_z_mm", float(self.safe_z_mm))
        object.__setattr__(self, "speed_mm_s", float(self.speed_mm_s))


class CodeRouteError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def build_code_route_plan(
    recognition: CodeRecognitionResult,
    *,
    routing_config: Mapping[str, Any],
    workspace: Mapping[str, Any],
    scene_part_ids: Collection[str],
    image_size: tuple[int, int],
) -> CodeRoutePlan:
    required = {
        "schema_version", "expected_count", "confidence_min",
        "calibration_matrix", "pick_z_mm", "safe_z_mm", "speed_mm_s", "routes",
    }
    if not isinstance(routing_config, Mapping) or set(routing_config) != required:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "routing schema is invalid")

    def number(value: Any, name: str) -> float:
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", f"{name} must be finite")
        return float(value)

    def vector(value: Any, length: int, name: str) -> tuple[float, ...]:
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", f"{name} has invalid size")
        return tuple(number(item, f"{name}[{index}]") for index, item in enumerate(value))

    if routing_config["schema_version"] != 1 or type(routing_config["expected_count"]) is not int:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "version or count is invalid")
    expected_count = routing_config["expected_count"]
    if expected_count != 4:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "V1-07 requires four routes")
    confidence_min = number(routing_config["confidence_min"], "confidence_min")
    if not 0.0 <= confidence_min <= 1.0:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "confidence is outside [0, 1]")
    matrix_raw = routing_config["calibration_matrix"]
    if not isinstance(matrix_raw, (list, tuple)) or len(matrix_raw) != 2:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "calibration matrix is invalid")
    matrix = tuple(vector(row, 3, "calibration_matrix") for row in matrix_raw)
    pick_z = number(routing_config["pick_z_mm"], "pick_z_mm")
    safe_z = number(routing_config["safe_z_mm"], "safe_z_mm")
    speed = number(routing_config["speed_mm_s"], "speed_mm_s")

    if not isinstance(workspace, Mapping) or set(workspace) != {"x_mm", "y_mm", "z_mm", "safe_z_mm"}:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "workspace is invalid")
    x_range = vector(workspace["x_mm"], 2, "workspace.x_mm")
    y_range = vector(workspace["y_mm"], 2, "workspace.y_mm")
    z_range = vector(workspace["z_mm"], 2, "workspace.z_mm")
    if safe_z != number(workspace["safe_z_mm"], "workspace.safe_z_mm") or speed <= 0.0:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "safety values do not match")
    if not (x_range[0] < x_range[1] and y_range[0] < y_range[1] and z_range[0] < z_range[1]):
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "workspace ranges are invalid")

    routes_raw = routing_config["routes"]
    route_fields = {"entry_id", "part_id", "code_type", "payload", "route_id", "drop_xyz_mm"}
    if not isinstance(routes_raw, (list, tuple)) or len(routes_raw) != expected_count:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "route count is invalid")
    routes: dict[str, dict[str, Any]] = {}
    entry_ids: set[str] = set()
    part_ids: set[str] = set()
    drops: set[tuple[float, ...]] = set()
    for raw in routes_raw:
        if not isinstance(raw, Mapping) or set(raw) != route_fields:
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "route schema is invalid")
        text_fields = ("entry_id", "part_id", "payload", "route_id")
        if any(type(raw[name]) is not str or not raw[name] for name in text_fields):
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "route ID is invalid")
        if raw["code_type"] not in {"qr", "ean13"}:
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "code type is invalid")
        drop = vector(raw["drop_xyz_mm"], 3, "drop_xyz_mm")
        if raw["payload"] in routes or raw["entry_id"] in entry_ids or raw["part_id"] in part_ids or drop in drops:
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "route identifiers must be unique")
        normalized = dict(raw)
        normalized["drop_xyz_mm"] = drop
        routes[raw["payload"]] = normalized
        entry_ids.add(raw["entry_id"])
        part_ids.add(raw["part_id"])
        drops.add(drop)
    if part_ids != set(scene_part_ids):
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "scene parts do not match routes")
    if safe_z <= max([pick_z, *(drop[2] for drop in drops)]):
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "safe Z must exceed pick and drop Z")

    if (
        not isinstance(recognition, CodeRecognitionResult)
        or recognition.status != "PASS"
        or recognition.image_size != image_size
        or len(recognition.readings) != expected_count
    ):
        raise CodeRouteError("CODE_ROUTE_RECOGNITION_INVALID", "recognition set is incomplete")
    width, height = image_size
    approved: list[ApprovedCodeRoute] = []
    seen_payloads: set[str] = set()
    for reading in recognition.readings:
        if (
            not reading.decoded
            or reading.data is None
            or reading.data in seen_payloads
            or reading.data not in routes
            or reading.confidence < confidence_min
            or reading.code_type != routes[reading.data]["code_type"]
        ):
            raise CodeRouteError("CODE_ROUTE_RECOGNITION_INVALID", "reading is not approved")
        u_px, v_px = reading.center_px
        if not all(math.isfinite(value) for value in (u_px, v_px)) or not (0 <= u_px < width and 0 <= v_px < height):
            raise CodeRouteError("CODE_ROUTE_RECOGNITION_INVALID", "centre is out of bounds")
        pick = (
            matrix[0][0] * u_px + matrix[0][1] * v_px + matrix[0][2],
            matrix[1][0] * u_px + matrix[1][1] * v_px + matrix[1][2],
            pick_z,
        )
        route = routes[reading.data]
        drop = route["drop_xyz_mm"]
        points = (
            pick,
            drop,
            (pick[0], pick[1], safe_z + 1.0),
            (drop[0], drop[1], safe_z + 1.0),
        )
        if any(
            not (x_range[0] <= point[0] <= x_range[1] and y_range[0] <= point[1] <= y_range[1] and z_range[0] <= point[2] <= z_range[1])
            for point in points
        ):
            raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "planned point is outside workspace")
        approved.append(ApprovedCodeRoute(
            route["entry_id"], route["part_id"], reading.code_type, reading.data,
            route["route_id"], tuple(float(value) for value in pick), drop,
            float(reading.confidence),
        ))
        seen_payloads.add(reading.data)
    approved.sort(key=lambda item: item.entry_id)
    canonical = {
        "entries": [asdict(item) for item in approved],
        "safe_z_mm": safe_z,
        "speed_mm_s": speed,
    }
    plan_id = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CodeRoutePlan(plan_id, tuple(approved), safe_z, speed)


def code_route_plan_to_dict(plan: CodeRoutePlan) -> dict[str, Any]:
    if not isinstance(plan, CodeRoutePlan):
        raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "plan type is invalid")
    return {
        "plan_id": plan.plan_id,
        "entries": [
            {
                **asdict(entry),
                "pick_xyz_mm": list(entry.pick_xyz_mm),
                "drop_xyz_mm": list(entry.drop_xyz_mm),
            }
            for entry in plan.entries
        ],
        "safe_z_mm": float(plan.safe_z_mm),
        "speed_mm_s": float(plan.speed_mm_s),
        "status": plan.status,
    }
```

Import `hashlib`, `json`, `math`, `asdict`, `dataclass`, `Any`, `Collection`, `Mapping`, and the two code-recognition models used above. Define the stable exception type exactly as shown. Compute `plan_id` from canonical JSON of the approved entries; never include wall-clock time or a filesystem path.

- [ ] **Step 3: Run focused GREEN and related algorithm regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_code_routing.py `
  tests/test_vision2d/test_code_recognition.py
git diff --check
```

- [ ] **Step 4: Register formal files and commit**

```powershell
git add -- vision_platform/experiments/code_routing.py `
  vision_platform/experiments/__init__.py `
  tests/test_experiments/test_code_routing.py RETAINED_FILES.txt
git commit -m "feat(experiments): add immutable code route plans"
```

## Task 3: Expose one no-argument SDK command with a frozen result

**Files:**

- Modify: `vision_platform/student/protocol.py`
- Modify: `vision_platform/student/sdk.py`
- Modify: `vision_platform/student/__init__.py`
- Modify: `tests/test_student_programs/test_protocol.py`
- Create: `tests/test_student_programs/test_v1_07_protocol_sdk.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write protocol RED tests**

Require exactly `vision2d.code_routes` in `ALLOWED_COMMANDS`. Verify an empty argument mapping is accepted by the message model and dangerous variants such as `vision2d.code_routes_file`, `vision2d.code_routes_configure`, raw payload arguments, path arguments, ROI arguments, and arbitrary command names remain rejected.

Add these exact protocol assertions to `test_protocol.py`:

```python
def test_code_routes_is_the_only_new_allowlisted_command() -> None:
    command = CommandMessage("route-000001", "vision2d.code_routes", {})
    assert command.to_dict()["args"] == {}
    assert "vision2d.code_routes" in ALLOWED_COMMANDS


@pytest.mark.parametrize(
    "name",
    [
        "vision2d.code_routes_file",
        "vision2d.code_routes_configure",
        "vision2d.decode_payload",
        "robot.execute_payload",
    ],
)
def test_code_route_command_variants_remain_forbidden(name: str) -> None:
    with pytest.raises(ValueError, match="COMMAND_NOT_ALLOWED"):
        CommandMessage("route-000001", name, {})
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_protocol.py -k code_routes
```

Expected RED: `COMMAND_NOT_ALLOWED`.

- [ ] **Step 2: Add only the exact allowlisted command**

Keep protocol schema version 1 and all existing command spellings unchanged.

- [ ] **Step 3: Write SDK RED tests for exact immutable data**

Require `ctx.vision2d.code_routes()` to send `{}` and return a frozen result containing only `schema_version`, `snapshot_id`, `vision_bundle_path`, `plan_id`, `status`, `safe_z_mm`, `speed_mm_s`, and `entries`. Each entry must expose the exact fields from `ApprovedCodeRoute`; nested lists and mappings must be copied and frozen. Reject missing/extra keys, booleans as numbers, NaN/Inf, malformed IDs, unexpected status, mutable aliasing, and more than four entries as `PROTOCOL_RESPONSE_INVALID`.

Create the SDK test file using the same fake connection style as V1-06:

```python
from __future__ import annotations

from copy import deepcopy

import pytest

from vision_platform.student.protocol import ResponseMessage
from vision_platform.student.sdk import StudentContext


class Connection:
    def __init__(self, value: object) -> None:
        self.value = value
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)

    def recv(self) -> dict:
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"], status="PASS", value=self.value, error=None
        ).to_dict()


def _entry(index: int) -> dict[str, object]:
    return {
        "entry_id": f"entry_{index}", "part_id": f"part_{index}",
        "code_type": "qr" if index < 2 else "ean13",
        "payload": f"payload-{index}", "route_id": "route_red" if index % 2 == 0 else "route_blue",
        "pick_xyz_mm": [40.0 + 10.0 * index, -45.0 + index, 18.0],
        "drop_xyz_mm": [116.0 + index, -60.0 if index % 2 == 0 else 60.0, 22.0],
        "confidence": 0.98,
    }


def _value() -> dict[str, object]:
    return {
        "schema_version": 1,
        "snapshot_id": "frame-000001",
        "vision_bundle_path": "vision-bundle-V1-07-frame-000001.json",
        "plan_id": "a" * 64,
        "status": "PASS",
        "safe_z_mm": 110.0,
        "speed_mm_s": 15.0,
        "entries": [_entry(index) for index in range(4)],
    }


def test_code_routes_sends_no_arguments_and_returns_frozen_result() -> None:
    raw = _value()
    connection = Connection(raw)
    result = StudentContext(connection).vision2d.code_routes()
    raw["entries"][0]["pick_xyz_mm"][0] = 999.0
    assert connection.sent[0]["name"] == "vision2d.code_routes"
    assert connection.sent[0]["args"] == {}
    assert result.plan_id == "a" * 64
    assert result.entries[0].pick_xyz_mm == (40.0, -45.0, 18.0)
    with pytest.raises((AttributeError, TypeError)):
        result.entries[0].pick_xyz_mm[0] = 1.0


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda value: value.update({"extra": 1}), "fields"),
        (lambda value: value.update({"schema_version": 2}), "schema_version"),
        (lambda value: value.update({"plan_id": "../plan"}), "plan_id"),
        (lambda value: value.update({"status": "PARTIAL"}), "status"),
        (lambda value: value.update({"safe_z_mm": float("nan")}), "safe_z_mm"),
        (lambda value: value["entries"].append(_entry(5)), "entries"),
        (lambda value: value["entries"].__setitem__(1, deepcopy(value["entries"][0])), "entries"),
        (lambda value: value["entries"][0].update({"payload": True}), "payload"),
    ],
)
def test_code_route_response_is_strict(mutation, field: str) -> None:
    value = deepcopy(_value())
    mutation(value)
    with pytest.raises(RuntimeError, match=field):
        StudentContext(Connection(value)).vision2d.code_routes()
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_v1_07_protocol_sdk.py
```

Expected RED: `StudentVision2D` has no `code_routes` method.

- [ ] **Step 4: Implement the strict SDK result**

Use a parameterless public method:

```python
@dataclass(frozen=True)
class StudentCodeRouteEntry:
    entry_id: str
    part_id: str
    code_type: str
    payload: str
    route_id: str
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True)
class CodeRoutePlanResult:
    schema_version: int
    snapshot_id: str
    vision_bundle_path: str
    plan_id: str
    status: str
    safe_z_mm: float
    speed_mm_s: float
    entries: tuple[StudentCodeRouteEntry, ...]


def _route_error(field: str) -> RuntimeError:
    return RuntimeError(f"PROTOCOL_RESPONSE_INVALID: vision2d.code_routes {field}")


def _route_number(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not isfinite(float(value)):
        raise _route_error(field)
    return float(value)


def _route_vector(value: Any, field: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise _route_error(field)
    return (
        _route_number(value[0], field),
        _route_number(value[1], field),
        _route_number(value[2], field),
    )


def _code_route_result(value: Any) -> CodeRoutePlanResult:
    fields = {
        "schema_version", "snapshot_id", "vision_bundle_path", "plan_id",
        "status", "safe_z_mm", "speed_mm_s", "entries",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _route_error("fields")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise _route_error("schema_version")
    snapshot_id = value["snapshot_id"]
    bundle = value["vision_bundle_path"]
    plan_id = value["plan_id"]
    if type(snapshot_id) is not str or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None:
        raise _route_error("snapshot_id")
    if type(bundle) is not str or _VISION_BUNDLE_NAME.fullmatch(bundle) is None or "/" in bundle or "\\" in bundle:
        raise _route_error("vision_bundle_path")
    if type(plan_id) is not str or len(plan_id) != 64 or any(character not in "0123456789abcdef" for character in plan_id):
        raise _route_error("plan_id")
    if value["status"] != "PASS":
        raise _route_error("status")
    entries_raw = value["entries"]
    if type(entries_raw) is not list or len(entries_raw) != 4:
        raise _route_error("entries")
    entry_fields = {
        "entry_id", "part_id", "code_type", "payload", "route_id",
        "pick_xyz_mm", "drop_xyz_mm", "confidence",
    }
    entries: list[StudentCodeRouteEntry] = []
    for index, raw in enumerate(entries_raw):
        if not isinstance(raw, Mapping) or set(raw) != entry_fields:
            raise _route_error(f"entries[{index}].fields")
        for name in ("entry_id", "part_id", "payload", "route_id"):
            if type(raw[name]) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}", raw[name]) is None:
                raise _route_error(name)
        if raw["code_type"] not in {"qr", "ean13"}:
            raise _route_error("code_type")
        confidence = _route_number(raw["confidence"], "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise _route_error("confidence")
        entries.append(StudentCodeRouteEntry(
            raw["entry_id"], raw["part_id"], raw["code_type"], raw["payload"],
            raw["route_id"], _route_vector(raw["pick_xyz_mm"], "pick_xyz_mm"),
            _route_vector(raw["drop_xyz_mm"], "drop_xyz_mm"), confidence,
        ))
    safe_z = _route_number(value["safe_z_mm"], "safe_z_mm")
    speed = _route_number(value["speed_mm_s"], "speed_mm_s")
    if (
        len({entry.entry_id for entry in entries}) != 4
        or len({entry.part_id for entry in entries}) != 4
        or len({(entry.code_type, entry.payload) for entry in entries}) != 4
        or len({entry.drop_xyz_mm for entry in entries}) != 4
        or speed <= 0.0
        or safe_z <= max(
            coordinate
            for entry in entries
            for coordinate in (entry.pick_xyz_mm[2], entry.drop_xyz_mm[2])
        )
    ):
        raise _route_error("entries")
    return CodeRoutePlanResult(
        1, snapshot_id, bundle, plan_id, "PASS",
        safe_z, speed, tuple(entries),
    )


class StudentVision2D:
    def code_routes(self) -> CodeRoutePlanResult:
        value = self._rpc.call("vision2d.code_routes")
        return _code_route_result(value)
```

Import `re` and reuse the existing `dataclass`, `isfinite`, `Any`, `Mapping`, `_SNAPSHOT_ID_PATTERN`, and `_VISION_BUNDLE_NAME` definitions. Add `"vision2d.code_routes"` to `ALLOWED_COMMANDS` and export the two result dataclasses through `vision_platform/student/__init__.py`. Do not expose OpenCV or NumPy objects to the child process.

- [ ] **Step 5: Run GREEN and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_protocol.py `
  tests/test_student_programs/test_sdk.py `
  tests/test_student_programs/test_v1_07_protocol_sdk.py
git diff --check
git add -- vision_platform/student tests/test_student_programs `
  RETAINED_FILES.txt
git commit -m "feat(student): expose controlled code route plans"
```

## Task 4: Build the host gateway and route finite-state guard

**Files:**

- Create: `vision_platform/student/code_route_guard.py`
- Modify: `vision_platform/student/experiment_gateway.py`
- Modify: `vision_platform/student/runner.py`
- Modify: `tests/test_student_programs/test_experiment_evidence.py`
- Modify: `tests/test_student_programs/test_runner.py`
- Modify: `vision_platform/experiments/capabilities.py`
- Modify: `vision_platform/vision_quality/controller.py`
- Create: `tests/test_student_programs/test_v1_07_route_guard.py`
- Create: `tests/test_student_programs/test_v1_07_gateway.py`
- Modify: `tests/test_experiments/test_capabilities.py`
- Modify: `tests/test_vision_quality/test_controller.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write capability and gateway RED tests**

Require `vision2d.code_routing` only when the selected formal experiment has a CoppeliaSim RGB camera, the fixed `standard` camera/lighting profile, code-recognition support, scene probes, robot motion, and suction. The gateway must reject replay, Hikvision, missing profiles, scene mismatch, malformed manifests, unavailable robot/tool, and every non-empty command argument before capture.

Use one deterministic synthetic frame with four approved codes and verify one capture, one recognition call, one complete plan, one raw layer, one annotated layer, copied JSON-native output, and no motion during planning. Add negative frames for an unknown payload, duplicate payload, one undecoded symbol, wrong count, low confidence, out-of-workspace mapping, and scene-part mismatch.

Create self-contained camera, evidence, application, definition, manifest, and recognition fixtures in `test_v1_07_gateway.py`; do not import another test module. Use this concrete fixture and core test:

```python
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import ExperimentAcceptance, ExperimentDefinition, ExperimentRunContext
from vision_platform.models import Frame
from vision_platform.student.experiment_gateway import StudentExperimentGateway
from vision_platform.vision2d.code_recognition import CodeReading, CodeRecognitionResult
from vision_platform.vision_quality.models import AppliedVisionProfile


class Camera:
    def __init__(self) -> None:
        self.read_calls = 0

    def read(self, timeout_s: float) -> Frame:
        assert timeout_s == 2.0
        self.read_calls += 1
        return Frame(
            image_bgr=np.full((1024, 1024, 3), 255, dtype=np.uint8),
            width=1024,
            height=1024,
            timestamp_s=1.0,
            source="coppeliasim",
            sequence_id=self.read_calls,
        )


class ProfileController:
    def current(self) -> AppliedVisionProfile:
        return AppliedVisionProfile(
            profile_id="standard", resolution=(1024, 1024),
            perspective_angle_deg=60.0, camera_rig_z_m=0.7,
            key_diffuse_rgb=(0.8, 0.8, 0.8), fill_diffuse_rgb=(0.35, 0.35, 0.35),
        )


class Evidence:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.json_calls: list[tuple[str, dict]] = []

    def record_snapshot(self, **payload: object) -> dict:
        path = self.root / "frames" / f"{payload['snapshot_id']}.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload["png_bytes"])
        return {
            "snapshot_id": payload["snapshot_id"],
            "path": path.relative_to(self.root).as_posix(),
            "sha256": hashlib.sha256(payload["png_bytes"]).hexdigest(),
            **payload["metadata"],
        }

    def record_json_artifact(self, name: str, payload: dict) -> str:
        self.json_calls.append((name, payload))
        (self.root / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return name


class Calls:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


def _reading(code_type: str, payload: str, center: tuple[float, float]) -> CodeReading:
    u, v = center
    return CodeReading(
        code_type, payload,
        ((u - 2, v - 2), (u + 2, v - 2), (u + 2, v + 2), (u - 2, v + 2)),
        (int(u - 2), int(v - 2), 4, 4), center, 0.98, True,
    )


def _recognition(case: str) -> CodeRecognitionResult:
    readings = [
        _reading("qr", "V1-07-A", (40.0, 45.0)),
        _reading("qr", "V1-07-B", (80.0, 45.0)),
        _reading("ean13", "6901234567892", (40.0, 95.0)),
        _reading("ean13", "6901234567809", (80.0, 95.0)),
    ]
    if case == "unknown":
        readings[0] = replace(readings[0], data="UNKNOWN")
    elif case == "duplicate":
        readings[1] = replace(readings[1], data=readings[0].data)
    elif case == "undecoded":
        readings[0] = replace(readings[0], data=None, decoded=False, failure_code="QR_DECODE_FAILED")
    elif case == "wrong_count":
        readings.pop()
    elif case == "low_confidence":
        readings[0] = replace(readings[0], confidence=0.2)
    elif case == "outside_workspace":
        readings[0] = replace(readings[0], center_px=(300.0, 45.0))
    return CodeRecognitionResult(
        "PASS", tuple(readings), (1024, 1024), None, ("qr", "ean13"), 4.0
    )


def _routing_config() -> dict[str, object]:
    return {
        "schema_version": 1, "expected_count": 4, "confidence_min": 0.90,
        "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, -90.0]],
        "pick_z_mm": 18.0, "safe_z_mm": 110.0, "speed_mm_s": 15.0,
        "routes": [
            {"entry_id": "entry_a", "part_id": "part_a", "code_type": "qr", "payload": "V1-07-A", "route_id": "route_red", "drop_xyz_mm": [116.0, -60.0, 22.0]},
            {"entry_id": "entry_b", "part_id": "part_b", "code_type": "qr", "payload": "V1-07-B", "route_id": "route_blue", "drop_xyz_mm": [116.0, 60.0, 22.0]},
            {"entry_id": "entry_c", "part_id": "part_c", "code_type": "ean13", "payload": "6901234567892", "route_id": "route_red", "drop_xyz_mm": [128.0, -60.0, 22.0]},
            {"entry_id": "entry_d", "part_id": "part_d", "code_type": "ean13", "payload": "6901234567809", "route_id": "route_blue", "drop_xyz_mm": [128.0, 60.0, 22.0]},
        ],
    }


def _gateway(tmp_path: Path, recognition_case: str = "pass"):
    scene = tmp_path / "BL23_vision_code_routing_lab.ttt"
    scene.write_bytes(b"v1-07-scene")
    asset_manifest = {
        "schema_version": 1,
        "entries": [
            {"part_id": route["part_id"], "code_type": route["code_type"], "payload": route["payload"]}
            for route in _routing_config()["routes"]
        ],
    }
    asset_path = tmp_path / "code_assets_manifest.json"
    asset_path.write_text(json.dumps(asset_manifest), encoding="utf-8")
    part_ids = ["part_a", "part_b", "part_c", "part_d"]
    if recognition_case == "scene_mismatch":
        part_ids.pop()
    initial_positions = {
        "part_a": [40.0, -45.0, 18.0],
        "part_b": [80.0, -45.0, 18.0],
        "part_c": [40.0, 5.0, 18.0],
        "part_d": [80.0, 5.0, 18.0],
    }
    if recognition_case == "calibration_mismatch":
        initial_positions["part_a"][0] += 10.0
    scene_manifest = {
        "schema_version": 1,
        "scene": {"path": str(scene), "sha256": hashlib.sha256(scene.read_bytes()).hexdigest()},
        "task_contracts": {},
        "code_assets_manifest": {"path": asset_path.name, "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest()},
        "code_routing": {
            "part_ids": part_ids,
            "initial_positions_mm": initial_positions,
            "calibration_plane_z_mm": 27.4,
            "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, -90.0]],
            "route_slots_mm": {
                "route_red": [[116.0, -60.0, 22.0], [128.0, -60.0, 22.0]],
                "route_blue": [[116.0, 60.0, 22.0], [128.0, 60.0, 22.0]],
            },
        },
    }
    scene_manifest_path = tmp_path / "scene_manifest.json"
    scene_manifest_path.write_text(json.dumps(scene_manifest), encoding="utf-8")
    public_parameters = {
        "camera_path": "/VisionCodeRoutingLab/CameraRig/Camera",
        "code_routing": _routing_config(),
    }
    workspace = {"x_mm": [20.0, 140.0], "y_mm": [-90.0, 90.0], "z_mm": [10.0, 140.0], "safe_z_mm": 110.0}
    definition = ExperimentDefinition(
        experiment_id="V1-07", pack_id="V1", title="Code routing", version="2.2.0",
        scene=scene, scene_manifest=scene_manifest_path,
        student_template=tmp_path / "student.py", guide=tmp_path / "guide.md",
        capabilities=("camera.rgb", "camera.profile", "lighting.profile", "vision2d.code_routing", "robot.home", "robot.pose", "robot.move_world", "tool.suction", "scene.probe"),
        workspace=workspace, public_parameters=public_parameters,
        acceptance=ExperimentAcceptance("code_route_occupancy", ("four_routes",), ("teacher_review",)),
        hardware_status="PENDING_HARDWARE",
    )
    context = ExperimentRunContext(
        "V1-07", "2.2.0", scene, scene_manifest["scene"]["sha256"],
        scene_manifest_path, public_parameters,
    )
    camera = Camera()
    application = SimpleNamespace(
        config=SimpleNamespace(camera_backend="sim", robot_backend="sim"),
        sim=object(), camera=camera, robot=Calls(), tool=Calls(),
    )
    evidence = Evidence(tmp_path / "evidence")
    gateway = StudentExperimentGateway(
        application=application, evidence=evidence, context=context,
        definition=definition, scene_manifest=scene_manifest,
        profile_controller=ProfileController(),
    )
    return gateway, camera, evidence, application, _recognition(recognition_case)


def test_gateway_builds_complete_plan_before_any_motion(tmp_path, monkeypatch) -> None:
    gateway, camera, evidence, application, recognition = _gateway(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.recognize_codes",
        lambda image: calls.append("recognize") or recognition,
    )

    value = gateway.dispatch("vision2d.code_routes", {})

    assert value["schema_version"] == 1
    assert value["status"] == "PASS"
    assert len(value["entries"]) == 4
    assert camera.read_calls == 1
    assert calls == ["recognize"]
    assert application.robot.calls == []
    assert application.tool.calls == []
    assert gateway.route_guard.completion()["active"] is True
    bundle = evidence.json_calls[-1][1]
    assert [layer["layer_id"] for layer in bundle["layers"]] == ["raw", "annotated"]
    assert bundle["result"]["plan_id"] == value["plan_id"]


@pytest.mark.parametrize("argument", [{"roi": [0, 0, 10, 10]}, {"payload": "V1-07-A"}, {"path": "code.png"}])
def test_gateway_rejects_all_student_detector_arguments_before_capture(tmp_path, argument) -> None:
    gateway, camera, evidence, application, recognition = _gateway(tmp_path)
    with pytest.raises(ValueError, match="does not accept arguments"):
        gateway.dispatch("vision2d.code_routes", argument)
    assert camera.read_calls == 0
    assert application.robot.calls == []


@pytest.mark.parametrize(
    ("recognition_case", "expected_code"),
    [
        ("unknown", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("duplicate", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("undecoded", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("wrong_count", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("low_confidence", "CODE_ROUTE_RECOGNITION_INVALID"),
        ("outside_workspace", "CODE_ROUTE_PLAN_INVALID"),
        ("scene_mismatch", "CODE_ROUTE_CONFIG_INVALID"),
        ("calibration_mismatch", "CODE_ROUTE_SCENE_POSITION_MISMATCH"),
    ],
)
def test_gateway_plan_failures_never_move(tmp_path, monkeypatch, recognition_case, expected_code) -> None:
    gateway, camera, evidence, application, recognition = _gateway(
        tmp_path, recognition_case=recognition_case
    )
    monkeypatch.setattr(
        "vision_platform.student.experiment_gateway.recognize_codes",
        lambda image: recognition,
    )
    with pytest.raises(VisionPlatformError) as captured:
        gateway.dispatch("vision2d.code_routes", {})
    assert captured.value.code == expected_code
    assert application.robot.calls == []
    assert application.tool.calls == []
    assert gateway.route_guard.completion()["active"] is False
```

Register only `vision2d.code_routing` as a new capability. Reuse the existing `robot.home`, `robot.pose`, `robot.move_world`, and `tool.suction` names exactly across the definition, capability checker, and tests. Do not mock `build_code_route_plan`, evidence writing, or route-guard activation.

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_capabilities.py -k code_routing `
  tests/test_student_programs/test_v1_07_gateway.py
```

Expected RED: capability and command are not implemented.

Add this capability matrix to `tests/test_experiments/test_capabilities.py` before GREEN:

```python
def test_code_routing_requires_complete_sim_camera_robot_and_tool_contract() -> None:
    required = (
        "camera.rgb", "camera.profile", "lighting.profile", "vision2d.code_routing",
        "robot.home", "robot.pose", "robot.move_world", "tool.suction", "scene.probe",
    )
    ready = SimpleNamespace(
        config=SimpleNamespace(camera_backend="sim", robot_backend="sim"),
        sim=object(), camera=object(), robot=object(), tool=object(),
    )
    assert check_capabilities(ready, required).ready
    for field in ("camera", "robot", "tool", "sim"):
        broken = SimpleNamespace(**ready.__dict__)
        setattr(broken, field, None)
        assert "vision2d.code_routing" in check_capabilities(broken, required).missing
    replay = SimpleNamespace(**ready.__dict__)
    replay.config = SimpleNamespace(camera_backend="replay", robot_backend="sim")
    assert "vision2d.code_routing" in check_capabilities(replay, required).missing
```

Register the capability by adding it to `_KNOWN` and placing this branch before the generic profile branch in `check_capabilities`:

```diff
_CODE_ROUTING_DEPENDENCIES = frozenset(
    {
        "camera.rgb", "camera.profile", "lighting.profile",
        "robot.home", "robot.pose", "robot.move_world",
        "tool.suction", "scene.probe",
    }
)


elif capability == "vision2d.code_routing":
    missing_dependencies = _CODE_ROUTING_DEPENDENCIES - requested_set
    if missing_dependencies:
        missing.append(capability)
        reasons[capability] = (
            "代码路由必须同时声明：" + ", ".join(sorted(missing_dependencies))
        )
    elif str(getattr(application.config, "camera_backend", "")) != "sim":
        missing.append(capability)
        reasons[capability] = "代码路由仅支持 CoppeliaSim 相机"
    elif str(getattr(application.config, "robot_backend", "")) != "sim":
        missing.append(capability)
        reasons[capability] = "代码路由仅支持 CoppeliaSim 机器人"
    elif any(
        getattr(application, name, None) is None
        for name in ("sim", "camera", "robot", "tool")
    ):
        missing.append(capability)
        reasons[capability] = "代码路由缺少场景、相机、机器人或吸盘"
    else:
        available.append(capability)
```

The snippet is an insertion into the existing `if/elif` chain, so retain its chain indentation and add `"vision2d.code_routing"` to `_KNOWN` rather than creating a second checker.

The existing profile controller must also support the new formal sensor without weakening its binding. Add focused tests in `tests/test_vision_quality/test_controller.py` that accept a CoppeliaSim camera only when `camera.sensor_path == catalog.sensor_path`, reject a mismatch, and preserve all legacy `/VisionQualityLab` cases. In `controller_for_experiment`, require the formal experiment's published `public_parameters.camera_path` to equal the hash-bound catalog sensor path before constructing the controller:

```python
published_camera_path = definition.public_parameters.get("camera_path")
if type(published_camera_path) is not str or published_camera_path != catalog.sensor_path:
    raise _profile_context_required()
```

Then replace the legacy constant check inside `_validate_backend` with the catalog-bound equality and remove `_SENSOR_PATH`:

```diff
-or catalog.sensor_path != _SENSOR_PATH
-or camera.sensor_path != _SENSOR_PATH
+or camera.sensor_path != catalog.sensor_path
```

This is a path-binding generalization, not permission for arbitrary sensors: the experiment definition, adjacent `profiles.json`, scene manifest hash, catalog sensor path, and live camera path must all agree. Run `tests/test_vision_quality/test_controller.py` together with the gateway tests before GREEN.

- [ ] **Step 2: Write route-guard RED tests before changing runner behavior**

Model these allowed per-entry states: `READY`, `AT_PICK_HOVER`, `AT_PICK`, `ATTACHED`, `LIFTED`, `AT_DROP_HOVER`, `AT_DROP`, `RELEASED`, `COMPLETE`. Permit the student to choose any incomplete entry while `READY`, but require every requested target to match the selected plan entry exactly and every observed current pose to be within 1.0 mm. Require `tool.on` only at the selected pick pose and `tool.off` only at its drop pose or cleanup. Reject movement before a plan, cross-entry mixing, altered coordinates, unsafe Z, repeated completion, tool misuse, and commands after guard failure.

Create these route-guard tests using the Task 2 plan fixture:

```python
import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.code_routing import ApprovedCodeRoute, CodeRoutePlan
from vision_platform.student.code_route_guard import CodeRouteGuard


@pytest.fixture
def plan() -> CodeRoutePlan:
    entries = tuple(
        ApprovedCodeRoute(
            entry_id=f"entry_{index}",
            part_id=f"part_{index}",
            code_type="qr" if index < 2 else "ean13",
            payload=f"payload-{index}",
            route_id="route_red" if index % 2 == 0 else "route_blue",
            pick_xyz_mm=(45.0 + 20.0 * index, -45.0 + 15.0 * index, 18.0),
            drop_xyz_mm=(116.0 + 4.0 * index, -60.0 if index % 2 == 0 else 60.0, 22.0),
            confidence=0.98,
        )
        for index in range(4)
    )
    return CodeRoutePlan("a" * 64, entries, 110.0, 15.0)


def _execute_one(guard: CodeRouteGuard, entry) -> None:
    pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], 111.0)
    drop_hover = (entry.drop_xyz_mm[0], entry.drop_xyz_mm[1], 111.0)
    guard.validate_move((20.0, 0.0, 111.0), pick_hover)
    guard.validate_move(pick_hover, entry.pick_xyz_mm)
    guard.validate_tool_on(entry.pick_xyz_mm)
    guard.validate_move(entry.pick_xyz_mm, pick_hover)
    guard.validate_move(pick_hover, drop_hover)
    guard.validate_move(drop_hover, entry.drop_xyz_mm)
    guard.validate_tool_off(entry.drop_xyz_mm)
    guard.validate_move(entry.drop_xyz_mm, drop_hover)


def test_guard_allows_student_selected_order_but_only_complete_trajectories(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    for entry in reversed(plan.entries):
        _execute_one(guard, entry)
    outcome = guard.completion()
    assert outcome["completed_entry_ids"] == sorted(entry.entry_id for entry in plan.entries)
    assert outcome["all_complete"] is True
    guard.validate_home()


def test_guard_rejects_motion_before_plan() -> None:
    guard = CodeRouteGuard()
    with pytest.raises(VisionPlatformError) as captured:
        guard.validate_move((20.0, 0.0, 111.0), (45.0, -45.0, 111.0))
    assert captured.value.code == "CODE_ROUTE_GUARD_INACTIVE"


def test_guard_latches_altered_coordinate_and_stops_future_commands(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    entry = plan.entries[0]
    with pytest.raises(VisionPlatformError) as captured:
        guard.validate_move((20.0, 0.0, 111.0), (entry.pick_xyz_mm[0] + 1.0, entry.pick_xyz_mm[1], 111.0))
    assert captured.value.code == "CODE_ROUTE_SEQUENCE_INVALID"
    with pytest.raises(VisionPlatformError, match="latched"):
        guard.validate_move((20.0, 0.0, 111.0), (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], 111.0))


def test_guard_requires_safe_height_before_lateral_pick_motion(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    entry = plan.entries[0]
    pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], plan.safe_z_mm + 1.0)
    with pytest.raises(VisionPlatformError) as captured:
        guard.validate_move((20.0, 0.0, 50.0), pick_hover)
    assert captured.value.code == "CODE_ROUTE_SEQUENCE_INVALID"


def test_tool_off_is_always_permitted_but_aborts_wrong_state(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    entry = plan.entries[0]
    pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], plan.safe_z_mm + 1.0)
    guard.validate_move((20.0, 0.0, plan.safe_z_mm + 1.0), pick_hover)
    guard.validate_tool_off(pick_hover)
    assert guard.completion()["failed"] is True


def test_host_reset_clears_the_plan_and_allows_a_fresh_activation(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    guard.reset()
    assert guard.completion() == {
        "active": False, "state": "INACTIVE", "selected_entry_id": None,
        "completed_entry_ids": [], "all_complete": False, "failed": False,
    }
    guard.activate(plan)
    assert guard.completion()["active"] is True
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_v1_07_route_guard.py
```

Expected RED: missing `CodeRouteGuard`.

- [ ] **Step 3: Implement the guard as a pure state machine**

Use explicit host callbacks; do not let the child mutate guard state:

```python
class CodeRouteGuard:
    def __init__(self) -> None:
        self._plan: CodeRoutePlan | None = None
        self._state = "INACTIVE"
        self._selected: ApprovedCodeRoute | None = None
        self._completed: set[str] = set()
        self._failed = False

    def activate(self, plan: CodeRoutePlan) -> None:
        if self._plan is not None or not isinstance(plan, CodeRoutePlan):
            raise VisionPlatformError(
                "CODE_ROUTE_PLAN_ALREADY_ACTIVE", "路线计划已经激活或无效"
            )
        self._plan = plan
        self._state = "READY"

    def reset(self) -> None:
        self._plan = None
        self._state = "INACTIVE"
        self._selected = None
        self._completed.clear()
        self._failed = False

    @staticmethod
    def _near(left, right, tolerance: float) -> bool:
        try:
            left_values = tuple(float(value) for value in left)
            right_values = tuple(float(value) for value in right)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            len(left_values) == 3
            and len(right_values) == 3
            and all(math.isfinite(value) for value in left_values + right_values)
            and all(abs(a - b) <= tolerance for a, b in zip(left_values, right_values))
        )

    def _reject(self, message: str) -> None:
        self._failed = True
        raise VisionPlatformError("CODE_ROUTE_SEQUENCE_INVALID", message)

    def _required(self) -> CodeRoutePlan:
        if self._plan is None:
            raise VisionPlatformError("CODE_ROUTE_GUARD_INACTIVE", "路线计划尚未激活")
        if self._failed:
            raise VisionPlatformError("CODE_ROUTE_SEQUENCE_INVALID", "route guard failure is latched")
        return self._plan

    def validate_move(
        self,
        current_xyz_mm: tuple[float, float, float],
        target_xyz_mm: tuple[float, float, float],
    ) -> None:
        plan = self._required()
        if not self._near(current_xyz_mm, current_xyz_mm, 0.0) or not self._near(target_xyz_mm, target_xyz_mm, 0.0):
            self._reject("坐标必须是三个有限值")
        if self._state == "READY":
            safe_lift = (current_xyz_mm[0], current_xyz_mm[1], plan.safe_z_mm + 1.0)
            if self._near(target_xyz_mm, safe_lift, 0.25):
                return
            for entry in plan.entries:
                pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], plan.safe_z_mm + 1.0)
                if entry.entry_id not in self._completed and self._near(target_xyz_mm, pick_hover, 0.25):
                    current_hover = (
                        current_xyz_mm[0], current_xyz_mm[1], plan.safe_z_mm + 1.0
                    )
                    if not self._near(current_xyz_mm, current_hover, 1.0):
                        self._reject("横向移动到抓取悬停点前必须先到安全高度")
                    self._selected = entry
                    self._state = "AT_PICK_HOVER"
                    return
            self._reject("READY 只允许安全抬升或未完成构件的抓取悬停点")
        entry = self._selected
        if entry is None:
            self._reject("没有选中的路线条目")
        pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], plan.safe_z_mm + 1.0)
        drop_hover = (entry.drop_xyz_mm[0], entry.drop_xyz_mm[1], plan.safe_z_mm + 1.0)
        transitions = {
            "AT_PICK_HOVER": (pick_hover, entry.pick_xyz_mm, "AT_PICK"),
            "ATTACHED": (entry.pick_xyz_mm, pick_hover, "LIFTED"),
            "LIFTED": (pick_hover, drop_hover, "AT_DROP_HOVER"),
            "AT_DROP_HOVER": (drop_hover, entry.drop_xyz_mm, "AT_DROP"),
            "RELEASED": (entry.drop_xyz_mm, drop_hover, "COMPLETE"),
        }
        if self._state not in transitions:
            self._reject(f"状态 {self._state} 不允许移动")
        expected_current, expected_target, next_state = transitions[self._state]
        if not self._near(current_xyz_mm, expected_current, 1.0) or not self._near(target_xyz_mm, expected_target, 0.25):
            self._reject(f"状态 {self._state} 的当前位置或目标不匹配")
        if next_state == "COMPLETE":
            self._completed.add(entry.entry_id)
            self._selected = None
            self._state = "READY"
        else:
            self._state = next_state

    def validate_tool_on(self, pose_xyz_mm: tuple[float, float, float]) -> None:
        self._required()
        if self._state != "AT_PICK" or self._selected is None or not self._near(pose_xyz_mm, self._selected.pick_xyz_mm, 1.0):
            self._reject("吸盘只能在当前条目的抓取点开启")
        self._state = "ATTACHED"

    def validate_tool_off(self, pose_xyz_mm: tuple[float, float, float]) -> bool:
        if self._plan is None:
            return True
        if self._state == "READY" and len(self._completed) == len(self._plan.entries):
            return True
        if self._state == "AT_DROP" and self._selected is not None and self._near(pose_xyz_mm, self._selected.drop_xyz_mm, 1.0):
            self._state = "RELEASED"
            return True
        self._failed = True
        return False

    def validate_home(self) -> None:
        plan = self._required()
        if self._state != "READY" or len(self._completed) != len(plan.entries):
            self._reject("全部路线完成前禁止学生程序回零")

    def completion(self) -> dict[str, Any]:
        count = 0 if self._plan is None else len(self._plan.entries)
        return {
            "active": self._plan is not None,
            "state": self._state,
            "selected_entry_id": None if self._selected is None else self._selected.entry_id,
            "completed_entry_ids": sorted(self._completed),
            "all_complete": count > 0 and len(self._completed) == count and not self._failed,
            "failed": self._failed,
        }
```

Import `math`, `Any`, `VisionPlatformError`, `ApprovedCodeRoute`, and `CodeRoutePlan`. The target whitelist consists only of the active entry's pick/drop poses, the existing helper's hover height `safe_z_mm + 1.0`, an initial lift at the current X/Y to that hover height, and `robot.home` after all entries complete. Requested targets use a 0.25 mm numerical tolerance and observed poses use 1.0 mm. Generic workspace and speed validation still runs in addition to this guard. Runner cleanup continues to call the physical adapter directly so emergency `tool.off`, safe lift, and home cannot be blocked by a latched student-route error.

- [ ] **Step 4: Implement host recognition, plan activation, and evidence**

Extend `StudentExperimentGateway.dispatch` with exactly:

```python
_CODE_ROUTE_CAPABILITIES = frozenset(
    {
        "camera.rgb", "camera.profile", "lighting.profile",
        "vision2d.code_routing", "robot.home", "robot.pose",
        "robot.move_world", "tool.suction", "scene.probe",
    }
)


if name == "vision2d.code_routes":
    self._require_exact_args(name, args, ())
    return self._code_routes()
```

Initialize `self.route_guard = CodeRouteGuard()` beside the existing snapshot/probe state in `StudentExperimentGateway.__init__`. Add these complete helpers and method:

```python
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotate_code_routes(
    image_bgr: np.ndarray,
    recognition: CodeRecognitionResult,
    plan: CodeRoutePlan,
) -> np.ndarray:
    annotated = image_bgr.copy()
    entry_by_payload = {entry.payload: entry for entry in plan.entries}
    for reading in recognition.readings:
        entry = entry_by_payload[reading.data]
        points = np.rint(np.asarray(reading.polygon_px, dtype=np.float64)).astype(np.int32)
        cv2.polylines(annotated, [points], True, (0, 180, 0), 2, cv2.LINE_AA)
        origin = (max(0, int(reading.bbox_px[0])), max(16, int(reading.bbox_px[1]) - 4))
        cv2.putText(
            annotated, f"{entry.part_id}->{entry.route_id}", origin,
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1, cv2.LINE_AA,
        )
    return annotated


def _validated_code_asset_binding(
    self,
) -> dict[str, tuple[float, float, float]]:
    binding = self._scene_manifest.get("code_assets_manifest")
    route_binding = self._scene_manifest.get("code_routing")
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256"}
        or type(binding.get("path")) is not str
        or type(binding.get("sha256")) is not str
        or len(binding["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in binding["sha256"])
        or not isinstance(route_binding, Mapping)
        or set(route_binding) != {
            "part_ids", "initial_positions_mm",
            "calibration_plane_z_mm", "calibration_matrix", "route_slots_mm",
        }
    ):
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面或构件绑定缺失")
    relative = Path(binding["path"])
    root = self._definition.scene_manifest.parent.resolve()
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单路径不安全")
    candidate = root / relative
    if candidate.is_symlink():
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单不得使用链接")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != root or _sha256_file(resolved) != binding["sha256"]:
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单哈希或路径不匹配")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "entries"} or payload["schema_version"] != 1:
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单格式无效")
    route_config = self._definition.public_parameters.get("code_routing")
    if not isinstance(route_config, Mapping):
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "路线配置无效")
    routes = route_config.get("routes")
    if not isinstance(payload["entries"], list) or not isinstance(routes, (tuple, list)):
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面条目无效")
    expected = {
        (route["part_id"], route["code_type"], route["payload"])
        for route in routes
    }
    entries = payload["entries"]
    required_entry_fields = {"part_id", "code_type", "payload"}
    if (
        len(entries) != len(routes)
        or any(
            not isinstance(entry, Mapping)
            or not required_entry_fields <= set(entry)
            or any(type(entry[field]) is not str for field in required_entry_fields)
            for entry in entries
        )
    ):
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面条目无效")
    actual = {
        (entry["part_id"], entry["code_type"], entry["payload"])
        for entry in entries
    }
    part_ids = route_binding["part_ids"]
    initial_positions = route_binding["initial_positions_mm"]

    def matrix(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "标定矩阵无效")
        rows = []
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "标定矩阵无效")
            if any(
                type(component) not in {int, float}
                or not math.isfinite(float(component))
                for component in row
            ):
                raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "标定矩阵无效")
            rows.append(tuple(float(component) for component in row))
        return (rows[0], rows[1])

    route_slots = route_binding["route_slots_mm"]
    if not isinstance(route_slots, Mapping) or any(
        type(route_id) is not str or not isinstance(slots, (list, tuple))
        for route_id, slots in route_slots.items()
    ):
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "仓位绑定无效")
    if sum(len(slots) for slots in route_slots.values()) != len(routes) or any(
        not isinstance(slot, (list, tuple))
        or len(slot) != 3
        or any(
            type(component) not in {int, float}
            or not math.isfinite(float(component))
            for component in slot
        )
        for slots in route_slots.values()
        for slot in slots
    ):
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "仓位绑定无效")
    manifest_slots = {
        (route_id, tuple(float(component) for component in slot))
        for route_id, slots in route_slots.items()
        for slot in slots
        if isinstance(slot, (list, tuple))
        and len(slot) == 3
        and all(
            type(component) in {int, float}
            and math.isfinite(float(component))
            for component in slot
        )
    }
    configured_slots = {
        (route["route_id"], tuple(float(component) for component in route["drop_xyz_mm"]))
        for route in routes
        if isinstance(route, Mapping)
        and type(route.get("route_id")) is str
        and isinstance(route.get("drop_xyz_mm"), (list, tuple))
        and len(route["drop_xyz_mm"]) == 3
        and all(
            type(component) in {int, float}
            and math.isfinite(float(component))
            for component in route["drop_xyz_mm"]
        )
    }

    if (
        actual != expected
        or type(part_ids) is not list
        or len(part_ids) != len(expected)
        or any(type(part_id) is not str for part_id in part_ids)
        or set(part_ids) != {item[0] for item in expected}
        or not isinstance(initial_positions, Mapping)
        or set(initial_positions) != set(part_ids)
        or any(
            not isinstance(position, (list, tuple))
            or len(position) != 3
            or any(
                type(component) not in {int, float}
                or not math.isfinite(float(component))
                for component in position
            )
            for position in initial_positions.values()
        )
        or type(route_binding["calibration_plane_z_mm"]) not in {int, float}
        or not math.isfinite(float(route_binding["calibration_plane_z_mm"]))
        or float(route_binding["calibration_plane_z_mm"]) != 27.4
        or matrix(route_binding["calibration_matrix"])
        != matrix(route_config.get("calibration_matrix"))
        or len(manifest_slots) != len(routes)
        or configured_slots != manifest_slots
    ):
        raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面、路线和场景构件不一致")
    return {
        part_id: tuple(float(component) for component in initial_positions[part_id])
        for part_id in part_ids
    }


def _code_routes(self) -> dict[str, Any]:
    if self.route_guard.completion()["active"]:
        raise VisionPlatformError("CODE_ROUTE_PLAN_ALREADY_ACTIVE", "本次运行已有路线计划")
    if not _CODE_ROUTE_CAPABILITIES <= set(self._definition.capabilities):
        raise VisionPlatformError("CODE_ROUTE_CONTEXT_REQUIRED", "当前实验没有代码路由能力")
    scene_part_positions = self._validated_code_asset_binding()
    profile = self._profile_public(
        self._profiles_required().current(), path="vision2d.code_routes.profile"
    )
    if profile.get("profile_id") != "standard" or profile.get("resolution") != [1024, 1024]:
        raise VisionPlatformError("CODE_ROUTE_PROFILE_MISMATCH", "代码路由必须使用 standard 1024x1024")
    recorded = self._capture_raw(profile)
    recognition = recognize_codes(recorded.image_bgr, max_codes=4)
    try:
        plan = build_code_route_plan(
            recognition,
            routing_config=self._definition.public_parameters["code_routing"],
            workspace=self._definition.workspace,
            scene_part_ids=set(scene_part_positions),
            image_size=(recorded.image_bgr.shape[1], recorded.image_bgr.shape[0]),
        )
    except CodeRouteError as error:
        raise VisionPlatformError(error.code, str(error)) from error
    for entry in plan.entries:
        expected = scene_part_positions[entry.part_id]
        if (
            math.hypot(
                entry.pick_xyz_mm[0] - expected[0],
                entry.pick_xyz_mm[1] - expected[1],
            ) > 3.0
            or abs(entry.pick_xyz_mm[2] - expected[2]) > 0.25
        ):
            raise VisionPlatformError(
                "CODE_ROUTE_SCENE_POSITION_MISMATCH",
                f"{entry.part_id} 的视觉坐标与绑定初态不一致",
            )
    annotated = _annotate_code_routes(recorded.image_bgr, recognition, plan)
    result = code_route_plan_to_dict(plan)
    snapshot_id = recorded.value["snapshot_id"]
    bundle = VisionResultBundle(
        schema_version=1,
        bundle_id=f"{self.context.experiment_id}-{snapshot_id}",
        experiment_id=self.context.experiment_id,
        source_snapshot_id=snapshot_id,
        status="PASS",
        layers=(
            VisionImageLayer("raw", "原图", recorded.image_bgr),
            VisionImageLayer("annotated", "代码与仓位标注", annotated),
        ),
        result=result,
        profile={**profile, "code_route_plan_id": plan.plan_id},
        hardware_status="PENDING_HARDWARE",
    )
    bundle_path = record_vision_bundle(
        self.evidence, bundle, existing_layer_records={"raw": recorded.record}
    )
    self.route_guard.activate(plan)
    public = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "vision_bundle_path": bundle_path,
        **result,
    }
    copied = _copy_json_native(public, path="vision2d.code_routes")
    assert isinstance(copied, dict)
    return copied
```

Import `hashlib`, `json`, `math`, `Path`, OpenCV, NumPy, the route models/functions/error, recognition models/function, and `CodeRouteGuard`. Activation deliberately occurs only after manifest validation, recognition, full-plan validation, the 3 mm scene-initial-position cross-check, annotation, and evidence persistence all succeed. A second call fails before capture.

- [ ] **Step 5: Enforce the guard in runner motion and tool dispatch**

Before physical action, call the route guard for V1-07; on any violation, keep the primary error, stop future student commands, turn the tool off during cleanup, and record the guard state. Existing experiments without `vision2d.code_routing` must remain byte-for-byte compatible at the protocol boundary.

Add these gateway pass-through methods:

```python
def route_validate_move(self, current, target) -> None:
    if "vision2d.code_routing" in self._definition.capabilities:
        self.route_guard.validate_move(current, target)


def route_validate_tool_on(self, pose) -> None:
    if "vision2d.code_routing" in self._definition.capabilities:
        self.route_guard.validate_tool_on(pose)


def route_note_tool_off(self, pose) -> bool:
    return (
        True
        if "vision2d.code_routing" not in self._definition.capabilities
        else self.route_guard.validate_tool_off(pose)
    )


def route_validate_home(self) -> None:
    if "vision2d.code_routing" in self._definition.capabilities:
        self.route_guard.validate_home()
```

In `StudentExperimentGateway.reset_environment`, call the existing profile reset first and then `self.route_guard.reset()` only as part of the host-owned reset path; Task 10 will also clear `_route_evidence` there. Add runner tests proving pause/continue do not mutate guard state, single-step advances exactly one accepted command, stop latches no later student command, and reset performs safe tool-off/scene recovery before allowing a fresh plan. Do not expose `reset()` to the student SDK.

Then make these exact runner insertions:

```python
# In the experiment-gateway command set:
"vision2d.code_routes",

# In _command_move_world, after generic validation and before physical movement:
if self._experiment_gateway is not None:
    self._experiment_gateway.route_validate_move(current, target)

# In _command_tool_on, after generic height validation and before physical tool.on:
if self._experiment_gateway is not None:
    self._experiment_gateway.route_validate_tool_on(pose)

# In _command_tool_off, physically release first, then preserve an abort result:
self._application.tool.off()
if (
    self._experiment_gateway is not None
    and not self._experiment_gateway.route_note_tool_off(self._read_pose())
):
    raise VisionPlatformError(
        "CODE_ROUTE_SEQUENCE_INVALID", "吸盘已安全关闭，但路线顺序已中止"
    )

# In _command_home, before the student-initiated physical home call:
if self._experiment_gateway is not None:
    self._experiment_gateway.route_validate_home()
```

Do not route runner cleanup through `route_validate_home`; cleanup must still release the tool and attempt safe recovery after a latched route error. Add runner tests that execute the exact happy command sequence, reject a 1 mm altered X target, allow safety `tool.off` before surfacing the route error, and preserve an injected physical-move exception as the primary error with route/cleanup details attached.

- [ ] **Step 6: Run focused GREEN and runner regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_capabilities.py `
  tests/test_student_programs/test_v1_07_gateway.py `
  tests/test_student_programs/test_v1_07_route_guard.py `
  tests/test_student_programs/test_experiment_gateway.py `
  tests/test_student_programs/test_runner.py `
  tests/test_student_programs/test_student_safety.py `
  tests/test_student_programs/test_experiment_evidence.py `
  tests/test_vision_quality/test_controller.py
git diff --check
```

- [ ] **Step 7: Commit the controlled host integration**

```powershell
git add -- vision_platform/student vision_platform/experiments/capabilities.py `
  vision_platform/vision_quality/controller.py `
  tests/test_student_programs tests/test_experiments/test_capabilities.py `
  tests/test_vision_quality/test_controller.py `
  RETAINED_FILES.txt
git commit -m "feat(student): guard V1-07 route execution"
```

## Task 5: Generate and bind original QR/EAN-13 assets

**Files:**

- Create: `tools/vision_lab/generate_v1_07_code_assets.py`
- Create: `simulation/vision_code_routing_lab/code_assets/qr_v1_07_a.png`
- Create: `simulation/vision_code_routing_lab/code_assets/qr_v1_07_b.png`
- Create: `simulation/vision_code_routing_lab/code_assets/ean_6901234567892.png`
- Create: `simulation/vision_code_routing_lab/code_assets/ean_6901234567809.png`
- Create: `simulation/vision_code_routing_lab/code_assets_manifest.json`
- Create: `tests/test_simulation/test_v1_07_code_assets.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write asset RED tests**

Require four exact manifest entries, normalized project-relative paths, lowercase SHA-256, dimensions/channels, payload and code type, two QR payloads `V1-07-A` and `V1-07-B`, two valid EAN-13 values `6901234567892` and `6901234567809`, no external URL/license ambiguity, deterministic regeneration, and successful decoding by the integrated production recognizer.

Create this concrete asset test:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

from tools.vision_lab.generate_v1_07_code_assets import generate
from vision_platform.vision2d.code_recognition import recognize_codes


EXPECTED = {
    ("part_a", "qr", "V1-07-A"),
    ("part_b", "qr", "V1-07-B"),
    ("part_c", "ean13", "6901234567892"),
    ("part_d", "ean13", "6901234567809"),
}


def test_generated_assets_are_bound_and_decoded_by_production(tmp_path: Path) -> None:
    manifest_path = generate(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "entries"}
    assert payload["schema_version"] == 1
    assert {
        (entry["part_id"], entry["code_type"], entry["payload"])
        for entry in payload["entries"]
    } == EXPECTED
    for entry in payload["entries"]:
        assert set(entry) == {
            "asset_id", "part_id", "code_type", "payload", "path", "sha256",
            "size_px", "channels", "generator",
        }
        path = tmp_path / Path(entry["path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        assert [image.shape[1], image.shape[0]] == entry["size_px"]
        assert entry["channels"] == 3
        result = recognize_codes(image, max_codes=1)
        assert result.status == "PASS"
        assert len(result.readings) == 1
        assert result.readings[0].code_type == entry["code_type"]
        assert result.readings[0].data == entry["payload"]


def test_check_mode_detects_no_drift(tmp_path: Path) -> None:
    generate(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    generate(tmp_path, check=True)
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_simulation/test_v1_07_code_assets.py
```

Expected RED: generator and manifest are absent.

- [ ] **Step 2: Implement deterministic generation**

Use `cv2.QRCodeEncoder_create()` for QR and the integrated `encode_ean13_payload()` helper for EAN-13. Use nearest-neighbor scaling, fixed quiet zones, BGR `uint8`, no timestamps, no random values, and canonical JSON with UTF-8 plus a final newline. The generator must support `--check` without writing.

Implement the generator with this exact data table and byte-comparison behavior:

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from vision_platform.vision2d.code_recognition import encode_ean13_payload


ASSETS = (
    ("qr_v1_07_a", "part_a", "qr", "V1-07-A", "qr_v1_07_a.png"),
    ("qr_v1_07_b", "part_b", "qr", "V1-07-B", "qr_v1_07_b.png"),
    ("ean_6901234567892", "part_c", "ean13", "6901234567892", "ean_6901234567892.png"),
    ("ean_6901234567809", "part_d", "ean13", "6901234567809", "ean_6901234567809.png"),
)


def _qr(payload: str):
    encoded = cv2.QRCodeEncoder_create().encode(payload)
    scaled = cv2.resize(encoded, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    bordered = cv2.copyMakeBorder(
        scaled, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255
    )
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)


def _png_bytes(image) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not ok:
        raise RuntimeError("V1-07 PNG encoding failed")
    return bytes(encoded)


def generate(
    output_dir: Path,
    *,
    manifest_path: Path | None = None,
    check: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    manifest_path = (
        output_dir / "code_assets_manifest.json"
        if manifest_path is None
        else Path(manifest_path)
    )
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    expected_files: dict[str, bytes] = {}
    for asset_id, part_id, code_type, payload, filename in ASSETS:
        image = _qr(payload) if code_type == "qr" else encode_ean13_payload(payload)
        png = _png_bytes(image)
        expected_files[filename] = png
        entries.append(
            {
                "asset_id": asset_id,
                "part_id": part_id,
                "code_type": code_type,
                "payload": payload,
                "path": f"simulation/vision_code_routing_lab/code_assets/{filename}",
                "sha256": hashlib.sha256(png).hexdigest(),
                "size_px": [int(image.shape[1]), int(image.shape[0])],
                "channels": 3,
                "generator": "robot-sim-opencv-v1",
            }
        )
    manifest_bytes = (
        json.dumps(
            {"schema_version": 1, "entries": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    for filename, expected in expected_files.items():
        path = output_dir / filename
        if check:
            if not path.is_file() or path.read_bytes() != expected:
                raise RuntimeError(f"V1-07 asset drift: {filename}")
        else:
            path.write_bytes(expected)
    if check:
        if not manifest_path.is_file() or manifest_path.read_bytes() != manifest_bytes:
            raise RuntimeError("V1-07 asset drift: code_assets_manifest.json")
    else:
        manifest_path.write_bytes(manifest_bytes)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / "simulation" / "vision_code_routing_lab" / "code_assets"
    generate(
        output,
        manifest_path=output.parent / "code_assets_manifest.json",
        check=arguments.check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The standalone generator writes the manifest at `simulation/vision_code_routing_lab/code_assets_manifest.json`; the test helper keeps it inside `tmp_path`. `--check` compares all four PNGs and the parent manifest without writing, moving, or deleting files.

- [ ] **Step 3: Generate, check, and decode every asset**

```powershell
.\.venv-vision\Scripts\python.exe `
  tools/vision_lab/generate_v1_07_code_assets.py
.\.venv-vision\Scripts\python.exe `
  tools/vision_lab/generate_v1_07_code_assets.py --check
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_simulation/test_v1_07_code_assets.py `
  tests/test_vision2d/test_code_recognition.py
```

Expected: `--check` reports all assets current and all tests pass.

- [ ] **Step 4: Commit source, generated assets, manifest, and tests together**

```powershell
git add -- tools/vision_lab/generate_v1_07_code_assets.py `
  simulation/vision_code_routing_lab/code_assets `
  simulation/vision_code_routing_lab/code_assets_manifest.json `
  tests/test_simulation/test_v1_07_code_assets.py RETAINED_FILES.txt
git commit -m "feat(simulation): add original V1-07 code assets"
```

## Task 6: Build the independent V1-07 CoppeliaSim scene

**Files:**

- Create: `tools/vision_lab/build_v1_07_scene.py`
- Create: `simulation/vision_code_routing_lab/scene_spec.json`
- Create: `simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt`
- Create: `simulation/vision_code_routing_lab/scene_manifest.json`
- Create: `simulation/vision_code_routing_lab/profiles.json`
- Modify: `simulation/training_scenes/build_scene.py`
- Create: `tests/test_simulation/test_v1_07_scene_contract.py`
- Modify: `tests/test_simulation/test_formal_training_scenes.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write static scene-contract RED tests**

Require the new scene and manifest only under `simulation/vision_code_routing_lab/`; bind the exact scene hash, profile catalog, and code-assets manifest hash; declare four unique graspable part paths, four code-face paths, camera, robot, suction, two bin paths, four unique drop-slot anchors, host-owned reset contract, initial/final probes, workspace, safe Z, and deterministic code-to-part-to-route bindings. Assert existing formal scene hashes and protected robot assets remain unchanged from the Task 1 baseline.

Create the scene specification exactly as follows; this is also the expected value table for `test_v1_07_scene_contract.py`:

```json
{
  "schema_version": 1,
  "scene_id": "vision-code-routing-lab",
  "template": "simulation/vision_lab/BL23_vision_lab.ttt",
  "output": "simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt",
  "remove_paths": ["/VisionLab"],
  "root_path": "/VisionCodeRoutingLab",
  "code_assets_manifest": "simulation/vision_code_routing_lab/code_assets_manifest.json",
  "profiles": "simulation/vision_code_routing_lab/profiles.json",
  "workspace": {"alias": "Workspace", "center_mm": [80, 0, 5], "size_mm": [140, 190, 10]},
  "camera": {"alias": "Camera", "rig_position_m": [0.085, 0.0, 0.5], "orientation_deg": [180, 0, 0], "code_face_plane_z_mm": 27.4},
  "safe_z_mm": 110,
  "parts": [
    {"alias": "part_a", "code_asset_id": "qr_v1_07_a", "position_mm": [40, -45, 18], "size_mm": [34, 34, 16]},
    {"alias": "part_b", "code_asset_id": "qr_v1_07_b", "position_mm": [80, -45, 18], "size_mm": [34, 34, 16]},
    {"alias": "part_c", "code_asset_id": "ean_6901234567892", "position_mm": [40, 5, 18], "size_mm": [38, 28, 16]},
    {"alias": "part_d", "code_asset_id": "ean_6901234567809", "position_mm": [80, 5, 18], "size_mm": [38, 28, 16]}
  ],
  "bins": [
    {"alias": "route_red", "color_rgb": [0.8, 0.15, 0.15], "slots": [{"alias": "red_1", "position_mm": [116, -60, 22]}, {"alias": "red_2", "position_mm": [128, -60, 22]}]},
    {"alias": "route_blue", "color_rgb": [0.15, 0.3, 0.85], "slots": [{"alias": "blue_1", "position_mm": [116, 60, 22]}, {"alias": "blue_2", "position_mm": [128, 60, 22]}]}
  ],
  "reset_contract": {"strategy": "scene_reload", "tool_off": true, "robot_home": true},
  "required_paths": [
    "/BLX_base_link", "/BLX_tool_suction", "/VisionCodeRoutingLab", "/VisionCodeRoutingLab/Workspace",
    "/VisionCodeRoutingLab/CameraRig/Camera", "/VisionCodeRoutingLab/Lighting", "/VisionCodeRoutingLab/Lighting/KeyLight", "/VisionCodeRoutingLab/Lighting/FillLight", "/VisionCodeRoutingLab/Parts",
    "/VisionCodeRoutingLab/Parts/part_a", "/VisionCodeRoutingLab/Parts/part_a/CodeFace",
    "/VisionCodeRoutingLab/Parts/part_b", "/VisionCodeRoutingLab/Parts/part_b/CodeFace",
    "/VisionCodeRoutingLab/Parts/part_c", "/VisionCodeRoutingLab/Parts/part_c/CodeFace",
    "/VisionCodeRoutingLab/Parts/part_d", "/VisionCodeRoutingLab/Parts/part_d/CodeFace",
    "/VisionCodeRoutingLab/Bins/route_red", "/VisionCodeRoutingLab/Bins/route_red/red_1", "/VisionCodeRoutingLab/Bins/route_red/red_2",
    "/VisionCodeRoutingLab/Bins/route_blue", "/VisionCodeRoutingLab/Bins/route_blue/blue_1", "/VisionCodeRoutingLab/Bins/route_blue/blue_2"
  ]
}
```

Create the adjacent profile catalog exactly as follows. The narrow fixed field of view gives every EAN-13 module enough pixels for the integrated decoder while keeping the full task area visible:

```json
{
  "schema_version": 1,
  "baseline_profile_id": "standard",
  "sensor_path": "/VisionCodeRoutingLab/CameraRig/Camera",
  "camera_rig_path": "/VisionCodeRoutingLab/CameraRig",
  "key_light_path": "/VisionCodeRoutingLab/Lighting/KeyLight",
  "fill_light_path": "/VisionCodeRoutingLab/Lighting/FillLight",
  "near_clip_m": 0.05,
  "far_clip_m": 1.0,
  "profiles": [
    {
      "profile_id": "standard",
      "label": "代码路由标准视图",
      "resolution": [1024, 1024],
      "perspective_angle_deg": 20,
      "camera_rig_z_m": 0.5,
      "key_diffuse_rgb": [0.8, 0.8, 0.8],
      "fill_diffuse_rgb": [0.35, 0.35, 0.35]
    }
  ]
}
```

The platform's existing reset operation reloads the scene, turns the tool off, and homes the robot; the `reset_contract` above binds that existing host-controlled behavior rather than adding an independently callable student/Lua reset command.

Create this static contract test:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "simulation" / "vision_code_routing_lab"


def test_v1_07_scene_release_is_self_consistent() -> None:
    spec = json.loads((LAB / "scene_spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((LAB / "scene_manifest.json").read_text(encoding="utf-8"))
    scene = LAB / "BL23_vision_code_routing_lab.ttt"
    assets = LAB / "code_assets_manifest.json"
    profiles = LAB / "profiles.json"
    assert spec["scene_id"] == manifest["scene_id"] == "vision-code-routing-lab"
    assert manifest["scene"]["path"] == scene.relative_to(ROOT).as_posix()
    assert manifest["scene"]["sha256"] == hashlib.sha256(scene.read_bytes()).hexdigest()
    assert manifest["code_assets_manifest"]["sha256"] == hashlib.sha256(assets.read_bytes()).hexdigest()
    assert manifest["profile_catalog"] == {
        "path": "simulation/vision_code_routing_lab/profiles.json",
        "sha256": hashlib.sha256(profiles.read_bytes()).hexdigest(),
    }
    assert manifest["required_paths"] == spec["required_paths"]
    assert manifest["protected_assets_unchanged"] is True
    assert manifest["reset_contract"] == {"strategy": "scene_reload", "tool_off": True, "robot_home": True}
    assert manifest["code_routing"]["calibration_plane_z_mm"] == 27.4
    assert manifest["code_routing"]["calibration_matrix"] == [
        [-0.1627580685211339, 0.0, 168.25075204856],
        [0.0, 0.1627580685211339, -83.25075204855999],
    ]
    assert manifest["code_routing"]["route_slots_mm"] == {
        "route_red": [[116, -60, 22], [128, -60, 22]],
        "route_blue": [[116, 60, 22], [128, 60, 22]],
    }


def test_scene_parts_bins_and_slots_are_unique_and_bounded() -> None:
    spec = json.loads((LAB / "scene_spec.json").read_text(encoding="utf-8"))
    part_aliases = [part["alias"] for part in spec["parts"]]
    asset_ids = [part["code_asset_id"] for part in spec["parts"]]
    slots = [tuple(slot["position_mm"]) for bin_spec in spec["bins"] for slot in bin_spec["slots"]]
    assert part_aliases == ["part_a", "part_b", "part_c", "part_d"]
    assert len(set(asset_ids)) == 4
    assert len(slots) == len(set(slots)) == 4
    assert all(20 <= x <= 140 and -90 <= y <= 90 and 10 <= z <= 140 for x, y, z in slots)
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_simulation/test_v1_07_scene_contract.py `
  tests/test_simulation/test_formal_training_scenes.py
```

Expected RED: new scene manifest is missing.

- [ ] **Step 2: Implement a reproducible scene builder**

Extend the existing atomic training-scene builder rather than introducing a second unsafe publish path. Add this formal registration beside the three existing scene registrations:

```python
_CODE_ROUTING_REQUIRED_PATHS = (
    "/BLX_base_link", "/BLX_tool_suction", "/VisionCodeRoutingLab",
    "/VisionCodeRoutingLab/Workspace", "/VisionCodeRoutingLab/CameraRig/Camera",
    "/VisionCodeRoutingLab/Lighting", "/VisionCodeRoutingLab/Lighting/KeyLight",
    "/VisionCodeRoutingLab/Lighting/FillLight",
    "/VisionCodeRoutingLab/Parts", "/VisionCodeRoutingLab/Parts/part_a",
    "/VisionCodeRoutingLab/Parts/part_a/CodeFace", "/VisionCodeRoutingLab/Parts/part_b",
    "/VisionCodeRoutingLab/Parts/part_b/CodeFace", "/VisionCodeRoutingLab/Parts/part_c",
    "/VisionCodeRoutingLab/Parts/part_c/CodeFace", "/VisionCodeRoutingLab/Parts/part_d",
    "/VisionCodeRoutingLab/Parts/part_d/CodeFace", "/VisionCodeRoutingLab/Bins/route_red",
    "/VisionCodeRoutingLab/Bins/route_red/red_1", "/VisionCodeRoutingLab/Bins/route_red/red_2",
    "/VisionCodeRoutingLab/Bins/route_blue", "/VisionCodeRoutingLab/Bins/route_blue/blue_1",
    "/VisionCodeRoutingLab/Bins/route_blue/blue_2",
)

_FORMAL_SCENES["simulation/vision_code_routing_lab/scene_spec.json"] = _FormalScene(
    spec_relative="simulation/vision_code_routing_lab/scene_spec.json",
    scene_id="vision-code-routing-lab",
    root_path="/VisionCodeRoutingLab",
    output_relative="simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt",
    required_paths=_CODE_ROUTING_REQUIRED_PATHS,
)
```

Add an exact-key validator branch for the JSON fields above. Reuse `_vector`, `_exact_keys`, `_project_file_from_relative`, and `_validate_spec`; require four parts, two bins, two slots per bin, all unique aliases/assets/slot positions, `safe_z_mm == 110`, the exact protected template/output/root/remove paths, and the exact required-path tuple.

Import `cv2`, NumPy, and `encode_ean13_payload`, then add these concrete builder helpers. They create a white plate and black, non-respondable top-face primitives from the manifest-bound payload; the final compound is movable and retains a `CodeFace` child path for probing.

```python
def _code_rectangles(code_type: str, payload: str) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    if code_type == "qr":
        encoded = cv2.QRCodeEncoder_create().encode(payload)
        gray = cv2.copyMakeBorder(
            encoded, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255
        )
    elif code_type == "ean13":
        gray = cv2.cvtColor(
            encode_ean13_payload(payload, module_px=3, bar_height_px=72),
            cv2.COLOR_BGR2GRAY,
        )
    else:
        raise ValueError(f"unsupported code type: {code_type}")
    mask = np.asarray(gray < 128, dtype=np.uint8)
    rectangles: list[tuple[int, int, int, int]] = []
    if code_type == "qr":
        for row, column in np.argwhere(mask):
            rectangles.append((int(column), int(row), 1, 1))
    else:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for component in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[component])
            if area > 0:
                rectangles.append((x, y, width, height))
    return gray, rectangles


def _code_part(sim: Any, part: dict[str, Any], asset: dict[str, Any], parent: int) -> int:
    center = [float(value) for value in part["position_mm"]]
    size = [float(value) for value in part["size_mm"]]
    pieces = [
        _shape(
            sim, name=f"{part['alias']}_body", shape="cuboid", size_mm=size,
            position_mm=center, color=[0.75, 0.75, 0.75], parent=parent,
            respondable=True,
        )
    ]
    face_width = size[0] - 4.0
    face_height = size[1] - 4.0
    face_z = center[2] + size[2] / 2.0 + 0.6
    pieces.append(
        _shape(
            sim, name=f"{part['alias']}_face_plate", shape="cuboid",
            size_mm=[face_width, face_height, 1.0],
            position_mm=[center[0], center[1], face_z],
            color=[0.98, 0.98, 0.98], parent=parent, respondable=False,
        )
    )
    gray, rectangles = _code_rectangles(asset["code_type"], asset["payload"])
    scale = min(face_width / float(gray.shape[1]), face_height / float(gray.shape[0]))
    origin_x = center[0] - gray.shape[1] * scale / 2.0
    origin_y = center[1] - gray.shape[0] * scale / 2.0
    for index, (x, y, width, height) in enumerate(rectangles):
        pieces.append(
            _shape(
                sim, name=f"{part['alias']}_code_{index:04d}", shape="cuboid",
                size_mm=[width * scale, height * scale, 0.3],
                position_mm=[
                    origin_x + (x + width / 2.0) * scale,
                    origin_y + (y + height / 2.0) * scale,
                    face_z + 0.65,
                ],
                color=[0.01, 0.01, 0.01], parent=parent, respondable=False,
            )
        )
    compound = int(sim.groupShapes(pieces, False))
    _alias(sim, compound, part["alias"])
    sim.setObjectParent(compound, parent, True)
    _set_int_parameter(sim, compound, sim.shapeintparam_static, 0)
    _set_int_parameter(sim, compound, sim.shapeintparam_respondable, 1)
    _dummy(sim, "CodeFace", compound)
    return compound


def _build_bin(sim: Any, specification: dict[str, Any], parent: int) -> None:
    group = _dummy(sim, specification["alias"], parent)
    slots = specification["slots"]
    center_x = sum(float(slot["position_mm"][0]) for slot in slots) / len(slots)
    center_y = sum(float(slot["position_mm"][1]) for slot in slots) / len(slots)
    color = [float(value) for value in specification["color_rgb"]]
    _shape(
        sim, name="Floor", shape="cuboid", size_mm=[34, 34, 2],
        position_mm=[center_x, center_y, 11], color=color, parent=group,
        respondable=True,
    )
    for name, dx, dy, sx, sy in (
        ("WallLeft", -18, 0, 2, 38), ("WallRight", 18, 0, 2, 38),
        ("WallFront", 0, -18, 38, 2), ("WallBack", 0, 18, 38, 2),
    ):
        _shape(
            sim, name=name, shape="cuboid", size_mm=[sx, sy, 12],
            position_mm=[center_x + dx, center_y + dy, 17], color=color,
            parent=group, respondable=True,
        )
    for slot in slots:
        handle = _dummy(sim, slot["alias"], group)
        sim.setObjectPosition(
            handle,
            [float(value) / 1000.0 for value in slot["position_mm"]],
            group,
        )


def _build_code_routing(
    sim: Any,
    spec: dict[str, Any],
    root: int,
    *,
    code_assets: dict[str, Any],
    profile_catalog: Any,
) -> None:
    _build_workspace(sim, spec["workspace"], root)
    parts_group = _dummy(sim, "Parts", root)
    bins_group = _dummy(sim, "Bins", root)
    camera_rig = _dummy(sim, "CameraRig", root)
    standard = profile_catalog.require(profile_catalog.baseline_profile_id)
    camera_spec = spec["camera"]
    sim.setObjectPosition(
        camera_rig,
        [float(value) for value in camera_spec["rig_position_m"]],
        sim.handle_world,
    )
    options = 1 | 2 | 4 | 64 | 128
    camera = int(
        sim.createVisionSensor(
            options,
            [int(standard.resolution[0]), int(standard.resolution[1]), 0, 0],
            [
                float(profile_catalog.near_clip_m),
                float(profile_catalog.far_clip_m),
                math.radians(float(standard.perspective_angle_deg)),
                0.02, 0.0, 0.0, 0.08, 0.08, 0.10, 0.0, 0.0,
            ],
        )
    )
    _alias(sim, camera, camera_spec["alias"])
    sim.setObjectParent(camera, camera_rig, False)
    sim.setObjectPosition(camera, [0.0, 0.0, 0.0], camera_rig)
    sim.setObjectOrientation(
        camera,
        [math.radians(float(value)) for value in camera_spec["orientation_deg"]],
        camera_rig,
    )
    assets = {entry["asset_id"]: entry for entry in code_assets["entries"]}
    if set(assets) != {part["code_asset_id"] for part in spec["parts"]}:
        raise ValueError("code assets and scene parts do not match")
    for part in spec["parts"]:
        _code_part(sim, part, assets[part["code_asset_id"]], parts_group)
    for bin_specification in spec["bins"]:
        _build_bin(sim, bin_specification, bins_group)
    _build_vision_lighting(sim, profile_catalog, root)
```

Refactor the existing logistics render-scope helper into a private helper that accepts only builder-owned literal root and camera paths, then preserve the logistics wrapper and add the V1-07 wrapper:

```python
def _attach_camera_scope(
    sim: Any,
    root: int,
    *,
    scene_root_path: str,
    camera_path: str,
) -> int:
    allowed = {
        ("/LogisticsLab", "/LogisticsLab/Camera"),
        ("/VisionCodeRoutingLab", "/VisionCodeRoutingLab/CameraRig/Camera"),
    }
    if (scene_root_path, camera_path) not in allowed:
        raise ValueError("camera render scope is not an approved formal scene")
    script_text = f"""
function sysCall_init()
    trainingCollection = sim.createCollection(1)
    sim.addItemToCollection(
        trainingCollection,
        sim.handle_tree,
        sim.getObject('{scene_root_path}'),
        0
    )
    local camera = sim.getObject('{camera_path}')
    sim.setObjectInt32Param(
        camera,
        sim.visionintparam_entity_to_render,
        trainingCollection
    )
end

function sysCall_cleanup()
    if trainingCollection then
        sim.destroyCollection(trainingCollection)
        trainingCollection = nil
    end
end
""".strip()
    handle = int(sim.createScript(sim.scripttype_simulation, script_text, 0, "lua"))
    _alias(sim, handle, "CameraRenderScope")
    sim.setObjectParent(handle, root, True)
    return handle


def _attach_logistics_camera_scope(sim: Any, root: int) -> int:
    return _attach_camera_scope(
        sim, root,
        scene_root_path="/LogisticsLab",
        camera_path="/LogisticsLab/Camera",
    )


def _attach_code_routing_camera_scope(sim: Any, root: int) -> int:
    return _attach_camera_scope(
        sim, root,
        scene_root_path="/VisionCodeRoutingLab",
        camera_path="/VisionCodeRoutingLab/CameraRig/Camera",
    )
```

Call `_attach_code_routing_camera_scope(sim, root)` immediately after `_build_code_routing`. This keeps the robot and unrelated template objects out of the recognition frame while preserving the existing logistics behavior and using no student-controlled Lua text.

The branch inserted into the existing `build_scene` conditional is exact:

```diff
elif formal.scene_id == "vision-code-routing-lab":
    code_assets_path = _project_file_from_relative(
        spec["code_assets_manifest"], label="code assets", must_exist=True
    )
    code_assets_bytes = code_assets_path.read_bytes()
    code_assets = json.loads(code_assets_bytes.decode("utf-8"))
    _build_code_routing(
        sim, spec, root,
        code_assets=code_assets,
        profile_catalog=profile_catalog,
    )
    _attach_code_routing_camera_scope(sim, root)
```

Generalize the existing profile-loading condition from only `vision-quality-lab` to both profile-owned scene IDs:

```diff
-if formal.scene_id == "vision-quality-lab":
+if formal.scene_id in {"vision-quality-lab", "vision-code-routing-lab"}:
```

Keep the existing catalog hashing, immutable byte check, manifest `profile_catalog` entry, atomic staging, and `_recoverable_release` validation unchanged for both scene families. The V1-07 exact-key validator must require `profiles`, the fixed rig position/orientation/plane, and equality between the baseline profile and the camera geometry used to derive the affine matrix.

After the common manifest dictionary is created, bind immutable routing metadata with:

```python
if formal.scene_id == "vision-code-routing-lab":
    standard = profile_catalog.require(profile_catalog.baseline_profile_id)
    plane_z_mm = float(spec["camera"]["code_face_plane_z_mm"])
    distance_mm = float(standard.camera_rig_z_m) * 1000.0 - plane_z_mm
    scale_mm_per_px = (
        2.0
        * distance_mm
        * math.tan(math.radians(float(standard.perspective_angle_deg)) / 2.0)
        / float(standard.resolution[0])
    )
    pixel_center = (float(standard.resolution[0]) - 1.0) / 2.0
    rig_x_mm = float(spec["camera"]["rig_position_m"][0]) * 1000.0
    rig_y_mm = float(spec["camera"]["rig_position_m"][1]) * 1000.0
    calibration_matrix = [
        [-scale_mm_per_px, 0.0, rig_x_mm + scale_mm_per_px * pixel_center],
        [0.0, scale_mm_per_px, rig_y_mm - scale_mm_per_px * pixel_center],
    ]
    manifest["code_assets_manifest"] = {
        "path": Path(spec["code_assets_manifest"]).name,
        "sha256": hashlib.sha256(code_assets_bytes).hexdigest(),
    }
    manifest["code_routing"] = {
        "part_ids": [part["alias"] for part in spec["parts"]],
        "initial_positions_mm": {
            part["alias"]: list(part["position_mm"]) for part in spec["parts"]
        },
        "calibration_plane_z_mm": plane_z_mm,
        "calibration_matrix": calibration_matrix,
        "route_slots_mm": {
            bin_specification["alias"]: [
                list(slot["position_mm"])
                for slot in bin_specification["slots"]
            ]
            for bin_specification in spec["bins"]
        },
    }
    manifest["reset_contract"] = dict(spec["reset_contract"])
```

Declare `code_assets_bytes` before the build branch as `None`; assert it is `bytes` before hashing in the manifest branch. Extend `_recoverable_release` with a second hash-bound auxiliary-file argument for the code-asset manifest, while preserving its existing profile-catalog validation. If the real BL23 reachability or the 3 mm online calibration check rejects any published coordinate, stop and return to architecture review rather than silently changing the course contract or widening a tolerance.

Create the thin launcher with no scene logic duplication:

```python
from pathlib import Path

from simulation.training_scenes.build_scene import build_scene


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    build_scene(
        spec_path=root / "simulation" / "vision_code_routing_lab" / "scene_spec.json",
        host="127.0.0.1",
        port=23007,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Generate the new scene under an owned CoppeliaSim process**

Use the existing process-ownership wrapper and a dedicated port. Confirm the source formal `.ttt` hashes before and after the builder. Generate `scene_manifest.json` only after saving the new scene and probing every declared path.

- [ ] **Step 4: Run static GREEN and protected-asset regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_simulation/test_v1_07_scene_contract.py `
  tests/test_simulation/test_formal_training_scenes.py `
  tests/test_simulation/test_robot_asset_provenance.py `
  tests/test_acceptance/test_delivery_contract.py
git diff --check
```

- [ ] **Step 5: Commit only the new scene family and its reproducible builder**

```powershell
git add -- tools/vision_lab/build_v1_07_scene.py `
  simulation/training_scenes/build_scene.py `
  simulation/vision_code_routing_lab `
  tests/test_simulation/test_v1_07_scene_contract.py `
  tests/test_simulation/test_formal_training_scenes.py RETAINED_FILES.txt
git commit -m "feat(simulation): add isolated V1-07 routing lab"
```

## Task 7: Publish the formal V1-07 course, template, and catalog entry

**Files:**

- Create: `config/experiments/V1-07.json`
- Modify: `config/experiments/catalog.json`
- Create: `docs/experiments/V1-07.md`
- Create: `student_programs/templates/v1_07_code_routing.py`
- Create: `tests/test_vision_quality/test_v1_07_materials.py`
- Modify: `tests/test_experiments/test_formal_catalog.py`
- Modify: `tests/test_experiments/test_student_templates.py`
- Modify: `tests/test_acceptance/test_experiment_guides.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write materials and public-template RED tests**

Require experiment version `2.2.0`, the new scene and manifest, capabilities for RGB/profile/code routing/robot/tool/probes, the exact four whitelist bindings, fixed calibration matrix, confidence floor, pick/drop/safe coordinates, command/runtime limits, automated and human checks, `PENDING_HARDWARE`, and guide sections for objectives, safety, procedure, evidence, error interpretation, simulation boundary, human acceptance, and hardware migration.

Require the student template to use public SDK only and the command sequence `experiment.info`, profile apply, `vision2d.code_routes`, complete-plan validation, guarded pick-and-place for each chosen entry, `robot.home`, checkpoint, and profile reset in `finally` without replacing the primary error.

Create these material tests and keep the existing formal-catalog/template parameterizations extended with `V1-07`:

```python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v1_07_definition_publishes_only_the_fixed_closed_loop() -> None:
    definition = json.loads(
        (ROOT / "config" / "experiments" / "V1-07.json").read_text(encoding="utf-8")
    )
    assert definition["experiment_id"] == "V1-07"
    assert definition["version"] == "2.2.0"
    assert definition["scene"] == "simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt"
    assert definition["hardware_status"] == "PENDING_HARDWARE"
    routing = definition["public_parameters"]["code_routing"]
    manifest = json.loads(
        (ROOT / definition["scene_manifest"]).read_text(encoding="utf-8")
    )
    assert routing["expected_count"] == 4
    assert {route["code_type"] for route in routing["routes"]} == {"qr", "ean13"}
    assert len({route["payload"] for route in routing["routes"]}) == 4
    assert len({tuple(route["drop_xyz_mm"]) for route in routing["routes"]}) == 4
    assert routing["calibration_matrix"] == manifest["code_routing"]["calibration_matrix"]
    assert not any(key in routing for key in ("path", "roi_px", "threshold", "command"))


def test_v1_07_guide_preserves_simulation_human_and_hardware_boundaries() -> None:
    guide = (ROOT / "docs" / "experiments" / "V1-07.md").read_text(encoding="utf-8")
    for heading in (
        "## 实验目标", "## 安全边界", "## 操作步骤", "## 证据说明",
        "## 错误解释", "## 自动检查", "## 人工验收", "## 真机迁移边界",
    ):
        assert heading in guide
    assert "PENDING_HUMAN_ACCEPTANCE" in guide
    assert "PENDING_HARDWARE" in guide
    assert "仿真通过不代表" in guide


def test_v1_07_template_contains_no_private_or_host_escape_api() -> None:
    source = (ROOT / "student_programs" / "templates" / "v1_07_code_routing.py").read_text(encoding="utf-8")
    assert "ctx.vision2d.code_routes()" in source
    assert "pick_and_place(" in source
    for forbidden in ("subprocess", "os.system", "cv2", "numpy", "sim.", "Path(", "open("):
        assert forbidden not in source
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_quality/test_v1_07_materials.py `
  tests/test_experiments/test_formal_catalog.py `
  tests/test_experiments/test_student_templates.py `
  tests/test_acceptance/test_experiment_guides.py
```

Expected RED: V1-07 materials are absent.

- [ ] **Step 2: Add the exact formal definition and catalog entry**

The route table must be immutable host configuration. Publish only IDs, teaching metadata, approved coordinates, and safety parameters through `ctx.experiment.info()`; do not publish filesystem paths or arbitrary detector settings.

Create `V1-07.json` with this exact content and append `"V1-07.json"` after `"V1-06.json"` in the catalog:

```json
{
  "schema_version": 1,
  "experiment_id": "V1-07",
  "pack_id": "V1",
  "title": "二维码与条形码识别及仓位路由",
  "version": "2.2.0",
  "scene": "simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt",
  "scene_manifest": "simulation/vision_code_routing_lab/scene_manifest.json",
  "student_template": "student_programs/templates/v1_07_code_routing.py",
  "guide": "docs/experiments/V1-07.md",
  "capabilities": [
    "camera.rgb", "camera.profile", "lighting.profile", "vision2d.code_routing",
    "experiment.info", "scene.probe", "robot.home", "robot.pose",
    "robot.move_world", "tool.suction"
  ],
  "workspace": {
    "x_mm": [20, 140], "y_mm": [-90, 90], "z_mm": [10, 140], "safe_z_mm": 110
  },
  "public_parameters": {
    "baseline_profile_id": "standard",
    "allowed_profile_ids": ["standard"],
    "camera_path": "/VisionCodeRoutingLab/CameraRig/Camera",
    "scene_group_path": "/VisionCodeRoutingLab",
    "pickables_path": "/VisionCodeRoutingLab/Parts",
    "code_routing": {
      "schema_version": 1,
      "expected_count": 4,
      "confidence_min": 0.9,
      "calibration_matrix": [[-0.1627580685211339, 0.0, 168.25075204856], [0.0, 0.1627580685211339, -83.25075204855999]],
      "pick_z_mm": 18.0,
      "safe_z_mm": 110.0,
      "speed_mm_s": 15.0,
      "routes": [
        {"entry_id": "entry_a", "part_id": "part_a", "code_type": "qr", "payload": "V1-07-A", "route_id": "route_red", "drop_xyz_mm": [116.0, -60.0, 22.0]},
        {"entry_id": "entry_b", "part_id": "part_b", "code_type": "qr", "payload": "V1-07-B", "route_id": "route_blue", "drop_xyz_mm": [116.0, 60.0, 22.0]},
        {"entry_id": "entry_c", "part_id": "part_c", "code_type": "ean13", "payload": "6901234567892", "route_id": "route_red", "drop_xyz_mm": [128.0, -60.0, 22.0]},
        {"entry_id": "entry_d", "part_id": "part_d", "code_type": "ean13", "payload": "6901234567809", "route_id": "route_blue", "drop_xyz_mm": [128.0, 60.0, 22.0]}
      ]
    }
  },
  "acceptance": {
    "probe_kind": "code_route_occupancy",
    "automated_checks": [
      "standard_profile", "four_codes", "complete_route_plan", "no_motion_before_plan",
      "four_guarded_transfers", "final_bin_occupancy", "robot_home", "tool_off",
      "same_run_evidence"
    ],
    "human_checks": [
      "学生解释 QR 与 EAN-13 的差异和校验边界",
      "学生解释白名单路由、计划先行和失败关闭"
    ]
  },
  "hardware_status": "PENDING_HARDWARE"
}
```

The published affine is derived from the fixed 1024x1024, 20-degree, 0.5 m top-down profile and the 27.4 mm code-face plane. It is not a generic camera calibration. The gateway's 3 mm manifest-position cross-check and the enabled online test must both pass before any robot motion; if the real scene readback violates that budget, stop and review the scene/profile/calibration contract rather than widening the tolerance.

- [ ] **Step 3: Implement the student template with plan-first validation**

Use the existing `r1_common.pick_and_place` helper only after all four entries have been validated. Sort by `entry_id` in the reference template while documenting that students may choose another order. Always call `tool.off` and profile reset during cleanup; retain the original exception as primary.

Create the reference template exactly as follows:

```python
from student_programs.templates.r1_common import pick_and_place


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    profile_id = parameters["baseline_profile_id"]
    if profile_id != "standard" or parameters["allowed_profile_ids"] != ["standard"]:
        raise RuntimeError("V1-07 只允许已发布的 standard 配置档")
    published = parameters["code_routing"]
    primary_error = None
    try:
        ctx.camera.apply_profile(profile_id)
        plan = ctx.vision2d.code_routes()
        if plan.status != "PASS" or len(plan.entries) != published["expected_count"]:
            raise RuntimeError("主机没有返回完整路线计划")
        if plan.safe_z_mm != published["safe_z_mm"] or plan.speed_mm_s != published["speed_mm_s"]:
            raise RuntimeError("路线计划安全参数与发布值不一致")
        published_routes = {
            route["entry_id"]: route for route in published["routes"]
        }
        if set(published_routes) != {entry.entry_id for entry in plan.entries}:
            raise RuntimeError("路线计划条目与发布白名单不一致")
        for entry in sorted(plan.entries, key=lambda item: item.entry_id):
            expected = published_routes[entry.entry_id]
            if (
                entry.part_id != expected["part_id"]
                or entry.code_type != expected["code_type"]
                or entry.payload != expected["payload"]
                or entry.route_id != expected["route_id"]
                or entry.drop_xyz_mm != tuple(expected["drop_xyz_mm"])
            ):
                raise RuntimeError(f"路线 {entry.entry_id} 未通过完整白名单校验")
        for entry in sorted(plan.entries, key=lambda item: item.entry_id):
            pick_and_place(
                ctx,
                pick_xy=entry.pick_xyz_mm[:2],
                drop_xyz=entry.drop_xyz_mm,
                pick_z_mm=entry.pick_xyz_mm[2],
                safe_z_mm=plan.safe_z_mm,
                speed=plan.speed_mm_s,
            )
            ctx.log(f"{entry.part_id}: {entry.code_type}/{entry.payload} -> {entry.route_id}")
        ctx.robot.home()
        ctx.checkpoint(f"V1-07 完成，plan_id={plan.plan_id}")
    except Exception as error:
        primary_error = error
        raise
    finally:
        cleanup_errors = []
        for label, cleanup in (
            ("tool.off", ctx.tool.off),
            ("camera.reset_profile", ctx.camera.reset_profile),
        ):
            try:
                cleanup()
            except Exception as cleanup_error:
                cleanup_errors.append((label, cleanup_error))
        if primary_error is not None:
            for label, cleanup_error in cleanup_errors:
                primary_error.add_note(f"{label} also failed: {cleanup_error}")
        elif cleanup_errors:
            label, first_error = cleanup_errors[0]
            for later_label, later_error in cleanup_errors[1:]:
                first_error.add_note(f"{later_label} also failed: {later_error}")
            raise first_error
```

Create `docs/experiments/V1-07.md` with the eight headings asserted above. State the four payload/part/bin mappings, the plan-before-motion rule, student-selectable order, raw/annotated/routes/motion/final-occupancy evidence, the fail-closed error codes, and this exact boundary sentence: `仿真通过不代表教学效果、海康相机、真实机械臂、急停、气路或物理抓取验收通过；前者保持 PENDING_HUMAN_ACCEPTANCE，后者保持 PENDING_HARDWARE。`

- [ ] **Step 4: Run GREEN and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_quality/test_v1_07_materials.py `
  tests/test_experiments/test_formal_catalog.py `
  tests/test_experiments/test_student_templates.py `
  tests/test_acceptance/test_experiment_guides.py
git diff --check
git add -- config/experiments docs/experiments/V1-07.md `
  student_programs/templates/v1_07_code_routing.py `
  tests/test_vision_quality/test_v1_07_materials.py `
  tests/test_experiments tests/test_acceptance/test_experiment_guides.py `
  RETAINED_FILES.txt
git commit -m "feat(curriculum): publish V1-07 code routing lab"
```

## Task 8: Add CLI and PowerShell entry points with exact process ownership

**Files:**

- Create: `tools/vision_lab/run_v1_07_code_routing.ps1`
- Modify: `tools/vision_lab/run_experiment.ps1`
- Create: `tests/test_experiments/test_v1_07_cli.py`
- Modify: `tests/test_acceptance/test_powershell_process_ownership.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write CLI and wrapper RED tests**

Require the generic `experiment-run --experiment V1-07` path and a convenience PowerShell wrapper that delegates to it, uses a dedicated port, records exact owned PID/port metadata, never kills unrelated CoppeliaSim processes, propagates nonzero exit codes, and leaves no child Python process after PASS, FAIL, stop, timeout, or Ctrl+C cleanup. Create this parser/catalog test:

```python
from vision_platform.cli import build_parser, main


def test_v1_07_uses_the_generic_formal_experiment_cli(capsys) -> None:
    arguments = build_parser().parse_args(
        ["experiment-run", "--experiment", "V1-07", "--port", "23007"]
    )
    assert arguments.experiment == "V1-07"
    assert arguments.port == "23007"
    assert main(["experiment-show", "--experiment", "V1-07"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["student_template"].endswith("v1_07_code_routing.py")
    assert payload["hardware_status"] == "PENDING_HARDWARE"
```

Add this static ownership test beside the existing V1-06 wrapper assertion:

```python
def test_v1_07_wrapper_delegates_to_owned_generic_launcher() -> None:
    source = (
        PROJECT_ROOT / "tools" / "vision_lab" / "run_v1_07_code_routing.ps1"
    ).read_text(encoding="utf-8-sig")
    assert "run_experiment.ps1" in source
    assert "'V1-07'" in source
    assert "23007" in source
    assert "Stop-Process" not in source
    assert "taskkill" not in source.lower()
    assert "exit $LASTEXITCODE" in source
```

- [ ] **Step 2: Implement only thin entry-point wiring**

Do not duplicate routing logic in CLI or PowerShell. The wrapper selects V1-07 and the published student template, then uses the existing bootstrap, launch, and process-ownership helpers.

Add `V1-07` to the existing `[ValidateSet]` in `run_experiment.ps1` and create this exact wrapper:

```powershell
[CmdletBinding()]
param(
    [string]$Program,
    [string]$HostName = '127.0.0.1',
    [int]$Port = 23007,
    [string]$Output = 'artifacts/vision_lab/experiment-runs'
)

$ErrorActionPreference = 'Stop'
$General = Join-Path $PSScriptRoot 'run_experiment.ps1'
$Arguments = @(
    '-Experiment', 'V1-07',
    '-HostName', $HostName,
    '-Port', [string]$Port,
    '-Output', $Output
)
if ($Program) {
    $Arguments += @('-Program', $Program)
}
& powershell.exe -ExecutionPolicy Bypass -File $General @Arguments
exit $LASTEXITCODE
```

No `vision_platform/cli.py` change is expected: catalog registration supplies V1-07 to the existing generic commands. If a CLI source change appears necessary, stop and first prove a missing generic behavior with a focused RED test.

- [ ] **Step 3: Run GREEN and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_experiments/test_v1_07_cli.py `
  tests/test_experiments/test_cli.py `
  tests/test_acceptance/test_powershell_process_ownership.py
git diff --check
git add -- tools/vision_lab `
  tests/test_experiments/test_v1_07_cli.py `
  tests/test_acceptance/test_powershell_process_ownership.py `
  RETAINED_FILES.txt
git commit -m "feat(cli): add V1-07 controlled launcher"
```

## Task 9: Show route and final-state evidence in PyQt

**Files:**

- Modify: `vision_platform/ui/vision_result_panel.py`
- Modify: `vision_platform/ui/student_program_panel.py`
- Create: `tests/test_vision_platform/test_v1_07_result_panel.py`
- Modify: `tests/test_vision_platform/test_student_program_panel.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write offscreen UI RED tests**

Require read-only sections for raw image, annotated image, decoded code type/payload, part ID, route ID, approved pick/drop coordinates, motion events, completed entries, final bin occupancy, error code, and hardware/human acceptance state. Payload text must render as inert escaped text and must never become a clickable path, command, or editable execution field.

Create a real recorded bundle and assert the read-only rendering:

```python
from __future__ import annotations

import numpy as np

from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.ui.vision_result_panel import VisionResultPanel
from vision_platform.vision_quality.evidence import record_vision_bundle
from vision_platform.vision_quality.results import make_result_bundle


def _record_route_bundle(tmp_path):
    program = tmp_path / "v1_07.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs", program_path=program,
        robot_backend="sim", run_id="v1-07-panel-run",
    )
    raw = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    annotated = raw.copy()
    result = {
        "plan_id": "a" * 64,
        "status": "PASS",
        "entries": [
            {
                "entry_id": "entry_a", "part_id": "part_a", "code_type": "qr",
                "payload": "[V1-07-A](file:///C:/unsafe)", "route_id": "route_red",
                "pick_xyz_mm": [40.0, -45.0, 18.0],
                "drop_xyz_mm": [116.0, -60.0, 22.0], "confidence": 0.98,
            }
        ],
        "motion_events": [{"name": "tool.off", "status": "PASS"}],
        "completed_entry_ids": ["entry_a"],
        "final_occupancy": {"route_red": ["part_a"], "route_blue": []},
        "error": None,
        "human_acceptance": "PENDING_HUMAN_ACCEPTANCE",
        "hardware_status": "PENDING_HARDWARE",
    }
    bundle = make_result_bundle(
        bundle_id="V1-07-frame-000001-final", experiment_id="V1-07",
        source_snapshot_id="frame-000001", status="PASS",
        layers={"raw": ("原图", raw), "annotated": ("代码与仓位标注", annotated)},
        result=result,
        profile={"profile_id": "standard", "resolution": [1024, 1024]},
    )
    record_vision_bundle(evidence, bundle)
    return evidence.directory


def test_panel_renders_v1_07_routes_as_inert_read_only_text(qtbot, tmp_path) -> None:
    panel = VisionResultPanel()
    qtbot.addWidget(panel)
    panel.load_run(_record_route_bundle(tmp_path))
    assert tuple(
        panel.layer_combo.itemData(index) for index in range(panel.layer_combo.count())
    ) == ("raw", "annotated")
    assert panel.route_text.isReadOnly()
    rendered = panel.route_text.toPlainText()
    assert "[V1-07-A](file:///C:/unsafe)" in rendered
    assert "part_a" in rendered and "route_red" in rendered
    assert "45.0, -45.0, 18.0" in rendered
    assert "tool.off" in rendered
    assert "PENDING_HUMAN_ACCEPTANCE" in rendered
    assert "PENDING_HARDWARE" in rendered
```

- [ ] **Step 2: Implement a V1-07 renderer without changing existing experiment views**

Route rendering must be selected by structured result type or experiment ID. Preserve V1-01 through V1-06 and R1 behavior. Empty/partial/failed evidence must display a stable failure state rather than pretending the route completed.

Add a read-only plain-text section in `_build_ui`, clear it in `clear_result`, and render only when the exact route fields are present:

```python
# In _build_ui after metrics_label:
self.route_summary_label = QLabel("代码路由：—")
self.route_summary_label.setObjectName("visionRouteSummary")
self.route_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
root.addWidget(self.route_summary_label)
self.route_text = QPlainTextEdit()
self.route_text.setObjectName("visionRouteDetails")
self.route_text.setReadOnly(True)
self.route_text.setMaximumBlockCount(400)
self.route_text.setMinimumHeight(150)
root.addWidget(self.route_text)

# In clear_result:
self.route_summary_label.setText("代码路由：—")
self.route_text.clear()
```

`QPlainTextEdit` has no link activation path. Do not replace it with `QTextBrowser`.

Add this formatter and call it from `_show_bundle` after obtaining `result`:

```python
def _show_code_routes(self, result: Any) -> None:
    required = {
        "plan_id", "status", "entries", "motion_events",
        "completed_entry_ids", "final_occupancy", "error",
        "human_acceptance", "hardware_status",
    }
    if not isinstance(result, dict) or not required <= set(result):
        self.route_summary_label.setText("代码路由：—")
        self.route_text.clear()
        return
    entries = result["entries"]
    if not isinstance(entries, list):
        self.route_summary_label.setText("代码路由：证据格式无效")
        self.route_text.clear()
        return
    lines = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"{entry.get('entry_id', '—')} | {entry.get('code_type', '—')} | "
            f"{entry.get('payload', '—')} | {entry.get('part_id', '—')} -> "
            f"{entry.get('route_id', '—')} | pick={entry.get('pick_xyz_mm', '—')} | "
            f"drop={entry.get('drop_xyz_mm', '—')}"
        )
    lines.append(f"motion_events={result['motion_events']}")
    lines.append(f"completed={result['completed_entry_ids']}")
    lines.append(f"final_occupancy={result['final_occupancy']}")
    lines.append(f"error={result['error']}")
    lines.append(f"human={result['human_acceptance']}")
    lines.append(f"hardware={result['hardware_status']}")
    self.route_summary_label.setText(
        f"代码路由：{result['status']}　计划：{result['plan_id']}"
    )
    self.route_text.setPlainText("\n".join(lines))


# In _show_bundle:
self._show_code_routes(result)
```

Import `QPlainTextEdit`. Use `setPlainText` only; never use `setHtml`, `setMarkdown`, anchors, URL handlers, or command callbacks. For failed evidence, the final route bundle from Task 10 supplies the same exact key set with `status="REJECTED"`, partial completion, final occupancy, and a structured error, so the panel displays failure rather than inferring success.

- [ ] **Step 3: Run UI GREEN and regression**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision_platform/test_v1_07_result_panel.py `
  tests/test_vision_platform/test_student_program_panel.py `
  tests/test_vision_platform/test_vision_result_panel.py `
  tests/test_vision_platform/test_experiment_catalog_panel.py
Remove-Item Env:QT_QPA_PLATFORM
```

- [ ] **Step 4: Perform and record 100% and 125% human screenshot checks**

Use the same successful V1-07 evidence in both scale checks. Confirm no clipping, overlap, hidden error text, editable route fields, or unreadable payloads. Store screenshots under runtime artifacts only; do not add them to Git. Record `PENDING_HUMAN_ACCEPTANCE` despite UI rendering success.

- [ ] **Step 5: Commit UI behavior and tests**

```powershell
git diff --check
git add -- vision_platform/ui tests/test_vision_platform `
  RETAINED_FILES.txt
git commit -m "feat(ui): display V1-07 routing evidence"
```

## Task 10: Prove the complete loop in real CoppeliaSim

**Files:**

- Create: `tests/test_acceptance/test_coppeliasim_v1_07.py`
- Modify: `vision_platform/experiments/probes.py`
- Modify: `vision_platform/student/experiment_gateway.py`
- Modify: `vision_platform/student/runner.py`
- Modify: `tests/test_student_programs/test_experiment_evidence.py`
- Modify: `tests/test_student_programs/test_runner.py`
- Modify: `tools/vision_lab/run_vision_quality_acceptance.ps1`
- Modify: `tests/test_acceptance/test_powershell_process_ownership.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write the explicitly enabled online RED test**

Require exact scene hash, initial probe PASS, one frame from the declared camera, two QR plus two EAN-13 readings, four approved routes, zero motion before plan activation, each part in its declared final drop slot/bin, all four guard entries complete, robot home, suction off, final probe PASS, same-run snapshot/plan/motion/final evidence linkage, no child process, and `PENDING_HARDWARE` plus `PENDING_HUMAN_ACCEPTANCE`.

Create the online test using the same subprocess boundary as V1-06:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vision_platform.vision_quality.evidence import load_recorded_bundle


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.coppeliasim
def test_v1_07_student_template_completes_guarded_code_routes(tmp_path, request) -> None:
    host = request.config.getoption("--coppelia-host") or os.environ.get("COPPELIA_HOST") or "127.0.0.1"
    configured_port = request.config.getoption("--coppelia-port")
    port = configured_port if configured_port is not None else int(os.environ.get("COPPELIA_PORT", "23007"))
    completed = subprocess.run(
        [
            sys.executable, "-m", "vision_platform.cli", "experiment-run",
            "--experiment", "V1-07", "--host", host, "--port", str(port),
            "--output", str(tmp_path / "runs"),
        ],
        cwd=ROOT, text=True, encoding="utf-8", capture_output=True,
        timeout=300, check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["status"] == "PASS"
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    evidence_dir = Path(payload["evidence"]).resolve()
    summary = json.loads(Path(payload["summary"]).read_text(encoding="utf-8"))
    commands = [
        json.loads(line)
        for line in (evidence_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    names = [item["name"] for item in commands]
    plan_index = names.index("vision2d.code_routes")
    first_motion = min(names.index("robot.move_world"), names.index("tool.on"))
    assert plan_index < first_motion
    assert names.count("vision2d.code_routes") == 1
    assert names.count("camera.capture") == 0
    assert names[-1] == "camera.profile.reset"
    assert summary["scene_probe_status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    final_probe = json.loads((evidence_dir / "scene-final.json").read_text(encoding="utf-8"))
    assert final_probe["status"] == "PASS"
    assert final_probe["matched"] == final_probe["expected"] == 4
    assert final_probe["final_occupancy"] == {
        "route_blue": ["part_b", "part_d"],
        "route_red": ["part_a", "part_c"],
    }
    artifacts = sorted(evidence_dir.glob("vision-bundle-*.json"))
    assert len(artifacts) == 2
    final_artifact = next(path for path in artifacts if "zfinal" in path.name)
    bundle = load_recorded_bundle(evidence_dir, final_artifact.name)
    result = bundle["result"]
    assert bundle["experiment_id"] == "V1-07"
    assert bundle["status"] == "PASS"
    assert [layer["layer_id"] for layer in bundle["layers"]] == ["raw", "annotated"]
    assert len(result["entries"]) == 4
    assert len(result["completed_entry_ids"]) == 4
    assert result["final_occupancy"] == final_probe["final_occupancy"]
    assert result["human_acceptance"] == "PENDING_HUMAN_ACCEPTANCE"
    assert result["hardware_status"] == "PENDING_HARDWARE"
    assert result["snapshot_id"] == bundle["source_snapshot_id"]
    assert all(event["plan_id"] == result["plan_id"] for event in result["motion_events"])
```

Run RED only against an owned CoppeliaSim process. A connection failure, missing executable, or skipped test is not acceptable RED; the expected failure must be an unimplemented V1-07 contract.

- [ ] **Step 2: Add fail-closed online cases**

Using deterministic scene overrides, verify unknown/duplicate/unreadable code fails before the first robot movement. Inject one mid-motion failure and verify tool off, no later route commands, original error preserved, evidence flushed, and exact process cleanup.

Keep unknown/duplicate/unreadable frames in the gateway tests from Task 4. For the online negative path, make a temporary copy of the formal scene, remove one `/CodeFace` through the owned Remote API, and run the normal binding/probe path against that copy; expect scene/hash or recognition failure before any movement. Do not commit the temporary `.ttt`. Inject the mid-motion exception through the existing fake robot adapter runner test, because altering the formal simulator transport to manufacture a motion fault would not be a production-faithful online test.

The runner/evidence unit tests must assert that a mid-motion primary error yields a `REJECTED` final bundle, retains the original error code, records only commands that actually reached dispatch, attempts physical `tool.off`, adds cleanup diagnostics without replacing the primary error, and never marks all entries complete. Rerun both unit files in this Task; do not rely only on the success-path online test.

Add the new probe kind and exact implementation:

```python
# Add to _KNOWN_PROBE_KINDS:
"code_route_occupancy",


def _probe_code_route_occupancy(
    sim: Any,
    definition: Any,
    *,
    experiment_id: str,
    phase: str,
    parameters: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tolerance_mm: float,
) -> dict[str, Any]:
    route_config = _mapping(parameters.get("code_routing"), "public_parameters.code_routing")
    routes = _sequence(route_config.get("routes"), "public_parameters.code_routing.routes", length=4)
    scene_contract = _mapping(manifest.get("code_routing"), "scene_manifest.code_routing")
    initial = _mapping(scene_contract.get("initial_positions_mm"), "scene_manifest.code_routing.initial_positions_mm")
    rows: list[dict[str, Any]] = []
    occupancy: dict[str, list[str]] = {}
    matched = 0
    for index, raw in enumerate(routes):
        route = _mapping(raw, f"public_parameters.code_routing.routes[{index}]")
        part_id = _alias(route.get("part_id"), f"routes[{index}].part_id")
        route_id = _alias(route.get("route_id"), f"routes[{index}].route_id")
        expected = (
            _position(initial.get(part_id), f"initial_positions_mm.{part_id}")
            if phase == "initial"
            else _position(route.get("drop_xyz_mm"), f"routes[{index}].drop_xyz_mm")
        )
        actual = _world_mm(sim, f"/VisionCodeRoutingLab/Parts/{part_id}")
        distance_mm = _distance_mm(actual, expected)
        is_match = distance_mm <= tolerance_mm
        matched += int(is_match)
        if phase == "final" and is_match:
            occupancy.setdefault(route_id, []).append(part_id)
        rows.append(
            {
                "part_id": part_id, "route_id": route_id,
                "position_mm": actual, "expected_mm": expected,
                "distance_mm": distance_mm, "matched": is_match,
            }
        )
    if phase == "final":
        for route_id in {row["route_id"] for row in rows}:
            occupancy.setdefault(route_id, [])
            occupancy[route_id].sort()
    return {
        "schema_version": 1, "experiment_id": experiment_id, "phase": phase,
        "status": "PASS" if matched == 4 else "FAIL",
        "matched": matched, "expected": 4, "tolerance_mm": tolerance_mm,
        "rows": rows, "final_occupancy": dict(sorted(occupancy.items())),
        "hardware_status": "PENDING_HARDWARE",
    }


# In probe_experiment before the logistics branches:
if kind == "code_route_occupancy":
    return _probe_code_route_occupancy(
        sim, definition, experiment_id=experiment_id, phase=phase,
        parameters=parameters, manifest=manifest, tolerance_mm=tolerance,
    )
```

For `phase="initial"`, `final_occupancy` is an empty mapping and all four initial positions must match. The test must assert both initial and final probe documents belong to V1-07 and have the same scene hash recorded by the run summary.

Persist the successful planning context after the first bundle is recorded and before activating the guard:

```python
self._route_evidence = {
    "plan": plan,
    "snapshot_id": snapshot_id,
    "raw": recorded.image_bgr.copy(),
    "annotated": annotated.copy(),
    "raw_record": dict(recorded.record),
    "profile": {**profile, "code_route_plan_id": plan.plan_id},
}
```

Initialize `_route_evidence` to `None` in the gateway constructor and clear it in `reset_environment`. Add this final-evidence method:

```python
def record_code_route_final(
    self,
    *,
    final_probe: Mapping[str, Any] | None,
    primary_error: Mapping[str, Any] | None,
) -> str | None:
    context = self._route_evidence
    if context is None:
        return None
    plan = context["plan"]
    completion = self.route_guard.completion()
    probe = {} if final_probe is None else _copy_json_native(final_probe, path="code_route.final_probe")
    commands_path = Path(self.evidence.directory) / "commands.jsonl"
    commands = []
    if commands_path.is_file():
        for line in commands_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("name") in {"robot.home", "robot.move_world", "tool.on", "tool.off"}:
                commands.append(
                    {
                        "plan_id": plan.plan_id,
                        "sequence": len(commands) + 1,
                        "name": item["name"],
                        "status": item.get("status", "UNKNOWN"),
                    }
                )
    probe_pass = probe.get("status") == "PASS"
    success = completion["all_complete"] and probe_pass and primary_error is None
    error = (
        None
        if success
        else _copy_json_native(
            primary_error
            or {"code": "CODE_ROUTE_FINAL_STATE_INVALID", "message": "路线终态未通过"},
            path="code_route.error",
        )
    )
    result = {
        **code_route_plan_to_dict(plan),
        "status": "PASS" if success else "REJECTED",
        "snapshot_id": context["snapshot_id"],
        "motion_events": commands,
        "completed_entry_ids": completion["completed_entry_ids"],
        "final_occupancy": probe.get("final_occupancy", {}),
        "error": error,
        "human_acceptance": "PENDING_HUMAN_ACCEPTANCE",
        "hardware_status": "PENDING_HARDWARE",
    }
    bundle = VisionResultBundle(
        schema_version=1,
        bundle_id=f"V1-07-zfinal-{context['snapshot_id']}",
        experiment_id=self.context.experiment_id,
        source_snapshot_id=context["snapshot_id"],
        status="PASS" if success else "REJECTED",
        layers=(
            VisionImageLayer("raw", "原图", context["raw"]),
            VisionImageLayer("annotated", "代码与仓位标注", context["annotated"]),
        ),
        result=result,
        profile=context["profile"],
        hardware_status="PENDING_HARDWARE",
    )
    return record_vision_bundle(
        self.evidence, bundle,
        existing_layer_records={"raw": context["raw_record"]},
    )
```

In runner terminalization, call `record_code_route_final(final_probe=final_probe_report, primary_error=self._error)` after the final probe and command records have been flushed but before the summary is written. Store the returned artifact name in the summary. If final-evidence recording fails, append `CODE_ROUTE_FINAL_EVIDENCE_FAILED` to cleanup diagnostics; preserve any existing student/motion error as primary, and fail an otherwise successful run because same-run final evidence is part of V1-07 completion.

- [ ] **Step 3: Extend the owned acceptance wrapper**

Add the enabled V1-07 online tests and experiment run without removing V1-01 through V1-06 coverage. Update exact JUnit expectations, timeout, JSON summary, PID ownership assertions, and final port-free assertion.

- [ ] **Step 4: Run online GREEN**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_student_programs/test_runner.py `
  tests/test_student_programs/test_experiment_evidence.py
powershell -ExecutionPolicy Bypass -File `
  .\tools\vision_lab\run_vision_quality_acceptance.ps1 `
  -OutputDir artifacts\vision_lab\v2-2-v1-07
```

Expected: wrapper exits 0; enabled JUnit contains zero failures, errors, or skips; V1-01 through V1-07 required runs pass; the dedicated port is free; no owned CoppeliaSim or Python process remains. Report exactly what was run.

- [ ] **Step 5: Commit acceptance coverage**

```powershell
git add -- vision_platform/experiments/probes.py `
  vision_platform/student/experiment_gateway.py vision_platform/student/runner.py `
  tests/test_student_programs/test_experiment_evidence.py `
  tests/test_student_programs/test_runner.py `
  tests/test_acceptance/test_coppeliasim_v1_07.py `
  tests/test_acceptance/test_powershell_process_ownership.py `
  tools/vision_lab/run_vision_quality_acceptance.ps1 RETAINED_FILES.txt
git commit -m "test(acceptance): verify V1-07 closed loop"
```

## Task 11: Run release, protected-asset, and full regression gates

**Files:**

- Modify: `tests/test_acceptance/test_delivery_contract.py`
- Verify: `RETAINED_FILES.txt`
- Verify: all files changed from the Task 1 base

- [ ] **Step 1: Audit branch scope and protected files**

```powershell
$base = git merge-base HEAD origin/main
git diff --name-status $base..HEAD
git diff --name-only $base..HEAD -- `
  "*.urdf" "*.stl" "*.dae" "*.obj" `
  simulation/vision_lab/BL23_vision_lab.ttt `
  simulation/vision_quality_lab/BL23_vision_quality_lab.ttt
```

Expected: only approved V1-07 paths changed; the protected-file query is empty.

- [ ] **Step 2: Validate retained-file union and release contract**

Before running the contract, add this explicit V1-07 release set and its retained assertion, append `V1-07` to the exact formal V1 ID tuples, and include V1-07 in the hardware-pending/no-grading loop:

```python
V22_V1_07_REQUIRED_RELEASE_PATHS = frozenset(
    {
        "config/experiments/V1-07.json",
        "docs/experiments/V1-07.md",
        "docs/superpowers/plans/2026-08-02-v1-07-code-routing-plan.md",
        "docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md",
        "simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt",
        "simulation/vision_code_routing_lab/code_assets/ean_6901234567809.png",
        "simulation/vision_code_routing_lab/code_assets/ean_6901234567892.png",
        "simulation/vision_code_routing_lab/code_assets/qr_v1_07_a.png",
        "simulation/vision_code_routing_lab/code_assets/qr_v1_07_b.png",
        "simulation/vision_code_routing_lab/code_assets_manifest.json",
        "simulation/vision_code_routing_lab/profiles.json",
        "simulation/vision_code_routing_lab/scene_manifest.json",
        "simulation/vision_code_routing_lab/scene_spec.json",
        "student_programs/templates/v1_07_code_routing.py",
        "tests/test_acceptance/test_coppeliasim_v1_07.py",
        "tests/test_experiments/test_code_routing.py",
        "tests/test_experiments/test_v1_07_cli.py",
        "tests/test_simulation/test_v1_07_code_assets.py",
        "tests/test_simulation/test_v1_07_scene_contract.py",
        "tests/test_student_programs/test_v1_07_gateway.py",
        "tests/test_student_programs/test_v1_07_protocol_sdk.py",
        "tests/test_student_programs/test_v1_07_route_guard.py",
        "tests/test_vision_platform/test_v1_07_result_panel.py",
        "tests/test_vision_quality/test_v1_07_materials.py",
        "tools/vision_lab/build_v1_07_scene.py",
        "tools/vision_lab/generate_v1_07_code_assets.py",
        "tools/vision_lab/run_v1_07_code_routing.ps1",
        "vision_platform/experiments/code_routing.py",
        "vision_platform/student/code_route_guard.py",
    }
)


def test_retained_files_contains_complete_v1_07_delivery() -> None:
    retained = _retained_release_paths()
    missing = sorted(V22_V1_07_REQUIRED_RELEASE_PATHS - retained)
    assert not missing, "missing V1-07 paths: " + ", ".join(missing)
    assert all((ROOT / path).is_file() for path in V22_V1_07_REQUIRED_RELEASE_PATHS)
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_acceptance/test_delivery_contract.py `
  tests/test_vision_quality/test_delivery.py `
  tests/test_vision2d/test_delivery.py
git diff --check
```

Expected: zero failures, no duplicate retained entries, no missing retained path, and no stale cache or runtime-evidence path. Reproducibly generated formal PNG/`.ttt` release assets are expected and hash-bound.

- [ ] **Step 3: Run all related static suites**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_vision2d `
  tests/test_experiments `
  tests/test_student_programs `
  tests/test_vision_quality `
  tests/test_vision_platform `
  tests/test_simulation
```

Expected: zero failures. Report skips by reason.

- [ ] **Step 4: Run the complete static suite from a clean process**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
git status --short --branch
```

Expected: zero failures and no uncommitted implementation files. Existing explicit online skips remain separately reported.

- [ ] **Step 5: Commit the explicit V1-07 release registration**

After rerunning Steps 2 through 4 with the registration above:

```powershell
git add -- tests/test_acceptance/test_delivery_contract.py RETAINED_FILES.txt
git commit -m "test(delivery): register V1-07 release files"
```


## Task 12: Independent review, push, and handoff

**Files:**

- Create: `docs/superpowers/reports/2026-08-02-v1-07-validation-report.md`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write the validation report from captured evidence**

Include base SHA, branch tip, Task commit list, RED/GREEN commands, static suite results, explicitly enabled online JUnit results, UI scale checks, protected-asset audit, retained-file audit, process cleanup, known limitations, `PENDING_HUMAN_ACCEPTANCE`, and `PENDING_HARDWARE`. Do not convert skips or unrun checks into PASS.

Use this exact report heading order and retain the boundary paragraph verbatim. Populate every evidence section only with literal command output captured in this run:

```markdown
# V2.2 V1-07 Validation Report

## Scope and claim boundary

本报告只记录 V1-07 在 CoppeliaSim 中的代码识别、白名单路线、受控运动、虚拟仓位和同次运行证据。教学效果保持 PENDING_HUMAN_ACCEPTANCE；海康相机、真实机械臂、急停、气路和物理抓取保持 PENDING_HARDWARE。

## Exact baseline and branch tip

## Task commits

## RED and GREEN evidence

## Static regression

## Explicitly enabled CoppeliaSim acceptance

## PyQt 100% and 125% checks

## Protected assets and retained files

## Process ownership and cleanup

## Independent review findings and fixes

## Remaining limitations
```

- [ ] **Step 2: Request independent P0/P1/P2 review**

The reviewer must compare the full diff against the approved coordination design and specifically inspect payload inertness, plan atomicity, motion-before-plan prevention, route-guard bypasses, error precedence, cleanup, scene/asset provenance, online process ownership, and evidence same-run linkage.

- [ ] **Step 3: Fix every accepted P0/P1 and relevant P2 through a fresh RED/GREEN cycle**

Make focused follow-up commits. Re-run the affected target suite, complete static suite, and online wrapper for any motion, scene, gateway, acceptance, or cleanup change.

- [ ] **Step 4: Commit the final report and retained entry**

```powershell
git add -- docs/superpowers/reports/2026-08-02-v1-07-validation-report.md `
  RETAINED_FILES.txt
git commit -m "docs: record V1-07 validation evidence"
```

- [ ] **Step 5: Perform final clean verification and push only this branch**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
git status --short --branch
git log --oneline origin/main..HEAD
git push -u origin codex/v2-2-v1-07-code-routing
git rev-parse HEAD
git rev-parse origin/codex/v2-2-v1-07-code-routing
```

Expected: local and remote full SHAs match, worktree is clean, and no merge to `main` occurred.

- [ ] **Step 6: Hand off exact claims**

Report changed files, commit range, static results, enabled online results, UI evidence status, remaining human/hardware gates, and the recommended integration order. The strongest permitted claim is “V1-07 CoppeliaSim virtual closed loop passed the recorded online acceptance”; do not claim real camera, real robot, physical picking, or teaching effectiveness passed.
