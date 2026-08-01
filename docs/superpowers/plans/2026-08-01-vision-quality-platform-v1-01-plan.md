# V2.2-C0 Vision Quality Platform and V1-01 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated CoppeliaSim vision-quality scene, allowlisted camera/lighting profiles, a reusable vision-result evidence contract, and the formal V1-01 virtual-vision experiment without modifying the parallel V1-02–V1-05 algorithm kernel.

**Architecture:** V1-01 publishes three immutable profile IDs. Student SDK commands pass through the existing protocol and experiment gateway to a rollback-safe `VisionProfileController`; captures become generic `VisionResultBundle` evidence that a read-only PyQt panel can display. The formal scene is generated from the protected vision-lab template into a new `.ttt`, and all online claims require explicit CoppeliaSim runs.

**Tech Stack:** Python 3.11, dataclasses, NumPy, OpenCV, PyQt5, CoppeliaSim ZeroMQ Remote API, JSON, pytest, PowerShell, Git worktrees.

---

## 0. Execution contract

Run every command from:

```powershell
cd 'C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-vision-quality-platform'
```

The implementation branch must remain:

```text
codex/v2-2-vision-quality-platform
```

The immutable base is:

```text
v2.2.0
37a385fd596d890716cbff9eb86e7feae31c11c9
```

Before Task 1, verify the branch and baseline:

```powershell
git status --short --branch
git merge-base HEAD v2.2.0
git rev-list --left-right --count v2.2.0...HEAD
```

Expected before implementation: the branch contains only the committed design/plan baseline on top of `v2.2.0`; the worktree is clean.

Do not modify these paths in any Task:

```text
vision_platform/vision2d/**
tests/test_vision2d/**
docs/experiments/V1-02.md
docs/experiments/V1-03.md
docs/experiments/V1-04.md
docs/experiments/V1-05.md
vision_platform/recognition/color_shape.py
simulation/vision_lab/BL23_vision_lab.ttt
simulation/vision_lab/robot_assets/**
*.urdf
*.stl
```

The parallel algorithm branch completed and was pushed at `f5bef91`. Its retained evidence reports `34 passed` for `tests/test_vision2d` and `658 passed, 3 skipped` for its V2.1-based full static suite. Those are algorithm-only results, not this branch's acceptance. Never merge or cherry-pick it during this plan. A later integration owner will join the branches after V1-01 is independently complete.

## 1. File and responsibility map

### New runtime files

```text
vision_platform/vision_quality/__init__.py
  Public V1-01 profile/result API only.

vision_platform/vision_quality/models.py
  Immutable profile, applied-state, image-layer and result-bundle models.

vision_platform/vision_quality/catalog.py
  Strict profiles.json loading, validation and SHA-256 verification.

vision_platform/vision_quality/controller.py
  Allowlisted CoppeliaSim profile apply/readback/rollback/reset logic.

vision_platform/vision_quality/results.py
  Image validation, bundle construction and JSON-native conversion.

vision_platform/vision_quality/evidence.py
  Public StudentRunEvidence adapter and safe recorded-bundle loader.

vision_platform/ui/vision_result_panel.py
  Read-only PyQt result-layer and JSON viewer.
```

### New formal experiment and scene files

```text
simulation/vision_quality_lab/__init__.py
simulation/vision_quality_lab/scene_spec.json
simulation/vision_quality_lab/profiles.json
simulation/vision_quality_lab/BL23_vision_quality_lab.ttt
simulation/vision_quality_lab/scene_manifest.json
config/experiments/V1-01.json
docs/experiments/V1-01.md
student_programs/templates/v1_01_virtual_vision.py
```

### New tests

```text
tests/test_vision_quality/__init__.py
tests/test_vision_quality/test_catalog.py
tests/test_vision_quality/test_controller.py
tests/test_vision_quality/test_results.py
tests/test_vision_quality/test_evidence.py
tests/test_vision_quality/test_v1_01_materials.py
tests/test_vision_quality/test_delivery.py
tests/test_vision_platform/test_vision_result_panel.py
tests/test_acceptance/test_coppeliasim_vision_quality_scene.py
tests/test_acceptance/test_coppeliasim_v1_01.py
```

### Minimal shared integration points

```text
simulation/training_scenes/build_scene.py
simulation/training_scenes/scene_contract.py
vision_platform/experiments/capabilities.py
vision_platform/experiments/probes.py
vision_platform/student/protocol.py
vision_platform/student/sdk.py
vision_platform/student/experiment_gateway.py
vision_platform/student/runner.py
vision_platform/ui/pyqt_app.py
config/experiments/catalog.json
tests/test_experiments/**
tests/test_student_programs/**
tests/test_vision_platform/**
tests/test_acceptance/test_delivery_contract.py
RETAINED_FILES.txt
```

## Task 1: Immutable visual-profile catalog

**Files:**

- Create: `vision_platform/vision_quality/__init__.py`
- Create: `vision_platform/vision_quality/models.py`
- Create: `vision_platform/vision_quality/catalog.py`
- Create: `tests/test_vision_quality/__init__.py`
- Create: `tests/test_vision_quality/test_catalog.py`

- [ ] **Step 1: Write the failing catalog tests**

Create `tests/test_vision_quality/__init__.py` as an empty package marker. Create `tests/test_vision_quality/test_catalog.py` with focused contract tests:

```python
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from vision_platform.vision_quality.catalog import load_profile_catalog


def _payload():
    return {
        "schema_version": 1,
        "baseline_profile_id": "standard",
        "sensor_path": "/VisionQualityLab/CameraRig/Camera",
        "camera_rig_path": "/VisionQualityLab/CameraRig",
        "key_light_path": "/VisionQualityLab/Lighting/KeyLight",
        "fill_light_path": "/VisionQualityLab/Lighting/FillLight",
        "near_clip_m": 0.05,
        "far_clip_m": 2.0,
        "profiles": [
            {
                "profile_id": "standard",
                "label": "标准视图",
                "resolution": [512, 512],
                "perspective_angle_deg": 60.0,
                "camera_rig_z_m": 0.70,
                "key_diffuse_rgb": [0.8, 0.8, 0.8],
                "fill_diffuse_rgb": [0.35, 0.35, 0.35],
            },
            {
                "profile_id": "wide_dim",
                "label": "宽视场弱光",
                "resolution": [256, 256],
                "perspective_angle_deg": 75.0,
                "camera_rig_z_m": 0.80,
                "key_diffuse_rgb": [0.35, 0.35, 0.35],
                "fill_diffuse_rgb": [0.15, 0.15, 0.15],
            },
        ],
    }


def _write(tmp_path, payload=None):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(_payload() if payload is None else payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_profile_catalog_loads_exact_immutable_contract(tmp_path):
    catalog = load_profile_catalog(_write(tmp_path))

    assert catalog.baseline_profile_id == "standard"
    assert catalog.require("wide_dim").resolution == (256, 256)
    assert catalog.require("standard").key_diffuse_rgb == (0.8, 0.8, 0.8)
    assert catalog.profile_ids == ("standard", "wide_dim")
    with pytest.raises(FrozenInstanceError):
        catalog.baseline_profile_id = "wide_dim"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"extra": True}), "fields mismatch"),
        (lambda p: p.update({"schema_version": 2}), "schema_version"),
        (lambda p: p.update({"baseline_profile_id": "missing"}), "baseline"),
        (lambda p: p["profiles"][0].update({"resolution": [64, 512]}), "resolution"),
        (lambda p: p["profiles"][0].update({"perspective_angle_deg": 91}), "angle"),
        (lambda p: p["profiles"][0].update({"camera_rig_z_m": 1.2}), "camera_rig_z_m"),
        (lambda p: p["profiles"][0].update({"key_diffuse_rgb": [1.1, 0, 0]}), "RGB"),
        (lambda p: p["profiles"].append(dict(p["profiles"][0])), "duplicate"),
    ],
)
def test_profile_catalog_rejects_unsafe_or_ambiguous_values(tmp_path, mutate, message):
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        load_profile_catalog(_write(tmp_path, payload))


def test_profile_catalog_requires_fixed_scene_paths(tmp_path):
    payload = _payload()
    payload["sensor_path"] = "/student/chosen/path"
    with pytest.raises(ValueError, match="sensor_path"):
        load_profile_catalog(_write(tmp_path, payload))
```

- [ ] **Step 2: Run the test and verify the missing package failure**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_quality/test_catalog.py -q
```

Expected: collection fails because `vision_platform.vision_quality` does not exist.

- [ ] **Step 3: Implement immutable models and strict loading**

Create `vision_platform/vision_quality/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionProfile:
    profile_id: str
    label: str
    resolution: tuple[int, int]
    perspective_angle_deg: float
    camera_rig_z_m: float
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class VisionProfileCatalog:
    baseline_profile_id: str
    sensor_path: str
    camera_rig_path: str
    key_light_path: str
    fill_light_path: str
    near_clip_m: float
    far_clip_m: float
    profiles: tuple[VisionProfile, ...]

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(item.profile_id for item in self.profiles)

    def require(self, profile_id: str) -> VisionProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(f"unknown vision profile: {profile_id}")


@dataclass(frozen=True)
class AppliedVisionProfile:
    profile_id: str
    resolution: tuple[int, int]
    perspective_angle_deg: float
    camera_rig_z_m: float
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "resolution": list(self.resolution),
            "perspective_angle_deg": self.perspective_angle_deg,
            "camera_rig_z_m": self.camera_rig_z_m,
            "key_diffuse_rgb": list(self.key_diffuse_rgb),
            "fill_diffuse_rgb": list(self.fill_diffuse_rgb),
        }
```

Create `vision_platform/vision_quality/catalog.py` with exact-field validation. Use these constants and checks rather than accepting arbitrary paths:

```python
_CATALOG_FIELDS = {
    "schema_version", "baseline_profile_id", "sensor_path",
    "camera_rig_path", "key_light_path", "fill_light_path",
    "near_clip_m", "far_clip_m", "profiles",
}
_PROFILE_FIELDS = {
    "profile_id", "label", "resolution", "perspective_angle_deg",
    "camera_rig_z_m", "key_diffuse_rgb", "fill_diffuse_rgb",
}
_FIXED_PATHS = {
    "sensor_path": "/VisionQualityLab/CameraRig/Camera",
    "camera_rig_path": "/VisionQualityLab/CameraRig",
    "key_light_path": "/VisionQualityLab/Lighting/KeyLight",
    "fill_light_path": "/VisionQualityLab/Lighting/FillLight",
}
_PROFILE_ID = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
```

Implement `load_profile_catalog(path)` so it:

```python
def load_profile_catalog(path: str | Path) -> VisionProfileCatalog:
    selected = Path(path).expanduser().resolve()
    payload = json.loads(
        selected.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    _exact_fields(payload, _CATALOG_FIELDS, "profile catalog")
    if payload["schema_version"] != 1:
        raise ValueError("profile catalog schema_version must be 1")
    for field, expected in _FIXED_PATHS.items():
        if payload[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    near_clip = _bounded(payload["near_clip_m"], "near_clip_m", 0.01, 0.20)
    far_clip = _bounded(payload["far_clip_m"], "far_clip_m", 1.0, 5.0)
    if near_clip >= far_clip:
        raise ValueError("near_clip_m must be less than far_clip_m")
    raw_profiles = payload["profiles"]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("profiles must be a non-empty list")
    profiles = tuple(_profile(item) for item in raw_profiles)
    ids = tuple(item.profile_id for item in profiles)
    if len(ids) != len(set(ids)):
        raise ValueError("profiles contain duplicate profile_id")
    baseline = _text(payload["baseline_profile_id"], "baseline_profile_id")
    if baseline not in ids:
        raise ValueError("baseline_profile_id must name a published profile")
    return VisionProfileCatalog(
        baseline_profile_id=baseline,
        sensor_path=payload["sensor_path"],
        camera_rig_path=payload["camera_rig_path"],
        key_light_path=payload["key_light_path"],
        fill_light_path=payload["fill_light_path"],
        near_clip_m=near_clip,
        far_clip_m=far_clip,
        profiles=profiles,
    )
```

The `_profile()` helper must enforce resolution components `128..1024`, perspective angle `20..90`, rig Z `0.50..0.90`, and RGB components `0..1`. Reject booleans and non-finite numbers. Return tuples, never caller-owned lists.

Create `vision_platform/vision_quality/__init__.py` exporting only:

```python
from vision_platform.vision_quality.catalog import load_profile_catalog
from vision_platform.vision_quality.models import (
    AppliedVisionProfile,
    VisionProfile,
    VisionProfileCatalog,
)

__all__ = [
    "AppliedVisionProfile",
    "VisionProfile",
    "VisionProfileCatalog",
    "load_profile_catalog",
]
```

- [ ] **Step 4: Run catalog tests and full model regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_quality/test_catalog.py `
  tests/test_experiments/test_catalog.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add vision_platform/vision_quality tests/test_vision_quality
git commit -m "feat(vision-quality): define immutable visual profiles"
```

## Task 2: Formal vision-quality scene specification and static contract

**Files:**

- Create: `simulation/vision_quality_lab/__init__.py`
- Create: `simulation/vision_quality_lab/profiles.json`
- Create: `simulation/vision_quality_lab/scene_spec.json`
- Modify: `simulation/training_scenes/build_scene.py`
- Modify: `simulation/training_scenes/scene_contract.py`
- Create: `tests/test_simulation/test_vision_quality_scene_contract.py`

- [ ] **Step 1: Write failing formal-scene tests**

Create `tests/test_simulation/test_vision_quality_scene_contract.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import simulation.training_scenes.build_scene as builder
from vision_platform.vision_quality.catalog import load_profile_catalog


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "simulation" / "vision_quality_lab" / "scene_spec.json"
PROFILES = ROOT / "simulation" / "vision_quality_lab" / "profiles.json"

REQUIRED_PATHS = [
    "/VisionQualityLab",
    "/VisionQualityLab/InspectionBoard",
    "/VisionQualityLab/Samples",
    "/VisionQualityLab/Samples/ReferenceRectangle",
    "/VisionQualityLab/Samples/ReferenceCircle",
    "/VisionQualityLab/Samples/ReferenceTriangle",
    "/VisionQualityLab/Samples/ResolutionTarget",
    "/VisionQualityLab/CameraRig",
    "/VisionQualityLab/CameraRig/Camera",
    "/VisionQualityLab/Lighting",
    "/VisionQualityLab/Lighting/KeyLight",
    "/VisionQualityLab/Lighting/FillLight",
]


def test_vision_quality_spec_is_an_approved_non_overwriting_scene():
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["scene_id"] == "vision-quality-lab"
    assert payload["template"] == "simulation/vision_lab/BL23_vision_lab.ttt"
    assert payload["output"] == (
        "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt"
    )
    assert payload["template"] != payload["output"]
    assert payload["root_path"] == "/VisionQualityLab"
    assert payload["required_paths"] == REQUIRED_PATHS
    assert payload["profiles"] == "simulation/vision_quality_lab/profiles.json"


def test_three_published_profiles_match_the_course_contract():
    catalog = load_profile_catalog(PROFILES)
    assert catalog.profile_ids == ("standard", "wide_dim", "detail_bright")
    assert catalog.require("standard").resolution == (512, 512)
    assert catalog.require("wide_dim").resolution == (256, 256)
    assert catalog.require("detail_bright").resolution == (768, 768)


def test_builder_accepts_only_the_canonical_vision_quality_spec():
    path, formal = builder._formal_spec_path(SPEC)
    assert path == SPEC
    assert formal.scene_id == "vision-quality-lab"
    with pytest.raises(ValueError, match="approved formal spec"):
        builder._formal_spec_path(PROFILES)


def test_profile_digest_is_stable_and_nonempty():
    digest = hashlib.sha256(PROFILES.read_bytes()).hexdigest()
    assert len(digest) == 64
    assert digest != "0" * 64
```

- [ ] **Step 2: Run and verify missing files**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_simulation/test_vision_quality_scene_contract.py -q
```

Expected: failure because the new scene directory/spec/profile files are absent.

- [ ] **Step 3: Add the fixed profile and scene declarations**

Create `simulation/vision_quality_lab/profiles.json` with all three exact profiles from the design:

```json
{
  "schema_version": 1,
  "baseline_profile_id": "standard",
  "sensor_path": "/VisionQualityLab/CameraRig/Camera",
  "camera_rig_path": "/VisionQualityLab/CameraRig",
  "key_light_path": "/VisionQualityLab/Lighting/KeyLight",
  "fill_light_path": "/VisionQualityLab/Lighting/FillLight",
  "near_clip_m": 0.05,
  "far_clip_m": 2.0,
  "profiles": [
    {
      "profile_id": "standard",
      "label": "标准视图",
      "resolution": [512, 512],
      "perspective_angle_deg": 60.0,
      "camera_rig_z_m": 0.7,
      "key_diffuse_rgb": [0.8, 0.8, 0.8],
      "fill_diffuse_rgb": [0.35, 0.35, 0.35]
    },
    {
      "profile_id": "wide_dim",
      "label": "宽视场弱光",
      "resolution": [256, 256],
      "perspective_angle_deg": 75.0,
      "camera_rig_z_m": 0.8,
      "key_diffuse_rgb": [0.35, 0.35, 0.35],
      "fill_diffuse_rgb": [0.15, 0.15, 0.15]
    },
    {
      "profile_id": "detail_bright",
      "label": "细节强光",
      "resolution": [768, 768],
      "perspective_angle_deg": 40.0,
      "camera_rig_z_m": 0.6,
      "key_diffuse_rgb": [1.0, 1.0, 1.0],
      "fill_diffuse_rgb": [0.55, 0.55, 0.55]
    }
  ]
}
```

Create `scene_spec.json` with exact top-level fields:

```json
{
  "schema_version": 1,
  "scene_id": "vision-quality-lab",
  "template": "simulation/vision_lab/BL23_vision_lab.ttt",
  "output": "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt",
  "remove_paths": ["/VisionLab"],
  "root_path": "/VisionQualityLab",
  "profiles": "simulation/vision_quality_lab/profiles.json",
  "workspace": {
    "center_m": [0.35, 0.0, 0.02],
    "size_m": [0.36, 0.28, 0.02]
  },
  "camera": {
    "rig_position_m": [0.35, 0.0, 0.7],
    "orientation_deg": [180.0, 0.0, 0.0]
  },
  "samples": {
    "ReferenceRectangle": {"shape": "cuboid", "position_m": [0.29, -0.06, 0.045], "size_m": [0.08, 0.04, 0.03], "color_rgb": [0.85, 0.1, 0.1]},
    "ReferenceCircle": {"shape": "cylinder", "position_m": [0.41, -0.06, 0.045], "size_m": [0.05, 0.05, 0.03], "color_rgb": [0.1, 0.3, 0.9]},
    "ReferenceTriangle": {"shape": "triangle", "position_m": [0.29, 0.07, 0.045], "size_m": [0.07, 0.06, 0.03], "color_rgb": [0.1, 0.75, 0.2]},
    "ResolutionTarget": {"shape": "resolution_target", "position_m": [0.41, 0.07, 0.031], "size_m": [0.08, 0.06, 0.002], "stripe_count": 12}
  },
  "required_paths": [
    "/VisionQualityLab",
    "/VisionQualityLab/InspectionBoard",
    "/VisionQualityLab/Samples",
    "/VisionQualityLab/Samples/ReferenceRectangle",
    "/VisionQualityLab/Samples/ReferenceCircle",
    "/VisionQualityLab/Samples/ReferenceTriangle",
    "/VisionQualityLab/Samples/ResolutionTarget",
    "/VisionQualityLab/CameraRig",
    "/VisionQualityLab/CameraRig/Camera",
    "/VisionQualityLab/Lighting",
    "/VisionQualityLab/Lighting/KeyLight",
    "/VisionQualityLab/Lighting/FillLight"
  ]
}
```

The triangle builder must create original geometry from vertices. The resolution target builder must create twelve alternating black/white cuboids; do not add an external image.

- [ ] **Step 4: Register and validate the third formal scene**

In `simulation/training_scenes/build_scene.py`:

1. Add `_VISION_QUALITY_REQUIRED_PATHS` equal to the test list.
2. Add a third `_FormalScene` entry for the canonical spec and output.
3. Change `_validate_spec()` from a two-way conditional to explicit three-way dispatch:

```python
if formal.scene_id == "robot-basics":
    detail_field = "markers"
elif formal.scene_id == "logistics-lab":
    detail_field = "tasks"
elif formal.scene_id == "vision-quality-lab":
    detail_field = "samples"
else:
    raise ValueError(f"unsupported formal scene: {formal.scene_id}")
```

For the vision scene, require the additional exact `profiles` field and validate it equals `simulation/vision_quality_lab/profiles.json`. Call `load_profile_catalog()` during static validation. Add `_validate_vision_samples()` to enforce the four exact aliases, shape allowlist, finite positions/sizes/colors and `stripe_count == 12`.

In `simulation/training_scenes/scene_contract.py`, when `scene_id == "vision-quality-lab"`, construct and require the manifest entry from the real file:

```python
profile_path = root / "simulation/vision_quality_lab/profiles.json"
expected_profile_entry = {
    "path": "simulation/vision_quality_lab/profiles.json",
    "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
}
assert manifest["profile_catalog"] == expected_profile_entry
```

Resolve the path inside `project_root`, verify the file hash and return `profile_sha256` in the validation report.

- [ ] **Step 5: Run static scene contracts and existing scene regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_simulation/test_vision_quality_scene_contract.py `
  tests/test_simulation/test_training_scene_contract.py `
  tests/test_simulation/test_formal_training_scenes.py -q
```

Expected: all selected static tests pass. No `.ttt` is created in this Task.

- [ ] **Step 6: Commit Task 2**

```powershell
git add simulation/vision_quality_lab simulation/training_scenes `
  tests/test_simulation/test_vision_quality_scene_contract.py
git commit -m "feat(simulation): define vision quality lab contract"
```

## Task 3: Rollback-safe CoppeliaSim profile controller

**Files:**

- Create: `vision_platform/vision_quality/controller.py`
- Modify: `vision_platform/vision_quality/__init__.py`
- Create: `tests/test_vision_quality/test_controller.py`

- [ ] **Step 1: Write fake-sim controller tests first**

Create `tests/test_vision_quality/test_controller.py`. The fake must expose only the Remote API methods the controller is allowed to call and record call order:

```python
from __future__ import annotations

import math

import pytest

from vision_platform.vision_quality.controller import VisionProfileController
from vision_platform.vision_quality.models import VisionProfile, VisionProfileCatalog


class FakeSim:
    visionintparam_resolution_x = 1002
    visionintparam_resolution_y = 1003
    visionfloatparam_perspective_angle = 1004
    visionfloatparam_near_clipping = 1000
    visionfloatparam_far_clipping = 1001
    handle_world = -1

    def __init__(self):
        self.handles = {
            "/VisionQualityLab/CameraRig/Camera": 1,
            "/VisionQualityLab/CameraRig": 2,
            "/VisionQualityLab/Lighting/KeyLight": 3,
            "/VisionQualityLab/Lighting/FillLight": 4,
        }
        self.int_params = {(1, 1002): 512, (1, 1003): 512}
        self.float_params = {(1, 1004): math.radians(60), (1, 1000): 0.05, (1, 1001): 2.0}
        self.positions = {2: [0.35, 0.0, 0.70]}
        self.lights = {3: (True, [0.8] * 3, [0.0] * 3), 4: (True, [0.35] * 3, [0.0] * 3)}
        self.calls = []
        self.fail_on = None

    def getObject(self, path):
        self.calls.append(("getObject", path))
        return self.handles[path]

    def getObjectInt32Param(self, handle, param):
        return self.int_params[(handle, param)]

    def setObjectInt32Param(self, handle, param, value):
        self._set(("int", handle, param), lambda: self.int_params.__setitem__((handle, param), int(value)))

    def getObjectFloatParam(self, handle, param):
        return self.float_params[(handle, param)]

    def setObjectFloatParam(self, handle, param, value):
        self._set(("float", handle, param), lambda: self.float_params.__setitem__((handle, param), float(value)))

    def getObjectPosition(self, handle, relative):
        assert relative == self.handle_world
        return list(self.positions[handle])

    def setObjectPosition(self, handle, relative, value):
        assert relative == self.handle_world
        self._set(("position", handle), lambda: self.positions.__setitem__(handle, list(value)))

    def getLightParameters(self, handle):
        return self.lights[handle]

    def setLightParameters(self, handle, enabled, diffuse, specular):
        self._set(("light", handle), lambda: self.lights.__setitem__(handle, (bool(enabled), list(diffuse), list(specular))))

    def _set(self, key, action):
        self.calls.append(key)
        if self.fail_on == key:
            self.fail_on = None
            raise RuntimeError("simulated set failure")
        action()


class FakeCamera:
    def __init__(self):
        self.read_count = 0

    def read(self, timeout_s=1.0):
        self.read_count += 1
        return object()


def _catalog():
    profiles = (
        VisionProfile("standard", "标准", (512, 512), 60.0, 0.70, (0.8,) * 3, (0.35,) * 3),
        VisionProfile("wide_dim", "宽弱", (256, 256), 75.0, 0.80, (0.35,) * 3, (0.15,) * 3),
    )
    return VisionProfileCatalog(
        "standard",
        "/VisionQualityLab/CameraRig/Camera",
        "/VisionQualityLab/CameraRig",
        "/VisionQualityLab/Lighting/KeyLight",
        "/VisionQualityLab/Lighting/FillLight",
        0.05,
        2.0,
        profiles,
    )


def test_apply_sets_all_fields_reads_back_and_discards_first_frame():
    sim = FakeSim()
    camera = FakeCamera()
    controller = VisionProfileController(
        sim=sim,
        camera=camera,
        catalog=_catalog(),
        allowed_profile_ids=("standard", "wide_dim"),
    )

    state = controller.apply("wide_dim")

    assert state.profile_id == "wide_dim"
    assert state.resolution == (256, 256)
    assert state.perspective_angle_deg == pytest.approx(75.0)
    assert camera.read_count == 1


def test_unknown_or_unallowed_profile_never_calls_a_setter():
    sim = FakeSim()
    controller = VisionProfileController(
        sim=sim,
        camera=FakeCamera(),
        catalog=_catalog(),
        allowed_profile_ids=("standard",),
    )
    with pytest.raises(ValueError, match="VISION_PROFILE_NOT_ALLOWED"):
        controller.apply("wide_dim")
    assert not any(call[0] in {"int", "float", "position", "light"} for call in sim.calls)


def test_apply_failure_rolls_back_every_previously_changed_value():
    sim = FakeSim()
    sim.fail_on = ("light", 4)
    controller = VisionProfileController(
        sim=sim,
        camera=FakeCamera(),
        catalog=_catalog(),
        allowed_profile_ids=("standard", "wide_dim"),
    )
    with pytest.raises(RuntimeError, match="VISION_PROFILE_APPLY_FAILED"):
        controller.apply("wide_dim")
    assert sim.int_params[(1, 1002)] == 512
    assert sim.int_params[(1, 1003)] == 512
    assert sim.positions[2][2] == pytest.approx(0.70)


def test_reset_is_idempotent_and_returns_standard():
    controller = VisionProfileController(
        sim=FakeSim(),
        camera=FakeCamera(),
        catalog=_catalog(),
        allowed_profile_ids=("standard", "wide_dim"),
    )
    controller.apply("wide_dim")
    assert controller.reset().profile_id == "standard"
    assert controller.reset().profile_id == "standard"
```

Add separate tests for missing fixed objects, readback mismatch, a second fake with permanent setter failure to prove `VISION_PROFILE_ROLLBACK_FAILED`, non-string profile IDs, duplicate allowed IDs and concurrent apply serialization.

- [ ] **Step 2: Run and verify controller is missing**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_quality/test_controller.py -q
```

Expected: import failure for `vision_platform.vision_quality.controller`.

- [ ] **Step 3: Implement the minimal controller with explicit snapshots**

Create `vision_platform/vision_quality/controller.py` with private immutable `_SceneSnapshot`. Resolve four fixed handles in `__init__`; validate the camera backend has callable `read`. Use `RLock` and these public methods:

```python
def current(self) -> AppliedVisionProfile:
    with self._lock:
        snapshot = self._read_snapshot()
        return self._match_published_profile(snapshot)

def apply(self, profile_id: str) -> AppliedVisionProfile:
    selected = self._allowed_profile(profile_id)
    with self._lock:
        before = self._read_snapshot()
        try:
            self._write_profile(selected)
            after = self._read_snapshot()
            self._assert_matches(after, selected)
            self._camera.read(timeout_s=2.0)
        except BaseException as original:
            try:
                self._restore_snapshot(before)
                self._assert_snapshot(self._read_snapshot(), before)
            except BaseException as rollback_error:
                raise RuntimeError("VISION_PROFILE_ROLLBACK_FAILED") from rollback_error
            raise RuntimeError("VISION_PROFILE_APPLY_FAILED") from original
        return self._public_state(selected)

def reset(self) -> AppliedVisionProfile:
    try:
        return self.apply(self._catalog.baseline_profile_id)
    except BaseException as error:
        raise RuntimeError("VISION_PROFILE_RESET_FAILED") from error
```

Write resolution X/Y and clipping params with integer/float APIs, FOV in radians, rig Z while preserving world X/Y, and enabled diffuse light values while preserving each light's specular component. Readback tolerances:

```text
angle: 0.05 degree
position: 0.0005 metre
light channels: 0.001
clipping: 0.0005 metre
resolution: exact integer equality
```

`_match_published_profile()` must return the unique matching profile or raise `VISION_PROFILE_READBACK_MISMATCH`. Never infer a nearest profile.

- [ ] **Step 4: Run controller tests twice for determinism**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_quality/test_controller.py -q
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_quality/test_controller.py -q
```

Expected: both runs pass with identical counts.

- [ ] **Step 5: Commit Task 3**

```powershell
git add vision_platform/vision_quality tests/test_vision_quality/test_controller.py
git commit -m "feat(vision-quality): control allowlisted camera profiles"
```

## Task 4: Student protocol and SDK profile commands

**Files:**

- Modify: `vision_platform/student/protocol.py`
- Modify: `vision_platform/student/sdk.py`
- Modify: `tests/test_student_programs/test_protocol.py`
- Modify: `tests/test_student_programs/test_sdk.py`

- [ ] **Step 1: Extend tests before the command allowlist**

Append to `tests/test_student_programs/test_protocol.py`:

```python
def test_v1_01_profile_commands_are_exactly_whitelisted():
    assert {
        "camera.profile.get",
        "camera.profile.apply",
        "camera.profile.reset",
    } <= ALLOWED_COMMANDS
    assert "sim.setObjectInt32Param" not in ALLOWED_COMMANDS
    assert "camera.profile.set_arbitrary" not in ALLOWED_COMMANDS
```

Add an RPC recorder test to `tests/test_student_programs/test_sdk.py`:

```python
def test_student_camera_exposes_profile_ids_not_raw_sim_parameters():
    connection = ScriptedConnection(
        [
            {"profile_id": "standard", "resolution": [512, 512], "perspective_angle_deg": 60.0, "camera_rig_z_m": 0.7, "key_diffuse_rgb": [0.8] * 3, "fill_diffuse_rgb": [0.35] * 3},
            {"profile_id": "wide_dim", "resolution": [256, 256], "perspective_angle_deg": 75.0, "camera_rig_z_m": 0.8, "key_diffuse_rgb": [0.35] * 3, "fill_diffuse_rgb": [0.15] * 3},
            {"profile_id": "standard", "resolution": [512, 512], "perspective_angle_deg": 60.0, "camera_rig_z_m": 0.7, "key_diffuse_rgb": [0.8] * 3, "fill_diffuse_rgb": [0.35] * 3},
        ]
    )
    ctx = StudentContext(connection)

    assert ctx.camera.get_profile().profile_id == "standard"
    assert ctx.camera.apply_profile("wide_dim").resolution == (256, 256)
    assert ctx.camera.reset_profile().profile_id == "standard"
    assert [item["name"] for item in connection.sent] == [
        "camera.profile.get", "camera.profile.apply", "camera.profile.reset"
    ]
    assert connection.sent[1]["args"] == {"profile_id": "wide_dim"}
```

Also test malformed response fields, non-finite floats, wrong resolution length and non-string IDs. Each must raise `PROTOCOL_RESPONSE_INVALID`.

- [ ] **Step 2: Run and verify missing commands/methods**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_student_programs/test_protocol.py `
  tests/test_student_programs/test_sdk.py -q
```

Expected: allowlist assertion and missing SDK methods fail.

- [ ] **Step 3: Add three commands and a strict public state model**

Add exactly three strings to `ALLOWED_COMMANDS` in `protocol.py`.

In `sdk.py`, add:

```python
@dataclass(frozen=True)
class StudentVisionProfile:
    profile_id: str
    resolution: tuple[int, int]
    perspective_angle_deg: float
    camera_rig_z_m: float
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]
```

Implement one private strict parser `_vision_profile(value)` that rejects booleans, NaN/Inf, extra/missing fields, invalid tuple lengths and unsafe IDs. Add to `StudentCamera`:

```python
def get_profile(self) -> StudentVisionProfile:
    return _vision_profile(self._rpc.call("camera.profile.get"))

def apply_profile(self, profile_id: str) -> StudentVisionProfile:
    if type(profile_id) is not str or _PROFILE_ID.fullmatch(profile_id) is None:
        raise ValueError("profile_id must be a published ASCII identifier")
    return _vision_profile(
        self._rpc.call("camera.profile.apply", profile_id=profile_id)
    )

def reset_profile(self) -> StudentVisionProfile:
    return _vision_profile(self._rpc.call("camera.profile.reset"))
```

Do not expose sensor paths, handles, arbitrary resolution, pose or light setters.

- [ ] **Step 4: Run protocol, SDK and validator regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_student_programs/test_protocol.py `
  tests/test_student_programs/test_sdk.py `
  tests/test_student_programs/test_validator.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add vision_platform/student/protocol.py vision_platform/student/sdk.py `
  tests/test_student_programs/test_protocol.py tests/test_student_programs/test_sdk.py
git commit -m "feat(student): expose allowlisted vision profiles"
```

## Task 5: Generic vision-result bundle and evidence adapter

**Files:**

- Modify: `vision_platform/vision_quality/models.py`
- Create: `vision_platform/vision_quality/results.py`
- Create: `vision_platform/vision_quality/evidence.py`
- Modify: `vision_platform/vision_quality/__init__.py`
- Create: `tests/test_vision_quality/test_results.py`
- Create: `tests/test_vision_quality/test_evidence.py`

- [ ] **Step 1: Write result-model tests**

Create `tests/test_vision_quality/test_results.py`:

```python
from __future__ import annotations

import json

import numpy as np
import pytest

from vision_platform.vision_quality.results import make_result_bundle, result_bundle_to_dict


def test_bundle_preserves_named_read_only_bgr_layers_and_json_result():
    raw = np.zeros((24, 32, 3), dtype=np.uint8)
    annotated = raw.copy()
    annotated[2:5, 3:8] = (0, 255, 0)

    bundle = make_result_bundle(
        bundle_id="frame-000001",
        experiment_id="V1-01",
        source_snapshot_id="frame-000001",
        status="PASS",
        layers={"raw": ("原图", raw), "annotated": ("标注图", annotated)},
        result={"observed": True, "count": 1},
        profile={"profile_id": "standard", "resolution": [32, 24]},
    )

    assert tuple(layer.layer_id for layer in bundle.layers) == ("raw", "annotated")
    assert bundle.layers[0].image_bgr.flags.writeable is False
    assert bundle.hardware_status == "PENDING_HARDWARE"
    payload = result_bundle_to_dict(
        bundle,
        layer_records={
            "raw": {"path": "frames/frame-000001.png", "sha256": "0" * 64, "width": 32, "height": 24},
            "annotated": {"path": "frames/frame-000001-annotated.png", "sha256": "1" * 64, "width": 32, "height": 24},
        },
    )
    assert payload["result"] == {"observed": True, "count": 1}
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_bundle_rejects_invalid_images(bad_image):
    with pytest.raises((TypeError, ValueError), match="image"):
        make_result_bundle(
            bundle_id="frame-000001",
            experiment_id="V1-01",
            source_snapshot_id="frame-000001",
            status="PASS",
            layers={"raw": ("原图", bad_image)},
            result={},
            profile={"profile_id": "standard"},
        )


def test_bundle_rejects_non_json_or_nonfinite_results():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises((TypeError, ValueError)):
        make_result_bundle(
            bundle_id="frame-000001",
            experiment_id="V1-01",
            source_snapshot_id="frame-000001",
            status="PASS",
            layers={"raw": ("原图", image)},
            result={"bad": float("nan")},
            profile={"profile_id": "standard"},
        )
```

Test stable layer ordering, duplicate/suspicious IDs, mutation of caller arrays/mappings after construction, and strict `hardware_status` validation: `type(hardware_status) is str` and the value is exactly `PENDING_HARDWARE`. Add `0 skipped` RED cases proving that direct `VisionImageLayer`/`VisionResultBundle` construction and `dataclasses.replace()` cannot bypass validation or defensive copying; later caller mutation cannot change stored images or nested result/profile values; surrogate-containing strings or mapping keys are rejected; and public bundle failures expose the stable code `VISION_RESULT_BUNDLE_INVALID`.

- [ ] **Step 2: Write evidence tests using the real evidence class**

Create `tests/test_vision_quality/test_evidence.py`:

```python
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.vision_quality.evidence import (
    load_recorded_bundle,
    record_vision_bundle,
)
from vision_platform.vision_quality.results import make_result_bundle


def _evidence(tmp_path):
    program = tmp_path / "student.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    return StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="vision-run",
    )


def test_record_bundle_reuses_existing_raw_snapshot_and_loads_safely(tmp_path):
    evidence = _evidence(tmp_path)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()
    raw_record = evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=png,
        metadata={"width": 32, "height": 24, "source": "coppeliasim"},
    )
    bundle = make_result_bundle(
        bundle_id="frame-000001",
        experiment_id="V1-01",
        source_snapshot_id="frame-000001",
        status="PASS",
        layers={"raw": ("原图", image)},
        result={"capture": "ok"},
        profile={"profile_id": "standard", "resolution": [32, 24]},
    )

    artifact = record_vision_bundle(
        evidence,
        bundle,
        existing_layer_records={"raw": raw_record},
    )

    assert artifact == "vision-bundle-frame-000001.json"
    assert len(list((evidence.directory / "frames").glob("*.png"))) == 1
    loaded = load_recorded_bundle(evidence.directory, artifact)
    assert loaded["bundle_id"] == "frame-000001"
    assert loaded["layers"][0]["sha256"] == hashlib.sha256(png).hexdigest()


def test_loader_rejects_path_escape_or_hash_mismatch(tmp_path):
    evidence = _evidence(tmp_path)
    artifact = evidence.directory / "vision-bundle-frame-000001.json"
    artifact.write_text(
        json.dumps({
            "schema_version": 1,
            "bundle_id": "frame-000001",
            "experiment_id": "V1-01",
            "source_snapshot_id": "frame-000001",
            "status": "PASS",
            "layers": [{"layer_id": "raw", "title": "原图", "path": "../outside.png", "sha256": "0" * 64, "width": 32, "height": 24}],
            "result": {},
            "profile": {"profile_id": "standard"},
            "hardware_status": "PENDING_HARDWARE",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inside evidence directory"):
        load_recorded_bundle(evidence.directory, artifact.name)
```

Also test duplicate artifact names, PNG encode failure cleanup, layer-record dimension mismatch, missing files, invalid JSON constants and changed file hashes. Add `0 skipped` RED cases proving that the loader rejects JPEG bytes disguised with a `.png` name; an `existing_layer_records["raw"]` record must correspond exactly to the bundle's `raw` layer; supplying that record when the bundle has no `raw` layer is rejected before any write; normal and overlength appended-layer snapshot IDs follow the deterministic rule below; all appended-layer ID collisions are rejected before any write; and public evidence failures expose `VISION_RESULT_EVIDENCE_INVALID`, preserve `__cause__`, and translate nested bundle validation failures to that evidence code.

Add RED cases with `annotated` ordered before `raw` proving that an invalid existing raw record is still rejected without leaving a new PNG; an appended-layer ID colliding with the reused raw `source_snapshot_id` is rejected without leaving a new PNG; a pre-existing second-layer target or a final payload that cannot be UTF-8 JSON serialized leaves no new PNG; and same-run arbitrary or repeated layer paths are rejected. An 80-character `bundle_id` must record and load normally with the deterministic limited artifact name below, while a valid short `bundle_id` equal to the old 61-hex digest alias can coexist in the same run. Add signature/IHDR pre-decode RED cases for encoded data over 64 MiB, grayscale, alpha, malformed IHDR and all dimension/pixel-limit violations.

- [ ] **Step 3: Run and verify result/evidence APIs are missing**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_quality/test_results.py `
  tests/test_vision_quality/test_evidence.py -q
```

Expected: RED from import failures for the new modules, or from the new evidence-contract assertions if the modules already exist. The run must report `0 skipped`; skipped contract cases are not an acceptable RED result.

- [ ] **Step 4: Implement immutable bundles and public evidence calls**

Add to `models.py`:

```python
from collections.abc import Mapping
from dataclasses import field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VisionImageLayer:
    layer_id: str
    title: str
    image_bgr: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class VisionResultBundle:
    schema_version: int
    bundle_id: str
    experiment_id: str
    source_snapshot_id: str
    status: str
    layers: tuple[VisionImageLayer, ...]
    result: Mapping[str, Any]
    profile: Mapping[str, Any]
    hardware_status: str = "PENDING_HARDWARE"
```

`VisionImageLayer` and `VisionResultBundle` are public immutable models and must be re-exported from `vision_platform.vision_quality`. Their own construction path, including `dataclasses.replace()`, must enforce all invariants; they are not passive containers whose safety depends on `make_result_bundle()`. In `models.py`, defensively copy every image into model-owned read-only storage and recursively copy then freeze result/profile mappings and sequences. Reject an invalid `schema_version`, identifier, status, layer collection, title, image, result, profile or `hardware_status`: require `type(schema_version) is int` and `schema_version == 1`; exact-string fields and mapping keys that are UTF-8 encodable with no surrogate code points; non-empty ordered layers with unique valid IDs; non-empty UTF-8 titles; non-empty HxWx3 `uint8` BGR images; JSON-native finite result/profile values; an allowed status; and the exact hardware boundary below. Caller mutation after construction must not affect the model.

Expose public `VisionPlatformError` subclasses for bundle/evidence validation with stable `.code` values `VISION_RESULT_BUNDLE_INVALID` and `VISION_RESULT_EVIDENCE_INVALID`. Preserve the original exception with `raise ... from exc`; `load_recorded_bundle()` must translate any nested bundle-model failure to the evidence exception/code rather than leak the bundle code. `make_result_bundle()` must delegate to the same model-owned normalization/invariants instead of maintaining a weaker parallel path. Accept statuses `PASS`, `PARTIAL`, `NO_TARGETS`, `REJECTED`; for V1-01 use `PASS`. Use ASCII patterns:

```python
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_LAYER_ID = re.compile(r"[a-z][a-z0-9_-]{0,39}\Z")
```

Do not change the public limits: `_ID` remains 80 ASCII characters, `_LAYER_ID` remains 40 ASCII characters, and `StudentRunEvidence` snapshot IDs remain capped at 80 characters. Validate `hardware_status` with `type(hardware_status) is str` and require it to equal `PENDING_HARDWARE` exactly.

`result_bundle_to_dict()` must require a record for every layer before serialization and produce only JSON-native values.

In `evidence.py`, use only:

```python
record = evidence.record_snapshot(...)
artifact = evidence.record_json_artifact(name, payload)
```

Before any write, `record_vision_bundle()` must reconstruct and fully normalize a fresh bundle through the model-owned invariant path rather than trust the incoming instance, encode every layer, and validate all encoded PNGs. If `existing_layer_records` contains `raw`, locate the normalized bundle's `raw` layer by `layer_id`, independently of layer order, and require it to exist. Derive and validate the existing record's snapshot ID from `frames/{snapshot_id}.png`, require it to equal `bundle.source_snapshot_id`, and verify that its SHA matches the pre-encoded raw layer and its recorded dimensions first match that PNG's IHDR. Supplying a raw record for a bundle without a raw layer, or failing any raw validation, must fail before any write.

For every layer that will be appended with `record_snapshot()`, form the full logical snapshot ID as `{bundle_id}-{layer_id}`. Use that ID unchanged when its length is at most 80 characters. When it is longer, use the deterministic ASCII limited ID `vision-` plus the lowercase hexadecimal SHA-256 digest of the full logical ID (71 characters total). Precompute the complete set of selected snapshot IDs before any write; when raw is reused, include its validated snapshot ID, which equals `bundle.source_snapshot_id`, in that set. Reject an appended-layer ID that collides with the reused raw ID or another appended-layer ID before performing any write. These rules must not relax the public identifier limits above.

Keep the public `StudentRunEvidence` artifact-stem limit at 75 characters. Use the ordinary artifact name `vision-bundle-{bundle_id}.json` when its stem is at most 75 characters, which permits `bundle_id` lengths through 61. For a longer valid `bundle_id`, use `vision-bundle-_` plus `hashlib.sha256(bundle_id.encode("ascii")).hexdigest()[:60]` plus `.json`; the stem is exactly 75 characters and the filename exactly 80. Because `_` cannot begin a valid `bundle_id`, this long-ID namespace cannot alias an ordinary short bundle artifact and still matches the Task 9/11 `vision-bundle-*.json` discovery contract.

Complete one deterministic preflight before the first `record_snapshot()` call: precompute every selected snapshot ID and artifact name, predicted layer record and final payload; actually serialize the final payload with finite-value checks to UTF-8 JSON bytes; resolve every to-be-created snapshot target and the artifact target under the real `evidence.directory`; and require all those targets not to exist. Only the existing validated raw source is excluded from create-target non-existence checks. Deterministic validation, collision, target-exists and serialization failures must leave no new PNG. Keep `record_snapshot()` and `record_json_artifact()` as the only write calls; do not extend `StudentRunEvidence` with a transaction API or claim multi-file atomicity under concurrent writers or mid-write I/O failure. Write JSON last so those unavoidable failures cannot leave a falsely complete bundle artifact.

`load_recorded_bundle(run_directory, artifact_name)` must strictly derive and match the ordinary or `_`-prefixed deterministic artifact name from the validated payload `bundle_id`; do not accept the old unprefixed long-ID hash alias, a stem longer than 75 characters, a filename longer than 80 characters or a `bundle_id` beyond the public `_ID` limit. Bind each non-reused layer path exactly to `frames/{_appended_snapshot_id(bundle_id, layer_id)}.png`. Bind `raw` only to `frames/{source_snapshot_id}.png` when reused or to its deterministic appended-layer path; reject arbitrary paths and any path repeated by two layers. Resolve every accepted layer path beneath the run directory, verify SHA-256 and return a copied JSON mapping plus resolved layer paths in a separate private field for UI use.

Use one PNG header validator for pre-encoded recording layers and loaded bytes. Before any `cv2` decode, require encoded size at most 64 MiB, the PNG signature `b"\x89PNG\r\n\x1a\n"`, an `IHDR` first chunk with length exactly 13, positive width/height each at most 4096, at most 16,777,216 pixels, bit depth 8, truecolor RGB color type 2, and compression/filter/interlace methods all 0. Require recorded width/height to match IHDR before decoding. Reject JPEG, grayscale, alpha, oversized or malformed bytes before `cv2` is called. Do not allow absolute paths, `..`, symlinks escaping the run, NaN or duplicate layer IDs.

- [ ] **Step 5: Run evidence, existing evidence and security regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_quality/test_results.py `
  tests/test_vision_quality/test_evidence.py `
  tests/test_student_programs/test_evidence.py -q
```

Expected: all selected tests pass with `0 skipped`, including public-model direct construction/replace, defensive copying and recursive freezing, UTF-8/surrogate rejection, stable chained error codes, full write-free record preflight, strict path binding, `_`-separated long-artifact coexistence, PNG IHDR/resource limits before decode, layer-order-independent exact raw-record correspondence, selected-ID collisions with reused raw, 80-character bundle record/load, public-limit and strict `hardware_status` regressions.

- [ ] **Step 6: Commit Task 5**

```powershell
git add vision_platform/vision_quality tests/test_vision_quality
git commit -m "feat(vision-quality): record reusable vision result bundles"
```

## Task 6: Gateway, capability and cleanup integration

**Files:**

- Modify: `vision_platform/experiments/capabilities.py`
- Modify: `vision_platform/student/experiment_gateway.py`
- Modify: `vision_platform/student/runner.py`
- Modify: `vision_platform/vision_quality/catalog.py`
- Modify: `vision_platform/vision_quality/controller.py`
- Modify: `tests/test_experiments/test_capabilities.py`
- Modify: `tests/test_student_programs/test_experiment_gateway.py`
- Modify: `tests/test_student_programs/test_runner.py`
- Modify: `tests/test_vision_quality/test_catalog.py`

- [ ] **Step 1: Write capability and gateway failures first**

Add to `tests/test_experiments/test_capabilities.py`:

```python
def test_sim_camera_and_scene_expose_profile_capabilities():
    application = _application(backend="sim")
    application.config.camera_backend = "sim"
    report = check_capabilities(
        application,
        ("camera.profile", "lighting.profile"),
    )
    assert report.ready


def test_replay_or_missing_sim_rejects_profile_capabilities():
    application = _application(sim=False)
    application.config.camera_backend = "replay"
    report = check_capabilities(application, ("camera.profile", "lighting.profile"))
    assert report.missing == ("camera.profile", "lighting.profile")
```

Add `0 skipped` RED cases proving `camera.profile` and `lighting.profile` are an indivisible declaration: either name requested alone is missing with a stable reason, while the pair is ready from configuration and presence checks only. Use sentinel `sim` and camera objects whose scene/Remote API methods raise if called, proving capability checking does not resolve handles, inspect a concrete camera type or touch the Remote API.

Add to `tests/test_vision_quality/test_catalog.py` `0 skipped` RED cases for `load_profile_catalog_bytes(content)`: valid UTF-8 bytes produce the same immutable catalog as the path loader; invalid UTF-8, duplicate object keys and non-finite constants are rejected by the same strict parser; and `load_profile_catalog(path)` calls `read_bytes()` exactly once before delegating.

Add to `tests/test_student_programs/test_experiment_gateway.py` a `FakeProfileController` with `current/apply/reset` counters. Extend the existing `FakeEvidence` with a faithful `record_json_artifact(name, payload)` recorder so the new bundle path is tested without bypassing the public evidence API. Construct `StudentExperimentGateway(..., profile_controller=fake)` and test:

```python
def test_gateway_dispatches_only_allowed_profile_ids_and_records_profile_capture(tmp_path):
    context, definition, manifest = _bundle(
        tmp_path,
        experiment_id="V1-01",
        public_parameters={
            "baseline_profile_id": "standard",
            "allowed_profile_ids": ["standard", "wide_dim", "detail_bright"],
        },
    )
    definition = replace(
        definition,
        capabilities=("camera.rgb", "camera.profile", "lighting.profile", "scene.probe"),
    )
    controller = FakeProfileController()
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(),
        context=context,
        definition=definition,
        scene_manifest=manifest,
        profile_controller=controller,
    )

    applied = gateway.dispatch("camera.profile.apply", {"profile_id": "wide_dim"})
    captured = gateway.dispatch("camera.capture", {})
    reset = gateway.dispatch("camera.profile.reset", {})

    assert applied["profile_id"] == "wide_dim"
    assert captured["vision_bundle_path"].startswith("vision-bundle-")
    assert reset["profile_id"] == "standard"
```

Add rejection tests for extra arguments, unallowed IDs, profile commands in R1 experiments, controller absence and non-JSON controller output. Add `0 skipped` RED cases proving the factory parses the exact bytes whose manifest hash it verified even if the catalog path is replaced immediately after `read_bytes()`; `public_parameters.baseline_profile_id` must be an exact string equal to `catalog.baseline_profile_id`; and missing, non-string or mismatched baselines reject the binding. Exercise the production controller error path through the gateway so its stable `.code` is not rewritten to a generic student error.

Treat the recorded raw snapshot as independent failure evidence. Add a `0 skipped` RED case that makes bundle JSON recording fail after `record_snapshot()`: the `camera.capture` command must fail, the raw snapshot and its metadata must remain, and no `vision-bundle-*.json`, synthetic `vision_bundle_path` or otherwise falsely complete bundle may remain.

- [ ] **Step 2: Add runner tests for cleanup ordering and failure**

In `tests/test_student_programs/test_runner.py`, extend the fake gateway and cleanup trace:

```python
def test_terminal_cleanup_resets_vision_before_tool_and_robot(tmp_path):
    controller, trace = _controller_with_cleanup_trace(tmp_path)
    controller._experiment_gateway = SimpleNamespace(
        reset_environment=lambda: trace.append("vision.reset")
    )
    controller._application.tool.off = lambda: trace.append("tool.off")
    controller._read_pose = lambda: trace.append("robot.pose") or (0.0, 0.0, 100.0)

    errors = controller._cleanup()

    assert errors == []
    assert trace.index("vision.reset") < trace.index("tool.off") < trace.index("robot.pose")
```

Test that reset failure is included as stage `vision.profile.reset`, changes a would-be PASS to FAILED and preserves `PENDING_HARDWARE`.

Add `0 skipped` startup RED coverage with a `BlockingSim.getObject`: after the client timeout is bounded, controller/gateway construction must run through the existing bounded backend-action path at stage `vision.profile.controller`; timeout or transport failure quarantines the connection, completes the run as FAILED and returns from the caller's `start()` thread within the test deadline.

Add `0 skipped` production-exception integration coverage using the real `VisionProfileController`, gateway and runner rather than a synthetic `RuntimeError`. Force an apply `BaseException` together with rollback setter/readback failures and prove the runner recognizes the structured rollback marker through direct attributes, notes, `BaseExceptionGroup`, `__cause__` and `__context__`, quarantines the backend and does not run later backend cleanup. Parameterize ordinary controller failures to prove `VISION_PROFILE_ID_INVALID`, `VISION_PROFILE_NOT_ALLOWED`, `VISION_PROFILE_APPLY_FAILED`, `VISION_PROFILE_RESET_FAILED` and the other existing `VISION_PROFILE_*` codes survive as student error `.code` values.

- [ ] **Step 3: Run and verify missing integration**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_experiments/test_capabilities.py `
  tests/test_student_programs/test_experiment_gateway.py `
  tests/test_student_programs/test_runner.py `
  tests/test_vision_quality/test_catalog.py -q
```

Expected: new capability, byte-binding, dispatch, startup, rollback and cleanup assertions fail with `0 skipped`.

- [ ] **Step 4: Implement profile controller construction and dispatch**

In `capabilities.py`, add both names to `_KNOWN`. Materialize the requested capabilities before evaluating them and require `camera.profile` and `lighting.profile` to be declared together. If only one is present, report that declared member missing with a stable paired-capability reason. Mark the complete pair available only when:

```python
application.config.camera_backend == "sim"
and application.sim is not None
and application.camera is not None
```

Use stable missing reasons and pure data/presence checks. Do not call scene objects or any Remote API method, and do not require a concrete camera class, sensor handle or light handle here; the bounded startup controller gate owns those checks.

In `catalog.py`, add a bytes entry point and make both module-level loaders share one strict parser and validation path:

```python
def load_profile_catalog_bytes(content: bytes) -> VisionProfileCatalog:
    if type(content) is not bytes:
        raise TypeError("profile catalog content must be bytes")
    payload = json.loads(
        content.decode("utf-8", errors="strict"),
        parse_constant=_reject_nonfinite,
        object_pairs_hook=_reject_duplicate_keys,
    )
    return _catalog_from_payload(payload)


def load_profile_catalog(path: str | Path) -> VisionProfileCatalog:
    selected = Path(path).expanduser().resolve()
    return load_profile_catalog_bytes(selected.read_bytes())
```

Move the existing exact-field, fixed-path, range, duplicate-profile and baseline validation behind `_catalog_from_payload()`. The path loader must perform exactly one `read_bytes()` and delegate; it must not decode or reopen independently. Invalid UTF-8, duplicate JSON keys and non-finite constants therefore have identical behavior for path and bytes callers.

In `controller.py`, add a `VisionPlatformError`-compatible profile exception type (and narrower subclasses if useful) so every public controller/factory failure has a stable `.code`. Replace string-only `RuntimeError`/`ValueError` signaling without renaming the existing codes, including `VISION_PROFILE_CONTEXT_REQUIRED`, `VISION_PROFILE_BACKEND_UNAVAILABLE`, `VISION_PROFILE_CATALOG_INVALID`, `VISION_PROFILE_CATALOG_AMBIGUOUS`, `VISION_PROFILE_OBJECT_MISSING`, `VISION_PROFILE_ID_INVALID`, `VISION_PROFILE_NOT_ALLOWED`, `VISION_PROFILE_READBACK_MISMATCH`, `VISION_PROFILE_APPLY_FAILED`, `VISION_PROFILE_ROLLBACK_FAILED` and `VISION_PROFILE_RESET_FAILED`.

Add the factory using the verified bytes directly:

```python
def controller_for_experiment(application, definition, scene_manifest):
    required = {"camera.profile", "lighting.profile"}
    declared = set(definition.capabilities)
    if not required & declared:
        return None
    if not required <= declared:
        raise VisionProfileError(
            "VISION_PROFILE_CONTEXT_REQUIRED",
            "vision profile capabilities must be declared together",
        )
    if (
        application.config.camera_backend != "sim"
        or application.sim is None
        or application.camera is None
    ):
        raise VisionProfileError(
            "VISION_PROFILE_BACKEND_UNAVAILABLE",
            "vision profile backend is unavailable",
        )
    profile_path = definition.scene.parent / "profiles.json"
    entry = scene_manifest.get("profile_catalog")
    if not isinstance(entry, Mapping):
        raise VisionProfileError(
            "VISION_PROFILE_CONTEXT_REQUIRED",
            "vision profile catalog binding is required",
        )
    expected_relative = profile_path.relative_to(definition.scene.parents[2]).as_posix()
    if entry.get("path") != expected_relative:
        raise VisionProfileError(
            "VISION_PROFILE_CONTEXT_REQUIRED",
            "vision profile catalog path is invalid",
        )
    content = profile_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry.get("sha256"):
        raise VisionProfileError(
            "VISION_PROFILE_CONTEXT_REQUIRED",
            "vision profile catalog digest is invalid",
        )
    catalog = load_profile_catalog_bytes(content)
    baseline = definition.public_parameters.get("baseline_profile_id")
    if type(baseline) is not str or baseline != catalog.baseline_profile_id:
        raise VisionProfileError(
            "VISION_PROFILE_CONTEXT_REQUIRED",
            "baseline_profile_id does not match the verified catalog",
        )
    allowed = definition.public_parameters.get("allowed_profile_ids")
    return VisionProfileController(
        sim=application.sim,
        camera=application.camera,
        catalog=catalog,
        allowed_profile_ids=tuple(allowed),
    )
```

Do not assume `parents[2]` silently: verify it is the project root by checking `config/experiments` and `simulation` children; otherwise reject the binding. Hash `content` and pass those same bytes to `load_profile_catalog_bytes()`; never call the path loader or otherwise reopen the verified file.

Every rollback-failure escape branch, including one that re-raises `KeyboardInterrupt`, `SystemExit` or another `BaseException`, must attach the exact structured attribute `vision_profile_rollback_failure = "VISION_PROFILE_ROLLBACK_FAILED"`. Preserve the original exception identity where control-flow semantics require it, retain diagnostic notes, and keep all rollback errors in the nested `BaseExceptionGroup`/cause graph. For ordinary exceptions, raise the stable `VisionPlatformError`-compatible `VISION_PROFILE_ROLLBACK_FAILED` error with the same marker. The runner detector must walk the structured marker, `.code`, notes, `__cause__`, `__context__` and nested exception groups; message-string matching may remain only as backward compatibility, not the primary contract. `_exception_error()` and gateway/runner boundaries must preserve profile `.code` and structured details instead of replacing them with generic codes.

In `StudentExperimentGateway.__init__`, accept optional injected `profile_controller`; otherwise call `controller_for_experiment`. Change `dispatch()` to validate arguments per command:

```python
if name == "camera.profile.get":
    _require_exact_args(args, ())
    return self._profiles_required().current().to_public_dict()
if name == "camera.profile.apply":
    _require_exact_args(args, ("profile_id",))
    return self._profiles_required().apply(args["profile_id"]).to_public_dict()
if name == "camera.profile.reset":
    _require_exact_args(args, ())
    return self._profiles_required().reset().to_public_dict()
```

In `_capture()`, when a profile controller exists:

1. obtain `current().to_public_dict()` before capture;
2. add it to snapshot metadata as `vision_profile`;
3. create a one-layer `VisionResultBundle` using the already recorded raw snapshot;
4. call `record_vision_bundle()` and return `vision_bundle_path`;
5. verify captured width/height equal the profile resolution.

Record the raw snapshot before constructing or recording its bundle. If bundle construction or JSON recording fails, propagate the command failure and retain that raw snapshot as independent failure evidence; do not delete it, return success, manufacture `vision_bundle_path` or leave a `vision-bundle-*.json` artifact. JSON remains the last bundle write, so an artifact on disk always denotes a complete bundle.

For experiments without a controller, preserve the exact previous return and evidence fields.

Add `reset_environment()`:

```python
def reset_environment(self) -> dict[str, object] | None:
    if self._profile_controller is None:
        return None
    return self._profile_controller.reset().to_public_dict()
```

- [ ] **Step 5: Route commands and invoke cleanup in the runner**

In `StudentProgramController.start()`, call `_bound_client_timeout()` before any gateway/controller construction can resolve a scene object. Construct `StudentExperimentGateway` (and therefore `controller_for_experiment`) through the existing `_dispatch_backend_action_bounded(stage="vision.profile.controller", ...)`. An `_BackendActionStuck` or transport failure must quarantine the backend, set `_starting` false, complete the run as FAILED and return promptly from `start()`; ordinary validation/controller errors must also complete and return while preserving any `VisionPlatformError.code`. Reuse the existing backend-action deadline, registration and quarantine machinery; do not add another worker-thread architecture.

In `StudentProgramController._dispatch`, include all three profile command names in the gateway-only set. Detect the shared structured rollback marker across the full exception graph for production command failures. In `_cleanup()`, call the gateway reset through the existing bounded backend-action mechanism before `tool.off`; record stage `vision.profile.reset`. A timeout or rollback failure must quarantine the backend using the same policy as other stuck CoppeliaSim actions.

Do not add profile commands to the robot motion guard; they remain experiment-gateway commands.

- [ ] **Step 6: Run gateway, runner and full student regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_experiments/test_capabilities.py `
  tests/test_student_programs/test_experiment_gateway.py `
  tests/test_student_programs/test_runner.py `
  tests/test_vision_quality/test_catalog.py `
  tests/test_student_programs -q
```

Expected: all selected tests pass with `0 skipped`. Record exact pass and skip counts. If a separate full static suite skips live CoppeliaSim cases, report those skips as unexecuted live coverage; skipped live tests are not PASS and do not satisfy this Task's zero-skip focused gate.

- [ ] **Step 7: Commit Task 6**

```powershell
git add vision_platform/experiments/capabilities.py `
  vision_platform/student/experiment_gateway.py vision_platform/student/runner.py `
  vision_platform/vision_quality/catalog.py `
  vision_platform/vision_quality/controller.py tests/test_experiments `
  tests/test_student_programs tests/test_vision_quality/test_catalog.py
git commit -m "feat(student): guard V1-01 profile execution"
```

## Task 7: V1-01 catalog definition, guide and student template

**Files:**

- Create: `config/experiments/V1-01.json`
- Create: `docs/experiments/V1-01.md`
- Create: `student_programs/templates/v1_01_virtual_vision.py`
- Create: `tests/test_vision_quality/test_v1_01_materials.py`

- [ ] **Step 1: Write formal-material tests**

Create `tests/test_vision_quality/test_v1_01_materials.py`:

```python
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v1_01_definition_binds_profiles_scene_and_pending_boundaries():
    payload = json.loads(
        (ROOT / "config" / "experiments" / "V1-01.json").read_text(encoding="utf-8")
    )
    assert payload["experiment_id"] == "V1-01"
    assert payload["pack_id"] == "V1"
    assert payload["scene"] == "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt"
    assert payload["scene_manifest"] == "simulation/vision_quality_lab/scene_manifest.json"
    assert payload["capabilities"] == [
        "camera.rgb", "camera.profile", "lighting.profile",
        "experiment.info", "scene.probe",
    ]
    assert payload["public_parameters"]["baseline_profile_id"] == "standard"
    assert payload["public_parameters"]["allowed_profile_ids"] == [
        "standard", "wide_dim", "detail_bright"
    ]
    assert payload["acceptance"]["probe_kind"] == "vision_profile_observation"
    assert payload["hardware_status"] == "PENDING_HARDWARE"


def test_v1_01_guide_contains_learning_and_acceptance_boundaries():
    text = (ROOT / "docs" / "experiments" / "V1-01.md").read_text(encoding="utf-8")
    for phrase in (
        "分辨率", "视场角", "相机高度", "光照", "不是课程成绩",
        "PENDING_HUMAN_ACCEPTANCE", "PENDING_HARDWARE",
    ):
        assert phrase in text


def test_v1_01_template_uses_public_sdk_only():
    path = ROOT / "student_programs" / "templates" / "v1_01_virtual_vision.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "apply_profile" in source
    assert "capture" in source
    assert "reset_profile" in source
    forbidden = ("RemoteAPIClient", "getObject", "setObject", "subprocess", "socket")
    assert not any(term in source for term in forbidden)
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in ast.walk(tree))
```

- [ ] **Step 2: Run and verify the three missing materials**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_quality/test_v1_01_materials.py -q
```

Expected: failure at missing `V1-01.json`.

- [ ] **Step 3: Create the strict V1-01 experiment definition**

Create `config/experiments/V1-01.json` with the common schema used by R1 definitions. Use:

```json
{
  "schema_version": 1,
  "experiment_id": "V1-01",
  "pack_id": "V1",
  "title": "虚拟视觉系统认知",
  "version": "2.2.0",
  "scene": "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt",
  "scene_manifest": "simulation/vision_quality_lab/scene_manifest.json",
  "student_template": "student_programs/templates/v1_01_virtual_vision.py",
  "guide": "docs/experiments/V1-01.md",
  "capabilities": ["camera.rgb", "camera.profile", "lighting.profile", "experiment.info", "scene.probe"],
  "workspace": {
    "x_mm": [20, 140],
    "y_mm": [-90, 90],
    "z_mm": [10, 140],
    "safe_z_mm": 100
  },
  "public_parameters": {
    "baseline_profile_id": "standard",
    "allowed_profile_ids": ["standard", "wide_dim", "detail_bright"],
    "comparison_order": ["standard", "wide_dim", "detail_bright"],
    "camera_path": "/VisionQualityLab/CameraRig/Camera"
  },
  "acceptance": {
    "probe_kind": "vision_profile_observation",
    "automated_checks": ["profile_readback", "three_captures", "baseline_reset", "evidence_hashes"],
    "human_checks": ["学生解释分辨率与视场差异", "学生解释光照变化对图像的影响"]
  },
  "hardware_status": "PENDING_HARDWARE"
}
```

Do not append `V1-01` to `config/experiments/catalog.json` in this Task. The catalog loader requires the formal `.ttt` and manifest to exist; registration is intentionally gated until Task 10 publishes and validates both files.

- [ ] **Step 4: Write the guide and executable template**

`docs/experiments/V1-01.md` must contain: objective, safety boundary, three profile table, student steps, expected evidence, questions, automated checks, human checks, and hardware boundary.

Create `student_programs/templates/v1_01_virtual_vision.py`:

```python
def main(ctx):
    info = ctx.experiment.info()
    parameters = info["public_parameters"]
    order = tuple(parameters["comparison_order"])
    allowed = set(parameters["allowed_profile_ids"])
    if set(order) - allowed:
        raise RuntimeError("实验配置档顺序超出公开白名单")

    try:
        for profile_id in order:
            state = ctx.camera.apply_profile(profile_id)
            frame = ctx.camera.capture()
            ctx.log(
                f"{profile_id}: {frame.width}x{frame.height}, "
                f"FOV={state.perspective_angle_deg:.1f} deg, "
                f"camera_z={state.camera_rig_z_m:.2f} m"
            )
            ctx.checkpoint()
    finally:
        ctx.camera.reset_profile()
```

Do not catch and suppress command failures. The runner must retain the real failure and still perform its own cleanup reset.

- [ ] **Step 5: Execute the template against a strict fake public SDK**

In `test_v1_01_materials.py`, load the template with `runpy.run_path`, pass a fake context whose camera implements only `apply_profile`, `capture` and `reset_profile`, and assert the template produces three apply/capture pairs plus a final reset. Validate the three captured widths are 512, 256 and 768. Make a capture fail and assert the template still calls explicit reset while preserving the original exception.

- [ ] **Step 6: Run raw-material and validator tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_quality/test_v1_01_materials.py `
  tests/test_student_programs/test_validator.py -q
```

Expected: all selected tests pass. The formal experiment list still contains only the five released R1 entries at this point.

- [ ] **Step 7: Commit Task 7**

```powershell
git add config/experiments/V1-01.json docs/experiments/V1-01.md `
  student_programs/templates/v1_01_virtual_vision.py `
  tests/test_vision_quality/test_v1_01_materials.py
git commit -m "feat(curriculum): add V1-01 virtual vision lab"
```

## Task 8: Vision-profile scene probe

**Files:**

- Modify: `vision_platform/experiments/probes.py`
- Modify: `tests/test_experiments/test_probes.py`

- [ ] **Step 1: Add probe tests with exact published values**

Append a fake profile-aware sim to `tests/test_experiments/test_probes.py` and test:

```python
def test_vision_profile_probe_reports_exact_standard_state():
    definition = _definition(
        "V1-01",
        "vision_profile_observation",
        {
            "baseline_profile_id": "standard",
            "allowed_profile_ids": ["standard", "wide_dim", "detail_bright"],
            "camera_path": "/VisionQualityLab/CameraRig/Camera",
        },
    )
    report = probe_experiment(
        FakeVisionProfileSim.standard(),
        definition,
        phase="initial",
        scene_manifest={"task_contracts": {}, "profile_catalog": VALID_PROFILE_ENTRY},
    )
    assert report["status"] == "PASS"
    assert report["matched"] == report["expected"]
    assert report["profile_id"] == "standard"
    assert report["hardware_status"] == "PENDING_HARDWARE"
```

Test an unknown parameter combination produces FAIL rather than nearest-profile PASS, missing light paths produce an exception, and both `initial` and `final` require baseline `standard`.

- [ ] **Step 2: Run and verify unsupported probe kind**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_experiments/test_probes.py -q
```

Expected: `unsupported probe_kind: vision_profile_observation`.

- [ ] **Step 3: Implement probe by reusing profile readback semantics**

Add the kind to `_KNOWN_PROBE_KINDS`. Do not instantiate the command controller because the probe is read-only. Add a read-only helper in `controller.py`:

```python
def inspect_published_profile(sim, catalog) -> AppliedVisionProfile:
    return VisionProfileController.inspect(sim=sim, catalog=catalog)
```

The helper resolves only fixed catalog paths and does not require a camera. In `probe_experiment`, for V1-01:

1. verify the manifest profile path/hash;
2. set `project_root = definition.scene.parents[2]`, require `project_root/config/experiments` and `project_root/simulation` to be directories, then load exactly `definition.scene.parent / "profiles.json"`;
3. read the current profile;
4. require `profile_id == baseline_profile_id` for initial and final phases;
5. return rows for sensor, camera rig, key light and fill light plus public profile fields.

Keep the existing function signature and call sites unchanged. Derive the profile file only from the already validated `definition.scene` and manifest digest; do not accept a student-provided path or new caller path argument.

- [ ] **Step 4: Run probe and existing experiment regressions**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_experiments/test_probes.py `
  tests/test_student_programs/test_experiment_gateway.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 8**

```powershell
git add vision_platform/experiments/probes.py `
  vision_platform/vision_quality/controller.py tests/test_experiments/test_probes.py
git commit -m "feat(experiments): probe V1-01 visual profiles"
```

## Task 9: Read-only PyQt vision-result panel

**Files:**

- Create: `vision_platform/ui/vision_result_panel.py`
- Modify: `vision_platform/ui/pyqt_app.py`
- Create: `tests/test_vision_platform/test_vision_result_panel.py`
- Modify: `tests/test_vision_platform/test_pyqt_smoke.py`

- [ ] **Step 1: Write panel tests for empty, valid and hostile evidence**

Create `tests/test_vision_platform/test_vision_result_panel.py`:

```python
from __future__ import annotations

import cv2
import numpy as np

from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.ui.vision_result_panel import VisionResultPanel
from vision_platform.vision_quality.evidence import record_vision_bundle
from vision_platform.vision_quality.results import make_result_bundle


def test_panel_has_stable_empty_state_and_pending_boundaries(qtbot):
    panel = VisionResultPanel()
    qtbot.addWidget(panel)
    assert "尚无视觉结果" in panel.status_label.text()
    assert panel.layer_combo.count() == 0
    assert "PENDING_HARDWARE" in panel.boundary_label.text()
    assert "不是课程成绩" in panel.boundary_label.text()


def test_panel_loads_latest_bundle_and_switches_layers(qtbot, tmp_path):
    run = _record_two_layer_bundle(tmp_path)
    panel = VisionResultPanel()
    qtbot.addWidget(panel)

    panel.load_run(run)

    assert panel.layer_combo.count() == 2
    assert panel.layer_combo.itemData(0) == "raw"
    assert "standard" in panel.profile_label.text()
    assert '"experiment_id": "V1-01"' in panel.result_text.toPlainText()
    assert not panel.preview_label.pixmap().isNull()
    panel.layer_combo.setCurrentIndex(1)
    assert panel.layer_combo.currentData() == "annotated"


def test_panel_contains_invalid_bundle_error(qtbot, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "vision-bundle-frame-000001.json").write_text("{}", encoding="utf-8")
    panel = VisionResultPanel()
    qtbot.addWidget(panel)
    panel.load_run(run)
    assert "VISION_RESULT_EVIDENCE_INVALID" in panel.status_label.text()
    assert panel.layer_combo.count() == 0
```

The helper `_record_two_layer_bundle()` must use the real evidence adapter from Task 5, not write a hand-crafted success file.

- [ ] **Step 2: Run and verify missing panel**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_vision_result_panel.py -q
```

Expected: import failure for `VisionResultPanel`.

- [ ] **Step 3: Implement a read-only evidence viewer**

Create `vision_platform/ui/vision_result_panel.py` with:

```python
class VisionResultPanel(QWidget):
    def __init__(self, *, controller=None, parent=None): ...
    def load_run(self, run_directory: str | Path) -> None: ...
    def clear_result(self, message: str = "尚无视觉结果") -> None: ...
    def release_subscription(self) -> None: ...
```

UI fields must be public for smoke tests:

```text
status_label
profile_label
metrics_label
layer_combo
preview_label
result_text
boundary_label
```

`load_run()` finds only filenames matching `vision-bundle-*.json` directly under the supplied evidence directory, selects the lexicographically latest, and calls `load_recorded_bundle()`. It must not recursively scan arbitrary paths. Render BGR via `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` into a detached `QImage.copy()` before creating `QPixmap`.

If `controller` is provided, subscribe to `StudentRunSnapshot`. When state enters PASSED, FAILED or CANCELLED and `evidence_dir` is present, load the latest bundle; terminal runs without a bundle show `本次运行没有视觉结果包`. Catch all callback errors and render a stable UI error instead of raising into the runner thread.

- [ ] **Step 4: Add the tab without changing existing student controls**

In `VisionLabWindow`:

1. initialize `_vision_result_unsubscribe = None`;
2. after successfully creating `StudentProgramController`, create `VisionResultPanel(controller=controller)`;
3. add tabs in this order: experiment catalog, calibration, recognition, acceptance, student programming, vision result;
4. store `self.vision_result_panel`;
5. release the panel subscription during window close before closing the controller;
6. preserve current behavior when student-controller setup fails.

Do not let the result panel initiate profile changes or run student code.

- [ ] **Step 5: Run panel and full PyQt smoke tests**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_vision_result_panel.py `
  tests/test_vision_platform/test_pyqt_smoke.py `
  tests/test_vision_platform/test_application_close.py -q
```

Expected: all selected tests pass; no Qt thread remains after close tests.

- [ ] **Step 6: Commit Task 9**

```powershell
git add vision_platform/ui/vision_result_panel.py vision_platform/ui/pyqt_app.py `
  tests/test_vision_platform
git commit -m "feat(ui): display reusable vision result evidence"
```

## Task 10: Build and publish the independent CoppeliaSim scene

**Files:**

- Modify: `simulation/training_scenes/build_scene.py`
- Create after verified online build: `simulation/vision_quality_lab/BL23_vision_quality_lab.ttt`
- Create after verified online build: `simulation/vision_quality_lab/scene_manifest.json`
- Modify: `config/experiments/catalog.json`
- Modify: `tools/vision_lab/run_experiment.ps1`
- Modify: `tests/test_simulation/test_formal_training_scenes.py`
- Modify: `tests/test_experiments/test_formal_catalog.py`
- Modify: `tests/test_experiments/test_student_templates.py`
- Modify: `tests/test_experiments/test_cli.py`
- Modify: `tests/test_acceptance/test_delivery_contract.py`
- Create: `tests/test_acceptance/test_coppeliasim_vision_quality_scene.py`

- [ ] **Step 1: Add static builder tests before scene generation**

Extend `tests/test_simulation/test_formal_training_scenes.py` with fakes that assert `_build_vision_quality()` creates exactly the required aliases beneath `/VisionQualityLab`, uses primitives only, sets the camera path from `profiles.json`, and never modifies `/BLX`, `/BLX_base`, `/BLX_tool_suction` or any existing mesh.

In the fake publisher test, inspect the manifest returned from its temporary build directory and assert it includes:

```python
assert manifest["profile_catalog"] == {
    "path": "simulation/vision_quality_lab/profiles.json",
    "sha256": hashlib.sha256(PROFILES.read_bytes()).hexdigest(),
}
```

Do not append `simulation/vision_quality_lab` to the real-module `SCENES` tuple yet. Before Step 5 there is intentionally no published `.ttt` or `scene_manifest.json`, so the existing two formal-scene regression cases must remain unchanged.

- [ ] **Step 2: Implement original scene objects in the existing safe publisher**

Add small focused helpers to `build_scene.py`:

```python
def _build_resolution_target(sim, spec, parent): ...
def _build_triangle_prism(sim, spec, parent): ...
def _build_vision_lighting(sim, catalog, parent): ...
def _build_vision_quality(sim, spec, root): ...
```

Requirements:

- create `/VisionQualityLab` after removing only `/VisionLab` from the staged template;
- create the inspection board and four stationary samples from primitives;
- create twelve alternating stripe cuboids under a model named `ResolutionTarget`;
- create `CameraRig/Camera` as a perspective vision sensor at the standard profile;
- set near/far clipping from the catalog;
- orient the camera from the exact scene spec and verify all four samples are in the returned frame;
- create key/fill lights with standard profile diffuse values;
- set the inspection board, three sample shapes and twelve stripe shapes to static=true and respondable=false so the robot cannot disturb them;
- call `sim.getObject()` for every required path before saving;
- include the profile catalog path/hash in the manifest;
- retain lock, staged-file, atomic-publish and protected-hash behavior.

Change build dispatch explicitly:

```python
if formal.scene_id == "robot-basics":
    _build_basics(sim, spec, root)
elif formal.scene_id == "logistics-lab":
    _build_logistics(sim, spec, root)
    _attach_logistics_camera_scope(sim, root)
elif formal.scene_id == "vision-quality-lab":
    _build_vision_quality(sim, spec, root)
else:
    raise RuntimeError(f"unsupported formal scene: {formal.scene_id}")
```

- [ ] **Step 3: Run static builder tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_simulation/test_vision_quality_scene_contract.py `
  tests/test_simulation/test_formal_training_scenes.py `
  tests/test_simulation/test_training_scene_contract.py -q
```

Expected: all selected unit/static tests pass before launching CoppeliaSim. Any manifest assertion in this step uses the fake publisher's temporary output, never a nonexistent repository manifest.

- [ ] **Step 4: Write the opt-in online scene test**

Create `tests/test_acceptance/test_coppeliasim_vision_quality_scene.py` using the existing explicit `coppeliasim` marker and endpoint options. Connect only to the prelaunched dedicated port; verify the loaded scene path equals the formal vision-quality scene before changing anything:

```python
@pytest.mark.coppeliasim
def test_vision_quality_scene_required_paths_profiles_and_frames(request):
    host = request.config.getoption("--coppelia-host") or os.environ.get("COPPELIA_HOST") or "127.0.0.1"
    configured_port = request.config.getoption("--coppelia-port")
    port = configured_port if configured_port is not None else int(os.environ.get("COPPELIA_PORT", "23005"))
    definition = ExperimentCatalog.load(
        ROOT / "config/experiments/catalog.json", project_root=ROOT
    ).require("V1-01")
    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    loaded = Path(sim.getStringParam(sim.stringparam_scene_path)).resolve()
    assert loaded == definition.scene.resolve()
    manifest = json.loads(definition.scene_manifest.read_text(encoding="utf-8"))
    report = validate_scene_contract(
        definition.scene.parent / "scene_spec.json",
        definition.scene_manifest,
        project_root=ROOT,
    )
    assert report["status"] == "PASS"
    for path in manifest["required_paths"]:
        sim.getObject(path)
    profile_catalog = load_profile_catalog(
        definition.scene.parent / "profiles.json"
    )
    camera = CoppeliaSimCamera(
        sensor_path=profile_catalog.sensor_path,
        sim=sim,
        client=client,
    )
    application = SimpleNamespace(
        sim=sim,
        camera=camera,
        config=SimpleNamespace(camera_backend="sim"),
    )
    controller = controller_for_experiment(application, definition, manifest)
    try:
        camera.open()
        observed = {}
        for profile_id in ("standard", "wide_dim", "detail_bright"):
            state = controller.apply(profile_id)
            frame = camera.read(timeout_s=5.0)
            observed[profile_id] = (
                frame.width,
                frame.height,
                hashlib.sha256(frame.image_bgr.tobytes()).hexdigest(),
            )
            assert (frame.width, frame.height) == state.resolution
        assert len({item[2] for item in observed.values()}) == 3
    finally:
        controller.reset()
        camera.close()
        client.close()
```

Do not mark the test as passed when the marker is skipped.

- [ ] **Step 5: Launch a dedicated build port and publish the scene**

Use port `23005` so active acceptance ports remain isolated. Launch the protected template through the existing owned-process script, run the builder, and stop only the returned owned process in `finally`:

```powershell
$port = 23005
$scene = 'simulation/vision_lab/BL23_vision_lab.ttt'
$launch = & '.\tools\vision_lab\launch_coppeliasim.ps1' `
  -Scene $scene -Port $port -Hidden
try {
  .\.venv-vision\Scripts\python.exe -m simulation.training_scenes.build_scene `
    --spec simulation/vision_quality_lab/scene_spec.json `
    --host 127.0.0.1 --port $port
  if ($LASTEXITCODE -ne 0) { throw 'vision quality scene build failed' }
} finally {
  . '.\tools\vision_lab\process_ownership.ps1'
  Stop-ExactOwnedProcess `
    -ProcessId $launch.ProcessId `
    -ProcessPath $launch.ProcessPath `
    -ProcessStartTimeUtcTicks $launch.ProcessStartTimeUtcTicks
}
```

Verify no listener remains on `23005`; do not stop by process name.

- [ ] **Step 6: Register V1-01 only after formal scene files exist**

Append `V1-01` to `config/experiments/catalog.json` after the five R1 entries. Extend existing formal-catalog expectations without changing any R1 contract. Add a strict fake camera to `tests/test_experiments/test_student_templates.py` and verify the V1-01 template applies/captures 512, 256 and 768 pixel profiles and resets in `finally`.

Now that both published files exist, append `ROOT / "simulation" / "vision_quality_lab"` to `SCENES` in `tests/test_simulation/test_formal_training_scenes.py` and extend its expected scene-id list with `"vision-quality-lab"`. This is the first step allowed to make the real formal-scene contract load the new manifest.

Extend `tests/test_experiments/test_cli.py`:

```python
def test_experiment_list_and_show_include_v1_01(capsys):
    assert main(["experiment-list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["experiments"][-1]["experiment_id"] == "V1-01"
    assert main(["experiment-show", "--experiment", "V1-01"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["student_template"].endswith("v1_01_virtual_vision.py")
```

Extend the generic `run_experiment.ps1` allowlist:

```powershell
[ValidateSet('R1-01', 'R1-02', 'R1-05', 'R1-06', 'R1-07', 'V1-01')]
```

Do not create a V1-01-specific launcher. Run the registration gate:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_simulation/test_formal_training_scenes.py `
  tests/test_experiments/test_formal_catalog.py `
  tests/test_experiments/test_student_templates.py `
  tests/test_experiments/test_cli.py `
  tests/test_acceptance/test_delivery_contract.py -q
```

Expected: all tests pass and the formal list contains six stable entries.

- [ ] **Step 7: Validate the generated files and run the explicit online test**

```powershell
$port = 23005
$scene = 'simulation/vision_quality_lab/BL23_vision_quality_lab.ttt'
$launch = & '.\tools\vision_lab\launch_coppeliasim.ps1' `
  -Scene $scene -Port $port -Hidden
try {
  .\.venv-vision\Scripts\python.exe -m simulation.training_scenes.verify_scene `
    --spec simulation/vision_quality_lab/scene_spec.json
  if ($LASTEXITCODE -ne 0) { throw 'vision quality scene verification failed' }
  .\.venv-vision\Scripts\python.exe -m pytest `
    tests/test_acceptance/test_coppeliasim_vision_quality_scene.py `
    -m coppeliasim -q `
    --coppelia-host 127.0.0.1 --coppelia-port $port
  if ($LASTEXITCODE -ne 0) { throw 'vision quality online test failed' }
} finally {
  . '.\tools\vision_lab\process_ownership.ps1'
  Stop-ExactOwnedProcess `
    -ProcessId $launch.ProcessId `
    -ProcessPath $launch.ProcessPath `
    -ProcessStartTimeUtcTicks $launch.ProcessStartTimeUtcTicks
}
if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
  throw "CoppeliaSim listener remains on port $port"
}
```

Expected: scene contract PASS and online test `1 passed, 0 skipped`. Record the exact observed count; do not use the expected text if the run differs.

- [ ] **Step 8: Recheck protected assets before commit**

```powershell
git diff --exit-code v2.2.0 -- `
  simulation/vision_lab/BL23_vision_lab.ttt `
  simulation/vision_lab/robot_assets_manifest.json `
  simulation/vision_lab/robot_assets
$protected = git diff --name-only v2.2.0 -- | Select-String -Pattern '\.(urdf|stl)$'
if ($protected) { $protected; throw 'Protected URDF/STL changed' }
```

Expected: no protected changes.

- [ ] **Step 9: Commit Task 10**

```powershell
git add simulation/training_scenes/build_scene.py `
  simulation/vision_quality_lab/BL23_vision_quality_lab.ttt `
  simulation/vision_quality_lab/scene_manifest.json `
  config/experiments/catalog.json tools/vision_lab/run_experiment.ps1 `
  tests/test_simulation/test_formal_training_scenes.py `
  tests/test_experiments tests/test_acceptance/test_delivery_contract.py `
  tests/test_acceptance/test_coppeliasim_vision_quality_scene.py
git commit -m "feat(simulation): publish vision quality lab scene"
```

## Task 11: End-to-end V1-01 online acceptance and UI evidence

**Files:**

- Create: `tests/test_acceptance/test_coppeliasim_v1_01.py`
- Create: `tools/vision_lab/run_vision_quality_acceptance.ps1`
- Modify: `tests/test_acceptance/test_powershell_process_ownership.py`
- Modify: `tests/test_vision_platform/test_pyqt_smoke.py`

- [ ] **Step 1: Write the end-to-end online test before the acceptance wrapper**

Create `tests/test_acceptance/test_coppeliasim_v1_01.py` with the explicit marker and existing endpoint options. Run the formal template through the public `experiment-run` CLI, which constructs `StudentProgramController`, and assert:

```python
@pytest.mark.coppeliasim
def test_v1_01_student_template_records_three_profiles_and_resets(tmp_path, request):
    host = request.config.getoption("--coppelia-host") or os.environ.get("COPPELIA_HOST") or "127.0.0.1"
    configured_port = request.config.getoption("--coppelia-port")
    port = configured_port if configured_port is not None else int(os.environ.get("COPPELIA_PORT", "23005"))
    output_root = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable, "-m", "vision_platform.cli", "experiment-run",
            "--experiment", "V1-01", "--host", host, "--port", str(port),
            "--output", str(output_root),
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["status"] == "PASS"
    summary_path = Path(payload["summary"]).resolve()
    evidence_dir = Path(payload["evidence"]).resolve()
    assert output_root.resolve() in evidence_dir.parents
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    commands = [
        json.loads(line)
        for line in (evidence_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    bundles = sorted(evidence_dir.glob("vision-bundle-*.json"))
    assert [item["args"].get("profile_id") for item in commands if item["name"] == "camera.profile.apply"] == [
        "standard", "wide_dim", "detail_bright"
    ]
    assert sum(item["name"] == "camera.capture" for item in commands) == 3
    assert commands[-1]["name"] == "camera.profile.reset"
    assert len(bundles) == 3
    assert summary["scene_probe_status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
```

Also read the final scene probe artifact and assert `profile_id == standard`. Every bundle must load through `load_recorded_bundle()` and contain one `raw` layer with dimensions matching its recorded profile.

- [ ] **Step 2: Run the test once to verify its real pre-wrapper behavior**

```powershell
$port = 23005
$scene = 'simulation/vision_quality_lab/BL23_vision_quality_lab.ttt'
$launch = & '.\tools\vision_lab\launch_coppeliasim.ps1' `
  -Scene $scene -Port $port -Hidden
try {
  .\.venv-vision\Scripts\python.exe -m pytest `
    tests/test_acceptance/test_coppeliasim_v1_01.py `
    -m coppeliasim -q `
    --coppelia-host 127.0.0.1 --coppelia-port $port
  if ($LASTEXITCODE -ne 0) { throw 'V1-01 pre-wrapper online test failed' }
} finally {
  . '.\tools\vision_lab\process_ownership.ps1'
  Stop-ExactOwnedProcess `
    -ProcessId $launch.ProcessId `
    -ProcessPath $launch.ProcessPath `
    -ProcessStartTimeUtcTicks $launch.ProcessStartTimeUtcTicks
}
if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
  throw "CoppeliaSim listener remains on port $port"
}
```

Expected after Tasks 1–10: test passes online with zero skips. If it fails, use systematic debugging; do not weaken profile, reset, evidence or probe assertions.

- [ ] **Step 3: Add a dedicated owned-process acceptance wrapper**

Create `tools/vision_lab/run_vision_quality_acceptance.ps1` modeled on `run_acceptance.ps1`, but limited to this branch. It must:

- accept output directory, Coppelia root, host and port (default `23005`);
- refuse an already occupied port;
- launch only `BL23_vision_quality_lab.ttt` with captured identity;
- run both online test files with JUnit XML;
- run `vision_platform.cli experiment-run --experiment V1-01` with the formal template;
- assert JUnit contains zero failures, errors and skips;
- write `acceptance-summary.json` atomically after cleanup status is known;
- stop only the exact owned process in `finally`;
- preserve and restore caller `QT_QPA_PLATFORM`;
- write `hardware_status=PENDING_HARDWARE` and `teaching_effect=PENDING_HUMAN_ACCEPTANCE`;
- return nonzero on cleanup failure.

Do not delete unrelated files under the output directory; only replace the wrapper-owned summary/JUnit names.

- [ ] **Step 4: Add PowerShell ownership and failure-order tests**

Extend `tests/test_acceptance/test_powershell_process_ownership.py` to assert the new wrapper:

```text
captures process path, PID and UTC start ticks
uses Stop-ExactOwnedProcess
performs cleanup before final summary status
marks any online skip as FAIL
restores location and QT_QPA_PLATFORM
does not use Stop-Process by name
does not claim hardware PASS
```

- [ ] **Step 5: Capture PyQt evidence at 100% and 125%**

Extend the existing UI screenshot harness to load a real recorded V1-01 result bundle and save:

```text
artifacts/vision_lab/v2-2-c0-v1-01/ui-100/vision-result.png
artifacts/vision_lab/v2-2-c0-v1-01/ui-125/vision-result.png
```

Automated assertions must verify nonzero dimensions, visible profile summary, selected raw layer, JSON text, and boundary label. Human review remains pending until a teacher checks the screenshots.

- [ ] **Step 6: Run the complete explicit acceptance wrapper**

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  .\tools\vision_lab\run_vision_quality_acceptance.ps1 `
  -OutputDir artifacts\vision_lab\v2-2-c0-v1-01 `
  -Port 23005
if ($LASTEXITCODE -ne 0) { throw 'V1-01 acceptance failed' }
```

Expected: wrapper summary PASS, online JUnit zero skips, exact owned process stopped, port `23005` free. Report actual test counts from JUnit.

- [ ] **Step 7: Commit Task 11**

Do not add `artifacts/` screenshots or run evidence to Git unless `RETAINED_FILES.txt` explicitly defines them as formal published files; this plan keeps them untracked.

```powershell
git add tests/test_acceptance/test_coppeliasim_v1_01.py `
  tests/test_acceptance/test_powershell_process_ownership.py `
  tests/test_vision_platform/test_pyqt_smoke.py `
  tools/vision_lab/run_vision_quality_acceptance.ps1
git commit -m "test(acceptance): verify V1-01 live workflow"
```

## Task 12: Delivery contract, full regression and clean handoff

**Files:**

- Create: `tests/test_vision_quality/test_delivery.py`
- Modify: `RETAINED_FILES.txt`
- Modify: `docs/superpowers/plans/2026-08-01-vision-quality-platform-v1-01-plan.md` only to mark genuinely completed checkboxes
- Evidence: `artifacts/vision_lab/v2-2-c0-v1-01/`

- [ ] **Step 1: Write the failing delivery whitelist test**

Create `tests/test_vision_quality/test_delivery.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v1_01_delivery_is_complete_retained_and_isolated():
    retained_lines = [
        line.strip()
        for line in (ROOT / "RETAINED_FILES.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    retained = set(retained_lines)
    required = {
        "config/experiments/V1-01.json",
        "docs/experiments/V1-01.md",
        "docs/superpowers/plans/2026-08-01-vision-quality-platform-v1-01-plan.md",
        "docs/superpowers/specs/2026-08-01-vision-quality-platform-v1-01-design.md",
        "simulation/vision_quality_lab/__init__.py",
        "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt",
        "simulation/vision_quality_lab/profiles.json",
        "simulation/vision_quality_lab/scene_manifest.json",
        "simulation/vision_quality_lab/scene_spec.json",
        "student_programs/templates/v1_01_virtual_vision.py",
        "tools/vision_lab/run_vision_quality_acceptance.ps1",
        "vision_platform/ui/vision_result_panel.py",
        "vision_platform/vision_quality/__init__.py",
        "vision_platform/vision_quality/catalog.py",
        "vision_platform/vision_quality/controller.py",
        "vision_platform/vision_quality/evidence.py",
        "vision_platform/vision_quality/models.py",
        "vision_platform/vision_quality/results.py",
        "tests/test_acceptance/test_coppeliasim_v1_01.py",
        "tests/test_acceptance/test_coppeliasim_vision_quality_scene.py",
        "tests/test_simulation/test_vision_quality_scene_contract.py",
        "tests/test_vision_platform/test_vision_result_panel.py",
        "tests/test_vision_quality/__init__.py",
        "tests/test_vision_quality/test_catalog.py",
        "tests/test_vision_quality/test_controller.py",
        "tests/test_vision_quality/test_delivery.py",
        "tests/test_vision_quality/test_evidence.py",
        "tests/test_vision_quality/test_results.py",
        "tests/test_vision_quality/test_v1_01_materials.py",
    }
    assert required <= retained
    assert len(retained_lines) == len(retained)
    assert all((ROOT / path).is_file() for path in required)
    assert not any(path.startswith("artifacts/") for path in retained)
```

- [ ] **Step 2: Run and verify missing retained paths**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_quality/test_delivery.py -q
```

Expected: failure listing new paths not yet in `RETAINED_FILES.txt`.

- [ ] **Step 3: Add every formal V1-01 path to `RETAINED_FILES.txt`**

Add all paths from the required set in repository-path order. Preserve every existing V2.1/V2.2 entry. Do not add caches, environments, generated acceptance evidence, student submissions or screenshots.

- [ ] **Step 4: Run focused static suites with saved evidence**

```powershell
$evidence = 'artifacts/vision_lab/v2-2-c0-v1-01'
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_quality `
  tests/test_experiments `
  tests/test_student_programs `
  tests/test_simulation `
  tests/test_vision_platform `
  -q 2>&1 | Tee-Object -FilePath (Join-Path $evidence 'pytest-focused.txt')
if ($LASTEXITCODE -ne 0) { throw 'focused regression failed' }
```

Expected: zero failures. Report exact pass/skip counts from the fresh output.

- [ ] **Step 5: Run the complete static regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q 2>&1 |
  Tee-Object -FilePath (Join-Path $evidence 'pytest-full.txt')
if ($LASTEXITCODE -ne 0) { throw 'full static regression failed' }
```

Expected: zero failures. Opt-in CoppeliaSim tests may skip here; report the exact skip count and do not call skips PASS.

- [ ] **Step 6: Re-run release and retained-file contracts**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_quality/test_delivery.py `
  tests/test_acceptance/test_delivery_contract.py -q
```

Expected: all selected tests pass, retained paths have zero duplicates and zero missing files.

- [ ] **Step 7: Verify parallel and protected boundaries are untouched**

```powershell
git diff --exit-code v2.2.0 -- `
  vision_platform/vision2d `
  tests/test_vision2d `
  docs/experiments/V1-02.md `
  docs/experiments/V1-03.md `
  docs/experiments/V1-04.md `
  docs/experiments/V1-05.md `
  vision_platform/recognition/color_shape.py `
  simulation/vision_lab/BL23_vision_lab.ttt `
  simulation/vision_lab/robot_assets_manifest.json `
  simulation/vision_lab/robot_assets
$protected = git diff --name-only v2.2.0 -- | Select-String -Pattern '\.(urdf|stl)$'
if ($protected) { $protected; throw 'Protected URDF/STL changed' }
```

Expected: zero differences for every listed path and no URDF/STL matches.

- [ ] **Step 8: Validate document encoding, fences, placeholders and whitespace**

```powershell
$documents = @(
  'docs/superpowers/specs/2026-08-01-vision-quality-platform-v1-01-design.md',
  'docs/superpowers/plans/2026-08-01-vision-quality-platform-v1-01-plan.md',
  'docs/experiments/V1-01.md'
)
$patterns = @(('T'+'BD'), ('TO'+'DO'), ('FIX'+'ME'), ('待'+'定'))
foreach ($document in $documents) {
  $bytes = [IO.File]::ReadAllBytes((Resolve-Path $document))
  if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    throw "UTF-8 BOM is not allowed: $document"
  }
  $text = Get-Content -LiteralPath $document -Raw -Encoding UTF8
  $fence = ([char]96).ToString() * 3
  if (([regex]::Matches($text, [regex]::Escape($fence))).Count % 2 -ne 0) {
    throw "Unbalanced Markdown fences: $document"
  }
}
Get-Content $documents -Encoding UTF8 | Select-String -Pattern $patterns
git diff --check
```

Expected: no BOM, no unresolved placeholder match, balanced fences and no whitespace errors.

- [ ] **Step 9: Re-run explicit online acceptance after final static changes**

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  .\tools\vision_lab\run_vision_quality_acceptance.ps1 `
  -OutputDir artifacts\vision_lab\v2-2-c0-v1-01 `
  -Port 23005
if ($LASTEXITCODE -ne 0) { throw 'final V1-01 online acceptance failed' }
```

Expected: fresh online PASS with zero skipped online tests, owned process stopped and port free. This is CoppeliaSim evidence only.

- [ ] **Step 10: Mark only executed checkboxes and commit the final handoff**

Only after Steps 1–9 have fresh evidence, change their checkboxes from `[ ]` to `[x]`. Do not mark manual teaching review or hardware validation complete.

```powershell
git add RETAINED_FILES.txt `
  tests/test_vision_quality/test_delivery.py `
  docs/superpowers/plans/2026-08-01-vision-quality-platform-v1-01-plan.md
git commit -m "docs: complete V1-01 vision quality delivery"
git status --short --branch
```

Expected: clean worktree.

- [ ] **Step 11: Push the feature branch without merging main**

```powershell
git push -u origin codex/v2-2-vision-quality-platform
```

Do not merge into `main` during implementation. Integration follows independent review of this branch and the completed algorithm-kernel branch.

## Completion report contract

The final implementation report must state:

- branch and final commit;
- Task-by-Task commit list;
- new/modified files by responsibility;
- exact focused and full static pass/skip counts;
- explicit CoppeliaSim online pass/skip counts and evidence directory;
- V1-01 command count, three applied profiles, three capture sizes and final reset state;
- initial/final scene probe statuses;
- PyQt screenshot paths and automated UI results;
- `RETAINED_FILES.txt` total, duplicate count and missing count;
- protected scene/asset/URDF/STL diff result;
- parallel `vision2d` path diff result;
- GitHub push status;
- `PENDING_HUMAN_ACCEPTANCE` for teaching effectiveness;
- `PENDING_HARDWARE` for Hikvision, real robot, emergency stop, pneumatics, physical grasping and real optical precision.

The branch may claim only the tests and online runs it actually executed. It must not describe static skips as online PASS or CoppeliaSim evidence as real-hardware acceptance.
