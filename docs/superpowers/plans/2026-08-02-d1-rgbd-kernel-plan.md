# D1 RGB-D Pure Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, dependency-light RGB-D foundation that validates in-memory color/depth frames, samples metric depth, deprojects pixels into 3D, applies explicit rigid transforms, and returns finite JSON-native measurement results proven by original synthetic tests.

**Architecture:** The package owns no camera, file, network, simulator, UI, robot, or student-runner integration. Immutable contracts copy and seal caller arrays; pure functions consume validated frames and intrinsics; recoverable missing depth returns a structured failure result while invalid contracts raise stable coded errors. All geometry uses metres and an explicit top-left integer pixel-centre convention.

**Tech Stack:** Python 3.11, NumPy, pytest, standard-library dataclasses/JSON/time, Git/GitHub. No Open3D, ROS, depth-camera SDK, model runtime, or new third-party dependency.

---

## 0. Required design and non-negotiable rules

Read completely before implementation:

- `docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md`
- `docs/superpowers/plans/2026-08-02-d1-rgbd-kernel-plan.md`

Execution rules:

- Work only on branch `codex/v2-2-d1-rgbd-kernel` in a dedicated worktree created from the gated `origin/main` described in Task 1.
- Do not begin implementation if Task 1 fails. Do not start from either V1 candidate branch.
- This branch owns only `vision_platform/rgbd/**`, `tests/test_rgbd/**`, its validation report, and append-only entries for those files in `RETAINED_FILES.txt`.
- Do not modify `vision_platform/vision2d/**`, `vision_platform/student/**`, `vision_platform/experiments/**`, UI, CLI, PowerShell, CoppeliaSim scenes, formal experiment definitions/guides, robot assets, URDF, STL, or tests owned by the V1-07 branch.
- Use `apply_patch` for every source, test, and Markdown edit.
- Every implementation Task follows RED, observed expected failure, minimal GREEN, target regression, related regression, and an independent commit.
- Inputs are in-memory only. No package function accepts `Path`, filename, URL, device handle, CoppeliaSim object, ROS message, or arbitrary callback.
- Color is BGR `uint8 HxWx3`; depth is `float32 HxW` in metres; positive finite values are valid; zero is invalid/missing; NaN and Inf are rejected at the frame boundary.
- The top-left pixel centre is `(0.0, 0.0)`. Deprojection is `x=(u-cx)z/fx`, `y=(v-cy)z/fy`, `z=z`; no implicit half-pixel offset is added.
- No formal D1 experiment, CoppeliaSim RGB-D, SDK, PyQt, robot loop, point cloud, registration, SLAM, grasp planning, real sensor model, or hardware claim is in scope.
- Add every new formal source/test/report file to `RETAINED_FILES.txt` in the same Task. Append only.
- Stop only for a real blocker or scope expansion. Report static results accurately; this branch has no online simulation PASS.

Use the worktree-local interpreter:

```powershell
.\.venv-vision\Scripts\python.exe
```

## Task 1: Establish the exact integrated baseline and isolated worktree

**Files:**

- Verify: `docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md`
- Verify: `docs/superpowers/plans/2026-08-02-d1-rgbd-kernel-plan.md`
- Verify: `config/experiments/V1-06.json`
- Verify: `vision_platform/vision2d/code_recognition.py`
- Verify: `vision_platform/vision2d/defect_detection.py`
- Verify: `tests/test_vision2d/test_defect_detection.py`

- [ ] **Step 1: Fetch and record the prospective base**

```powershell
git fetch origin --prune
git status --short --branch
git rev-parse origin/main
```

Expected: clean source checkout. Record the exact `origin/main` SHA.

- [ ] **Step 2: Prove the shared integration gate is complete**

```powershell
git cat-file -e origin/main:config/experiments/V1-06.json
git cat-file -e origin/main:vision_platform/vision2d/template_matching.py
git cat-file -e origin/main:vision_platform/vision2d/code_recognition.py
git cat-file -e origin/main:vision_platform/vision2d/ocr.py
git cat-file -e origin/main:vision_platform/vision2d/defect_detection.py
git cat-file -e origin/main:docs/superpowers/specs/2026-08-02-v1-07-d1-parallel-coordination-design.md
git cat-file -e origin/main:docs/superpowers/plans/2026-08-02-d1-rgbd-kernel-plan.md
```

Expected: all commands exit 0. Also read `tests/test_vision2d/test_defect_detection.py` and confirm the final identical multi-component PASS plus single-component-split `broken` regressions are present. If not, report `BASELINE_GATE_NOT_READY` and stop.

- [ ] **Step 3: Create the isolated branch from that exact base**

```powershell
git worktree add `
  C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-rgbd-kernel `
  -b codex/v2-2-d1-rgbd-kernel origin/main
Set-Location C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-rgbd-kernel
git rev-parse HEAD
git status --short --branch
```

Expected: `HEAD` equals the recorded `origin/main` SHA and the worktree is clean.

- [ ] **Step 4: Run the complete static baseline**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: zero failures. Report skips separately and do not classify them as online PASS.

- [ ] **Step 5: Record the baseline without an empty commit**

Record base SHA, test result, skip reasons, interpreter path, and worktree path in the eventual validation report.

## Task 2: Define strict immutable RGB-D data contracts

**Files:**

- Create: `vision_platform/rgbd/errors.py`
- Create: `vision_platform/rgbd/models.py`
- Create: `tests/test_rgbd/__init__.py`
- Create: `tests/test_rgbd/test_models.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write model RED tests**

Require positive integer image dimensions; positive finite `fx_px` and `fy_px`; finite `cx_px` and `cy_px`; exact BGR `uint8 HxWx3`; exact depth `float32 HxW`; equal spatial size; no NaN, Inf, negative depth, or boolean substitutions; zero depth accepted as missing; caller arrays unmodified; stored arrays copied, C-contiguous, and read-only; stable equality for scalar models; and stable error codes.

Create the test file with these concrete cases:

```python
from __future__ import annotations

import numpy as np
import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame


def test_intrinsics_normalize_finite_builtin_values() -> None:
    value = CameraIntrinsics(4, 3, 100.0, 110.0, 1.5, 1.0)
    assert value.width_px == 4
    assert value.height_px == 3
    assert value.fx_px == 100.0
    assert value.pixel_center_convention == "integer_center_top_left_zero"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width_px", 0),
        ("width_px", 4.0),
        ("height_px", -1),
        ("height_px", False),
        ("width_px", True),
        ("fx_px", 0.0),
        ("fy_px", -1.0),
        ("fy_px", float("inf")),
        ("cx_px", float("nan")),
        ("cx_px", float("inf")),
        ("cy_px", True),
    ],
)
def test_intrinsics_reject_invalid_values(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "width_px": 4,
        "height_px": 3,
        "fx_px": 100.0,
        "fy_px": 110.0,
        "cx_px": 1.5,
        "cy_px": 1.0,
    }
    arguments[field] = value
    with pytest.raises(RgbdContractError) as captured:
        CameraIntrinsics(**arguments)
    assert captured.value.code == "RGBD_INTRINSICS_INVALID"


def test_frame_copies_and_seals_exact_arrays() -> None:
    image = np.full((3, 4, 3), 17, dtype=np.uint8)
    depth = np.full((3, 4), 0.75, dtype=np.float32)
    frame = RgbdFrame(image, depth)
    image[:] = 99
    depth[:] = 1.5
    assert int(frame.image_bgr[0, 0, 0]) == 17
    assert float(frame.depth_m[0, 0]) == pytest.approx(0.75)
    assert frame.image_bgr.flags.c_contiguous
    assert frame.depth_m.flags.c_contiguous
    assert not frame.image_bgr.flags.writeable
    assert not frame.depth_m.flags.writeable


@pytest.mark.parametrize(
    "depth",
    [
        np.ones((3, 4), dtype=np.float64),
        np.full((3, 4), -0.1, dtype=np.float32),
        np.full((3, 4), np.nan, dtype=np.float32),
        np.full((3, 4), np.inf, dtype=np.float32),
    ],
)
def test_frame_rejects_invalid_depth(depth: np.ndarray) -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    with pytest.raises(RgbdContractError) as captured:
        RgbdFrame(image, depth)
    assert captured.value.code == "RGBD_FRAME_INVALID"


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
        np.zeros((3, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_frame_rejects_invalid_image(image: np.ndarray) -> None:
    with pytest.raises(RgbdContractError) as captured:
        RgbdFrame(image, np.ones((3, 4), dtype=np.float32))
    assert captured.value.code == "RGBD_FRAME_INVALID"


def test_frame_rejects_spatial_size_mismatch() -> None:
    with pytest.raises(RgbdContractError) as captured:
        RgbdFrame(
            np.zeros((3, 4, 3), dtype=np.uint8),
            np.ones((4, 3), dtype=np.float32),
        )
    assert captured.value.code == "RGBD_FRAME_INVALID"


def test_frame_accepts_zero_as_missing_depth() -> None:
    frame = RgbdFrame(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
    )
    assert np.count_nonzero(frame.depth_m) == 0
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd/test_models.py
```

Expected RED: import failure for `vision_platform.rgbd`.

- [ ] **Step 2: Implement stable coded errors and intrinsics**

Use this public shape:

```python
class RgbdContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    pixel_center_convention: str = "integer_center_top_left_zero"

    def __post_init__(self) -> None:
        if (
            type(self.width_px) is not int
            or type(self.height_px) is not int
            or self.width_px <= 0
            or self.height_px <= 0
        ):
            raise RgbdContractError(
                "RGBD_INTRINSICS_INVALID", "image size must be positive integers"
            )
        values = {
            "fx_px": self.fx_px,
            "fy_px": self.fy_px,
            "cx_px": self.cx_px,
            "cy_px": self.cy_px,
        }
        normalized: dict[str, float] = {}
        for name, raw in values.items():
            if type(raw) not in {int, float} or not math.isfinite(float(raw)):
                raise RgbdContractError(
                    "RGBD_INTRINSICS_INVALID", f"{name} must be finite"
                )
            normalized[name] = float(raw)
        if normalized["fx_px"] <= 0.0 or normalized["fy_px"] <= 0.0:
            raise RgbdContractError(
                "RGBD_INTRINSICS_INVALID", "focal lengths must be positive"
            )
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        if self.pixel_center_convention != "integer_center_top_left_zero":
            raise RgbdContractError(
                "RGBD_INTRINSICS_INVALID", "pixel centre convention is unsupported"
            )
```

Place `import math` and `from dataclasses import dataclass` at the top of `models.py`; import `RgbdContractError` from `.errors`. Reject booleans explicitly as shown and normalize accepted built-in integers/floats to built-in immutable values.

- [ ] **Step 3: Implement frame ownership and validity rules**

Use one constructor that copies and seals both arrays:

```python
@dataclass(frozen=True, eq=False)
class RgbdFrame:
    image_bgr: np.ndarray
    depth_m: np.ndarray

    def __post_init__(self) -> None:
        source_image = self.image_bgr
        source_depth = self.depth_m
        if (
            not isinstance(source_image, np.ndarray)
            or source_image.dtype != np.uint8
            or source_image.ndim != 3
            or source_image.shape[2] != 3
            or source_image.shape[0] <= 0
            or source_image.shape[1] <= 0
        ):
            raise RgbdContractError(
                "RGBD_FRAME_INVALID", "image_bgr must be non-empty uint8 HxWx3"
            )
        if (
            not isinstance(source_depth, np.ndarray)
            or source_depth.dtype != np.float32
            or source_depth.ndim != 2
            or source_depth.shape != source_image.shape[:2]
            or not np.all(np.isfinite(source_depth))
            or np.any(source_depth < 0.0)
        ):
            raise RgbdContractError(
                "RGBD_FRAME_INVALID",
                "depth_m must be finite non-negative float32 HxW matching image_bgr",
            )
        image = np.array(self.image_bgr, dtype=np.uint8, copy=True, order="C")
        depth = np.array(self.depth_m, dtype=np.float32, copy=True, order="C")
        image.setflags(write=False)
        depth.setflags(write=False)
        object.__setattr__(self, "image_bgr", image)
        object.__setattr__(self, "depth_m", depth)
```

Do not silently cast an invalid source dtype; the shown array conversion happens only after exact source validation.

- [ ] **Step 4: Run GREEN, input-mutation checks, and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd/test_models.py
git diff --check
git add -- vision_platform/rgbd/errors.py vision_platform/rgbd/models.py `
  tests/test_rgbd/__init__.py tests/test_rgbd/test_models.py `
  RETAINED_FILES.txt
git commit -m "feat(rgbd): define immutable frame contracts"
```

## Task 3: Implement bounded deterministic depth sampling

**Files:**

- Create: `vision_platform/rgbd/sampling.py`
- Create: `tests/test_rgbd/test_sampling.py`
- Modify: `vision_platform/rgbd/models.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write depth-sampling RED tests**

Cover exact single-pixel sampling; 3x3, 5x5, 7x7, and 9x9 median sampling; boundary clipping without wraparound; even/zero/greater-than-nine windows rejected; integer pixel index requirement; out-of-bounds rejection; minimum valid count; zero exclusion; deterministic even-count median; source arrays unchanged; valid count and sampled bounds; and structured `NO_VALID_DEPTH` rather than NaN.

Add these exact tests; the window parameterization covers every allowed odd size:

```python
from __future__ import annotations

import numpy as np
import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.models import RgbdFrame
from vision_platform.rgbd.sampling import sample_depth


def _frame(depth: np.ndarray) -> RgbdFrame:
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    return RgbdFrame(image, np.asarray(depth, dtype=np.float32))


def test_single_pixel_sample_is_exact() -> None:
    sample = sample_depth(_frame(np.array([[0.5]], dtype=np.float32)), 0, 0)
    assert sample.status == "PASS"
    assert sample.depth_m == pytest.approx(0.5)
    assert sample.valid_count == 1
    assert sample.failure_code is None


@pytest.mark.parametrize("window_size", [3, 5, 7, 9])
def test_odd_window_uses_valid_median(window_size: int) -> None:
    depth = np.zeros((9, 9), dtype=np.float32)
    depth[3:6, 3:6] = np.array(
        [[0.0, 0.3, 0.0], [0.1, 0.2, 0.4], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    sample = sample_depth(_frame(depth), 4, 4, window_size=window_size)
    assert sample.depth_m == pytest.approx(0.25)
    assert sample.valid_count == 4


def test_boundary_window_clips_without_wrapping() -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[0, 0] = 0.2
    depth[0, 1] = 0.4
    depth[-1, -1] = 9.0
    sample = sample_depth(_frame(depth), 0, 0, window_size=3)
    assert sample.depth_m == pytest.approx(0.3)
    assert sample.valid_count == 2


def test_no_valid_depth_is_structured() -> None:
    sample = sample_depth(_frame(np.zeros((3, 3), np.float32)), 1, 1, window_size=3)
    assert sample.status == "NO_VALID_DEPTH"
    assert sample.depth_m is None
    assert sample.failure_code == "NO_VALID_DEPTH"


@pytest.mark.parametrize(
    "arguments",
    [
        {"u_px": -1, "v_px": 0},
        {"u_px": 0, "v_px": 3},
        {"u_px": True, "v_px": 0},
        {"u_px": 0, "v_px": 0, "window_size": 2},
        {"u_px": 0, "v_px": 0, "window_size": 11},
        {"u_px": 0, "v_px": 0, "min_valid_count": 0},
    ],
)
def test_sample_rejects_invalid_arguments(arguments: dict[str, object]) -> None:
    with pytest.raises(RgbdContractError) as captured:
        sample_depth(_frame(np.ones((3, 3), np.float32)), **arguments)
    assert captured.value.code == "RGBD_SAMPLE_INVALID"
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd/test_sampling.py
```

Expected RED: sampling API missing.

- [ ] **Step 2: Add the structured sample result**

```python
@dataclass(frozen=True)
class DepthSample:
    status: str
    u_px: int
    v_px: int
    window_size: int
    valid_count: int
    depth_m: float | None
    failure_code: str | None

    def __post_init__(self) -> None:
        if (
            type(self.u_px) is not int
            or type(self.v_px) is not int
            or self.u_px < 0
            or self.v_px < 0
            or type(self.window_size) is not int
            or self.window_size not in {1, 3, 5, 7, 9}
            or type(self.valid_count) is not int
            or not 0 <= self.valid_count <= self.window_size * self.window_size
        ):
            raise RgbdContractError(
                "RGBD_SAMPLE_INVALID", "sample index, window, or count is invalid"
            )
        if self.status == "PASS":
            if (
                self.valid_count <= 0
                or type(self.depth_m) is not float
                or not math.isfinite(self.depth_m)
                or self.depth_m <= 0.0
                or self.failure_code is not None
            ):
                raise RgbdContractError(
                    "RGBD_SAMPLE_INVALID", "PASS sample fields are inconsistent"
                )
        elif self.status == "NO_VALID_DEPTH":
            if self.depth_m is not None or self.failure_code != "NO_VALID_DEPTH":
                raise RgbdContractError(
                    "RGBD_SAMPLE_INVALID", "missing-depth fields are inconsistent"
                )
        else:
            raise RgbdContractError(
                "RGBD_SAMPLE_INVALID", "sample status is unsupported"
            )
```

Import `math` and `RgbdContractError` in `models.py`. Allowed statuses are `PASS` and `NO_VALID_DEPTH`. A PASS has finite positive `depth_m` and no failure code; a failure has `depth_m=None` and `failure_code="NO_VALID_DEPTH"`.

- [ ] **Step 3: Implement bounded median sampling**

```python
def sample_depth(
    frame: RgbdFrame,
    u_px: int,
    v_px: int,
    *,
    window_size: int = 1,
    min_valid_count: int = 1,
) -> DepthSample:
    if not isinstance(frame, RgbdFrame):
        raise RgbdContractError("RGBD_SAMPLE_INVALID", "frame is invalid")
    if type(u_px) is not int or type(v_px) is not int:
        raise RgbdContractError(
            "RGBD_SAMPLE_INVALID", "pixel indices must be built-in integers"
        )
    height, width = frame.depth_m.shape
    if not 0 <= u_px < width or not 0 <= v_px < height:
        raise RgbdContractError("RGBD_SAMPLE_INVALID", "pixel is out of bounds")
    if (
        type(window_size) is not int
        or window_size not in {1, 3, 5, 7, 9}
        or type(min_valid_count) is not int
        or min_valid_count <= 0
        or min_valid_count > window_size * window_size
    ):
        raise RgbdContractError(
            "RGBD_SAMPLE_INVALID", "window or valid-count limit is invalid"
        )
    radius = window_size // 2
    left = max(0, u_px - radius)
    right = min(width, u_px + radius + 1)
    top = max(0, v_px - radius)
    bottom = min(height, v_px + radius + 1)
    window = frame.depth_m[top:bottom, left:right]
    valid = window[window > 0.0]
    valid_count = int(valid.size)
    if valid_count < min_valid_count:
        return DepthSample(
            "NO_VALID_DEPTH",
            u_px,
            v_px,
            window_size,
            valid_count,
            None,
            "NO_VALID_DEPTH",
        )
    return DepthSample(
        "PASS",
        u_px,
        v_px,
        window_size,
        valid_count,
        float(np.median(valid)),
        None,
    )
```

Import NumPy, `RgbdContractError`, `DepthSample`, and `RgbdFrame` at the top of `sampling.py`. Slice only the clipped odd neighbourhood, select values strictly greater than zero, and use `np.median` on that bounded view. Convert the result and count to built-in `float` and `int`.

- [ ] **Step 4: Run GREEN and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd/test_sampling.py tests/test_rgbd/test_models.py
git diff --check
git add -- vision_platform/rgbd/sampling.py vision_platform/rgbd/models.py `
  tests/test_rgbd/test_sampling.py RETAINED_FILES.txt
git commit -m "feat(rgbd): add bounded metric depth sampling"
```

## Task 4: Implement explicit pinhole deprojection

**Files:**

- Create: `vision_platform/rgbd/geometry.py`
- Create: `tests/test_rgbd/test_geometry.py`
- Modify: `vision_platform/rgbd/models.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write geometry RED tests from analytic values**

Require principal-point deprojection to `(0, 0, z)`, corners and boundaries, asymmetric focal lengths, subpixel coordinates, known positive and negative X/Y values, zero/negative/non-finite depth rejection, out-of-image pixels, invalid intrinsics, built-in finite output, deterministic output, and no hidden 0.5-pixel offset.

Use exact analytic cases including `width=640`, `height=480`, `fx=400`, `fy=500`, `cx=319.5`, `cy=239.5`, `u=419.5`, `v=339.5`, `z=2.0`, whose expected point is `(0.5, 0.4, 2.0)`.

Create these concrete tests:

```python
from __future__ import annotations

import math

import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.geometry import deproject_pixel
from vision_platform.rgbd.models import CameraIntrinsics


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(640, 480, 400.0, 500.0, 319.5, 239.5)


def test_principal_point_has_zero_lateral_coordinates() -> None:
    point = deproject_pixel(
        _intrinsics(), u_px=319.5, v_px=239.5, depth_m=2.0
    )
    assert (point.x_m, point.y_m, point.z_m) == pytest.approx((0.0, 0.0, 2.0))


def test_asymmetric_intrinsics_match_analytic_point() -> None:
    point = deproject_pixel(
        _intrinsics(), u_px=419.5, v_px=339.5, depth_m=2.0
    )
    assert (point.x_m, point.y_m, point.z_m) == pytest.approx((0.5, 0.4, 2.0))
    assert all(type(value) is float for value in (point.x_m, point.y_m, point.z_m))


def test_top_left_pixel_uses_integer_center_without_half_offset() -> None:
    point = deproject_pixel(
        CameraIntrinsics(3, 3, 2.0, 2.0, 1.0, 1.0),
        u_px=0.0,
        v_px=0.0,
        depth_m=1.0,
    )
    assert (point.x_m, point.y_m, point.z_m) == pytest.approx((-0.5, -0.5, 1.0))


@pytest.mark.parametrize(
    ("u_px", "v_px", "depth_m"),
    [
        (-0.1, 0.0, 1.0),
        (640.0, 0.0, 1.0),
        (0.0, 480.0, 1.0),
        (True, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, math.inf),
        (0.0, 0.0, math.nan),
    ],
)
def test_deprojection_rejects_invalid_values(
    u_px: object, v_px: object, depth_m: object
) -> None:
    with pytest.raises(RgbdContractError) as captured:
        deproject_pixel(_intrinsics(), u_px=u_px, v_px=v_px, depth_m=depth_m)
    assert captured.value.code == "RGBD_DEPROJECTION_INVALID"
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd/test_geometry.py
```

Expected RED: deprojection API missing.

- [ ] **Step 2: Add a finite immutable point model and deprojection**

```python
@dataclass(frozen=True)
class Point3M:
    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise RgbdContractError(
                    "RGBD_POINT_INVALID", f"{name} must be finite"
                )
            object.__setattr__(self, name, float(value))


def deproject_pixel(
    intrinsics: CameraIntrinsics,
    *,
    u_px: float,
    v_px: float,
    depth_m: float,
) -> Point3M:
    if not isinstance(intrinsics, CameraIntrinsics):
        raise RgbdContractError(
            "RGBD_DEPROJECTION_INVALID", "intrinsics are invalid"
        )
    normalized: list[float] = []
    for name, raw in (("u_px", u_px), ("v_px", v_px), ("depth_m", depth_m)):
        if type(raw) not in {int, float} or not math.isfinite(float(raw)):
            raise RgbdContractError(
                "RGBD_DEPROJECTION_INVALID", f"{name} must be finite"
            )
        normalized.append(float(raw))
    u_value, v_value, depth_value = normalized
    if (
        not 0.0 <= u_value <= intrinsics.width_px - 1
        or not 0.0 <= v_value <= intrinsics.height_px - 1
        or depth_value <= 0.0
    ):
        raise RgbdContractError(
            "RGBD_DEPROJECTION_INVALID", "pixel or depth is outside its domain"
        )
    x_m = (u_value - intrinsics.cx_px) * depth_value / intrinsics.fx_px
    y_m = (v_value - intrinsics.cy_px) * depth_value / intrinsics.fy_px
    return Point3M(x_m, y_m, depth_value)
```

Import `math`, `dataclass`, `RgbdContractError`, and `CameraIntrinsics` where shown. Validate every argument before arithmetic; `Point3M` validates the final result after arithmetic.

- [ ] **Step 3: Run GREEN with model/sampling regression and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd/test_geometry.py `
  tests/test_rgbd/test_sampling.py `
  tests/test_rgbd/test_models.py
git diff --check
git add -- vision_platform/rgbd/geometry.py vision_platform/rgbd/models.py `
  tests/test_rgbd/test_geometry.py RETAINED_FILES.txt
git commit -m "feat(rgbd): deproject pixels into metric points"
```

## Task 5: Validate and apply explicit rigid transforms

**Files:**

- Create: `vision_platform/rgbd/transforms.py`
- Create: `tests/test_rgbd/test_transforms.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write transform RED tests**

Cover identity, translation, 90-degree rotations about X/Y/Z, combined rotation/translation, input point non-mutation, copied read-only matrix, wrong shape, wrong dtype, NaN/Inf, bad homogeneous row, scale, shear, reflection, non-orthonormal rotation, determinant not +1, and non-finite output. Use absolute validation tolerance `1e-6` and analytic expected points.

Create the tests with analytic matrices rather than calling the implementation to build expected values:

```python
from __future__ import annotations

import numpy as np
import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.models import Point3M
from vision_platform.rgbd.transforms import RigidTransform, transform_point


def test_identity_and_translation_are_applied_without_aliasing() -> None:
    matrix = np.array(
        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, -0.2],
         [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = RigidTransform.from_matrix(matrix)
    matrix[0, 3] = 99.0
    result = transform_point(Point3M(1.0, 2.0, 3.0), transform)
    assert (result.x_m, result.y_m, result.z_m) == pytest.approx((1.5, 1.8, 4.0))
    assert not transform.matrix.flags.writeable


def test_z_rotation_matches_analytic_point() -> None:
    matrix = np.array(
        [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    result = transform_point(
        Point3M(1.0, 0.0, 2.0), RigidTransform.from_matrix(matrix)
    )
    assert (result.x_m, result.y_m, result.z_m) == pytest.approx((0.0, 1.0, 2.0))


def _invalid_matrices() -> list[np.ndarray]:
    wrong_shape = np.eye(3, dtype=np.float64)
    non_finite = np.eye(4, dtype=np.float64)
    non_finite[0, 0] = np.nan
    bad_row = np.eye(4, dtype=np.float64)
    bad_row[3, 0] = 1.0
    scaled = np.eye(4, dtype=np.float64)
    scaled[0, 0] = 2.0
    sheared = np.eye(4, dtype=np.float64)
    sheared[0, 1] = 0.2
    reflected = np.eye(4, dtype=np.float64)
    reflected[0, 0] = -1.0
    return [wrong_shape, non_finite, bad_row, scaled, sheared, reflected]


@pytest.mark.parametrize("matrix", _invalid_matrices())
def test_rigid_transform_rejects_non_rigid_matrix(matrix: np.ndarray) -> None:
    with pytest.raises(RgbdContractError) as captured:
        RigidTransform.from_matrix(matrix)
    assert captured.value.code == "RGBD_TRANSFORM_INVALID"


def test_rigid_transform_rejects_object_dtype() -> None:
    with pytest.raises(RgbdContractError):
        RigidTransform.from_matrix(np.eye(4, dtype=object))
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd/test_transforms.py
```

Expected RED: transform API missing.

- [ ] **Step 2: Implement the rigid transform contract**

```python
def _validated_matrix(matrix: np.ndarray) -> np.ndarray:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (4, 4)
        or matrix.dtype.kind not in {"f", "i", "u"}
        or not np.all(np.isfinite(matrix))
    ):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "matrix must be finite numeric 4x4"
        )
    value = np.array(matrix, dtype=np.float64, copy=True, order="C")
    if not np.allclose(
        value[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-6
    ):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "homogeneous row is invalid"
        )
    rotation = value[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "rotation must be proper and orthonormal"
        )
    value.setflags(write=False)
    return value


@dataclass(frozen=True, eq=False)
class RigidTransform:
    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _validated_matrix(self.matrix))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "RigidTransform":
        return cls(matrix)


def transform_point(point: Point3M, transform: RigidTransform) -> Point3M:
    if not isinstance(point, Point3M) or not isinstance(transform, RigidTransform):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "point or transform has invalid type"
        )
    homogeneous = np.array(
        [point.x_m, point.y_m, point.z_m, 1.0], dtype=np.float64
    )
    result = transform.matrix @ homogeneous
    if not np.all(np.isfinite(result)) or not math.isclose(
        float(result[3]), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "transformed point is invalid"
        )
    return Point3M(float(result[0]), float(result[1]), float(result[2]))
```

Import `math`, NumPy, `RgbdContractError`, and `Point3M` at the top of `transforms.py`. Require an exact numeric 4x4 source array, all finite values, final row `[0, 0, 0, 1]`, `R.T @ R` equal to identity within `1e-6`, and determinant equal to `+1` within `1e-6`. Copy to `float64`, make the copy read-only, and never normalize an invalid matrix into validity.

- [ ] **Step 3: Run GREEN and geometry regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd/test_transforms.py tests/test_rgbd/test_geometry.py
git diff --check
```

- [ ] **Step 4: Commit**

```powershell
git add -- vision_platform/rgbd/transforms.py `
  tests/test_rgbd/test_transforms.py RETAINED_FILES.txt
git commit -m "feat(rgbd): add explicit rigid point transforms"
```

## Task 6: Compose measurement results and JSON-native serialization

**Files:**

- Create: `vision_platform/rgbd/serialization.py`
- Create: `vision_platform/rgbd/measurement.py`
- Create: `tests/test_rgbd/test_serialization.py`
- Modify: `vision_platform/rgbd/models.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write result and serialization RED tests**

Require PASS measurements with sample, camera point, optional target point, coordinate-frame IDs, and schema version 1; `NO_VALID_DEPTH` measurements with no points; exact key sets; finite built-in scalars; lists rather than tuples; `None` rather than NaN; stable JSON encoding; fresh returned containers; and rejection of every value that is not an `RgbdMeasurement`.

Create the test file with these exact end-to-end cases:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.measurement import measure_pixel
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame
from vision_platform.rgbd.serialization import measurement_to_dict
from vision_platform.rgbd.transforms import RigidTransform


def _inputs(depth_value: float) -> tuple[RgbdFrame, CameraIntrinsics]:
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    depth = np.full((3, 3), depth_value, dtype=np.float32)
    return RgbdFrame(image, depth), CameraIntrinsics(3, 3, 2.0, 2.0, 1.0, 1.0)


def test_measurement_serializes_camera_and_target_points() -> None:
    frame, intrinsics = _inputs(2.0)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    result = measure_pixel(
        frame,
        intrinsics,
        u_px=2,
        v_px=1,
        camera_frame_id="camera_optical",
        target_frame_id="world",
        camera_to_target=RigidTransform.from_matrix(matrix),
    )
    payload = measurement_to_dict(result)
    assert payload == {
        "schema_version": 1,
        "status": "PASS",
        "pixel_px": [2, 1],
        "window_size": 1,
        "valid_count": 1,
        "depth_m": 2.0,
        "camera_frame_id": "camera_optical",
        "target_frame_id": "world",
        "point_camera_m": [1.0, 0.0, 2.0],
        "point_target_m": [2.0, 2.0, 5.0],
        "failure_code": None,
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    payload["point_camera_m"][0] = 99.0
    assert measurement_to_dict(result)["point_camera_m"][0] == 1.0


def test_missing_depth_serializes_null_points() -> None:
    frame, intrinsics = _inputs(0.0)
    payload = measurement_to_dict(
        measure_pixel(frame, intrinsics, u_px=1, v_px=1)
    )
    assert payload["status"] == "NO_VALID_DEPTH"
    assert payload["depth_m"] is None
    assert payload["point_camera_m"] is None
    assert payload["point_target_m"] is None
    assert payload["failure_code"] == "NO_VALID_DEPTH"


def test_target_frame_and_transform_must_be_paired() -> None:
    frame, intrinsics = _inputs(1.0)
    with pytest.raises(RgbdContractError) as captured:
        measure_pixel(frame, intrinsics, u_px=1, v_px=1, target_frame_id="world")
    assert captured.value.code == "RGBD_MEASUREMENT_INVALID"


def test_measurement_rejects_frame_intrinsics_size_mismatch() -> None:
    frame, _ = _inputs(1.0)
    mismatched = CameraIntrinsics(4, 3, 2.0, 2.0, 1.5, 1.0)
    with pytest.raises(RgbdContractError) as captured:
        measure_pixel(frame, mismatched, u_px=1, v_px=1)
    assert captured.value.code == "RGBD_MEASUREMENT_INVALID"


def test_invalid_transform_is_rejected_even_when_depth_is_missing() -> None:
    frame, intrinsics = _inputs(0.0)
    with pytest.raises(RgbdContractError) as captured:
        measure_pixel(
            frame, intrinsics, u_px=1, v_px=1,
            target_frame_id="world", camera_to_target=object(),
        )
    assert captured.value.code == "RGBD_MEASUREMENT_INVALID"


@pytest.mark.parametrize(
    "invalid",
    [None, np.float32(1.0), np.zeros(1), Path("frame.json"), b"bytes", {}],
)
def test_serializer_rejects_arbitrary_objects(invalid: object) -> None:
    with pytest.raises(RgbdContractError) as captured:
        measurement_to_dict(invalid)
    assert captured.value.code == "RGBD_SERIALIZATION_INVALID"
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd/test_serialization.py
```

Expected RED: measurement and serialization APIs missing.

- [ ] **Step 2: Add the exact result model and composition function**

```python
@dataclass(frozen=True)
class RgbdMeasurement:
    status: str
    sample: DepthSample
    camera_frame_id: str
    target_frame_id: str | None
    point_camera_m: Point3M | None
    point_target_m: Point3M | None
    failure_code: str | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.sample, DepthSample):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "sample type is invalid"
            )
        if self.point_camera_m is not None and not isinstance(self.point_camera_m, Point3M):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "camera point type is invalid"
            )
        if self.point_target_m is not None and not isinstance(self.point_target_m, Point3M):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "target point type is invalid"
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "schema version must be 1"
            )
        for name in ("camera_frame_id",):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value) > 80:
                raise RgbdContractError(
                    "RGBD_MEASUREMENT_INVALID", f"{name} is invalid"
                )
        if self.target_frame_id is not None and (
            type(self.target_frame_id) is not str
            or not self.target_frame_id
            or len(self.target_frame_id) > 80
        ):
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "target_frame_id is invalid"
            )
        if self.status == "PASS":
            if (
                self.sample.status != "PASS"
                or not isinstance(self.point_camera_m, Point3M)
                or self.failure_code is not None
                or (self.target_frame_id is None) != (self.point_target_m is None)
            ):
                raise RgbdContractError(
                    "RGBD_MEASUREMENT_INVALID", "PASS fields are inconsistent"
                )
        elif self.status == "NO_VALID_DEPTH":
            if (
                self.sample.status != "NO_VALID_DEPTH"
                or self.point_camera_m is not None
                or self.point_target_m is not None
                or self.failure_code != "NO_VALID_DEPTH"
            ):
                raise RgbdContractError(
                    "RGBD_MEASUREMENT_INVALID", "failure fields are inconsistent"
                )
        else:
            raise RgbdContractError(
                "RGBD_MEASUREMENT_INVALID", "measurement status is unsupported"
            )


def measure_pixel(
    frame: RgbdFrame,
    intrinsics: CameraIntrinsics,
    *,
    u_px: int,
    v_px: int,
    window_size: int = 1,
    min_valid_count: int = 1,
    camera_frame_id: str = "camera",
    target_frame_id: str | None = None,
    camera_to_target: RigidTransform | None = None,
) -> RgbdMeasurement:
    if (target_frame_id is None) != (camera_to_target is None):
        raise RgbdContractError(
            "RGBD_MEASUREMENT_INVALID",
            "target frame and transform must be supplied together",
        )
    if camera_to_target is not None and not isinstance(camera_to_target, RigidTransform):
        raise RgbdContractError(
            "RGBD_MEASUREMENT_INVALID", "camera_to_target is invalid"
        )
    if (
        not isinstance(frame, RgbdFrame)
        or not isinstance(intrinsics, CameraIntrinsics)
        or frame.depth_m.shape != (intrinsics.height_px, intrinsics.width_px)
    ):
        raise RgbdContractError(
            "RGBD_MEASUREMENT_INVALID", "frame and intrinsics size do not match"
        )
    if type(camera_frame_id) is not str or not camera_frame_id:
        raise RgbdContractError(
            "RGBD_MEASUREMENT_INVALID", "camera frame ID is invalid"
        )
    sample = sample_depth(
        frame,
        u_px,
        v_px,
        window_size=window_size,
        min_valid_count=min_valid_count,
    )
    if sample.status == "NO_VALID_DEPTH":
        return RgbdMeasurement(
            "NO_VALID_DEPTH",
            sample,
            camera_frame_id,
            target_frame_id,
            None,
            None,
            "NO_VALID_DEPTH",
        )
    assert sample.depth_m is not None
    camera_point = deproject_pixel(
        intrinsics,
        u_px=float(u_px),
        v_px=float(v_px),
        depth_m=sample.depth_m,
    )
    target_point = (
        None
        if camera_to_target is None
        else transform_point(camera_point, camera_to_target)
    )
    return RgbdMeasurement(
        "PASS",
        sample,
        camera_frame_id,
        target_frame_id,
        camera_point,
        target_point,
        None,
    )
```

Place this function in `measurement.py` and import all referenced public types and functions explicitly. An absent target transform produces only `point_camera_m`. A provided transform requires a non-empty target frame ID. Missing valid depth returns a structured failure without calling deprojection or transform.

- [ ] **Step 3: Implement result-specific JSON serialization**

```python
def measurement_to_dict(result: RgbdMeasurement) -> dict[str, object]:
    if not isinstance(result, RgbdMeasurement):
        raise RgbdContractError(
            "RGBD_SERIALIZATION_INVALID", "result must be RgbdMeasurement"
        )

    def point(value: Point3M | None) -> list[float] | None:
        if value is None:
            return None
        return [float(value.x_m), float(value.y_m), float(value.z_m)]

    payload: dict[str, object] = {
        "schema_version": int(result.schema_version),
        "status": result.status,
        "pixel_px": [int(result.sample.u_px), int(result.sample.v_px)],
        "window_size": int(result.sample.window_size),
        "valid_count": int(result.sample.valid_count),
        "depth_m": (
            None if result.sample.depth_m is None else float(result.sample.depth_m)
        ),
        "camera_frame_id": result.camera_frame_id,
        "target_frame_id": result.target_frame_id,
        "point_camera_m": point(result.point_camera_m),
        "point_target_m": point(result.point_target_m),
        "failure_code": result.failure_code,
    }
    json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return payload
```

Import `json`, `RgbdContractError`, `Point3M`, and `RgbdMeasurement` at the top of `serialization.py`. Serialize fields explicitly rather than recursively accepting arbitrary objects. `json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))` is the final finite JSON-native guard, and each call creates fresh list containers.

- [ ] **Step 4: Run GREEN and all RGB-D tests to date**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add -- vision_platform/rgbd/serialization.py vision_platform/rgbd/measurement.py `
  vision_platform/rgbd/models.py `
  tests/test_rgbd/test_serialization.py RETAINED_FILES.txt
git commit -m "feat(rgbd): compose serializable 3D measurements"
```

## Task 7: Add original synthetic contracts, determinism, and CPU budget

**Files:**

- Create: `tests/test_rgbd/synthetic_factory.py`
- Create: `tests/test_rgbd/test_synthetic_factory.py`
- Create: `tests/test_rgbd/test_pipeline_contract.py`
- Create: `tests/test_rgbd/test_performance.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write synthetic-factory RED tests**

Require deterministic in-memory plane, depth step, raised box, tilted plane, rectangular zero-depth hole, and boundary-pixel fixtures. Each factory accepts explicit dimensions and a fixed integer seed where noise is used, returns exact BGR/depth dtypes, writes no files, has known analytic depth at selected pixels, and does not share mutable arrays across calls.

Implement the test-only factory with this exact public surface:

```python
from __future__ import annotations

import numpy as np


def _base(width: int, height: int, depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("synthetic size must be positive integers")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (32, 96, 160)
    depth = np.full((height, width), depth_m, dtype=np.float32)
    return image, depth


def make_plane(width: int, height: int, depth_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return _base(width, height, depth_m)


def make_step(
    width: int, height: int, left_m: float = 0.8, right_m: float = 1.2
) -> tuple[np.ndarray, np.ndarray]:
    image, depth = _base(width, height, left_m)
    depth[:, width // 2 :] = np.float32(right_m)
    return image, depth


def make_box(
    width: int, height: int, plane_m: float = 1.0, box_m: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    image, depth = _base(width, height, plane_m)
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    image[y0:y1, x0:x1] = (20, 180, 40)
    depth[y0:y1, x0:x1] = np.float32(box_m)
    return image, depth


def make_tilted_plane(
    width: int,
    height: int,
    *,
    base_m: float = 0.6,
    du_m: float = 0.001,
    dv_m: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    image, _ = _base(width, height, base_m)
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    depth = np.asarray(base_m + du_m * u + dv_m * v, dtype=np.float32)
    return image, depth


def make_hole(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    image, depth = _base(width, height, 1.0)
    depth[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 0.0
    return image, depth


def make_seeded_noise(
    width: int, height: int, *, seed: int, sigma_m: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    image, depth = _base(width, height, 1.0)
    rng = np.random.default_rng(seed)
    depth += rng.normal(0.0, sigma_m, depth.shape).astype(np.float32)
    return image, depth
```

- [ ] **Step 2: Implement only original synthetic generators**

Use the code above, NumPy formulas, and fixed BGR fills. Do not download images, copy external datasets, or add binary fixtures. The tilted plane equation is `z=base_m+du_m*u+dv_m*v`, in metres with integer-centre pixel coordinates.

- [ ] **Step 3: Write cross-contract and deterministic pipeline tests**

For every fixture, construct `RgbdFrame`, sample selected points, deproject, optionally transform, serialize, round-trip through `json.dumps`/`json.loads`, and compare against analytic values. Require repeated calls and process-independent fixed-seed fixtures to produce identical values. Verify errors stay within `1e-6 m` for exact planes and `1e-5 m` for median windows crossing an allowed synthetic slope.

Create `test_synthetic_factory.py` and `test_pipeline_contract.py` with these core tests; add direct value assertions for step, box, hole, and all four boundaries using the same factory functions:

```python
import json

import numpy as np
import pytest

from tests.test_rgbd.synthetic_factory import (
    make_box,
    make_hole,
    make_plane,
    make_seeded_noise,
    make_step,
    make_tilted_plane,
)
from vision_platform.rgbd.measurement import measure_pixel
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame
from vision_platform.rgbd.serialization import measurement_to_dict


def test_seeded_factory_is_deterministic_and_independent() -> None:
    first_image, first_depth = make_seeded_noise(32, 24, seed=1707)
    second_image, second_depth = make_seeded_noise(32, 24, seed=1707)
    assert np.array_equal(first_image, second_image)
    assert np.array_equal(first_depth, second_depth)
    first_depth[0, 0] = 9.0
    assert second_depth[0, 0] != 9.0


def test_all_factories_have_exact_contracts() -> None:
    for factory in (make_plane, make_step, make_box, make_tilted_plane, make_hole):
        image, depth = factory(12, 10)
        assert image.shape == (10, 12, 3)
        assert image.dtype == np.uint8
        assert depth.shape == (10, 12)
        assert depth.dtype == np.float32


def test_tilted_plane_measurement_matches_independent_equation() -> None:
    image, depth = make_tilted_plane(16, 12, base_m=0.6, du_m=0.001, dv_m=0.002)
    frame = RgbdFrame(image, depth)
    intrinsics = CameraIntrinsics(16, 12, 100.0, 120.0, 7.5, 5.5)
    result = measure_pixel(frame, intrinsics, u_px=10, v_px=8, window_size=3)
    payload = measurement_to_dict(result)
    expected_z = 0.6 + 0.001 * 10 + 0.002 * 8
    assert payload["depth_m"] == pytest.approx(expected_z, abs=1e-5)
    assert payload["point_camera_m"] == pytest.approx(
        [(10 - 7.5) * expected_z / 100.0, (8 - 5.5) * expected_z / 120.0, expected_z],
        abs=1e-5,
    )
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_hole_and_boundaries_use_defined_failure_and_success_contracts() -> None:
    image, depth = make_hole(9, 9)
    frame = RgbdFrame(image, depth)
    intrinsics = CameraIntrinsics(9, 9, 100.0, 100.0, 4.0, 4.0)
    center = measurement_to_dict(measure_pixel(frame, intrinsics, u_px=4, v_px=4))
    assert center["status"] == "NO_VALID_DEPTH"
    for u_px, v_px in ((0, 0), (8, 0), (0, 8), (8, 8)):
        boundary = measurement_to_dict(
            measure_pixel(frame, intrinsics, u_px=u_px, v_px=v_px, window_size=3)
        )
        assert boundary["status"] == "PASS"
        assert boundary["depth_m"] == pytest.approx(1.0)
```

- [ ] **Step 4: Add a generous but explicit CPU budget**

On a 512x512 plane frame, construct and seal the frame, perform 1,024 bounded 3x3 samples plus deprojections, and serialize the results in less than 2.0 seconds on the project baseline machine. Warm up once, measure with `time.perf_counter`, do not include interpreter startup, and mark no test as network, GPU, online, or hardware dependent.

Create the performance test exactly as follows so it measures the approved path and does not hide work in setup:

```python
from time import perf_counter

from tests.test_rgbd.synthetic_factory import make_plane
from vision_platform.rgbd.measurement import measure_pixel
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame
from vision_platform.rgbd.serialization import measurement_to_dict


def test_512_frame_and_1024_measurements_fit_cpu_budget() -> None:
    image, depth = make_plane(512, 512, 1.0)
    intrinsics = CameraIntrinsics(512, 512, 400.0, 400.0, 255.5, 255.5)
    warm_frame = RgbdFrame(image, depth)
    measurement_to_dict(
        measure_pixel(warm_frame, intrinsics, u_px=16, v_px=16, window_size=3)
    )
    started = perf_counter()
    frame = RgbdFrame(image, depth)
    for v_px in range(8, 512, 16):
        for u_px in range(8, 512, 16):
            measurement_to_dict(
                measure_pixel(
                    frame, intrinsics, u_px=u_px, v_px=v_px, window_size=3
                )
            )
    elapsed_s = perf_counter() - started
    assert elapsed_s < 2.0, f"RGB-D CPU budget exceeded: {elapsed_s:.3f}s"
```

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_rgbd/test_synthetic_factory.py `
  tests/test_rgbd/test_pipeline_contract.py `
  tests/test_rgbd/test_performance.py
```

Expected GREEN: all synthetic, end-to-end, determinism, and CPU-budget tests pass without skips.

- [ ] **Step 5: Run full RGB-D regression and commit**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd
git diff --check
git add -- tests/test_rgbd RETAINED_FILES.txt
git commit -m "test(rgbd): prove synthetic 3D measurement contracts"
```

## Task 8: Publish the package boundary and run release regressions

**Files:**

- Create: `vision_platform/rgbd/__init__.py`
- Create: `tests/test_rgbd/test_public_api.py`
- Create: `tests/test_rgbd/test_delivery.py`
- Modify: `RETAINED_FILES.txt`
- Verify: dependency files and all protected paths

- [ ] **Step 1: Write public API and delivery RED tests**

Require an explicit `__all__` containing only the approved contracts and functions; no OpenCV, file, network, ROS, CoppeliaSim, UI, SDK, or robot import; import has no thread/process/file side effect; all RGB-D source/tests/report paths are retained; and no generated caches or runtime artifacts are retained.

Create these concrete public and delivery tests:

```python
from __future__ import annotations

import ast
from pathlib import Path

import vision_platform.rgbd as rgbd


EXPECTED_PUBLIC_API = {
    "CameraIntrinsics",
    "DepthSample",
    "Point3M",
    "RgbdContractError",
    "RgbdFrame",
    "RgbdMeasurement",
    "RigidTransform",
    "deproject_pixel",
    "measure_pixel",
    "measurement_to_dict",
    "sample_depth",
    "transform_point",
}


def test_public_api_is_exact_and_resolvable() -> None:
    assert set(rgbd.__all__) == EXPECTED_PUBLIC_API
    assert len(rgbd.__all__) == len(EXPECTED_PUBLIC_API)
    for name in rgbd.__all__:
        assert getattr(rgbd, name) is not None


def test_rgbd_sources_have_no_forbidden_imports() -> None:
    root = Path(rgbd.__file__).resolve().parent
    forbidden = {"cv2", "open3d", "rclpy", "zmq", "PyQt5", "socket", "subprocess"}
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden, f"{source.name}: {sorted(imported & forbidden)}"
```

```python
from pathlib import Path


def test_all_rgbd_delivery_files_are_retained() -> None:
    root = Path(__file__).resolve().parents[2]
    retained = {
        line.strip()
        for line in (root / "RETAINED_FILES.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    expected = {
        path.relative_to(root).as_posix()
        for base in (root / "vision_platform" / "rgbd", root / "tests" / "test_rgbd")
        for path in base.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert expected <= retained
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in retained)
```

- [ ] **Step 2: Export the minimal API**

The public surface is exactly:

```python
__all__ = [
    "CameraIntrinsics",
    "DepthSample",
    "Point3M",
    "RgbdContractError",
    "RgbdFrame",
    "RgbdMeasurement",
    "RigidTransform",
    "deproject_pixel",
    "measure_pixel",
    "measurement_to_dict",
    "sample_depth",
    "transform_point",
]
```

Place these imports above `__all__` in `vision_platform/rgbd/__init__.py`:

```python
from .errors import RgbdContractError
from .geometry import deproject_pixel
from .measurement import measure_pixel
from .models import (
    CameraIntrinsics,
    DepthSample,
    Point3M,
    RgbdFrame,
    RgbdMeasurement,
)
from .sampling import sample_depth
from .serialization import measurement_to_dict
from .transforms import RigidTransform, transform_point
```

- [ ] **Step 3: Prove no dependency or ownership drift**

```powershell
$base = git merge-base HEAD origin/main
git diff --name-only $base..HEAD
git diff --exit-code $base..HEAD -- `
  pyproject.toml requirements.txt requirements-dev.txt `
  vision_platform/vision2d vision_platform/student vision_platform/experiments `
  vision_platform/ui config tools simulation
```

Expected: the ownership/dependency diff exits 0. If a listed dependency file does not exist in the repository, omit only that nonexistent path from the command; do not create it.

- [ ] **Step 4: Run package, release, and full static regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd
.\.venv-vision\Scripts\python.exe -m pytest -q `
  tests/test_acceptance/test_delivery_contract.py
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
git status --short --branch
```

Expected: zero failures; all RGB-D tests have zero skips; full-suite skips are reported separately. No online CoppeliaSim test is added or claimed.

- [ ] **Step 5: Commit package exports and delivery tests**

```powershell
git add -- vision_platform/rgbd/__init__.py `
  tests/test_rgbd/test_public_api.py tests/test_rgbd/test_delivery.py `
  RETAINED_FILES.txt
git commit -m "feat(rgbd): publish the pure kernel API"
```

## Task 9: Independent review, validation report, push, and handoff

**Files:**

- Create: `docs/superpowers/reports/2026-08-02-d1-rgbd-kernel-validation-report.md`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Request independent P0/P1/P2 review**

The reviewer must compare the complete branch diff with the approved coordination design and inspect dtype rejection before copying, zero/NaN/Inf rules, array aliasing, boundary clipping, pixel-centre convention, transform rigidity, JSON finiteness, performance-test stability, import side effects, dependency drift, and file ownership.

- [ ] **Step 2: Fix every accepted P0/P1 and relevant P2 with RED/GREEN evidence**

Make focused commits and rerun the affected tests plus all `tests/test_rgbd`. If any public contract changes, rerun the full static suite.

- [ ] **Step 3: Write the validation report from fresh outputs**

Record base SHA, tip SHA, Task commits, RED/GREEN evidence, RGB-D test result, full static result and skips, performance timing, retained audit, dependency/ownership audit, and limitations. State precisely that only the D1 pure RGB-D software kernel was validated; CoppeliaSim RGB-D, D1 curriculum, SDK/UI/robot integration, real cameras, and physical hardware were not run.

Use this exact report heading order and retain the boundary paragraph verbatim. Populate each evidence section from fresh literal command output:

```markdown
# D1 RGB-D Pure Kernel Validation Report

## Scope and claim boundary

本报告只验证 D1 纯内存 RGB-D 数据合同、深度采样、针孔反投影、刚体变换、JSON 序列化和原创合成测试。CoppeliaSim RGB-D、正式 D1 课程、学生 SDK、PyQt、机器人闭环、真实深度相机和物理硬件均未验收。

## Exact baseline and branch tip

## Task commits

## RED and GREEN evidence

## RGB-D focused tests

## Full static regression and skips

## 512x512 CPU timing

## Dependency, ownership, and retained-file audit

## Independent review findings and fixes

## Remaining limitations
```

- [ ] **Step 4: Commit the report and retained entry**

```powershell
git add -- docs/superpowers/reports/2026-08-02-d1-rgbd-kernel-validation-report.md `
  RETAINED_FILES.txt
git commit -m "docs: record D1 RGB-D kernel validation"
```

- [ ] **Step 5: Run final clean verification and push only this branch**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_rgbd
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
git status --short --branch
git log --oneline origin/main..HEAD
git push -u origin codex/v2-2-d1-rgbd-kernel
git rev-parse HEAD
git rev-parse origin/codex/v2-2-d1-rgbd-kernel
```

Expected: local and remote full SHAs match, worktree is clean, and no merge to `main` occurred.

- [ ] **Step 6: Hand off exact claims and integration order**

Report changed files, commit range, tests, performance evidence, review findings, and remaining gates. Recommend integrating V1-07 first and D1 second. Do not describe D1 static tests as a simulator, course, robot, or hardware acceptance.
