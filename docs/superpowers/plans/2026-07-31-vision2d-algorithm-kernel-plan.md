# V2.2-C1 Vision2D Algorithm Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, multi-object 2D vision algorithm kernel for V1-02 through V1-05 without touching the active V2.2 integration surfaces.

**Architecture:** Add an isolated `vision_platform.vision2d` package whose layers validate and preprocess BGR images, segment independent objects, measure geometry, classify appearance, and serialize structured results. The package has no camera, CoppeliaSim, robot, student-runner, CLI, or PyQt dependency; later integration will adapt its result model into the multi-experiment platform.

**Tech Stack:** Python 3.11, NumPy, OpenCV, frozen dataclasses, pytest, Windows PowerShell, Git worktrees.

---

## 1. Execution baseline and hard boundaries

Use this existing worktree:

```text
C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-vision2d-algorithm-kernel
```

Expected branch and design baseline:

```text
codex/v2-2-vision2d-algorithm-kernel
38f1914 docs: design V1 vision2d algorithm kernel
```

Before Task 1, run:

```powershell
git status --short --branch
git log -2 --oneline
.\.venv-vision\Scripts\python.exe -m pytest -q
```

Expected baseline: clean worktree and `624 passed, 3 skipped`. The three skips are opt-in CoppeliaSim tests and are not an online acceptance PASS.

Do not modify these active integration surfaces:

```text
vision_platform/cli.py
vision_platform/application.py
vision_platform/session.py
vision_platform/student/
vision_platform/ui/
vision_platform/recognition/color_shape.py
tools/vision_lab/run_acceptance.ps1
config/experiments/
simulation/**/*.ttt
```

Do not add generated images, `.venv-vision/`, `artifacts/`, caches, or personal submissions to Git. Do not claim CoppeliaSim, robot motion, PyQt, teaching effectiveness, Hikvision, or real-hardware acceptance from this branch.

## 2. File map

Runtime files to create:

```text
vision_platform/vision2d/__init__.py
vision_platform/vision2d/models.py
vision_platform/vision2d/preprocessing.py
vision_platform/vision2d/segmentation.py
vision_platform/vision2d/geometry.py
vision_platform/vision2d/appearance.py
vision_platform/vision2d/pipeline.py
vision_platform/vision2d/serialization.py
```

Test files to create:

```text
tests/test_vision2d/__init__.py
tests/test_vision2d/synthetic_factory.py
tests/test_vision2d/test_models.py
tests/test_vision2d/test_synthetic_factory.py
tests/test_vision2d/test_preprocessing.py
tests/test_vision2d/test_segmentation.py
tests/test_vision2d/test_geometry.py
tests/test_vision2d/test_appearance.py
tests/test_vision2d/test_pipeline.py
tests/test_vision2d/test_serialization.py
tests/test_vision2d/test_compatibility.py
tests/test_vision2d/test_guides.py
tests/test_vision2d/test_delivery.py
```

Course files to create:

```text
docs/experiments/V1-02.md
docs/experiments/V1-03.md
docs/experiments/V1-04.md
docs/experiments/V1-05.md
```

Files to modify:

```text
RETAINED_FILES.txt
docs/superpowers/plans/2026-07-31-vision2d-algorithm-kernel-plan.md
```

## Task 1: Package skeleton and immutable result contracts

**Files:**
- Create: `vision_platform/vision2d/__init__.py`
- Create: `vision_platform/vision2d/models.py`
- Create: `tests/test_vision2d/__init__.py`
- Create: `tests/test_vision2d/test_models.py`

- [ ] **Step 1: Write the failing model tests**

Create `tests/test_vision2d/__init__.py` as an empty UTF-8 file.

Create `tests/test_vision2d/test_models.py`:

```python
import math

import numpy as np
import pytest

from vision_platform.vision2d.models import (
    PixelScale,
    RejectedTarget,
    TargetMeasurement,
    Vision2DAnalysis,
    Vision2DConfig,
    Vision2DResult,
)


@pytest.mark.parametrize("value", [0.0, -0.1, math.inf, -math.inf, math.nan])
def test_pixel_scale_requires_finite_positive_values(value):
    with pytest.raises(ValueError, match="finite positive"):
        PixelScale(mm_per_pixel_x=value, mm_per_pixel_y=0.2)


def test_config_validates_ratios_and_odd_kernel_sizes():
    with pytest.raises(ValueError, match="area ratios"):
        Vision2DConfig(min_area_ratio=0.5, max_area_ratio=0.4)
    with pytest.raises(ValueError, match="positive odd"):
        Vision2DConfig(gaussian_kernel_size=4)
    with pytest.raises(ValueError, match="positive odd"):
        Vision2DConfig(morphology_kernel_size=0)


def test_result_contract_keeps_numpy_only_at_analysis_boundary():
    contour = np.array([[[0, 0]], [[2, 0]], [[2, 2]]], dtype=np.int32)
    target = TargetMeasurement(
        detection_id="det-001",
        center_px=(1.0, 1.0),
        axis_aligned_bbox_px=(0, 0, 3, 3),
        rotated_box_px=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
        long_side_px=2.0,
        short_side_px=2.0,
        angle_deg=0.0,
        area_px2=4.0,
        perimeter_px=8.0,
        size_mm=None,
        area_mm2=None,
        perimeter_mm=None,
        color="red",
        shape="square",
        vertex_count=4,
        circularity=0.785,
        aspect_ratio=1.0,
        quality_flags=("ANGLE_AMBIGUOUS_FOR_SQUARE",),
        contour=contour,
    )
    rejected = RejectedTarget(
        candidate_index=2,
        center_px=None,
        area_px2=1.0,
        code="AREA_TOO_SMALL",
        reason="candidate area is below configured minimum",
        contour=contour,
    )
    result = Vision2DResult(
        status="PARTIAL",
        image_size=(320, 240),
        targets=(target,),
        rejected_targets=(rejected,),
    )
    analysis = Vision2DAnalysis(
        result=result,
        intermediate_images={"gray": np.zeros((240, 320), dtype=np.uint8)},
    )

    assert analysis.result.targets[0].detection_id == "det-001"
    assert analysis.result.image_size == (320, 240)
    assert analysis.intermediate_images["gray"].shape == (240, 320)


def test_result_rejects_unknown_status_and_invalid_image_size():
    with pytest.raises(ValueError, match="status"):
        Vision2DResult(status="OK", image_size=(320, 240))
    with pytest.raises(ValueError, match="image_size"):
        Vision2DResult(status="NO_TARGETS", image_size=(0, 240))
```

- [ ] **Step 2: Run the model tests and verify the import failure**

Run:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d/test_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'vision_platform.vision2d'`.

- [ ] **Step 3: Implement the immutable contracts**

Create `vision_platform/vision2d/models.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


def _finite_positive(value: float, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite positive")
    return normalized


@dataclass(frozen=True)
class PixelScale:
    mm_per_pixel_x: float
    mm_per_pixel_y: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mm_per_pixel_x",
            _finite_positive(self.mm_per_pixel_x, "mm_per_pixel_x"),
        )
        object.__setattr__(
            self,
            "mm_per_pixel_y",
            _finite_positive(self.mm_per_pixel_y, "mm_per_pixel_y"),
        )


@dataclass(frozen=True)
class Vision2DConfig:
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.40
    saturation_min: int = 60
    value_min: int = 40
    gaussian_kernel_size: int = 5
    morphology_kernel_size: int = 3
    border_margin_px: int = 1
    polygon_epsilon_ratio: float = 0.03
    square_aspect_min: float = 0.85
    circle_circularity_min: float = 0.70
    pixel_scale: PixelScale | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.min_area_ratio < self.max_area_ratio <= 1.0:
            raise ValueError("area ratios must satisfy 0 < min < max <= 1")
        for name in ("gaussian_kernel_size", "morphology_kernel_size"):
            value = int(getattr(self, name))
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be positive odd")
        if not 0 <= self.saturation_min <= 255:
            raise ValueError("saturation_min must be between 0 and 255")
        if not 0 <= self.value_min <= 255:
            raise ValueError("value_min must be between 0 and 255")
        if self.border_margin_px < 0:
            raise ValueError("border_margin_px must be non-negative")
        if not 0.0 < self.polygon_epsilon_ratio < 1.0:
            raise ValueError("polygon_epsilon_ratio must be between 0 and 1")
        if not 0.0 < self.square_aspect_min <= 1.0:
            raise ValueError("square_aspect_min must be between 0 and 1")
        if not 0.0 < self.circle_circularity_min <= 1.0:
            raise ValueError("circle_circularity_min must be between 0 and 1")


@dataclass(frozen=True)
class TargetMeasurement:
    detection_id: str
    center_px: tuple[float, float]
    axis_aligned_bbox_px: tuple[int, int, int, int]
    rotated_box_px: tuple[tuple[float, float], ...]
    long_side_px: float
    short_side_px: float
    angle_deg: float | None
    area_px2: float
    perimeter_px: float
    size_mm: tuple[float, float] | None
    area_mm2: float | None
    perimeter_mm: float | None
    color: str
    shape: str
    vertex_count: int
    circularity: float
    aspect_ratio: float
    quality_flags: tuple[str, ...] = ()
    contour: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty((0, 1, 2), dtype=np.int32))


@dataclass(frozen=True)
class RejectedTarget:
    candidate_index: int
    center_px: tuple[float, float] | None
    area_px2: float
    code: str
    reason: str
    contour: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class Vision2DResult:
    status: str
    image_size: tuple[int, int]
    targets: tuple[TargetMeasurement, ...] = ()
    rejected_targets: tuple[RejectedTarget, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "PARTIAL", "NO_TARGETS", "REJECTED"}:
            raise ValueError("status is not a supported Vision2D result status")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("image_size must contain positive width and height")


@dataclass(frozen=True)
class Vision2DAnalysis:
    result: Vision2DResult
    intermediate_images: Mapping[str, np.ndarray]
```

Create `vision_platform/vision2d/__init__.py`:

```python
from vision_platform.vision2d.models import (
    PixelScale,
    RejectedTarget,
    TargetMeasurement,
    Vision2DAnalysis,
    Vision2DConfig,
    Vision2DResult,
)

__all__ = [
    "PixelScale",
    "RejectedTarget",
    "TargetMeasurement",
    "Vision2DAnalysis",
    "Vision2DConfig",
    "Vision2DResult",
]
```

- [ ] **Step 4: Run model tests and the existing models regression**

Run:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_models.py `
  tests/test_vision_platform/test_models.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add vision_platform/vision2d tests/test_vision2d
git commit -m "feat(vision2d): define immutable analysis contracts"
```

## Task 2: Deterministic original synthetic-image factory

**Files:**
- Create: `tests/test_vision2d/synthetic_factory.py`
- Create: `tests/test_vision2d/test_synthetic_factory.py`

- [ ] **Step 1: Write the failing factory tests**

Create `tests/test_vision2d/test_synthetic_factory.py`:

```python
import numpy as np

from tests.test_vision2d.synthetic_factory import (
    SyntheticObject,
    add_gaussian_noise,
    adjust_brightness,
    make_scene,
)


def test_scene_factory_returns_original_image_and_truth():
    objects = (
        SyntheticObject("rectangle", "red", (90, 80), (60, 30), 30.0),
        SyntheticObject("circle", "blue", (220, 150), (40, 40), 0.0),
    )

    image, truth = make_scene((320, 240), objects)

    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8
    assert [item.shape for item in truth] == ["rectangle", "circle"]
    assert [item.color for item in truth] == ["red", "blue"]


def test_noise_is_reproducible_for_fixed_seed():
    image, _ = make_scene(
        (160, 120),
        (SyntheticObject("square", "green", (80, 60), (40, 40), 15.0),),
    )

    first = add_gaussian_noise(image, sigma=4.0, seed=20260731)
    second = add_gaussian_noise(image, sigma=4.0, seed=20260731)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)


def test_brightness_adjustment_is_explicit_and_deterministic():
    image, _ = make_scene(
        (160, 120),
        (SyntheticObject("rectangle", "yellow", (80, 60), (50, 30), 0.0),),
    )

    adjusted = adjust_brightness(image, factor=0.65)

    assert adjusted.dtype == np.uint8
    assert adjusted.shape == image.shape
    assert np.array_equal(adjusted, adjust_brightness(image, factor=0.65))
    assert not np.array_equal(adjusted, image)
```

- [ ] **Step 2: Run the factory tests and verify the missing-module failure**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_synthetic_factory.py -q
```

Expected: collection fails because `synthetic_factory` does not exist.

- [ ] **Step 3: Implement the deterministic test factory**

Create `tests/test_vision2d/synthetic_factory.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


COLORS_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "gray": (160, 160, 160),
}


@dataclass(frozen=True)
class SyntheticObject:
    shape: str
    color: str
    center_px: tuple[int, int]
    size_px: tuple[int, int]
    angle_deg: float


def make_scene(
    image_size: tuple[int, int],
    objects: tuple[SyntheticObject, ...],
    *,
    background_bgr: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, tuple[SyntheticObject, ...]]:
    width, height = image_size
    image = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    for item in objects:
        color = COLORS_BGR[item.color]
        if item.shape in {"rectangle", "square"}:
            rectangle = (item.center_px, item.size_px, item.angle_deg)
            points = np.rint(cv2.boxPoints(rectangle)).astype(np.int32)
            cv2.fillConvexPoly(image, points, color)
        elif item.shape == "circle":
            cv2.circle(image, item.center_px, item.size_px[0] // 2, color, -1)
        elif item.shape == "triangle":
            radius = item.size_px[0] / 2.0
            angles = np.deg2rad(np.array([-90.0, 30.0, 150.0]) + item.angle_deg)
            points = np.column_stack(
                (
                    item.center_px[0] + radius * np.cos(angles),
                    item.center_px[1] + radius * np.sin(angles),
                )
            )
            cv2.fillConvexPoly(image, np.rint(points).astype(np.int32), color)
        else:
            raise ValueError(f"unsupported synthetic shape: {item.shape}")
    return image, objects


def add_gaussian_noise(
    image_bgr: np.ndarray,
    *,
    sigma: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image_bgr.shape)
    return np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def adjust_brightness(image_bgr: np.ndarray, *, factor: float) -> np.ndarray:
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("factor must be finite positive")
    return np.clip(
        image_bgr.astype(np.float32) * float(factor),
        0,
        255,
    ).astype(np.uint8)
```

- [ ] **Step 4: Run the factory and model tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d -q
```

Expected: all current `test_vision2d` tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add tests/test_vision2d
git commit -m "test(vision2d): add deterministic synthetic fixtures"
```

## Task 3: Input validation and preprocessing pipeline

**Files:**
- Create: `vision_platform/vision2d/preprocessing.py`
- Create: `tests/test_vision2d/test_preprocessing.py`

- [ ] **Step 1: Write preprocessing tests**

Create `tests/test_vision2d/test_preprocessing.py`:

```python
import numpy as np
import pytest

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d.models import Vision2DConfig
from vision_platform.vision2d.preprocessing import preprocess_image


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((20, 30), dtype=np.uint8),
        np.zeros((20, 30, 4), dtype=np.uint8),
        np.zeros((20, 30, 3), dtype=np.float32),
        np.zeros((0, 30, 3), dtype=np.uint8),
    ],
)
def test_preprocess_rejects_non_uint8_bgr_images(image):
    with pytest.raises((TypeError, ValueError), match="BGR|uint8|non-empty"):
        preprocess_image(image, Vision2DConfig())


def test_preprocess_returns_named_images_without_mutating_input():
    image, _ = make_scene(
        (240, 180),
        (SyntheticObject("rectangle", "red", (120, 90), (80, 40), 0.0),),
    )
    before = image.copy()

    output = preprocess_image(image, Vision2DConfig())

    assert set(output) == {"gray", "hsv", "foreground_mask", "cleaned_mask"}
    assert output["gray"].shape == image.shape[:2]
    assert output["hsv"].shape == image.shape
    assert output["cleaned_mask"].dtype == np.uint8
    assert output["cleaned_mask"][90, 120] == 255
    assert np.array_equal(image, before)
```

- [ ] **Step 2: Verify the tests fail because preprocessing is missing**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_preprocessing.py -q
```

Expected: import failure for `vision_platform.vision2d.preprocessing`.

- [ ] **Step 3: Implement preprocessing**

Create `vision_platform/vision2d/preprocessing.py`:

```python
from __future__ import annotations

import cv2
import numpy as np

from vision_platform.vision2d.models import Vision2DConfig


def validate_bgr_image(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy BGR array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have non-empty HxWx3 BGR shape")
    if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
        raise ValueError("image_bgr must be non-empty")
    if image_bgr.dtype != np.uint8:
        raise TypeError("image_bgr must use uint8 BGR values")


def preprocess_image(
    image_bgr: np.ndarray,
    config: Vision2DConfig,
) -> dict[str, np.ndarray]:
    validate_bgr_image(image_bgr)
    blurred = cv2.GaussianBlur(
        image_bgr,
        (config.gaussian_kernel_size, config.gaussian_kernel_size),
        0,
    )
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    foreground = cv2.inRange(
        hsv,
        np.array([0, config.saturation_min, config.value_min], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel_size, config.morphology_kernel_size),
    )
    cleaned = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return {
        "gray": gray,
        "hsv": hsv,
        "foreground_mask": foreground,
        "cleaned_mask": cleaned,
    }
```

- [ ] **Step 4: Run preprocessing and existing color-shape tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_preprocessing.py `
  tests/test_vision_platform/test_color_shape_recognition.py -q
```

Expected: all tests pass; the existing recognizer remains unchanged.

- [ ] **Step 5: Commit Task 3**

```powershell
git add vision_platform/vision2d/preprocessing.py tests/test_vision2d/test_preprocessing.py
git commit -m "feat(vision2d): add validated image preprocessing"
```

## Task 4: Multi-object contour segmentation and explicit rejection

**Files:**
- Create: `vision_platform/vision2d/segmentation.py`
- Create: `tests/test_vision2d/test_segmentation.py`

- [ ] **Step 1: Write segmentation tests**

Create `tests/test_vision2d/test_segmentation.py`:

```python
import cv2
import numpy as np

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d.models import Vision2DConfig
from vision_platform.vision2d.preprocessing import preprocess_image
from vision_platform.vision2d.segmentation import segment_contours


def test_segmentation_returns_two_independent_contours():
    image, _ = make_scene(
        (320, 240),
        (
            SyntheticObject("rectangle", "red", (80, 80), (60, 30), 0.0),
            SyntheticObject("circle", "blue", (240, 160), (50, 50), 0.0),
        ),
    )
    processed = preprocess_image(image, Vision2DConfig())

    accepted, rejected = segment_contours(
        processed["cleaned_mask"], Vision2DConfig()
    )

    assert len(accepted) == 2
    assert rejected == ()


def test_segmentation_rejects_small_and_border_touching_candidates():
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(mask, (0, 30), (40, 90), 255, -1)
    cv2.circle(mask, (150, 100), 3, 255, -1)
    config = Vision2DConfig(min_area_ratio=0.001, border_margin_px=1)

    accepted, rejected = segment_contours(mask, config)

    assert accepted == ()
    assert {item.code for item in rejected} == {
        "AREA_TOO_SMALL",
        "TOUCHES_IMAGE_BORDER",
    }


def test_segmentation_rejects_degenerate_line_contour():
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.line(mask, (150, 30), (150, 170), 255, 1)

    accepted, rejected = segment_contours(
        mask,
        Vision2DConfig(min_area_ratio=0.000001),
    )

    assert accepted == ()
    assert [item.code for item in rejected] == [
        "INSUFFICIENT_CONTOUR_POINTS"
    ]
```

- [ ] **Step 2: Run and confirm the missing segmentation module**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_segmentation.py -q
```

Expected: import failure for `vision_platform.vision2d.segmentation`.

- [ ] **Step 3: Implement contour segmentation**

Create `vision_platform/vision2d/segmentation.py`:

```python
from __future__ import annotations

import cv2
import numpy as np

from vision_platform.vision2d.models import RejectedTarget, Vision2DConfig


def _touches_border(
    contour: np.ndarray,
    *,
    width: int,
    height: int,
    margin: int,
) -> bool:
    points = contour.reshape(-1, 2)
    return bool(
        np.any(points[:, 0] <= margin)
        or np.any(points[:, 0] >= width - 1 - margin)
        or np.any(points[:, 1] <= margin)
        or np.any(points[:, 1] >= height - 1 - margin)
    )


def _center_or_none(contour: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        return None
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def segment_contours(
    cleaned_mask: np.ndarray,
    config: Vision2DConfig,
) -> tuple[tuple[np.ndarray, ...], tuple[RejectedTarget, ...]]:
    if cleaned_mask.ndim != 2 or cleaned_mask.dtype != np.uint8:
        raise ValueError("cleaned_mask must be a uint8 HxW image")
    height, width = cleaned_mask.shape
    image_area = float(width * height)
    minimum = image_area * config.min_area_ratio
    maximum = image_area * config.max_area_ratio
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    accepted: list[np.ndarray] = []
    rejected: list[RejectedTarget] = []
    for index, contour in enumerate(contours, start=1):
        area = float(cv2.contourArea(contour))
        code = None
        reason = None
        if len(contour) < 3:
            code = "INSUFFICIENT_CONTOUR_POINTS"
            reason = "candidate contour has fewer than three points"
        elif area < minimum:
            code = "AREA_TOO_SMALL"
            reason = "candidate area is below configured minimum"
        elif area > maximum:
            code = "AREA_TOO_LARGE"
            reason = "candidate area is above configured maximum"
        elif _touches_border(
            contour,
            width=width,
            height=height,
            margin=config.border_margin_px,
        ):
            code = "TOUCHES_IMAGE_BORDER"
            reason = "candidate touches the configured image border"
        elif _center_or_none(contour) is None:
            code = "DEGENERATE_MOMENT"
            reason = "candidate contour has a degenerate spatial moment"
        else:
            _, (box_width, box_height), _ = cv2.minAreaRect(contour)
            if box_width <= 0.0 or box_height <= 0.0:
                code = "DEGENERATE_ROTATED_BOX"
                reason = "candidate rotated box has a zero-length edge"
        if code is None:
            accepted.append(contour.copy())
        else:
            rejected.append(
                RejectedTarget(
                    candidate_index=index,
                    center_px=_center_or_none(contour),
                    area_px2=area,
                    code=code,
                    reason=reason,
                    contour=contour.copy(),
                )
            )
    return tuple(accepted), tuple(rejected)
```

- [ ] **Step 4: Run segmentation regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_segmentation.py `
  tests/test_vision2d/test_preprocessing.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add vision_platform/vision2d/segmentation.py tests/test_vision2d/test_segmentation.py
git commit -m "feat(vision2d): segment and reject contour candidates"
```

## Task 5: Geometry, angle semantics, and calibrated units

**Files:**
- Create: `vision_platform/vision2d/geometry.py`
- Create: `tests/test_vision2d/test_geometry.py`

- [ ] **Step 1: Write geometry tests**

Create `tests/test_vision2d/test_geometry.py`:

```python
import cv2
import numpy as np
import pytest

from vision_platform.vision2d.geometry import measure_geometry
from vision_platform.vision2d.models import PixelScale


def _rectangle_contour(
    center=(100.0, 80.0), size=(60.0, 30.0), angle=30.0
):
    points = cv2.boxPoints((center, size, angle))
    return points.reshape(-1, 1, 2).astype(np.float32)


def _axis_error(actual: float, expected: float) -> float:
    direct = abs(actual - expected) % 180.0
    return min(direct, 180.0 - direct)


def test_geometry_measures_center_edges_area_perimeter_and_axis_angle():
    geometry = measure_geometry(_rectangle_contour(), pixel_scale=None)

    assert geometry.center_px == pytest.approx((100.0, 80.0), abs=1e-3)
    assert geometry.long_side_px == pytest.approx(60.0, abs=1e-3)
    assert geometry.short_side_px == pytest.approx(30.0, abs=1e-3)
    assert _axis_error(geometry.angle_deg, 30.0) <= 1e-3
    assert geometry.area_px2 == pytest.approx(1800.0, rel=1e-3)
    assert geometry.perimeter_px == pytest.approx(180.0, rel=1e-3)
    assert geometry.size_mm is None
    assert geometry.area_mm2 is None
    assert geometry.perimeter_mm is None


def test_geometry_uses_anisotropic_scale_for_edges_area_and_perimeter():
    contour = _rectangle_contour(size=(40.0, 20.0), angle=0.0)
    scale = PixelScale(mm_per_pixel_x=0.2, mm_per_pixel_y=0.5)

    geometry = measure_geometry(contour, pixel_scale=scale)

    assert geometry.size_mm == pytest.approx((8.0, 10.0), abs=1e-3)
    assert geometry.area_mm2 == pytest.approx(80.0, rel=1e-3)
    assert geometry.perimeter_mm == pytest.approx(36.0, rel=1e-3)
```

- [ ] **Step 2: Run and verify the missing geometry module**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d/test_geometry.py -q
```

Expected: import failure for `vision_platform.vision2d.geometry`.

- [ ] **Step 3: Implement geometry and physical conversion**

Create `vision_platform/vision2d/geometry.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision_platform.vision2d.models import PixelScale


@dataclass(frozen=True)
class GeometryMeasurement:
    center_px: tuple[float, float]
    axis_aligned_bbox_px: tuple[int, int, int, int]
    rotated_box_px: tuple[tuple[float, float], ...]
    long_side_px: float
    short_side_px: float
    angle_deg: float
    area_px2: float
    perimeter_px: float
    size_mm: tuple[float, float] | None
    area_mm2: float | None
    perimeter_mm: float | None


def _normalize_axis_angle(angle_deg: float) -> float:
    return ((float(angle_deg) + 90.0) % 180.0) - 90.0


def _scaled_length(vector: np.ndarray, scale: PixelScale) -> float:
    return math.hypot(
        float(vector[0]) * scale.mm_per_pixel_x,
        float(vector[1]) * scale.mm_per_pixel_y,
    )


def measure_geometry(
    contour: np.ndarray,
    *,
    pixel_scale: PixelScale | None,
) -> GeometryMeasurement:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        raise ValueError("contour has degenerate moment")
    center = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    x, y, width, height = cv2.boundingRect(contour)
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
    edges = [box[(index + 1) % 4] - box[index] for index in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_index = int(np.argmax(lengths))
    short_index = int(np.argmin(lengths))
    long_edge = edges[long_index]
    short_edge = edges[short_index]
    long_side = lengths[long_index]
    short_side = lengths[short_index]
    angle = _normalize_axis_angle(
        math.degrees(math.atan2(float(long_edge[1]), float(long_edge[0])))
    )
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    size_mm = None
    area_mm2 = None
    perimeter_mm = None
    if pixel_scale is not None:
        size_mm = (
            _scaled_length(long_edge, pixel_scale),
            _scaled_length(short_edge, pixel_scale),
        )
        area_mm2 = (
            area
            * pixel_scale.mm_per_pixel_x
            * pixel_scale.mm_per_pixel_y
        )
        points = contour.reshape(-1, 2).astype(np.float64)
        closed = np.vstack((points, points[0]))
        perimeter_mm = sum(
            _scaled_length(vector, pixel_scale)
            for vector in np.diff(closed, axis=0)
        )
    return GeometryMeasurement(
        center_px=(round(center[0], 6), round(center[1], 6)),
        axis_aligned_bbox_px=(int(x), int(y), int(width), int(height)),
        rotated_box_px=tuple(
            (round(float(point[0]), 6), round(float(point[1]), 6))
            for point in box
        ),
        long_side_px=long_side,
        short_side_px=short_side,
        angle_deg=angle,
        area_px2=area,
        perimeter_px=perimeter,
        size_mm=size_mm,
        area_mm2=area_mm2,
        perimeter_mm=perimeter_mm,
    )
```

- [ ] **Step 4: Run geometry and model tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_geometry.py `
  tests/test_vision2d/test_models.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add vision_platform/vision2d/geometry.py tests/test_vision2d/test_geometry.py
git commit -m "feat(vision2d): measure geometry and calibrated units"
```

## Task 6: Color, shape, and contour appearance features

**Files:**
- Create: `vision_platform/vision2d/appearance.py`
- Create: `tests/test_vision2d/test_appearance.py`

- [ ] **Step 1: Write appearance tests**

Create `tests/test_vision2d/test_appearance.py`:

```python
import cv2
import numpy as np

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d.appearance import measure_appearance
from vision_platform.vision2d.models import Vision2DConfig


def _single_contour(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([179, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return hsv, max(contours, key=cv2.contourArea)


def test_appearance_labels_supported_colors_and_shapes():
    cases = (
        ("red", "rectangle"),
        ("yellow", "square"),
        ("green", "triangle"),
        ("blue", "circle"),
    )
    for color, shape in cases:
        size = (60, 35) if shape == "rectangle" else (50, 50)
        image, _ = make_scene(
            (180, 160),
            (SyntheticObject(shape, color, (90, 80), size, 0.0),),
        )
        hsv, contour = _single_contour(image)

        result = measure_appearance(hsv, contour, Vision2DConfig())

        assert result.color == color
        assert result.shape == shape


def test_low_saturation_target_is_unknown_color():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (120, 90), (160, 160, 160), -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contour = np.array([[[40, 30]], [[120, 30]], [[120, 90]], [[40, 90]]])

    result = measure_appearance(hsv, contour, Vision2DConfig())

    assert result.color == "unknown"
    assert result.shape == "rectangle"


def test_irregular_concave_contour_is_polygon():
    image = np.zeros((140, 180, 3), dtype=np.uint8)
    points = np.array(
        [[20, 50], [80, 20], [160, 30], [105, 65], [155, 115], [65, 100]],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [points], (0, 255, 0))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contour = points.reshape(-1, 1, 2)

    result = measure_appearance(hsv, contour, Vision2DConfig())

    assert result.color == "green"
    assert result.shape == "polygon"
```

- [ ] **Step 2: Verify the missing appearance module**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d/test_appearance.py -q
```

Expected: import failure for `vision_platform.vision2d.appearance`.

- [ ] **Step 3: Implement appearance measurement**

Create `vision_platform/vision2d/appearance.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision_platform.vision2d.models import Vision2DConfig


@dataclass(frozen=True)
class AppearanceMeasurement:
    color: str
    shape: str
    vertex_count: int
    circularity: float
    aspect_ratio: float


def _color_label(
    hsv: np.ndarray,
    contour: np.ndarray,
    *,
    saturation_min: int,
) -> str:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        return "unknown"
    hue = float(np.median(pixels[:, 0]))
    saturation = float(np.median(pixels[:, 1]))
    if saturation < float(saturation_min):
        return "unknown"
    if hue <= 10.0 or hue >= 170.0:
        return "red"
    if 15.0 <= hue <= 40.0:
        return "yellow"
    if 40.0 < hue <= 90.0:
        return "green"
    if 90.0 < hue <= 140.0:
        return "blue"
    return "unknown"


def measure_appearance(
    hsv: np.ndarray,
    contour: np.ndarray,
    config: Vision2DConfig,
) -> AppearanceMeasurement:
    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(contour))
    approximation = cv2.approxPolyDP(
        contour,
        config.polygon_epsilon_ratio * perimeter,
        True,
    )
    vertex_count = len(approximation)
    _, (width, height), _ = cv2.minAreaRect(contour)
    longer = max(float(width), float(height))
    shorter = min(float(width), float(height))
    aspect_ratio = shorter / longer if longer > 0.0 else 0.0
    circularity = (
        4.0 * math.pi * area / (perimeter * perimeter)
        if perimeter > 0.0
        else 0.0
    )
    if vertex_count == 3:
        shape = "triangle"
    elif vertex_count == 4:
        shape = "square" if aspect_ratio >= config.square_aspect_min else "rectangle"
    elif vertex_count >= 5 and circularity >= config.circle_circularity_min:
        shape = "circle"
    elif vertex_count >= 5:
        shape = "polygon"
    else:
        shape = "unknown"
    return AppearanceMeasurement(
        color=_color_label(
            hsv,
            contour,
            saturation_min=config.saturation_min,
        ),
        shape=shape,
        vertex_count=vertex_count,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
    )
```

- [ ] **Step 4: Run appearance and一期 recognizer tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_appearance.py `
  tests/test_vision_platform/test_color_shape_recognition.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

```powershell
git add vision_platform/vision2d/appearance.py tests/test_vision2d/test_appearance.py
git commit -m "feat(vision2d): classify color and contour shape"
```

## Task 7: Shared multi-object analysis pipeline

**Files:**
- Create: `vision_platform/vision2d/pipeline.py`
- Create: `tests/test_vision2d/test_pipeline.py`
- Modify: `vision_platform/vision2d/__init__.py`

- [ ] **Step 1: Write end-to-end pipeline tests**

Create `tests/test_vision2d/test_pipeline.py`:

```python
import numpy as np

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d import Vision2DConfig, analyze_image


def test_pipeline_returns_spatially_sorted_multi_object_results():
    image, _ = make_scene(
        (360, 260),
        (
            SyntheticObject("circle", "blue", (260, 190), (50, 50), 0.0),
            SyntheticObject("rectangle", "red", (90, 70), (70, 35), 30.0),
        ),
    )
    before = image.copy()

    analysis = analyze_image(image, Vision2DConfig())

    assert analysis.result.status == "PASS"
    assert [item.detection_id for item in analysis.result.targets] == [
        "det-001",
        "det-002",
    ]
    assert [(item.color, item.shape) for item in analysis.result.targets] == [
        ("red", "rectangle"),
        ("blue", "circle"),
    ]
    assert analysis.result.targets[1].angle_deg is None
    assert "ANGLE_UNDEFINED_FOR_CIRCLE" in analysis.result.targets[1].quality_flags
    assert set(analysis.intermediate_images) == {
        "gray",
        "hsv",
        "foreground_mask",
        "cleaned_mask",
        "annotated",
    }
    assert np.array_equal(image, before)


def test_pipeline_distinguishes_no_targets_and_rejected_candidates():
    empty = np.zeros((200, 300, 3), dtype=np.uint8)
    assert analyze_image(empty).result.status == "NO_TARGETS"

    border, _ = make_scene(
        (300, 200),
        (SyntheticObject("rectangle", "red", (20, 70), (40, 50), 0.0),),
    )
    assert analyze_image(border).result.status == "REJECTED"


def test_pipeline_reports_partial_for_valid_and_rejected_candidates():
    image, _ = make_scene(
        (300, 200),
        (
            SyntheticObject("rectangle", "red", (170, 100), (70, 40), 0.0),
            SyntheticObject("circle", "blue", (60, 60), (8, 8), 0.0),
        ),
    )

    result = analyze_image(
        image,
        Vision2DConfig(
            min_area_ratio=0.002,
            gaussian_kernel_size=1,
            morphology_kernel_size=1,
        ),
    ).result

    assert result.status == "PARTIAL"
    assert len(result.targets) == 1
    assert {item.code for item in result.rejected_targets} == {"AREA_TOO_SMALL"}
```

- [ ] **Step 2: Run and verify `analyze_image` is missing**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d/test_pipeline.py -q
```

Expected: import error for `analyze_image`.

- [ ] **Step 3: Implement the pipeline**

Create `vision_platform/vision2d/pipeline.py`:

```python
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from vision_platform.vision2d.appearance import measure_appearance
from vision_platform.vision2d.geometry import measure_geometry
from vision_platform.vision2d.models import (
    TargetMeasurement,
    Vision2DAnalysis,
    Vision2DConfig,
    Vision2DResult,
)
from vision_platform.vision2d.preprocessing import preprocess_image
from vision_platform.vision2d.segmentation import segment_contours


def _status(target_count: int, rejected_count: int) -> str:
    if target_count and rejected_count:
        return "PARTIAL"
    if target_count:
        return "PASS"
    if rejected_count:
        return "REJECTED"
    return "NO_TARGETS"


def analyze_image(
    image_bgr: np.ndarray,
    config: Vision2DConfig | None = None,
) -> Vision2DAnalysis:
    effective = config or Vision2DConfig()
    intermediate = preprocess_image(image_bgr, effective)
    contours, rejected = segment_contours(
        intermediate["cleaned_mask"], effective
    )
    targets: list[TargetMeasurement] = []
    for contour in contours:
        geometry = measure_geometry(
            contour,
            pixel_scale=effective.pixel_scale,
        )
        appearance = measure_appearance(
            intermediate["hsv"], contour, effective
        )
        flags: list[str] = []
        angle = geometry.angle_deg
        if appearance.shape == "circle":
            angle = None
            flags.append("ANGLE_UNDEFINED_FOR_CIRCLE")
        elif appearance.shape == "square":
            flags.append("ANGLE_AMBIGUOUS_FOR_SQUARE")
        targets.append(
            TargetMeasurement(
                detection_id="pending",
                center_px=geometry.center_px,
                axis_aligned_bbox_px=geometry.axis_aligned_bbox_px,
                rotated_box_px=geometry.rotated_box_px,
                long_side_px=geometry.long_side_px,
                short_side_px=geometry.short_side_px,
                angle_deg=angle,
                area_px2=geometry.area_px2,
                perimeter_px=geometry.perimeter_px,
                size_mm=geometry.size_mm,
                area_mm2=geometry.area_mm2,
                perimeter_mm=geometry.perimeter_mm,
                color=appearance.color,
                shape=appearance.shape,
                vertex_count=appearance.vertex_count,
                circularity=appearance.circularity,
                aspect_ratio=appearance.aspect_ratio,
                quality_flags=tuple(flags),
                contour=contour.copy(),
            )
        )
    targets.sort(
        key=lambda item: (item.center_px[1], item.center_px[0], item.area_px2)
    )
    numbered = tuple(
        replace(item, detection_id=f"det-{index:03d}")
        for index, item in enumerate(targets, start=1)
    )
    annotated = image_bgr.copy()
    for item in numbered:
        cv2.drawContours(annotated, [item.contour], -1, (255, 255, 255), 2)
        center = tuple(int(round(value)) for value in item.center_px)
        cv2.circle(annotated, center, 3, (255, 255, 255), -1)
        cv2.putText(
            annotated,
            f"{item.detection_id} {item.color}/{item.shape}",
            (max(0, center[0] - 50), max(16, center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    result = Vision2DResult(
        status=_status(len(numbered), len(rejected)),
        image_size=(int(image_bgr.shape[1]), int(image_bgr.shape[0])),
        targets=numbered,
        rejected_targets=rejected,
    )
    return Vision2DAnalysis(
        result=result,
        intermediate_images={**intermediate, "annotated": annotated},
    )
```

Append to `vision_platform/vision2d/__init__.py`:

```python
from vision_platform.vision2d.pipeline import analyze_image

__all__.append("analyze_image")
```

- [ ] **Step 4: Run the complete algorithm suite so far**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d -q
```

Expected: all current `test_vision2d` tests pass.

- [ ] **Step 5: Commit Task 7**

```powershell
git add vision_platform/vision2d tests/test_vision2d/test_pipeline.py
git commit -m "feat(vision2d): compose deterministic analysis pipeline"
```

## Task 8: JSON-safe serialization without image leakage

**Files:**
- Create: `vision_platform/vision2d/serialization.py`
- Create: `tests/test_vision2d/test_serialization.py`
- Modify: `vision_platform/vision2d/__init__.py`

- [ ] **Step 1: Write serialization tests**

Create `tests/test_vision2d/test_serialization.py`:

```python
import json

import numpy as np

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d import analyze_image
from vision_platform.vision2d.serialization import result_to_dict


def _contains_numpy(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return True
    if isinstance(value, dict):
        return any(_contains_numpy(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numpy(item) for item in value)
    return False


def test_result_serialization_is_json_safe_and_excludes_intermediate_images():
    image, _ = make_scene(
        (240, 180),
        (SyntheticObject("rectangle", "red", (120, 90), (70, 35), 15.0),),
    )
    analysis = analyze_image(image)

    payload = result_to_dict(analysis.result)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert payload["schema_version"] == 1
    assert payload["targets"][0]["detection_id"] == "det-001"
    assert "intermediate_images" not in payload
    assert not _contains_numpy(payload)
    assert encoded == json.dumps(
        result_to_dict(analyze_image(image).result),
        ensure_ascii=False,
        sort_keys=True,
    )
```

- [ ] **Step 2: Run and verify serialization is missing**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_serialization.py -q
```

Expected: import failure for `vision_platform.vision2d.serialization`.

- [ ] **Step 3: Implement explicit serialization**

Create `vision_platform/vision2d/serialization.py`:

```python
from __future__ import annotations

from typing import Any

from vision_platform.vision2d.models import (
    RejectedTarget,
    TargetMeasurement,
    Vision2DResult,
)


def _target_to_dict(target: TargetMeasurement) -> dict[str, Any]:
    return {
        "detection_id": target.detection_id,
        "center_px": list(target.center_px),
        "axis_aligned_bbox_px": list(target.axis_aligned_bbox_px),
        "rotated_box_px": [list(point) for point in target.rotated_box_px],
        "long_side_px": float(target.long_side_px),
        "short_side_px": float(target.short_side_px),
        "angle_deg": None if target.angle_deg is None else float(target.angle_deg),
        "area_px2": float(target.area_px2),
        "perimeter_px": float(target.perimeter_px),
        "size_mm": None if target.size_mm is None else list(target.size_mm),
        "area_mm2": target.area_mm2,
        "perimeter_mm": target.perimeter_mm,
        "color": target.color,
        "shape": target.shape,
        "vertex_count": int(target.vertex_count),
        "circularity": float(target.circularity),
        "aspect_ratio": float(target.aspect_ratio),
        "quality_flags": list(target.quality_flags),
        "contour_px": target.contour.reshape(-1, 2).tolist(),
    }


def _rejected_to_dict(target: RejectedTarget) -> dict[str, Any]:
    return {
        "candidate_index": int(target.candidate_index),
        "center_px": None if target.center_px is None else list(target.center_px),
        "area_px2": float(target.area_px2),
        "code": target.code,
        "reason": target.reason,
        "contour_px": target.contour.reshape(-1, 2).tolist(),
    }


def result_to_dict(result: Vision2DResult) -> dict[str, Any]:
    return {
        "schema_version": int(result.schema_version),
        "status": result.status,
        "image_size": list(result.image_size),
        "targets": [_target_to_dict(item) for item in result.targets],
        "rejected_targets": [
            _rejected_to_dict(item) for item in result.rejected_targets
        ],
    }
```

Append to `vision_platform/vision2d/__init__.py`:

```python
from vision_platform.vision2d.serialization import result_to_dict

__all__.append("result_to_dict")
```

- [ ] **Step 4: Run serialization and pipeline tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_serialization.py `
  tests/test_vision2d/test_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 8**

```powershell
git add vision_platform/vision2d tests/test_vision2d/test_serialization.py
git commit -m "feat(vision2d): serialize analysis results safely"
```

## Task 9: Precision, robustness, and一期 compatibility contracts

**Files:**
- Create: `tests/test_vision2d/test_compatibility.py`
- Modify: algorithm files only if a new test exposes a defect

- [ ] **Step 1: Add the precision and compatibility tests**

Create `tests/test_vision2d/test_compatibility.py`:

```python
from pathlib import Path

import pytest

from tests.test_vision2d.synthetic_factory import (
    SyntheticObject,
    add_gaussian_noise,
    adjust_brightness,
    make_scene,
)
from vision_platform.recognition.color_shape import ColorShapeRecognizer
from vision_platform.vision2d import analyze_image


def _axis_error(actual: float, expected: float) -> float:
    direct = abs(actual - expected) % 180.0
    return min(direct, 180.0 - direct)


def test_clean_rotated_rectangle_meets_precision_budget():
    image, _ = make_scene(
        (400, 300),
        (SyntheticObject("rectangle", "red", (180, 140), (100, 50), 30.0),),
    )

    target = analyze_image(image).result.targets[0]

    assert target.center_px == pytest.approx((180.0, 140.0), abs=1.5)
    assert target.long_side_px == pytest.approx(100.0, abs=2.0)
    assert target.short_side_px == pytest.approx(50.0, abs=2.0)
    assert _axis_error(target.angle_deg, 30.0) <= 2.0
    assert target.area_px2 == pytest.approx(5000.0, rel=0.05)
    assert target.perimeter_px == pytest.approx(300.0, rel=0.05)


def test_light_noise_preserves_target_count_and_labels():
    image, _ = make_scene(
        (360, 260),
        (
            SyntheticObject("rectangle", "yellow", (90, 80), (70, 35), 15.0),
            SyntheticObject("circle", "blue", (250, 180), (50, 50), 0.0),
        ),
    )
    noisy = add_gaussian_noise(image, sigma=4.0, seed=20260731)
    noisy = adjust_brightness(noisy, factor=0.75)

    result = analyze_image(noisy).result

    assert result.status == "PASS"
    assert [(item.color, item.shape) for item in result.targets] == [
        ("yellow", "rectangle"),
        ("blue", "circle"),
    ]


def test_new_and_existing_recognizers_share_core_label_semantics():
    image, _ = make_scene(
        (360, 260),
        (
            SyntheticObject("square", "green", (90, 80), (50, 50), 0.0),
            SyntheticObject("circle", "blue", (250, 180), (50, 50), 0.0),
        ),
    )

    legacy = ColorShapeRecognizer().detect(image)
    modern = analyze_image(image).result.targets

    assert [(item.detection_id, item.color, item.shape) for item in modern] == [
        (item.detection_id, item.color, item.shape) for item in legacy
    ]


def test_runtime_package_does_not_import_legacy_private_implementation():
    package = Path(__file__).resolve().parents[2] / "vision_platform" / "vision2d"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )

    assert "ColorShapeRecognizer" not in source
    assert "vision_platform.recognition.color_shape" not in source
```

- [ ] **Step 2: Run the compatibility tests and observe real failures**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_compatibility.py -q
```

Expected: all compatibility, precision, and light-noise tests pass. If they do not, stop this Task and use systematic debugging before changing code; the confirmed `1.5 px`, `2 px`, `2°`, and `5%` budgets may not be relaxed, and the legacy recognizer may not be modified.

- [ ] **Step 3: Run the entire Vision2D suite twice**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d -q
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d -q
```

Expected: both runs pass with identical test counts.

- [ ] **Step 4: Commit Task 9**

```powershell
git add vision_platform/vision2d tests/test_vision2d/test_compatibility.py
git commit -m "test(vision2d): enforce precision and compatibility budgets"
```

## Task 10: Four experiment guides with explicit evidence boundaries

**Files:**
- Create: `docs/experiments/V1-02.md`
- Create: `docs/experiments/V1-03.md`
- Create: `docs/experiments/V1-04.md`
- Create: `docs/experiments/V1-05.md`
- Create: `tests/test_vision2d/test_guides.py`

- [ ] **Step 1: Write the guide contract tests**

Create `tests/test_vision2d/test_guides.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_four_guides_define_algorithm_inputs_outputs_and_boundaries():
    required = {
        "V1-02": ("像素尺寸", "毫米"),
        "V1-03": ("中心", "旋转角"),
        "V1-04": ("面积", "周长"),
        "V1-05": ("颜色", "形状"),
    }
    for experiment_id, terms in required.items():
        path = ROOT / "docs" / "experiments" / f"{experiment_id}.md"
        text = path.read_text(encoding="utf-8")
        assert experiment_id in text
        assert all(term in text for term in terms)
        assert "原创合成图" in text
        assert "不是课程成绩" in text
        assert "PENDING_HARDWARE" in text
        assert "CoppeliaSim 在线验收不在本分支范围" in text
```

- [ ] **Step 2: Run and verify the four missing files**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d/test_guides.py -q
```

Expected: failure at the first missing guide path.

- [ ] **Step 3: Create `V1-02.md`**

Create `docs/experiments/V1-02.md`:

```markdown
# V1-02 像素尺寸与构件尺寸测量

## 学习目标

使用轮廓和旋转包围框测量构件长边、短边和面积，理解像素尺寸与物理尺寸的显式换算。

## 输入与步骤

输入为 BGR `uint8` 原创合成图。先观察前景掩膜和轮廓，再读取像素长短边。只有教师或实验配置显式提供 `mm_per_pixel_x` 和 `mm_per_pixel_y` 时，才解释毫米尺寸。

## 输出与复核

复核 `center_px`、`long_side_px`、`short_side_px`、`area_px2`、`size_mm` 和 `area_mm2`。无标定时毫米字段必须为空。

## 验收边界

算法状态不是课程成绩。CoppeliaSim 在线验收不在本分支范围。真实相机像素尺寸、镜头畸变和机械臂抓取均为 `PENDING_HARDWARE`。
```

- [ ] **Step 4: Create `V1-03.md`**

Create `docs/experiments/V1-03.md`:

```markdown
# V1-03 物体定位与旋转角测量

## 学习目标

使用轮廓矩计算中心，使用旋转包围框长轴计算旋转角，理解无向轴角度和姿态歧义。

## 输入与步骤

输入为包含一个或多个分离目标的 BGR `uint8` 原创合成图。观察中心、旋转框和标注图，对比设定角与测量角。

## 输出与复核

复核 `center_px`、`rotated_box_px`、`angle_deg` 和 `quality_flags`。圆形角度为空；正方形必须提示角度歧义。

## 验收边界

算法状态不是课程成绩。CoppeliaSim 在线验收不在本分支范围。相机外参、世界坐标和真实机器人定向抓取均为 `PENDING_HARDWARE`。
```

- [ ] **Step 5: Create `V1-04.md`**

Create `docs/experiments/V1-04.md`:

```markdown
# V1-04 边缘长度、周长和面积测量

## 学习目标

理解轮廓点、闭合周长、面积、圆度和长宽比，并比较噪声与分割参数对几何结果的影响。

## 输入与步骤

输入为矩形、圆形、三角形等原创合成图。依次观察灰度图、前景掩膜、清理后掩膜、轮廓和标注图。

## 输出与复核

复核 `area_px2`、`perimeter_px`、`circularity`、`aspect_ratio` 和轮廓点。提供各向异性像素尺寸时，物理周长按每段 X/Y 分量换算。

## 验收边界

算法状态不是课程成绩。CoppeliaSim 在线验收不在本分支范围。真实工件边缘、镜头误差和质量判定均为 `PENDING_HARDWARE`。
```

- [ ] **Step 6: Create `V1-05.md`**

Create `docs/experiments/V1-05.md`:

```markdown
# V1-05 颜色、形状与轮廓识别

## 学习目标

使用 HSV 色相和饱和度识别颜色，使用顶点数、长宽比和圆度识别形状，并理解 `unknown` 和拒绝结果。

## 输入与步骤

输入为红、黄、绿、蓝的矩形、正方形、三角形和圆形原创合成图。对比原图、HSV、前景掩膜和标注结果。

## 输出与复核

复核 `color`、`shape`、`vertex_count`、`circularity`、`aspect_ratio` 和 `quality_flags`。低饱和度或未定义色相返回 `unknown`。

## 验收边界

算法状态不是课程成绩。CoppeliaSim 在线验收不在本分支范围。真实水果、建材、海康相机和机器人分类均为 `PENDING_HARDWARE`。
```

- [ ] **Step 7: Run the guide tests**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d/test_guides.py -q
```

Expected: all guide tests pass.

- [ ] **Step 8: Commit Task 10**

```powershell
git add docs/experiments tests/test_vision2d/test_guides.py
git commit -m "docs(vision2d): add four measurement lab guides"
```

## Task 11: Delivery whitelist and isolated-branch contract

**Files:**
- Create: `tests/test_vision2d/test_delivery.py`
- Modify: `RETAINED_FILES.txt`

- [ ] **Step 1: Write the failing delivery test**

Create `tests/test_vision2d/test_delivery.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_retained_files_contains_complete_vision2d_kernel_delivery():
    retained = {
        line.strip()
        for line in (ROOT / "RETAINED_FILES.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    required = {
        "docs/experiments/V1-02.md",
        "docs/experiments/V1-03.md",
        "docs/experiments/V1-04.md",
        "docs/experiments/V1-05.md",
        "docs/superpowers/plans/2026-07-31-vision2d-algorithm-kernel-plan.md",
        "docs/superpowers/specs/2026-07-31-vision2d-algorithm-kernel-design.md",
        "vision_platform/vision2d/__init__.py",
        "vision_platform/vision2d/models.py",
        "vision_platform/vision2d/preprocessing.py",
        "vision_platform/vision2d/segmentation.py",
        "vision_platform/vision2d/geometry.py",
        "vision_platform/vision2d/appearance.py",
        "vision_platform/vision2d/pipeline.py",
        "vision_platform/vision2d/serialization.py",
        "tests/test_vision2d/__init__.py",
        "tests/test_vision2d/synthetic_factory.py",
        "tests/test_vision2d/test_models.py",
        "tests/test_vision2d/test_synthetic_factory.py",
        "tests/test_vision2d/test_preprocessing.py",
        "tests/test_vision2d/test_segmentation.py",
        "tests/test_vision2d/test_geometry.py",
        "tests/test_vision2d/test_appearance.py",
        "tests/test_vision2d/test_pipeline.py",
        "tests/test_vision2d/test_serialization.py",
        "tests/test_vision2d/test_compatibility.py",
        "tests/test_vision2d/test_guides.py",
        "tests/test_vision2d/test_delivery.py",
    }
    assert required <= retained
    assert all((ROOT / path).is_file() for path in required)
    assert len(retained) == len(
        [
            line.strip()
            for line in (ROOT / "RETAINED_FILES.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    )
    assert not any(path.startswith("artifacts/") for path in retained)
```

- [ ] **Step 2: Run and verify the missing whitelist entries**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d/test_delivery.py -q
```

Expected: failure showing the missing runtime, test, guide, or plan paths.

- [ ] **Step 3: Add every new formal path to `RETAINED_FILES.txt`**

Add the exact paths from the `required` set above in repository-path order. Keep all V2.1 entries. Do not add generated images, artifacts, environments, caches, or student submissions.

- [ ] **Step 4: Run delivery and existing release contracts**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_delivery.py `
  tests/test_acceptance/test_delivery_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify forbidden shared files are unchanged**

```powershell
git diff --exit-code 4bc638f -- `
  vision_platform/cli.py `
  vision_platform/application.py `
  vision_platform/session.py `
  vision_platform/student `
  vision_platform/ui `
  vision_platform/recognition/color_shape.py `
  tools/vision_lab/run_acceptance.ps1 `
  config/experiments `
  simulation
```

Expected: exit code 0 and no changed paths.

- [ ] **Step 6: Commit Task 11**

```powershell
git add RETAINED_FILES.txt tests/test_vision2d/test_delivery.py
git commit -m "docs(vision2d): retain isolated algorithm delivery"
```

## Task 12: Full regression, document QA, and clean handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-31-vision2d-algorithm-kernel-plan.md` only to mark genuinely completed checkboxes
- Evidence: `artifacts/vision_lab/v2-2-vision2d-kernel-final/`

- [ ] **Step 1: Run the complete Vision2D suite with evidence output**

```powershell
$evidence='artifacts/vision_lab/v2-2-vision2d-kernel-final'
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision2d -q 2>&1 |
  Tee-Object -FilePath (Join-Path $evidence 'pytest-vision2d.txt')
if ($LASTEXITCODE -ne 0) { throw 'Vision2D tests failed' }
```

Expected: zero failures and zero skips in `tests/test_vision2d`.

- [ ] **Step 2: Run the full static regression**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q 2>&1 |
  Tee-Object -FilePath (Join-Path $evidence 'pytest-full.txt')
if ($LASTEXITCODE -ne 0) { throw 'Full static regression failed' }
```

Expected: zero failures. Opt-in CoppeliaSim tests may be skipped, but the exact pass and skip counts must be reported and skips must not be called PASS.

- [ ] **Step 3: Verify formatting, encoding, fences, and placeholders**

```powershell
$documents=@(
  'docs/superpowers/specs/2026-07-31-vision2d-algorithm-kernel-design.md',
  'docs/superpowers/plans/2026-07-31-vision2d-algorithm-kernel-plan.md',
  'docs/experiments/V1-02.md',
  'docs/experiments/V1-03.md',
  'docs/experiments/V1-04.md',
  'docs/experiments/V1-05.md'
)
$patterns=@(('T'+'BD'),('TO'+'DO'),('FIX'+'ME'),('待'+'实现'),('待'+'定'))
foreach($document in $documents){
  $bytes=[IO.File]::ReadAllBytes((Resolve-Path $document))
  if($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF){
    throw "UTF-8 BOM is not allowed: $document"
  }
}
Get-Content $documents -Encoding UTF8 | Select-String -Pattern $patterns
git diff --check
```

Expected: no BOM, no unresolved placeholder match, and no whitespace error.

- [ ] **Step 4: Recheck publication and protected boundaries**

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision2d/test_delivery.py `
  tests/test_acceptance/test_delivery_contract.py -q
git diff --exit-code 4bc638f -- `
  vision_platform/cli.py `
  vision_platform/application.py `
  vision_platform/session.py `
  vision_platform/student `
  vision_platform/ui `
  vision_platform/recognition/color_shape.py `
  tools/vision_lab/run_acceptance.ps1 `
  config/experiments `
  simulation
```

Expected: release tests pass and forbidden shared files have zero differences.

- [ ] **Step 5: Mark only verified plan steps complete and commit the handoff**

```powershell
git add docs/superpowers/plans/2026-07-31-vision2d-algorithm-kernel-plan.md
git commit -m "docs: complete vision2d algorithm kernel delivery"
git status --short --branch
```

Expected: clean worktree.

- [ ] **Step 6: Push the isolated feature branch without merging main**

```powershell
git push -u origin codex/v2-2-vision2d-algorithm-kernel
```

Do not merge `origin/main` and do not merge into the active V2.2 first-batch integration branch. The single integration owner will cherry-pick this branch after the first-batch V2.2 acceptance is complete.

## Completion report contract

The final handoff must state:

- branch and final commit;
- Task-by-Task commit list;
- files created and modified;
- exact `tests/test_vision2d` pass count;
- exact full static pass and skip counts;
- `RETAINED_FILES.txt` total, duplicate count, and missing count;
- forbidden shared-file diff result;
- evidence directory;
- GitHub push status;
- explicit statement that CoppeliaSim, PyQt, robot, teaching-effectiveness, Hikvision, and real-hardware acceptance were not performed by this branch;
- `PENDING_HARDWARE` remains unchanged.

The branch may claim only: “V1-02 through V1-05 pure 2D vision algorithm kernel and original synthetic tests passed.”
