"""Reference-difference surface defect detection for deterministic 2D images."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class DefectConfig:
    """Relative thresholds and bounded morphology/alignment settings."""

    missing_ratio: float = 0.01
    hole_ratio: float = 0.005
    foreign_ratio: float = 0.002
    dimension_ratio: float = 0.08
    min_component_ratio: float = 0.002
    morphology_kernel_size: int = 3
    max_alignment_shift_px: float = 8.0
    min_contrast: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "missing_ratio",
            "hole_ratio",
            "foreign_ratio",
            "dimension_ratio",
            "min_component_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be finite in (0, 1)")
        if isinstance(self.morphology_kernel_size, (bool, np.bool_)) or not isinstance(
            self.morphology_kernel_size, (int, np.integer)
        ):
            raise ValueError("morphology_kernel_size must be a non-bool integer")
        kernel = int(self.morphology_kernel_size)
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("morphology_kernel_size must be positive odd")
        if not math.isfinite(float(self.max_alignment_shift_px)) or self.max_alignment_shift_px < 0.0:
            raise ValueError("max_alignment_shift_px must be finite non-negative")
        if not math.isfinite(float(self.min_contrast)) or self.min_contrast <= 0.0:
            raise ValueError("min_contrast must be positive")


@dataclass(frozen=True)
class DefectFinding:
    defect_type: str
    bbox_px: tuple[int, int, int, int]
    area_px2: float
    relative_area: float
    metric: float
    threshold: float
    confidence: float

    def __post_init__(self) -> None:
        if self.defect_type not in {"missing", "hole", "foreign", "broken", "dimension"}:
            raise ValueError("unsupported defect type")
        if len(self.bbox_px) != 4 or self.bbox_px[2] <= 0 or self.bbox_px[3] <= 0:
            raise ValueError("defect bbox must be positive")
        for value in (
            self.area_px2,
            self.relative_area,
            self.metric,
            self.threshold,
            self.confidence,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("defect metrics must be finite non-negative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class DefectResult:
    status: str
    defects: tuple[DefectFinding, ...]
    reference_metrics: Mapping[str, object]
    candidate_metrics: Mapping[str, object]
    alignment_shift_px: tuple[float, float]
    thresholds: Mapping[str, float]
    failure_code: str | None
    processing_ms: float
    image_size: tuple[int, int]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "PARTIAL", "NO_TARGETS", "REJECTED"}:
            raise ValueError("unsupported defect result status")
        if len(self.image_size) != 2 or any(int(value) <= 0 for value in self.image_size):
            raise ValueError("image_size must be positive")
        if len(self.alignment_shift_px) != 2 or any(
            not math.isfinite(float(value)) for value in self.alignment_shift_px
        ):
            raise ValueError("alignment shift must be finite")
        if not math.isfinite(float(self.processing_ms)) or self.processing_ms < 0.0:
            raise ValueError("processing_ms must be finite non-negative")


def _invalid_result(
    code: str,
    image_size: tuple[int, int] = (1, 1),
) -> DefectResult:
    return DefectResult(
        status="REJECTED",
        defects=(),
        reference_metrics={},
        candidate_metrics={},
        alignment_shift_px=(0.0, 0.0),
        thresholds={},
        failure_code=code,
        processing_ms=0.0,
        image_size=image_size,
    )


def _validate_image(image: object) -> tuple[np.ndarray | None, tuple[int, int]]:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        return None, (1, 1)
    if image.ndim == 2 and image.size:
        return image, (int(image.shape[1]), int(image.shape[0]))
    if image.ndim == 3 and image.shape[2] == 3 and image.size:
        return image, (int(image.shape[1]), int(image.shape[0]))
    return None, (1, 1)


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _segment(image: np.ndarray, config: DefectConfig) -> tuple[np.ndarray, float]:
    gray = _gray(image)
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    if contrast < config.min_contrast:
        return np.zeros(gray.shape, dtype=np.uint8), contrast
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    interior = gray[gray.shape[0] // 4 : (3 * gray.shape[0]) // 4, gray.shape[1] // 4 : (3 * gray.shape[1]) // 4]
    invert = float(np.median(border)) >= float(np.median(interior))
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _threshold, mask = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    kernel = np.ones((config.morphology_kernel_size, config.morphology_kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask, contrast


def _components(mask: np.ndarray, min_area: float) -> list[dict[str, object]]:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    output: list[dict[str, object]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < min_area:
            continue
        output.append(
            {
                "label": index,
                "area_px2": float(area),
                "bbox_px": (x, y, width, height),
                "center_px": (float(centroids[index][0]), float(centroids[index][1])),
            }
        )
    output.sort(key=lambda item: float(item["area_px2"]), reverse=True)
    return output


def _metrics(components: list[dict[str, object]], mask: np.ndarray) -> dict[str, object]:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    if components:
        largest = components[0]
        bbox = largest["bbox_px"]
        area = float(largest["area_px2"])
    else:
        bbox = (0, 0, 0, 0)
        area = 0.0
    return {
        "area_px2": area,
        "component_count": len(components),
        "bbox_px": bbox,
        "perimeter_px": perimeter,
        "foreground_px2": float(np.count_nonzero(mask)),
    }


def _bbox_from_contour(contour: np.ndarray) -> tuple[int, int, int, int]:
    x, y, width, height = cv2.boundingRect(contour)
    return int(x), int(y), int(width), int(height)


def _bbox_from_components(components: list[dict[str, object]]) -> tuple[int, int, int, int]:
    if not components:
        return (0, 0, 1, 1)
    boxes = [item["bbox_px"] for item in components]
    x0 = min(int(box[0]) for box in boxes)
    y0 = min(int(box[1]) for box in boxes)
    x1 = max(int(box[0]) + int(box[2]) for box in boxes)
    y1 = max(int(box[1]) + int(box[3]) for box in boxes)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _split_candidate_components(
    reference_mask: np.ndarray,
    reference_components: list[dict[str, object]],
    candidate_mask: np.ndarray,
    candidate_components: list[dict[str, object]],
    min_area: float,
    missing_ratio: float,
) -> list[dict[str, object]]:
    """Return candidate pieces when one reference subject maps to many pieces."""
    _reference_count, reference_labels, _reference_stats, _reference_centroids = (
        cv2.connectedComponentsWithStats(reference_mask, 8)
    )
    _candidate_count, candidate_labels, _candidate_stats, _candidate_centroids = (
        cv2.connectedComponentsWithStats(candidate_mask, 8)
    )
    split_labels: set[int] = set()
    for reference_component in reference_components:
        reference_pixels = reference_labels == int(reference_component["label"])
        overlap_min_area = max(
            min_area,
            float(reference_component["area_px2"]) * missing_ratio,
        )
        matching_labels: list[int] = []
        for candidate_component in candidate_components:
            candidate_label = int(candidate_component["label"])
            overlap_area = float(
                np.count_nonzero(reference_pixels & (candidate_labels == candidate_label))
            )
            candidate_area = max(float(candidate_component["area_px2"]), 1.0)
            if overlap_area >= overlap_min_area and overlap_area / candidate_area >= 0.25:
                matching_labels.append(candidate_label)
        if len(matching_labels) >= 2:
            split_labels.update(matching_labels)
    return [
        component
        for component in candidate_components
        if int(component["label"]) in split_labels
    ]


def _hole_regions(mask: np.ndarray, min_area: float) -> list[dict[str, object]]:
    """Return enclosed background contours with their aligned-image geometry."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    holes: list[dict[str, object]] = []
    for index, contour in enumerate(contours):
        if int(hierarchy[0][index][3]) < 0:
            continue
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area:
            continue
        bbox = _bbox_from_contour(contour)
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) > 1e-9:
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        else:
            center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        holes.append({"bbox_px": bbox, "area_px2": area, "center_px": center})
    return holes


def _hole_matches(reference_hole: dict[str, object], candidate_hole: dict[str, object]) -> bool:
    reference_bbox = reference_hole["bbox_px"]
    candidate_bbox = candidate_hole["bbox_px"]
    reference_center = reference_hole["center_px"]
    candidate_center = candidate_hole["center_px"]
    distance = math.dist(reference_center, candidate_center)
    size = max(float(reference_bbox[2]), float(reference_bbox[3]), 1.0)
    if distance <= max(3.0, 0.25 * size):
        return True
    x0 = max(int(reference_bbox[0]), int(candidate_bbox[0]))
    y0 = max(int(reference_bbox[1]), int(candidate_bbox[1]))
    x1 = min(int(reference_bbox[0]) + int(reference_bbox[2]), int(candidate_bbox[0]) + int(candidate_bbox[2]))
    y1 = min(int(reference_bbox[1]) + int(reference_bbox[3]), int(candidate_bbox[1]) + int(candidate_bbox[3]))
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = int(reference_bbox[2]) * int(reference_bbox[3]) + int(candidate_bbox[2]) * int(candidate_bbox[3]) - intersection
    return union > 0 and intersection / union >= 0.35


def _finding(
    defect_type: str,
    bbox: tuple[int, int, int, int],
    area: float,
    reference_area: float,
    metric: float,
    threshold: float,
) -> DefectFinding:
    relative = area / max(reference_area, 1.0)
    confidence = float(np.clip(0.55 + 0.45 * min(1.0, metric / max(threshold, 1e-6)), 0.0, 1.0))
    return DefectFinding(
        defect_type=defect_type,
        bbox_px=bbox,
        area_px2=float(area),
        relative_area=float(relative),
        metric=float(metric),
        threshold=float(threshold),
        confidence=confidence,
    )


def detect_surface_defects(
    reference_image: object,
    candidate_image: object,
    *,
    config: DefectConfig | None = None,
) -> DefectResult:
    """Compare a candidate surface to a reference using explicit thresholds."""
    started = time.perf_counter()
    if config is None:
        config = DefectConfig()
    if not isinstance(config, DefectConfig):
        return _invalid_result("CONFIG_INVALID")
    reference, reference_size = _validate_image(reference_image)
    candidate, candidate_size = _validate_image(candidate_image)
    if reference is None or candidate is None:
        return _invalid_result("INPUT_INVALID")
    if reference_size != candidate_size:
        return _invalid_result("SIZE_MISMATCH", reference_size)
    height, width = reference.shape[:2]
    image_size = (width, height)
    reference_mask, reference_contrast = _segment(reference, config)
    candidate_mask, candidate_contrast = _segment(candidate, config)
    image_area = float(height * width)
    min_area = max(4.0, image_area * config.min_component_ratio)
    reference_components = _components(reference_mask, min_area)
    candidate_components_raw = _components(candidate_mask, min_area)
    if not reference_components:
        return DefectResult(
            status="REJECTED",
            defects=(),
            reference_metrics={"contrast": reference_contrast},
            candidate_metrics={"contrast": candidate_contrast},
            alignment_shift_px=(0.0, 0.0),
            thresholds={},
            failure_code="REFERENCE_EMPTY",
            processing_ms=(time.perf_counter() - started) * 1000.0,
            image_size=image_size,
        )
    if not candidate_components_raw:
        return DefectResult(
            status="PARTIAL",
            defects=(
                _finding(
                    "missing",
                    reference_components[0]["bbox_px"],
                    float(reference_components[0]["area_px2"]),
                    float(reference_components[0]["area_px2"]),
                    float(reference_components[0]["area_px2"]),
                    max(1.0, float(reference_components[0]["area_px2"]) * config.missing_ratio),
                ),
            ),
            reference_metrics=_metrics(reference_components, reference_mask),
            candidate_metrics={"contrast": candidate_contrast, "area_px2": 0.0},
            alignment_shift_px=(0.0, 0.0),
            thresholds={"missing_ratio": config.missing_ratio},
            failure_code="CANDIDATE_EMPTY",
            processing_ms=(time.perf_counter() - started) * 1000.0,
            image_size=image_size,
        )

    reference_main = reference_components[0]
    candidate_main_raw = candidate_components_raw[0]
    reference_center = reference_main["center_px"]
    candidate_center = candidate_main_raw["center_px"]
    dx = float(candidate_center[0]) - float(reference_center[0])
    dy = float(candidate_center[1]) - float(reference_center[1])
    if abs(dx) > config.max_alignment_shift_px or abs(dy) > config.max_alignment_shift_px:
        dx = dy = 0.0
    if abs(dx) < 1.0:
        dx = 0.0
    if abs(dy) < 1.0:
        dy = 0.0
    aligned_candidate = cv2.warpAffine(
        candidate_mask,
        np.asarray([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    # All candidate-derived metrics after this point use the aligned mask and
    # are therefore expressed in the same coordinate system as the reference.
    candidate_components = _components(aligned_candidate, min_area)
    if not candidate_components:
        return _invalid_result("CANDIDATE_EMPTY_AFTER_ALIGNMENT", image_size)
    candidate_main = candidate_components[0]
    missing_mask = cv2.bitwise_and(reference_mask, cv2.bitwise_not(aligned_candidate))
    extra_mask = cv2.bitwise_and(aligned_candidate, cv2.bitwise_not(reference_mask))
    kernel = np.ones((config.morphology_kernel_size, config.morphology_kernel_size), dtype=np.uint8)
    missing_mask = cv2.morphologyEx(missing_mask, cv2.MORPH_OPEN, kernel)
    extra_mask = cv2.morphologyEx(extra_mask, cv2.MORPH_OPEN, kernel)
    reference_area = float(reference_main["area_px2"])
    reference_bbox = reference_main["bbox_px"]
    candidate_bbox = candidate_main["bbox_px"]
    candidate_area = float(candidate_main["area_px2"])
    area_ratio = abs(candidate_area - reference_area) / max(reference_area, 1.0)
    width_ratio = abs(float(candidate_bbox[2]) - float(reference_bbox[2])) / max(
        float(reference_bbox[2]), 1.0
    )
    height_ratio = abs(float(candidate_bbox[3]) - float(reference_bbox[3])) / max(
        float(reference_bbox[3]), 1.0
    )
    dimension_metric = max(area_ratio, width_ratio, height_ratio)
    thresholds = {
            "missing_ratio": float(config.missing_ratio),
            "hole_ratio": float(config.hole_ratio),
            "foreign_ratio": float(config.foreign_ratio),
            "dimension_ratio": float(config.dimension_ratio),
            "missing_px2": max(1.0, reference_area * config.missing_ratio),
            "hole_px2": max(1.0, reference_area * config.hole_ratio),
            "foreign_px2": max(1.0, image_area * config.foreign_ratio),
        }
    findings: list[DefectFinding] = []

    # A subject is broken only when one reference component maps to at least
    # two substantial candidate components. Separate legitimate reference
    # subjects therefore remain separate without manufacturing a defect.
    broken_components = _split_candidate_components(
        reference_mask,
        reference_components,
        aligned_candidate,
        candidate_components,
        min_area,
        config.missing_ratio,
    )
    broken_detected = bool(broken_components)

    # Extra foreground is the candidate-only mask.  A significant component
    # in that mask is new material, including material that fills a legal
    # reference hole.  A dimensional enlargement is handled by the explicit
    # dimension metric instead of being mislabeled as foreign material.
    foreign_components = (
        _components(extra_mask, thresholds["foreign_px2"])
        if dimension_metric < config.dimension_ratio
        else []
    )
    for component in foreign_components:
        area = float(component["area_px2"])
        findings.append(
            _finding(
                "foreign",
                component["bbox_px"],
                area,
                reference_area,
                area,
                thresholds["foreign_px2"],
            )
        )

    # A candidate hole is a defect only when it is not already present in the
    # reference.  Legal holes are part of the reference geometry contract.
    reference_holes = _hole_regions(reference_mask, thresholds["hole_px2"])
    candidate_holes = _hole_regions(aligned_candidate, thresholds["hole_px2"])
    new_hole_boxes: list[tuple[int, int, int, int]] = []
    for candidate_hole in candidate_holes:
        if any(_hole_matches(reference_hole, candidate_hole) for reference_hole in reference_holes):
            continue
        bbox = candidate_hole["bbox_px"]
        area = float(candidate_hole["area_px2"])
        new_hole_boxes.append(bbox)
        findings.append(_finding("hole", bbox, area, reference_area, area, thresholds["hole_px2"]))

    missing_components = _components(missing_mask, thresholds["missing_px2"])
    if not broken_detected:
        for component in missing_components:
            bbox = component["bbox_px"]
            center = component["center_px"]
            is_hole = any(
                int(hole[0]) <= float(center[0]) <= int(hole[0]) + int(hole[2])
                and int(hole[1]) <= float(center[1]) <= int(hole[1]) + int(hole[3])
                for hole in new_hole_boxes
            )
            if not is_hole:
                area = float(component["area_px2"])
                findings.append(
                    _finding("missing", bbox, area, reference_area, area, thresholds["missing_px2"])
                )

    if broken_detected:
        area = float(sum(float(item["area_px2"]) for item in broken_components))
        findings.append(
            _finding(
                "broken",
                _bbox_from_components(broken_components),
                area,
                reference_area,
                float(len(broken_components)),
                2.0,
            )
        )

    explicit_types = {finding.defect_type for finding in findings}
    if dimension_metric >= config.dimension_ratio and not explicit_types.intersection(
        {"missing", "hole", "foreign", "broken"}
    ):
        findings.append(
            _finding(
                "dimension",
                candidate_bbox,
                abs(candidate_area - reference_area),
                reference_area,
                dimension_metric,
                config.dimension_ratio,
            )
        )
    # Avoid duplicate findings generated by a noisy contour pass.
    unique: list[DefectFinding] = []
    for finding in findings:
        if not any(
            existing.defect_type == finding.defect_type
            and abs(existing.bbox_px[0] - finding.bbox_px[0]) <= 2
            and abs(existing.bbox_px[1] - finding.bbox_px[1]) <= 2
            for existing in unique
        ):
            unique.append(finding)
    unique.sort(key=lambda finding: (finding.defect_type, finding.bbox_px[1], finding.bbox_px[0]))
    status = "PARTIAL" if unique else "PASS"
    return DefectResult(
        status=status,
        defects=tuple(unique),
        reference_metrics=_metrics(reference_components, reference_mask),
        candidate_metrics=_metrics(candidate_components, aligned_candidate),
        alignment_shift_px=(dx, dy),
        thresholds=thresholds,
        failure_code=None if not unique else "DEFECTS_FOUND",
        processing_ms=(time.perf_counter() - started) * 1000.0,
        image_size=image_size,
    )


__all__ = ["DefectConfig", "DefectFinding", "DefectResult", "detect_surface_defects"]
