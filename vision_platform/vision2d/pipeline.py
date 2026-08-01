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
