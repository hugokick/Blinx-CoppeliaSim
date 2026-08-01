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
