"""Read-only teaching overlays for V1-09 defect findings."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_platform.vision2d.defect_detection import DefectResult


@dataclass(frozen=True)
class DefectTeachingLayers:
    annotated_frame: np.ndarray
    finding_mask: np.ndarray
    legend: str = "缺陷区域示意"

    def __post_init__(self) -> None:
        annotated = np.ascontiguousarray(self.annotated_frame).copy()
        mask = np.ascontiguousarray(self.finding_mask).copy()
        annotated.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "annotated_frame", annotated)
        object.__setattr__(self, "finding_mask", mask)


def render_defect_teaching_layers(frame: np.ndarray, result: DefectResult) -> DefectTeachingLayers:
    """Draw published finding bboxes only; never feeds overlays to detection."""

    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be uint8 BGR")
    if not isinstance(result, DefectResult):
        raise ValueError("result must be a DefectResult")
    annotated = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for finding in result.defects:
        x, y, width, height = finding.bbox_px
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
            raise ValueError("finding bbox is outside the frame")
        cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), (0, 0, 255), 2)
        mask[y : y + height, x : x + width] = 255
    return DefectTeachingLayers(annotated_frame=annotated, finding_mask=mask)


__all__ = ["DefectTeachingLayers", "render_defect_teaching_layers"]
