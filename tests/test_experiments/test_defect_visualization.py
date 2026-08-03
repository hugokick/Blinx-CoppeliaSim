from __future__ import annotations

import numpy as np

from vision_platform.experiments.defect_visualization import render_defect_teaching_layers
from vision_platform.vision2d.defect_detection import DefectFinding, DefectResult


def test_visualization_returns_bbox_only_mask_and_teaching_label() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = DefectResult(
        status="PARTIAL",
        defects=(DefectFinding("hole", (10, 12, 20, 18), 100.0, 0.02, 0.1, 0.05, 0.9),),
        reference_metrics={},
        candidate_metrics={},
        alignment_shift_px=(0.0, 0.0),
        thresholds={"hole_ratio": 0.005},
        failure_code="DEFECTS_FOUND",
        processing_ms=0.1,
        image_size=(64, 64),
    )
    layers = render_defect_teaching_layers(frame, result)
    assert layers.annotated_frame.shape == frame.shape
    assert layers.finding_mask.dtype == np.uint8
    assert layers.finding_mask[12, 10] == 255
    assert layers.finding_mask[29, 29] == 255
    assert layers.finding_mask[31, 30] == 0
    assert layers.legend == "缺陷区域示意"
    assert np.array_equal(frame, np.zeros((64, 64, 3), dtype=np.uint8))
