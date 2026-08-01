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
        rotated_box_px=(
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
        ),
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
        intermediate_images={
            "gray": np.zeros((240, 320), dtype=np.uint8)
        },
    )

    assert analysis.result.targets[0].detection_id == "det-001"
    assert analysis.result.image_size == (320, 240)
    assert analysis.intermediate_images["gray"].shape == (240, 320)


def test_result_rejects_unknown_status_and_invalid_image_size():
    with pytest.raises(ValueError, match="status"):
        Vision2DResult(status="OK", image_size=(320, 240))
    with pytest.raises(ValueError, match="image_size"):
        Vision2DResult(status="NO_TARGETS", image_size=(0, 240))
