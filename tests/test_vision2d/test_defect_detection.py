from __future__ import annotations

import math

import numpy as np
import pytest

from vision_platform.vision2d.defect_detection import DefectConfig, detect_surface_defects

from .synthetic_v107_v109 import make_surface_pair


def test_identical_surface_passes_under_brightness_and_noise_shift() -> None:
    reference, candidate = make_surface_pair("pass", brightness=12, noise_sigma=1.5)

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PASS"
    assert result.defects == ()
    assert result.failure_code is None
    assert result.thresholds
    assert result.alignment_shift_px[0] == 0.0
    assert result.alignment_shift_px[1] == 0.0


def test_missing_material_is_reported_with_region_metric_and_threshold() -> None:
    reference, candidate = make_surface_pair("missing")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PARTIAL"
    finding = next(item for item in result.defects if item.defect_type == "missing")
    assert finding.area_px2 > 0
    assert finding.relative_area > 0
    assert finding.threshold > 0
    assert all(math.isfinite(float(value)) for value in finding.bbox_px)


def test_hole_is_distinguished_from_external_missing_material() -> None:
    reference, candidate = make_surface_pair("hole")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PARTIAL"
    assert "hole" in {item.defect_type for item in result.defects}


def test_same_legal_hole_in_reference_and_candidate_passes() -> None:
    reference, candidate = make_surface_pair("same_hole")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PASS"
    assert result.defects == ()


def test_candidate_alignment_is_reflected_in_components_and_metrics() -> None:
    reference, candidate = make_surface_pair("shifted")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PASS"
    assert result.alignment_shift_px == pytest.approx((5.0, 3.0), abs=1.0)
    assert result.candidate_metrics["bbox_px"] == result.reference_metrics["bbox_px"]
    assert result.candidate_metrics["area_px2"] == pytest.approx(
        result.reference_metrics["area_px2"], abs=20.0
    )
    assert result.candidate_metrics["foreground_px2"] == pytest.approx(
        result.reference_metrics["foreground_px2"], abs=20.0
    )


def test_foreign_component_is_reported() -> None:
    reference, candidate = make_surface_pair("foreign")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PARTIAL"
    assert [item.defect_type for item in result.defects] == ["foreign"]


def test_broken_component_is_reported() -> None:
    reference, candidate = make_surface_pair("broken")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PARTIAL"
    assert [item.defect_type for item in result.defects] == ["broken"]


def test_identical_multi_component_surface_is_not_broken() -> None:
    reference, candidate = make_surface_pair("multi_component_pass")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PASS"
    assert result.defects == ()


def test_only_the_split_subject_in_a_multi_component_surface_is_broken() -> None:
    reference, candidate = make_surface_pair("multi_component_broken")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PARTIAL"
    assert [item.defect_type for item in result.defects] == ["broken"]
    split_bbox = result.defects[0].bbox_px
    assert split_bbox[0] + split_bbox[2] <= 90


def test_filling_a_legal_reference_hole_reports_only_new_foreign_material() -> None:
    reference, candidate = make_surface_pair("filled_hole")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PARTIAL"
    assert [item.defect_type for item in result.defects] == ["foreign"]
    finding = result.defects[0]
    assert finding.area_px2 > 0
    assert 95 <= finding.bbox_px[0] <= 100
    assert 93 <= finding.bbox_px[1] <= 98


def test_dimension_change_is_reported_using_configured_relative_threshold() -> None:
    reference, candidate = make_surface_pair("dimension")
    config = DefectConfig(dimension_ratio=0.08)

    result = detect_surface_defects(reference, candidate, config=config)

    assert result.status == "PARTIAL"
    finding = next(item for item in result.defects if item.defect_type == "dimension")
    assert finding.metric >= finding.threshold
    assert result.thresholds["dimension_ratio"] == 0.08


def test_invalid_inputs_have_explicit_failure_codes() -> None:
    reference, candidate = make_surface_pair("pass")
    blank = np.full_like(reference, 128)
    wrong_size = candidate[:, :-1]
    wrong_dtype = candidate.astype(np.float32)

    assert detect_surface_defects(blank, candidate).status == "REJECTED"
    assert detect_surface_defects(blank, candidate).failure_code == "REFERENCE_EMPTY"
    assert detect_surface_defects(reference, wrong_size).failure_code == "SIZE_MISMATCH"
    assert detect_surface_defects(wrong_dtype, candidate).failure_code == "INPUT_INVALID"


@pytest.mark.parametrize("kernel", [True, False, 3.0, 2, 0, -1])
def test_morphology_kernel_size_requires_non_bool_positive_odd_integer(kernel: object) -> None:
    with pytest.raises(ValueError, match="morphology_kernel_size"):
        DefectConfig(morphology_kernel_size=kernel)  # type: ignore[arg-type]
