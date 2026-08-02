from __future__ import annotations

import math

import numpy as np

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


def test_foreign_component_is_reported() -> None:
    reference, candidate = make_surface_pair("foreign")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PARTIAL"
    assert "foreign" in {item.defect_type for item in result.defects}


def test_broken_component_is_reported() -> None:
    reference, candidate = make_surface_pair("broken")

    result = detect_surface_defects(reference, candidate)

    assert result.status == "PARTIAL"
    assert "broken" in {item.defect_type for item in result.defects}


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

