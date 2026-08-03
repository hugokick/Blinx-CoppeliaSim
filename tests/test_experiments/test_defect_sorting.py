from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import FrozenInstanceError

import pytest

from vision_platform.experiments.defect_sorting import (
    ApprovedDefectSortEntry,
    DefectObservation,
    DefectSortError,
    DefectSortPlan,
    DefectSortReceipt,
    build_defect_sort_plan,
    classify_defect_result,
    defect_sort_plan_to_dict,
    defect_sort_receipt_to_dict,
)
from vision_platform.vision2d.defect_detection import DefectFinding, DefectResult


IMAGE_SIZE = (1024, 1024)
SCENE_SHA = "1" * 64
CONFIG_SHA = "2" * 64
ASSET_SHA = "3" * 64

EXPECTED = (
    ("entry_a", "part_a", "qualified", "route_qualified", "slot_qualified"),
    ("entry_b", "part_b", "missing", "route_missing", "slot_missing"),
    ("entry_c", "part_c", "hole", "route_hole", "slot_hole"),
    ("entry_d", "part_d", "foreign", "route_foreign", "slot_foreign"),
    ("entry_e", "part_e", "broken", "route_broken", "slot_broken"),
    ("entry_f", "part_f", "dimension", "route_dimension", "slot_dimension"),
)


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "expected_count": 6,
        "image_size": list(IMAGE_SIZE),
        "safe_z_mm": 110.0,
        "speed_mm_s": 15.0,
        "workspace": {
            "x_mm": [20.0, 155.0],
            "y_mm": [-95.0, 95.0],
            "z_mm": [10.0, 140.0],
            "safe_z_mm": 110.0,
        },
        "routes": [
            {
                "entry_id": entry_id,
                "part_id": part_id,
                "route_id": route_id,
                "slot_id": slot_id,
                "pick_xyz_mm": [32.0 + (index % 3) * 28.0, -58.0 if index < 3 else 10.0, 18.0],
                "drop_xyz_mm": [112.0 + (index % 3) * 14.0, -78.0 if index < 3 else 78.0, 22.0],
            }
            for index, (entry_id, part_id, _decision, route_id, slot_id) in enumerate(EXPECTED)
        ],
    }


def _result(decision: str) -> DefectResult:
    if decision == "qualified":
        return DefectResult(
            status="PASS",
            defects=(),
            reference_metrics={"component_count": 1},
            candidate_metrics={"component_count": 1},
            alignment_shift_px=(0.0, 0.0),
            thresholds={"missing_ratio": 0.01},
            failure_code=None,
            processing_ms=1.0,
            image_size=IMAGE_SIZE,
        )
    finding = DefectFinding(
        defect_type=decision,
        bbox_px=(20, 20, 30, 30),
        area_px2=900.0,
        relative_area=0.02,
        metric=0.02,
        threshold=0.01,
        confidence=0.95,
    )
    return DefectResult(
        status="PARTIAL",
        defects=(finding,),
        reference_metrics={"component_count": 1},
        candidate_metrics={"component_count": 1},
        alignment_shift_px=(0.0, 0.0),
        thresholds={"missing_ratio": 0.01},
        failure_code="DEFECTS_FOUND",
        processing_ms=1.0,
        image_size=IMAGE_SIZE,
    )


def _observations() -> tuple[DefectObservation, ...]:
    return tuple(
        DefectObservation(
            entry_id=entry_id,
            part_id=part_id,
            result=_result(decision),
            roi_px=(80 + (index % 3) * 336, 56 + (index // 3) * 328, 192, 192),
            reference_crop_sha256="a" * 64,
            candidate_crop_sha256=("b" if index == 0 else format(index + 2, "x")) * 64,
        )
        for index, (entry_id, part_id, decision, _route_id, _slot_id) in enumerate(EXPECTED)
    )


def _build(config: dict[str, object] | None = None, observations: tuple[DefectObservation, ...] | None = None) -> DefectSortPlan:
    return build_defect_sort_plan(
        _config() if config is None else config,
        _observations() if observations is None else observations,
        run_id="run-001",
        frame_id="frame-001",
        scene_sha256=SCENE_SHA,
        config_sha256=CONFIG_SHA,
        asset_manifest_sha256=ASSET_SHA,
    )


def test_complete_batch_freezes_all_six_decisions() -> None:
    config = _config()
    untouched = deepcopy(config)
    plan = _build(config)

    assert tuple((entry.entry_id, entry.part_id, entry.decision, entry.slot_id) for entry in plan.entries) == tuple(
        (entry_id, part_id, decision, slot_id)
        for entry_id, part_id, decision, _route_id, slot_id in EXPECTED
    )
    assert plan.status == "PASS"
    assert plan.config_sha256 == CONFIG_SHA
    assert len(plan.plan_id) == 64
    assert config == untouched
    payload = defect_sort_plan_to_dict(plan)
    assert payload["plan_id"] == plan.plan_id
    json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)


def test_plan_and_receipt_models_are_frozen_and_json_native() -> None:
    plan = _build()
    with pytest.raises(FrozenInstanceError):
        plan.status = "FAILED"  # type: ignore[misc]
    entry = plan.entries[0]
    with pytest.raises(TypeError):
        entry.pick_xyz_mm[0] = 99.0  # type: ignore[index]
    receipt = DefectSortReceipt(
        run_id=plan.run_id,
        plan_id=plan.plan_id,
        entry_id=entry.entry_id,
        part_id=entry.part_id,
        decision=entry.decision,
        slot_id=entry.slot_id,
        status="COMPLETE",
        evidence_id="evidence-001",
        hardware_status="PENDING_HARDWARE",
    )
    assert defect_sort_receipt_to_dict(receipt)["status"] == "COMPLETE"


@pytest.mark.parametrize(
    ("status", "failure_code"),
    [("PARTIAL", "CANDIDATE_EMPTY"), ("NO_TARGETS", None), ("REJECTED", "INVALID_IMAGE")],
)
def test_non_decidable_results_fail_closed(status: str, failure_code: str | None) -> None:
    result = _result("missing")
    result = DefectResult(
        status=status,
        defects=result.defects,
        reference_metrics=result.reference_metrics,
        candidate_metrics=result.candidate_metrics,
        alignment_shift_px=result.alignment_shift_px,
        thresholds=result.thresholds,
        failure_code=failure_code,
        processing_ms=result.processing_ms,
        image_size=result.image_size,
    )
    with pytest.raises(DefectSortError, match="DEFECT_SORT_ANALYSIS_REJECTED"):
        classify_defect_result(result)


def test_multiple_types_are_ambiguous() -> None:
    result = _result("missing")
    findings = (
        result.defects[0],
        DefectFinding("hole", (60, 20, 20, 20), 400.0, 0.01, 0.02, 0.01, 0.90),
    )
    ambiguous = DefectResult(
        status="PARTIAL",
        defects=findings,
        reference_metrics=result.reference_metrics,
        candidate_metrics=result.candidate_metrics,
        alignment_shift_px=result.alignment_shift_px,
        thresholds=result.thresholds,
        failure_code="DEFECTS_FOUND",
        processing_ms=result.processing_ms,
        image_size=result.image_size,
    )
    with pytest.raises(DefectSortError, match="DEFECT_SORT_DECISION_AMBIGUOUS"):
        classify_defect_result(ambiguous)


def test_candidate_empty_is_not_missing() -> None:
    result = _result("missing")
    empty = DefectResult(
        status="PARTIAL",
        defects=result.defects,
        reference_metrics=result.reference_metrics,
        candidate_metrics=result.candidate_metrics,
        alignment_shift_px=result.alignment_shift_px,
        thresholds=result.thresholds,
        failure_code="CANDIDATE_EMPTY",
        processing_ms=result.processing_ms,
        image_size=result.image_size,
    )
    with pytest.raises(DefectSortError, match="DEFECT_SORT_ANALYSIS_REJECTED"):
        classify_defect_result(empty)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda cfg: cfg["routes"].pop(),
        lambda cfg: cfg["routes"].__setitem__(1, {**cfg["routes"][1], "entry_id": "entry_a"}),
        lambda cfg: cfg["routes"].__setitem__(1, {**cfg["routes"][1], "slot_id": "slot_qualified"}),
        lambda cfg: cfg.__setitem__("expected_count", 5),
    ],
)
def test_invalid_fixed_batch_contracts_fail_closed(mutator) -> None:
    config = _config()
    mutator(config)
    with pytest.raises(DefectSortError):
        _build(config)


def test_invalid_observation_roi_and_digest_fail_closed() -> None:
    observations = list(_observations())
    observations[0] = DefectObservation(
        entry_id="entry_a",
        part_id="part_a",
        result=_result("qualified"),
        roi_px=(900, 900, 192, 192),
        reference_crop_sha256="d" * 64,
        candidate_crop_sha256="b" * 64,
    )
    with pytest.raises(DefectSortError):
        _build(observations=tuple(observations))


def test_entry_rejects_nonfinite_and_non_builtin_numbers() -> None:
    with pytest.raises((DefectSortError, ValueError)):
        ApprovedDefectSortEntry(
            entry_id="entry_a",
            part_id="part_a",
            decision="qualified",
            route_id="route_qualified",
            slot_id="slot_qualified",
            roi_px=(0, 0, 10, 10),
            pick_xyz_mm=(32.0, -58.0, float("nan")),
            drop_xyz_mm=(112.0, -78.0, 22.0),
            reference_crop_sha256="a" * 64,
            candidate_crop_sha256="b" * 64,
            findings_sha256="c" * 64,
            run_id="run-001",
            frame_id="frame-001",
            scene_sha256=SCENE_SHA,
            config_sha256=CONFIG_SHA,
            asset_manifest_sha256=ASSET_SHA,
        )
