from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest

from vision_platform.experiments.ocr_sorting import (
    OcrAnalysis,
    OcrAssetSpec,
    OcrObservation,
    OcrRouteSpec,
    OcrSortError,
    OcrSortPlan,
    ApprovedOcrSortEntry,
    build_ocr_sort_plan,
    ocr_sort_plan_to_dict,
    validate_training_report,
)
from vision_platform.vision2d.ocr import OCRCharacter, OCRResult, TrainingReport


IMAGE_SIZE = (320, 240)
SCENE_PARTS = {"part_a", "part_b", "part_c", "part_d"}
WORKSPACE = {
    "x_mm": [20.0, 140.0],
    "y_mm": [-90.0, 90.0],
    "z_mm": [10.0, 140.0],
    "safe_z_mm": 110.0,
}


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "expected_count": 4,
        "training_accuracy_min": 0.90,
        "confidence_min": 0.80,
        "safe_z_mm": 110.0,
        "speed_mm_s": 15.0,
        "workspace": deepcopy(WORKSPACE),
        "routes": [
            {
                "entry_id": "entry_a",
                "part_id": "part_a",
                "identifier": "A1",
                "route_id": "route_alpha",
                "pick_xyz_mm": [40.0, -45.0, 18.0],
                "drop_xyz_mm": [116.0, -60.0, 22.0],
            },
            {
                "entry_id": "entry_b",
                "part_id": "part_b",
                "identifier": "A2",
                "route_id": "route_alpha",
                "pick_xyz_mm": [80.0, -45.0, 18.0],
                "drop_xyz_mm": [116.0, 60.0, 22.0],
            },
            {
                "entry_id": "entry_c",
                "part_id": "part_c",
                "identifier": "B1",
                "route_id": "route_beta",
                "pick_xyz_mm": [40.0, 45.0, 18.0],
                "drop_xyz_mm": [128.0, -60.0, 22.0],
            },
            {
                "entry_id": "entry_d",
                "part_id": "part_d",
                "identifier": "B2",
                "route_id": "route_beta",
                "pick_xyz_mm": [80.0, 45.0, 18.0],
                "drop_xyz_mm": [128.0, 60.0, 22.0],
            },
        ],
    }


def _ocr(identifier: str, *, confidence: float = 0.97, status: str = "PASS", image_size: tuple[int, int] = (40, 24)) -> OCRResult:
    characters = tuple(
        OCRCharacter(
            character=character,
            bbox_px=(5 + index * 15, 4, 10, 16),
            confidence=confidence,
        )
        for index, character in enumerate(identifier)
    )
    return OCRResult(
        status=status,
        text=identifier,
        characters=characters,
        image_size=image_size,
        threshold_method="adaptive_gaussian",
        character_count=2,
        failure_code=None if status == "PASS" else "EXPECTED_TEXT_MISMATCH",
        processing_ms=1.0,
    )


def _analysis(observations: tuple[OcrObservation, ...] | None = None) -> OcrAnalysis:
    items = observations or tuple(
        OcrObservation(identifier=identifier, result=_ocr(identifier), roi_px=roi)
        for identifier, roi in zip(
            ("A1", "A2", "B1", "B2"),
            ((10, 20, 40, 24), (70, 20, 40, 24), (10, 90, 40, 24), (70, 90, 40, 24)),
        )
    )
    return OcrAnalysis(
        training_report=TrainingReport(train_count=24, test_count=8, held_out_accuracy=0.96, seed=20260802),
        observations=items,
        image_size=IMAGE_SIZE,
    )


def _build(config: dict[str, object] | None = None, analysis: OcrAnalysis | None = None):
    return build_ocr_sort_plan(
        _config() if config is None else config,
        _analysis() if analysis is None else analysis,
        scene_part_ids=SCENE_PARTS,
    )


def test_builds_deterministic_complete_ocr_sort_plan_and_json() -> None:
    config = _config()
    untouched = deepcopy(config)
    plan = _build(config)

    assert config == untouched
    assert tuple(entry.identifier for entry in plan.entries) == ("A1", "A2", "B1", "B2")
    assert tuple(entry.entry_id for entry in plan.entries) == ("entry_a", "entry_b", "entry_c", "entry_d")
    assert plan.status == "PASS"
    assert plan.entries[0].pick_xyz_mm == pytest.approx((40.0, -45.0, 18.0))
    assert plan.entries[0].roi_px == (10, 20, 40, 24)
    assert len(plan.plan_id) == 64
    payload = ocr_sort_plan_to_dict(plan)
    assert payload["status"] == "PASS"
    json.dumps(payload, allow_nan=False)


def test_ocr_models_are_frozen_and_validate_manifest_assets() -> None:
    asset = OcrAssetSpec(
        asset_id="glyph_A_01",
        path="training/A/01.png",
        sha256="a" * 64,
        size_px=(20, 20),
        channels=1,
        generator="v1_08_bitmap_v1",
    )
    assert asset.size_px == (20, 20)
    with pytest.raises(FrozenInstanceError):
        asset.asset_id = "changed"
    route = OcrRouteSpec(
        entry_id="entry_a",
        part_id="part_a",
        identifier="A1",
        route_id="route_alpha",
        pick_xyz_mm=(40.0, -45.0, 18.0),
        drop_xyz_mm=(116.0, -60.0, 22.0),
    )
    routes = tuple(
        OcrRouteSpec(
            entry_id=f"entry_{letter}",
            part_id=f"part_{letter}",
            identifier=identifier,
            route_id="route_alpha" if identifier.startswith("A") else "route_beta",
            pick_xyz_mm=(40.0 + index, -45.0, 18.0),
            drop_xyz_mm=(116.0 + index, -60.0, 22.0),
        )
        for index, (letter, identifier) in enumerate((("a", "A1"), ("b", "A2"), ("c", "B1"), ("d", "B2")))
    )
    entries = tuple(
        ApprovedOcrSortEntry(
            entry_id=route.entry_id,
            part_id=route.part_id,
            identifier=route.identifier,
            route_id=route.route_id,
            roi_px=(10 + index * 40, 20, 40, 24),
            pick_xyz_mm=route.pick_xyz_mm,
            drop_xyz_mm=route.drop_xyz_mm,
            confidence=0.97,
        )
        for index, route in enumerate(routes)
    )
    plan = OcrSortPlan("a" * 64, entries, 110.0, 15.0, status="PASS")
    assert plan.entries[0].identifier == "A1"
    with pytest.raises(FrozenInstanceError):
        plan.status = "REJECTED"


def test_training_report_must_meet_finite_minimum() -> None:
    report = TrainingReport(train_count=24, test_count=8, held_out_accuracy=0.96, seed=1)
    assert validate_training_report(report, 0.90) == report
    with pytest.raises(OcrSortError):
        validate_training_report(replace(report, held_out_accuracy=0.80), 0.90)
    malformed = object.__new__(TrainingReport)
    object.__setattr__(malformed, "train_count", 24)
    object.__setattr__(malformed, "test_count", 8)
    object.__setattr__(malformed, "held_out_accuracy", float("nan"))
    object.__setattr__(malformed, "seed", 1)
    with pytest.raises(OcrSortError):
        validate_training_report(malformed, 0.90)
    with pytest.raises(OcrSortError):
        validate_training_report(report, True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config, analysis: config.update({"extra": True}),
        lambda config, analysis: config["routes"].append(deepcopy(config["routes"][0])),
        lambda config, analysis: config["routes"][1].update({"identifier": "A1"}),
        lambda config, analysis: config["routes"][0].update({"pick_xyz_mm": [True, 0.0, 0.0]}),
        lambda config, analysis: config.update({"speed_mm_s": float("inf")}),
    ],
)
def test_config_schema_and_route_whitelist_fail_closed(mutation) -> None:
    config = _config()
    mutation(config, _analysis())
    with pytest.raises(OcrSortError):
        _build(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda items: items[:3],
        lambda items: items[:1] + (items[0],) + items[2:],
        lambda items: (replace(items[0], identifier="C9"),) + items[1:],
        lambda items: (replace(items[0], result=_ocr("A1", status="PARTIAL")),) + items[1:],
        lambda items: (replace(items[0], result=_ocr("B1")),) + items[1:],
        lambda items: (replace(items[0], result=_ocr("A1", confidence=0.20)),) + items[1:],
        lambda items: (replace(items[0], roi_px=(310, 20, 40, 24)),) + items[1:],
        lambda items: (replace(items[0], result=replace(items[0].result, characters=(replace(items[0].result.characters[0], bbox_px=(39, 4, 10, 16)), items[0].result.characters[1]))),) + items[1:],
    ],
)
def test_analysis_observations_require_complete_exact_pass_identifiers(mutation) -> None:
    base = _analysis()
    analysis = replace(base, observations=mutation(base.observations))
    with pytest.raises(OcrSortError):
        _build(analysis=analysis)


def test_scene_parts_must_match_the_route_parts() -> None:
    with pytest.raises(OcrSortError):
        build_ocr_sort_plan(_config(), _analysis(), scene_part_ids={"part_a", "part_b"})


def test_source_analysis_is_not_mutated_and_results_are_immutable() -> None:
    analysis = _analysis()
    before = analysis
    plan = _build(analysis=analysis)
    assert analysis == before
    with pytest.raises(FrozenInstanceError):
        analysis.observations = ()
    assert all(math.isfinite(entry.confidence) for entry in plan.entries)
