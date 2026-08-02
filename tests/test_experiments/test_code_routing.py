from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from vision_platform.experiments.code_routing import (
    ApprovedCodeRoute,
    CodeRouteError,
    CodeRoutePlan,
    build_code_route_plan,
    code_route_plan_to_dict,
)
from vision_platform.vision2d.code_recognition import CodeReading, CodeRecognitionResult


def _reading(code_type: str, payload: str | None, center: tuple[float, float], *, decoded: bool = True, confidence: float = 0.98) -> CodeReading:
    u, v = center
    return CodeReading(
        code_type=code_type,
        data=payload,
        polygon_px=((u - 2, v - 2), (u + 2, v - 2), (u + 2, v + 2), (u - 2, v + 2)),
        bbox_px=(int(u - 2), int(v - 2), 4, 4),
        center_px=center,
        confidence=confidence,
        decoded=decoded,
        failure_code=None if decoded else "QR_DECODE_FAILED",
    )


def _recognition() -> CodeRecognitionResult:
    return CodeRecognitionResult(
        status="PASS",
        readings=(
            _reading("qr", "V1-07-B", (80.0, 45.0)),
            _reading("ean13", "6901234567892", (40.0, 95.0)),
            _reading("qr", "V1-07-A", (40.0, 45.0)),
            _reading("ean13", "6901234567809", (80.0, 95.0)),
        ),
        image_size=(512, 512),
        failure_code=None,
        detector_order=("qr", "ean13"),
        processing_ms=4.0,
    )


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "expected_count": 4,
        "confidence_min": 0.90,
        "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, -90.0]],
        "pick_z_mm": 18.0,
        "safe_z_mm": 110.0,
        "speed_mm_s": 15.0,
        "routes": [
            {"entry_id": "entry_a", "part_id": "part_a", "code_type": "qr", "payload": "V1-07-A", "route_id": "route_red", "drop_xyz_mm": [116.0, -60.0, 22.0]},
            {"entry_id": "entry_b", "part_id": "part_b", "code_type": "qr", "payload": "V1-07-B", "route_id": "route_blue", "drop_xyz_mm": [116.0, 60.0, 22.0]},
            {"entry_id": "entry_c", "part_id": "part_c", "code_type": "ean13", "payload": "6901234567892", "route_id": "route_red", "drop_xyz_mm": [128.0, -60.0, 22.0]},
            {"entry_id": "entry_d", "part_id": "part_d", "code_type": "ean13", "payload": "6901234567809", "route_id": "route_blue", "drop_xyz_mm": [128.0, 60.0, 22.0]},
        ],
    }


WORKSPACE = {"x_mm": [20.0, 140.0], "y_mm": [-90.0, 90.0], "z_mm": [10.0, 140.0], "safe_z_mm": 110.0}
SCENE_PARTS = {"part_a", "part_b", "part_c", "part_d"}


def _build(config: dict[str, object] | None = None, recognition: CodeRecognitionResult | None = None):
    return build_code_route_plan(
        _recognition() if recognition is None else recognition,
        routing_config=_config() if config is None else config,
        workspace=WORKSPACE,
        scene_part_ids=SCENE_PARTS,
        image_size=(512, 512),
    )


def test_builds_deterministic_complete_route_plan() -> None:
    config = _config()
    untouched = deepcopy(config)
    plan = _build(config)
    assert config == untouched
    assert tuple(entry.entry_id for entry in plan.entries) == (
        "entry_a", "entry_b", "entry_c", "entry_d"
    )
    assert plan.entries[0].pick_xyz_mm == pytest.approx((40.0, -45.0, 18.0))
    assert plan.entries[2].drop_xyz_mm == (128.0, -60.0, 22.0)
    assert len(plan.plan_id) == 64
    payload = code_route_plan_to_dict(plan)
    assert payload["status"] == "PASS"
    json.dumps(payload, allow_nan=False)


def test_plan_id_binds_the_approved_safety_parameters() -> None:
    first = _build()
    changed = _config()
    changed["speed_mm_s"] = 14.0
    second = _build(changed)
    assert first.plan_id != second.plan_id


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda config, recognition: config.update({"extra": True}), "CODE_ROUTE_CONFIG_INVALID"),
        (lambda config, recognition: config.update({"confidence_min": float("nan")}), "CODE_ROUTE_CONFIG_INVALID"),
        (lambda config, recognition: config["routes"].append(deepcopy(config["routes"][0])), "CODE_ROUTE_CONFIG_INVALID"),
    ],
)
def test_route_plan_fails_closed(mutation, error_code: str) -> None:
    config = _config()
    recognition = _recognition()
    mutation(config, recognition)
    with pytest.raises(CodeRouteError) as captured:
        _build(config, recognition)
    assert captured.value.code == error_code


@pytest.mark.parametrize(
    "case",
    ["unknown", "undecoded", "low_confidence", "duplicate", "wrong_size", "out_of_bounds"],
)
def test_recognition_set_must_be_complete_and_approved(case: str) -> None:
    recognition = _recognition()
    readings = list(recognition.readings)
    if case == "unknown":
        readings[0] = replace(readings[0], data="UNKNOWN")
    elif case == "undecoded":
        readings[0] = replace(readings[0], data=None, decoded=False, failure_code="QR_DECODE_FAILED")
    elif case == "low_confidence":
        readings[0] = replace(readings[0], confidence=0.2)
    elif case == "duplicate":
        readings[1] = replace(readings[1], data=readings[0].data, code_type=readings[0].code_type)
    elif case == "out_of_bounds":
        readings[0] = replace(readings[0], center_px=(700.0, 45.0))
    recognition = replace(
        recognition,
        readings=tuple(readings),
        image_size=(320, 240) if case == "wrong_size" else recognition.image_size,
    )
    with pytest.raises(CodeRouteError) as captured:
        _build(recognition=recognition)
    assert captured.value.code == "CODE_ROUTE_RECOGNITION_INVALID"


@pytest.mark.parametrize(
    ("geometry_mutation", "case_name"),
    [
        (lambda reading: replace(reading, bbox_px=(38, 43, 0, 4)), "bbox_width_zero"),
        (lambda reading: replace(reading, bbox_px=(38, 43, 4, 0)), "bbox_height_zero"),
        (lambda reading: replace(reading, bbox_px=(510, 43, 4, 4)), "bbox_out_of_bounds"),
        (
            lambda reading: replace(
                reading,
                polygon_px=((38.0, 43.0), (42.0, 43.0), (float("nan"), 47.0), (38.0, 47.0)),
            ),
            "polygon_non_finite",
        ),
        (
            lambda reading: replace(
                reading,
                polygon_px=((38.0, 43.0), (42.0, 43.0), (42.0, 512.0), (38.0, 47.0)),
            ),
            "polygon_out_of_bounds",
        ),
    ],
)
def test_recognition_geometry_must_be_finite_and_inside_image(geometry_mutation, case_name: str) -> None:
    recognition = _recognition()
    readings = list(recognition.readings)
    readings[0] = geometry_mutation(readings[0])
    with pytest.raises(CodeRouteError) as captured:
        _build(recognition=replace(recognition, readings=tuple(readings)))
    assert captured.value.code == "CODE_ROUTE_RECOGNITION_INVALID", case_name


@pytest.mark.parametrize("case", ["boolean", "duplicate_part", "duplicate_drop", "scene_mismatch"])
def test_config_and_scene_binding_fail_closed(case: str) -> None:
    config = _config()
    scene_parts = set(SCENE_PARTS)
    if case == "boolean":
        config["pick_z_mm"] = True
    elif case == "duplicate_part":
        config["routes"][1]["part_id"] = config["routes"][0]["part_id"]
    elif case == "duplicate_drop":
        config["routes"][1]["drop_xyz_mm"] = config["routes"][0]["drop_xyz_mm"]
    else:
        scene_parts.remove("part_d")
    with pytest.raises(CodeRouteError) as captured:
        build_code_route_plan(
            _recognition(), routing_config=config, workspace=WORKSPACE,
            scene_part_ids=scene_parts, image_size=(512, 512),
        )
    assert captured.value.code == "CODE_ROUTE_CONFIG_INVALID"


def test_route_models_are_frozen_and_reject_invalid_values() -> None:
    routes = tuple(
        ApprovedCodeRoute(
            f"entry_{letter}", f"part_{letter}", "qr", f"V1-07-{letter.upper()}",
            f"route_{letter}", (40.0 + index, -45.0, 18.0),
            (116.0 + index, -60.0, 22.0), 0.98,
        )
        for index, letter in enumerate(("a", "b", "c", "d"))
    )
    plan = CodeRoutePlan("a" * 64, routes, 110.0, 15.0)
    with pytest.raises(FrozenInstanceError):
        routes[0].payload = "changed"
    with pytest.raises(FrozenInstanceError):
        plan.status = "REJECTED"
    with pytest.raises(CodeRouteError):
        ApprovedCodeRoute(
            "entry_a", "part_a", "qr", "V1-07-A", "route_red",
            (True, -45.0, 18.0), (116.0, -60.0, 22.0), 0.98,
        )


def test_code_route_plan_rejects_duplicate_entries_and_unsafe_height() -> None:
    routes = tuple(
        ApprovedCodeRoute(
            f"entry_{letter}", f"part_{letter}", "qr", f"V1-07-{letter.upper()}",
            f"route_{letter}", (40.0 + index, -45.0, 18.0),
            (116.0 + index, -60.0, 22.0), 0.98,
        )
        for index, letter in enumerate(("a", "b", "c", "d"))
    )
    route = routes[0]
    with pytest.raises(CodeRouteError):
        CodeRoutePlan("a" * 64, (route, route, route, route), 110.0, 15.0)
    with pytest.raises(CodeRouteError):
        CodeRoutePlan("a" * 64, routes, 20.0, 15.0)
