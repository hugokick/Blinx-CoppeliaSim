from __future__ import annotations

from collections.abc import Mapping
from math import dist, isfinite
import re
from typing import Any


_KNOWN_PROBE_KINDS = frozenset(
    {
        "motion_observation",
        "stack_2x3",
        "ordered_slots",
        "class_zones",
    }
)
_GROUP_BY_KIND = {
    "stack_2x3": "/LogisticsLab/Tasks/Stack",
    "ordered_slots": "/LogisticsLab/Tasks/Digits",
    "class_zones": "/LogisticsLab/Tasks/Classes",
}
_TASK_BY_KIND = {
    "stack_2x3": "Stack",
    "ordered_slots": "Digits",
    "class_zones": "Classes",
}
_JOINT_PATHS = tuple(f"/BLX_joint{index}" for index in range(1, 7))
_TOOL_PATH = "/BLX_tool_suction"
_ALIAS_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str, *, length: int | None = None) -> list[Any]:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{name} must be a list")
    result = list(value)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def _number(value: Any, name: str) -> float:
    if (
        type(value) not in {int, float}
        or not isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _position(value: Any, name: str) -> list[float]:
    raw = _sequence(value, name, length=3)
    return [
        _number(item, f"{name}[{index}]")
        for index, item in enumerate(raw)
    ]


def _alias(value: Any, name: str) -> str:
    if type(value) is not str or _ALIAS_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable object alias")
    return value


def _world_mm(sim: Any, path: str) -> list[float]:
    handle = sim.getObject(path)
    raw = sim.getObjectPosition(handle, sim.handle_world)
    metres = _position(raw, f"sim position for {path}")
    world_mm = [value * 1000.0 for value in metres]
    if not all(isfinite(value) for value in world_mm):
        raise ValueError("world position_mm must be finite")
    return world_mm


def _distance_mm(left: list[float], right: list[float]) -> float:
    try:
        value = dist(left, right)
    except OverflowError as error:
        raise ValueError("distance_mm must be finite") from error
    if not isfinite(value):
        raise ValueError("distance_mm must be finite")
    return value


def _maximum_assignment(
    actual: list[list[float]],
    expected: list[list[float]],
    tolerance_mm: float,
) -> dict[int, int]:
    adjacency: list[list[int]] = []
    for position in actual:
        candidates = []
        for index, target in enumerate(expected):
            distance_mm = _distance_mm(position, target)
            if distance_mm <= tolerance_mm:
                candidates.append((distance_mm, index))
        candidates.sort(key=lambda item: (item[0], item[1]))
        adjacency.append([index for _, index in candidates])

    expected_owner: list[int | None] = [None] * len(expected)

    def augment(actual_index: int, seen: set[int]) -> bool:
        for expected_index in adjacency[actual_index]:
            if expected_index in seen:
                continue
            seen.add(expected_index)
            owner = expected_owner[expected_index]
            if owner is None or augment(owner, seen):
                expected_owner[expected_index] = actual_index
                return True
        return False

    for actual_index in range(len(actual)):
        augment(actual_index, set())
    return {
        actual_index: expected_index
        for expected_index, actual_index in enumerate(expected_owner)
        if actual_index is not None
    }


def _validate_motion(parameters: Mapping[str, Any]) -> tuple[str, ...]:
    raw_paths = parameters.get("joint_paths", _JOINT_PATHS)
    paths = _sequence(raw_paths, "public_parameters.joint_paths", length=6)
    if any(type(path) is not str for path in paths):
        raise ValueError("public_parameters.joint_paths must contain strings")
    if tuple(paths) != _JOINT_PATHS:
        raise ValueError("public_parameters.joint_paths must use the six fixed BLX joints")
    return (*_JOINT_PATHS, _TOOL_PATH)


def _validate_logistics(
    kind: str,
    parameters: Mapping[str, Any],
) -> tuple[str, list[str], Any]:
    group = parameters.get("scene_group_path")
    expected_group = _GROUP_BY_KIND[kind]
    if type(group) is not str or group != expected_group:
        raise ValueError(
            f"public_parameters.scene_group_path must be {expected_group}"
        )

    if kind == "stack_2x3":
        expected = [
            _position(slot, f"public_parameters.stack_slots_mm[{index}]")
            for index, slot in enumerate(
                _sequence(
                    parameters.get("stack_slots_mm"),
                    "public_parameters.stack_slots_mm",
                    length=6,
                )
            )
        ]
        aliases = [f"stack_{index:02d}" for index in range(1, 7)]
        return group, aliases, expected

    if kind == "ordered_slots":
        raw_order = _sequence(
            parameters.get("order"),
            "public_parameters.order",
            length=3,
        )
        if any(
            type(value) is not int or value < 0 or value > 9
            for value in raw_order
        ) or len(set(raw_order)) != 3:
            raise ValueError(
                "public_parameters.order must contain three distinct digits"
            )
        slots = [
            _position(slot, f"public_parameters.drop_slots_mm[{index}]")
            for index, slot in enumerate(
                _sequence(
                    parameters.get("drop_slots_mm"),
                    "public_parameters.drop_slots_mm",
                    length=3,
                )
            )
        ]
        aliases = [f"digit_{digit}" for digit in raw_order]
        return group, aliases, (list(raw_order), slots)

    raw_classes = _sequence(
        parameters.get("classes"),
        "public_parameters.classes",
        length=4,
    )
    classes = [
        _alias(value, f"public_parameters.classes[{index}]")
        for index, value in enumerate(raw_classes)
    ]
    if len(set(classes)) != 4:
        raise ValueError("public_parameters.classes must be unique")
    raw_routes = _mapping(
        parameters.get("drop_poses_mm"),
        "public_parameters.drop_poses_mm",
    )
    if set(raw_routes) != set(classes):
        raise ValueError(
            "public_parameters.drop_poses_mm must match classes exactly"
        )
    routes = {
        class_id: _position(
            raw_routes[class_id],
            f"public_parameters.drop_poses_mm.{class_id}",
        )
        for class_id in classes
    }
    return group, classes, routes


def _initial_objects(
    scene_manifest: Mapping[str, Any],
    *,
    kind: str,
    expected_aliases: list[str],
) -> list[tuple[str, list[float]]]:
    task_contracts = _mapping(
        scene_manifest.get("task_contracts"),
        "scene_manifest.task_contracts",
    )
    task_name = _TASK_BY_KIND[kind]
    contract = _mapping(
        task_contracts.get(task_name),
        f"scene_manifest.task_contracts.{task_name}",
    )
    raw_objects = _sequence(
        contract.get("objects"),
        f"scene_manifest.task_contracts.{task_name}.objects",
        length=len(expected_aliases),
    )
    result: list[tuple[str, list[float]]] = []
    for index, raw_item in enumerate(raw_objects):
        item = _mapping(
            raw_item,
            f"scene_manifest.task_contracts.{task_name}.objects[{index}]",
        )
        alias = _alias(
            item.get("alias"),
            f"scene_manifest.task_contracts.{task_name}.objects[{index}].alias",
        )
        position = _position(
            item.get("position_mm"),
            f"scene_manifest.task_contracts.{task_name}.objects[{index}].position_mm",
        )
        result.append((alias, position))
    aliases = [alias for alias, _ in result]
    if len(set(aliases)) != len(aliases) or set(aliases) != set(expected_aliases):
        raise ValueError(
            f"scene_manifest.task_contracts.{task_name}.objects aliases "
            "must match the selected experiment exactly"
        )
    return result


def _report(
    *,
    experiment_id: str,
    phase: str,
    matched: int,
    expected: int,
    tolerance_mm: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "phase": phase,
        "status": "PASS" if matched == expected else "FAIL",
        "matched": matched,
        "expected": expected,
        "tolerance_mm": float(tolerance_mm),
        "rows": rows,
        "hardware_status": "PENDING_HARDWARE",
    }


def probe_experiment(
    sim: Any,
    definition: Any,
    *,
    phase: str,
    scene_manifest: Mapping[str, Any],
    tolerance_mm: float = 6.0,
) -> dict[str, Any]:
    if type(phase) is not str or phase not in {"initial", "final"}:
        raise ValueError("phase must be initial or final")
    tolerance = _number(tolerance_mm, "tolerance_mm")
    if tolerance <= 0:
        raise ValueError("tolerance_mm must be greater than zero")
    manifest = _mapping(scene_manifest, "scene_manifest")
    _mapping(
        manifest.get("task_contracts"),
        "scene_manifest.task_contracts",
    )

    try:
        experiment_id = definition.experiment_id
        parameters_raw = definition.public_parameters
        acceptance = definition.acceptance
        kind = acceptance.probe_kind
    except AttributeError as error:
        raise ValueError("definition is missing required probe fields") from error
    if type(experiment_id) is not str or not experiment_id.strip():
        raise ValueError("definition.experiment_id must be a non-empty string")
    parameters = _mapping(
        parameters_raw,
        "definition.public_parameters",
    )
    if type(kind) is not str or kind not in _KNOWN_PROBE_KINDS:
        raise ValueError(f"unsupported probe_kind: {kind}")

    if kind == "motion_observation":
        paths = _validate_motion(parameters)
        rows: list[dict[str, Any]] = []
        for path in paths:
            sim.getObject(path)
            rows.append(
                {
                    "object_path": path,
                    "expected_path": path,
                    "distance_mm": None,
                }
            )
        return _report(
            experiment_id=experiment_id,
            phase=phase,
            matched=len(paths),
            expected=len(paths),
            tolerance_mm=tolerance,
            rows=rows,
        )

    group, aliases, final_contract = _validate_logistics(kind, parameters)
    if phase == "initial":
        reset_objects = _initial_objects(
            manifest,
            kind=kind,
            expected_aliases=aliases,
        )
        rows = []
        matched = 0
        for alias, expected_position in reset_objects:
            actual = _world_mm(sim, f"{group}/Pickables/{alias}")
            distance_mm = _distance_mm(actual, expected_position)
            rows.append(
                {
                    "object": alias,
                    "position_mm": actual,
                    "expected_mm": expected_position,
                    "distance_mm": distance_mm,
                    "matched": distance_mm <= tolerance,
                }
            )
            matched += int(distance_mm <= tolerance)
        return _report(
            experiment_id=experiment_id,
            phase=phase,
            matched=matched,
            expected=len(reset_objects),
            tolerance_mm=tolerance,
            rows=rows,
        )

    if kind == "stack_2x3":
        expected_positions = final_contract
        actual_positions = [
            _world_mm(sim, f"{group}/Pickables/{alias}")
            for alias in aliases
        ]
        assignment = _maximum_assignment(
            actual_positions,
            expected_positions,
            tolerance,
        )
        rows = []
        for actual_index, (alias, actual) in enumerate(
            zip(aliases, actual_positions)
        ):
            expected_index = assignment.get(actual_index)
            matched_row = expected_index is not None
            if expected_index is None:
                expected_index = min(
                    range(len(expected_positions)),
                    key=lambda index: (
                        _distance_mm(actual, expected_positions[index]),
                        index,
                    ),
                )
            expected_position = expected_positions[expected_index]
            rows.append(
                {
                    "object": alias,
                    "position_mm": actual,
                    "expected_index": expected_index,
                    "expected_mm": expected_position,
                    "distance_mm": _distance_mm(actual, expected_position),
                    "matched": matched_row,
                }
            )
        return _report(
            experiment_id=experiment_id,
            phase=phase,
            matched=len(assignment),
            expected=len(expected_positions),
            tolerance_mm=tolerance,
            rows=rows,
        )

    rows = []
    matched = 0
    if kind == "ordered_slots":
        order, expected_positions = final_contract
        for index, (digit, alias, expected_position) in enumerate(
            zip(order, aliases, expected_positions)
        ):
            actual = _world_mm(sim, f"{group}/Pickables/{alias}")
            distance_mm = _distance_mm(actual, expected_position)
            rows.append(
                {
                    "digit": digit,
                    "slot_index": index,
                    "position_mm": actual,
                    "expected_mm": expected_position,
                    "distance_mm": distance_mm,
                    "matched": distance_mm <= tolerance,
                }
            )
            matched += int(distance_mm <= tolerance)
    else:
        routes = final_contract
        for class_id in aliases:
            expected_position = routes[class_id]
            actual = _world_mm(sim, f"{group}/Pickables/{class_id}")
            distance_mm = _distance_mm(actual, expected_position)
            rows.append(
                {
                    "class_id": class_id,
                    "position_mm": actual,
                    "expected_mm": expected_position,
                    "distance_mm": distance_mm,
                    "matched": distance_mm <= tolerance,
                }
            )
            matched += int(distance_mm <= tolerance)
    return _report(
        experiment_id=experiment_id,
        phase=phase,
        matched=matched,
        expected=len(aliases),
        tolerance_mm=tolerance,
        rows=rows,
    )
