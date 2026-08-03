from __future__ import annotations

from collections.abc import Mapping
import hashlib
from math import dist, isfinite
from pathlib import Path
import re
from typing import Any

from vision_platform.vision_quality.catalog import load_profile_catalog_bytes
from vision_platform.vision_quality.controller import (
    inspect_published_profile_readback,
)

_KNOWN_PROBE_KINDS = frozenset(
    {
        "motion_observation",
        "stack_2x3",
        "ordered_slots",
        "class_zones",
        "vision_profile_observation",
        "code_route_occupancy",
        "ocr_sort_occupancy",
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


def _vision_profile_catalog(
    definition: Any,
    parameters: Mapping[str, Any],
    manifest: Mapping[str, Any],
):
    try:
        scene_path = Path(definition.scene).expanduser().resolve()
        project_root = scene_path.parents[2]
        if not (project_root / "config" / "experiments").is_dir():
            raise ValueError("project experiment directory is missing")
        if not (project_root / "simulation").is_dir():
            raise ValueError("project simulation directory is missing")

        profile_path = (scene_path.parent / "profiles.json").resolve()
        profile_path.relative_to(project_root / "simulation")
        expected_path = profile_path.relative_to(project_root).as_posix()
        entry = _mapping(
            manifest.get("profile_catalog"),
            "scene_manifest.profile_catalog",
        )
        if set(entry) != {"path", "sha256"}:
            raise ValueError("profile catalog entry fields are invalid")
        digest = entry["sha256"]
        if (
            type(entry["path"]) is not str
            or entry["path"] != expected_path
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("profile catalog entry is invalid")
        content = profile_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("profile catalog hash does not match")
        catalog = load_profile_catalog_bytes(content)

        allowed = _sequence(
            parameters.get("allowed_profile_ids"),
            "public_parameters.allowed_profile_ids",
        )
        baseline = parameters.get("baseline_profile_id")
        camera_path = parameters.get("camera_path")
        if (
            any(type(profile_id) is not str for profile_id in allowed)
            or tuple(allowed) != catalog.profile_ids
            or type(baseline) is not str
            or baseline != catalog.baseline_profile_id
            or type(camera_path) is not str
            or camera_path != catalog.sensor_path
        ):
            raise ValueError("profile catalog binding is invalid")
        return catalog
    except (AttributeError, IndexError, KeyError, OSError, TypeError) as error:
        raise ValueError("profile catalog context is invalid") from error
    except ValueError as error:
        raise ValueError("profile catalog context is invalid") from error


def _vision_profile_rows(
    catalog: Any,
    inspection: Any,
    matched: bool,
):
    return [
        {
            "component": "sensor",
            "object_path": catalog.sensor_path,
            "resolution": list(inspection.resolution),
            "perspective_angle_deg": inspection.perspective_angle_deg,
            "near_clip_m": inspection.near_clip_m,
            "far_clip_m": inspection.far_clip_m,
            "matched": matched,
        },
        {
            "component": "camera_rig",
            "object_path": catalog.camera_rig_path,
            "position_m": list(inspection.camera_rig_position_m),
            "camera_rig_z_m": inspection.camera_rig_position_m[2],
            "matched": matched,
        },
        {
            "component": "key_light",
            "object_path": catalog.key_light_path,
            "state": inspection.key_light_state,
            "enabled": inspection.key_light_state > 0,
            "diffuse_rgb": list(inspection.key_diffuse_rgb),
            "matched": matched,
        },
        {
            "component": "fill_light",
            "object_path": catalog.fill_light_path,
            "state": inspection.fill_light_state,
            "enabled": inspection.fill_light_state > 0,
            "diffuse_rgb": list(inspection.fill_diffuse_rgb),
            "matched": matched,
        },
    ]


def _probe_vision_profile(
    sim: Any,
    definition: Any,
    *,
    experiment_id: str,
    phase: str,
    parameters: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = _vision_profile_catalog(definition, parameters, manifest)
    inspection = inspect_published_profile_readback(sim, catalog)
    profile = inspection.profile

    baseline = parameters["baseline_profile_id"]
    matched = profile is not None and profile.profile_id == baseline
    public = None if profile is None else profile.to_public_dict()
    expected = 4
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "phase": phase,
        "status": "PASS" if matched else "FAIL",
        "matched": expected if matched else 0,
        "expected": expected,
        "profile_id": None if profile is None else profile.profile_id,
        "baseline_profile_id": baseline,
        "profile": public,
        "rows": _vision_profile_rows(
            catalog,
            inspection,
            matched,
        ),
        "hardware_status": "PENDING_HARDWARE",
    }


def _probe_code_route_occupancy(
    sim: Any,
    definition: Any,
    *,
    experiment_id: str,
    phase: str,
    parameters: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tolerance_mm: float,
) -> dict[str, Any]:
    route_config = _mapping(
        parameters.get("code_routing"),
        "public_parameters.code_routing",
    )
    routes = _sequence(
        route_config.get("routes"),
        "public_parameters.code_routing.routes",
        length=4,
    )
    scene_contract = _mapping(
        manifest.get("code_routing"),
        "scene_manifest.code_routing",
    )
    initial = _mapping(
        scene_contract.get("initial_positions_mm"),
        "scene_manifest.code_routing.initial_positions_mm",
    )
    rows: list[dict[str, Any]] = []
    occupancy: dict[str, list[str]] = {}
    matched = 0
    for index, raw in enumerate(routes):
        route = _mapping(raw, f"public_parameters.code_routing.routes[{index}]")
        part_id = _alias(route.get("part_id"), f"routes[{index}].part_id")
        route_id = _alias(route.get("route_id"), f"routes[{index}].route_id")
        expected = (
            _position(initial.get(part_id), f"initial_positions_mm.{part_id}")
            if phase == "initial"
            else _position(route.get("drop_xyz_mm"), f"routes[{index}].drop_xyz_mm")
        )
        actual = _world_mm(sim, f"/VisionCodeRoutingLab/Parts/{part_id}")
        distance_mm = _distance_mm(actual, expected)
        is_match = distance_mm <= tolerance_mm
        matched += int(is_match)
        if phase == "final" and is_match:
            occupancy.setdefault(route_id, []).append(part_id)
        rows.append(
            {
                "part_id": part_id,
                "route_id": route_id,
                "position_mm": actual,
                "expected_mm": expected,
                "distance_mm": distance_mm,
                "matched": is_match,
            }
        )
    if phase == "final":
        for route_id in {row["route_id"] for row in rows}:
            occupancy.setdefault(route_id, [])
            occupancy[route_id].sort()
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "phase": phase,
        "status": "PASS" if matched == 4 else "FAIL",
        "matched": matched,
        "expected": 4,
        "tolerance_mm": tolerance_mm,
        "rows": rows,
        "final_occupancy": dict(sorted(occupancy.items())),
        "hardware_status": "PENDING_HARDWARE",
    }


# V1-08 has a deliberately separate probe contract.  The legacy probes above
# remain position-only and retain their historical output shape; this layer
# binds every observation to the host-owned OCR route, run, scene and
# snapshot evidence before reporting success.
_OCR_ENTRY_IDS = ("entry_a", "entry_b", "entry_c", "entry_d")
_OCR_PART_IDS = ("part_a", "part_b", "part_c", "part_d")
_OCR_ROUTE_IDS = ("route_alpha", "route_beta")
_OCR_SCENE_PREFIX = "/VisionOcrSortingLab/Parts/"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _ocr_sort_config(definition: Any) -> tuple[dict[str, dict[str, Any]], dict[str, list[list[float]]]]:
    try:
        public = _mapping(definition.public_parameters, "definition.public_parameters")
        selected = _mapping(public.get("ocr_sorting"), "public_parameters.ocr_sorting")
        config = _mapping(selected.get("sort_config"), "public_parameters.ocr_sorting.sort_config")
        raw_routes = _sequence(
            config.get("routes"),
            "public_parameters.ocr_sorting.sort_config.routes",
            length=4,
        )
    except AttributeError as error:
        raise ValueError("V1-08 definition is missing OCR sorting configuration") from error

    expected = {
        "entry_a": ("part_a", "A1", "route_alpha"),
        "entry_b": ("part_b", "A2", "route_alpha"),
        "entry_c": ("part_c", "B1", "route_beta"),
        "entry_d": ("part_d", "B2", "route_beta"),
    }
    routes: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_routes):
        route = _mapping(raw, f"ocr_sorting.sort_config.routes[{index}]")
        entry_id = _alias(route.get("entry_id"), f"routes[{index}].entry_id")
        if entry_id in routes or entry_id not in expected:
            raise ValueError("V1-08 route entry whitelist is invalid")
        part_id, identifier, route_id = expected[entry_id]
        if (
            route.get("part_id") != part_id
            or route.get("identifier") != identifier
            or route.get("route_id") != route_id
        ):
            raise ValueError("V1-08 route binding is invalid")
        drop = _position(route.get("drop_xyz_mm"), f"routes[{index}].drop_xyz_mm")
        pick = _position(route.get("pick_xyz_mm"), f"routes[{index}].pick_xyz_mm")
        routes[entry_id] = {
            "entry_id": entry_id,
            "part_id": part_id,
            "identifier": identifier,
            "route_id": route_id,
            "drop_xyz_mm": drop,
            "pick_xyz_mm": pick,
        }
    if tuple(routes) != _OCR_ENTRY_IDS:
        raise ValueError("V1-08 route entries must contain the four fixed IDs")

    selected_binding = _mapping(
        _mapping(definition.public_parameters, "definition.public_parameters")
        .get("ocr_sorting"),
        "public_parameters.ocr_sorting",
    )
    # The route slots are authoritative scene data.  If a small unit fixture
    # omits them, derive the exact same two slots from the host sort config.
    raw_slots = selected_binding.get("route_slots_mm")
    if raw_slots is None:
        raw_slots = {
            route_id: [
                routes[entry_id]["drop_xyz_mm"]
                for entry_id in _OCR_ENTRY_IDS
                if routes[entry_id]["route_id"] == route_id
            ]
            for route_id in _OCR_ROUTE_IDS
        }
    slots_mapping = _mapping(raw_slots, "public_parameters.ocr_sorting.route_slots_mm")
    slots: dict[str, list[list[float]]] = {}
    for route_id in _OCR_ROUTE_IDS:
        raw_route_slots = _sequence(
            slots_mapping.get(route_id),
            f"route_slots_mm.{route_id}",
            length=2,
        )
        slots[route_id] = [
            _position(value, f"route_slots_mm.{route_id}[{index}]")
            for index, value in enumerate(raw_route_slots)
        ]
    return routes, slots


def _ocr_manifest_slots(
    scene_manifest: Mapping[str, Any],
    routes: Mapping[str, Mapping[str, Any]],
    slots: Mapping[str, list[list[float]]],
) -> dict[str, list[list[float]]]:
    binding = scene_manifest.get("ocr_sorting")
    if binding is None:
        return {route_id: [list(value) for value in values] for route_id, values in slots.items()}
    selected = _mapping(binding, "scene_manifest.ocr_sorting")
    raw = selected.get("route_slots_mm")
    if raw is None:
        return {route_id: [list(value) for value in values] for route_id, values in slots.items()}
    declared = _mapping(raw, "scene_manifest.ocr_sorting.route_slots_mm")
    result: dict[str, list[list[float]]] = {}
    for route_id in _OCR_ROUTE_IDS:
        values = _sequence(
            declared.get(route_id),
            f"scene_manifest.ocr_sorting.route_slots_mm.{route_id}",
            length=2,
        )
        result[route_id] = [
            _position(value, f"scene_manifest.ocr_sorting.route_slots_mm.{route_id}[{index}]")
            for index, value in enumerate(values)
        ]
    # A scene cannot silently redefine the host route slots.
    for route_id in _OCR_ROUTE_IDS:
        if result[route_id] != slots[route_id]:
            raise ValueError("scene manifest route slots do not match OCR config")
    return result


def _ocr_binding_text(value: Any, name: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError(f"{name} must be a bounded string")
    if name == "scene_hash" and _HEX64.fullmatch(value) is None:
        raise ValueError("scene_hash must be a lowercase SHA-256 digest")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _ocr_actual_position(sim: Any, part_id: str) -> list[float]:
    get_object = getattr(sim, "getObject", None)
    get_position = getattr(sim, "getObjectPosition", None)
    if not callable(get_object) or not callable(get_position):
        raise ValueError("sim does not expose read-only object position methods")
    path = f"{_OCR_SCENE_PREFIX}{part_id}"
    try:
        handle = get_object(path)
        raw = get_position(handle, getattr(sim, "handle_world", -1))
        metres = _position(raw, f"sim position for {path}")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"could not read OCR part position: {part_id}") from error
    return [value * 1000.0 for value in metres]


def _ocr_entry_for_id(routes: Mapping[str, Mapping[str, Any]], entry_id: Any) -> Mapping[str, Any]:
    if type(entry_id) is not str or entry_id not in routes:
        raise ValueError("entry_id is not an approved V1-08 route")
    return routes[entry_id]


def _ocr_evidence_id(entry_id: str, run_id: str, scene_hash: str, snapshot_id: str) -> str:
    digest = hashlib.sha256(
        f"{run_id}\0{scene_hash}\0{snapshot_id}\0{entry_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"ocr-entry-{entry_id}-{digest}"


def probe_ocr_entry(
    sim: Any,
    definition: Any,
    *,
    entry_id: str,
    run_id: str,
    scene_hash: str,
    snapshot_id: str,
    scene_manifest: Mapping[str, Any] | None = None,
    tolerance_mm: float = 6.0,
) -> dict[str, Any]:
    """Return a bounded same-run proof for one configured OCR route slot."""

    routes, slots = _ocr_sort_config(definition)
    entry = _ocr_entry_for_id(routes, entry_id)
    route_id = str(entry["route_id"])
    _ocr_manifest_slots(
        scene_manifest if scene_manifest is not None else {},
        routes,
        slots,
    )
    run_id = _ocr_binding_text(run_id, "run_id")
    scene_hash = _ocr_binding_text(scene_hash, "scene_hash")
    snapshot_id = _ocr_binding_text(snapshot_id, "snapshot_id")
    tolerance = _number(tolerance_mm, "tolerance_mm")
    if tolerance <= 0:
        raise ValueError("tolerance_mm must be greater than zero")

    ordered = sorted(
        (item for item in routes.values() if item["route_id"] == route_id),
        key=lambda item: tuple(item["drop_xyz_mm"]),
    )
    slot_id = f"slot_{ordered.index(entry) + 1}"
    expected = list(slots[route_id][int(slot_id.rsplit("_", 1)[1]) - 1])
    actual = _ocr_actual_position(sim, str(entry["part_id"]))
    distance_mm = _distance_mm(actual, expected)
    passed = distance_mm <= tolerance
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "entry_id": entry_id,
        "part_id": entry["part_id"],
        "route_id": route_id,
        "slot_id": slot_id,
        "run_id": run_id,
        "scene_hash": scene_hash,
        "snapshot_id": snapshot_id,
        "position_mm": actual,
        "expected_mm": expected,
        "distance_mm": distance_mm,
        "tolerance_mm": tolerance,
        "hardware_status": "PENDING_HARDWARE",
    }
    if passed:
        evidence_id = _ocr_evidence_id(entry_id, run_id, scene_hash, snapshot_id)
        result["evidence_id"] = evidence_id
        # This exact six-field object is the only value that a pure guard
        # needs to consume.  The surrounding report is intentionally richer
        # and remains read-only evidence for the host/UI.
        result["evidence_ref"] = {
            "run_id": run_id,
            "scene_hash": scene_hash,
            "part_id": entry["part_id"],
            "route_id": route_id,
            "slot_id": slot_id,
            "evidence_id": evidence_id,
        }
    return result


def _ocr_runtime_flag(sim: Any, name: str, explicit: Any) -> bool:
    if explicit is not None:
        if type(explicit) is not bool:
            raise ValueError(f"{name} must be a boolean")
        return explicit
    getter = getattr(sim, f"get_{name}", None)
    if callable(getter):
        value = getter()
        if type(value) is not bool:
            raise ValueError(f"sim {name} state must be a boolean")
        return value
    value = getattr(sim, name, None)
    if type(value) is bool:
        return value
    return False


def probe_ocr_final(
    sim: Any,
    definition: Any,
    *,
    run_id: str,
    scene_hash: str,
    snapshot_id: str,
    entry_evidence: Any,
    consumed_entry_ids: Any,
    scene_manifest: Mapping[str, Any] | None = None,
    robot_home: bool | None = None,
    tool_on: bool | None = None,
    tolerance_mm: float = 6.0,
) -> dict[str, Any]:
    """Verify V1-08's exact four-slot terminal state and evidence binding."""

    routes, slots = _ocr_sort_config(definition)
    _ocr_manifest_slots(
        scene_manifest if scene_manifest is not None else {},
        routes,
        slots,
    )
    run_id = _ocr_binding_text(run_id, "run_id")
    scene_hash = _ocr_binding_text(scene_hash, "scene_hash")
    snapshot_id = _ocr_binding_text(snapshot_id, "snapshot_id")
    tolerance = _number(tolerance_mm, "tolerance_mm")
    if tolerance <= 0:
        raise ValueError("tolerance_mm must be greater than zero")
    refs = _sequence(entry_evidence, "entry_evidence")
    consumed = _sequence(consumed_entry_ids, "consumed_entry_ids")
    if len(refs) != 4 or len(consumed) != 4:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "matched": 0,
            "expected": 4,
            "error": {"code": "OCR_SORT_EVIDENCE_INCOMPLETE"},
            "hardware_status": "PENDING_HARDWARE",
        }
    if any(type(item) is not str for item in consumed) or set(consumed) != set(_OCR_ENTRY_IDS):
        return {
            "schema_version": 1,
            "status": "FAIL",
            "matched": 0,
            "expected": 4,
            "error": {"code": "OCR_SORT_EVIDENCE_INCOMPLETE"},
            "hardware_status": "PENDING_HARDWARE",
        }

    normalized_refs: list[dict[str, Any]] = []
    seen_entries: set[str] = set()
    same_run = True
    for index, raw in enumerate(refs):
        ref = _mapping(raw, f"entry_evidence[{index}]")
        if "evidence_ref" in ref:
            ref = _mapping(ref["evidence_ref"], f"entry_evidence[{index}].evidence_ref")
        required = {"part_id", "route_id", "slot_id", "run_id", "scene_hash", "evidence_id"}
        optional = {"entry_id", "snapshot_id"}
        if not required <= set(ref) or not set(ref) <= required | optional:
            return {
                "schema_version": 1,
                "status": "FAIL",
                "matched": 0,
                "expected": 4,
                "error": {"code": "OCR_SORT_EVIDENCE_INVALID"},
                "hardware_status": "PENDING_HARDWARE",
            }
        entry_id = ref.get("entry_id")
        if entry_id is None:
            candidates = [
                candidate_id
                for candidate_id, candidate in routes.items()
                if candidate["part_id"] == ref.get("part_id")
                and candidate["route_id"] == ref.get("route_id")
            ]
            entry_id = candidates[0] if len(candidates) == 1 else None
        if type(entry_id) is not str or entry_id not in routes or entry_id in seen_entries:
            same_run = False
        seen_entries.add(entry_id)
        for field in ("run_id", "scene_hash", "part_id", "route_id", "slot_id", "evidence_id"):
            try:
                _ocr_binding_text(ref[field], field)
            except ValueError:
                same_run = False
        if "snapshot_id" in ref:
            try:
                _ocr_binding_text(ref["snapshot_id"], "snapshot_id")
            except ValueError:
                same_run = False
        same_run = same_run and (
            ref["run_id"] == run_id
            and ref["scene_hash"] == scene_hash
            and ref.get("snapshot_id", snapshot_id) == snapshot_id
        )
        entry = routes.get(entry_id)
        if entry is None:
            continue
        ordered_route_entries = sorted(
            (item for item in routes.values() if item["route_id"] == entry["route_id"]),
            key=lambda item: tuple(item["drop_xyz_mm"]),
        )
        configured_slot_id = f"slot_{ordered_route_entries.index(entry) + 1}"
        same_run = same_run and (
            ref["part_id"] == entry["part_id"]
            and ref["route_id"] == entry["route_id"]
            and ref["slot_id"] == configured_slot_id
        )
        normalized = dict(ref)
        normalized["entry_id"] = entry_id
        normalized["snapshot_id"] = ref.get("snapshot_id", snapshot_id)
        normalized_refs.append(normalized)

    rows: list[dict[str, Any]] = []
    occupancy: dict[str, dict[str, str]] = {
        route_id: {} for route_id in _OCR_ROUTE_IDS
    }
    matched = 0
    for entry_id in _OCR_ENTRY_IDS:
        entry = routes[entry_id]
        route_id = str(entry["route_id"])
        slot_id = next(
            (
                str(ref["slot_id"])
                for ref in normalized_refs
                if ref.get("entry_id") == entry_id
            ),
            "",
        )
        expected_slot = slots[route_id][int(slot_id.rsplit("_", 1)[1]) - 1] if slot_id in {"slot_1", "slot_2"} else list(entry["drop_xyz_mm"])
        actual = _ocr_actual_position(sim, str(entry["part_id"]))
        distance_mm = _distance_mm(actual, expected_slot)
        position_match = distance_mm <= tolerance
        duplicate = slot_id in occupancy[route_id] if slot_id in {"slot_1", "slot_2"} else True
        if position_match and not duplicate:
            occupancy[route_id][slot_id] = str(entry["part_id"])
            matched += 1
        same_run = same_run and position_match and not duplicate
        rows.append(
            {
                "entry_id": entry_id,
                "part_id": entry["part_id"],
                "route_id": route_id,
                "slot_id": slot_id,
                "position_mm": actual,
                "expected_mm": expected_slot,
                "distance_mm": distance_mm,
                "matched": position_match and not duplicate,
            }
        )
    home = _ocr_runtime_flag(sim, "robot_home", robot_home)
    suction_on = _ocr_runtime_flag(sim, "tool_on", tool_on)
    tool_off = not suction_on
    passed = (
        matched == 4
        and same_run
        and home
        and tool_off
        and set(seen_entries) == set(_OCR_ENTRY_IDS)
        and len(normalized_refs) == 4
    )
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "matched": matched,
        "expected": 4,
        "run_id": run_id,
        "scene_hash": scene_hash,
        "snapshot_id": snapshot_id,
        "rows": rows,
        "final_occupancy": {
            route_id: dict(sorted(values.items()))
            for route_id, values in sorted(occupancy.items())
        },
        "entry_evidence": normalized_refs,
        "consumed_entry_ids": list(consumed),
        "same_run_evidence": same_run,
        "robot_home": home,
        "tool_off": tool_off,
        "hardware_status": "PENDING_HARDWARE",
    }


def _probe_ocr_initial(
    sim: Any,
    definition: Any,
    *,
    scene_manifest: Mapping[str, Any],
    tolerance_mm: float,
) -> dict[str, Any]:
    routes, _ = _ocr_sort_config(definition)
    binding = _mapping(
        scene_manifest.get("ocr_sorting"),
        "scene_manifest.ocr_sorting",
    )
    initial = _mapping(
        binding.get("initial_positions_mm"),
        "scene_manifest.ocr_sorting.initial_positions_mm",
    )
    rows: list[dict[str, Any]] = []
    matched = 0
    for entry_id in _OCR_ENTRY_IDS:
        entry = routes[entry_id]
        expected = _position(
            initial.get(entry["part_id"]),
            f"initial_positions_mm.{entry['part_id']}",
        )
        actual = _ocr_actual_position(sim, str(entry["part_id"]))
        distance = _distance_mm(actual, expected)
        is_match = distance <= tolerance_mm
        matched += int(is_match)
        rows.append(
            {
                "entry_id": entry_id,
                "part_id": entry["part_id"],
                "position_mm": actual,
                "expected_mm": expected,
                "distance_mm": distance,
                "matched": is_match,
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": str(definition.experiment_id),
        "phase": "initial",
        "status": "PASS" if matched == 4 else "FAIL",
        "matched": matched,
        "expected": 4,
        "tolerance_mm": tolerance_mm,
        "rows": rows,
        "final_occupancy": {},
        "hardware_status": "PENDING_HARDWARE",
    }


def probe_experiment(
    sim: Any,
    definition: Any,
    *,
    phase: str,
    scene_manifest: Mapping[str, Any],
    tolerance_mm: float = 6.0,
    run_context: Mapping[str, Any] | None = None,
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

    if kind == "vision_profile_observation":
        return _probe_vision_profile(
            sim,
            definition,
            experiment_id=experiment_id,
            phase=phase,
            parameters=parameters,
            manifest=manifest,
        )

    if kind == "code_route_occupancy":
        return _probe_code_route_occupancy(
            sim,
            definition,
            experiment_id=experiment_id,
            phase=phase,
            parameters=parameters,
            manifest=manifest,
            tolerance_mm=tolerance,
        )

    if kind == "ocr_sort_occupancy":
        if phase == "initial":
            return _probe_ocr_initial(
                sim,
                definition,
                scene_manifest=manifest,
                tolerance_mm=tolerance,
            )
        if run_context is None or not isinstance(run_context, Mapping):
            return {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "phase": "final",
                "status": "FAIL",
                "matched": 0,
                "expected": 4,
                "error": {"code": "OCR_SORT_EVIDENCE_CONTEXT_REQUIRED"},
                "hardware_status": "PENDING_HARDWARE",
            }
        context = dict(run_context)
        try:
            return probe_ocr_final(
                sim,
                definition,
                run_id=context["run_id"],
                scene_hash=context["scene_hash"],
                snapshot_id=context["snapshot_id"],
                entry_evidence=context["entry_evidence"],
                consumed_entry_ids=context["consumed_entry_ids"],
                robot_home=context.get("robot_home"),
                tool_on=context.get("tool_on"),
                scene_manifest=manifest,
                tolerance_mm=tolerance,
            )
        except (KeyError, TypeError) as error:
            return {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "phase": "final",
                "status": "FAIL",
                "matched": 0,
                "expected": 4,
                "error": {
                    "code": "OCR_SORT_EVIDENCE_CONTEXT_INVALID",
                    "type": type(error).__name__,
                },
                "hardware_status": "PENDING_HARDWARE",
            }

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
