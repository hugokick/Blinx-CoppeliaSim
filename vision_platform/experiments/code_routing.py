"""Host-owned immutable planning contract for V1-07 code routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Collection, Mapping

from vision_platform.vision2d.code_recognition import CodeRecognitionResult


class CodeRouteError(ValueError):
    """Stable fail-closed error raised by the code-route contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or len(value) > 80:
        raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", f"{name} is invalid")
    return value


def _finite_number(value: Any, code: str, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise CodeRouteError(code, f"{name} must be finite")
    return float(value)


def _route_coordinate(value: Any, name: str) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", f"{name} is invalid")
    return tuple(
        _finite_number(component, "CODE_ROUTE_PLAN_INVALID", f"{name}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class ApprovedCodeRoute:
    entry_id: str
    part_id: str
    code_type: str
    payload: str
    route_id: str
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        for name in ("entry_id", "part_id", "payload", "route_id"):
            _text(getattr(self, name), name)
        if self.code_type not in {"qr", "ean13"}:
            raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "code_type is invalid")
        object.__setattr__(self, "pick_xyz_mm", _route_coordinate(self.pick_xyz_mm, "pick_xyz_mm"))
        object.__setattr__(self, "drop_xyz_mm", _route_coordinate(self.drop_xyz_mm, "drop_xyz_mm"))
        confidence = _finite_number(self.confidence, "CODE_ROUTE_PLAN_INVALID", "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "confidence is invalid")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class CodeRoutePlan:
    plan_id: str
    entries: tuple[ApprovedCodeRoute, ...]
    safe_z_mm: float
    speed_mm_s: float
    status: str = "PASS"

    def __post_init__(self) -> None:
        if (
            type(self.plan_id) is not str
            or len(self.plan_id) != 64
            or any(character not in "0123456789abcdef" for character in self.plan_id)
            or type(self.entries) is not tuple
            or len(self.entries) != 4
            or any(not isinstance(entry, ApprovedCodeRoute) for entry in self.entries)
            or len({entry.entry_id for entry in self.entries}) != 4
            or len({entry.part_id for entry in self.entries}) != 4
            or len({(entry.code_type, entry.payload) for entry in self.entries}) != 4
            or len({entry.drop_xyz_mm for entry in self.entries}) != 4
            or self.status != "PASS"
            or type(self.safe_z_mm) not in {int, float}
            or type(self.speed_mm_s) not in {int, float}
            or not math.isfinite(float(self.safe_z_mm))
            or not math.isfinite(float(self.speed_mm_s))
            or float(self.speed_mm_s) <= 0.0
            or float(self.safe_z_mm)
            <= max(
                coordinate
                for entry in self.entries
                for coordinate in (entry.pick_xyz_mm[2], entry.drop_xyz_mm[2])
            )
        ):
            raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "plan fields are invalid")
        object.__setattr__(self, "safe_z_mm", float(self.safe_z_mm))
        object.__setattr__(self, "speed_mm_s", float(self.speed_mm_s))


def _number(value: Any, name: str) -> float:
    return _finite_number(value, "CODE_ROUTE_CONFIG_INVALID", name)


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", f"{name} has invalid size")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _validate_reading_geometry(
    reading: Any,
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Validate pixel geometry before it can influence a motion plan."""

    code = "CODE_ROUTE_RECOGNITION_INVALID"
    bbox = getattr(reading, "bbox_px", None)
    if (
        type(bbox) is not tuple
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
        or bbox[0] < 0
        or bbox[1] < 0
        or bbox[2] <= 0
        or bbox[3] <= 0
        or bbox[0] + bbox[2] > width
        or bbox[1] + bbox[3] > height
    ):
        raise CodeRouteError(code, "bounding box is invalid")

    polygon = getattr(reading, "polygon_px", None)
    if type(polygon) is not tuple or len(polygon) < 4:
        raise CodeRouteError(code, "polygon is invalid")
    for point in polygon:
        if (
            type(point) is not tuple
            or len(point) != 2
            or any(type(value) not in {int, float} for value in point)
            or any(not math.isfinite(float(value)) for value in point)
            or not (0 <= float(point[0]) < width and 0 <= float(point[1]) < height)
        ):
            raise CodeRouteError(code, "polygon is invalid")

    center = getattr(reading, "center_px", None)
    if (
        type(center) is not tuple
        or len(center) != 2
        or any(type(value) not in {int, float} for value in center)
        or any(not math.isfinite(float(value)) for value in center)
    ):
        raise CodeRouteError(code, "centre is invalid")
    u_px, v_px = (float(center[0]), float(center[1]))
    if not (0 <= u_px < width and 0 <= v_px < height):
        raise CodeRouteError(code, "centre is out of bounds")
    if not (bbox[0] <= u_px < bbox[0] + bbox[2] and bbox[1] <= v_px < bbox[1] + bbox[3]):
        raise CodeRouteError(code, "centre is outside bounding box")
    return u_px, v_px


def build_code_route_plan(
    recognition: CodeRecognitionResult,
    *,
    routing_config: Mapping[str, Any],
    workspace: Mapping[str, Any],
    scene_part_ids: Collection[str],
    image_size: tuple[int, int],
) -> CodeRoutePlan:
    required = {
        "schema_version", "expected_count", "confidence_min",
        "calibration_matrix", "pick_z_mm", "safe_z_mm", "speed_mm_s", "routes",
    }
    if not isinstance(routing_config, Mapping) or set(routing_config) != required:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "routing schema is invalid")
    if type(routing_config["schema_version"]) is not int or routing_config["schema_version"] != 1:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "version or count is invalid")
    if type(routing_config["expected_count"]) is not int or routing_config["expected_count"] != 4:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "version or count is invalid")
    expected_count = routing_config["expected_count"]
    confidence_min = _number(routing_config["confidence_min"], "confidence_min")
    if not 0.0 <= confidence_min <= 1.0:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "confidence is outside [0, 1]")
    matrix_raw = routing_config["calibration_matrix"]
    if not isinstance(matrix_raw, (list, tuple)) or len(matrix_raw) != 2:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "calibration matrix is invalid")
    matrix = tuple(_vector(row, 3, "calibration_matrix") for row in matrix_raw)
    pick_z = _number(routing_config["pick_z_mm"], "pick_z_mm")
    safe_z = _number(routing_config["safe_z_mm"], "safe_z_mm")
    speed = _number(routing_config["speed_mm_s"], "speed_mm_s")

    if not isinstance(workspace, Mapping) or set(workspace) != {"x_mm", "y_mm", "z_mm", "safe_z_mm"}:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "workspace is invalid")
    x_range = _vector(workspace["x_mm"], 2, "workspace.x_mm")
    y_range = _vector(workspace["y_mm"], 2, "workspace.y_mm")
    z_range = _vector(workspace["z_mm"], 2, "workspace.z_mm")
    if safe_z != _number(workspace["safe_z_mm"], "workspace.safe_z_mm") or speed <= 0.0:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "safety values do not match")
    if not (x_range[0] < x_range[1] and y_range[0] < y_range[1] and z_range[0] < z_range[1]):
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "workspace ranges are invalid")

    routes_raw = routing_config["routes"]
    route_fields = {"entry_id", "part_id", "code_type", "payload", "route_id", "drop_xyz_mm"}
    if not isinstance(routes_raw, (list, tuple)) or len(routes_raw) != expected_count:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "route count is invalid")
    routes: dict[str, dict[str, Any]] = {}
    entry_ids: set[str] = set()
    part_ids: set[str] = set()
    drops: set[tuple[float, ...]] = set()
    for raw in routes_raw:
        if not isinstance(raw, Mapping) or set(raw) != route_fields:
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "route schema is invalid")
        text_fields = ("entry_id", "part_id", "payload", "route_id")
        if any(type(raw[name]) is not str or not raw[name] or len(raw[name]) > 80 for name in text_fields):
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "route ID is invalid")
        if raw["code_type"] not in {"qr", "ean13"}:
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "code type is invalid")
        drop = _vector(raw["drop_xyz_mm"], 3, "drop_xyz_mm")
        if raw["payload"] in routes or raw["entry_id"] in entry_ids or raw["part_id"] in part_ids or drop in drops:
            raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "route identifiers must be unique")
        normalized = dict(raw)
        normalized["drop_xyz_mm"] = drop
        routes[raw["payload"]] = normalized
        entry_ids.add(raw["entry_id"])
        part_ids.add(raw["part_id"])
        drops.add(drop)
    try:
        scene_parts = set(scene_part_ids)
    except (TypeError, ValueError):
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "scene parts are invalid") from None
    if part_ids != scene_parts:
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "scene parts do not match routes")
    if safe_z <= max([pick_z, *(drop[2] for drop in drops)]):
        raise CodeRouteError("CODE_ROUTE_CONFIG_INVALID", "safe Z must exceed pick and drop Z")

    if (
        not isinstance(recognition, CodeRecognitionResult)
        or recognition.status != "PASS"
        or recognition.image_size != image_size
        or len(recognition.readings) != expected_count
    ):
        raise CodeRouteError("CODE_ROUTE_RECOGNITION_INVALID", "recognition set is incomplete")
    if (
        type(image_size) is not tuple
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise CodeRouteError("CODE_ROUTE_RECOGNITION_INVALID", "image size is invalid")
    width, height = image_size
    approved: list[ApprovedCodeRoute] = []
    seen_payloads: set[str] = set()
    for reading in recognition.readings:
        if not hasattr(reading, "data") or type(reading.data) is not str:
            raise CodeRouteError("CODE_ROUTE_RECOGNITION_INVALID", "reading is not approved")
        if (
            not reading.decoded
            or reading.data in seen_payloads
            or reading.data not in routes
            or type(reading.confidence) not in {int, float}
            or not math.isfinite(float(reading.confidence))
            or float(reading.confidence) < confidence_min
            or reading.code_type != routes[reading.data]["code_type"]
        ):
            raise CodeRouteError("CODE_ROUTE_RECOGNITION_INVALID", "reading is not approved")
        u_px, v_px = _validate_reading_geometry(reading, width=width, height=height)
        pick = (
            matrix[0][0] * u_px + matrix[0][1] * v_px + matrix[0][2],
            matrix[1][0] * u_px + matrix[1][1] * v_px + matrix[1][2],
            pick_z,
        )
        route = routes[reading.data]
        drop = route["drop_xyz_mm"]
        points = (
            pick,
            drop,
            (pick[0], pick[1], safe_z + 1.0),
            (drop[0], drop[1], safe_z + 1.0),
        )
        if any(
            not (
                x_range[0] <= point[0] <= x_range[1]
                and y_range[0] <= point[1] <= y_range[1]
                and z_range[0] <= point[2] <= z_range[1]
            )
            for point in points
        ):
            raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "planned point is outside workspace")
        approved.append(
            ApprovedCodeRoute(
                route["entry_id"], route["part_id"], reading.code_type, reading.data,
                route["route_id"], tuple(float(value) for value in pick), drop,
                float(reading.confidence),
            )
        )
        seen_payloads.add(reading.data)
    approved.sort(key=lambda item: item.entry_id)
    canonical = {
        "entries": [asdict(item) for item in approved],
        "safe_z_mm": safe_z,
        "speed_mm_s": speed,
    }
    plan_id = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CodeRoutePlan(plan_id, tuple(approved), safe_z, speed)


def code_route_plan_to_dict(plan: CodeRoutePlan) -> dict[str, Any]:
    if not isinstance(plan, CodeRoutePlan):
        raise CodeRouteError("CODE_ROUTE_PLAN_INVALID", "plan type is invalid")
    return {
        "plan_id": plan.plan_id,
        "entries": [
            {
                "entry_id": entry.entry_id,
                "part_id": entry.part_id,
                "code_type": entry.code_type,
                "payload": entry.payload,
                "route_id": entry.route_id,
                "pick_xyz_mm": list(entry.pick_xyz_mm),
                "drop_xyz_mm": list(entry.drop_xyz_mm),
                "confidence": float(entry.confidence),
            }
            for entry in plan.entries
        ],
        "safe_z_mm": float(plan.safe_z_mm),
        "speed_mm_s": float(plan.speed_mm_s),
        "status": plan.status,
    }
