from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vision_platform.errors import VisionPlatformError
from vision_platform.models import Detection
from vision_platform.tasks.classify import PlacementEvidence


@dataclass(frozen=True)
class _Zone:
    handle: int
    bounds: tuple[float, float, float, float, float, float]


class CoppeliaSimPlacementVerifier:
    def __init__(
        self,
        *,
        sim: Any,
        zone_paths: Mapping[str, str],
        margin_m: float = 0.0,
        check_z: bool = False,
    ) -> None:
        if margin_m < 0:
            raise ValueError("margin_m must be non-negative")
        self.sim = sim
        self.zone_paths = dict(zone_paths)
        self.margin_m = float(margin_m)
        self.check_z = bool(check_z)
        self._zones: dict[str, _Zone] = {}

    def verify(
        self,
        object_ref: Any,
        expected_zone: str,
    ) -> PlacementEvidence:
        zone = self._resolve_zone(expected_zone)
        object_handle = self._object_handle(object_ref)
        position = tuple(
            float(value)
            for value in self.sim.getObjectPosition(object_handle, zone.handle)
        )
        if len(position) != 3:
            raise VisionPlatformError(
                "PLACEMENT_VERIFICATION_FAILED",
                "CoppeliaSim returned an invalid object position",
                details={"position": list(position)},
            )
        min_x, max_x, min_y, max_y, min_z, max_z = zone.bounds
        margin = self.margin_m
        inside_x = min_x - margin <= position[0] <= max_x + margin
        inside_y = min_y - margin <= position[1] <= max_y + margin
        inside_z = min_z - margin <= position[2] <= max_z + margin
        ok = inside_x and inside_y and (inside_z if self.check_z else True)
        return PlacementEvidence(
            ok=ok,
            expected_zone=expected_zone,
            object_ref=object_handle,
            details={
                "zone_path": self.zone_paths[expected_zone],
                "zone_handle": zone.handle,
                "relative_position_m": list(position),
                "bounds_m": list(zone.bounds),
                "inside_x": inside_x,
                "inside_y": inside_y,
                "inside_z": inside_z,
                "check_z": self.check_z,
            },
        )

    def _resolve_zone(self, name: str) -> _Zone:
        if name in self._zones:
            return self._zones[name]
        if name not in self.zone_paths:
            raise VisionPlatformError(
                "CLASSIFICATION_ZONE_MISSING",
                f"No CoppeliaSim zone path is configured for {name}",
                details={"zone": name},
            )
        handle = int(self.sim.getObject(self.zone_paths[name]))
        parameters = (
            self.sim.objfloatparam_objbbox_min_x,
            self.sim.objfloatparam_objbbox_max_x,
            self.sim.objfloatparam_objbbox_min_y,
            self.sim.objfloatparam_objbbox_max_y,
            self.sim.objfloatparam_objbbox_min_z,
            self.sim.objfloatparam_objbbox_max_z,
        )
        bounds = tuple(
            self._float_param(handle, parameter) for parameter in parameters
        )
        zone = _Zone(handle=handle, bounds=bounds)
        self._zones[name] = zone
        return zone

    def _float_param(self, handle: int, parameter: int) -> float:
        value = self.sim.getObjectFloatParam(handle, parameter)
        if isinstance(value, (tuple, list)):
            value = value[-1]
        return float(value)

    @staticmethod
    def _object_handle(object_ref: Any) -> int:
        if isinstance(object_ref, Detection):
            object_ref = object_ref.metadata.get("object_handle")
        if isinstance(object_ref, bool) or not isinstance(object_ref, int):
            raise VisionPlatformError(
                "PLACEMENT_VERIFICATION_FAILED",
                "Placement verification requires a CoppeliaSim object handle",
                details={"object_ref": repr(object_ref)},
            )
        return object_ref
