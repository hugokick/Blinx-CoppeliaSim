"""Finite JSON-native serialization for RGB-D measurements."""

from __future__ import annotations

import json

from .errors import RgbdContractError
from .models import Point3M, RgbdMeasurement


def measurement_to_dict(result: RgbdMeasurement) -> dict[str, object]:
    if not isinstance(result, RgbdMeasurement):
        raise RgbdContractError(
            "RGBD_SERIALIZATION_INVALID", "result must be RgbdMeasurement"
        )

    def point(value: Point3M | None) -> list[float] | None:
        if value is None:
            return None
        return [float(value.x_m), float(value.y_m), float(value.z_m)]

    payload: dict[str, object] = {
        "schema_version": int(result.schema_version),
        "status": result.status,
        "pixel_px": [int(result.sample.u_px), int(result.sample.v_px)],
        "window_size": int(result.sample.window_size),
        "valid_count": int(result.sample.valid_count),
        "depth_m": (
            None if result.sample.depth_m is None else float(result.sample.depth_m)
        ),
        "camera_frame_id": result.camera_frame_id,
        "target_frame_id": result.target_frame_id,
        "point_camera_m": point(result.point_camera_m),
        "point_target_m": point(result.point_target_m),
        "failure_code": result.failure_code,
    }
    json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return payload


__all__ = ["measurement_to_dict"]
