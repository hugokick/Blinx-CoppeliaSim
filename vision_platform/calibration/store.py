from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vision_platform.calibration.affine import (
    AffineCalibration,
    CalibrationMetrics,
)
from vision_platform.errors import CalibrationError


def save_calibration(
    calibration: AffineCalibration,
    path: str | Path,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = calibration.to_record().to_dict()
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_calibration(path: str | Path) -> AffineCalibration:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise CalibrationError(
            f"Calibration file does not exist: {source}",
            code="CALIBRATION_MISSING",
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise CalibrationError(
            f"Unsupported calibration schema: {payload.get('schema_version')}"
        )
    calibration = AffineCalibration(
        matrix=np.asarray(payload["matrix"], dtype=np.float64),
        pixel_points=tuple(tuple(point) for point in payload["pixel_points"]),
        world_points_mm=tuple(
            tuple(point) for point in payload["world_points_mm"]
        ),
        image_size=tuple(payload["image_size"]),
        plane_z_mm=float(payload["plane_z_mm"]),
        source=str(payload.get("source", "unknown")),
        scene_version=payload.get("scene_version"),
        created_at=payload.get("created_at"),
    )
    if (
        payload.get("rms_error_mm") is not None
        and payload.get("max_error_mm") is not None
    ):
        calibration.metrics = CalibrationMetrics(
            rms_error_mm=float(payload["rms_error_mm"]),
            max_error_mm=float(payload["max_error_mm"]),
            sample_count=int(payload.get("validation_sample_count", 0)),
        )
    return calibration
