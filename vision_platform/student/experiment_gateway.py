from __future__ import annotations

from itertools import count
from math import isfinite
from typing import Any, Mapping

import cv2
import numpy as np

from vision_platform.experiments.models import ExperimentRunContext


class StudentExperimentGateway:
    def __init__(
        self,
        *,
        application: Any,
        evidence: Any,
        context: ExperimentRunContext,
        capture_timeout_s: float = 2.0,
    ) -> None:
        self.application = application
        self.evidence = evidence
        self.context = context
        self.capture_timeout_s = float(capture_timeout_s)
        self._snapshot_ids = count(1)

    def dispatch(self, name: str, args: Mapping[str, Any]) -> Any:
        if args:
            raise ValueError(f"{name} does not accept arguments")
        if name == "experiment.info":
            return self._public_experiment_info()
        if name == "camera.capture":
            return self._capture()
        raise ValueError(f"COMMAND_NOT_ALLOWED: {name}")

    def _public_experiment_info(self) -> dict[str, Any]:
        public = self.context.to_public_dict()
        value = {
            "experiment_id": public["experiment_id"],
            "experiment_version": public["experiment_version"],
            "scene_sha256": public["scene_sha256"],
            "public_parameters": public["public_parameters"],
            "hardware_status": "PENDING_HARDWARE",
        }
        return _copy_json_native(value, path="experiment.info")

    def _capture(self) -> dict[str, Any]:
        frame = self.application.camera.read(
            timeout_s=self.capture_timeout_s
        )
        image = frame.image_bgr
        if (
            type(image) is not np.ndarray
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise RuntimeError("CAMERA_SNAPSHOT_FRAME_INVALID")
        width = frame.width
        height = frame.height
        timestamp_s = frame.timestamp_s
        source = frame.source
        sequence_id = frame.sequence_id
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or image.shape[:2] != (height, width)
            or type(timestamp_s) not in {int, float}
            or not isfinite(float(timestamp_s))
            or type(source) is not str
            or source != "coppeliasim"
            or type(sequence_id) is not int
            or sequence_id < 0
        ):
            raise RuntimeError("CAMERA_SNAPSHOT_FRAME_INVALID")

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("CAMERA_SNAPSHOT_ENCODE_FAILED")
        snapshot_id = f"frame-{next(self._snapshot_ids):06d}"
        png_bytes = encoded.tobytes()
        metadata = {
            "width": width,
            "height": height,
            "timestamp_s": float(timestamp_s),
            "source": source,
            "sequence_id": sequence_id,
        }
        record = self.evidence.record_snapshot(
            snapshot_id=snapshot_id,
            png_bytes=png_bytes,
            metadata=metadata,
        )
        return {
            "snapshot_id": snapshot_id,
            "png_bytes": png_bytes,
            **metadata,
            "evidence_path": record["path"],
        }


def _copy_json_native(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} must contain finite floats")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            result[key] = _copy_json_native(
                nested,
                path=f"{path}.{key}",
            )
        return result
    if type(value) in {list, tuple}:
        return [
            _copy_json_native(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise TypeError(
        f"{path} must contain only JSON-native values, not "
        f"{type(value).__name__}"
    )
