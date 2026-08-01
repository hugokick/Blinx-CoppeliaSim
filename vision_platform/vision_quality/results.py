from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import numpy as np

from .models import (
    VisionImageLayer,
    VisionResultBundle,
    VisionResultBundleError,
    _mapping_items,
    _thaw_json,
    _utf8_text,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def make_result_bundle(
    *,
    bundle_id: str,
    experiment_id: str,
    source_snapshot_id: str,
    status: str,
    layers: Mapping[str, tuple[str, np.ndarray]],
    result: Mapping[str, Any],
    profile: Mapping[str, Any],
    hardware_status: str = "PENDING_HARDWARE",
) -> VisionResultBundle:
    try:
        copied_layers: list[VisionImageLayer] = []
        for layer_id, definition in _mapping_items(layers, name="layers"):
            if type(definition) not in {tuple, list} or len(definition) != 2:
                raise ValueError("each layer must contain a title and image")
            title, image = definition
            copied_layers.append(VisionImageLayer(layer_id, title, image))
        return VisionResultBundle(
            schema_version=1,
            bundle_id=bundle_id,
            experiment_id=experiment_id,
            source_snapshot_id=source_snapshot_id,
            status=status,
            layers=tuple(copied_layers),
            result=result,
            profile=profile,
            hardware_status=hardware_status,
        )
    except VisionResultBundleError:
        raise
    except Exception as error:
        raise VisionResultBundleError(str(error)) from error


def _validate_layer_record(value: Any, *, layer_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"layer record for {layer_id} must be a mapping")
    required = ("path", "sha256", "width", "height")
    if any(field not in value for field in required):
        raise ValueError(f"layer record for {layer_id} is missing fields")
    path = value["path"]
    digest = value["sha256"]
    width = value["width"]
    height = value["height"]
    try:
        path = _utf8_text(
            path,
            name=f"layer record for {layer_id} path",
            nonempty=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"layer record for {layer_id} has an invalid path"
        ) from error
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"layer record for {layer_id} has an invalid sha256")
    if type(width) is not int or width <= 0:
        raise ValueError(f"layer record for {layer_id} has an invalid width")
    if type(height) is not int or height <= 0:
        raise ValueError(f"layer record for {layer_id} has an invalid height")
    return {
        "path": path,
        "sha256": digest,
        "width": width,
        "height": height,
    }


def result_bundle_to_dict(
    bundle: VisionResultBundle,
    *,
    layer_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        if not isinstance(bundle, VisionResultBundle):
            raise TypeError("bundle must be a VisionResultBundle")
        canonical = VisionResultBundle(
            schema_version=bundle.schema_version,
            bundle_id=bundle.bundle_id,
            experiment_id=bundle.experiment_id,
            source_snapshot_id=bundle.source_snapshot_id,
            status=bundle.status,
            layers=tuple(bundle.layers),
            result=bundle.result,
            profile=bundle.profile,
            hardware_status=bundle.hardware_status,
        )
        expected_ids = tuple(layer.layer_id for layer in canonical.layers)
        supplied_ids = tuple(
            key for key, _ in _mapping_items(layer_records, name="layer records")
        )
        if len(set(supplied_ids)) != len(supplied_ids):
            raise ValueError("layer records contain duplicate IDs")
        if set(supplied_ids) != set(expected_ids):
            raise ValueError("layer records must match every bundle layer exactly")

        serialized_layers = []
        for layer in canonical.layers:
            record = _validate_layer_record(
                layer_records[layer.layer_id], layer_id=layer.layer_id
            )
            height, width = layer.image_bgr.shape[:2]
            if record["width"] != width or record["height"] != height:
                raise ValueError(
                    f"layer record for {layer.layer_id} has mismatched dimensions"
                )
            serialized_layers.append(
                {
                    "layer_id": layer.layer_id,
                    "title": layer.title,
                    **record,
                }
            )

        return {
            "schema_version": 1,
            "bundle_id": canonical.bundle_id,
            "experiment_id": canonical.experiment_id,
            "source_snapshot_id": canonical.source_snapshot_id,
            "status": canonical.status,
            "layers": serialized_layers,
            "result": _thaw_json(canonical.result),
            "profile": _thaw_json(canonical.profile),
            "hardware_status": canonical.hardware_status,
        }
    except VisionResultBundleError:
        raise
    except Exception as error:
        raise VisionResultBundleError(str(error)) from error
