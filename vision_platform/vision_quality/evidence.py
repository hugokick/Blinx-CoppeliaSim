from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any

import cv2
import numpy as np

from .models import (
    VisionImageLayer,
    VisionResultBundle,
    VisionResultEvidenceError,
)
from .results import make_result_bundle, result_bundle_to_dict


_BUNDLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_SNAPSHOT_ID = _BUNDLE_ID
_LAYER_ID = re.compile(r"[a-z][a-z0-9_-]{0,39}\Z")
_ARTIFACT_NAME = re.compile(
    r"vision-bundle-[A-Za-z0-9_][A-Za-z0-9_-]{0,60}\.json\Z"
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "bundle_id",
        "experiment_id",
        "source_snapshot_id",
        "status",
        "layers",
        "result",
        "profile",
        "hardware_status",
    }
)
_LAYER_FIELDS = frozenset(
    {"layer_id", "title", "path", "sha256", "width", "height"}
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_BYTES = 64 * 1024 * 1024
_MAX_PNG_DIMENSION = 4096
_MAX_PNG_PIXELS = 16_777_216


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _validate_png_header(payload: bytes, *, layer_id: str) -> tuple[int, int]:
    if len(payload) > _MAX_PNG_BYTES:
        raise ValueError(f"layer {layer_id} PNG exceeds 64 MiB")
    if len(payload) < 33 or not payload.startswith(_PNG_SIGNATURE):
        raise ValueError(f"layer {layer_id} has an invalid PNG signature")
    chunk_length = struct.unpack(">I", payload[8:12])[0]
    chunk_type = payload[12:16]
    if chunk_length != 13 or chunk_type != b"IHDR":
        raise ValueError(f"layer {layer_id} PNG must start with a 13-byte IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", payload[16:29])
    )
    if (
        width <= 0
        or height <= 0
        or width > _MAX_PNG_DIMENSION
        or height > _MAX_PNG_DIMENSION
        or width * height > _MAX_PNG_PIXELS
    ):
        raise ValueError(f"layer {layer_id} PNG dimensions are outside limits")
    if bit_depth != 8 or color_type != 2:
        raise ValueError(f"layer {layer_id} PNG must be 8-bit truecolor")
    if compression != 0 or filtering != 0 or interlace != 0:
        raise ValueError(f"layer {layer_id} PNG uses unsupported IHDR methods")
    return width, height


def _png_bytes(image: np.ndarray, *, layer_id: str) -> bytes:
    image_height, image_width = image.shape[:2]
    if (
        image_width > _MAX_PNG_DIMENSION
        or image_height > _MAX_PNG_DIMENSION
        or image_width * image_height > _MAX_PNG_PIXELS
    ):
        raise ValueError(f"layer {layer_id} PNG dimensions are outside limits")
    try:
        ok, encoded = cv2.imencode(".png", image)
    except cv2.error as error:
        raise ValueError(f"PNG encoding failed for layer {layer_id}") from error
    if not ok or encoded is None:
        raise ValueError(f"PNG encoding failed for layer {layer_id}")
    payload = encoded.tobytes()
    if not payload:
        raise ValueError(f"PNG encoding produced empty data for {layer_id}")
    width, height = _validate_png_header(payload, layer_id=layer_id)
    if (width, height) != (image_width, image_height):
        raise ValueError(f"layer {layer_id} PNG dimensions do not match image")
    return payload


def _read_bounded_png(
    path: Path, *, layer_id: str
) -> tuple[bytes, tuple[int, int]]:
    with path.open("rb") as stream:
        file_stat = os.fstat(stream.fileno())
        identity = (file_stat.st_dev, file_stat.st_ino)
        if file_stat.st_size > _MAX_PNG_BYTES:
            raise ValueError(f"layer {layer_id} PNG exceeds 64 MiB")
        payload = stream.read(_MAX_PNG_BYTES + 1)
    if len(payload) > _MAX_PNG_BYTES:
        raise ValueError(f"layer {layer_id} PNG exceeds 64 MiB")
    return payload, identity


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_existing_record(
    evidence: Any,
    bundle: VisionResultBundle,
    record: Any,
    png_bytes: bytes,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("existing raw record must be a mapping")
    snapshot_id = record.get("snapshot_id")
    if (
        type(snapshot_id) is not str
        or _SNAPSHOT_ID.fullmatch(snapshot_id) is None
    ):
        raise ValueError("existing raw record has an invalid snapshot ID")
    expected_path = f"frames/{snapshot_id}.png"
    expected_sha = hashlib.sha256(png_bytes).hexdigest()
    raw_layer = next(
        (layer for layer in bundle.layers if layer.layer_id == "raw"), None
    )
    if raw_layer is None:
        raise ValueError("existing raw record requires a raw bundle layer")
    height, width = raw_layer.image_bgr.shape[:2]
    if (
        snapshot_id != bundle.source_snapshot_id
        or record.get("path") != expected_path
        or record.get("sha256") != expected_sha
        or type(record.get("width")) is not int
        or record.get("width") != width
        or type(record.get("height")) is not int
        or record.get("height") != height
    ):
        raise ValueError("existing raw record does not exactly match the bundle")

    directory = getattr(evidence, "directory", None)
    if directory is not None:
        root = Path(directory).resolve()
        target = root / PurePosixPath(expected_path)
        try:
            resolved = target.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("existing raw record file is missing") from error
        if not _is_inside(root, resolved) or not resolved.is_file():
            raise ValueError("existing raw record path escapes the evidence run")
        actual_bytes, _ = _read_bounded_png(resolved, layer_id="raw")
        if actual_bytes != png_bytes:
            raise ValueError("existing raw record PNG bytes do not match")
    return {
        "path": expected_path,
        "sha256": expected_sha,
        "width": width,
        "height": height,
    }


def _validate_recorded_layer(
    record: Any,
    *,
    snapshot_id: str,
    png_bytes: bytes,
    width: int,
    height: int,
) -> dict[str, Any]:
    expected = {
        "path": f"frames/{snapshot_id}.png",
        "sha256": hashlib.sha256(png_bytes).hexdigest(),
        "width": width,
        "height": height,
    }
    if not isinstance(record, Mapping) or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("recorded layer metadata does not match PNG bytes")
    if record.get("snapshot_id", snapshot_id) != snapshot_id:
        raise ValueError("recorded layer snapshot_id does not match")
    return expected


def _appended_snapshot_id(bundle_id: str, layer_id: str) -> str:
    logical_id = f"{bundle_id}-{layer_id}"
    if len(logical_id) <= 80:
        return logical_id
    return "vision-" + hashlib.sha256(logical_id.encode("ascii")).hexdigest()


def _artifact_name(bundle_id: Any) -> str:
    if type(bundle_id) is not str or _BUNDLE_ID.fullmatch(bundle_id) is None:
        raise ValueError("bundle_id must be a portable 1-80 character ID")
    suffix = bundle_id
    if len(bundle_id) > 61:
        suffix = "_" + hashlib.sha256(bundle_id.encode("ascii")).hexdigest()[:60]
    name = f"vision-bundle-{suffix}.json"
    if (
        len(Path(name).stem) > 75
        or len(name) > 80
        or _ARTIFACT_NAME.fullmatch(name) is None
    ):
        raise ValueError("vision bundle artifact name is invalid")
    return name


def _evidence_root(evidence: Any) -> Path:
    directory = getattr(evidence, "directory", None)
    if directory is None:
        raise TypeError("evidence must expose a directory")
    try:
        root = Path(directory).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("evidence directory does not exist") from error
    if not root.is_dir():
        raise ValueError("evidence directory must be a directory")
    return root


def _preflight_new_target(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    target = root.joinpath(*pure.parts)
    try:
        resolved = target.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("evidence target path is invalid") from error
    if not _is_inside(root, resolved):
        raise ValueError("evidence target path escapes the evidence run")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"evidence target already exists: {relative_path}")
    return target


def _fresh_bundle(bundle: Any) -> VisionResultBundle:
    if not isinstance(bundle, VisionResultBundle):
        raise TypeError("bundle must be a VisionResultBundle")
    return VisionResultBundle(
        schema_version=bundle.schema_version,
        bundle_id=bundle.bundle_id,
        experiment_id=bundle.experiment_id,
        source_snapshot_id=bundle.source_snapshot_id,
        status=bundle.status,
        layers=tuple(
            VisionImageLayer(layer.layer_id, layer.title, layer.image_bgr)
            for layer in bundle.layers
        ),
        result=bundle.result,
        profile=bundle.profile,
        hardware_status=bundle.hardware_status,
    )


def record_vision_bundle(
    evidence: Any,
    bundle: VisionResultBundle,
    *,
    existing_layer_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    try:
        return _record_vision_bundle(
            evidence,
            bundle,
            existing_layer_records=existing_layer_records,
        )
    except VisionResultEvidenceError:
        raise
    except Exception as error:
        raise VisionResultEvidenceError(str(error)) from error


def _record_vision_bundle(
    evidence: Any,
    bundle: VisionResultBundle,
    *,
    existing_layer_records: Mapping[str, Mapping[str, Any]] | None,
) -> str:
    bundle = _fresh_bundle(bundle)
    existing = {} if existing_layer_records is None else existing_layer_records
    if not isinstance(existing, Mapping):
        raise TypeError("existing_layer_records must be a mapping")
    if any(layer_id != "raw" for layer_id in existing):
        raise ValueError("only an existing raw layer record may be reused")
    layer_ids = {layer.layer_id for layer in bundle.layers}
    if "raw" in existing and "raw" not in layer_ids:
        raise ValueError("existing raw record requires a bundle raw layer")

    artifact_name = _artifact_name(bundle.bundle_id)
    root = _evidence_root(evidence)

    encoded_layers = {
        layer.layer_id: _png_bytes(layer.image_bgr, layer_id=layer.layer_id)
        for layer in bundle.layers
    }
    layer_records: dict[str, dict[str, Any]] = {}
    selected_snapshot_ids: set[str] = set()
    if "raw" in existing:
        layer_records["raw"] = _validate_existing_record(
            evidence, bundle, existing["raw"], encoded_layers["raw"]
        )
        selected_snapshot_ids.add(bundle.source_snapshot_id)

    appended_snapshot_ids: dict[str, str] = {}
    for layer in bundle.layers:
        if layer.layer_id == "raw" and "raw" in existing:
            continue
        snapshot_id = _appended_snapshot_id(
            bundle.bundle_id, layer.layer_id
        )
        if snapshot_id in selected_snapshot_ids:
            raise ValueError(
                f"appended snapshot ID collision: {snapshot_id}"
            )
        selected_snapshot_ids.add(snapshot_id)
        appended_snapshot_ids[layer.layer_id] = snapshot_id

    predicted_records = dict(layer_records)
    for layer in bundle.layers:
        if layer.layer_id == "raw" and "raw" in existing:
            continue
        snapshot_id = appended_snapshot_ids[layer.layer_id]
        png_bytes = encoded_layers[layer.layer_id]
        width, height = _validate_png_header(
            png_bytes, layer_id=layer.layer_id
        )
        predicted_records[layer.layer_id] = {
            "path": f"frames/{snapshot_id}.png",
            "sha256": hashlib.sha256(png_bytes).hexdigest(),
            "width": width,
            "height": height,
        }

    payload = result_bundle_to_dict(bundle, layer_records=predicted_records)
    json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ).encode("utf-8")

    _preflight_new_target(root, artifact_name)
    for layer in bundle.layers:
        if layer.layer_id == "raw" and "raw" in existing:
            continue
        snapshot_id = appended_snapshot_ids[layer.layer_id]
        _preflight_new_target(root, f"frames/{snapshot_id}.png")

    for layer in bundle.layers:
        if layer.layer_id == "raw" and "raw" in existing:
            continue
        snapshot_id = appended_snapshot_ids[layer.layer_id]
        png_bytes = encoded_layers[layer.layer_id]
        expected = predicted_records[layer.layer_id]
        record = evidence.record_snapshot(
            snapshot_id=snapshot_id,
            png_bytes=png_bytes,
            metadata={
                "width": expected["width"],
                "height": expected["height"],
                "source_snapshot_id": bundle.source_snapshot_id,
            },
        )
        layer_records[layer.layer_id] = _validate_recorded_layer(
            record,
            snapshot_id=snapshot_id,
            png_bytes=png_bytes,
            width=expected["width"],
            height=expected["height"],
        )

    if layer_records != predicted_records:
        raise ValueError("recorded layer metadata changed after preflight")
    return evidence.record_json_artifact(artifact_name, payload)


def _require_exact_fields(
    value: Any, fields: frozenset[str], *, name: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != fields:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _direct_artifact_path(root: Path, artifact_name: Any) -> Path:
    if (
        type(artifact_name) is not str
        or _ARTIFACT_NAME.fullmatch(artifact_name) is None
        or len(Path(artifact_name).stem) > 75
        or len(artifact_name) > 80
        or "/" in artifact_name
        or "\\" in artifact_name
        or Path(artifact_name).is_absolute()
    ):
        raise ValueError("artifact must be directly inside evidence directory")
    path = root / artifact_name
    if path.is_symlink():
        raise ValueError("artifact must be directly inside evidence directory")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("artifact is missing inside evidence directory") from error
    if resolved.parent != root or not resolved.is_file():
        raise ValueError("artifact must be directly inside evidence directory")
    return resolved


def _resolve_layer_path(root: Path, value: Any) -> Path:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("layer path must stay inside evidence directory")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {".", ".."} for part in pure.parts)
    ):
        raise ValueError("layer path must stay inside evidence directory")
    if Path(value).is_absolute() or Path(value).drive:
        raise ValueError("layer path must stay inside evidence directory")
    target = root.joinpath(*pure.parts)
    try:
        resolved = target.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("layer file is missing") from error
    if not _is_inside(root, resolved) or not resolved.is_file():
        raise ValueError("layer path must stay inside evidence directory")
    return resolved


def load_recorded_bundle(
    run_directory: str | Path, artifact_name: str
) -> dict[str, Any]:
    try:
        return _load_recorded_bundle(run_directory, artifact_name)
    except VisionResultEvidenceError:
        raise
    except Exception as error:
        raise VisionResultEvidenceError(str(error)) from error


def _load_recorded_bundle(
    run_directory: str | Path, artifact_name: str
) -> dict[str, Any]:
    try:
        root = Path(run_directory).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("evidence directory does not exist") from error
    if not root.is_dir():
        raise ValueError("evidence directory must be a directory")
    artifact = _direct_artifact_path(root, artifact_name)
    try:
        payload = json.loads(
            artifact.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("vision bundle JSON is invalid") from error
    bundle_payload = _require_exact_fields(
        payload, _TOP_LEVEL_FIELDS, name="vision bundle"
    )
    if _artifact_name(bundle_payload.get("bundle_id")) != artifact_name:
        raise ValueError("artifact name does not match vision bundle ID")
    if type(bundle_payload["schema_version"]) is not int or bundle_payload[
        "schema_version"
    ] != 1:
        raise ValueError("vision bundle schema_version must equal integer 1")
    source_snapshot_id = bundle_payload.get("source_snapshot_id")
    if (
        type(source_snapshot_id) is not str
        or _SNAPSHOT_ID.fullmatch(source_snapshot_id) is None
    ):
        raise ValueError("vision bundle source_snapshot_id is invalid")
    raw_layers = bundle_payload["layers"]
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("vision bundle layers must be a non-empty list")

    layer_definitions: dict[str, tuple[str, np.ndarray]] = {}
    layer_records: dict[str, dict[str, Any]] = {}
    resolved_paths: dict[str, Path] = {}
    used_relative_paths: set[str] = set()
    used_physical_files: set[tuple[int, int]] = set()
    for raw_layer in raw_layers:
        layer = _require_exact_fields(
            raw_layer, _LAYER_FIELDS, name="vision bundle layer"
        )
        layer_id = layer.get("layer_id")
        if type(layer_id) is not str or _LAYER_ID.fullmatch(layer_id) is None:
            raise ValueError("vision bundle layer_id is invalid")
        if layer_id in layer_definitions:
            raise ValueError(f"duplicate layer_id: {layer_id}")
        appended_path = (
            "frames/"
            + _appended_snapshot_id(bundle_payload["bundle_id"], layer_id)
            + ".png"
        )
        allowed_paths = {appended_path}
        if layer_id == "raw":
            allowed_paths.add(f"frames/{source_snapshot_id}.png")
        layer_path = layer.get("path")
        if layer_path not in allowed_paths:
            raise ValueError(
                "layer path must stay inside evidence directory and match "
                f"its deterministic ID: {layer_id}"
            )
        if layer_path in used_relative_paths:
            raise ValueError("vision bundle layer paths must be unique")
        used_relative_paths.add(layer_path)
        resolved = _resolve_layer_path(root, layer_path)
        png_bytes, physical_identity = _read_bounded_png(
            resolved, layer_id=layer_id
        )
        if physical_identity in used_physical_files:
            raise ValueError("vision bundle layer paths must be unique files")
        used_physical_files.add(physical_identity)
        digest = layer.get("sha256")
        if (
            type(digest) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or hashlib.sha256(png_bytes).hexdigest() != digest
        ):
            raise ValueError(f"layer {layer_id} SHA-256 does not match")
        width = layer.get("width")
        height = layer.get("height")
        if (
            type(width) is not int
            or width <= 0
            or type(height) is not int
            or height <= 0
        ):
            raise ValueError(f"layer {layer_id} dimensions are invalid")
        header_width, header_height = _validate_png_header(
            png_bytes, layer_id=layer_id
        )
        if (width, height) != (header_width, header_height):
            raise ValueError(f"layer {layer_id} PNG dimensions do not match")
        try:
            image = cv2.imdecode(
                np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except cv2.error as error:
            raise ValueError(f"layer {layer_id} PNG is invalid") from error
        if image is None or image.shape != (height, width, 3):
            raise ValueError(f"layer {layer_id} PNG dimensions do not match")
        layer_definitions[layer_id] = (layer.get("title"), image)
        layer_records[layer_id] = {
            "path": layer_path,
            "sha256": digest,
            "width": width,
            "height": height,
        }
        resolved_paths[layer_id] = resolved

    bundle = make_result_bundle(
        bundle_id=bundle_payload.get("bundle_id"),
        experiment_id=bundle_payload.get("experiment_id"),
        source_snapshot_id=bundle_payload.get("source_snapshot_id"),
        status=bundle_payload.get("status"),
        layers=layer_definitions,
        result=bundle_payload.get("result"),
        profile=bundle_payload.get("profile"),
        hardware_status=bundle_payload.get("hardware_status"),
    )
    normalized = result_bundle_to_dict(bundle, layer_records=layer_records)
    if normalized != bundle_payload:
        raise ValueError("vision bundle is not in canonical form")
    normalized["_resolved_layer_paths"] = dict(resolved_paths)
    return normalized
