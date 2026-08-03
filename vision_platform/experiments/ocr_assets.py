"""Hash-bound, host-owned OCR training assets for V1-08.

The loader in this module is deliberately the sole file reader in the V1-08
OCR path.  It validates the manifest and every referenced PNG before exposing
immutable, read-only image copies to the training service.  It never opens a
camera, simulator, robot or tool connection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np


ALPHABET: tuple[str, ...] = ("1", "2", "A", "B")
IDENTIFIERS: tuple[str, ...] = ("A1", "A2", "B1", "B2")
EXPECTED_SCENE_ID = "V1-08"
EXPECTED_GENERATOR = "tools.vision_lab.generate_ocr_assets"
EXPECTED_SEED = 20260802
EXPECTED_METHOD = "knn"
EXPECTED_TEST_FRACTION = 0.25
EXPECTED_SIZE_PX = (64, 96)
EXPECTED_CHANNELS = 3
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {
    "schema_version",
    "generator",
    "generator_version",
    "seed",
    "alphabet",
    "training_parameters",
    "allowed_scene_ids",
    "training",
    "labels",
}
_TRAINING_PARAMETERS = {
    "method",
    "test_fraction",
    "seed",
    "variants_per_glyph",
    "bitmap_size_px",
    "image_size_px",
    "channels",
}
_TRAINING_RECORD_KEYS = {
    "glyph",
    "path",
    "purpose",
    "size_px",
    "channels",
    "sha256",
}
_LABEL_RECORD_KEYS = {
    "identifier",
    "path",
    "purpose",
    "size_px",
    "channels",
    "sha256",
}


class OcrAssetError(ValueError):
    """Stable fail-closed error for manifest or asset validation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(code: str, message: str) -> OcrAssetError:
    return OcrAssetError(code, message)


def _is_builtin_int(value: Any, *, positive: bool = False) -> bool:
    return type(value) is int and (not positive or value > 0)


def _is_builtin_float(value: Any) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _size(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(_is_builtin_int(item, positive=True) for item in value)
    ):
        raise _fail("OCR_ASSET_SCHEMA_INVALID", f"{name} must be two positive integers")
    return int(value[0]), int(value[1])


def _safe_text(value: Any, name: str) -> str:
    if type(value) is not str or not value or any(
        ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise _fail("OCR_ASSET_SCHEMA_INVALID", f"{name} must be safe text")
    return value


def _canonical_relative_path(value: Any) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise _fail("OCR_ASSET_PATH_INVALID", "asset path is not canonical")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _fail("OCR_ASSET_PATH_INVALID", "asset path must stay under the manifest root")
    normalized = path.as_posix()
    if normalized != value or not normalized.lower().endswith(".png"):
        raise _fail("OCR_ASSET_PATH_INVALID", "asset path is not canonical PNG")
    return normalized


def _require_regular_file(root: Path, relative_path: str) -> Path:
    """Resolve one path while rejecting symlink/reparse components."""

    candidate = root.joinpath(*relative_path.split("/"))
    current = root
    # Checking every component prevents a symlinked directory from escaping
    # the asset root even when the final path itself is not a link.
    for component in relative_path.split("/"):
        current = current / component
        try:
            information = current.lstat()
        except OSError as exc:
            raise _fail("OCR_ASSET_FILE_INVALID", f"asset is unavailable: {relative_path}") from exc
        if stat.S_ISLNK(information.st_mode) or bool(
            getattr(information, "st_file_attributes", 0) & 0x0400
        ):
            raise _fail("OCR_ASSET_FILE_INVALID", f"asset must not be a symlink or reparse point: {relative_path}")
    try:
        final_info = candidate.lstat()
    except OSError as exc:
        raise _fail("OCR_ASSET_FILE_INVALID", f"asset is unavailable: {relative_path}") from exc
    if not stat.S_ISREG(final_info.st_mode):
        raise _fail("OCR_ASSET_FILE_INVALID", f"asset is not a regular file: {relative_path}")
    # Resolve containment once more after the component checks.  This also
    # protects callers on platforms whose filesystem exposes unusual links.
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise _fail("OCR_ASSET_PATH_INVALID", f"asset escaped the manifest root: {relative_path}") from exc
    return candidate


@dataclass(frozen=True)
class OcrAssetRecord:
    """Validated manifest metadata for one immutable PNG."""

    path: str
    purpose: str
    size_px: tuple[int, int]
    channels: int
    sha256: str
    glyph: str | None = None
    identifier: str | None = None
    image: np.ndarray | None = None


@dataclass(frozen=True)
class OcrTrainingAssets:
    """Read-only OCR assets ready for one host training session."""

    manifest_path: Path
    asset_root: Path
    scene_id: str
    generator: str
    generator_version: str
    seed: int
    alphabet: tuple[str, ...]
    training_parameters: Mapping[str, object]
    samples: Mapping[str, tuple[np.ndarray, ...]]
    labels_images: Mapping[str, np.ndarray]
    records: tuple[OcrAssetRecord, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "asset_root", Path(self.asset_root))
        object.__setattr__(self, "alphabet", tuple(self.alphabet))
        object.__setattr__(self, "training_parameters", MappingProxyType(dict(self.training_parameters)))
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))
        object.__setattr__(self, "labels_images", MappingProxyType(dict(self.labels_images)))

    @property
    def training_images(self) -> Mapping[str, tuple[np.ndarray, ...]]:
        """Alias used by host callers; arrays remain read-only copies."""

        return self.samples

    @property
    def labels(self) -> tuple[str, ...]:
        """Identifiers in the deterministic scene-label order."""

        return IDENTIFIERS


def _decode_png(payload: bytes, record: Mapping[str, Any]) -> np.ndarray:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise _fail("OCR_ASSET_DTYPE_MISMATCH", f"asset is not a decodable PNG: {record.get('path')}") from exc
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != EXPECTED_CHANNELS:
        raise _fail("OCR_ASSET_DTYPE_MISMATCH", f"asset is not an 8-bit BGR PNG: {record.get('path')}")
    expected_size = _size(record.get("size_px"), "asset.size_px")
    expected_channels = record.get("channels")
    if not _is_builtin_int(expected_channels, positive=True):
        raise _fail("OCR_ASSET_SCHEMA_INVALID", f"asset.channels is invalid: {record.get('path')}")
    if expected_size != EXPECTED_SIZE_PX or expected_channels != EXPECTED_CHANNELS:
        raise _fail("OCR_ASSET_SHAPE_MISMATCH", f"asset shape metadata is not {EXPECTED_SIZE_PX}x{EXPECTED_CHANNELS}: {record.get('path')}")
    if (int(image.shape[1]), int(image.shape[0])) != expected_size or int(image.shape[2]) != expected_channels:
        raise _fail("OCR_ASSET_SHAPE_MISMATCH", f"asset pixels do not match metadata: {record.get('path')}")
    copy = np.ascontiguousarray(image).copy()
    copy.setflags(write=False)
    return copy


def _validate_record(record: Any, *, kind: str, seen_paths: set[str]) -> tuple[dict[str, Any], str]:
    if not isinstance(record, Mapping):
        raise _fail("OCR_ASSET_SCHEMA_INVALID", f"{kind} asset record must be an object")
    expected_keys = _TRAINING_RECORD_KEYS if kind == "training" else _LABEL_RECORD_KEYS
    if set(record) != expected_keys:
        raise _fail("OCR_ASSET_SCHEMA_INVALID", f"{kind} asset record schema is invalid")
    path = _canonical_relative_path(record["path"])
    if path in seen_paths:
        raise _fail("OCR_ASSET_DUPLICATE", f"asset path is duplicated: {path}")
    seen_paths.add(path)
    digest = record["sha256"]
    if type(digest) is not str or _HEX64.fullmatch(digest) is None:
        raise _fail("OCR_ASSET_SCHEMA_INVALID", f"asset hash is invalid: {path}")
    purpose = record["purpose"]
    expected_purpose = "training" if kind == "training" else "scene_label"
    if purpose != expected_purpose:
        raise _fail("OCR_ASSET_SCHEMA_INVALID", f"asset purpose is invalid: {path}")
    size = _size(record["size_px"], "asset.size_px")
    channels = record["channels"]
    if not _is_builtin_int(channels, positive=True):
        raise _fail("OCR_ASSET_SCHEMA_INVALID", f"asset.channels is invalid: {path}")
    if size != EXPECTED_SIZE_PX or channels != EXPECTED_CHANNELS:
        raise _fail("OCR_ASSET_SHAPE_MISMATCH", f"asset metadata is not fixed V1-08 shape: {path}")
    if kind == "training":
        glyph = record["glyph"]
        if glyph not in ALPHABET:
            raise _fail("OCR_ASSET_SCHEMA_INVALID", f"training glyph is not whitelisted: {path}")
    else:
        identifier = record["identifier"]
        if identifier not in IDENTIFIERS:
            raise _fail("OCR_ASSET_SCHEMA_INVALID", f"scene identifier is not whitelisted: {path}")
    return dict(record), path


def load_ocr_assets(
    manifest_path: str | os.PathLike[str],
    *,
    expected_scene_id: str = EXPECTED_SCENE_ID,
) -> OcrTrainingAssets:
    """Load, hash-check and decode the committed V1-08 OCR asset set."""

    if type(expected_scene_id) is not str or not expected_scene_id:
        raise _fail("OCR_ASSET_SCENE_MISMATCH", "expected scene id is invalid")
    try:
        manifest = Path(manifest_path)
    except TypeError as exc:
        raise _fail("OCR_ASSET_MANIFEST_INVALID", "manifest path is invalid") from exc
    try:
        manifest_input = manifest
        information = manifest_input.lstat()
    except OSError as exc:
        raise _fail("OCR_ASSET_MANIFEST_INVALID", "manifest is unavailable") from exc
    if manifest.name != "ocr_assets_manifest.json" or stat.S_ISLNK(information.st_mode) or bool(
        getattr(information, "st_file_attributes", 0) & 0x0400
    ) or not stat.S_ISREG(information.st_mode):
        raise _fail("OCR_ASSET_MANIFEST_INVALID", "manifest must be a regular canonical file")
    try:
        manifest = manifest.resolve(strict=True)
    except OSError as exc:
        raise _fail("OCR_ASSET_MANIFEST_INVALID", "manifest is unavailable") from exc
    try:
        manifest_bytes = manifest.read_bytes()
    except OSError as exc:
        raise _fail("OCR_ASSET_MANIFEST_INVALID", "manifest cannot be read") from exc
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("OCR_ASSET_MANIFEST_INVALID", "manifest must be UTF-8 JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_KEYS:
        raise _fail("OCR_ASSET_SCHEMA_INVALID", "manifest schema is invalid")
    if payload["schema_version"] != 1 or payload["generator"] != EXPECTED_GENERATOR:
        raise _fail("OCR_ASSET_CONFIG_INVALID", "manifest generator/schema is not the V1-08 generator")
    generator_version = _safe_text(payload["generator_version"], "generator_version")
    if payload["seed"] != EXPECTED_SEED:
        raise _fail("OCR_ASSET_CONFIG_INVALID", "manifest seed is not the fixed V1-08 seed")
    if payload["alphabet"] != list(ALPHABET):
        raise _fail("OCR_ASSET_CONFIG_INVALID", "manifest alphabet is not the fixed V1-08 alphabet")
    allowed_scenes = payload["allowed_scene_ids"]
    if not isinstance(allowed_scenes, list) or expected_scene_id not in allowed_scenes or any(type(item) is not str for item in allowed_scenes):
        raise _fail("OCR_ASSET_SCENE_MISMATCH", "manifest is not bound to the expected scene")
    parameters = payload["training_parameters"]
    if not isinstance(parameters, Mapping) or set(parameters) != _TRAINING_PARAMETERS:
        raise _fail("OCR_ASSET_CONFIG_INVALID", "training parameters schema is invalid")
    if (
        parameters["method"] != EXPECTED_METHOD
        or parameters["seed"] != EXPECTED_SEED
        or parameters["test_fraction"] != EXPECTED_TEST_FRACTION
        or not _is_builtin_int(parameters["variants_per_glyph"], positive=True)
        or parameters["variants_per_glyph"] < 8
        or _size(parameters["bitmap_size_px"], "bitmap_size_px") != (5, 7)
        or _size(parameters["image_size_px"], "image_size_px") != EXPECTED_SIZE_PX
        or parameters["channels"] != EXPECTED_CHANNELS
    ):
        raise _fail("OCR_ASSET_CONFIG_INVALID", "training parameters are not fixed V1-08 values")

    training_records = payload["training"]
    label_records = payload["labels"]
    if not isinstance(training_records, list) or not isinstance(label_records, list):
        raise _fail("OCR_ASSET_SCHEMA_INVALID", "training and labels must be arrays")
    seen_paths: set[str] = set()
    parsed_training: list[tuple[dict[str, Any], str]] = []
    parsed_labels: list[tuple[dict[str, Any], str]] = []
    for record in training_records:
        parsed_training.append(_validate_record(record, kind="training", seen_paths=seen_paths))
    for record in label_records:
        parsed_labels.append(_validate_record(record, kind="labels", seen_paths=seen_paths))
    by_glyph: dict[str, list[dict[str, Any]]] = {glyph: [] for glyph in ALPHABET}
    for record, _path in parsed_training:
        by_glyph[record["glyph"]].append(record)
    if any(len(items) < 8 for items in by_glyph.values()) or any(
        len(items) != parameters["variants_per_glyph"] for items in by_glyph.values()
    ):
        raise _fail("OCR_ASSET_CONFIG_INVALID", "each glyph must have the declared deterministic variant count")
    by_identifier: dict[str, dict[str, Any]] = {}
    for record, _path in parsed_labels:
        identifier = record["identifier"]
        if identifier in by_identifier:
            raise _fail("OCR_ASSET_DUPLICATE", f"scene label is duplicated: {identifier}")
        by_identifier[identifier] = record
    if tuple(by_identifier) != IDENTIFIERS and set(by_identifier) != set(IDENTIFIERS):
        raise _fail("OCR_ASSET_CONFIG_INVALID", "scene labels must be exactly A1, A2, B1 and B2")

    root = manifest.parent
    decoded_records: list[OcrAssetRecord] = []
    samples: dict[str, list[np.ndarray]] = {glyph: [] for glyph in ALPHABET}
    labels: dict[str, np.ndarray] = {}
    for record, path in parsed_training:
        file_path = _require_regular_file(root, path)
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            raise _fail("OCR_ASSET_FILE_INVALID", f"asset cannot be read: {path}") from exc
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise _fail("OCR_ASSET_HASH_MISMATCH", f"asset bytes do not match the manifest: {path}")
        image = _decode_png(data, record)
        samples[record["glyph"]].append(image)
        decoded_records.append(
            OcrAssetRecord(path=path, purpose=record["purpose"], size_px=tuple(record["size_px"]), channels=record["channels"], sha256=record["sha256"], glyph=record["glyph"], image=image)
        )
    for identifier in IDENTIFIERS:
        record = by_identifier[identifier]
        path = _canonical_relative_path(record["path"])
        file_path = _require_regular_file(root, path)
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            raise _fail("OCR_ASSET_FILE_INVALID", f"asset cannot be read: {path}") from exc
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise _fail("OCR_ASSET_HASH_MISMATCH", f"asset bytes do not match the manifest: {path}")
        image = _decode_png(data, record)
        labels[identifier] = image
        decoded_records.append(
            OcrAssetRecord(path=path, purpose=record["purpose"], size_px=tuple(record["size_px"]), channels=record["channels"], sha256=record["sha256"], identifier=identifier, image=image)
        )
    return OcrTrainingAssets(
        manifest_path=manifest,
        asset_root=root,
        scene_id=expected_scene_id,
        generator=payload["generator"],
        generator_version=generator_version,
        seed=EXPECTED_SEED,
        alphabet=ALPHABET,
        training_parameters=parameters,
        samples={glyph: tuple(samples[glyph]) for glyph in ALPHABET},
        labels_images=labels,
        records=tuple(decoded_records),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


class OcrAssetLoader:
    """Small object wrapper for callers that prefer dependency injection."""

    def __init__(self, manifest_path: str | os.PathLike[str], *, expected_scene_id: str = EXPECTED_SCENE_ID) -> None:
        self._manifest_path = Path(manifest_path)
        self._expected_scene_id = expected_scene_id

    def load(self) -> OcrTrainingAssets:
        return load_ocr_assets(self._manifest_path, expected_scene_id=self._expected_scene_id)


load_hash_bound_ocr_assets = load_ocr_assets


__all__ = [
    "ALPHABET",
    "EXPECTED_SCENE_ID",
    "IDENTIFIERS",
    "OcrAssetError",
    "OcrAssetLoader",
    "OcrAssetRecord",
    "OcrTrainingAssets",
    "load_hash_bound_ocr_assets",
    "load_ocr_assets",
]
