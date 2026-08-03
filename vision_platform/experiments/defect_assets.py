"""Hash-bound, host-owned V1-09 surface image assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np


EXPECTED_SCENE_ID = "V1-09"
EXPECTED_GENERATOR = "tools.vision_lab.generate_v1_09_defect_assets"
EXPECTED_SEED = 20260803
EXPECTED_SIZE_PX = (256, 256)
EXPECTED_ASSET_IDS: tuple[str, ...] = ("reference", "entry_a", "entry_b", "entry_c", "entry_d", "entry_e", "entry_f")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {"assets", "generator", "generator_version", "scene_id", "schema_version", "seed", "source"}
_ASSET_KEYS = {"asset_id", "generator", "generator_version", "path", "purpose", "sha256", "size_px", "source"}


class DefectAssetError(ValueError):
    """Stable fail-closed error for the V1-09 asset contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(message: str) -> DefectAssetError:
    return DefectAssetError("DEFECT_SORT_ASSET_INVALID", message)


@dataclass(frozen=True)
class DefectAssetRecord:
    asset_id: str
    path: str
    purpose: str
    size_px: tuple[int, int]
    sha256: str
    source: str
    image: np.ndarray

    def __post_init__(self) -> None:
        image = np.ascontiguousarray(self.image).copy()
        image.setflags(write=False)
        object.__setattr__(self, "image", image)


@dataclass(frozen=True)
class DefectAssets:
    manifest_path: Path
    asset_root: Path
    scene_id: str
    generator: str
    generator_version: str
    seed: int
    records: tuple[DefectAssetRecord, ...]
    images: Mapping[str, np.ndarray]
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "asset_root", Path(self.asset_root))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "images", MappingProxyType(dict(self.images)))

    def image(self, asset_id: str) -> np.ndarray:
        try:
            return self.images[asset_id]
        except (KeyError, TypeError) as exc:
            raise _fail("unknown V1-09 asset id") from exc


def _regular_file(path: Path, label: str) -> Path:
    try:
        information = path.lstat()
    except OSError as exc:
        raise _fail(f"{label} is unavailable") from exc
    if not stat.S_ISREG(information.st_mode) or path.is_symlink():
        raise _fail(f"{label} must be a regular file")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(path.parent.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise _fail(f"{label} escaped its root") from exc
    return path


def _canonical_relative(value: Any) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise _fail("asset path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _fail("asset path must stay below the manifest root")
    normalized = path.as_posix()
    if normalized != value or not normalized.startswith("assets/") or not normalized.endswith(".png"):
        raise _fail("asset path is not canonical")
    return normalized


def _decode_image(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise _fail("asset bytes cannot be read") from exc
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise _fail("asset bytes do not match the manifest")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise _fail("asset cannot be decoded") from exc
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise _fail("asset must be a uint8 BGR image")
    if (int(image.shape[1]), int(image.shape[0])) != EXPECTED_SIZE_PX or record["size_px"] != [256, 256]:
        raise _fail("asset must be exactly 256x256")
    copy = np.ascontiguousarray(image).copy()
    copy.setflags(write=False)
    return copy


def load_defect_assets(manifest_path: str | Path, *, expected_scene_id: str = EXPECTED_SCENE_ID) -> DefectAssets:
    """Load and verify the canonical seven-image V1-09 asset set."""

    manifest = Path(manifest_path)
    _regular_file(manifest, "manifest")
    if manifest.name != "defect_assets_manifest.json":
        raise _fail("manifest filename is invalid")
    try:
        payload_bytes = manifest.read_bytes()
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("manifest must be readable UTF-8 JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_KEYS:
        raise _fail("manifest schema is invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["generator"] != EXPECTED_GENERATOR
        or payload["scene_id"] != expected_scene_id
        or payload["seed"] != EXPECTED_SEED
        or payload["source"] != "project-original-generated"
        or type(payload["generator_version"]) is not str
    ):
        raise _fail("manifest identity is invalid")
    records_payload = payload["assets"]
    if not isinstance(records_payload, list) or len(records_payload) != len(EXPECTED_ASSET_IDS):
        raise _fail("manifest must contain exactly seven assets")
    root = manifest.parent
    records: list[DefectAssetRecord] = []
    images: dict[str, np.ndarray] = {}
    seen_paths: set[str] = set()
    for raw, expected_id in zip(records_payload, EXPECTED_ASSET_IDS):
        if not isinstance(raw, Mapping) or set(raw) != _ASSET_KEYS:
            raise _fail("asset record schema is invalid")
        if raw["asset_id"] != expected_id:
            raise _fail("asset identity is invalid")
        path = _canonical_relative(raw["path"])
        if path in seen_paths:
            raise _fail("asset path is duplicated")
        seen_paths.add(path)
        if raw["generator"] != EXPECTED_GENERATOR or raw["generator_version"] != payload["generator_version"]:
            raise _fail("asset generator binding is invalid")
        if raw["source"] != payload["source"] or type(raw["purpose"]) is not str:
            raise _fail("asset provenance is invalid")
        if type(raw["size_px"]) is not list or raw["size_px"] != [256, 256]:
            raise _fail("asset dimensions are invalid")
        if type(raw["sha256"]) is not str or _HEX64.fullmatch(raw["sha256"]) is None:
            raise _fail("asset hash is invalid")
        path_obj = root / Path(*path.split("/"))
        _regular_file(path_obj, path)
        image = _decode_image(path_obj, raw)
        record = DefectAssetRecord(
            asset_id=expected_id,
            path=path,
            purpose=raw["purpose"],
            size_px=(256, 256),
            sha256=raw["sha256"],
            source=raw["source"],
            image=image,
        )
        records.append(record)
        images[expected_id] = image
    return DefectAssets(
        manifest_path=manifest,
        asset_root=root,
        scene_id=expected_scene_id,
        generator=payload["generator"],
        generator_version=payload["generator_version"],
        seed=EXPECTED_SEED,
        records=tuple(records),
        images=images,
        manifest_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


class DefectAssetLoader:
    def __init__(self, manifest_path: str | Path, *, expected_scene_id: str = EXPECTED_SCENE_ID) -> None:
        self.manifest_path = Path(manifest_path)
        self.expected_scene_id = expected_scene_id

    def load(self) -> DefectAssets:
        return load_defect_assets(self.manifest_path, expected_scene_id=self.expected_scene_id)


load_hash_bound_defect_assets = load_defect_assets

__all__ = [
    "EXPECTED_ASSET_IDS",
    "EXPECTED_SCENE_ID",
    "DefectAssetError",
    "DefectAssetLoader",
    "DefectAssetRecord",
    "DefectAssets",
    "load_defect_assets",
    "load_hash_bound_defect_assets",
]
