"""Strict, read-only binding between the D1-01 scene manifest and captures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .depth_model import DepthAnchor
from .errors import RgbdSimContractError

_MANIFEST_KEYS = {
    "schema_version",
    "generated_at",
    "generator",
    "port",
    "protected_assets_unchanged",
    "template",
    "scene",
    "sensor",
    "source_depth_model",
    "required_paths",
    "probe_anchors",
    "validation_rois",
    "depth_order",
}
_SCENE_KEYS = {"path", "sha256", "size_bytes"}
_SENSOR_KEYS = {
    "path",
    "resolution",
    "perspective_angle_deg",
    "near_clip_m",
    "far_clip_m",
    "explicit_handling",
    "perspective",
    "rgb_enabled",
    "depth_enabled",
    "position_m",
    "orientation_quaternion",
}
_ANCHOR_KEYS = {"path", "position_m", "pixel", "optical_z_m", "ray_range_m"}
_ROI_NAMES = {"near_block", "far_block", "step_low", "step_high"}
_DEPTH_ORDER_KEYS = {
    "near_block_less_than_far_block",
    "step_high_less_than_step_low",
    "minimum_margin_m",
}
_REQUIRED_PATHS = {
    "/RgbdLab",
    "/RgbdLab/CameraRig/RgbdSensor",
    "/RgbdLab/ReferencePlane",
    "/RgbdLab/Targets/near_block",
    "/RgbdLab/Targets/far_block",
    "/RgbdLab/Targets/step_low",
    "/RgbdLab/Targets/step_high",
    "/RgbdLab/ProbeAnchors/center",
    "/RgbdLab/ProbeAnchors/off_axis_left",
    "/RgbdLab/ProbeAnchors/off_axis_right",
}
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class RoiSpec:
    """Half-open image ROI expressed as ``x0, y0, x1, y1`` pixels."""

    name: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class SceneBinding:
    manifest_path: Path
    scene_file: Path
    scene_path: str
    scene_sha256: str
    port: int
    sensor_path: str
    resolution: tuple[int, int]
    perspective_angle_rad: float
    near_clip_m: float
    far_clip_m: float
    expected_source_depth_model: str
    required_paths: tuple[str, ...]
    anchors: tuple[DepthAnchor, ...]
    rois: Mapping[str, RoiSpec]
    depth_order_margin_m: float
    source_model_tolerance_m: float = 0.02

    def __post_init__(self) -> None:
        if self.port != 23009:
            raise RgbdSimContractError(
                "RGBD_SIM_BINDING_INVALID", "D1-01 must use dedicated port 23009"
            )
        if self.expected_source_depth_model not in {"optical_z", "ray_range"}:
            raise RgbdSimContractError(
                "RGBD_SIM_BINDING_INVALID", "source depth model is unsupported"
            )
        object.__setattr__(self, "resolution", tuple(self.resolution))
        object.__setattr__(self, "required_paths", tuple(self.required_paths))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "rois", MappingProxyType(dict(self.rois)))


def _fail(code: str, message: str) -> None:
    raise RgbdSimContractError(code, message)


def _object(value: object, code: str, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code, f"{label} must be an object")
    unknown = set(value) - keys
    missing = keys - set(value)
    if unknown or missing:
        _fail(code, f"{label} keys are not exact (unknown={sorted(unknown)}, missing={sorted(missing)})")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        _fail("RGBD_SIM_BINDING_INVALID", f"{label} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        _fail("RGBD_SIM_BINDING_INVALID", f"{label} must be positive")
    return result


def _safe_relative_file(raw: object, root: Path, manifest_parent: Path, code: str) -> tuple[Path, str]:
    if type(raw) is not str or not raw or Path(raw).is_absolute():
        _fail(code, "manifest file path must be relative")
    raw_path = Path(raw)
    candidates = [(manifest_parent / raw_path), (root / raw_path)]
    for candidate in candidates:
        current = candidate
        while current != root and current.parent != current:
            if current.is_symlink():
                _fail("RGBD_SIM_BINDING_PATH_INVALID", "manifest scene path may not use symlinks")
            current = current.parent
    chosen: Path | None = None
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.exists():
            chosen = resolved
            break
    if chosen is None:
        # Resolve once more so an escaping path reports the path-specific code.
        try:
            resolved = candidates[0].resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, ValueError):
            _fail(code, "manifest file path escapes repository root")
        _fail(code, "manifest file does not exist")
    current = chosen
    while True:
        if current.is_symlink():
            _fail("RGBD_SIM_BINDING_PATH_INVALID", "manifest scene path may not use symlinks")
        if current == root:
            break
        if current.parent == current:
            break
        current = current.parent
    return chosen, chosen.relative_to(root).as_posix()


def _verify_sha256(path: Path, expected: object) -> str:
    if type(expected) is not str or len(expected) != _SHA256_LENGTH or any(
        char not in "0123456789abcdef" for char in expected
    ):
        _fail("RGBD_SIM_BINDING_SCENE_INVALID", "scene sha256 must be lowercase SHA-256")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        _fail("RGBD_SIM_BINDING_SCENE_INVALID", "scene sha256 does not match manifest")
    return digest


def _parse_rois(raw: object, width: int, height: int) -> Mapping[str, RoiSpec]:
    if not isinstance(raw, dict) or set(raw) != _ROI_NAMES:
        _fail("RGBD_SIM_BINDING_ROI_INVALID", "validation_rois keys are not exact")
    parsed: dict[str, RoiSpec] = {}
    for name, coords in raw.items():
        if (
            not isinstance(coords, list)
            or len(coords) != 4
            or any(type(value) is not int for value in coords)
        ):
            _fail("RGBD_SIM_BINDING_ROI_INVALID", f"ROI {name} coordinates are invalid")
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            _fail("RGBD_SIM_BINDING_ROI_INVALID", f"ROI {name} is out of bounds")
        parsed[name] = RoiSpec(name, x0, y0, x1, y1)
    names = list(parsed)
    for index, left_name in enumerate(names):
        left = parsed[left_name]
        for right_name in names[index + 1 :]:
            right = parsed[right_name]
            if left.x0 < right.x1 and right.x0 < left.x1 and left.y0 < right.y1 and right.y0 < left.y1:
                _fail("RGBD_SIM_BINDING_ROI_INVALID", f"ROIs {left_name} and {right_name} overlap")
    return MappingProxyType(parsed)


def _parse_anchors(raw: object, width: int, height: int) -> tuple[DepthAnchor, ...]:
    if not isinstance(raw, list) or not raw:
        _fail("RGBD_SIM_BINDING_ANCHOR_INVALID", "probe_anchors must be a non-empty list")
    anchors: list[DepthAnchor] = []
    paths: set[str] = set()
    pixels: set[tuple[int, int]] = set()
    for item in raw:
        value = _object(item, "RGBD_SIM_BINDING_ANCHOR_INVALID", "probe anchor", _ANCHOR_KEYS)
        path = value["path"]
        if type(path) is not str or not path.startswith("/") or path in paths:
            _fail("RGBD_SIM_BINDING_ANCHOR_INVALID", "anchor path is invalid or duplicated")
        pixel = value["pixel"]
        if (
            not isinstance(pixel, list)
            or len(pixel) != 2
            or any(type(item_value) is not int for item_value in pixel)
            or not (0 <= pixel[0] < width and 0 <= pixel[1] < height)
            or (pixel[0], pixel[1]) in pixels
        ):
            _fail("RGBD_SIM_BINDING_ANCHOR_INVALID", "anchor pixel is invalid or duplicated")
        tolerance = 0.02
        anchors.append(
            DepthAnchor(
                pixel[0],
                pixel[1],
                _finite(value["optical_z_m"], "anchor optical_z_m", positive=True),
                _finite(value["ray_range_m"], "anchor ray_range_m", positive=True),
                tolerance,
            )
        )
        paths.add(path)
        pixels.add((pixel[0], pixel[1]))
    return tuple(anchors)


def load_scene_binding(
    manifest_path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str] | None = None,
) -> SceneBinding:
    """Load and verify one D1-01 scene manifest without executing its contents."""

    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest_input = manifest
    if manifest_input.is_symlink():
        _fail("RGBD_SIM_BINDING_PATH_INVALID", "manifest path must not be a symlink")
    manifest = manifest.resolve(strict=False)
    try:
        manifest.relative_to(root)
    except ValueError:
        _fail("RGBD_SIM_BINDING_PATH_INVALID", "manifest path escapes repository root")
    if manifest.is_symlink() or not manifest.is_file():
        _fail("RGBD_SIM_BINDING_PATH_INVALID", "manifest path must be a regular file")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("RGBD_SIM_BINDING_MANIFEST_INVALID", f"manifest cannot be read: {exc}")
    value = _object(payload, "RGBD_SIM_BINDING_MANIFEST_INVALID", "manifest", _MANIFEST_KEYS)
    if value["schema_version"] != 1 or value["port"] != 23009 or value["protected_assets_unchanged"] is not True:
        _fail("RGBD_SIM_BINDING_MANIFEST_INVALID", "schema, dedicated port, or protected flag is invalid")
    if type(value["source_depth_model"]) is not str or value["source_depth_model"] not in {"optical_z", "ray_range"}:
        _fail("RGBD_SIM_BINDING_MANIFEST_INVALID", "source depth model is unsupported")

    scene_value = _object(value["scene"], "RGBD_SIM_BINDING_SCENE_INVALID", "scene", _SCENE_KEYS)
    scene_file, scene_path = _safe_relative_file(scene_value["path"], root, manifest.parent, "RGBD_SIM_BINDING_PATH_INVALID")
    scene_sha256 = _verify_sha256(scene_file, scene_value["sha256"])
    if type(scene_value["size_bytes"]) is not int or scene_value["size_bytes"] != scene_file.stat().st_size:
        _fail("RGBD_SIM_BINDING_SCENE_INVALID", "scene size does not match manifest")

    sensor = _object(value["sensor"], "RGBD_SIM_BINDING_SENSOR_INVALID", "sensor", _SENSOR_KEYS)
    if type(sensor["path"]) is not str or not sensor["path"].startswith("/"):
        _fail("RGBD_SIM_BINDING_SENSOR_INVALID", "sensor path is invalid")
    resolution = sensor["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(item) is not int or item <= 0 for item in resolution)
    ):
        _fail("RGBD_SIM_BINDING_SENSOR_INVALID", "sensor resolution is invalid")
    if any(sensor[name] is not True for name in ("explicit_handling", "perspective", "rgb_enabled", "depth_enabled")):
        _fail("RGBD_SIM_BINDING_SENSOR_INVALID", "sensor contract flags must be true")
    angle_rad = math.radians(_finite(sensor["perspective_angle_deg"], "perspective_angle_deg", positive=True))
    near_clip_m = _finite(sensor["near_clip_m"], "near_clip_m", positive=True)
    far_clip_m = _finite(sensor["far_clip_m"], "far_clip_m", positive=True)
    if far_clip_m <= near_clip_m or angle_rad >= math.pi:
        _fail("RGBD_SIM_BINDING_SENSOR_INVALID", "sensor clipping or angle is invalid")

    required_paths = value["required_paths"]
    if (
        not isinstance(required_paths, list)
        or not required_paths
        or any(type(path) is not str or not path.startswith("/") for path in required_paths)
        or len(set(required_paths)) != len(required_paths)
    ):
        _fail("RGBD_SIM_BINDING_PATH_INVALID", "required scene paths are invalid")
    if set(required_paths) != _REQUIRED_PATHS:
        _fail("RGBD_SIM_BINDING_PATH_INVALID", "required scene paths do not match the D1-01 scene contract")
    rois = _parse_rois(value["validation_rois"], resolution[0], resolution[1])
    anchors = _parse_anchors(value["probe_anchors"], resolution[0], resolution[1])
    order = _object(value["depth_order"], "RGBD_SIM_BINDING_DEPTH_ORDER_INVALID", "depth_order", _DEPTH_ORDER_KEYS)
    if order["near_block_less_than_far_block"] is not True or order["step_high_less_than_step_low"] is not True:
        _fail("RGBD_SIM_BINDING_DEPTH_ORDER_INVALID", "depth-order contracts must be enabled")
    margin = _finite(order["minimum_margin_m"], "minimum_margin_m", positive=True)

    return SceneBinding(
        manifest_path=manifest,
        scene_file=scene_file,
        scene_path=scene_path,
        scene_sha256=scene_sha256,
        port=23009,
        sensor_path=sensor["path"],
        resolution=(resolution[0], resolution[1]),
        perspective_angle_rad=angle_rad,
        near_clip_m=near_clip_m,
        far_clip_m=far_clip_m,
        expected_source_depth_model=value["source_depth_model"],
        required_paths=tuple(required_paths),
        anchors=anchors,
        rois=rois,
        depth_order_margin_m=margin,
    )


# Names used by callers that prefer an explicit verb.
bind_scene_manifest = load_scene_binding


__all__ = ["RoiSpec", "SceneBinding", "bind_scene_manifest", "load_scene_binding"]
