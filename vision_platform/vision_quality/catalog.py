from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from .models import VisionProfile, VisionProfileCatalog


_CATALOG_FIELDS = frozenset(
    {
        "schema_version",
        "baseline_profile_id",
        "sensor_path",
        "camera_rig_path",
        "key_light_path",
        "fill_light_path",
        "near_clip_m",
        "far_clip_m",
        "profiles",
    }
)
_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "label",
        "resolution",
        "perspective_angle_deg",
        "camera_rig_z_m",
        "key_diffuse_rgb",
        "fill_diffuse_rgb",
    }
)
_FIXED_PATHS = {
    "sensor_path": "/VisionQualityLab/CameraRig/Camera",
    "camera_rig_path": "/VisionQualityLab/CameraRig",
    "key_light_path": "/VisionQualityLab/Lighting/KeyLight",
    "fill_light_path": "/VisionQualityLab/Lighting/FillLight",
}
_APPROVED_SCENE_ROOTS = ("/VisionQualityLab", "/VisionCodeRoutingLab")
_PROFILE_ID = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _require_exact_fields(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = fields - value.keys()
    extra = value.keys() - fields
    if missing:
        raise ValueError(f"{name} missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{name} has unexpected fields: {', '.join(sorted(extra))}")
    return value


def _finite_number(value: Any, name: str, minimum: float, maximum: float) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _validate_profile(value: Any) -> VisionProfile:
    profile = _require_exact_fields(value, _PROFILE_FIELDS, "profile")
    profile_id = _nonempty_text(profile["profile_id"], "profile_id")
    if not profile_id.isascii() or not _PROFILE_ID.fullmatch(profile_id):
        raise ValueError("profile_id must match [a-z][a-z0-9_]{0,31}")
    label = _nonempty_text(profile["label"], "label")

    resolution = profile["resolution"]
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("resolution must contain exactly two integers")
    if any(isinstance(component, bool) or not isinstance(component, int) for component in resolution):
        raise ValueError("resolution must contain integers")
    if any(component < 128 or component > 1024 for component in resolution):
        raise ValueError("resolution components must be between 128 and 1024")

    angle = _finite_number(profile["perspective_angle_deg"], "perspective_angle_deg", 20, 90)
    rig_z = _finite_number(profile["camera_rig_z_m"], "camera_rig_z_m", 0.50, 0.90)
    key_rgb = _validate_rgb(profile["key_diffuse_rgb"], "key_diffuse_rgb")
    fill_rgb = _validate_rgb(profile["fill_diffuse_rgb"], "fill_diffuse_rgb")
    return VisionProfile(profile_id, label, tuple(resolution), angle, rig_z, key_rgb, fill_rgb)


def _validate_rgb(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numeric components")
    if any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in value):
        raise ValueError(f"{name} must contain finite numeric components")
    for component in value:
        if isinstance(component, float) and not math.isfinite(component):
            raise ValueError(f"{name} components must be finite values from 0 to 1")
        if component < 0 or component > 1:
            raise ValueError(f"{name} components must be finite values from 0 to 1")
    return tuple(float(component) for component in value)


def load_profile_catalog_bytes(content: bytes) -> VisionProfileCatalog:
    if type(content) is not bytes:
        raise TypeError("catalog content must be exact bytes")
    payload = json.loads(
        content.decode("utf-8"),
        parse_constant=_reject_nonfinite,
        object_pairs_hook=_reject_duplicate_keys,
    )
    catalog = _require_exact_fields(payload, _CATALOG_FIELDS, "catalog")

    schema_version = catalog["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise ValueError("schema_version must equal integer 1")
    baseline_profile_id = _nonempty_text(catalog["baseline_profile_id"], "baseline_profile_id")
    expected_suffixes = {
        "sensor_path": "/CameraRig/Camera",
        "camera_rig_path": "/CameraRig",
        "key_light_path": "/Lighting/KeyLight",
        "fill_light_path": "/Lighting/FillLight",
    }
    roots = set()
    for field, suffix in expected_suffixes.items():
        value = catalog[field]
        if not isinstance(value, str) or not value.endswith(suffix):
            raise ValueError(f"{field} must equal fixed formal-scene path")
        roots.add(value[: -len(suffix)])
    if len(roots) != 1 or next(iter(roots), None) not in _APPROVED_SCENE_ROOTS:
        raise ValueError("catalog paths must belong to an approved formal scene")
    scene_root = next(iter(roots))
    resolved_paths = {
        field: f"{scene_root}{suffix}"
        for field, suffix in expected_suffixes.items()
    }

    near_clip_m = _finite_number(catalog["near_clip_m"], "near_clip_m", 0.01, 0.20)
    far_clip_m = _finite_number(catalog["far_clip_m"], "far_clip_m", 1.0, 5.0)
    if near_clip_m >= far_clip_m:
        raise ValueError("near_clip_m must be less than far_clip_m")
    raw_profiles = catalog["profiles"]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("profiles must be a nonempty list")
    profiles = tuple(_validate_profile(profile) for profile in raw_profiles)
    profile_ids = tuple(profile.profile_id for profile in profiles)
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("duplicate profile_id")
    if baseline_profile_id not in profile_ids:
        raise ValueError("baseline_profile_id must name a profile")

    return VisionProfileCatalog(
        baseline_profile_id=baseline_profile_id,
        sensor_path=resolved_paths["sensor_path"],
        camera_rig_path=resolved_paths["camera_rig_path"],
        key_light_path=resolved_paths["key_light_path"],
        fill_light_path=resolved_paths["fill_light_path"],
        near_clip_m=near_clip_m,
        far_clip_m=far_clip_m,
        profiles=profiles,
    )


def load_profile_catalog(path: str | Path) -> VisionProfileCatalog:
    catalog_path = Path(path).expanduser().resolve()
    return load_profile_catalog_bytes(catalog_path.read_bytes())
