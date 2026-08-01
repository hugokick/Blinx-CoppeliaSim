from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Any

import numpy as np

from vision_platform.errors import VisionPlatformError


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_LAYER_ID = re.compile(r"[a-z][a-z0-9_-]{0,39}\Z")
_STATUSES = frozenset({"PASS", "PARTIAL", "NO_TARGETS", "REJECTED"})
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "AUX",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "CON",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
        "NUL",
        "PRN",
    }
)


class VisionResultBundleError(VisionPlatformError, ValueError):
    def __init__(self, message: str) -> None:
        super().__init__("VISION_RESULT_BUNDLE_INVALID", message)


class VisionResultEvidenceError(VisionPlatformError, ValueError):
    def __init__(self, message: str) -> None:
        super().__init__("VISION_RESULT_EVIDENCE_INVALID", message)


def _utf8_text(value: Any, *, name: str, nonempty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact text")
    if nonempty and not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8 text") from error
    return value


def _validate_id(
    value: Any, *, name: str, pattern: re.Pattern[str]
) -> str:
    selected = _utf8_text(value, name=name, nonempty=True)
    if pattern.fullmatch(selected) is None:
        raise ValueError(f"{name} must be a portable ASCII id")
    if selected.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{name} must not use a reserved id")
    return selected


def _mapping_items(value: Any, *, name: str):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        return value.items()
    except Exception as error:
        raise TypeError(f"{name} must expose mapping items") from error


def _freeze_json(
    value: Any,
    *,
    path: str,
    active: set[int] | None = None,
) -> Any:
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is str:
        return _utf8_text(value, name=path)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return value

    seen = set() if active is None else active
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{path} must not contain a cycle")
        seen.add(identity)
        try:
            copied: dict[str, Any] = {}
            for key, item in _mapping_items(value, name=path):
                selected_key = _utf8_text(key, name=f"{path} key")
                if selected_key in copied:
                    raise ValueError(
                        f"{path} contains a duplicate key: {selected_key}"
                    )
                copied[selected_key] = _freeze_json(
                    item, path=f"{path}.{selected_key}", active=seen
                )
            return MappingProxyType(copied)
        finally:
            seen.remove(identity)

    if type(value) in {list, tuple}:
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{path} must not contain a cycle")
        seen.add(identity)
        try:
            return tuple(
                _freeze_json(item, path=f"{path}[{index}]", active=seen)
                for index, item in enumerate(value)
            )
        finally:
            seen.remove(identity)

    raise TypeError(
        f"{path} must contain only JSON-native values, not "
        f"{type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_thaw_json(item) for item in value]
    return value


def _copy_read_only_bgr(image: Any) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8 BGR pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image dimensions must be positive")
    copied = np.frombuffer(image.tobytes(order="C"), dtype=np.uint8)
    return copied.reshape(image.shape)


@dataclass(frozen=True)
class VisionProfile:
    profile_id: str
    label: str
    resolution: tuple[int, int]
    perspective_angle_deg: float | int
    camera_rig_z_m: float | int
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class VisionProfileCatalog:
    baseline_profile_id: str
    sensor_path: str
    camera_rig_path: str
    key_light_path: str
    fill_light_path: str
    near_clip_m: float | int
    far_clip_m: float | int
    profiles: tuple[VisionProfile, ...]

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(profile.profile_id for profile in self.profiles)

    def require(self, profile_id: str) -> VisionProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(f"unknown vision profile: {profile_id}")


@dataclass(frozen=True)
class AppliedVisionProfile:
    profile_id: str
    resolution: tuple[int, int]
    perspective_angle_deg: float | int
    camera_rig_z_m: float | int
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "resolution": list(self.resolution),
            "perspective_angle_deg": self.perspective_angle_deg,
            "camera_rig_z_m": self.camera_rig_z_m,
            "key_diffuse_rgb": list(self.key_diffuse_rgb),
            "fill_diffuse_rgb": list(self.fill_diffuse_rgb),
        }


@dataclass(frozen=True)
class VisionImageLayer:
    layer_id: str
    title: str
    image_bgr: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            layer_id = _validate_id(
                self.layer_id, name="layer_id", pattern=_LAYER_ID
            )
            title = _utf8_text(
                self.title, name="layer title", nonempty=True
            )
            image = _copy_read_only_bgr(self.image_bgr)
            object.__setattr__(self, "layer_id", layer_id)
            object.__setattr__(self, "title", title)
            object.__setattr__(self, "image_bgr", image)
        except VisionResultBundleError:
            raise
        except Exception as error:
            raise VisionResultBundleError(str(error)) from error


@dataclass(frozen=True)
class VisionResultBundle:
    schema_version: int
    bundle_id: str
    experiment_id: str
    source_snapshot_id: str
    status: str
    layers: tuple[VisionImageLayer, ...]
    result: Mapping[str, Any]
    profile: Mapping[str, Any]
    hardware_status: str = "PENDING_HARDWARE"

    def __post_init__(self) -> None:
        try:
            if type(self.schema_version) is not int or self.schema_version != 1:
                raise ValueError("schema_version must equal integer 1")
            bundle_id = _validate_id(
                self.bundle_id, name="bundle_id", pattern=_ID
            )
            experiment_id = _validate_id(
                self.experiment_id, name="experiment_id", pattern=_ID
            )
            source_snapshot_id = _validate_id(
                self.source_snapshot_id,
                name="source_snapshot_id",
                pattern=_ID,
            )
            status = _utf8_text(self.status, name="status", nonempty=True)
            if status not in _STATUSES:
                raise ValueError(
                    "status must be PASS, PARTIAL, NO_TARGETS or REJECTED"
                )
            if type(self.layers) not in {tuple, list} or not self.layers:
                raise ValueError("layers must be a non-empty ordered collection")
            layers: list[VisionImageLayer] = []
            layer_ids: set[str] = set()
            for layer in self.layers:
                if type(layer) is not VisionImageLayer:
                    raise TypeError("layers must contain VisionImageLayer values")
                copied = VisionImageLayer(
                    layer.layer_id, layer.title, layer.image_bgr
                )
                if copied.layer_id in layer_ids:
                    raise ValueError(f"duplicate layer_id: {copied.layer_id}")
                layer_ids.add(copied.layer_id)
                layers.append(copied)
            result = _freeze_json(self.result, path="result")
            profile = _freeze_json(self.profile, path="profile")
            if not isinstance(result, Mapping):
                raise TypeError("result must be a mapping")
            if not isinstance(profile, Mapping):
                raise TypeError("profile must be a mapping")
            if (
                type(self.hardware_status) is not str
                or self.hardware_status != "PENDING_HARDWARE"
            ):
                raise ValueError(
                    "hardware_status must remain PENDING_HARDWARE"
                )
            object.__setattr__(self, "bundle_id", bundle_id)
            object.__setattr__(self, "experiment_id", experiment_id)
            object.__setattr__(self, "source_snapshot_id", source_snapshot_id)
            object.__setattr__(self, "status", status)
            object.__setattr__(self, "layers", tuple(layers))
            object.__setattr__(self, "result", result)
            object.__setattr__(self, "profile", profile)
            object.__setattr__(
                self, "hardware_status", "PENDING_HARDWARE"
            )
        except VisionResultBundleError:
            raise
        except Exception as error:
            raise VisionResultBundleError(str(error)) from error
