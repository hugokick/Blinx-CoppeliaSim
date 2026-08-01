from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from threading import RLock
from typing import Any

from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.errors import VisionPlatformError

from .catalog import load_profile_catalog_bytes
from .models import AppliedVisionProfile, VisionProfile, VisionProfileCatalog


_PROFILE_ID = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
_ANGLE_TOLERANCE_RAD = math.radians(0.05)
_POSITION_TOLERANCE_M = 0.0005
_LIGHT_TOLERANCE = 0.001
_CLIPPING_TOLERANCE_M = 0.0005
_SENSOR_PATH = "/VisionQualityLab/CameraRig/Camera"

_LightState = tuple[
    int,
    tuple[float, float, float],
    tuple[float, float, float],
]


class _ReadbackMismatch(RuntimeError):
    pass


class VisionProfileError(VisionPlatformError):
    def __init__(self, code: str) -> None:
        super().__init__(code, code)


class VisionProfileValueError(VisionProfileError, ValueError):
    pass


class VisionProfileTypeError(VisionProfileError, TypeError):
    pass


def _profile_context_required() -> VisionProfileValueError:
    return VisionProfileValueError("VISION_PROFILE_CONTEXT_REQUIRED")


def _mark_rollback_failure(error: BaseException) -> None:
    error.vision_profile_rollback_failure = (
        "VISION_PROFILE_ROLLBACK_FAILED"
    )


def controller_for_experiment(
    application: Any,
    definition: Any,
    scene_manifest: Mapping[str, Any],
) -> VisionProfileController | None:
    required = {"camera.profile", "lighting.profile"}
    declared = set(definition.capabilities)
    profile_capabilities = required & declared
    if not profile_capabilities:
        return None
    if profile_capabilities != required:
        raise _profile_context_required()

    config = getattr(application, "config", None)
    sim = getattr(application, "sim", None)
    camera = getattr(application, "camera", None)
    if (
        getattr(config, "camera_backend", None) != "sim"
        or sim is None
        or camera is None
    ):
        raise VisionProfileError("VISION_PROFILE_BACKEND_UNAVAILABLE")

    try:
        scene_path = Path(definition.scene).expanduser().resolve()
        profile_path = (scene_path.parent / "profiles.json").resolve()
        project_root = next(
            candidate
            for candidate in scene_path.parents
            if (candidate / "config" / "experiments").is_dir()
            and (candidate / "simulation").is_dir()
        )
        scene_path.relative_to(project_root / "simulation")
        profile_path.relative_to(project_root / "simulation")
    except (OSError, RuntimeError, StopIteration, TypeError, ValueError) as error:
        raise _profile_context_required() from error

    entry = scene_manifest.get("profile_catalog")
    if not isinstance(entry, Mapping):
        raise _profile_context_required()
    try:
        if set(entry) != {"path", "sha256"}:
            raise _profile_context_required()
        expected_path = profile_path.relative_to(project_root).as_posix()
        if type(entry["path"]) is not str or entry["path"] != expected_path:
            raise _profile_context_required()
        expected_sha256 = entry["sha256"]
        if (
            type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            )
        ):
            raise _profile_context_required()
        content = profile_path.read_bytes()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error) == (
            "VISION_PROFILE_CONTEXT_REQUIRED"
        ):
            raise
        raise _profile_context_required() from error
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise _profile_context_required()

    try:
        catalog = load_profile_catalog_bytes(content)
        allowed = definition.public_parameters.get("allowed_profile_ids")
        baseline = definition.public_parameters.get("baseline_profile_id")
        if (
            type(allowed) is not tuple
            or not allowed
            or any(type(profile_id) is not str for profile_id in allowed)
            or type(baseline) is not str
            or baseline != catalog.baseline_profile_id
        ):
            raise _profile_context_required()
    except VisionProfileValueError:
        raise
    except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as error:
        raise _profile_context_required() from error
    return VisionProfileController(
        sim=sim,
        camera=camera,
        catalog=catalog,
        allowed_profile_ids=allowed,
    )


@dataclass(frozen=True)
class _SceneSnapshot:
    resolution: tuple[int, int]
    perspective_angle_rad: float
    near_clip_m: float
    far_clip_m: float
    rig_position: tuple[float, float, float]
    key_light: _LightState
    fill_light: _LightState


@dataclass(frozen=True)
class PublishedVisionProfileInspection:
    profile: AppliedVisionProfile | None
    resolution: tuple[int, int]
    perspective_angle_deg: float
    near_clip_m: float
    far_clip_m: float
    camera_rig_position_m: tuple[float, float, float]
    key_light_state: int
    key_diffuse_rgb: tuple[float, float, float]
    fill_light_state: int
    fill_diffuse_rgb: tuple[float, float, float]


class VisionProfileController:
    """Control an allowlisted CoppeliaSim profile atomically.

    Transport deadlines and backend quarantine are preconditions enforced by
    the owning runner boundary; this controller does not create worker threads.
    """

    def __init__(
        self,
        sim: Any,
        camera: Any,
        catalog: VisionProfileCatalog,
        allowed_profile_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(catalog, VisionProfileCatalog):
            raise VisionProfileTypeError("VISION_PROFILE_CATALOG_INVALID")
        self._validate_catalog_ambiguity(catalog)
        self._validate_allowed_profile_ids(catalog, allowed_profile_ids)
        self._validate_backend(sim, camera, catalog)

        self._sim = sim
        self._camera = camera
        self._catalog = catalog
        self._allowed_profile_ids = allowed_profile_ids
        self._lock = RLock()

        try:
            self._sensor_handle = sim.getObject(catalog.sensor_path)
            self._camera_rig_handle = sim.getObject(catalog.camera_rig_path)
            self._key_light_handle = sim.getObject(catalog.key_light_path)
            self._fill_light_handle = sim.getObject(catalog.fill_light_path)
        except Exception as error:
            raise VisionProfileError("VISION_PROFILE_OBJECT_MISSING") from error

    @classmethod
    def inspect(
        cls,
        *,
        sim: Any,
        catalog: VisionProfileCatalog,
    ) -> AppliedVisionProfile:
        """Read one published profile without constructing a command controller."""

        inspection = cls.inspect_readback(sim=sim, catalog=catalog)
        if inspection.profile is None:
            raise VisionProfileError("VISION_PROFILE_READBACK_MISMATCH")
        return inspection.profile

    @classmethod
    def inspect_readback(
        cls,
        *,
        sim: Any,
        catalog: VisionProfileCatalog,
    ) -> PublishedVisionProfileInspection:
        """Read one immutable scene snapshot and match it to the catalog."""

        if not isinstance(catalog, VisionProfileCatalog):
            raise VisionProfileTypeError("VISION_PROFILE_CATALOG_INVALID")
        cls._validate_catalog_ambiguity(catalog)

        inspector = cls.__new__(cls)
        inspector._sim = sim
        inspector._catalog = catalog
        try:
            inspector._sensor_handle = sim.getObject(catalog.sensor_path)
            inspector._camera_rig_handle = sim.getObject(
                catalog.camera_rig_path
            )
            inspector._key_light_handle = sim.getObject(
                catalog.key_light_path
            )
            inspector._fill_light_handle = sim.getObject(
                catalog.fill_light_path
            )
        except Exception as error:
            raise VisionProfileError(
                "VISION_PROFILE_OBJECT_MISSING"
            ) from error

        try:
            snapshot = inspector._read_snapshot()
        except _ReadbackMismatch as error:
            raise VisionProfileError(
                "VISION_PROFILE_READBACK_MISMATCH"
            ) from error
        except Exception as error:
            raise VisionProfileError(
                "VISION_PROFILE_BACKEND_UNAVAILABLE"
            ) from error

        matches = [
            profile
            for profile in catalog.profiles
            if inspector._matches_profile(snapshot, profile)
        ]
        profile = (
            inspector._public_state(matches[0])
            if len(matches) == 1
            else None
        )
        return PublishedVisionProfileInspection(
            profile=profile,
            resolution=snapshot.resolution,
            perspective_angle_deg=math.degrees(
                snapshot.perspective_angle_rad
            ),
            near_clip_m=snapshot.near_clip_m,
            far_clip_m=snapshot.far_clip_m,
            camera_rig_position_m=snapshot.rig_position,
            key_light_state=snapshot.key_light[0],
            key_diffuse_rgb=snapshot.key_light[1],
            fill_light_state=snapshot.fill_light[0],
            fill_diffuse_rgb=snapshot.fill_light[1],
        )

    @classmethod
    def _validate_catalog_ambiguity(cls, catalog: VisionProfileCatalog) -> None:
        for index, left in enumerate(catalog.profiles):
            for right in catalog.profiles[index + 1 :]:
                if cls._profiles_overlap(left, right):
                    raise VisionProfileValueError(
                        "VISION_PROFILE_CATALOG_AMBIGUOUS"
                    )

    @classmethod
    def _profiles_overlap(cls, left: VisionProfile, right: VisionProfile) -> bool:
        return (
            left.resolution == right.resolution
            and cls._close(
                math.radians(float(left.perspective_angle_deg)),
                math.radians(float(right.perspective_angle_deg)),
                2 * _ANGLE_TOLERANCE_RAD,
            )
            and cls._close(
                float(left.camera_rig_z_m),
                float(right.camera_rig_z_m),
                2 * _POSITION_TOLERANCE_M,
            )
            and cls._components_close(
                left.key_diffuse_rgb,
                right.key_diffuse_rgb,
                2 * _LIGHT_TOLERANCE,
            )
            and cls._components_close(
                left.fill_diffuse_rgb,
                right.fill_diffuse_rgb,
                2 * _LIGHT_TOLERANCE,
            )
        )

    @staticmethod
    def _validate_backend(
        sim: Any,
        camera: Any,
        catalog: VisionProfileCatalog,
    ) -> None:
        if (
            not isinstance(camera, CoppeliaSimCamera)
            or not callable(getattr(camera, "read", None))
            or catalog.sensor_path != _SENSOR_PATH
            or camera.sensor_path != _SENSOR_PATH
        ):
            raise VisionProfileError("VISION_PROFILE_BACKEND_UNAVAILABLE")
        resolver = getattr(camera, "_resolver", None)
        injected_sim = getattr(resolver, "_injected_sim", None)
        if getattr(camera, "_sim", None) is not sim and injected_sim is not sim:
            raise VisionProfileError("VISION_PROFILE_BACKEND_UNAVAILABLE")

    @staticmethod
    def _validate_allowed_profile_ids(
        catalog: VisionProfileCatalog,
        allowed_profile_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(allowed_profile_ids, tuple) or not allowed_profile_ids:
            raise VisionProfileValueError("VISION_PROFILE_NOT_ALLOWED")
        if any(
            not isinstance(profile_id, str)
            or not profile_id.isascii()
            or _PROFILE_ID.fullmatch(profile_id) is None
            for profile_id in allowed_profile_ids
        ):
            raise VisionProfileValueError("VISION_PROFILE_ID_INVALID")
        if len(set(allowed_profile_ids)) != len(allowed_profile_ids):
            raise VisionProfileValueError("VISION_PROFILE_NOT_ALLOWED")
        if any(profile_id not in catalog.profile_ids for profile_id in allowed_profile_ids):
            raise VisionProfileValueError("VISION_PROFILE_NOT_ALLOWED")
        if catalog.baseline_profile_id not in allowed_profile_ids:
            raise VisionProfileValueError("VISION_PROFILE_NOT_ALLOWED")

    def current(self) -> AppliedVisionProfile:
        with self._lock:
            try:
                snapshot = self._read_snapshot()
            except _ReadbackMismatch as error:
                raise VisionProfileError(
                    "VISION_PROFILE_READBACK_MISMATCH"
                ) from error
            except Exception as error:
                raise VisionProfileError(
                    "VISION_PROFILE_BACKEND_UNAVAILABLE"
                ) from error
            return self._match_published_profile(snapshot)

    def apply(self, profile_id: str) -> AppliedVisionProfile:
        selected = self._allowed_profile(profile_id)
        with self._lock:
            try:
                before = self._read_snapshot()
            except Exception as error:
                raise VisionProfileError("VISION_PROFILE_APPLY_FAILED") from error

            target = self._target_snapshot(before, selected)
            try:
                self._write_snapshot(target)
                after = self._read_snapshot()
                self._assert_snapshot(after, target)
                self._camera.read(timeout_s=2.0)
            except BaseException as original:
                rollback_errors = self._best_effort_restore(before)
                if not isinstance(original, Exception):
                    if rollback_errors:
                        _mark_rollback_failure(original)
                        evidence = BaseExceptionGroup(
                            "VISION_PROFILE_ROLLBACK_EVIDENCE",
                            rollback_errors,
                        )
                        original.add_note(
                            "VISION_PROFILE_ROLLBACK_FAILED during best-effort rollback"
                        )
                        raise original from evidence
                    raise
                for index, rollback_error in enumerate(rollback_errors):
                    if isinstance(rollback_error, Exception):
                        continue
                    evidence = BaseExceptionGroup(
                        "VISION_PROFILE_ROLLBACK_EVIDENCE",
                        [
                            original,
                            *rollback_errors[:index],
                            *rollback_errors[index + 1 :],
                        ],
                    )
                    rollback_error.add_note(
                        f"original apply failure: {type(original).__name__}: {original}"
                    )
                    _mark_rollback_failure(rollback_error)
                    raise rollback_error from evidence
                if rollback_errors:
                    evidence = BaseExceptionGroup(
                        "VISION_PROFILE_ROLLBACK_EVIDENCE",
                        rollback_errors,
                    )
                    failure = VisionProfileError(
                        "VISION_PROFILE_ROLLBACK_FAILED"
                    )
                    failure.add_note(
                        f"original apply failure: {type(original).__name__}: {original}"
                    )
                    _mark_rollback_failure(failure)
                    raise failure from evidence
                raise VisionProfileError(
                    "VISION_PROFILE_APPLY_FAILED"
                ) from original
            return self._public_state(selected)

    def reset(self) -> AppliedVisionProfile:
        with self._lock:
            try:
                return self.apply(self._catalog.baseline_profile_id)
            except Exception as error:
                failure = VisionProfileError("VISION_PROFILE_RESET_FAILED")
                if (
                    getattr(
                        error,
                        "vision_profile_rollback_failure",
                        None,
                    )
                    == "VISION_PROFILE_ROLLBACK_FAILED"
                    or getattr(error, "code", None)
                    == "VISION_PROFILE_ROLLBACK_FAILED"
                ):
                    _mark_rollback_failure(failure)
                raise failure from error

    def _allowed_profile(self, profile_id: str) -> VisionProfile:
        if (
            not isinstance(profile_id, str)
            or not profile_id.isascii()
            or _PROFILE_ID.fullmatch(profile_id) is None
            or profile_id not in self._catalog.profile_ids
        ):
            raise VisionProfileValueError("VISION_PROFILE_ID_INVALID")
        if profile_id not in self._allowed_profile_ids:
            raise VisionProfileValueError("VISION_PROFILE_NOT_ALLOWED")
        return self._catalog.require(profile_id)

    def _read_snapshot(self) -> _SceneSnapshot:
        sim = self._sim
        resolution = (
            self._positive_native_int(
                sim.getObjectInt32Param(
                    self._sensor_handle,
                    sim.visionintparam_resolution_x,
                ),
            ),
            self._positive_native_int(
                sim.getObjectInt32Param(
                    self._sensor_handle,
                    sim.visionintparam_resolution_y,
                ),
            ),
        )
        perspective_angle_rad = self._native_number(
            sim.getObjectFloatParam(
                self._sensor_handle,
                sim.visionfloatparam_perspective_angle,
            )
        )
        near_clip_m = self._native_number(
            sim.getObjectFloatParam(
                self._sensor_handle,
                sim.visionfloatparam_near_clipping,
            )
        )
        far_clip_m = self._native_number(
            sim.getObjectFloatParam(
                self._sensor_handle,
                sim.visionfloatparam_far_clipping,
            )
        )
        rig_position = self._triple(
            sim.getObjectPosition(self._camera_rig_handle, sim.handle_world)
        )
        key_light = self._light_state(sim.getLightParameters(self._key_light_handle))
        fill_light = self._light_state(sim.getLightParameters(self._fill_light_handle))
        return _SceneSnapshot(
            resolution=resolution,
            perspective_angle_rad=perspective_angle_rad,
            near_clip_m=near_clip_m,
            far_clip_m=far_clip_m,
            rig_position=rig_position,
            key_light=key_light,
            fill_light=fill_light,
        )

    @staticmethod
    def _positive_native_int(value: Any) -> int:
        if type(value) is not int or value <= 0:
            raise _ReadbackMismatch("VISION_PROFILE_READBACK_MISMATCH")
        return value

    @staticmethod
    def _native_number(value: Any) -> float:
        if type(value) not in (int, float):
            raise _ReadbackMismatch("VISION_PROFILE_READBACK_MISMATCH")
        result = float(value)
        if not math.isfinite(result):
            raise _ReadbackMismatch("VISION_PROFILE_READBACK_MISMATCH")
        return result

    @staticmethod
    def _triple(value: Any) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise _ReadbackMismatch("VISION_PROFILE_READBACK_MISMATCH")
        result = tuple(VisionProfileController._native_number(component) for component in value)
        return result  # type: ignore[return-value]

    @classmethod
    def _light_state(cls, value: Any) -> _LightState:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise _ReadbackMismatch("VISION_PROFILE_READBACK_MISMATCH")
        state, _, diffuse, specular = value
        if type(state) is not int or state < 0:
            raise _ReadbackMismatch("VISION_PROFILE_READBACK_MISMATCH")
        return state, cls._triple(diffuse), cls._triple(specular)

    def _target_snapshot(
        self,
        before: _SceneSnapshot,
        profile: VisionProfile,
    ) -> _SceneSnapshot:
        key_enabled, _, key_specular = before.key_light
        fill_enabled, _, fill_specular = before.fill_light
        return _SceneSnapshot(
            resolution=profile.resolution,
            perspective_angle_rad=math.radians(float(profile.perspective_angle_deg)),
            near_clip_m=float(self._catalog.near_clip_m),
            far_clip_m=float(self._catalog.far_clip_m),
            rig_position=(
                before.rig_position[0],
                before.rig_position[1],
                float(profile.camera_rig_z_m),
            ),
            key_light=(key_enabled, profile.key_diffuse_rgb, key_specular),
            fill_light=(fill_enabled, profile.fill_diffuse_rgb, fill_specular),
        )

    def _write_snapshot(self, snapshot: _SceneSnapshot) -> None:
        sim = self._sim
        sim.setObjectInt32Param(
            self._sensor_handle,
            sim.visionintparam_resolution_x,
            snapshot.resolution[0],
        )
        sim.setObjectInt32Param(
            self._sensor_handle,
            sim.visionintparam_resolution_y,
            snapshot.resolution[1],
        )
        sim.setObjectFloatParam(
            self._sensor_handle,
            sim.visionfloatparam_perspective_angle,
            snapshot.perspective_angle_rad,
        )
        sim.setObjectFloatParam(
            self._sensor_handle,
            sim.visionfloatparam_near_clipping,
            snapshot.near_clip_m,
        )
        sim.setObjectFloatParam(
            self._sensor_handle,
            sim.visionfloatparam_far_clipping,
            snapshot.far_clip_m,
        )
        sim.setObjectPosition(
            self._camera_rig_handle,
            sim.handle_world,
            list(snapshot.rig_position),
        )
        self._set_light(self._key_light_handle, snapshot.key_light)
        self._set_light(self._fill_light_handle, snapshot.fill_light)

    def _best_effort_restore(self, snapshot: _SceneSnapshot) -> list[BaseException]:
        sim = self._sim
        actions = (
            lambda: self._set_light(self._fill_light_handle, snapshot.fill_light),
            lambda: self._set_light(self._key_light_handle, snapshot.key_light),
            lambda: sim.setObjectPosition(
                self._camera_rig_handle,
                sim.handle_world,
                list(snapshot.rig_position),
            ),
            lambda: sim.setObjectFloatParam(
                self._sensor_handle,
                sim.visionfloatparam_far_clipping,
                snapshot.far_clip_m,
            ),
            lambda: sim.setObjectFloatParam(
                self._sensor_handle,
                sim.visionfloatparam_near_clipping,
                snapshot.near_clip_m,
            ),
            lambda: sim.setObjectFloatParam(
                self._sensor_handle,
                sim.visionfloatparam_perspective_angle,
                snapshot.perspective_angle_rad,
            ),
            lambda: sim.setObjectInt32Param(
                self._sensor_handle,
                sim.visionintparam_resolution_y,
                snapshot.resolution[1],
            ),
            lambda: sim.setObjectInt32Param(
                self._sensor_handle,
                sim.visionintparam_resolution_x,
                snapshot.resolution[0],
            ),
        )
        errors: list[BaseException] = []
        for action in actions:
            try:
                action()
            except BaseException as error:
                errors.append(error)
        try:
            self._assert_snapshot(self._read_snapshot(), snapshot)
        except BaseException as error:
            errors.append(error)
        return errors

    def _set_light(self, handle: Any, state: _LightState) -> None:
        light_state, diffuse, specular = state
        self._sim.setLightParameters(
            handle,
            light_state,
            None,
            list(diffuse),
            list(specular),
        )

    @staticmethod
    def _close(left: float, right: float, tolerance: float) -> bool:
        difference = abs(left - right)
        roundoff = 8 * max(
            math.ulp(left),
            math.ulp(right),
            math.ulp(difference),
            math.ulp(tolerance),
        )
        return difference <= tolerance + roundoff

    @classmethod
    def _assert_snapshot(cls, actual: _SceneSnapshot, expected: _SceneSnapshot) -> None:
        matches = (
            actual.resolution == expected.resolution
            and cls._close(
                actual.perspective_angle_rad,
                expected.perspective_angle_rad,
                _ANGLE_TOLERANCE_RAD,
            )
            and cls._close(actual.near_clip_m, expected.near_clip_m, _CLIPPING_TOLERANCE_M)
            and cls._close(actual.far_clip_m, expected.far_clip_m, _CLIPPING_TOLERANCE_M)
            and cls._components_close(
                actual.rig_position,
                expected.rig_position,
                _POSITION_TOLERANCE_M,
            )
            and cls._lights_close(actual.key_light, expected.key_light)
            and cls._lights_close(actual.fill_light, expected.fill_light)
        )
        if not matches:
            raise _ReadbackMismatch("VISION_PROFILE_READBACK_MISMATCH")

    @classmethod
    def _components_close(
        cls,
        actual: tuple[float, float, float],
        expected: tuple[float, float, float],
        tolerance: float,
    ) -> bool:
        return all(
            cls._close(actual_component, expected_component, tolerance)
            for actual_component, expected_component in zip(actual, expected)
        )

    @classmethod
    def _lights_close(cls, actual: _LightState, expected: _LightState) -> bool:
        return (
            actual[0] == expected[0]
            and cls._components_close(actual[1], expected[1], _LIGHT_TOLERANCE)
            and cls._components_close(actual[2], expected[2], _LIGHT_TOLERANCE)
        )

    def _match_published_profile(self, snapshot: _SceneSnapshot) -> AppliedVisionProfile:
        matches = [
            profile
            for profile in self._catalog.profiles
            if self._matches_profile(snapshot, profile)
        ]
        if len(matches) != 1:
            raise VisionProfileError("VISION_PROFILE_READBACK_MISMATCH")
        return self._public_state(matches[0])

    def _matches_profile(self, snapshot: _SceneSnapshot, profile: VisionProfile) -> bool:
        return (
            snapshot.resolution == profile.resolution
            and snapshot.key_light[0] > 0
            and snapshot.fill_light[0] > 0
            and self._close(
                snapshot.perspective_angle_rad,
                math.radians(float(profile.perspective_angle_deg)),
                _ANGLE_TOLERANCE_RAD,
            )
            and self._close(
                snapshot.near_clip_m,
                float(self._catalog.near_clip_m),
                _CLIPPING_TOLERANCE_M,
            )
            and self._close(
                snapshot.far_clip_m,
                float(self._catalog.far_clip_m),
                _CLIPPING_TOLERANCE_M,
            )
            and self._close(
                snapshot.rig_position[2],
                float(profile.camera_rig_z_m),
                _POSITION_TOLERANCE_M,
            )
            and self._components_close(
                snapshot.key_light[1],
                profile.key_diffuse_rgb,
                _LIGHT_TOLERANCE,
            )
            and self._components_close(
                snapshot.fill_light[1],
                profile.fill_diffuse_rgb,
                _LIGHT_TOLERANCE,
            )
        )

    @staticmethod
    def _public_state(profile: VisionProfile) -> AppliedVisionProfile:
        return AppliedVisionProfile(
            profile_id=profile.profile_id,
            resolution=profile.resolution,
            perspective_angle_deg=profile.perspective_angle_deg,
            camera_rig_z_m=profile.camera_rig_z_m,
            key_diffuse_rgb=profile.key_diffuse_rgb,
            fill_diffuse_rgb=profile.fill_diffuse_rgb,
        )


def inspect_published_profile(
    sim: Any,
    catalog: VisionProfileCatalog,
) -> AppliedVisionProfile:
    return VisionProfileController.inspect(sim=sim, catalog=catalog)


def inspect_published_profile_readback(
    sim: Any,
    catalog: VisionProfileCatalog,
) -> PublishedVisionProfileInspection:
    return VisionProfileController.inspect_readback(sim=sim, catalog=catalog)
