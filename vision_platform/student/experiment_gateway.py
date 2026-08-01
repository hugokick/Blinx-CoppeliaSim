from __future__ import annotations

import errno
import hashlib
import json
from dataclasses import dataclass
from itertools import count
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import cv2
import numpy as np

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import (
    ExperimentDefinition,
    ExperimentRunContext,
)
from vision_platform.experiments.probes import probe_experiment
from vision_platform.vision2d import (
    analyze_image,
    mask_to_curriculum_roi,
    parse_curriculum_config,
    result_to_dict,
)
from vision_platform.vision_quality.controller import controller_for_experiment
from vision_platform.vision_quality.evidence import record_vision_bundle
from vision_platform.vision_quality.models import (
    VisionImageLayer,
    VisionResultBundle,
)


_PROFILE_CONTROLLER_UNSET = object()
_PROFILE_CAPABILITIES = frozenset(
    {"camera.profile", "lighting.profile"}
)
_VISION2D_CAPABILITIES = frozenset(
    {
        "camera.rgb",
        "camera.profile",
        "lighting.profile",
        "vision2d.analysis",
    }
)


@dataclass(frozen=True)
class _RecordedCapture:
    image_bgr: np.ndarray
    value: dict[str, Any]
    record: Mapping[str, Any]
    profile: Mapping[str, Any] | None


@dataclass(frozen=True)
class FileBytesSeal:
    """An immutable exact-byte handoff seal for one small catalog file."""

    path: Path
    size: int
    sha256: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if not self.path.is_absolute():
            raise ValueError("path must be absolute")
        if self.path != self.path.expanduser().resolve():
            raise ValueError("path must be resolved")
        if type(self.size) is not int or self.size < 0:
            raise TypeError("size must be a non-negative integer")
        if type(self.sha256) is not str or (
            len(self.sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.sha256
            )
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if type(self.content) is not bytes:
            raise TypeError("content must be exact bytes")
        if self.size != len(self.content):
            raise ValueError("size does not match content")
        if hashlib.sha256(self.content).hexdigest() != self.sha256:
            raise ValueError("sha256 does not match content")

    @classmethod
    def capture(
        cls,
        path: Path,
        *,
        content: bytes | None = None,
    ) -> FileBytesSeal:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")
        selected = path.expanduser().resolve()
        exact = selected.read_bytes() if content is None else content
        if type(exact) is not bytes:
            raise TypeError("content must be exact bytes")
        return cls(
            path=selected,
            size=len(exact),
            sha256=hashlib.sha256(exact).hexdigest(),
            content=exact,
        )

    def read_verified(self, expected_path: Path) -> bytes:
        if not isinstance(expected_path, Path):
            raise TypeError("expected_path must be a pathlib.Path")
        selected = expected_path.expanduser().resolve()
        if selected != self.path:
            raise ValueError("sealed file path does not match expected path")
        current = selected.read_bytes()
        if (
            len(current) != self.size
            or hashlib.sha256(current).hexdigest() != self.sha256
            or current != self.content
        ):
            raise RuntimeError(
                "sealed file changed during experiment handoff"
            )
        return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _invalid_binding(reason: str) -> VisionPlatformError:
    return VisionPlatformError(
        "EXPERIMENT_CONTEXT_INVALID",
        "实验上下文、定义与场景清单不一致",
        details={"reason": reason},
    )


def _json_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (
            left.keys() == right.keys()
            and all(
                _json_values_equal(left[key], right[key])
                for key in left
            )
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def validate_scene_manifest_file_bytes(
    scene_manifest: Mapping[str, Any],
    file_bytes: bytes,
) -> dict[str, Any]:
    """Match a canonical manifest to one already-read exact file value."""
    try:
        if type(file_bytes) is not bytes:
            raise TypeError("file_bytes must be exact bytes")
        copied_manifest = _copy_json_native(
            scene_manifest,
            path="scene_manifest",
        )
        if not isinstance(copied_manifest, dict):
            raise ValueError("scene_manifest must be a mapping")
        file_manifest = json.loads(
            file_bytes.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
        copied_file_manifest = _copy_json_native(
            file_manifest,
            path="scene_manifest file",
        )
    except BaseException as error:
        raise _invalid_binding(
            f"scene manifest cannot be validated: {type(error).__name__}"
        ) from error
    if not _json_values_equal(copied_manifest, copied_file_manifest):
        raise _invalid_binding("scene_manifest does not match its file")
    return copied_manifest


def _is_probe_transport_error(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    eagain_values = {errno.EAGAIN, errno.EWOULDBLOCK}
    while pending:
        candidate = pending.pop()
        identity = id(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        if getattr(candidate, "errno", None) in eagain_values:
            return True
        candidate_type = type(candidate)
        if (
            candidate_type.__name__ == "Again"
            and candidate_type.__module__.split(".", 1)[0] == "zmq"
        ):
            return True
        for related in (
            getattr(candidate, "__cause__", None),
            getattr(candidate, "__context__", None),
        ):
            if isinstance(related, BaseException):
                pending.append(related)
    return False


def validate_experiment_binding(
    context: ExperimentRunContext | None,
    definition: ExperimentDefinition | None,
    scene_manifest: Mapping[str, Any] | None,
    *,
    scene_manifest_file_bytes: bytes | None = None,
) -> dict[str, Any] | None:
    supplied = (
        context is not None,
        definition is not None,
        scene_manifest is not None,
    )
    if not any(supplied):
        return None
    if not all(supplied):
        raise _invalid_binding(
            "context, definition and scene_manifest must be supplied together"
        )
    if not isinstance(context, ExperimentRunContext):
        raise _invalid_binding("context has the wrong type")
    if not isinstance(definition, ExperimentDefinition):
        raise _invalid_binding("definition has the wrong type")
    assert scene_manifest is not None

    identity_fields = (
        ("context.experiment_id", context.experiment_id),
        ("definition.experiment_id", definition.experiment_id),
        ("context.experiment_version", context.experiment_version),
        ("definition.version", definition.version),
        ("context.scene_sha256", context.scene_sha256),
    )
    for field_name, value in identity_fields:
        if type(value) is not str or not value.strip():
            raise _invalid_binding(
                f"{field_name} must be a non-empty string"
            )

    try:
        if scene_manifest_file_bytes is None:
            manifest_file_bytes = definition.scene_manifest.read_bytes()
        elif type(scene_manifest_file_bytes) is bytes:
            manifest_file_bytes = scene_manifest_file_bytes
        else:
            raise TypeError(
                "scene_manifest_file_bytes must be exact bytes"
            )
    except BaseException as error:
        raise _invalid_binding(
            f"scene manifest cannot be validated: {type(error).__name__}"
        ) from error
    copied_manifest = validate_scene_manifest_file_bytes(
        scene_manifest,
        manifest_file_bytes,
    )
    schema_version = copied_manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise _invalid_binding("scene_manifest schema_version must be 1")
    scene = copied_manifest.get("scene")
    if not isinstance(scene, dict):
        raise _invalid_binding("scene_manifest.scene must be a mapping")
    if not isinstance(copied_manifest.get("task_contracts"), dict):
        raise _invalid_binding(
            "scene_manifest.task_contracts must be a mapping"
        )
    declared_path = scene.get("path")
    declared_hash = scene.get("sha256")
    if type(declared_path) is not str or not declared_path.strip():
        raise _invalid_binding("scene_manifest.scene.path is invalid")
    if (
        type(declared_hash) is not str
        or len(declared_hash) != 64
        or any(character not in "0123456789abcdef" for character in declared_hash)
    ):
        raise _invalid_binding("scene_manifest.scene.sha256 is invalid")

    definition_scene = definition.scene.expanduser().resolve()
    declared = Path(declared_path).expanduser()
    if declared.is_absolute():
        path_matches = declared.resolve() == definition_scene
    else:
        declared_parts = declared.parts
        path_matches = bool(
            declared_parts
            and not any(part in {".", ".."} for part in declared_parts)
            and len(declared_parts) <= len(definition_scene.parts)
            and tuple(part.casefold() for part in declared_parts)
            == tuple(
                part.casefold()
                for part in definition_scene.parts[-len(declared_parts):]
            )
        )
    if not path_matches:
        raise _invalid_binding(
            "scene_manifest.scene.path does not match definition.scene"
        )

    try:
        actual_hash = _sha256(definition_scene)
    except BaseException as error:
        raise _invalid_binding(
            f"definition.scene cannot be hashed: {type(error).__name__}"
        ) from error
    if not (
        context.experiment_id == definition.experiment_id
        and context.experiment_version == definition.version
        and context.scene_path.expanduser().resolve() == definition_scene
        and context.scene_manifest_path.expanduser().resolve()
        == definition.scene_manifest.expanduser().resolve()
        and context.scene_sha256 == declared_hash == actual_hash
    ):
        raise _invalid_binding("context identity or scene digest is inconsistent")
    if (
        context.hardware_status != "PENDING_HARDWARE"
        or definition.hardware_status != "PENDING_HARDWARE"
    ):
        raise _invalid_binding("hardware_status must remain PENDING_HARDWARE")
    try:
        context_parameters = _copy_json_native(
            context.public_parameters,
            path="context.public_parameters",
        )
        definition_parameters = _copy_json_native(
            definition.public_parameters,
            path="definition.public_parameters",
        )
    except BaseException as error:
        raise _invalid_binding(
            f"public parameters cannot be validated: {type(error).__name__}"
        ) from error
    if not _json_values_equal(context_parameters, definition_parameters):
        raise _invalid_binding("context public_parameters do not match definition")
    return copied_manifest


class StudentExperimentGateway:
    def __init__(
        self,
        *,
        application: Any,
        evidence: Any,
        context: ExperimentRunContext,
        definition: ExperimentDefinition,
        scene_manifest: Mapping[str, Any],
        capture_timeout_s: float = 2.0,
        profile_controller: Any = _PROFILE_CONTROLLER_UNSET,
    ) -> None:
        copied_manifest = validate_experiment_binding(
            context,
            definition,
            scene_manifest,
        )
        assert copied_manifest is not None
        self.application = application
        self.evidence = evidence
        self.context = context
        self._definition = definition
        self._scene_manifest = copied_manifest
        self._profile_controller = (
            controller_for_experiment(
                application,
                definition,
                copied_manifest,
            )
            if profile_controller is _PROFILE_CONTROLLER_UNSET
            else profile_controller
        )
        self.capture_timeout_s = float(capture_timeout_s)
        self._snapshot_ids = count(1)
        self._probe_lock = RLock()

    def dispatch(self, name: str, args: Mapping[str, Any]) -> Any:
        if name == "experiment.info":
            self._require_exact_args(name, args, ())
            return self._public_experiment_info()
        if name == "camera.capture":
            self._require_exact_args(name, args, ())
            return self._capture()
        if name == "camera.profile.get":
            self._require_exact_args(name, args, ())
            return self._profile_public(
                self._profiles_required().current(),
                path=name,
            )
        if name == "camera.profile.apply":
            self._require_exact_args(name, args, ("profile_id",))
            return self._profile_public(
                self._profiles_required().apply(args["profile_id"]),
                path=name,
            )
        if name == "camera.profile.reset":
            self._require_exact_args(name, args, ())
            return self._profile_public(
                self._profiles_required().reset(),
                path=name,
            )
        if name == "vision2d.analyze":
            self._require_exact_args(name, args, ())
            return self._analyze_vision2d()
        raise ValueError(f"COMMAND_NOT_ALLOWED: {name}")

    @staticmethod
    def _require_exact_args(
        name: str,
        args: Mapping[str, Any],
        expected: tuple[str, ...],
    ) -> None:
        try:
            keys = tuple(args.keys())
        except BaseException as error:
            raise ValueError(f"{name} arguments are invalid") from error
        if len(keys) == len(expected) and set(keys) == set(expected):
            return
        if not expected:
            raise ValueError(f"{name} does not accept arguments")
        raise ValueError(
            f"{name} arguments must be exactly: {', '.join(expected)}"
        )

    def _profiles_required(self) -> Any:
        if (
            not _PROFILE_CAPABILITIES
            <= set(self._definition.capabilities)
            or self._profile_controller is None
        ):
            raise VisionPlatformError(
                "VISION_PROFILE_CONTEXT_REQUIRED",
                "当前实验没有可用的受控视觉配置",
            )
        return self._profile_controller

    @staticmethod
    def _profile_public(value: Any, *, path: str) -> dict[str, Any]:
        to_public_dict = getattr(value, "to_public_dict", None)
        if not callable(to_public_dict):
            raise TypeError(
                f"{path} must return a JSON-native public profile"
            )
        public = _copy_json_native(
            to_public_dict(),
            path=path,
        )
        if not isinstance(public, dict):
            raise TypeError(
                f"{path} must return a JSON-native public profile"
            )
        json.dumps(public, ensure_ascii=False, allow_nan=False)
        return public

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

    def _capture_profile(self) -> dict[str, Any] | None:
        profile_controller = (
            self._profile_controller
            if _PROFILE_CAPABILITIES
            <= set(self._definition.capabilities)
            else None
        )
        return (
            self._profile_public(
                profile_controller.current(),
                path="camera.capture.vision_profile",
            )
            if profile_controller is not None
            else None
        )

    def _capture(self) -> dict[str, Any]:
        recorded = self._capture_raw(self._capture_profile())
        value = recorded.value
        profile = recorded.profile
        if profile is not None:
            bundle = VisionResultBundle(
                schema_version=1,
                bundle_id=(
                    f"{self.context.experiment_id}-{value['snapshot_id']}"
                ),
                experiment_id=self.context.experiment_id,
                source_snapshot_id=value["snapshot_id"],
                status="PASS",
                layers=(
                    VisionImageLayer("raw", "原图", recorded.image_bgr),
                ),
                result={"snapshot_id": value["snapshot_id"]},
                profile=profile,
                hardware_status="PENDING_HARDWARE",
            )
            value["vision_bundle_path"] = record_vision_bundle(
                self.evidence,
                bundle,
                existing_layer_records={"raw": recorded.record},
            )
        return value

    def _capture_raw(
        self,
        profile: Mapping[str, Any] | None,
    ) -> _RecordedCapture:
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
        if profile is not None:
            resolution = profile.get("resolution")
            if (
                type(resolution) is not list
                or len(resolution) != 2
                or any(type(component) is not int for component in resolution)
                or resolution != [width, height]
            ):
                raise RuntimeError("VISION_PROFILE_RESOLUTION_MISMATCH")

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
        if profile is not None:
            metadata["vision_profile"] = profile
        record = self.evidence.record_snapshot(
            snapshot_id=snapshot_id,
            png_bytes=png_bytes,
            metadata=metadata,
        )
        value = {
            "snapshot_id": snapshot_id,
            "png_bytes": png_bytes,
            **metadata,
            "evidence_path": record["path"],
        }
        return _RecordedCapture(
            image_bgr=image,
            value=value,
            record=record,
            profile=profile,
        )

    def _vision2d_profiles_required(self) -> Any:
        if not _VISION2D_CAPABILITIES <= set(self._definition.capabilities):
            raise VisionPlatformError(
                "VISION2D_CONTEXT_REQUIRED",
                "当前实验没有受控二维视觉分析能力",
            )
        return self._profiles_required()

    def _analyze_vision2d(self) -> dict[str, Any]:
        profile = self._profile_public(
            self._vision2d_profiles_required().current(),
            path="vision2d.analyze.profile",
        )
        resolution = profile.get("resolution")
        if (
            type(resolution) is not list
            or len(resolution) != 2
            or any(type(component) is not int for component in resolution)
        ):
            raise VisionPlatformError(
                "VISION2D_PROFILE_MISMATCH",
                "当前视觉配置没有有效分辨率",
            )
        config = parse_curriculum_config(
            self.context.public_parameters,
            image_size=(resolution[0], resolution[1]),
        )
        if profile.get("profile_id") != config.profile_id:
            raise VisionPlatformError(
                "VISION2D_PROFILE_MISMATCH",
                "当前视觉配置档与实验发布配置不一致",
            )

        recorded = self._capture_raw(profile)
        roi_input = mask_to_curriculum_roi(recorded.image_bgr, config)
        try:
            analysis = analyze_image(roi_input, config.vision2d_config)
            result = result_to_dict(analysis.result)
        except Exception as error:
            raise VisionPlatformError(
                "VISION2D_ANALYSIS_FAILED",
                "二维视觉分析执行失败",
                details={"error_type": type(error).__name__},
            ) from error

        try:
            foreground = cv2.cvtColor(
                analysis.intermediate_images["foreground_mask"],
                cv2.COLOR_GRAY2BGR,
            )
            cleaned = cv2.cvtColor(
                analysis.intermediate_images["cleaned_mask"],
                cv2.COLOR_GRAY2BGR,
            )
            annotated = analysis.intermediate_images["annotated"]
        except (KeyError, TypeError, cv2.error) as error:
            raise VisionPlatformError(
                "VISION2D_ANALYSIS_FAILED",
                "二维视觉分析没有产生完整中间图",
                details={"error_type": type(error).__name__},
            ) from error

        snapshot_id = recorded.value["snapshot_id"]
        bundle_profile = {
            **profile,
            "vision2d": config.to_public_dict(),
        }
        bundle = VisionResultBundle(
            schema_version=1,
            bundle_id=f"{self.context.experiment_id}-{snapshot_id}",
            experiment_id=self.context.experiment_id,
            source_snapshot_id=snapshot_id,
            status=analysis.result.status,
            layers=(
                VisionImageLayer("raw", "原图", recorded.image_bgr),
                VisionImageLayer("roi-input", "分析区域", roi_input),
                VisionImageLayer("foreground-mask", "前景掩膜", foreground),
                VisionImageLayer("cleaned-mask", "清理后掩膜", cleaned),
                VisionImageLayer("annotated", "检测标注", annotated),
            ),
            result=result,
            profile=bundle_profile,
            hardware_status="PENDING_HARDWARE",
        )
        bundle_path = record_vision_bundle(
            self.evidence,
            bundle,
            existing_layer_records={"raw": recorded.record},
        )
        response = {
            "snapshot_id": snapshot_id,
            "vision_bundle_path": bundle_path,
            "profile_id": config.profile_id,
            "status": result["status"],
            "image_size": result["image_size"],
            "targets": result["targets"],
            "rejected_targets": result["rejected_targets"],
        }
        public = _copy_json_native(response, path="vision2d.analyze")
        assert isinstance(public, dict)
        return public

    def reset_environment(self) -> dict[str, Any] | None:
        profile_controller = (
            self._profile_controller
            if _PROFILE_CAPABILITIES
            <= set(self._definition.capabilities)
            else None
        )
        if profile_controller is None:
            return None
        return self._profile_public(
            profile_controller.reset(),
            path="vision.profile.reset",
        )

    def record_probe(self, phase: str) -> dict[str, Any]:
        report = self.collect_probe(phase)
        return self.record_probe_report(phase, report)

    def collect_probe(self, phase: str) -> dict[str, Any]:
        if type(phase) is not str or phase not in {"initial", "final"}:
            raise ValueError("phase must be initial or final")
        with self._probe_lock:
            try:
                report = probe_experiment(
                    self.application.sim,
                    self._definition,
                    phase=phase,
                    scene_manifest=self._scene_manifest,
                )
            except BaseException as error:
                try:
                    message = str(error)
                except BaseException:
                    message = "<unprintable>"
                return self._probe_error_report(
                    phase=phase,
                    code=(
                        "SCENE_PROBE_TRANSPORT_FAILED"
                        if _is_probe_transport_error(error)
                        else "SCENE_PROBE_FAILED"
                    ),
                    message=message,
                    error_type=(
                        f"{type(error).__module__}."
                        f"{type(error).__qualname__}"
                    ),
                )
            return _copy_json_native(report, path=f"scene-{phase}")

    def record_probe_report(
        self,
        phase: str,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        if type(phase) is not str or phase not in {"initial", "final"}:
            raise ValueError("phase must be initial or final")
        return self._record_probe_report(phase, report)

    def record_probe_error(
        self,
        phase: str,
        *,
        code: str,
        message: str,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        if type(phase) is not str or phase not in {"initial", "final"}:
            raise ValueError("phase must be initial or final")
        report = self._probe_error_report(
            phase=phase,
            code=code,
            message=message,
            error_type=error_type,
        )
        return self._record_probe_report(phase, report)

    def _probe_error_report(
        self,
        *,
        phase: str,
        code: str,
        message: str,
        error_type: str | None,
    ) -> dict[str, Any]:
        error = {
            "code": str(code)[:200],
            "message": str(message)[:2000],
        }
        if error_type is not None:
            error["type"] = str(error_type)[:200]
        report = {
            "schema_version": 1,
            "experiment_id": self.context.experiment_id,
            "phase": phase,
            "status": "ERROR",
            "matched": 0,
            "expected": None,
            "tolerance_mm": None,
            "rows": [],
            "error": error,
            "hardware_status": "PENDING_HARDWARE",
        }
        return report

    def _record_probe_report(
        self,
        phase: str,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        canonical_report = _copy_json_native(
            report,
            path=f"scene-{phase}",
        )
        assert isinstance(canonical_report, dict)
        self.evidence.record_json_artifact(
            f"scene-{phase}.json",
            canonical_report,
        )
        return canonical_report


def _copy_json_native(
    value: Any,
    *,
    path: str,
    active_containers: set[int] | None = None,
) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} must contain finite floats")
        return value
    if isinstance(value, Mapping):
        active = active_containers if active_containers is not None else set()
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} must not contain cycles")
        active.add(identity)
        result: dict[str, Any] = {}
        try:
            for key, nested in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} keys must be strings")
                result[key] = _copy_json_native(
                    nested,
                    path=f"{path}.{key}",
                    active_containers=active,
                )
            return result
        finally:
            active.remove(identity)
    if type(value) in {list, tuple}:
        active = active_containers if active_containers is not None else set()
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} must not contain cycles")
        active.add(identity)
        try:
            return [
                _copy_json_native(
                    nested,
                    path=f"{path}[{index}]",
                    active_containers=active,
                )
                for index, nested in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise TypeError(
        f"{path} must contain only JSON-native values, not "
        f"{type(value).__name__}"
    )
