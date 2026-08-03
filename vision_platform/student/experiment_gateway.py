from __future__ import annotations

import errno
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import count
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import (
    ExperimentDefinition,
    ExperimentRunContext,
)
from vision_platform.experiments.code_routing import (
    CodeRouteError,
    build_code_route_plan,
    code_route_plan_to_dict,
)
from vision_platform.experiments.ocr_assets import (
    IDENTIFIERS,
    load_ocr_assets,
)
from vision_platform.experiments.ocr_service import (
    OcrServiceError,
    OcrSortingService,
)
from vision_platform.experiments.ocr_sorting import (
    OcrSortPlan,
    ocr_sort_plan_to_dict,
)
from vision_platform.experiments.probes import probe_experiment
from vision_platform.vision2d import (
    analyze_image,
    mask_to_curriculum_roi,
    parse_curriculum_config,
    result_to_dict,
)
from vision_platform.vision2d.ocr import OCRCharacter, OCRResult
from vision_platform.vision2d.template_matching import (
    TemplateMatchConfig,
    TemplateMatchError,
    annotate_template_match,
    match_template,
    template_match_result_to_dict,
)
from vision_platform.vision2d.code_recognition import CodeRecognitionResult, recognize_codes
from vision_platform.student.code_route_guard import CodeRouteGuard
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
_TEMPLATE_MATCH_CAPABILITIES = frozenset(
    {
        "camera.rgb",
        "camera.profile",
        "lighting.profile",
        "vision2d.template_matching",
    }
)
_CODE_ROUTE_CAPABILITIES = frozenset(
    {
        "camera.rgb", "camera.profile", "lighting.profile",
        "vision2d.code_routing", "robot.home", "robot.pose",
        "robot.move_world", "tool.suction", "scene.probe",
    }
)
_OCR_SORTING_CAPABILITIES = frozenset(
    {"camera.rgb", "camera.profile", "lighting.profile", "vision2d.ocr_sorting"}
)


def _ocr_results_public(
    results: object,
    *,
    frame_size: tuple[int, int],
    confidence_min: float,
) -> list[dict[str, Any]]:
    """Validate and copy the four host-owned OCR results for the student API."""

    if type(results) is not tuple or len(results) != len(IDENTIFIERS):
        raise VisionPlatformError(
            "OCR_SORT_RESULT_INVALID", "OCR 必须返回固定四条识别结果"
        )
    if type(confidence_min) not in {int, float} or not math.isfinite(float(confidence_min)):
        raise VisionPlatformError(
            "OCR_SORT_RESULT_INVALID", "OCR 置信度门槛无效"
        )
    minimum = float(confidence_min)
    if not 0.0 <= minimum <= 1.0:
        raise VisionPlatformError(
            "OCR_SORT_RESULT_INVALID", "OCR 置信度门槛超出范围"
        )
    frame_width, frame_height = frame_size
    if (
        type(frame_width) is not int
        or type(frame_height) is not int
        or frame_width <= 0
        or frame_height <= 0
    ):
        raise VisionPlatformError(
            "OCR_SORT_RESULT_INVALID", "OCR 图像尺寸无效"
        )
    public: list[dict[str, Any]] = []
    for index, (identifier, result) in enumerate(zip(IDENTIFIERS, results)):
        if not isinstance(result, OCRResult):
            raise VisionPlatformError(
                "OCR_SORT_RESULT_INVALID", f"OCR 结果 {identifier} 类型无效"
            )
        if (
            result.status != "PASS"
            or result.text != identifier
            or result.failure_code is not None
            or type(result.schema_version) is not int
            or result.schema_version != 1
            or type(result.character_count) is not int
            or result.character_count != 2
            or type(result.characters) is not tuple
            or len(result.characters) != 2
            or type(result.threshold_method) is not str
            or not result.threshold_method
            or type(result.confidence_method) is not str
            or not result.confidence_method
        ):
            raise VisionPlatformError(
                "OCR_SORT_RESULT_INVALID", f"OCR 结果 {identifier} 不完整"
            )
        image_size = result.image_size
        if (
            type(image_size) is not tuple
            or len(image_size) != 2
            or any(type(value) is not int or value <= 0 for value in image_size)
            or image_size[0] > frame_width
            or image_size[1] > frame_height
        ):
            raise VisionPlatformError(
                "OCR_SORT_RESULT_INVALID", f"OCR 结果 {identifier} 图像尺寸无效"
            )
        processing_ms = result.processing_ms
        if (
            type(processing_ms) not in {int, float}
            or not math.isfinite(float(processing_ms))
            or float(processing_ms) < 0.0
        ):
            raise VisionPlatformError(
                "OCR_SORT_RESULT_INVALID", f"OCR 结果 {identifier} 处理时长无效"
            )
        characters: list[dict[str, Any]] = []
        for char_index, character in enumerate(result.characters):
            if not isinstance(character, OCRCharacter):
                raise VisionPlatformError(
                    "OCR_SORT_RESULT_INVALID",
                    f"OCR 结果 {identifier} 字符类型无效",
                )
            if character.character != identifier[char_index]:
                raise VisionPlatformError(
                    "OCR_SORT_RESULT_INVALID",
                    f"OCR 结果 {identifier} 字符序列无效",
                )
            bbox = character.bbox_px
            if (
                type(bbox) is not tuple
                or len(bbox) != 4
                or any(type(value) is not int for value in bbox)
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] <= 0
                or bbox[3] <= 0
                or bbox[0] + bbox[2] > image_size[0]
                or bbox[1] + bbox[3] > image_size[1]
            ):
                raise VisionPlatformError(
                    "OCR_SORT_RESULT_INVALID",
                    f"OCR 结果 {identifier} 字符框越出图像",
                )
            confidence = character.confidence
            if (
                type(confidence) not in {int, float}
                or not math.isfinite(float(confidence))
                or not minimum <= float(confidence) <= 1.0
                or character.failure_code is not None
                or type(character.confidence_method) is not str
                or not character.confidence_method
            ):
                raise VisionPlatformError(
                    "OCR_SORT_RESULT_INVALID",
                    f"OCR 结果 {identifier} 字符置信度无效",
                )
            characters.append(
                {
                    "character": character.character,
                    "bbox_px": list(bbox),
                    "confidence": float(confidence),
                    "failure_code": None,
                    "confidence_method": character.confidence_method,
                }
            )
        public.append(
            {
                "identifier": identifier,
                "status": "PASS",
                "text": identifier,
                "characters": characters,
                "image_size": list(image_size),
                "threshold_method": result.threshold_method,
                "character_count": 2,
                "failure_code": None,
                "processing_ms": float(processing_ms),
                "schema_version": 1,
                "confidence_method": result.confidence_method,
            }
        )
    return public


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


def _sha256_file(path: Path) -> str:
    return _sha256(path)


def _annotate_code_routes(
    image_bgr: np.ndarray,
    recognition: CodeRecognitionResult,
    plan: Any,
) -> np.ndarray:
    annotated = image_bgr.copy()
    entry_by_payload = {entry.payload: entry for entry in plan.entries}
    for reading in recognition.readings:
        entry = entry_by_payload[reading.data]
        points = np.rint(np.asarray(reading.polygon_px, dtype=np.float64)).astype(np.int32)
        cv2.polylines(annotated, [points], True, (0, 180, 0), 2, cv2.LINE_AA)
        origin = (max(0, int(reading.bbox_px[0])), max(16, int(reading.bbox_px[1]) - 4))
        cv2.putText(
            annotated,
            f"{entry.part_id}->{entry.route_id}",
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 100, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _template_root(scene_manifest_path: Path) -> Path:
    """Return the bounded root from which fixed template files may load."""

    resolved = scene_manifest_path.expanduser().resolve()
    for parent in (resolved.parent, *resolved.parents):
        if parent.name.casefold() == "simulation":
            return parent.parent.resolve()
    return resolved.parent


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_template_file(path: Path, root: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = candidate.absolute()
    bounded_root = root.expanduser().resolve()
    if not _path_is_under(candidate, bounded_root):
        raise ValueError("template path is outside the allowed template root")

    current = candidate
    while True:
        if current.is_symlink():
            raise ValueError("template path cannot use a symlink")
        if current == bounded_root or current.parent == current:
            break
        current = current.parent

    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not _path_is_under(resolved, bounded_root):
        raise ValueError("template path is outside the allowed template root")
    if resolved != candidate:
        raise ValueError("template path cannot resolve through a link")
    return resolved


def _find_template_file(base_dir: Path, relative_path: Path, root: Path) -> Path:
    bounded_root = root.expanduser().resolve()
    for parent in (base_dir.expanduser().resolve(), *base_dir.expanduser().resolve().parents):
        candidate = parent / relative_path
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if not _path_is_under(parent, bounded_root):
            return _safe_template_file(candidate, bounded_root)
        return _safe_template_file(candidate, bounded_root)
    raise FileNotFoundError(f"template file was not found: {relative_path}")


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
        ocr_guard_activator: Callable[[OcrSortPlan], Any] | None = None,
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
        self.route_guard = CodeRouteGuard()
        self._ocr_guard_activator = ocr_guard_activator
        self._ocr_evidence: dict[str, Any] | None = None
        self._route_evidence: dict[str, Any] | None = None
        self._route_completion_before_reset: dict[str, Any] | None = None

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
        if name == "vision2d.template_match":
            self._require_exact_args(name, args, ())
            return self._template_match()
        if name == "vision2d.code_routes":
            self._require_exact_args(name, args, ())
            return self._code_routes()
        if name == "vision2d.ocr_sorting":
            self._require_exact_args(name, args, ())
            return self._ocr_sorting()
        if name == "vision2d.ocr_sort_entry":
            self._require_exact_args(name, args, ("entry_id",))
            # The runner owns the private action executor.  This branch is
            # intentionally never used for motion; it exists only to make a
            # direct gateway misuse fail closed with a stable code.
            raise VisionPlatformError(
                "OCR_SORT_ENTRY_RUNNER_ONLY",
                "OCR 分拣动作必须由运行器的私有安全执行器调用",
            )
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

    def _template_profiles_required(self) -> Any:
        if not _TEMPLATE_MATCH_CAPABILITIES <= set(self._definition.capabilities):
            raise VisionPlatformError(
                "VISION_TEMPLATE_CONTEXT_REQUIRED",
                "当前实验没有受控模板匹配能力",
            )
        return self._profiles_required()

    def _resolve_catalog_path(self, relative_path: str) -> Path:
        selected = Path(relative_path)
        if selected.is_absolute() or any(part in {".", ".."} for part in selected.parts):
            raise ValueError("template catalog path must be a safe relative path")
        manifest_path = self._definition.scene_manifest.expanduser().resolve()
        return _find_template_file(
            manifest_path.parent,
            selected,
            _template_root(manifest_path),
        )

    def _template_asset(self) -> tuple[np.ndarray, TemplateMatchConfig, dict[str, Any]]:
        binding = self._scene_manifest.get("template_catalog")
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise VisionPlatformError(
                "VISION_TEMPLATE_ASSET_INVALID",
                "场景清单没有完整模板清单绑定",
            )
        catalog_path_value = binding.get("path")
        catalog_hash = binding.get("sha256")
        if (
            type(catalog_path_value) is not str
            or not catalog_path_value
            or type(catalog_hash) is not str
            or len(catalog_hash) != 64
            or any(character not in "0123456789abcdef" for character in catalog_hash)
        ):
            raise VisionPlatformError(
                "VISION_TEMPLATE_ASSET_INVALID",
                "模板清单绑定格式无效",
            )
        try:
            catalog_path = self._resolve_catalog_path(catalog_path_value)
            catalog_bytes = catalog_path.read_bytes()
            if _sha256(catalog_path) != catalog_hash:
                raise ValueError("template catalog hash mismatch")
            catalog = json.loads(catalog_bytes.decode("utf-8"))
            if not isinstance(catalog, Mapping):
                raise ValueError("template catalog must be a mapping")
            required = {
                "schema_version",
                "template_id",
                "template_version",
                "asset_path",
                "sha256",
                "size_px",
                "channels",
                "method",
                "threshold",
                "search_roi_px",
            }
            if not required <= set(catalog):
                raise ValueError("template catalog fields are incomplete")
            asset_path_value = catalog["asset_path"]
            if type(asset_path_value) is not str or not asset_path_value:
                raise ValueError("template asset path is invalid")
            asset_relative = Path(asset_path_value)
            if asset_relative.is_absolute() or any(
                part in {".", ".."} for part in asset_relative.parts
            ):
                raise ValueError("template asset path is unsafe")
            asset_path = _find_template_file(
                catalog_path.parent,
                asset_relative,
                _template_root(self._definition.scene_manifest),
            )
            if _sha256(asset_path) != catalog["sha256"]:
                raise ValueError("template asset hash mismatch")
            image = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
            if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("template asset is not an 8-bit BGR image")
            size_px = catalog["size_px"]
            if (
                type(size_px) is not list
                or len(size_px) != 2
                or any(type(value) is not int for value in size_px)
                or [image.shape[1], image.shape[0]] != size_px
                or catalog["channels"] != 3
            ):
                raise ValueError("template asset dimensions do not match catalog")
            config = TemplateMatchConfig(
                template_id=catalog["template_id"],
                template_version=catalog["template_version"],
                threshold=catalog["threshold"],
                search_roi_px=tuple(catalog["search_roi_px"]),
                method=catalog["method"],
            )
            return image, config, {
                "template_id": config.template_id,
                "template_version": config.template_version,
                "method": config.method,
                "threshold": config.threshold,
                "search_roi_px": list(config.search_roi_px),
                "sha256": catalog["sha256"],
            }
        except VisionPlatformError:
            raise
        except Exception as error:
            raise VisionPlatformError(
                "VISION_TEMPLATE_ASSET_INVALID",
                "固定模板资产校验失败",
                details={"error_type": type(error).__name__},
            ) from error

    def _template_match(self) -> dict[str, Any]:
        profile = self._profile_public(
            self._template_profiles_required().current(),
            path="vision2d.template_match.profile",
        )
        resolution = profile.get("resolution")
        if (
            type(resolution) is not list
            or len(resolution) != 2
            or any(type(component) is not int for component in resolution)
        ):
            raise VisionPlatformError(
                "VISION_TEMPLATE_PROFILE_MISMATCH",
                "当前视觉配置没有有效分辨率",
            )
        template, config, template_public = self._template_asset()
        recorded = self._capture_raw(profile)
        try:
            result = match_template(recorded.image_bgr, template, config)
            annotated = annotate_template_match(recorded.image_bgr, result)
        except TemplateMatchError as error:
            raise VisionPlatformError(
                error.code,
                "模板匹配执行失败",
                details={"error_type": type(error).__name__},
            ) from error
        result_dict = template_match_result_to_dict(result)
        snapshot_id = recorded.value["snapshot_id"]
        result_dict["snapshot_id"] = snapshot_id
        bundle = VisionResultBundle(
            schema_version=1,
            bundle_id=f"{self.context.experiment_id}-{snapshot_id}",
            experiment_id=self.context.experiment_id,
            source_snapshot_id=snapshot_id,
            status="PASS" if result.matched else "PARTIAL",
            layers=(
                VisionImageLayer("raw", "原图", recorded.image_bgr),
                VisionImageLayer("template", "固定模板", template),
                VisionImageLayer("annotated", "模板匹配标注", annotated),
            ),
            result=result_dict,
            profile={**profile, "template": template_public},
            hardware_status="PENDING_HARDWARE",
        )
        bundle_path = record_vision_bundle(
            self.evidence,
            bundle,
            existing_layer_records={"raw": recorded.record},
        )
        response = {
            "schema_version": result.schema_version,
            "snapshot_id": snapshot_id,
            "vision_bundle_path": bundle_path,
            "template_id": result.template_id,
            "template_version": result.template_version,
            "matched": result.matched,
            "status": result.status,
            "score": result.score,
            "threshold": result.threshold,
            "bbox_px": list(result.bbox_px) if result.bbox_px is not None else None,
            "center_px": list(result.center_px) if result.center_px is not None else None,
            "image_size": list(result.image_size),
            "search_roi_px": list(result.search_roi_px),
            "method": result.method,
        }
        public = _copy_json_native(response, path="vision2d.template_match")
        assert isinstance(public, dict)
        return public

    def _ocr_assets_path(self) -> Path:
        binding = self._scene_manifest.get("ocr_assets_manifest")
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or type(binding.get("path")) is not str
            or type(binding.get("sha256")) is not str
            or len(binding["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in binding["sha256"])
        ):
            raise VisionPlatformError(
                "OCR_SORT_ASSET_INVALID", "OCR 资产清单绑定缺失或格式无效"
            )
        try:
            manifest_path = self._resolve_catalog_path(binding["path"])
            if _sha256(manifest_path) != binding["sha256"]:
                raise ValueError("OCR asset manifest hash mismatch")
            if manifest_path.name != "ocr_assets_manifest.json":
                raise ValueError("OCR asset manifest has an invalid name")
            return manifest_path
        except VisionPlatformError:
            raise
        except Exception as error:
            raise VisionPlatformError(
                "OCR_SORT_ASSET_INVALID", "OCR 资产清单校验失败",
                details={"error_type": type(error).__name__},
            ) from error

    def _ocr_fixed_rois(self, image_size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
        binding = self._scene_manifest.get("ocr_sorting")
        parameters = self._definition.public_parameters.get("ocr_sorting")
        candidates: Any = None
        for source in (parameters, binding):
            if isinstance(source, Mapping) and "fixed_rois_px" in source:
                candidates = source["fixed_rois_px"]
                break
        if not isinstance(candidates, Mapping) or set(candidates) != set(IDENTIFIERS):
            raise VisionPlatformError(
                "OCR_SORT_ROI_INVALID", "OCR 固定 ROI 必须完整绑定 A1、A2、B1、B2"
            )
        width, height = image_size
        result: dict[str, tuple[int, int, int, int]] = {}
        for identifier in IDENTIFIERS:
            value = candidates[identifier]
            if (
                type(value) not in {list, tuple}
                or len(value) != 4
                or any(type(item) is not int for item in value)
            ):
                raise VisionPlatformError(
                    "OCR_SORT_ROI_INVALID", f"{identifier} 的固定 ROI 格式无效"
                )
            x, y, roi_width, roi_height = value
            if (
                x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0
                or x + roi_width > width or y + roi_height > height
            ):
                raise VisionPlatformError(
                    "OCR_SORT_ROI_INVALID", f"{identifier} 的固定 ROI 越出图像"
                )
            result[identifier] = (x, y, roi_width, roi_height)
        return result

    def _ocr_sort_config(self) -> Mapping[str, object] | None:
        parameters = self._definition.public_parameters.get("ocr_sorting")
        if not isinstance(parameters, Mapping):
            return None
        selected = parameters.get("sort_config")
        if selected is None:
            required = {
                "schema_version", "expected_count", "training_accuracy_min",
                "confidence_min", "safe_z_mm", "speed_mm_s", "workspace", "routes",
            }
            if required <= set(parameters):
                selected = parameters
        return selected if isinstance(selected, Mapping) else None

    def _ocr_scene_part_ids(self) -> tuple[str, ...]:
        binding = self._scene_manifest.get("ocr_sorting")
        if not isinstance(binding, Mapping):
            raise VisionPlatformError("OCR_SORT_CONFIG_INVALID", "场景缺少 OCR 分拣绑定")
        values = binding.get("part_ids")
        if type(values) is not list or tuple(values) != ("part_a", "part_b", "part_c", "part_d"):
            raise VisionPlatformError("OCR_SORT_CONFIG_INVALID", "场景构件集合不是固定四件")
        return tuple(values)

    def _ocr_sorting(self) -> dict[str, Any]:
        if not _OCR_SORTING_CAPABILITIES <= set(self._definition.capabilities):
            raise VisionPlatformError(
                "OCR_SORT_CONTEXT_REQUIRED", "当前实验没有受控 OCR 分拣能力"
            )
        if self._ocr_evidence is not None:
            raise VisionPlatformError(
                "OCR_SORT_PLAN_ALREADY_ACTIVE", "本次运行已有 OCR 分拣计划"
            )
        if not callable(self._ocr_guard_activator):
            raise VisionPlatformError(
                "OCR_SORT_EXECUTOR_REQUIRED", "OCR 分拣需要运行器安全守卫"
            )
        try:
            manifest_path = self._ocr_assets_path()
            profile = self._profile_public(
                self._profiles_required().current(),
                path="vision2d.ocr_sorting.profile",
            )
            if profile.get("profile_id") != "standard" or profile.get("resolution") != [1024, 1024]:
                raise VisionPlatformError(
                    "OCR_SORT_PROFILE_MISMATCH", "OCR 分拣必须使用 standard 1024x1024"
                )
            recorded = self._capture_raw(profile)
            fixed_rois = self._ocr_fixed_rois(
                (int(recorded.image_bgr.shape[1]), int(recorded.image_bgr.shape[0]))
            )
            sort_config = self._ocr_sort_config()
            service = OcrSortingService.from_manifest(
                manifest_path,
                minimum_accuracy=float(
                    (sort_config or {}).get("training_accuracy_min", 0.95)
                ),
                sort_config=sort_config,
                scene_part_ids=self._ocr_scene_part_ids(),
            )
            output = service.analyze(recorded.image_bgr, fixed_rois)
            results_public = _ocr_results_public(
                output.results,
                frame_size=(
                    int(recorded.image_bgr.shape[1]),
                    int(recorded.image_bgr.shape[0]),
                ),
                confidence_min=(sort_config or {}).get("confidence_min", 0.40),
            )
            plan = output.plan
            plan_public = ocr_sort_plan_to_dict(plan)
            report = output.training_report
            training_public = {
                "train_count": int(report.train_count),
                "test_count": int(report.test_count),
                "held_out_accuracy": float(report.held_out_accuracy),
                "seed": int(report.seed),
            }
            annotated = np.ascontiguousarray(output.annotated_frame).copy()
            bundle_profile = {
                **profile,
                "ocr": {
                    "scene_id": output.scene_id,
                    "manifest_sha256": output.manifest_sha256,
                    "plan_id": plan.plan_id,
                },
            }
            bundle_result = {
                "plan": plan_public,
                "training": training_public,
                "results": results_public,
            }
            bundle = VisionResultBundle(
                schema_version=1,
                bundle_id=f"{self.context.experiment_id}-{recorded.value['snapshot_id']}",
                experiment_id=self.context.experiment_id,
                source_snapshot_id=recorded.value["snapshot_id"],
                status="PASS",
                layers=(
                    VisionImageLayer("raw", "原图", recorded.image_bgr),
                    VisionImageLayer("annotated", "OCR 编号标注", annotated),
                ),
                result=bundle_result,
                profile=bundle_profile,
                hardware_status="PENDING_HARDWARE",
            )
            bundle_path = record_vision_bundle(
                self.evidence,
                bundle,
                existing_layer_records={"raw": recorded.record},
            )
            self._ocr_guard_activator(plan)
        except VisionPlatformError:
            raise
        except OcrServiceError as error:
            raise VisionPlatformError(error.code, "OCR 识别或分拣计划校验失败") from error
        except Exception as error:
            raise VisionPlatformError(
                getattr(error, "code", "OCR_SORT_RECOGNITION_FAILED"),
                "OCR 识别或分拣计划生成失败",
                details={"error_type": type(error).__name__},
            ) from error

        bundle_id = f"{self.context.experiment_id}-{recorded.value['snapshot_id']}"
        evidence_public = {
            "raw_path": recorded.record["path"],
            "annotated_path": f"frames/{bundle_id}-annotated.png",
            "bundle_path": bundle_path,
        }
        response = {
            "schema_version": 1,
            "snapshot_id": recorded.value["snapshot_id"],
            "vision_bundle_path": bundle_path,
            "manifest_sha256": output.manifest_sha256,
            "scene_id": output.scene_id,
            "status": "PASS",
            "training": training_public,
            "results": results_public,
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "part_id": entry.part_id,
                    "identifier": entry.identifier,
                    "route_id": entry.route_id,
                    "roi_px": list(entry.roi_px),
                    "confidence": float(entry.confidence),
                    "status": "APPROVED",
                }
                for entry in plan.entries
            ],
            "plan_id": plan.plan_id,
            "safe_z_mm": float(plan.safe_z_mm),
            "speed_mm_s": float(plan.speed_mm_s),
            "evidence": evidence_public,
            "hardware_status": "PENDING_HARDWARE",
        }
        copied = _copy_json_native(response, path="vision2d.ocr_sorting")
        assert isinstance(copied, dict)
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
        self._ocr_evidence = {
            "plan": plan,
            "snapshot_id": recorded.value["snapshot_id"],
            "raw": recorded.image_bgr.copy(),
            "annotated": annotated.copy(),
            "raw_record": dict(recorded.record),
            "profile": bundle_profile,
            "manifest_sha256": output.manifest_sha256,
            "scene_id": output.scene_id,
        }
        return copied

    def _validated_code_asset_binding(self) -> dict[str, tuple[float, float, float]]:
        binding = self._scene_manifest.get("code_assets_manifest")
        route_binding = self._scene_manifest.get("code_routing")
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or type(binding.get("path")) is not str
            or type(binding.get("sha256")) is not str
            or len(binding["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in binding["sha256"])
            or not isinstance(route_binding, Mapping)
            or set(route_binding)
            != {
                "part_ids", "initial_positions_mm",
                "calibration_plane_z_mm", "calibration_matrix", "route_slots_mm",
            }
        ):
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面或构件绑定缺失")
        relative = Path(binding["path"])
        root = self._definition.scene_manifest.parent.resolve()
        if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单路径不安全")
        candidate = root / relative
        if candidate.is_symlink():
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单不得使用链接")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != root or _sha256_file(resolved) != binding["sha256"]:
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单哈希或路径不匹配")
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as error:
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单格式无效") from error
        if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "entries"} or payload["schema_version"] != 1:
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面清单格式无效")
        route_config = self._definition.public_parameters.get("code_routing")
        if not isinstance(route_config, Mapping):
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "路线配置无效")
        routes = route_config.get("routes")
        if not isinstance(payload["entries"], list) or not isinstance(routes, (tuple, list)):
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面条目无效")
        try:
            expected = {
                (route["part_id"], route["code_type"], route["payload"])
                for route in routes
            }
        except (KeyError, TypeError) as error:
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "路线配置无效") from error
        entries = payload["entries"]
        required_entry_fields = {"part_id", "code_type", "payload"}
        if (
            len(entries) != len(routes)
            or any(
                not isinstance(entry, Mapping)
                or not required_entry_fields <= set(entry)
                or any(type(entry[field]) is not str for field in required_entry_fields)
                for entry in entries
            )
        ):
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面条目无效")
        actual = {(entry["part_id"], entry["code_type"], entry["payload"]) for entry in entries}
        part_ids = route_binding["part_ids"]
        initial_positions = route_binding["initial_positions_mm"]

        def matrix(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "标定矩阵无效")
            rows = []
            for row in value:
                if not isinstance(row, (list, tuple)) or len(row) != 3:
                    raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "标定矩阵无效")
                if any(type(component) not in {int, float} or not math.isfinite(float(component)) for component in row):
                    raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "标定矩阵无效")
                rows.append(tuple(float(component) for component in row))
            return (rows[0], rows[1])  # type: ignore[return-value]

        route_slots = route_binding["route_slots_mm"]
        if not isinstance(route_slots, Mapping) or any(
            type(route_id) is not str or not isinstance(slots, (list, tuple))
            for route_id, slots in route_slots.items()
        ):
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "仓位绑定无效")
        if sum(len(slots) for slots in route_slots.values()) != len(routes) or any(
            not isinstance(slot, (list, tuple))
            or len(slot) != 3
            or any(type(component) not in {int, float} or not math.isfinite(float(component)) for component in slot)
            for slots in route_slots.values()
            for slot in slots
        ):
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "仓位绑定无效")
        manifest_slots = {
            (route_id, tuple(float(component) for component in slot))
            for route_id, slots in route_slots.items()
            for slot in slots
        }
        configured_slots = {
            (route["route_id"], tuple(float(component) for component in route["drop_xyz_mm"]))
            for route in routes
        }
        if (
            actual != expected
            or type(part_ids) is not list
            or len(part_ids) != len(expected)
            or len(set(part_ids)) != len(part_ids)
            or any(type(part_id) is not str for part_id in part_ids)
            or set(part_ids) != {item[0] for item in expected}
            or not isinstance(initial_positions, Mapping)
            or set(initial_positions) != set(part_ids)
            or any(
                not isinstance(position, (list, tuple))
                or len(position) != 3
                or any(type(component) not in {int, float} or not math.isfinite(float(component)) for component in position)
                for position in initial_positions.values()
            )
            or type(route_binding["calibration_plane_z_mm"]) not in {int, float}
            or not math.isfinite(float(route_binding["calibration_plane_z_mm"]))
            or float(route_binding["calibration_plane_z_mm"]) != 27.4
            or matrix(route_binding["calibration_matrix"]) != matrix(route_config.get("calibration_matrix"))
            or len(manifest_slots) != len(routes)
            or configured_slots != manifest_slots
        ):
            raise VisionPlatformError("CODE_ROUTE_CONFIG_INVALID", "码面、路线和场景构件不一致")
        return {
            part_id: tuple(float(component) for component in initial_positions[part_id])
            for part_id in part_ids
        }

    def _code_routes(self) -> dict[str, Any]:
        if self.route_guard.completion()["active"]:
            raise VisionPlatformError("CODE_ROUTE_PLAN_ALREADY_ACTIVE", "本次运行已有路线计划")
        if not _CODE_ROUTE_CAPABILITIES <= set(self._definition.capabilities):
            raise VisionPlatformError("CODE_ROUTE_CONTEXT_REQUIRED", "当前实验没有代码路由能力")
        scene_part_positions = self._validated_code_asset_binding()
        profile = self._profile_public(self._profiles_required().current(), path="vision2d.code_routes.profile")
        if profile.get("profile_id") != "standard" or profile.get("resolution") != [1024, 1024]:
            raise VisionPlatformError("CODE_ROUTE_PROFILE_MISMATCH", "代码路由必须使用 standard 1024x1024")
        recorded = self._capture_raw(profile)
        recognition = recognize_codes(recorded.image_bgr, max_codes=4)
        try:
            plan = build_code_route_plan(
                recognition,
                routing_config=self._definition.public_parameters["code_routing"],
                workspace=self._definition.workspace,
                scene_part_ids=set(scene_part_positions),
                image_size=(recorded.image_bgr.shape[1], recorded.image_bgr.shape[0]),
            )
        except CodeRouteError as error:
            raise VisionPlatformError(error.code, str(error)) from error
        for entry in plan.entries:
            expected = scene_part_positions[entry.part_id]
            if (
                math.hypot(entry.pick_xyz_mm[0] - expected[0], entry.pick_xyz_mm[1] - expected[1]) > 3.0
                or abs(entry.pick_xyz_mm[2] - expected[2]) > 0.25
            ):
                raise VisionPlatformError("CODE_ROUTE_SCENE_POSITION_MISMATCH", f"{entry.part_id} 的视觉坐标与绑定初态不一致")
        annotated = _annotate_code_routes(recorded.image_bgr, recognition, plan)
        result = code_route_plan_to_dict(plan)
        snapshot_id = recorded.value["snapshot_id"]
        bundle = VisionResultBundle(
            schema_version=1,
            bundle_id=f"{self.context.experiment_id}-{snapshot_id}",
            experiment_id=self.context.experiment_id,
            source_snapshot_id=snapshot_id,
            status="PASS",
            layers=(
                VisionImageLayer("raw", "原图", recorded.image_bgr),
                VisionImageLayer("annotated", "代码与仓位标注", annotated),
            ),
            result=result,
            profile={**profile, "code_route_plan_id": plan.plan_id},
            hardware_status="PENDING_HARDWARE",
        )
        bundle_path = record_vision_bundle(
            self.evidence,
            bundle,
            existing_layer_records={"raw": recorded.record},
        )
        self.route_guard.activate(plan)
        public = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "vision_bundle_path": bundle_path,
            **result,
        }
        copied = _copy_json_native(public, path="vision2d.code_routes")
        assert isinstance(copied, dict)
        self._route_evidence = {
            "plan": plan,
            "snapshot_id": snapshot_id,
            "raw": recorded.image_bgr.copy(),
            "annotated": annotated.copy(),
            "raw_record": dict(recorded.record),
            "profile": {**profile, "code_route_plan_id": plan.plan_id},
        }
        self._route_completion_before_reset = None
        return copied

    def record_code_route_final(
        self,
        *,
        final_probe: Mapping[str, Any] | None,
        primary_error: Mapping[str, Any] | None,
    ) -> str | None:
        context = self._route_evidence
        if context is None:
            return None
        plan = context["plan"]
        completion = (
            getattr(self, "_route_completion_before_reset", None)
            or self.route_guard.completion()
        )
        probe = (
            {}
            if final_probe is None
            else _copy_json_native(final_probe, path="code_route.final_probe")
        )
        commands_path = Path(self.evidence.directory) / "commands.jsonl"
        commands = []
        if commands_path.is_file():
            for line in commands_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item.get("name") in {"robot.home", "robot.move_world", "tool.on", "tool.off"}:
                    commands.append(
                        {
                            "plan_id": plan.plan_id,
                            "sequence": len(commands) + 1,
                            "name": item["name"],
                            "status": item.get("status", "UNKNOWN"),
                        }
                    )
        probe_pass = isinstance(probe, Mapping) and probe.get("status") == "PASS"
        success = completion["all_complete"] and probe_pass and primary_error is None
        error = (
            None
            if success
            else _copy_json_native(
                primary_error
                or {"code": "CODE_ROUTE_FINAL_STATE_INVALID", "message": "路线终态未通过"},
                path="code_route.error",
            )
        )
        result = {
            **code_route_plan_to_dict(plan),
            "status": "PASS" if success else "REJECTED",
            "snapshot_id": context["snapshot_id"],
            "motion_events": commands,
            "completed_entry_ids": completion["completed_entry_ids"],
            "final_occupancy": probe.get("final_occupancy", {}) if isinstance(probe, Mapping) else {},
            "error": error,
            "human_acceptance": "PENDING_HUMAN_ACCEPTANCE",
            "hardware_status": "PENDING_HARDWARE",
        }
        bundle = VisionResultBundle(
            schema_version=1,
            bundle_id=f"V1-07-zfinal-{context['snapshot_id']}",
            experiment_id=self.context.experiment_id,
            source_snapshot_id=context["snapshot_id"],
            status="PASS" if success else "REJECTED",
            layers=(
                VisionImageLayer("raw", "原图", context["raw"]),
                VisionImageLayer("annotated", "代码与仓位标注", context["annotated"]),
            ),
            result=result,
            profile=context["profile"],
            hardware_status="PENDING_HARDWARE",
        )
        artifact = record_vision_bundle(
            self.evidence,
            bundle,
            existing_layer_records={"raw": context["raw_record"]},
        )
        self._route_evidence = None
        self._route_completion_before_reset = None
        return artifact

    def route_validate_move(self, current: Any, target: Any) -> None:
        if "vision2d.code_routing" in self._definition.capabilities:
            self.route_guard.validate_move(current, target)

    def route_validate_tool_on(self, pose: Any) -> None:
        if "vision2d.code_routing" in self._definition.capabilities:
            self.route_guard.validate_tool_on(pose)

    def route_note_tool_off(self, pose: Any) -> bool:
        if "vision2d.code_routing" not in self._definition.capabilities:
            return True
        return self.route_guard.validate_tool_off(pose)

    def route_validate_home(self) -> None:
        if "vision2d.code_routing" in self._definition.capabilities:
            self.route_guard.validate_home()

    def reset_environment(self) -> dict[str, Any] | None:
        profile_controller = (
            self._profile_controller
            if _PROFILE_CAPABILITIES
            <= set(self._definition.capabilities)
            else None
        )
        result = None
        if profile_controller is not None:
            result = self._profile_public(
                profile_controller.reset(),
                path="vision.profile.reset",
            )
        if self._route_evidence is not None:
            self._route_completion_before_reset = self.route_guard.completion()
        self._ocr_evidence = None
        self.route_guard.reset()
        return result

    def collect_ocr_entry_probe(
        self,
        entry_id: str,
        *,
        run_id: str,
        tolerance_mm: float = 6.0,
    ) -> dict[str, Any]:
        """Read one bound scene part and return the guard's six-field proof."""

        context = self._ocr_evidence
        if not isinstance(context, Mapping):
            raise VisionPlatformError(
                "OCR_SORT_PLAN_NOT_ACTIVE", "OCR 分拣计划尚未激活"
            )
        plan = context.get("plan")
        entries = getattr(plan, "entries", ())
        entry = next((item for item in entries if getattr(item, "entry_id", None) == entry_id), None)
        if entry is None:
            raise VisionPlatformError(
                "OCR_SORT_ENTRY_INVALID", "entry_id 不在当前 OCR 分拣计划中"
            )
        if type(run_id) is not str or not run_id:
            raise VisionPlatformError("OCR_SORT_PROBE_INVALID", "run_id 无效")
        try:
            tolerance = float(tolerance_mm)
        except (TypeError, ValueError, OverflowError):
            raise VisionPlatformError("OCR_SORT_PROBE_INVALID", "探针容差无效") from None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise VisionPlatformError("OCR_SORT_PROBE_INVALID", "探针容差无效")

        route_entries = [item for item in entries if getattr(item, "route_id", None) == entry.route_id]
        ordered = sorted(
            route_entries,
            key=lambda item: tuple(float(value) for value in item.drop_xyz_mm),
        )
        slot_id = f"slot_{ordered.index(entry) + 1}"
        expected = tuple(float(value) for value in entry.drop_xyz_mm)
        sim = self.application.sim
        get_object = getattr(sim, "getObject", None)
        get_position = getattr(sim, "getObjectPosition", None)
        if not callable(get_object) or not callable(get_position):
            raise VisionPlatformError(
                "OCR_SORT_PROBE_FAILED", "CoppeliaSim 不支持只读条目探针"
            )
        path = f"/VisionOcrSortingLab/Parts/{entry.part_id}"
        try:
            handle = get_object(path)
            raw = get_position(handle, getattr(sim, "handle_world", -1))
            actual = tuple(float(value) * 1000.0 for value in raw)
            if len(actual) != 3 or not all(math.isfinite(value) for value in actual):
                raise ValueError("scene position is invalid")
            distance = math.dist(actual, expected)
        except VisionPlatformError:
            raise
        except Exception as error:
            raise VisionPlatformError(
                "OCR_SORT_PROBE_FAILED", "读取 OCR 条目场景位置失败",
                details={"error_type": type(error).__name__},
            ) from error
        if distance > tolerance:
            raise VisionPlatformError(
                "OCR_SORT_PROBE_FAILED", "OCR 条目未进入绑定仓位",
                details={"distance_mm": distance, "tolerance_mm": tolerance},
            )

        evidence_id = f"ocr-entry-{entry_id}-{len(getattr(self, '_ocr_probe_ids', [])) + 1:03d}"
        self._ocr_probe_ids = [*getattr(self, "_ocr_probe_ids", []), evidence_id]
        reference = {
            "run_id": run_id,
            "scene_hash": self.context.scene_sha256,
            "part_id": entry.part_id,
            "route_id": entry.route_id,
            "slot_id": slot_id,
            "evidence_id": evidence_id,
        }
        self.evidence.record_json_artifact(
            f"{evidence_id}.json",
            {
                **reference,
                "status": "PASS",
                "expected_xyz_mm": list(expected),
                "actual_xyz_mm": list(actual),
                "distance_mm": distance,
                "hardware_status": "PENDING_HARDWARE",
            },
        )
        return _copy_json_native(reference, path="ocr-sort-entry.probe")

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
