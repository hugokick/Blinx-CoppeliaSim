from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from vision_platform.config import VisionLabConfig, WorkspaceConfig
from vision_platform.experiments.capabilities import check_capabilities
from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.experiments.models import (
    ExperimentDefinition,
    ExperimentRunContext,
)
from vision_platform.session import VisionLabSession


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_public_path(definition: ExperimentDefinition, name: str) -> str:
    value = definition.public_parameters.get(name)
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(
            f"{definition.experiment_id} public_parameters.{name} "
            "must be an absolute CoppeliaSim object path"
        )
    return value


def _expected_scene_hash(
    definition: ExperimentDefinition,
    project_root: Path,
) -> str:
    payload = json.loads(definition.scene_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("scene manifest schema_version must be 1")
    if not isinstance(payload.get("scene"), dict):
        raise ValueError("scene manifest must contain a scene object")
    scene = payload["scene"]
    raw_path = scene.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("scene manifest scene.path must be a non-empty path")
    root = project_root.expanduser().resolve()
    declared_path = Path(raw_path).expanduser()
    if not declared_path.is_absolute():
        declared_path = root / declared_path
    declared_path = declared_path.resolve()
    if declared_path != root and root not in declared_path.parents:
        raise ValueError("scene manifest scene.path must stay inside project")
    if declared_path != definition.scene.resolve():
        raise ValueError(
            "scene manifest scene.path does not match experiment scene"
        )

    expected = scene.get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("scene manifest scene.sha256 must be lowercase SHA-256")
    return expected


class ExperimentSession:
    """Select isolated experiment scenes through the shared V2.1 session."""

    def __init__(
        self,
        *,
        catalog: ExperimentCatalog,
        vision_session: VisionLabSession,
        base_config: VisionLabConfig,
        application_factory: Callable[[VisionLabConfig], Any],
        student_is_idle: Callable[[], bool],
    ) -> None:
        if not isinstance(base_config, VisionLabConfig):
            raise TypeError("base_config must be VisionLabConfig")
        self.catalog = catalog
        self.vision_session = vision_session
        self.base_config = base_config
        self.application_factory = application_factory
        self.student_is_idle = student_is_idle
        self.current: ExperimentRunContext | None = None
        self._lock = RLock()

    def select(self, experiment_id: str) -> ExperimentRunContext:
        with self._lock:
            return self._select_locked(experiment_id)

    def _select_locked(self, experiment_id: str) -> ExperimentRunContext:
        if not self.student_is_idle():
            raise RuntimeError("学生程序运行、暂停或清理期间不能切换实验")

        definition = self.catalog.require(experiment_id)
        config = self._config_for(definition)
        validated_context: ExperimentRunContext | None = None
        validated_application: Any | None = None

        def validate(application: Any) -> None:
            nonlocal validated_application, validated_context
            report = check_capabilities(application, definition.capabilities)
            if not report.ready:
                reasons = "; ".join(
                    f"{name}: {report.reasons[name]}" for name in report.missing
                )
                raise RuntimeError(f"实验能力不可用：{reasons}")

            actual_hash = _sha256(definition.scene)
            expected_hash = _expected_scene_hash(
                definition,
                self.base_config.project_root,
            )
            if actual_hash != expected_hash:
                raise RuntimeError(
                    "场景哈希不匹配："
                    f"expected={expected_hash}, actual={actual_hash}"
                )
            validated_context = ExperimentRunContext(
                experiment_id=definition.experiment_id,
                experiment_version=definition.version,
                scene_path=definition.scene,
                scene_sha256=actual_hash,
                scene_manifest_path=definition.scene_manifest,
                public_parameters=definition.public_parameters,
                hardware_status=definition.hardware_status,
            )
            validated_application = application

        self.current = None
        try:
            self.vision_session.replace_application(
                lambda: self.application_factory(config),
                scene_path=definition.scene,
                validate=validate,
            )
        except BaseException:
            if (
                validated_context is not None
                and self.vision_session.application is validated_application
            ):
                self.current = validated_context
            raise
        if validated_context is None:  # pragma: no cover - defensive contract
            raise RuntimeError("实验场景未完成验证")
        self.current = validated_context
        return validated_context

    def _config_for(self, definition: ExperimentDefinition) -> VisionLabConfig:
        workspace = WorkspaceConfig(
            x_mm=tuple(float(value) for value in definition.workspace["x_mm"]),
            y_mm=tuple(float(value) for value in definition.workspace["y_mm"]),
            z_mm=tuple(float(value) for value in definition.workspace["z_mm"]),
            safe_z_mm=float(definition.workspace["safe_z_mm"]),
        )
        camera_options = deepcopy(dict(self.base_config.camera_options))
        sim_camera = deepcopy(dict(camera_options.get("sim", {})))
        sim_camera["sensor_path"] = _required_public_path(
            definition,
            "camera_path",
        )
        camera_options["sim"] = sim_camera

        task_options = deepcopy(dict(self.base_config.task))
        if "pickables_path" in definition.public_parameters:
            task_options["pickables_path"] = _required_public_path(
                definition,
                "pickables_path",
            )

        return replace(
            self.base_config,
            camera_backend="sim",
            robot_backend="sim",
            coppelia_scene=definition.scene,
            camera_options=camera_options,
            task=task_options,
            workspace=workspace,
        )
