from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "vision_lab.default.json"
DEFAULT_STUDENT_CONFIG: Mapping[str, Any] = {
    "allow_real_backend": False,
    "max_runtime_s": 60,
    "max_commands": 200,
    "command_timeout_s": 10,
    "max_sleep_s": 5,
    "speed_range": [1, 30],
    "tool_on_max_z_mm": 35,
    "output": "artifacts/vision_lab/student-runs",
}


@dataclass(frozen=True)
class WorkspaceConfig:
    x_mm: tuple[float, float]
    y_mm: tuple[float, float]
    z_mm: tuple[float, float]
    safe_z_mm: float

    def __post_init__(self) -> None:
        for name, bounds in (
            ("x_mm", self.x_mm),
            ("y_mm", self.y_mm),
            ("z_mm", self.z_mm),
        ):
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must be [minimum, maximum]")
        if not self.z_mm[0] <= self.safe_z_mm <= self.z_mm[1]:
            raise ValueError("safe_z_mm must be inside z_mm bounds")


@dataclass(frozen=True)
class VisionLabConfig:
    camera_backend: str
    robot_backend: str
    coppeliasim_root: Path
    coppelia_host: str
    coppelia_port: int
    coppelia_scene: Path
    camera_options: Mapping[str, Any]
    workspace: WorkspaceConfig
    calibration: Mapping[str, Any]
    recognition: Mapping[str, Any]
    task: Mapping[str, Any]
    student: Mapping[str, Any]
    ui: Mapping[str, Any]
    config_path: Path
    project_root: Path


def _absolute(value: str | os.PathLike[str], root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _resolve_camera_paths(camera_options: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = json.loads(json.dumps(camera_options))
    for backend_options in resolved.values():
        if not isinstance(backend_options, dict):
            continue
        for key, value in tuple(backend_options.items()):
            if key in {"manifest", "path", "video"} and isinstance(value, str):
                backend_options[key] = str(_absolute(value, root))
            elif key == "paths" and isinstance(value, list):
                backend_options[key] = [
                    str(_absolute(item, root)) for item in value
                ]
    return resolved


def _student_options(payload: Any) -> dict[str, Any]:
    options = deepcopy(dict(DEFAULT_STUDENT_CONFIG))
    options.update(deepcopy(dict(payload)))
    return options


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    project_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> VisionLabConfig:
    env = os.environ if environ is None else environ
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    selected_path = _absolute(
        config_path or env.get("VISION_LAB_CONFIG", DEFAULT_CONFIG_PATH),
        root,
    )
    payload = json.loads(selected_path.read_text(encoding="utf-8"))

    camera_backend = env.get(
        "VISION_BACKEND", payload.get("camera_backend", "replay")
    ).strip().lower()
    robot_backend = env.get(
        "ROBOT_BACKEND", payload.get("robot_backend", "sim")
    ).strip().lower()
    if camera_backend not in {"sim", "replay", "hik"}:
        raise ValueError(f"Unsupported camera backend: {camera_backend}")
    if robot_backend not in {"sim", "real"}:
        raise ValueError(f"Unsupported robot backend: {robot_backend}")

    coppelia = dict(payload.get("coppeliasim", {}))
    coppeliasim_root = _absolute(
        env.get("COPPELIASIM_ROOT", coppelia.get("root", "E:/CoppeliaSim")),
        root,
    )
    coppelia_scene = _absolute(
        env.get(
            "COPPELIA_SCENE",
            coppelia.get("scene", "simulation/vision_lab/BL23_vision_lab.ttt"),
        ),
        root,
    )
    coppelia_host = env.get("COPPELIA_HOST", coppelia.get("host", "127.0.0.1"))
    coppelia_port = int(env.get("COPPELIA_PORT", coppelia.get("port", 23000)))

    workspace_payload = payload["workspace"]
    workspace = WorkspaceConfig(
        x_mm=tuple(float(value) for value in workspace_payload["x_mm"]),
        y_mm=tuple(float(value) for value in workspace_payload["y_mm"]),
        z_mm=tuple(float(value) for value in workspace_payload["z_mm"]),
        safe_z_mm=float(workspace_payload["safe_z_mm"]),
    )

    return VisionLabConfig(
        camera_backend=camera_backend,
        robot_backend=robot_backend,
        coppeliasim_root=coppeliasim_root,
        coppelia_host=str(coppelia_host),
        coppelia_port=coppelia_port,
        coppelia_scene=coppelia_scene,
        camera_options=_resolve_camera_paths(payload.get("camera", {}), root),
        workspace=workspace,
        calibration=dict(payload.get("calibration", {})),
        recognition=dict(payload.get("recognition", {})),
        task=dict(payload.get("task", {})),
        student=_student_options(payload.get("student", {})),
        ui=dict(payload.get("ui", {})),
        config_path=selected_path,
        project_root=root,
    )
