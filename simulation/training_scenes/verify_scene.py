from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Callable

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.training_scenes.scene_contract import validate_scene_contract
from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim
from vision_platform.coppeliasim_readiness import close_remote_client
from vision_platform.experiments.scene_setup import (
    LOGISTICS_GROUPS,
    activate_scene_group,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FORMAL_SCENES = {
    "robot-basics": {
        "directory": PROJECT_ROOT / "simulation" / "robot_basics",
        "scene": PROJECT_ROOT
        / "simulation"
        / "robot_basics"
        / "BL23_robot_basics.ttt",
    },
    "logistics-lab": {
        "directory": PROJECT_ROOT / "simulation" / "logistics_lab",
        "scene": PROJECT_ROOT
        / "simulation"
        / "logistics_lab"
        / "BL23_logistics_lab.ttt",
    },
}
_CAPTURE_NAMES = {
    "robot-basics": ("overview.png",),
    "logistics-lab": ("group-1.png", "group-2.png", "group-3.png"),
}


def _safe_exception_text(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException as formatting_error:
        error_type = f"{type(error).__module__}.{type(error).__qualname__}"
        formatting_type = (
            f"{type(formatting_error).__module__}."
            f"{type(formatting_error).__qualname__}"
        )
        return (
            f"<unprintable {error_type}; __str__ raised {formatting_type}>"
        )


class SceneCleanupError(RuntimeError):
    def __init__(self, failures: list[tuple[str, BaseException]]) -> None:
        self.failures = tuple(failures)
        details = "; ".join(
            f"{stage}: {type(error).__name__}: {_safe_exception_text(error)}"
            for stage, error in failures
        )
        super().__init__(f"training scene cleanup failed: {details}")


def _project_path(raw: str | Path) -> Path:
    selected = Path(raw).expanduser()
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    return selected.resolve()


def _required_file(raw: str | Path, label: str) -> Path:
    selected = _project_path(raw)
    if not selected.is_file():
        raise FileNotFoundError(f"{label} does not exist: {selected}")
    return selected


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _formal_scene_id_for_cleanup(
    scene_directory: str | Path | None,
    *,
    scene: str | Path | None,
    spec: str | Path | None,
    manifest: str | Path | None,
) -> str | None:
    # Secondary selectors are validated later; only an unambiguous primary
    # selector is safe enough to identify fixed evidence for early cleanup.
    if scene_directory is not None and spec is not None:
        return None
    try:
        selected_directory = (
            _project_path(scene_directory)
            if scene_directory is not None
            else None
        )
        selected_spec = _project_path(spec) if spec is not None else None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None

    for scene_id, formal in _FORMAL_SCENES.items():
        formal_directory = formal["directory"].resolve()
        formal_spec = (formal_directory / "scene_spec.json").resolve()
        if selected_directory is not None:
            if selected_directory == formal_directory:
                return scene_id
        elif selected_spec == formal_spec:
            return scene_id
    return None


def _resolve_inputs(
    scene_directory: str | Path | None,
    *,
    scene: str | Path | None,
    spec: str | Path | None,
    manifest: str | Path | None,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    if scene_directory is not None and spec is not None:
        raise ValueError("use scene_directory or spec, not both")
    if scene_directory is not None:
        directory = _project_path(scene_directory)
        if not directory.is_dir():
            raise NotADirectoryError(
                f"training scene directory does not exist: {directory}"
            )
        spec_path = _required_file(directory / "scene_spec.json", "scene spec")
        manifest_path = _required_file(
            directory / "scene_manifest.json",
            "scene manifest",
        )
    elif spec is not None:
        spec_path = _required_file(spec, "scene spec")
        directory = spec_path.parent
        manifest_path = _required_file(
            manifest or directory / "scene_manifest.json",
            "scene manifest",
        )
    else:
        raise ValueError("scene_directory or spec is required")

    spec_payload = _load_object(spec_path, "scene spec")
    declared_scene = _required_file(spec_payload.get("output", ""), "scene")
    scene_path = _required_file(scene or declared_scene, "scene")
    if scene_path != declared_scene:
        raise ValueError(
            "explicit scene does not match scene_spec output: "
            f"explicit={scene_path}, declared={declared_scene}"
        )
    contract = validate_scene_contract(
        spec_path,
        manifest_path,
        project_root=PROJECT_ROOT,
    )
    scene_id = spec_payload.get("scene_id")
    formal = _FORMAL_SCENES.get(scene_id)
    if formal is None:
        raise ValueError(f"unsupported formal training scene: {scene_id!r}")
    formal_directory = formal["directory"].resolve()
    formal_scene = formal["scene"].resolve()
    if (
        directory.resolve() != formal_directory
        or spec_path != (formal_directory / "scene_spec.json").resolve()
        or manifest_path != (formal_directory / "scene_manifest.json").resolve()
        or scene_path != formal_scene
    ):
        raise ValueError(
            "training scene inputs must use the checked-in formal scene, "
            "spec and manifest"
        )
    return scene_path, spec_path, manifest_path, spec_payload, contract


def _validate_endpoint(host: str, port: int) -> tuple[str, int]:
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(f"port must be an integer in 1..65535, got {port!r}")
    return host, port


def _wait_for_state(
    sim: Any,
    expected: int,
    *,
    timeout_s: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while (
        int(sim.getSimulationState()) != int(expected)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if int(sim.getSimulationState()) != int(expected):
        raise RuntimeError(
            f"CoppeliaSim state did not reach {expected} within {timeout_s}s"
        )


def _stop(sim: Any) -> None:
    if int(sim.getSimulationState()) == int(sim.simulation_stopped):
        return
    sim.stopSimulation()
    _wait_for_state(sim, int(sim.simulation_stopped))


def _start(sim: Any) -> None:
    sim.startSimulation()
    deadline = time.monotonic() + 10.0
    while (
        int(sim.getSimulationState()) == int(sim.simulation_stopped)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if int(sim.getSimulationState()) == int(sim.simulation_stopped):
        raise RuntimeError("CoppeliaSim simulation did not start")
    time.sleep(0.2)


def _cleanup_call(
    failures: list[tuple[str, BaseException]],
    stage: str,
    action: Callable[[], Any],
) -> None:
    try:
        action()
    except BaseException as error:
        failures.append((stage, error))


def _raise_with_cleanup(
    primary: tuple[BaseException, TracebackType | None] | None,
    failures: list[tuple[str, BaseException]],
) -> None:
    if primary is not None:
        error, traceback = primary
        existing = error.__cause__
        if failures:
            previous_cause = existing
            if isinstance(existing, SceneCleanupError):
                failures = [*existing.failures, *failures]
                previous_cause = existing.__cause__
            cleanup_error = SceneCleanupError(failures)
            if previous_cause is not None:
                cleanup_error.__cause__ = previous_cause
                cleanup_error.__suppress_context__ = True
            raise error.with_traceback(traceback) from cleanup_error
        raise error.with_traceback(traceback)
    if failures:
        cleanup_error = SceneCleanupError(failures)
        raise cleanup_error from failures[0][1]


def _activate_logistics_group(
    sim: Any,
    active_path: str,
) -> dict[str, list[float]]:
    activate_scene_group(sim, active_path=active_path)
    positions: dict[str, list[float]] = {}
    for path in LOGISTICS_GROUPS:
        handle = int(sim.getObject(path))
        raw = sim.getObjectPosition(handle, sim.handle_world)
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise RuntimeError(f"invalid logistics group position: {path}")
        position = [float(value) for value in raw]
        if not all(np.isfinite(value) for value in position):
            raise RuntimeError(f"non-finite logistics group position: {path}")
        positions[path] = position
    return positions


def _frame_summary(
    frame: Any,
    *,
    expected_resolution: tuple[int, int],
) -> dict[str, Any]:
    width = getattr(frame, "width", None)
    height = getattr(frame, "height", None)
    image = getattr(frame, "image_bgr", None)
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
    ):
        raise RuntimeError("camera frame has invalid dimensions")
    if (width, height) != expected_resolution:
        raise RuntimeError(
            "camera frame resolution mismatch: "
            f"expected={list(expected_resolution)}, actual={[width, height]}"
        )
    if (
        type(image) is not np.ndarray
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape != (height, width, 3)
        or image.size == 0
        or image.nbytes == 0
    ):
        raise RuntimeError("camera frame has invalid image data")
    if not np.isfinite(image).all():
        raise RuntimeError("camera frame contains non-finite values")
    channel_minimums = image.min(axis=(0, 1))
    channel_maximums = image.max(axis=(0, 1))
    channel_spatial_ranges = (
        channel_maximums.astype(np.int16)
        - channel_minimums.astype(np.int16)
    )
    spatial_dynamic_range = int(channel_spatial_ranges.max())
    if spatial_dynamic_range <= 0:
        raise RuntimeError("camera frame is blank (no spatial variation)")
    minimum = int(channel_minimums.min())
    maximum = int(channel_maximums.max())
    dynamic_range = maximum - minimum
    return {
        "non_blank": True,
        "pixel_min": minimum,
        "pixel_max": maximum,
        "dynamic_range": dynamic_range,
        "spatial_dynamic_range": spatial_dynamic_range,
        "channel_spatial_ranges": [
            int(value) for value in channel_spatial_ranges
        ],
        "mean": round(float(image.mean()), 6),
    }


def _capture(
    *,
    sim: Any,
    client: Any,
    sensor_path: str,
    expected_resolution: tuple[int, int],
    output: Path,
) -> tuple[list[int], dict[str, Any]]:
    sensor_handle: int | None = None
    explicit_attempted = False
    camera: Any | None = None
    primary: tuple[BaseException, TracebackType | None] | None = None
    result: tuple[list[int], dict[str, Any]] | None = None
    try:
        sensor_handle = int(sim.getObject(sensor_path))
        explicit_attempted = True
        sim.setExplicitHandling(sensor_handle, 1)
        sim.handleVisionSensor(sensor_handle)
        camera = CoppeliaSimCamera(
            sim=sim,
            client=client,
            sensor_path=sensor_path,
        )
        camera.open()
        frame = camera.read(timeout_s=5.0)
        summary = _frame_summary(
            frame,
            expected_resolution=expected_resolution,
        )
        if not cv2.imwrite(str(output), frame.image_bgr):
            raise RuntimeError(f"Could not write {output}")
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"Camera evidence is empty: {output}")
        result = ([int(frame.width), int(frame.height)], summary)
    except BaseException as error:
        primary = (error, error.__traceback__)

    cleanup_failures: list[tuple[str, BaseException]] = []
    if camera is not None:
        _cleanup_call(cleanup_failures, "camera.close", camera.close)
    if explicit_attempted and sensor_handle is not None:
        _cleanup_call(
            cleanup_failures,
            "vision_sensor.explicit_handling_reset",
            lambda: sim.setExplicitHandling(sensor_handle, 0),
        )
    _raise_with_cleanup(primary, cleanup_failures)
    assert result is not None
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _evidence_paths(report_path: Path, scene_id: str) -> tuple[Path, tuple[Path, ...]]:
    evidence_dir = report_path.parent / f"{scene_id}-frames"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    resolved_parent = report_path.parent.resolve()
    resolved_evidence = evidence_dir.resolve()
    if resolved_evidence.parent != resolved_parent:
        raise ValueError("training scene evidence directory escaped report directory")
    paths = tuple(resolved_evidence / name for name in _CAPTURE_NAMES[scene_id])
    return resolved_evidence, paths


def _remove_paths(
    paths: tuple[Path, ...],
    failures: list[tuple[str, BaseException]] | None = None,
) -> None:
    for path in paths:
        if failures is None:
            path.unlink(missing_ok=True)
        else:
            _cleanup_call(
                failures,
                f"evidence.remove:{path.name}",
                lambda selected=path: selected.unlink(missing_ok=True),
            )


def verify_training_scene(
    scene_directory: str | Path | None = None,
    *,
    host: str,
    port: int,
    output: str | Path,
    scene: str | Path | None = None,
    spec: str | Path | None = None,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    report_path = Path(output).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.unlink(missing_ok=True)
    cleanup_scene_id = _formal_scene_id_for_cleanup(
        scene_directory,
        scene=scene,
        spec=spec,
        manifest=manifest,
    )
    if cleanup_scene_id is not None:
        _, stale_evidence = _evidence_paths(
            report_path,
            cleanup_scene_id,
        )
        _remove_paths(stale_evidence)
    selected_host, selected_port = _validate_endpoint(host, port)
    (
        scene_path,
        spec_path,
        manifest_path,
        spec_payload,
        contract,
    ) = _resolve_inputs(
        scene_directory,
        scene=scene,
        spec=spec,
        manifest=manifest,
    )
    scene_id = str(spec_payload["scene_id"])
    _, expected_evidence = _evidence_paths(report_path, scene_id)
    _remove_paths(expected_evidence)
    expected_resolution = tuple(
        int(value) for value in spec_payload["camera"]["resolution"]
    )
    if len(expected_resolution) != 2 or any(
        value <= 0 for value in expected_resolution
    ):
        raise ValueError("scene camera resolution must contain two positive integers")

    client: Any | None = None
    sim: Any | None = None
    primary: tuple[BaseException, TracebackType | None] | None = None
    captures: list[dict[str, Any]] = []
    handles: dict[str, int] = {}
    reload_handles: dict[str, int] = {}
    reload_verified = False
    staged_scene: Path | None = None
    try:
        client = RemoteAPIClient(host=selected_host, port=selected_port)
        sim = client.require("sim")
        staged_scene = stage_scene_for_coppeliasim(scene_path)
        _stop(sim)
        sim.loadScene(staged_scene.as_posix())
        handles = {
            path: int(sim.getObject(path))
            for path in spec_payload["required_paths"]
        }
        groups: tuple[str | None, ...] = (
            tuple(LOGISTICS_GROUPS)
            if scene_id == "logistics-lab"
            else (None,)
        )
        for index, group in enumerate(groups, start=1):
            _start(sim)
            positions = (
                _activate_logistics_group(sim, group)
                if group is not None
                else {}
            )
            if group is not None:
                time.sleep(0.2)
            capture_path = expected_evidence[index - 1]
            size, frame_summary = _capture(
                sim=sim,
                client=client,
                sensor_path=spec_payload["camera"]["path"],
                expected_resolution=expected_resolution,
                output=capture_path,
            )
            captures.append(
                {
                    "group": group,
                    "group_positions_m": positions,
                    "path": capture_path.relative_to(
                        report_path.parent
                    ).as_posix(),
                    "size": size,
                    "sha256": _sha256(capture_path),
                    "frame_summary": frame_summary,
                }
            )
            _stop(sim)

        sim.loadScene(staged_scene.as_posix())
        reload_handles = {
            path: int(sim.getObject(path))
            for path in spec_payload["required_paths"]
        }
        _start(sim)
        _stop(sim)
        reload_verified = True
    except BaseException as error:
        primary = (error, error.__traceback__)

    cleanup_failures: list[tuple[str, BaseException]] = []
    if sim is not None:
        _cleanup_call(cleanup_failures, "simulation.stop", lambda: _stop(sim))
    if client is not None:
        _cleanup_call(
            cleanup_failures,
            "remote_client.close",
            lambda: close_remote_client(client),
        )
    if primary is not None or cleanup_failures:
        _remove_paths(expected_evidence, cleanup_failures)
    _raise_with_cleanup(primary, cleanup_failures)

    assert staged_scene is not None
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scene_id": scene_id,
        "scene": str(scene_path),
        "loaded_scene": str(staged_scene),
        "spec": str(spec_path),
        "manifest": str(manifest_path),
        "endpoint": {"host": selected_host, "port": selected_port},
        "contract": contract,
        "required_paths": handles,
        "captures": captures,
        "reload_required_paths": reload_handles,
        "reload_verified": reload_verified,
        "hardware_status": "PENDING_HARDWARE",
    }
    try:
        _atomic_write_json(report_path, report)
    except BaseException as error:
        report_path.unlink(missing_ok=True)
        evidence_cleanup: list[tuple[str, BaseException]] = []
        _remove_paths(expected_evidence, evidence_cleanup)
        _raise_with_cleanup((error, error.__traceback__), evidence_cleanup)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify one checked-in formal training scene in CoppeliaSim",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene-directory", type=Path)
    source.add_argument("--spec", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", "--output", dest="report", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.spec is not None and args.scene is None:
        parser.error("--scene is required when --spec is used")
    report = verify_training_scene(
        args.scene_directory,
        scene=args.scene,
        spec=args.spec,
        manifest=args.manifest,
        host=args.host,
        port=args.port,
        output=args.report,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
