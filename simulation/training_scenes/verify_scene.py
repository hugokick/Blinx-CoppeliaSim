from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.training_scenes.scene_contract import validate_scene_contract
from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGISTICS_GROUPS = (
    "/LogisticsLab/Tasks/Stack",
    "/LogisticsLab/Tasks/Digits",
    "/LogisticsLab/Tasks/Classes",
)


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


def _activate_logistics_group(
    sim: Any,
    active_path: str,
) -> dict[str, list[float]]:
    if active_path not in LOGISTICS_GROUPS:
        raise ValueError(f"unknown logistics group: {active_path}")
    positions = {}
    for index, path in enumerate(LOGISTICS_GROUPS):
        handle = int(sim.getObject(path))
        z_m = 0.0 if path == active_path else -float(index + 3)
        sim.setObjectPosition(
            handle,
            [0.0, 0.0, z_m],
            sim.handle_world,
        )
        positions[path] = [
            float(value)
            for value in sim.getObjectPosition(handle, sim.handle_world)
        ]
    return positions


def _capture(
    *,
    sim: Any,
    client: Any,
    sensor_path: str,
    output: Path,
) -> list[int]:
    sensor_handle = int(sim.getObject(sensor_path))
    sim.setExplicitHandling(sensor_handle, 1)
    sim.handleVisionSensor(sensor_handle)
    camera = CoppeliaSimCamera(
        sim=sim,
        client=client,
        sensor_path=sensor_path,
    )
    camera.open()
    try:
        frame = camera.read(timeout_s=5.0)
    finally:
        camera.close()
        sim.setExplicitHandling(sensor_handle, 0)
    if not cv2.imwrite(str(output), frame.image_bgr):
        raise RuntimeError(f"Could not write {output}")
    return [int(frame.width), int(frame.height)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_training_scene(
    scene_directory: str | Path,
    *,
    host: str,
    port: int,
    output: str | Path,
) -> dict[str, Any]:
    directory = Path(scene_directory).expanduser().resolve()
    spec = json.loads(
        (directory / "scene_spec.json").read_text(encoding="utf-8")
    )
    contract = validate_scene_contract(
        directory / "scene_spec.json",
        directory / "scene_manifest.json",
        project_root=PROJECT_ROOT,
    )
    report_path = Path(output).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir = report_path.parent / f"{spec['scene_id']}-frames"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    scene = stage_scene_for_coppeliasim(PROJECT_ROOT / spec["output"])
    _stop(sim)
    sim.loadScene(scene.as_posix())
    handles = {
        path: int(sim.getObject(path)) for path in spec["required_paths"]
    }
    captures = []
    groups = (
        LOGISTICS_GROUPS
        if spec["scene_id"] == "logistics-lab"
        else (None,)
    )
    try:
        for index, group in enumerate(groups, start=1):
            _start(sim)
            positions = (
                _activate_logistics_group(sim, group)
                if group is not None
                else {}
            )
            if group is not None:
                time.sleep(0.2)
            name = (
                f"group-{index}.png"
                if group is not None
                else "overview.png"
            )
            path = evidence_dir / name
            size = _capture(
                sim=sim,
                client=client,
                sensor_path=spec["camera"]["path"],
                output=path,
            )
            captures.append(
                {
                    "group": group,
                    "group_positions_m": positions,
                    "path": path.relative_to(report_path.parent).as_posix(),
                    "size": size,
                    "sha256": _sha256(path),
                }
            )
            _stop(sim)

        sim.loadScene(scene.as_posix())
        for path in spec["required_paths"]:
            sim.getObject(path)
        _start(sim)
        _stop(sim)
        reload_verified = True
    finally:
        _stop(sim)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scene_id": spec["scene_id"],
        "contract": contract,
        "required_paths": handles,
        "captures": captures,
        "reload_verified": reload_verified,
        "hardware_status": "PENDING_HARDWARE",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify_training_scene(
        args.scene_directory,
        host=args.host,
        port=args.port,
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
