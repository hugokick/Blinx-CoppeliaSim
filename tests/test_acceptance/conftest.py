from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_SOURCE = PROJECT_ROOT / "simulation" / "vision_lab" / "BL23_vision_lab.ttt"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "coppeliasim: requires a live CoppeliaSim process and the BL23 vision scene",
    )


def pytest_collection_modifyitems(config, items):
    marker_expression = config.getoption("-m") or ""
    if "coppeliasim" in marker_expression:
        return
    skip = pytest.mark.skip(
        reason=(
            "live CoppeliaSim acceptance is opt-in; rerun with "
            "'-m coppeliasim' (a skip is not an acceptance PASS)"
        )
    )
    for item in items:
        if item.get_closest_marker("coppeliasim"):
            item.add_marker(skip)


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_is_open(host, port):
            return
        time.sleep(0.1)
    pytest.fail(f"CoppeliaSim ZMQ port did not open at {host}:{port}")


def _wait_for_stopped(sim, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while (
        int(sim.getSimulationState()) != int(sim.simulation_stopped)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if int(sim.getSimulationState()) != int(sim.simulation_stopped):
        pytest.fail("CoppeliaSim did not stop before loading the acceptance scene")


@pytest.fixture
def running_vision_scene(tmp_path, request):
    if "coppeliasim" not in (request.config.getoption("-m") or ""):
        pytest.skip(
            "live CoppeliaSim acceptance requires '-m coppeliasim'; "
            "this skip is not a PASS"
        )

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    from vision_platform.application import VisionLabApplication
    from vision_platform.config import load_config

    host = "127.0.0.1"
    port = 23000
    scene_copy = tmp_path / "BL23_vision_lab_acceptance.ttt"
    shutil.copy2(SCENE_SOURCE, scene_copy)
    process = None
    owns_process = not _port_is_open(host, port)
    if owns_process:
        executable = Path("E:/CoppeliaSim/coppeliaSim.exe")
        if not executable.is_file():
            pytest.fail(f"CoppeliaSim executable is missing: {executable}")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [
                str(executable),
                "-h",
                f"-GzmqRemoteApi.rpcPort={port}",
                str(scene_copy),
            ],
            cwd=str(executable.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        _wait_for_port(host, port)

    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    if int(sim.getSimulationState()) != int(sim.simulation_stopped):
        sim.stopSimulation()
        _wait_for_stopped(sim)
    sim.loadScene(scene_copy.as_posix())
    sim.startSimulation()
    deadline = time.monotonic() + 5.0
    while (
        int(sim.getSimulationState()) == int(sim.simulation_stopped)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if int(sim.getSimulationState()) == int(sim.simulation_stopped):
        pytest.fail("CoppeliaSim acceptance scene did not start")
    time.sleep(0.25)

    environ = dict(os.environ)
    environ.update(
        {
            "VISION_BACKEND": "sim",
            "ROBOT_BACKEND": "sim",
            "COPPELIA_HOST": host,
            "COPPELIA_PORT": str(port),
            "COPPELIA_SCENE": str(scene_copy),
        }
    )
    config = load_config(project_root=PROJECT_ROOT, environ=environ)
    application = VisionLabApplication.from_config(
        config,
        sim=sim,
        client=client,
    )
    try:
        yield application
    finally:
        application.close()
        try:
            if int(sim.getSimulationState()) != int(sim.simulation_stopped):
                sim.stopSimulation()
                _wait_for_stopped(sim)
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
