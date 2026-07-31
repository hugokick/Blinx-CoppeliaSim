from __future__ import annotations

import os
import shutil
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from vision_platform.coppeliasim_readiness import (
    close_remote_client,
    require_expected_scene,
    wait_for_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_SOURCE = PROJECT_ROOT / "simulation" / "vision_lab" / "BL23_vision_lab.ttt"
LAUNCH_COMMAND = (
    "powershell -ExecutionPolicy Bypass "
    "-File tools\\vision_lab\\launch_coppeliasim.ps1"
)


def pytest_addoption(parser):
    group = parser.getgroup("coppeliasim")
    group.addoption(
        "--coppelia-host",
        action="store",
        default=None,
        help=(
            "CoppeliaSim ZMQ Remote API host "
            "(default: COPPELIA_HOST or 127.0.0.1)"
        ),
    )
    group.addoption(
        "--coppelia-port",
        action="store",
        type=int,
        default=None,
        help=(
            "CoppeliaSim ZMQ Remote API port "
            "(default: COPPELIA_PORT or 23000)"
        ),
    )


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


def _resolve_coppelia_endpoint(config, *, environ=None) -> tuple[str, int]:
    selected_environ = os.environ if environ is None else environ
    host = (
        config.getoption("--coppelia-host")
        or selected_environ.get("COPPELIA_HOST")
        or "127.0.0.1"
    )
    configured_port = config.getoption("--coppelia-port")
    raw_port = (
        configured_port
        if configured_port is not None
        else selected_environ.get("COPPELIA_PORT", "23000")
    )
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as error:
        raise pytest.UsageError(
            f"invalid CoppeliaSim port: {raw_port!r}"
        ) from error
    if not 1 <= port <= 65535:
        raise pytest.UsageError(
            f"CoppeliaSim port must be in 1..65535, got {port}"
        )
    return str(host), port


def _wait_for_remote_api(
    client_factory,
    *,
    host: str,
    port: int,
    timeout_s: float = 30.0,
    retry_interval_s: float = 0.1,
    state_predicate=None,
    state_description: str = "usable simulation state",
    ready_action=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Wait for a usable sim RPC, replacing a failed REQ client each try."""
    deadline = clock() + timeout_s
    last_error: Exception | None = None
    while True:
        client = None
        try:
            client = client_factory(host=host, port=port)
            sim = client.require("sim")
            state = int(sim.getSimulationState())
            if (
                state_predicate is not None
                and not state_predicate(state, sim)
            ):
                raise RuntimeError(
                    f"CoppeliaSim state {state} is not "
                    f"{state_description}"
                )
            if ready_action is not None:
                ready_action(state, sim)
            return client, sim
        except Exception as error:
            last_error = error
            if client is not None:
                close_remote_client(client)

        remaining_s = deadline - clock()
        if remaining_s <= 0:
            raise TimeoutError(
                "CoppeliaSim Remote API was not usable at "
                f"{host}:{port} within {timeout_s:.3f}s; "
                f"last RPC error: {last_error}"
            ) from last_error
        sleep(min(retry_interval_s, remaining_s))


def _prepare_running_scene(
    client_factory,
    *,
    host: str,
    port: int,
    scene_path: Path,
    timeout_s: float = 30.0,
    retry_interval_s: float = 0.1,
    clock=time.monotonic,
    sleep=time.sleep,
):
    def start_expected_scene(state, sim) -> None:
        require_expected_scene(sim, scene_path)
        sim.startSimulation()

    setup_client = None
    try:
        setup_client, setup_sim = _wait_for_remote_api(
            client_factory,
            host=host,
            port=port,
            timeout_s=timeout_s,
            retry_interval_s=retry_interval_s,
            clock=clock,
            sleep=sleep,
        )
        if (
            int(setup_sim.getSimulationState())
            != int(setup_sim.simulation_stopped)
        ):
            setup_sim.stopSimulation()
            _wait_for_stopped(setup_sim)
        setup_sim.loadScene(Path(scene_path).as_posix())
    finally:
        if setup_client is not None:
            close_remote_client(setup_client)

    start_client = None
    try:
        start_client, _ = _wait_for_remote_api(
            client_factory,
            host=host,
            port=port,
            timeout_s=timeout_s,
            retry_interval_s=retry_interval_s,
            state_predicate=lambda state, sim: (
                state == int(sim.simulation_stopped)
            ),
            state_description="expected scene stopped",
            ready_action=start_expected_scene,
            clock=clock,
            sleep=sleep,
        )
    finally:
        if start_client is not None:
            close_remote_client(start_client)

    return _wait_for_remote_api(
        client_factory,
        host=host,
        port=port,
        timeout_s=timeout_s,
        retry_interval_s=retry_interval_s,
        state_predicate=lambda state, sim: (
            state != int(sim.simulation_stopped)
        ),
        state_description="expected scene running",
        ready_action=lambda state, sim: require_expected_scene(
            sim,
            scene_path,
        ),
        clock=clock,
        sleep=sleep,
    )


def _stop_simulation_with_fresh_clients(
    client_factory,
    *,
    host: str,
    port: int,
    restore_scene_path: Path | None = None,
    timeout_s: float = 10.0,
    retry_interval_s: float = 0.1,
    clock=time.monotonic,
    sleep=time.sleep,
) -> None:
    deadline = clock() + timeout_s
    stop_client = None
    verification_client = None
    try:
        try:
            stop_client, _ = _wait_for_remote_api(
                client_factory,
                host=host,
                port=port,
                timeout_s=timeout_s,
                retry_interval_s=retry_interval_s,
                ready_action=lambda state, sim: (
                    sim.stopSimulation()
                    if state != int(sim.simulation_stopped)
                    else None
                ),
                clock=clock,
                sleep=sleep,
            )
        finally:
            if stop_client is not None:
                close_remote_client(stop_client)

        remaining_s = deadline - clock()
        if remaining_s <= 0:
            raise TimeoutError(
                "CoppeliaSim cleanup timed out before stop verification"
            )
        try:
            verification_client, _ = _wait_for_remote_api(
                client_factory,
                host=host,
                port=port,
                timeout_s=remaining_s,
                retry_interval_s=retry_interval_s,
                state_predicate=lambda state, sim: (
                    state == int(sim.simulation_stopped)
                ),
                state_description="simulation stopped",
                ready_action=(
                    (
                        lambda state, sim: sim.loadScene(
                            restore_scene_path.as_posix()
                        )
                    )
                    if restore_scene_path is not None
                    else None
                ),
                clock=clock,
                sleep=sleep,
            )
        finally:
            if verification_client is not None:
                close_remote_client(verification_client)
    except Exception as error:
        raise TimeoutError(
            "CoppeliaSim simulation cleanup did not complete at "
            f"{host}:{port} within {timeout_s:.3f}s; "
            f"last RPC error: {error}"
        ) from error


@contextmanager
def _running_scene_connection(
    client_factory,
    *,
    host: str,
    port: int,
    scene_path: Path,
    restore_scene_path: Path | None = None,
    readiness_timeout_s: float = 30.0,
    cleanup_timeout_s: float = 10.0,
    retry_interval_s: float = 0.1,
    clock=time.monotonic,
    sleep=time.sleep,
):
    client = None

    def release_connection_and_scene() -> None:
        close_error = None
        if client is not None:
            try:
                close_remote_client(client)
            except BaseException as error:
                close_error = error
        try:
            _stop_simulation_with_fresh_clients(
                client_factory,
                host=host,
                port=port,
                restore_scene_path=restore_scene_path,
                timeout_s=cleanup_timeout_s,
                retry_interval_s=retry_interval_s,
                clock=clock,
                sleep=sleep,
            )
        except BaseException as stop_error:
            if close_error is not None:
                raise close_error from stop_error
            raise
        if close_error is not None:
            raise close_error

    try:
        client, sim = _prepare_running_scene(
            client_factory,
            host=host,
            port=port,
            scene_path=scene_path,
            timeout_s=readiness_timeout_s,
            retry_interval_s=retry_interval_s,
            clock=clock,
            sleep=sleep,
        )
    except BaseException as preparation_error:
        try:
            release_connection_and_scene()
        except BaseException as cleanup_error:
            raise preparation_error from cleanup_error
        raise

    try:
        yield client, sim
    except BaseException as body_error:
        try:
            release_connection_and_scene()
        except BaseException as cleanup_error:
            raise body_error from cleanup_error
        raise
    else:
        release_connection_and_scene()


def _wait_for_stopped(sim, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while (
        int(sim.getSimulationState()) != int(sim.simulation_stopped)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if int(sim.getSimulationState()) != int(sim.simulation_stopped):
        pytest.fail("CoppeliaSim did not stop before loading the acceptance scene")


def _require_prelaunched_scene(
    client_factory,
    *,
    host: str,
    port: int,
    scene_path: Path,
    readiness_timeout_s: float = 10.0,
    port_probe=None,
    wait_for_scene_fn=None,
) -> None:
    probe = port_probe or _port_is_open
    readiness = wait_for_scene_fn or wait_for_scene
    if not probe(host, port):
        pytest.fail(
            f"No CoppeliaSim listener is available at {host}:{port}. "
            f"Start it first with: {LAUNCH_COMMAND}"
        )

    client = None
    try:
        client, _ = readiness(
            client_factory,
            host=host,
            port=port,
            scene_path=scene_path,
            timeout_s=readiness_timeout_s,
        )
    except Exception as error:
        raise pytest.fail.Exception(
            f"CoppeliaSim listener at {host}:{port} is not ready: {error}. "
            f"Start or reload the BL23 scene with: {LAUNCH_COMMAND}"
        ) from error
    finally:
        if client is not None:
            close_remote_client(client)


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

    host, port = _resolve_coppelia_endpoint(request.config)
    _require_prelaunched_scene(
        RemoteAPIClient,
        host=host,
        port=port,
        scene_path=SCENE_SOURCE,
    )

    scene_copy = tmp_path / "BL23_vision_lab_acceptance.ttt"
    shutil.copy2(SCENE_SOURCE, scene_copy)
    with _running_scene_connection(
        RemoteAPIClient,
        host=host,
        port=port,
        scene_path=scene_copy,
        restore_scene_path=SCENE_SOURCE,
    ) as (client, sim):
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
            application.close_quarantined()
