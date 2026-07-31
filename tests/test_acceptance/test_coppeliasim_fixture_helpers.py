from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_acceptance import conftest as acceptance_conftest


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class SocketProbe:
    def __init__(self) -> None:
        self.close_calls: list[int | None] = []

    def close(self, linger=None) -> None:
        self.close_calls.append(linger)


class ContextProbe:
    def __init__(self) -> None:
        self.term_calls = 0

    def term(self) -> None:
        self.term_calls += 1


class SimProbe:
    simulation_stopped = 0

    def __init__(self, outcome) -> None:
        self.outcome = outcome

    def getSimulationState(self) -> int:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return int(self.outcome)


class ClientProbe:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.socket = SocketProbe()
        self.context = ContextProbe()
        self.require_calls: list[str] = []

    def require(self, name: str):
        self.require_calls.append(name)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if isinstance(self.outcome, tuple):
            stage, state_outcome = self.outcome
            assert stage == "state"
            return SimProbe(state_outcome)
        return SimProbe(self.outcome)


class LifecycleSimProbe:
    simulation_stopped = 0
    stringparam_scene_path_and_name = 13

    def __init__(
        self,
        state: int,
        *,
        scene_path: str = "C:/fixture/default-scene.ttt",
        missing_objects: tuple[str, ...] = (),
    ) -> None:
        self.state = int(state)
        self.scene_path = str(scene_path)
        self.missing_objects = frozenset(missing_objects)
        self.operations: list[tuple] = []
        self.poisoned = False

    def getSimulationState(self) -> int:
        self.operations.append(("getSimulationState",))
        if self.poisoned:
            raise RuntimeError("poisoned yielded client was reused")
        return self.state

    def loadScene(self, path: str) -> None:
        self.operations.append(("loadScene", path))
        self.scene_path = str(path)

    def startSimulation(self) -> None:
        self.operations.append(("startSimulation",))
        self.state = 16

    def stopSimulation(self) -> None:
        self.operations.append(("stopSimulation",))
        self.state = self.simulation_stopped

    def getStringParam(self, parameter: int) -> str:
        self.operations.append(("getStringParam", parameter))
        assert parameter == self.stringparam_scene_path_and_name
        return self.scene_path

    def getObject(self, path: str) -> int:
        self.operations.append(("getObject", path))
        if path in self.missing_objects:
            raise RuntimeError(f"missing scene object: {path}")
        return {
            "/VisionLab": 10,
            "/BLX_base_link": 20,
        }[path]


class LifecycleClientProbe:
    def __init__(self, sim: LifecycleSimProbe) -> None:
        self.sim = sim
        self.socket = SocketProbe()
        self.context = ContextProbe()
        self.require_calls: list[str] = []

    def require(self, name: str) -> LifecycleSimProbe:
        self.require_calls.append(name)
        return self.sim


class FailingClientProbe:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.socket = SocketProbe()
        self.context = ContextProbe()

    def require(self, name: str):
        raise self.error


class ConfigProbe:
    def __init__(self, **options) -> None:
        self.options = dict(options)

    def getoption(self, name: str):
        return self.options.get(name)


def test_endpoint_options_override_environment_and_support_port_23001():
    host, port = acceptance_conftest._resolve_coppelia_endpoint(
        ConfigProbe(
            **{
                "--coppelia-host": "192.0.2.10",
                "--coppelia-port": 23001,
            }
        ),
        environ={
            "COPPELIA_HOST": "127.0.0.2",
            "COPPELIA_PORT": "23002",
        },
    )

    assert host == "192.0.2.10"
    assert port == 23001


def test_endpoint_uses_environment_then_stable_defaults():
    assert acceptance_conftest._resolve_coppelia_endpoint(
        ConfigProbe(),
        environ={
            "COPPELIA_HOST": "127.0.0.2",
            "COPPELIA_PORT": "23001",
        },
    ) == ("127.0.0.2", 23001)
    assert acceptance_conftest._resolve_coppelia_endpoint(
        ConfigProbe(),
        environ={},
    ) == ("127.0.0.1", 23000)


def test_remote_api_readiness_retries_with_fresh_released_clients():
    clock = FakeClock()
    outcomes = iter(
        [
            ConnectionError("require unavailable"),
            ("state", TimeoutError("state unavailable")),
            0,
        ]
    )
    clients: list[ClientProbe] = []

    def client_factory(*, host: str, port: int) -> ClientProbe:
        assert host == "127.0.0.1"
        assert port == 23000
        client = ClientProbe(next(outcomes))
        clients.append(client)
        return client

    client, sim = acceptance_conftest._wait_for_remote_api(
        client_factory,
        host="127.0.0.1",
        port=23000,
        timeout_s=1.0,
        retry_interval_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )

    assert client is clients[2]
    assert sim.getSimulationState() == 0
    assert [item.require_calls for item in clients] == [
        ["sim"],
        ["sim"],
        ["sim"],
    ]
    assert clients[0].socket.close_calls == [0]
    assert clients[0].context.term_calls == 1
    assert clients[1].socket.close_calls == [0]
    assert clients[1].context.term_calls == 1
    assert clients[2].socket.close_calls == []
    assert clients[2].context.term_calls == 0


def test_remote_api_readiness_timeout_is_bounded_and_preserves_last_error():
    clock = FakeClock()
    clients: list[ClientProbe] = []

    def client_factory(*, host: str, port: int) -> ClientProbe:
        client = ClientProbe(
            ConnectionError(f"RPC unavailable attempt {len(clients) + 1}")
        )
        clients.append(client)
        return client

    with pytest.raises(
        TimeoutError,
        match=r"127\.0\.0\.1:23000.*RPC unavailable attempt",
    ) as captured:
        acceptance_conftest._wait_for_remote_api(
            client_factory,
            host="127.0.0.1",
            port=23000,
            timeout_s=0.25,
            retry_interval_s=0.1,
            clock=clock,
            sleep=clock.sleep,
        )

    assert clock.now == pytest.approx(0.25)
    assert len(clients) == 4
    assert all(item.socket.close_calls == [0] for item in clients)
    assert all(item.context.term_calls == 1 for item in clients)
    assert captured.value.__cause__ is not None
    assert str(captured.value.__cause__) == "RPC unavailable attempt 4"


def test_remote_api_readiness_rejects_stopped_simulation_and_retries_fresh():
    clock = FakeClock()
    outcomes = iter([0, 16])
    clients: list[ClientProbe] = []

    def client_factory(*, host: str, port: int) -> ClientProbe:
        client = ClientProbe(next(outcomes))
        clients.append(client)
        return client

    client, sim = acceptance_conftest._wait_for_remote_api(
        client_factory,
        host="127.0.0.1",
        port=23000,
        timeout_s=1.0,
        retry_interval_s=0.1,
        state_predicate=lambda state, candidate: (
            state != int(candidate.simulation_stopped)
        ),
        state_description="simulation running",
        clock=clock,
        sleep=clock.sleep,
    )

    assert client is clients[1]
    assert sim.getSimulationState() == 16
    assert clients[0].socket.close_calls == [0]
    assert clients[0].context.term_calls == 1
    assert clients[1].socket.close_calls == []
    assert clients[1].context.term_calls == 0


def test_running_state_readiness_timeout_is_bounded_and_preserves_state_error():
    clock = FakeClock()
    clients: list[ClientProbe] = []

    def client_factory(*, host: str, port: int) -> ClientProbe:
        client = ClientProbe(0)
        clients.append(client)
        return client

    with pytest.raises(
        TimeoutError,
        match="simulation running",
    ) as captured:
        acceptance_conftest._wait_for_remote_api(
            client_factory,
            host="127.0.0.1",
            port=23000,
            timeout_s=0.25,
            retry_interval_s=0.1,
            state_predicate=lambda state, candidate: (
                state != int(candidate.simulation_stopped)
            ),
            state_description="simulation running",
            clock=clock,
            sleep=clock.sleep,
        )

    assert clock.now == pytest.approx(0.25)
    assert len(clients) == 4
    assert all(item.socket.close_calls == [0] for item in clients)
    assert all(item.context.term_calls == 1 for item in clients)
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "state 0 is not simulation running" in str(
        captured.value.__cause__
    )


def test_prepare_running_scene_rotates_load_start_and_running_clients(
    tmp_path: Path,
):
    clock = FakeClock()
    scene_path = tmp_path / "acceptance scene.ttt"
    setup = LifecycleClientProbe(LifecycleSimProbe(0))
    post_start_stopped = LifecycleClientProbe(
        LifecycleSimProbe(0, scene_path=scene_path.as_posix())
    )
    post_start_running = LifecycleClientProbe(
        LifecycleSimProbe(16, scene_path=scene_path.as_posix())
    )
    pending = iter([setup, post_start_stopped, post_start_running])

    def client_factory(*, host: str, port: int) -> LifecycleClientProbe:
        return next(pending)

    client, sim = acceptance_conftest._prepare_running_scene(
        client_factory,
        host="127.0.0.1",
        port=23000,
        scene_path=scene_path,
        timeout_s=1.0,
        retry_interval_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )

    assert ("startSimulation",) not in setup.sim.operations
    assert setup.socket.close_calls == [0]
    assert setup.context.term_calls == 1
    assert ("startSimulation",) in post_start_stopped.sim.operations
    assert post_start_stopped.socket.close_calls == [0]
    assert post_start_stopped.context.term_calls == 1
    assert client is post_start_running
    assert sim is post_start_running.sim
    assert post_start_running.socket.close_calls == []
    assert post_start_running.context.term_calls == 0


def test_borrowed_scene_reloads_then_uses_fresh_clients_to_start_and_yield(
    tmp_path: Path,
):
    clock = FakeClock()
    scene_path = tmp_path / "borrowed acceptance.ttt"
    load_client = LifecycleClientProbe(
        LifecycleSimProbe(
            16,
            scene_path="C:/teacher/previous-scene.ttt",
        )
    )
    scene_ready = LifecycleClientProbe(
        LifecycleSimProbe(0, scene_path=scene_path.as_posix())
    )
    post_start = LifecycleClientProbe(
        LifecycleSimProbe(16, scene_path=scene_path.as_posix())
    )
    pending = iter([load_client, scene_ready, post_start])

    client, sim = acceptance_conftest._prepare_running_scene(
        lambda **kwargs: next(pending),
        host="127.0.0.1",
        port=23000,
        scene_path=scene_path,
        timeout_s=1.0,
        retry_interval_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )

    assert ("stopSimulation",) in load_client.sim.operations
    assert ("loadScene", scene_path.as_posix()) in (
        load_client.sim.operations
    )
    assert ("startSimulation",) not in load_client.sim.operations
    assert load_client.socket.close_calls == [0]
    assert ("getObject", "/VisionLab") in scene_ready.sim.operations
    assert ("getObject", "/BLX_base_link") in scene_ready.sim.operations
    assert ("startSimulation",) in scene_ready.sim.operations
    assert scene_ready.socket.close_calls == [0]
    assert client is post_start
    assert sim is post_start.sim
    assert ("getObject", "/VisionLab") in post_start.sim.operations
    assert post_start.socket.close_calls == []


def test_prepare_running_scene_releases_setup_client_when_scene_load_fails(
    tmp_path: Path,
):
    clock = FakeClock()
    setup = LifecycleClientProbe(LifecycleSimProbe(0))

    def fail_load(path: str) -> None:
        raise RuntimeError("scene load failed")

    setup.sim.loadScene = fail_load

    with pytest.raises(RuntimeError, match="scene load failed"):
        acceptance_conftest._prepare_running_scene(
            lambda **kwargs: setup,
            host="127.0.0.1",
            port=23000,
            scene_path=tmp_path / "broken scene.ttt",
            timeout_s=1.0,
            retry_interval_s=0.1,
            clock=clock,
            sleep=clock.sleep,
        )

    assert setup.socket.close_calls == [0]
    assert setup.context.term_calls == 1


def test_start_failure_after_load_stops_and_restores_borrowed_scene(
    tmp_path: Path,
):
    clock = FakeClock()
    scene_path = tmp_path / "temporary acceptance.ttt"
    restore_scene_path = tmp_path / "teacher source.ttt"
    setup = LifecycleClientProbe(LifecycleSimProbe(0))
    failing_start_sim = LifecycleSimProbe(
        0,
        scene_path=scene_path.as_posix(),
    )

    def fail_start() -> None:
        failing_start_sim.operations.append(("startSimulation",))
        raise RuntimeError("start refused")

    failing_start_sim.startSimulation = fail_start
    failing_start = LifecycleClientProbe(failing_start_sim)
    cleanup_stop = LifecycleClientProbe(LifecycleSimProbe(16))
    cleanup_verify = LifecycleClientProbe(LifecycleSimProbe(0))
    pending = iter(
        [setup, failing_start, cleanup_stop, cleanup_verify]
    )

    with pytest.raises(
        TimeoutError,
        match="start refused",
    ) as captured:
        with acceptance_conftest._running_scene_connection(
            lambda **kwargs: next(pending),
            host="127.0.0.1",
            port=23000,
            scene_path=scene_path,
            restore_scene_path=restore_scene_path,
            readiness_timeout_s=0.0,
            cleanup_timeout_s=1.0,
            retry_interval_s=0.1,
            clock=clock,
            sleep=clock.sleep,
        ):
            raise AssertionError("body must not run")

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert ("loadScene", scene_path.as_posix()) in setup.sim.operations
    assert ("stopSimulation",) in cleanup_stop.sim.operations
    assert ("loadScene", restore_scene_path.as_posix()) in (
        cleanup_verify.sim.operations
    )
    assert all(
        client.socket.close_calls == [0]
        for client in (
            setup,
            failing_start,
            cleanup_stop,
            cleanup_verify,
        )
    )


def test_running_readiness_failure_preserves_primary_and_cleanup_cause(
    tmp_path: Path,
):
    clock = FakeClock()
    scene_path = tmp_path / "temporary acceptance.ttt"
    restore_scene_path = tmp_path / "teacher source.ttt"
    setup = LifecycleClientProbe(LifecycleSimProbe(0))
    scene_ready = LifecycleClientProbe(
        LifecycleSimProbe(0, scene_path=scene_path.as_posix())
    )
    not_running = LifecycleClientProbe(
        LifecycleSimProbe(0, scene_path=scene_path.as_posix())
    )
    initial = iter([setup, scene_ready, not_running])
    cleanup_clients: list[FailingClientProbe] = []

    def client_factory(*, host: str, port: int):
        try:
            return next(initial)
        except StopIteration:
            client = FailingClientProbe(
                RuntimeError("cleanup RPC refused")
            )
            cleanup_clients.append(client)
            return client

    with pytest.raises(
        TimeoutError,
        match="expected scene running",
    ) as captured:
        with acceptance_conftest._running_scene_connection(
            client_factory,
            host="127.0.0.1",
            port=23000,
            scene_path=scene_path,
            restore_scene_path=restore_scene_path,
            readiness_timeout_s=0.0,
            cleanup_timeout_s=0.0,
            retry_interval_s=0.1,
            clock=clock,
            sleep=clock.sleep,
        ):
            raise AssertionError("body must not run")

    assert isinstance(captured.value.__cause__, TimeoutError)
    assert "cleanup did not complete" in str(captured.value.__cause__)
    assert ("loadScene", scene_path.as_posix()) in setup.sim.operations
    assert ("startSimulation",) in scene_ready.sim.operations
    assert not_running.socket.close_calls == [0]
    assert len(cleanup_clients) == 1
    assert cleanup_clients[0].socket.close_calls == [0]


def test_teardown_stops_simulation_with_fresh_locally_released_clients():
    clock = FakeClock()
    stop_client = LifecycleClientProbe(LifecycleSimProbe(16))
    stopped_client = LifecycleClientProbe(LifecycleSimProbe(0))
    pending = iter([stop_client, stopped_client])

    def client_factory(*, host: str, port: int) -> LifecycleClientProbe:
        return next(pending)

    acceptance_conftest._stop_simulation_with_fresh_clients(
        client_factory,
        host="127.0.0.1",
        port=23000,
        timeout_s=1.0,
        retry_interval_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )

    assert ("stopSimulation",) in stop_client.sim.operations
    assert stop_client.socket.close_calls == [0]
    assert stop_client.context.term_calls == 1
    assert stopped_client.socket.close_calls == [0]
    assert stopped_client.context.term_calls == 1


def test_running_scene_teardown_never_reuses_poisoned_yielded_client(
    tmp_path: Path,
):
    clock = FakeClock()
    scene_path = tmp_path / "acceptance scene.ttt"
    restore_scene_path = tmp_path / "BL23 source.ttt"
    setup = LifecycleClientProbe(LifecycleSimProbe(0))
    scene_ready = LifecycleClientProbe(
        LifecycleSimProbe(0, scene_path=scene_path.as_posix())
    )
    yielded = LifecycleClientProbe(
        LifecycleSimProbe(16, scene_path=scene_path.as_posix())
    )
    cleanup_stop = LifecycleClientProbe(LifecycleSimProbe(16))
    cleanup_verify = LifecycleClientProbe(LifecycleSimProbe(0))
    pending = iter(
        [setup, scene_ready, yielded, cleanup_stop, cleanup_verify]
    )

    def client_factory(*, host: str, port: int) -> LifecycleClientProbe:
        return next(pending)

    with acceptance_conftest._running_scene_connection(
        client_factory,
        host="127.0.0.1",
        port=23000,
        scene_path=scene_path,
        restore_scene_path=restore_scene_path,
        readiness_timeout_s=1.0,
        cleanup_timeout_s=1.0,
        retry_interval_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    ) as (client, sim):
        assert client is yielded
        assert sim is yielded.sim
        operations_before_poison = list(sim.operations)
        sim.poisoned = True

    assert yielded.sim.operations == operations_before_poison
    assert yielded.socket.close_calls == [0]
    assert yielded.context.term_calls == 1
    assert ("stopSimulation",) in cleanup_stop.sim.operations
    assert cleanup_stop.socket.close_calls == [0]
    assert cleanup_verify.socket.close_calls == [0]
    assert ("loadScene", restore_scene_path.as_posix()) in (
        cleanup_verify.sim.operations
    )


def test_cleanup_failure_is_chained_without_masking_original_body_error(
    tmp_path: Path,
):
    clock = FakeClock()
    scene_path = tmp_path / "acceptance scene.ttt"
    setup = LifecycleClientProbe(LifecycleSimProbe(0))
    scene_ready = LifecycleClientProbe(
        LifecycleSimProbe(0, scene_path=scene_path.as_posix())
    )
    yielded = LifecycleClientProbe(
        LifecycleSimProbe(16, scene_path=scene_path.as_posix())
    )
    initial = iter([setup, scene_ready, yielded])
    cleanup_clients: list[FailingClientProbe] = []

    def client_factory(*, host: str, port: int):
        try:
            return next(initial)
        except StopIteration:
            client = FailingClientProbe(
                TimeoutError(
                    f"cleanup RPC unavailable {len(cleanup_clients) + 1}"
                )
            )
            cleanup_clients.append(client)
            return client

    with pytest.raises(
        AssertionError,
        match="original test failure",
    ) as captured:
        with acceptance_conftest._running_scene_connection(
            client_factory,
            host="127.0.0.1",
            port=23000,
            scene_path=scene_path,
            readiness_timeout_s=1.0,
            cleanup_timeout_s=0.25,
            retry_interval_s=0.1,
            clock=clock,
            sleep=clock.sleep,
        ):
            raise AssertionError("original test failure")

    assert isinstance(captured.value.__cause__, TimeoutError)
    assert "cleanup did not complete" in str(captured.value.__cause__)
    assert len(cleanup_clients) == 4
    assert all(
        client.socket.close_calls == [0]
        for client in cleanup_clients
    )
    assert all(
        client.context.term_calls == 1
        for client in cleanup_clients
    )


def test_fixture_uses_post_start_connection_and_quarantined_local_close():
    source = Path(acceptance_conftest.__file__).read_text(encoding="utf-8")
    fixture_source = source.split(
        "def running_vision_scene",
        1,
    )[1]

    assert "with _running_scene_connection(" in fixture_source
    assert "application.close_quarantined()" in fixture_source
    assert "_prepare_running_scene(" not in fixture_source
    assert "_stop_simulation_with_fresh_clients(" not in fixture_source
    assert "_close_remote_client(client)" not in fixture_source
    assert "_resolve_coppelia_endpoint(request.config)" in fixture_source


def test_prelaunched_scene_requirement_fails_with_clear_start_command(
    tmp_path: Path,
):
    readiness_called = False

    def unexpected_readiness(*args, **kwargs):
        nonlocal readiness_called
        readiness_called = True
        raise AssertionError("readiness must not run without a listener")

    with pytest.raises(
        pytest.fail.Exception,
        match=r"launch_coppeliasim\.ps1",
    ):
        acceptance_conftest._require_prelaunched_scene(
            lambda **kwargs: None,
            host="127.0.0.1",
            port=23000,
            scene_path=tmp_path / "BL23_vision_lab.ttt",
            port_probe=lambda host, port: False,
            wait_for_scene_fn=unexpected_readiness,
        )

    assert readiness_called is False


def test_prelaunched_scene_requirement_reports_wrong_scene(
    tmp_path: Path,
):
    with pytest.raises(
        pytest.fail.Exception,
        match=r"unexpected scene.*launch_coppeliasim\.ps1",
    ):
        acceptance_conftest._require_prelaunched_scene(
            lambda **kwargs: None,
            host="127.0.0.1",
            port=23000,
            scene_path=tmp_path / "BL23_vision_lab.ttt",
            port_probe=lambda host, port: True,
            wait_for_scene_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
                TimeoutError("unexpected scene")
            ),
        )


def test_fixture_never_starts_or_terminates_coppeliasim_processes():
    source = Path(acceptance_conftest.__file__).read_text(encoding="utf-8")

    assert "launch_coppeliasim.ps1" in source
    assert "subprocess" not in source
    assert "Popen" not in source
    assert "_managed_coppeliasim_process" not in source
    assert "terminate(" not in source
    assert ".kill(" not in source
