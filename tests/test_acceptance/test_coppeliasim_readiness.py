from io import StringIO
from pathlib import Path

import pytest

from vision_platform.coppeliasim_readiness import (
    close_remote_client,
    main,
    require_expected_scene,
    scene_paths_match,
    wait_for_scene,
)


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
    stringparam_scene_path_and_name = 13

    def __init__(
        self,
        *,
        scene_path: str,
        missing_objects: tuple[str, ...] = (),
    ) -> None:
        self.scene_path = scene_path
        self.missing_objects = frozenset(missing_objects)
        self.object_calls: list[str] = []

    def getSimulationState(self) -> int:
        return 0

    def getStringParam(self, parameter: int) -> str:
        assert parameter == self.stringparam_scene_path_and_name
        return self.scene_path

    def getObject(self, path: str) -> int:
        self.object_calls.append(path)
        if path in self.missing_objects:
            raise RuntimeError(f"missing scene object: {path}")
        return len(self.object_calls)


class ClientProbe:
    def __init__(self, sim: SimProbe) -> None:
        self.sim = sim
        self.socket = SocketProbe()
        self.context = ContextProbe()

    def require(self, name: str) -> SimProbe:
        assert name == "sim"
        return self.sim


class FailingClientProbe:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.socket = SocketProbe()
        self.context = ContextProbe()

    def require(self, name: str):
        assert name == "sim"
        raise self.error


class CleanupFailingSocketProbe(SocketProbe):
    def close(self, linger=None) -> None:
        super().close(linger=linger)
        raise RuntimeError("socket close failed")


class CleanupFailingContextProbe(ContextProbe):
    def term(self) -> None:
        super().term()
        raise RuntimeError("context term failed")


class FactoryMustNotRun(BaseException):
    pass


@pytest.mark.parametrize(
    "timeout_s",
    (float("nan"), float("inf"), float("-inf"), -0.001),
)
def test_invalid_timeout_is_rejected_before_client_factory(
    tmp_path: Path,
    timeout_s: float,
):
    factory_calls = 0

    def client_factory(**kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise FactoryMustNotRun(
            "invalid timeout must fail before factory"
        )

    with pytest.raises(ValueError, match="timeout_s"):
        wait_for_scene(
            client_factory,
            host="127.0.0.1",
            port=23000,
            scene_path=tmp_path / "scene.ttt",
            timeout_s=timeout_s,
        )

    assert factory_calls == 0


@pytest.mark.parametrize(
    "retry_interval_s",
    (float("nan"), float("inf"), float("-inf"), -0.001, 0.0),
)
def test_invalid_retry_interval_is_rejected_before_client_factory(
    tmp_path: Path,
    retry_interval_s: float,
):
    factory_calls = 0

    def client_factory(**kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise FactoryMustNotRun(
            "invalid retry interval must fail before factory"
        )

    with pytest.raises(ValueError, match="retry_interval_s"):
        wait_for_scene(
            client_factory,
            host="127.0.0.1",
            port=23000,
            scene_path=tmp_path / "scene.ttt",
            retry_interval_s=retry_interval_s,
        )

    assert factory_calls == 0


def test_zero_timeout_allows_one_immediate_attempt(tmp_path: Path):
    client = FailingClientProbe(ConnectionError("immediate failure"))
    factory_calls = 0

    def client_factory(**kwargs):
        nonlocal factory_calls
        factory_calls += 1
        return client

    with pytest.raises(TimeoutError, match="immediate failure"):
        wait_for_scene(
            client_factory,
            host="127.0.0.1",
            port=23000,
            scene_path=tmp_path / "scene.ttt",
            timeout_s=0.0,
            retry_interval_s=0.1,
        )

    assert factory_calls == 1
    assert client.socket.close_calls == [0]
    assert client.context.term_calls == 1


def test_scene_path_comparison_accepts_coppelia_slashes_and_windows_case():
    assert scene_paths_match(
        r"C:\Temp\Blinx Lab\Acceptance.TTT",
        "c:/temp/blinx lab/acceptance.ttt",
    )
    assert not scene_paths_match(
        r"C:\Temp\Blinx Lab\Acceptance.TTT",
        "c:/temp/blinx lab/other.ttt",
    )


def test_readiness_returns_client_only_after_scene_and_sentinels_match(
    tmp_path: Path,
):
    scene_path = tmp_path / "BL23 vision lab.ttt"
    sim = SimProbe(scene_path=scene_path.as_posix())
    client = ClientProbe(sim)

    ready_client, ready_sim = wait_for_scene(
        lambda **kwargs: client,
        host="127.0.0.1",
        port=23000,
        scene_path=scene_path,
    )

    assert ready_client is client
    assert ready_sim is sim
    assert sim.object_calls == ["/VisionLab", "/BLX_base_link"]
    assert client.socket.close_calls == []
    assert client.context.term_calls == 0


@pytest.mark.parametrize(
    ("filename", "expected_root"),
    (
        ("BL23_vision_lab.ttt", "/VisionLab"),
        ("BL23_robot_basics.ttt", "/VisionLab"),
        ("BL23_logistics_lab.ttt", "/VisionLab"),
        ("BL23_vision_quality_lab.ttt", "/VisionQualityLab"),
        ("BL23_vision_code_routing_lab.ttt", "/VisionCodeRoutingLab"),
        ("BL23_vision_ocr_sorting_lab.ttt", "/VisionOcrSortingLab"),
    ),
)
def test_readiness_selects_scene_specific_root_without_weakening_robot_sentinel(
    tmp_path: Path,
    filename: str,
    expected_root: str,
):
    scene_path = tmp_path / filename
    sim = SimProbe(scene_path=scene_path.as_posix())

    require_expected_scene(sim, scene_path)

    assert sim.object_calls == [expected_root, "/BLX_base_link"]


def test_wrong_scene_retries_with_a_fresh_released_client(tmp_path: Path):
    clock = FakeClock()
    scene_path = tmp_path / "BL23 vision lab.ttt"
    wrong = ClientProbe(SimProbe(scene_path="C:/other/scene.ttt"))
    ready = ClientProbe(SimProbe(scene_path=scene_path.as_posix()))
    pending = iter([wrong, ready])

    client, _ = wait_for_scene(
        lambda **kwargs: next(pending),
        host="127.0.0.1",
        port=23000,
        scene_path=scene_path,
        timeout_s=1.0,
        retry_interval_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )

    assert client is ready
    assert wrong.socket.close_calls == [0]
    assert wrong.context.term_calls == 1
    assert ready.socket.close_calls == []
    assert clock.now == pytest.approx(0.1)


def test_missing_sentinel_retries_with_a_fresh_released_client(
    tmp_path: Path,
):
    clock = FakeClock()
    scene_path = tmp_path / "BL23 vision lab.ttt"
    missing = ClientProbe(
        SimProbe(
            scene_path=scene_path.as_posix(),
            missing_objects=("/VisionLab",),
        )
    )
    ready = ClientProbe(SimProbe(scene_path=scene_path.as_posix()))
    pending = iter([missing, ready])

    client, _ = wait_for_scene(
        lambda **kwargs: next(pending),
        host="127.0.0.1",
        port=23000,
        scene_path=scene_path,
        timeout_s=1.0,
        retry_interval_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )

    assert client is ready
    assert missing.socket.close_calls == [0]
    assert missing.context.term_calls == 1


def test_rpc_error_retries_with_a_fresh_released_client(tmp_path: Path):
    clock = FakeClock()
    scene_path = tmp_path / "BL23 vision lab.ttt"
    failed = FailingClientProbe(TimeoutError("RPC receive timed out"))
    ready = ClientProbe(SimProbe(scene_path=scene_path.as_posix()))
    pending = iter([failed, ready])

    client, _ = wait_for_scene(
        lambda **kwargs: next(pending),
        host="127.0.0.1",
        port=23000,
        scene_path=scene_path,
        timeout_s=1.0,
        retry_interval_s=0.1,
        clock=clock,
        sleep=clock.sleep,
    )

    assert client is ready
    assert failed.socket.close_calls == [0]
    assert failed.context.term_calls == 1


def test_readiness_timeout_is_bounded_and_preserves_last_rpc_cause(
    tmp_path: Path,
):
    clock = FakeClock()
    clients: list[FailingClientProbe] = []

    def client_factory(**kwargs) -> FailingClientProbe:
        client = FailingClientProbe(
            ConnectionError(f"RPC unavailable attempt {len(clients) + 1}")
        )
        clients.append(client)
        return client

    with pytest.raises(
        TimeoutError,
        match=r"127\.0\.0\.1:23000.*RPC unavailable attempt 4",
    ) as captured:
        wait_for_scene(
            client_factory,
            host="127.0.0.1",
            port=23000,
            scene_path=tmp_path / "BL23 vision lab.ttt",
            timeout_s=0.25,
            retry_interval_s=0.1,
            clock=clock,
            sleep=clock.sleep,
        )

    assert clock.now == pytest.approx(0.25)
    assert len(clients) == 4
    assert all(client.socket.close_calls == [0] for client in clients)
    assert all(client.context.term_calls == 1 for client in clients)
    assert isinstance(captured.value.__cause__, ConnectionError)
    assert str(captured.value.__cause__) == "RPC unavailable attempt 4"


def test_readiness_cli_returns_zero_and_releases_success_client(
    tmp_path: Path,
):
    scene_path = tmp_path / "BL23 vision lab.ttt"
    client = ClientProbe(SimProbe(scene_path=scene_path.as_posix()))
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "23000",
            "--scene",
            str(scene_path),
            "--timeout",
            "1",
        ],
        client_factory=lambda **kwargs: client,
        out=stdout,
        err=stderr,
    )

    assert exit_code == 0
    assert "READY" in stdout.getvalue()
    assert stderr.getvalue() == ""
    assert client.socket.close_calls == [0]
    assert client.context.term_calls == 1


def test_readiness_cli_does_not_print_ready_when_success_cleanup_fails(
    tmp_path: Path,
):
    scene_path = tmp_path / "BL23 vision lab.ttt"
    client = ClientProbe(SimProbe(scene_path=scene_path.as_posix()))
    client.socket = CleanupFailingSocketProbe()
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--scene", str(scene_path), "--timeout", "1"],
        client_factory=lambda **kwargs: client,
        out=stdout,
        err=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "NOT_READY" in stderr.getvalue()
    assert "socket close failed" in stderr.getvalue()
    assert client.socket.close_calls == [0]
    assert client.context.term_calls == 1


def test_client_cleanup_keeps_socket_error_primary_and_chains_context_error(
    tmp_path: Path,
):
    client = ClientProbe(
        SimProbe(scene_path=(tmp_path / "scene.ttt").as_posix())
    )
    client.socket = CleanupFailingSocketProbe()
    client.context = CleanupFailingContextProbe()

    with pytest.raises(RuntimeError, match="socket close failed") as captured:
        close_remote_client(client)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "context term failed"
    assert client.socket.close_calls == [0]
    assert client.context.term_calls == 1


@pytest.mark.parametrize("timeout", ("nan", "inf", "-inf"))
def test_readiness_cli_rejects_nonfinite_timeout_without_connecting(
    tmp_path: Path,
    timeout: str,
):
    factory_calls = 0
    stdout = StringIO()
    stderr = StringIO()

    def client_factory(**kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise FactoryMustNotRun("invalid CLI timeout must not connect")

    exit_code = main(
        [
            "--scene",
            str(tmp_path / "scene.ttt"),
            f"--timeout={timeout}",
        ],
        client_factory=client_factory,
        out=stdout,
        err=stderr,
    )

    assert exit_code == 1
    assert factory_calls == 0
    assert stdout.getvalue() == ""
    assert "NOT_READY" in stderr.getvalue()
    assert "timeout_s" in stderr.getvalue()


def test_readiness_cli_failure_keeps_last_rpc_error_on_stderr(
    tmp_path: Path,
):
    client = FailingClientProbe(ConnectionError("RPC handshake failed"))
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        [
            "--scene",
            str(tmp_path / "BL23 vision lab.ttt"),
            "--timeout",
            "0",
        ],
        client_factory=lambda **kwargs: client,
        out=stdout,
        err=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "NOT_READY" in stderr.getvalue()
    assert "RPC handshake failed" in stderr.getvalue()
    assert client.socket.close_calls == [0]
    assert client.context.term_calls == 1
