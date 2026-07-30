from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision_platform.application import VisionLabApplication


class LocalCloseProbe:
    def __init__(self, error: Exception | None = None) -> None:
        self.close_calls = 0
        self.error = error

    def close(self) -> None:
        self.close_calls += 1
        if self.error is not None:
            raise self.error


class RemoteToolProbe:
    def __init__(self) -> None:
        self.off_calls = 0

    def off(self) -> None:
        self.off_calls += 1
        raise AssertionError("quarantined close must not issue tool RPC")


class SocketProbe:
    def __init__(self, error: Exception | None = None) -> None:
        self.close_calls: list[int | None] = []
        self.error = error

    def close(self, linger=None) -> None:
        self.close_calls.append(linger)
        if self.error is not None:
            raise self.error


class ContextProbe:
    def __init__(self) -> None:
        self.term_calls = 0

    def term(self) -> None:
        self.term_calls += 1


class PublicCloseClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.close_calls = 0
        self.error = error
        self.socket = SocketProbe()
        self.context = ContextProbe()

    def close(self) -> None:
        self.close_calls += 1
        if self.error is not None:
            raise self.error


def make_application(
    *,
    camera_error: Exception | None = None,
    robot_error: Exception | None = None,
    client=None,
):
    camera = LocalCloseProbe(camera_error)
    robot = LocalCloseProbe(robot_error)
    tool = RemoteToolProbe()
    socket = SocketProbe()
    context = ContextProbe()
    selected_client = client or SimpleNamespace(
        socket=socket,
        context=context,
    )
    application = VisionLabApplication(
        config=SimpleNamespace(),
        camera=camera,
        base_recognizer=object(),
        robot_backend=object(),
        robot=robot,
        tool=tool,
        verifier=None,
        event_bus=object(),
        workspace=object(),
        zones={},
        scene_spec={},
        sim=object(),
        client=selected_client,
        owns_client=True,
        calibration=None,
    )
    return application, camera, robot, tool, socket, context


def test_quarantined_close_is_local_idempotent_and_never_calls_tool_rpc():
    application, camera, robot, tool, socket, context = make_application()

    application.close_quarantined()
    application.close_quarantined()

    assert tool.off_calls == 0
    assert camera.close_calls == 1
    assert robot.close_calls == 1
    assert socket.close_calls == [0]
    assert context.term_calls == 1
    assert application._closed is True


def test_normal_close_releases_actual_remote_api_transport_without_close():
    application, camera, robot, tool, socket, context = make_application()

    application.close()
    application.close()

    assert tool.off_calls == 1
    assert camera.close_calls == 1
    assert robot.close_calls == 1
    assert socket.close_calls == [0]
    assert context.term_calls == 1
    assert application.client is None
    assert application.sim is None
    assert application._closed is True


def test_normal_close_calls_public_client_close_once_when_available():
    client = PublicCloseClient()
    application, camera, robot, tool, socket, context = make_application(
        client=client,
    )

    application.close()
    application.close()

    assert tool.off_calls == 1
    assert camera.close_calls == 1
    assert robot.close_calls == 1
    assert client.close_calls == 1
    assert client.socket.close_calls == []
    assert client.context.term_calls == 0
    assert socket.close_calls == []
    assert context.term_calls == 0
    assert application.client is None
    assert application.sim is None


def test_public_client_close_error_is_primary_and_local_fallback_runs():
    close_error = RuntimeError("public-client-close-failed")
    client = PublicCloseClient(close_error)
    application, camera, robot, tool, socket, context = make_application(
        client=client,
    )

    with pytest.raises(RuntimeError, match="public-client-close-failed"):
        application.close()
    application.close()

    assert tool.off_calls == 1
    assert camera.close_calls == 1
    assert robot.close_calls == 1
    assert client.close_calls == 1
    assert client.socket.close_calls == [0]
    assert client.context.term_calls == 1
    assert socket.close_calls == []
    assert context.term_calls == 0
    assert application.client is None
    assert application.sim is None
    assert application._closed is True


def test_public_close_remains_primary_when_local_fallback_also_fails():
    public_error = RuntimeError("public-client-close-failed")
    local_error = RuntimeError("local-socket-close-failed")
    client = PublicCloseClient(public_error)
    client.socket = SocketProbe(local_error)
    application, _, _, _, _, _ = make_application(client=client)

    with pytest.raises(
        RuntimeError,
        match="public-client-close-failed",
    ) as captured:
        application.close()
    application.close()

    assert captured.value is public_error
    assert captured.value.__cause__ is local_error
    assert client.close_calls == 1
    assert client.socket.close_calls == [0]
    assert client.context.term_calls == 1
    assert application.client is None
    assert application.sim is None
    assert application._closed is True


@pytest.mark.parametrize("stage", ["camera", "robot"])
def test_normal_close_releases_transport_and_seals_state_after_local_error(
    stage,
):
    close_error = RuntimeError(f"{stage}-close-failed")
    application, camera, robot, tool, socket, context = make_application(
        camera_error=close_error if stage == "camera" else None,
        robot_error=close_error if stage == "robot" else None,
    )

    with pytest.raises(RuntimeError, match=f"{stage}-close-failed"):
        application.close()
    application.close()

    assert tool.off_calls == 1
    assert camera.close_calls == 1
    assert robot.close_calls == 1
    assert socket.close_calls == [0]
    assert context.term_calls == 1
    assert application.client is None
    assert application.sim is None
    assert application._closed is True
