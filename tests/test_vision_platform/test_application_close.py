from __future__ import annotations

from types import SimpleNamespace

from vision_platform.application import VisionLabApplication


class LocalCloseProbe:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class RemoteToolProbe:
    def __init__(self) -> None:
        self.off_calls = 0

    def off(self) -> None:
        self.off_calls += 1
        raise AssertionError("quarantined close must not issue tool RPC")


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


def make_application():
    camera = LocalCloseProbe()
    robot = LocalCloseProbe()
    tool = RemoteToolProbe()
    socket = SocketProbe()
    context = ContextProbe()
    client = SimpleNamespace(socket=socket, context=context)
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
        client=client,
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
