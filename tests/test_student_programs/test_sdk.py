from __future__ import annotations

import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.student.protocol import CommandMessage, ResponseMessage
from vision_platform.student.safety import (
    StudentExecutionPolicy,
    StudentMotionGuard,
)
from vision_platform.student.sdk import StudentContext


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)

    def recv(self) -> dict:
        command_id = self.sent[-1]["command_id"]
        name = self.sent[-1]["name"]
        value = [100.0, 20.0, 120.0] if name == "robot.pose" else None
        return ResponseMessage(
            command_id=command_id,
            status="PASS",
            value=value,
            error=None,
        ).to_dict()


class PoseResponseConnection(FakeConnection):
    def __init__(self, value: object) -> None:
        super().__init__()
        self.value = value

    def recv(self) -> dict:
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"],
            status="PASS",
            value=self.value,
            error=None,
        ).to_dict()


class GuardLoopConnection(FakeConnection):
    def __init__(self) -> None:
        super().__init__()
        self.guard = StudentMotionGuard(
            workspace=WorkspacePolicy(
                x_mm=(20, 140),
                y_mm=(-90, 90),
                z_mm=(10, 140),
                safe_z_mm=100,
            ),
            policy=StudentExecutionPolicy(
                min_speed=1,
                max_speed=30,
                max_runtime_s=60,
                max_commands=200,
                command_timeout_s=10,
                max_sleep_s=5,
                tool_on_max_z_mm=35,
            ),
        )

    def recv(self) -> dict:
        command = CommandMessage.from_dict(self.sent[-1])
        try:
            if command.name == "robot.move_world":
                self.guard.validate_move(
                    (100, 20, 120),
                    (
                        command.args["x_mm"],
                        command.args["y_mm"],
                        command.args["z_mm"],
                    ),
                    speed=command.args["speed"],
                )
            elif command.name == "context.sleep":
                self.guard.validate_sleep(command.args["seconds"])
        except VisionPlatformError as error:
            return ResponseMessage(
                command_id=command.command_id,
                status="FAIL",
                value=None,
                error={
                    "code": error.code,
                    "message": str(error),
                    "details": error.details,
                },
            ).to_dict()
        return ResponseMessage(
            command_id=command.command_id,
            status="PASS",
            value=None,
            error=None,
        ).to_dict()


def test_student_context_emits_whitelisted_commands() -> None:
    connection = FakeConnection()
    ctx = StudentContext(connection)

    ctx.log("开始")
    ctx.robot.home()
    ctx.robot.move_world(100, 20, 120, speed=15)
    pose = ctx.robot.pose()
    ctx.tool.on()
    ctx.tool.off()

    assert [item["name"] for item in connection.sent] == [
        "context.log",
        "robot.home",
        "robot.move_world",
        "robot.pose",
        "tool.on",
        "tool.off",
    ]
    assert [item["command_id"] for item in connection.sent] == [
        "000001",
        "000002",
        "000003",
        "000004",
        "000005",
        "000006",
    ]
    assert connection.sent[2]["args"] == {
        "x_mm": 100,
        "y_mm": 20,
        "z_mm": 120,
        "speed": 15,
    }
    assert all(
        type(value) is int for value in connection.sent[2]["args"].values()
    )
    assert pose == (100.0, 20.0, 120.0)


def test_failed_response_becomes_student_runtime_error() -> None:
    connection = FakeConnection()

    def failed_recv() -> dict:
        command_id = connection.sent[-1]["command_id"]
        return ResponseMessage(
            command_id=command_id,
            status="FAIL",
            value=None,
            error={"code": "TARGET_OUT_OF_WORKSPACE", "message": "目标越界"},
        ).to_dict()

    connection.recv = failed_recv  # type: ignore[method-assign]
    ctx = StudentContext(connection)

    with pytest.raises(RuntimeError, match="TARGET_OUT_OF_WORKSPACE.*目标越界"):
        ctx.robot.move_world(500, 0, 20, speed=15)


def test_mismatched_response_command_id_is_rejected() -> None:
    connection = FakeConnection()

    def mismatched_recv() -> dict:
        return ResponseMessage(
            command_id="999999",
            status="PASS",
            value=None,
            error=None,
        ).to_dict()

    connection.recv = mismatched_recv  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="PROTOCOL_CORRELATION_ERROR"):
        StudentContext(connection).robot.home()


def test_sleep_preserves_argument_and_checkpoint_converts_label() -> None:
    connection = FakeConnection()
    ctx = StudentContext(connection)

    ctx.sleep("1.25")
    ctx.checkpoint(42)

    assert connection.sent[0]["args"] == {"seconds": "1.25"}
    assert connection.sent[1]["args"] == {"label": "42"}


@pytest.mark.parametrize(
    ("action", "error_code", "argument_name", "expected_type"),
    [
        (
            lambda ctx: ctx.robot.move_world(100, 20, 120, speed=True),
            "STUDENT_SPEED_INVALID",
            "speed",
            bool,
        ),
        (
            lambda ctx: ctx.sleep(True),
            "STUDENT_SLEEP_INVALID",
            "seconds",
            bool,
        ),
        (
            lambda ctx: ctx.robot.move_world("100", 20, 120, speed=15),
            "TARGET_OUT_OF_WORKSPACE",
            "x_mm",
            str,
        ),
    ],
)
def test_student_arguments_reach_parent_guard_without_sdk_coercion(
    action, error_code, argument_name, expected_type
) -> None:
    connection = GuardLoopConnection()
    ctx = StudentContext(connection)

    with pytest.raises(RuntimeError, match=error_code):
        action(ctx)

    assert type(connection.sent[-1]["args"][argument_name]) is expected_type


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        "123",
        {"x": 1.0, "y": 2.0, "z": 3.0},
        [1.0, object(), 3.0],
        [1.0, float("nan"), 3.0],
        [1.0, float("inf"), 3.0],
    ],
)
def test_pose_rejects_malformed_or_non_finite_response(value: object) -> None:
    connection = PoseResponseConnection(value)

    with pytest.raises(RuntimeError, match="PROTOCOL_RESPONSE_INVALID"):
        StudentContext(connection).robot.pose()
