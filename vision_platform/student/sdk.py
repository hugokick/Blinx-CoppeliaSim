from __future__ import annotations

from collections.abc import Sequence
from itertools import count
from math import isfinite
from typing import Any

from vision_platform.student.protocol import CommandMessage, ResponseMessage


class _StudentProgramCancelled(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class _Rpc:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self._ids = count(1)

    def call(self, name: str, **args: Any) -> Any:
        command_id = f"{next(self._ids):06d}"
        command = CommandMessage(command_id, name, args)
        self.connection.send(command.to_dict())
        response = ResponseMessage.from_dict(self.connection.recv())
        if response.command_id != command_id:
            raise RuntimeError("PROTOCOL_CORRELATION_ERROR")
        if response.status == "CANCELLED":
            assert response.error is not None
            raise _StudentProgramCancelled(
                code=str(
                    response.error.get(
                        "code",
                        "STUDENT_PROGRAM_CANCELLED",
                    )
                ),
                message=str(
                    response.error.get(
                        "message",
                        "学生程序通信已取消",
                    )
                ),
            )
        if response.status != "PASS":
            assert response.error is not None
            raise RuntimeError(
                f"{response.error.get('code', 'COMMAND_FAILED')}: "
                f"{response.error.get('message', '命令失败')}"
            )
        return response.value


class StudentRobot:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def home(self) -> None:
        self._rpc.call("robot.home")

    def move_world(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        *,
        speed: float,
    ) -> None:
        self._rpc.call(
            "robot.move_world",
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            speed=speed,
        )

    def pose(self) -> tuple[float, float, float]:
        value = self._rpc.call("robot.pose")
        if (
            isinstance(value, (str, bytes, bytearray))
            or not isinstance(value, Sequence)
            or len(value) != 3
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: robot.pose")
        try:
            pose = (float(value[0]), float(value[1]), float(value[2]))
        except Exception as error:
            raise RuntimeError(
                "PROTOCOL_RESPONSE_INVALID: robot.pose"
            ) from error
        if not all(isfinite(component) for component in pose):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: robot.pose")
        return pose


class StudentTool:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def on(self) -> None:
        self._rpc.call("tool.on")

    def off(self) -> None:
        self._rpc.call("tool.off")


class StudentContext:
    def __init__(self, connection: Any) -> None:
        self._rpc = _Rpc(connection)
        self.robot = StudentRobot(self._rpc)
        self.tool = StudentTool(self._rpc)

    def log(self, message: str) -> None:
        self._rpc.call("context.log", message=str(message))

    def sleep(self, seconds: float) -> None:
        self._rpc.call("context.sleep", seconds=seconds)

    def checkpoint(self, label: str) -> None:
        self._rpc.call("context.checkpoint", label=str(label))
