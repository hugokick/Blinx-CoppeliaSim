from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from vision_platform.errors import MotionSafetyError, VisionPlatformError
from vision_platform.robot.safety import Point3, WorkspacePolicy


@dataclass(frozen=True)
class StudentExecutionPolicy:
    min_speed: float
    max_speed: float
    max_runtime_s: float
    max_commands: int
    command_timeout_s: float
    max_sleep_s: float
    tool_on_max_z_mm: float

    def __post_init__(self) -> None:
        numeric_fields = (
            "min_speed",
            "max_speed",
            "max_runtime_s",
            "command_timeout_s",
            "max_sleep_s",
            "tool_on_max_z_mm",
        )
        normalized: dict[str, float] = {}
        for name in numeric_fields:
            value = getattr(self, name)
            if type(value) not in {int, float}:
                raise ValueError(f"{name} must be a number")
            try:
                number = float(value)
            except (ArithmeticError, TypeError, ValueError):
                raise ValueError(f"{name} must be finite") from None
            if not isfinite(number):
                raise ValueError(f"{name} must be finite")
            normalized[name] = number
        if type(self.max_commands) is not int or self.max_commands <= 0:
            raise ValueError("student max_commands must be a positive integer")

        for name, number in normalized.items():
            object.__setattr__(self, name, number)
        object.__setattr__(self, "max_commands", int(self.max_commands))
        if self.min_speed <= 0 or self.max_speed < self.min_speed:
            raise ValueError("student speed range is invalid")
        if self.max_runtime_s <= 0 or self.command_timeout_s <= 0:
            raise ValueError("student time limits must be positive")
        if self.max_sleep_s <= 0:
            raise ValueError("student max_sleep_s must be positive")
        if (
            self.min_speed < 1
            or self.max_speed > 30
            or self.max_runtime_s > 60
            or self.max_commands > 200
            or self.command_timeout_s > 10
            or self.max_sleep_s > 5
            or self.tool_on_max_z_mm > 35
        ):
            raise ValueError(
                "student policy exceeds V2.1 hard safety limit"
            )


class StudentMotionGuard:
    def __init__(
        self,
        *,
        workspace: WorkspacePolicy,
        policy: StudentExecutionPolicy,
    ) -> None:
        self.workspace = workspace
        self.policy = policy

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        if type(value) not in {int, float}:
            return None
        try:
            number = float(value)
        except (ArithmeticError, TypeError, ValueError):
            return None
        return number if isfinite(number) else None

    @staticmethod
    def _safe_repr(value: Any) -> str:
        try:
            rendered = repr(value)
        except BaseException:
            return "<repr unavailable>"
        if type(rendered) is not str:
            return "<invalid repr>"
        return rendered[:200]

    @staticmethod
    def _safe_type_name(value: Any) -> str:
        try:
            value_type = type(value)
            module = value_type.__module__
            name = value_type.__qualname__
            return f"{module}.{name}"[:200]
        except BaseException:
            return "<type unavailable>"

    @classmethod
    def _invalid_scalar_details(
        cls,
        name: str,
        value: Any,
    ) -> dict[str, str]:
        return {
            name: cls._safe_repr(value),
            f"{name}_type": cls._safe_type_name(value),
        }

    def _validate_point(self, point: Any, *, endpoint: str) -> Point3:
        try:
            if isinstance(point, (str, bytes, bytearray)):
                raise TypeError
            values = tuple(point)
        except BaseException:
            raise MotionSafetyError(
                "坐标必须包含三个有限数值",
                endpoint=endpoint,
                axis="point",
                **self._invalid_scalar_details("value", point),
            ) from None
        if len(values) != 3:
            raise MotionSafetyError(
                "坐标必须包含三个有限数值",
                endpoint=endpoint,
                axis="point",
                **self._invalid_scalar_details("value", point),
            )
        normalized: list[float] = []
        for axis, value in zip("xyz", values):
            number = self._finite_number(value)
            if number is None:
                raise MotionSafetyError(
                    "坐标必须使用有限的内置数值",
                    endpoint=endpoint,
                    axis=axis,
                    **self._invalid_scalar_details("value", value),
                )
            normalized.append(number)
        return self.workspace.validate(tuple(normalized))

    def validate_move(
        self,
        current: Point3,
        target: Point3,
        *,
        speed: float,
    ) -> Point3:
        start = self._validate_point(current, endpoint="current")
        end = self._validate_point(target, endpoint="target")
        value = self._finite_number(speed)
        if (
            value is None
            or value < self.policy.min_speed
            or value > self.policy.max_speed
        ):
            raise VisionPlatformError(
                "STUDENT_SPEED_INVALID",
                (
                    f"速度必须在 {self.policy.min_speed:g} 到 "
                    f"{self.policy.max_speed:g} 之间"
                ),
                details=(
                    self._invalid_scalar_details("speed", speed)
                    if value is None
                    else {"speed": value}
                ),
            )
        horizontal = abs(start[0] - end[0]) > 1e-6 or abs(
            start[1] - end[1]
        ) > 1e-6
        if horizontal and (
            start[2] < self.workspace.safe_z_mm
            or end[2] < self.workspace.safe_z_mm
        ):
            raise MotionSafetyError(
                "低于安全高度时禁止水平移动",
                start=list(start),
                target=list(end),
                safe_z_mm=float(self.workspace.safe_z_mm),
            )
        return end

    def validate_tool_on(self, pose: Point3) -> None:
        current = self._validate_point(pose, endpoint="pose")
        if current[2] > self.policy.tool_on_max_z_mm:
            raise VisionPlatformError(
                "STUDENT_TOOL_HEIGHT_INVALID",
                (
                    "吸盘只能在抓取高度开启，"
                    f"当前 Z={current[2]:.1f} mm"
                ),
                details={
                    "pose_mm": list(current),
                    "tool_on_max_z_mm": float(
                        self.policy.tool_on_max_z_mm
                    ),
                },
            )

    def validate_sleep(self, seconds: float) -> float:
        value = self._finite_number(seconds)
        if value is None or value < 0 or value > self.policy.max_sleep_s:
            raise VisionPlatformError(
                "STUDENT_SLEEP_INVALID",
                f"单次等待必须在 0 到 {self.policy.max_sleep_s:g} 秒之间",
                details=(
                    self._invalid_scalar_details("seconds", seconds)
                    if value is None
                    else {"seconds": value}
                ),
            )
        return value

    @staticmethod
    def validate_log(message: str) -> str:
        text = str(message)
        if len(text) > 500:
            raise VisionPlatformError(
                "STUDENT_LOG_TOO_LONG",
                "学生日志单条不得超过 500 个字符",
                details={"length": len(text)},
            )
        return text
