from __future__ import annotations

import json
from math import inf, nan

import pytest

from vision_platform.errors import MotionSafetyError, VisionPlatformError
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.student.safety import (
    StudentExecutionPolicy,
    StudentMotionGuard,
)


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


class _ExplodingRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr must not escape the safety boundary")


_HUGE_INTEGER = 10**10000


def _policy(**overrides):
    values = {
        "min_speed": 1,
        "max_speed": 30,
        "max_runtime_s": 60,
        "max_commands": 200,
        "command_timeout_s": 10,
        "max_sleep_s": 5,
        "tool_on_max_z_mm": 35,
    }
    values.update(overrides)
    return StudentExecutionPolicy(**values)


def _guard():
    workspace = WorkspacePolicy(
        x_mm=(20, 140),
        y_mm=(-90, 90),
        z_mm=(10, 140),
        safe_z_mm=100,
    )
    return StudentMotionGuard(workspace=workspace, policy=_policy())


def test_safe_vertical_and_high_horizontal_moves_pass():
    guard = _guard()

    assert guard.validate_move(
        (100, 20, 120), (100, 20, 25), speed=8
    ) == (100.0, 20.0, 25.0)
    assert guard.validate_move(
        (100, 20, 120), (120, -20, 120), speed=15
    ) == (120.0, -20.0, 120.0)


def test_low_horizontal_move_is_rejected():
    guard = _guard()

    with pytest.raises(MotionSafetyError, match="低于安全高度") as exc:
        guard.validate_move((100, 20, 25), (120, 20, 25), speed=8)

    assert exc.value.code == "TARGET_OUT_OF_WORKSPACE"
    assert exc.value.details == {
        "start": [100.0, 20.0, 25.0],
        "target": [120.0, 20.0, 25.0],
        "safe_z_mm": 100,
    }


def test_horizontal_tolerance_and_safe_height_boundary_are_allowed():
    guard = _guard()

    assert guard.validate_move(
        (100, 20, 25), (100.000001, 20, 25), speed=1
    ) == (100.000001, 20.0, 25.0)
    assert guard.validate_move(
        (100, 20, 100), (120, 20, 100), speed=30
    ) == (120.0, 20.0, 100.0)


def test_workspace_and_speed_are_checked():
    guard = _guard()

    with pytest.raises(MotionSafetyError, match="workspace"):
        guard.validate_move((100, 20, 120), (500, 20, 120), speed=8)
    with pytest.raises(VisionPlatformError, match="速度") as exc:
        guard.validate_move((100, 20, 120), (100, 20, 100), speed=50)

    assert exc.value.code == "STUDENT_SPEED_INVALID"
    assert exc.value.details == {"speed": 50.0}


@pytest.mark.parametrize("speed", [0, 31, nan, inf, -inf, True, "fast"])
def test_move_rejects_non_finite_boolean_and_non_numeric_speed(speed):
    guard = _guard()

    with pytest.raises(VisionPlatformError, match="速度") as exc:
        guard.validate_move((100, 20, 120), (100, 20, 100), speed=speed)

    assert exc.value.code == "STUDENT_SPEED_INVALID"
    assert "speed" in exc.value.details


@pytest.mark.parametrize(
    ("case_name", "value"),
    [
        ("nan", nan),
        ("positive infinity", inf),
        ("negative infinity", -inf),
        ("boolean", True),
        ("numeric string", "100"),
        ("plain object", object()),
        ("exploding repr", _ExplodingRepr()),
        ("int subclass", _IntSubclass(100)),
        ("float subclass", _FloatSubclass(100.0)),
    ],
)
@pytest.mark.parametrize("endpoint", ["current", "target"])
def test_move_coordinate_errors_are_strictly_json_serializable(
    case_name, value, endpoint
):
    del case_name
    guard = _guard()
    current = (100, 20, 120)
    target = (100, 20, 100)
    if endpoint == "current":
        current = (value, 20, 120)
    else:
        target = (value, 20, 100)

    with pytest.raises(MotionSafetyError) as exc:
        guard.validate_move(current, target, speed=8)

    assert exc.value.code == "TARGET_OUT_OF_WORKSPACE"
    assert exc.value.details["endpoint"] == endpoint
    assert exc.value.details["axis"] == "x"
    assert "value_type" in exc.value.details
    json.dumps(exc.value.details, allow_nan=False)


def test_huge_integer_coordinate_has_structured_safety_error():
    guard = _guard()

    with pytest.raises(MotionSafetyError, match="坐标") as exc:
        guard.validate_move(
            (100, 20, 120),
            (_HUGE_INTEGER, 20, 100),
            speed=8,
        )

    assert exc.value.code == "TARGET_OUT_OF_WORKSPACE"
    json.dumps(exc.value.details, allow_nan=False)


def test_tool_on_requires_low_pick_height():
    guard = _guard()

    assert guard.validate_tool_on((100, 20, 35)) is None
    with pytest.raises(VisionPlatformError, match="吸盘") as exc:
        guard.validate_tool_on((100, 20, 100))

    assert exc.value.code == "STUDENT_TOOL_HEIGHT_INVALID"
    assert exc.value.details == {
        "pose_mm": [100.0, 20.0, 100.0],
        "tool_on_max_z_mm": 35,
    }


def test_sleep_and_log_limits_are_checked():
    guard = _guard()

    assert guard.validate_sleep(0) == 0.0
    assert guard.validate_sleep(5) == 5.0
    assert guard.validate_log("学生日志") == "学生日志"
    assert guard.validate_log(123) == "123"
    assert guard.validate_log("x" * 500) == "x" * 500
    with pytest.raises(VisionPlatformError, match="5") as sleep_exc:
        guard.validate_sleep(6)
    with pytest.raises(VisionPlatformError, match="500") as log_exc:
        guard.validate_log("x" * 501)

    assert sleep_exc.value.code == "STUDENT_SLEEP_INVALID"
    assert sleep_exc.value.details == {"seconds": 6.0}
    assert log_exc.value.code == "STUDENT_LOG_TOO_LONG"
    assert log_exc.value.details == {"length": 501}


@pytest.mark.parametrize("seconds", [nan, inf, -inf, True, "later"])
def test_sleep_rejects_non_finite_boolean_and_non_numeric_values(seconds):
    guard = _guard()

    with pytest.raises(VisionPlatformError, match="5") as exc:
        guard.validate_sleep(seconds)

    assert exc.value.code == "STUDENT_SLEEP_INVALID"
    assert "seconds" in exc.value.details


@pytest.mark.parametrize(
    "value",
    [nan, inf, -inf, True, "15", object(), _ExplodingRepr()],
)
@pytest.mark.parametrize(
    ("method", "code", "detail_name"),
    [
        ("speed", "STUDENT_SPEED_INVALID", "speed"),
        ("sleep", "STUDENT_SLEEP_INVALID", "seconds"),
    ],
)
def test_scalar_safety_errors_are_strictly_json_serializable(
    value, method, code, detail_name
):
    guard = _guard()

    with pytest.raises(VisionPlatformError) as exc:
        if method == "speed":
            guard.validate_move(
                (100, 20, 120),
                (100, 20, 100),
                speed=value,
            )
        else:
            guard.validate_sleep(value)

    assert exc.value.code == code
    assert f"{detail_name}_type" in exc.value.details
    json.dumps(exc.value.details, allow_nan=False)


@pytest.mark.parametrize(
    ("method", "code", "message"),
    [
        ("speed", "STUDENT_SPEED_INVALID", "速度"),
        ("sleep", "STUDENT_SLEEP_INVALID", "等待"),
    ],
)
def test_huge_integer_scalar_has_structured_safety_error(
    method, code, message
):
    guard = _guard()

    with pytest.raises(VisionPlatformError, match=message) as exc:
        if method == "speed":
            guard.validate_move(
                (100, 20, 120),
                (100, 20, 100),
                speed=_HUGE_INTEGER,
            )
        else:
            guard.validate_sleep(_HUGE_INTEGER)

    assert exc.value.code == code
    json.dumps(exc.value.details, allow_nan=False)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"min_speed": 0}, "speed range"),
        ({"min_speed": 31}, "speed range"),
        ({"min_speed": nan}, "finite"),
        ({"max_speed": inf}, "finite"),
        ({"max_speed": "30"}, "number"),
        ({"max_runtime_s": 0}, "positive"),
        ({"max_runtime_s": nan}, "finite"),
        ({"command_timeout_s": inf}, "finite"),
        ({"max_sleep_s": -1}, "positive"),
        ({"tool_on_max_z_mm": nan}, "finite"),
        ({"max_commands": 0}, "positive integer"),
        ({"max_commands": 1.5}, "positive integer"),
        ({"max_commands": nan}, "positive integer"),
        ({"max_commands": inf}, "positive integer"),
        ({"max_commands": True}, "positive integer"),
    ],
)
def test_execution_policy_rejects_invalid_limits(override, message):
    with pytest.raises(ValueError, match=message):
        _policy(**override)


@pytest.mark.parametrize(
    "field",
    [
        "min_speed",
        "max_speed",
        "max_runtime_s",
        "command_timeout_s",
        "max_sleep_s",
        "tool_on_max_z_mm",
    ],
)
def test_execution_policy_rejects_boolean_numeric_fields(field):
    with pytest.raises(ValueError, match="number"):
        _policy(**{field: True})


@pytest.mark.parametrize(
    "override",
    [
        {"min_speed": 0.999},
        {"max_speed": 30.001},
        {"max_runtime_s": 60.001},
        {"max_commands": 201},
        {"command_timeout_s": 10.001},
        {"max_sleep_s": 5.001},
        {"tool_on_max_z_mm": 35.001},
    ],
)
def test_execution_policy_cannot_relax_v21_hard_limits(override):
    with pytest.raises(ValueError, match="hard safety limit"):
        _policy(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"min_speed": _IntSubclass(1)},
        {"max_speed": _FloatSubclass(30)},
        {"max_commands": _IntSubclass(200)},
    ],
)
def test_execution_policy_rejects_numeric_subclasses(override):
    with pytest.raises(ValueError):
        _policy(**override)


@pytest.mark.parametrize(
    "field",
    [
        "min_speed",
        "max_speed",
        "max_runtime_s",
        "command_timeout_s",
        "max_sleep_s",
        "tool_on_max_z_mm",
    ],
)
def test_execution_policy_rejects_huge_integer_as_non_finite(field):
    with pytest.raises(ValueError, match="finite"):
        _policy(**{field: _HUGE_INTEGER})


def test_execution_policy_accepts_finite_boundary_values():
    policy = _policy(
        min_speed=1,
        max_speed=1,
        max_runtime_s=0.001,
        max_commands=1,
        command_timeout_s=0.001,
        max_sleep_s=0.001,
        tool_on_max_z_mm=0,
    )

    assert type(policy.min_speed) is float
    assert type(policy.max_speed) is float
    assert type(policy.max_runtime_s) is float
    assert type(policy.command_timeout_s) is float
    assert type(policy.max_sleep_s) is float
    assert type(policy.tool_on_max_z_mm) is float
    assert type(policy.max_commands) is int
    assert policy.max_speed == 1.0
    assert policy.max_commands == 1
