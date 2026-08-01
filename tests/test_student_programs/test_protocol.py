import pytest

from vision_platform.student.protocol import (
    ALLOWED_COMMANDS,
    CommandMessage,
    ResponseMessage,
    RunState,
)


def test_command_message_round_trip_uses_schema_version_one():
    message = CommandMessage(
        command_id="000001",
        name="robot.move_world",
        args={
            "x_mm": 100.0,
            "y_mm": 20.0,
            "z_mm": 120.0,
            "speed": 15.0,
        },
    )

    payload = message.to_dict()

    assert payload["schema_version"] == 1
    assert CommandMessage.from_dict(payload) == message


def test_command_message_rejects_unknown_command():
    with pytest.raises(ValueError, match="COMMAND_NOT_ALLOWED"):
        CommandMessage(
            command_id="000001",
            name="sim.setObjectPosition",
            args={},
        )


def test_response_message_round_trip_preserves_structured_error():
    message = ResponseMessage(
        command_id="000002",
        status="FAIL",
        value=None,
        error={"code": "TARGET_OUT_OF_WORKSPACE", "message": "越界"},
    )

    assert ResponseMessage.from_dict(message.to_dict()) == message


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        (
            CommandMessage,
            {
                "kind": "command",
                "command_id": "000003",
                "name": "robot.home",
                "args": {},
            },
        ),
        (
            ResponseMessage,
            {
                "kind": "result",
                "command_id": "000003",
                "status": "PASS",
                "value": None,
                "error": None,
            },
        ),
    ],
)
@pytest.mark.parametrize("schema_version", [1.9, True, "1", None, []])
def test_from_dict_strictly_rejects_invalid_schema_versions(
    message_type,
    payload,
    schema_version,
):
    invalid_payload = {**payload, "schema_version": schema_version}

    with pytest.raises(ValueError, match="PROTOCOL_VERSION_UNSUPPORTED"):
        message_type.from_dict(invalid_payload)


@pytest.mark.parametrize(
    "error",
    [
        [("code", "BROKEN_ERROR"), ("message", "畸形错误")],
        "BROKEN_ERROR",
    ],
)
def test_response_from_dict_rejects_non_mapping_error(error):
    payload = {
        "schema_version": 1,
        "kind": "result",
        "command_id": "000004",
        "status": "FAIL",
        "value": None,
        "error": error,
    }

    with pytest.raises(ValueError, match="error must be a mapping"):
        ResponseMessage.from_dict(payload)


def test_run_state_values_are_stable_and_ordered():
    assert [state.value for state in RunState] == [
        "EMPTY",
        "LOADED",
        "VALIDATED",
        "RUNNING",
        "PAUSED",
        "PASSED",
        "FAILED",
        "CANCELLED",
        "RESETTING",
    ]


def test_allowed_commands_are_exactly_the_public_student_api():
    assert ALLOWED_COMMANDS == frozenset(
        {
            "camera.capture",
            "camera.profile.apply",
            "camera.profile.get",
            "camera.profile.reset",
            "context.log",
            "context.sleep",
            "context.checkpoint",
            "experiment.info",
            "robot.home",
            "robot.move_world",
            "robot.pose",
            "tool.on",
            "tool.off",
            "vision2d.analyze",
        }
    )


def test_v2_2_read_only_commands_are_whitelisted():
    assert "camera.capture" in ALLOWED_COMMANDS
    assert "experiment.info" in ALLOWED_COMMANDS


def test_v1_01_profile_commands_are_exactly_whitelisted():
    assert {
        "camera.profile.get",
        "camera.profile.apply",
        "camera.profile.reset",
    } <= ALLOWED_COMMANDS
    assert "sim.setObjectInt32Param" not in ALLOWED_COMMANDS
    assert "camera.profile.set_arbitrary" not in ALLOWED_COMMANDS


def test_v1_02_vision2d_analysis_is_exactly_whitelisted():
    assert "vision2d.analyze" in ALLOWED_COMMANDS
    assert "vision2d.configure" not in ALLOWED_COMMANDS
    assert "vision2d.analyze_file" not in ALLOWED_COMMANDS
    assert "opencv.execute" not in ALLOWED_COMMANDS

    command = CommandMessage("000001", "vision2d.analyze", {})

    assert command.to_dict()["args"] == {}
