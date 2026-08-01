from __future__ import annotations

import cv2
import numpy as np
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


class ScriptedConnection:
    def __init__(self, values: list[object]) -> None:
        self.values = list(values)
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)

    def recv(self) -> dict:
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"],
            status="PASS",
            value=self.values.pop(0),
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


def _profile_value(**overrides):
    value = {
        "profile_id": "standard",
        "resolution": [512, 512],
        "perspective_angle_deg": 60.0,
        "camera_rig_z_m": 0.7,
        "key_diffuse_rgb": [0.8, 0.8, 0.8],
        "fill_diffuse_rgb": [0.35, 0.35, 0.35],
    }
    value.update(overrides)
    return value


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


def _png_value(**overrides):
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:, :, 1] = 200
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    value = {
        "snapshot_id": "frame-000001",
        "png_bytes": encoded.tobytes(),
        "width": 4,
        "height": 3,
        "source": "coppeliasim",
        "sequence_id": 9,
    }
    value.update(overrides)
    return value


def test_student_camera_decodes_png_into_irreversibly_read_only_bgr():
    connection = ScriptedConnection([_png_value()])

    frame = StudentContext(connection).camera.capture()

    assert frame.snapshot_id == "frame-000001"
    assert frame.image_bgr.shape == (3, 4, 3)
    assert frame.image_bgr.dtype == np.uint8
    assert frame.image_bgr.flags.writeable is False
    with pytest.raises(ValueError):
        frame.image_bgr.setflags(write=True)
    with pytest.raises(ValueError):
        frame.image_bgr[0, 0, 0] = 255
    assert connection.sent[0]["name"] == "camera.capture"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"png_bytes": b"not-png"}, "camera.capture"),
        ({"png_bytes": bytearray(b"not-png")}, "camera.capture"),
        ({"width": True}, "camera dimensions"),
        ({"width": 5}, "camera dimensions"),
        ({"height": 0}, "camera dimensions"),
        ({"source": "hikvision"}, "camera source"),
        ({"source": 3}, "camera source"),
        ({"sequence_id": True}, "camera sequence"),
        ({"sequence_id": -1}, "camera sequence"),
        ({"snapshot_id": "../escape"}, "snapshot_id"),
    ],
)
def test_student_camera_strictly_rejects_invalid_snapshot_metadata(
    overrides, message
):
    connection = ScriptedConnection([_png_value(**overrides)])

    with pytest.raises(RuntimeError, match=message):
        StudentContext(connection).camera.capture()


def test_student_camera_exposes_profile_ids_not_raw_sim_parameters():
    connection = ScriptedConnection(
        [
            _profile_value(),
            _profile_value(
                profile_id="wide_dim",
                resolution=[256, 256],
                perspective_angle_deg=75.0,
                camera_rig_z_m=0.8,
                key_diffuse_rgb=[0.35, 0.35, 0.35],
                fill_diffuse_rgb=[0.15, 0.15, 0.15],
            ),
            _profile_value(),
        ]
    )
    ctx = StudentContext(connection)

    assert ctx.camera.get_profile().profile_id == "standard"
    assert ctx.camera.apply_profile("wide_dim").resolution == (256, 256)
    assert ctx.camera.reset_profile().profile_id == "standard"
    assert [item["name"] for item in connection.sent] == [
        "camera.profile.get",
        "camera.profile.apply",
        "camera.profile.reset",
    ]
    assert connection.sent[1]["args"] == {"profile_id": "wide_dim"}


@pytest.mark.parametrize(
    "value",
    [
        _profile_value(
            resolution=[128, 128],
            perspective_angle_deg=20,
            camera_rig_z_m=0.50,
            key_diffuse_rgb=[0, 0, 0],
            fill_diffuse_rgb=[0, 0, 0],
        ),
        _profile_value(
            resolution=[1024, 1024],
            perspective_angle_deg=90,
            camera_rig_z_m=0.90,
            key_diffuse_rgb=[1, 1, 1],
            fill_diffuse_rgb=[1, 1, 1],
        ),
    ],
)
def test_student_camera_accepts_profile_numeric_contract_boundaries(value):
    profile = StudentContext(ScriptedConnection([value])).camera.get_profile()

    assert 128 <= profile.resolution[0] <= 1024
    assert 128 <= profile.resolution[1] <= 1024
    assert 20 <= profile.perspective_angle_deg <= 90
    assert 0.50 <= profile.camera_rig_z_m <= 0.90
    assert all(0 <= component <= 1 for component in profile.key_diffuse_rgb)
    assert all(0 <= component <= 1 for component in profile.fill_diffuse_rgb)


@pytest.mark.parametrize(
    "value",
    [
        [],
        _profile_value(extra="not-public"),
        {
            key: value
            for key, value in _profile_value().items()
            if key != "camera_rig_z_m"
        },
        _profile_value(profile_id=3),
        _profile_value(profile_id="../unsafe"),
        _profile_value(profile_id="Wide-Dim"),
        _profile_value(profile_id="a" * 33),
        _profile_value(profile_id="\u6807\u51c6"),
        _profile_value(resolution=[512]),
        _profile_value(resolution=[512, 512, 512]),
        _profile_value(resolution=[True, 512]),
        _profile_value(resolution=[512.0, 512]),
        _profile_value(resolution=[127, 512]),
        _profile_value(resolution=[512, 1025]),
        _profile_value(resolution=[10**400, 512]),
        _profile_value(perspective_angle_deg=True),
        _profile_value(perspective_angle_deg=19.999),
        _profile_value(perspective_angle_deg=90.001),
        _profile_value(perspective_angle_deg=float("nan")),
        _profile_value(perspective_angle_deg=10**400),
        _profile_value(camera_rig_z_m=0.499),
        _profile_value(camera_rig_z_m=0.901),
        _profile_value(camera_rig_z_m=float("inf")),
        _profile_value(key_diffuse_rgb=[0.8, 0.8]),
        _profile_value(key_diffuse_rgb=[0.8, True, 0.8]),
        _profile_value(key_diffuse_rgb=[-0.001, 0.8, 0.8]),
        _profile_value(key_diffuse_rgb=[1.001, 0.8, 0.8]),
        _profile_value(key_diffuse_rgb=[10**400, 0.8, 0.8]),
        _profile_value(fill_diffuse_rgb=[-0.001, 0.35, 0.35]),
        _profile_value(fill_diffuse_rgb=[1.001, 0.35, 0.35]),
        _profile_value(fill_diffuse_rgb=[0.35, 0.35, float("-inf")]),
    ],
)
def test_student_camera_rejects_malformed_profile_response(value):
    connection = ScriptedConnection([value])

    with pytest.raises(RuntimeError, match="PROTOCOL_RESPONSE_INVALID"):
        StudentContext(connection).camera.get_profile()


@pytest.mark.parametrize(
    "profile_id",
    [
        None,
        True,
        7,
        "",
        "../standard",
        "wide dim",
        "Wide-Dim",
        "a" * 33,
        "\u6807\u51c6",
    ],
)
def test_student_camera_rejects_unsafe_profile_id_before_rpc(profile_id):
    connection = ScriptedConnection([])

    with pytest.raises(
        ValueError,
        match="profile_id must be a published ASCII identifier",
    ):
        StudentContext(connection).camera.apply_profile(profile_id)

    assert connection.sent == []


def test_student_experiment_returns_detached_public_json_info():
    value = {
        "experiment_id": "R1-05",
        "experiment_version": "2.2.0",
        "scene_sha256": "a" * 64,
        "public_parameters": {"safe_z_mm": 100, "labels": ["A", "B"]},
        "hardware_status": "PENDING_HARDWARE",
    }
    connection = ScriptedConnection([value])

    info = StudentContext(connection).experiment.info()
    info["public_parameters"]["labels"].append("student-change")

    assert info["experiment_id"] == "R1-05"
    assert info["hardware_status"] == "PENDING_HARDWARE"
    assert value["public_parameters"]["labels"] == ["A", "B"]
    assert connection.sent[0]["name"] == "experiment.info"


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"hardware_status": "PASS"},
        {
            "experiment_id": "R1-05",
            "hardware_status": "PENDING_HARDWARE",
            "scene_path": "secret.ttt",
        },
        {
            "experiment_id": "R1-05",
            "hardware_status": "PENDING_HARDWARE",
            "public_parameters": {"bad": object()},
        },
        {
            "experiment_id": "R1-05",
            "hardware_status": "PENDING_HARDWARE",
            "public_parameters": {"bad": float("nan")},
        },
    ],
)
def test_student_experiment_rejects_non_public_or_non_json_info(value):
    connection = ScriptedConnection([value])

    with pytest.raises(RuntimeError, match="PROTOCOL_RESPONSE_INVALID"):
        StudentContext(connection).experiment.info()
