from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from math import isfinite
import re
from typing import Any, Mapping

import cv2
import numpy as np

from vision_platform.student.protocol import CommandMessage, ResponseMessage


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PROFILE_ID = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
_SNAPSHOT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_VISION_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "resolution",
        "perspective_angle_deg",
        "camera_rig_z_m",
        "key_diffuse_rgb",
        "fill_diffuse_rgb",
    }
)
_PUBLIC_EXPERIMENT_FIELDS = frozenset(
    {
        "experiment_id",
        "experiment_version",
        "scene_sha256",
        "public_parameters",
        "hardware_status",
    }
)


def _copy_json_native(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise RuntimeError(
                f"PROTOCOL_RESPONSE_INVALID: {path} must be finite"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise RuntimeError(
                    f"PROTOCOL_RESPONSE_INVALID: {path} keys"
                )
            result[key] = _copy_json_native(
                nested,
                path=f"{path}.{key}",
            )
        return result
    if type(value) is list:
        return [
            _copy_json_native(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise RuntimeError(f"PROTOCOL_RESPONSE_INVALID: {path} JSON type")


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


@dataclass(frozen=True)
class StudentFrame:
    snapshot_id: str
    image_bgr: np.ndarray
    width: int
    height: int
    source: str
    sequence_id: int


@dataclass(frozen=True)
class StudentVisionProfile:
    profile_id: str
    resolution: tuple[int, int]
    perspective_angle_deg: float
    camera_rig_z_m: float
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]


def _profile_error(field: str) -> RuntimeError:
    return RuntimeError(f"PROTOCOL_RESPONSE_INVALID: camera.profile {field}")


def _bounded_profile_number(
    value: Any,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if type(value) not in {int, float}:
        raise _profile_error(field)
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise _profile_error(field) from error
    if not isfinite(result) or not minimum <= result <= maximum:
        raise _profile_error(field)
    return result


def _profile_rgb(value: Any, field: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise _profile_error(field)
    return (
        _bounded_profile_number(value[0], f"{field}[0]", 0.0, 1.0),
        _bounded_profile_number(value[1], f"{field}[1]", 0.0, 1.0),
        _bounded_profile_number(value[2], f"{field}[2]", 0.0, 1.0),
    )


def _vision_profile(value: Any) -> StudentVisionProfile:
    if not isinstance(value, Mapping) or set(value) != _VISION_PROFILE_FIELDS:
        raise _profile_error("fields")

    profile_id = value["profile_id"]
    if type(profile_id) is not str or _PROFILE_ID.fullmatch(profile_id) is None:
        raise _profile_error("profile_id")

    resolution = value["resolution"]
    if (
        type(resolution) is not list
        or len(resolution) != 2
        or type(resolution[0]) is not int
        or type(resolution[1]) is not int
        or not 128 <= resolution[0] <= 1024
        or not 128 <= resolution[1] <= 1024
    ):
        raise _profile_error("resolution")

    return StudentVisionProfile(
        profile_id=profile_id,
        resolution=(resolution[0], resolution[1]),
        perspective_angle_deg=_bounded_profile_number(
            value["perspective_angle_deg"],
            "perspective_angle_deg",
            20.0,
            90.0,
        ),
        camera_rig_z_m=_bounded_profile_number(
            value["camera_rig_z_m"],
            "camera_rig_z_m",
            0.50,
            0.90,
        ),
        key_diffuse_rgb=_profile_rgb(
            value["key_diffuse_rgb"],
            "key_diffuse_rgb",
        ),
        fill_diffuse_rgb=_profile_rgb(
            value["fill_diffuse_rgb"],
            "fill_diffuse_rgb",
        ),
    )


class StudentCamera:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def get_profile(self) -> StudentVisionProfile:
        return _vision_profile(self._rpc.call("camera.profile.get"))

    def apply_profile(self, profile_id: str) -> StudentVisionProfile:
        if (
            type(profile_id) is not str
            or _PROFILE_ID.fullmatch(profile_id) is None
        ):
            raise ValueError(
                "profile_id must be a published ASCII identifier"
            )
        return _vision_profile(
            self._rpc.call("camera.profile.apply", profile_id=profile_id)
        )

    def reset_profile(self) -> StudentVisionProfile:
        return _vision_profile(self._rpc.call("camera.profile.reset"))

    def capture(self) -> StudentFrame:
        value = self._rpc.call("camera.capture")
        if not isinstance(value, Mapping):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera.capture")

        snapshot_id = value.get("snapshot_id")
        if (
            type(snapshot_id) is not str
            or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: snapshot_id")

        png_bytes = value.get("png_bytes")
        if (
            type(png_bytes) is not bytes
            or not png_bytes.startswith(_PNG_SIGNATURE)
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera.capture PNG")

        width_value = value.get("width")
        height_value = value.get("height")
        if (
            type(width_value) is not int
            or type(height_value) is not int
            or width_value <= 0
            or height_value <= 0
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera dimensions")

        source = value.get("source")
        if type(source) is not str or source != "coppeliasim":
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera source")
        sequence_id = value.get("sequence_id")
        if type(sequence_id) is not int or sequence_id < 0:
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera sequence")

        encoded = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if (
            image is None
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera.capture PNG")
        height, width = image.shape[:2]
        if width != width_value or height != height_value:
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera dimensions")

        immutable_pixels = image.tobytes(order="C")
        read_only_image = np.frombuffer(
            immutable_pixels,
            dtype=np.uint8,
        ).reshape((height, width, 3))
        return StudentFrame(
            snapshot_id=snapshot_id,
            image_bgr=read_only_image,
            width=width,
            height=height,
            source=source,
            sequence_id=sequence_id,
        )


class StudentExperiment:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def info(self) -> dict[str, Any]:
        value = self._rpc.call("experiment.info")
        if not isinstance(value, Mapping):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: experiment.info")
        if not set(value).issubset(_PUBLIC_EXPERIMENT_FIELDS):
            raise RuntimeError(
                "PROTOCOL_RESPONSE_INVALID: experiment.info public fields"
            )
        required = {
            "experiment_id",
            "public_parameters",
            "hardware_status",
        }
        if not required.issubset(value):
            raise RuntimeError(
                "PROTOCOL_RESPONSE_INVALID: experiment.info fields"
            )
        if (
            type(value.get("experiment_id")) is not str
            or not value["experiment_id"]
            or value.get("hardware_status") != "PENDING_HARDWARE"
            or not isinstance(value.get("public_parameters"), Mapping)
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: experiment.info")
        if "experiment_version" in value and (
            type(value["experiment_version"]) is not str
            or not value["experiment_version"]
        ):
            raise RuntimeError(
                "PROTOCOL_RESPONSE_INVALID: experiment_version"
            )
        if "scene_sha256" in value and (
            type(value["scene_sha256"]) is not str
            or len(value["scene_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value["scene_sha256"]
            )
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: scene_sha256")
        result = _copy_json_native(value, path="experiment.info")
        assert isinstance(result, dict)
        return result


class StudentContext:
    def __init__(self, connection: Any) -> None:
        self._rpc = _Rpc(connection)
        self.robot = StudentRobot(self._rpc)
        self.tool = StudentTool(self._rpc)
        self.camera = StudentCamera(self._rpc)
        self.experiment = StudentExperiment(self._rpc)

    def log(self, message: str) -> None:
        self._rpc.call("context.log", message=str(message))

    def sleep(self, seconds: float) -> None:
        self._rpc.call("context.sleep", seconds=seconds)

    def checkpoint(self, label: str) -> None:
        self._rpc.call("context.checkpoint", label=str(label))
