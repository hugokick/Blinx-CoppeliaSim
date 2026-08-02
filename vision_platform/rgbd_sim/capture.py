"""One-cycle CoppeliaSim RGB-D source capture."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vision_platform.coppelia import CoppeliaClientResolver

from .errors import RgbdSimContractError
from .models import RgbdSensorMetadata, RgbdSourceCapture


class CoppeliaRgbdCapture:
    """Capture aligned raw BGR and metric source-depth buffers from one sensor cycle."""

    def __init__(
        self,
        *,
        sensor_path: str = "/RgbdLab/CameraRig/RgbdSensor",
        scene_binding: Any,
        sim: Any | None = None,
        client: Any | None = None,
        resolver: CoppeliaClientResolver | None = None,
        host: str = "127.0.0.1",
        port: int = 23009,
    ) -> None:
        if not isinstance(scene_binding, object) or scene_binding is None:
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "scene_binding is required"
            )
        self.sensor_path = str(sensor_path)
        self.scene_binding = scene_binding
        self._resolver = resolver or CoppeliaClientResolver(
            sim=sim, client=client, host=host, port=port
        )
        self._sequence_id = 0

    def read_source(self) -> RgbdSourceCapture:
        try:
            connection = self._resolver.resolve()
            sim = connection.sim
            handle = int(sim.getObject(self.sensor_path))
            explicit = sim.getExplicitHandling(handle)
            if type(explicit) not in {int, float} or int(explicit) != 1:
                raise RgbdSimContractError(
                    "RGBD_SIM_CAPTURE_INVALID",
                    "sensor explicit handling must equal one",
                )
            sim.handleVisionSensor(handle)
            image_result = sim.getVisionSensorImg(handle)
            image_raw, image_resolution = self._split_buffer_result(image_result)
            depth_result = sim.getVisionSensorDepth(handle, 1)
            depth_raw, depth_resolution = self._split_buffer_result(depth_result)
            resolution = self._validate_resolution(image_resolution)
            if depth_resolution is not None and self._validate_resolution(depth_resolution) != resolution:
                raise RgbdSimContractError(
                    "RGBD_SIM_CAPTURE_INVALID", "color and depth resolutions differ"
                )
            reported_resolution = self._read_resolution(sim, handle, resolution)
            if reported_resolution != resolution:
                raise RgbdSimContractError(
                    "RGBD_SIM_CAPTURE_INVALID", "sensor resolution changed during capture"
                )
            image = self._decode_image(image_raw, resolution)
            source_depth = self._decode_depth(depth_raw, resolution)
            metadata = self._read_metadata(sim, handle, resolution)
            return RgbdSourceCapture(metadata, image, source_depth)
        except RgbdSimContractError:
            raise
        except Exception as error:
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", f"CoppeliaSim RGB-D capture failed: {error}"
            ) from error

    def close(self) -> None:
        self._resolver.close_owned()

    @staticmethod
    def _split_buffer_result(result: Any) -> tuple[bytes, list[int] | None]:
        if isinstance(result, (tuple, list)) and len(result) == 2:
            raw, resolution = result
            if not isinstance(resolution, (tuple, list)):
                raise RgbdSimContractError(
                    "RGBD_SIM_CAPTURE_INVALID", "sensor buffer resolution is invalid"
                )
            return CoppeliaRgbdCapture._buffer_bytes(raw), list(resolution)
        return CoppeliaRgbdCapture._buffer_bytes(result), None

    @staticmethod
    def _buffer_bytes(raw: Any) -> bytes:
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "sensor buffer must be bytes-like"
            )
        return bytes(raw)

    @staticmethod
    def _validate_resolution(resolution: Any) -> tuple[int, int]:
        if (
            not isinstance(resolution, (tuple, list))
            or len(resolution) != 2
            or any(type(value) is not int or value <= 0 for value in resolution)
        ):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "sensor resolution is invalid"
            )
        return int(resolution[0]), int(resolution[1])

    @staticmethod
    def _decode_image(raw: bytes, resolution: tuple[int, int]) -> np.ndarray:
        width, height = resolution
        expected = width * height * 3
        if len(raw) != expected:
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "RGB buffer byte count is invalid"
            )
        rgb_bottom_up = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        return np.flipud(rgb_bottom_up)[:, :, ::-1].copy()

    @staticmethod
    def _decode_depth(raw: bytes, resolution: tuple[int, int]) -> np.ndarray:
        width, height = resolution
        expected = width * height * 4
        if len(raw) != expected:
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "depth buffer byte count is invalid"
            )
        depth_bottom_up = np.frombuffer(raw, dtype="<f4").reshape(height, width)
        depth = np.flipud(depth_bottom_up).copy()
        if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "source depth values are invalid"
            )
        return np.asarray(depth, dtype=np.float32)

    def _read_resolution(
        self, sim: Any, handle: int, fallback: tuple[int, int]
    ) -> tuple[int, int]:
        getter = getattr(sim, "getVisionSensorResolution", None)
        if not callable(getter):
            return fallback
        return self._validate_resolution(getter(handle))

    def _read_metadata(
        self, sim: Any, handle: int, resolution: tuple[int, int]
    ) -> RgbdSensorMetadata:
        int_getter = getattr(sim, "getObjectInt32Param", None)
        float_getter = getattr(sim, "getObjectFloatParam", None)
        if not callable(int_getter) or not callable(float_getter):
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "sensor parameter API is unavailable"
            )
        try:
            perspective = int_getter(
                handle, getattr(sim, "visionintparam_perspective_operation")
            )
            rgb_ignored = int_getter(
                handle, getattr(sim, "visionintparam_rgbignored")
            )
            depth_ignored = int_getter(
                handle, getattr(sim, "visionintparam_depthignored")
            )
            angle = float_getter(
                handle, getattr(sim, "visionfloatparam_perspective_angle")
            )
            near = float_getter(handle, getattr(sim, "visionfloatparam_near_clipping"))
            far = float_getter(handle, getattr(sim, "visionfloatparam_far_clipping"))
        except Exception as error:
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", f"sensor parameters could not be read: {error}"
            ) from error
        try:
            return RgbdSensorMetadata(
                sensor_path=self.sensor_path,
                resolution=resolution,
                near_clip_m=near,
                far_clip_m=far,
                perspective_angle_rad=angle,
                scene_path=str(self.scene_binding.scene_path),
                scene_sha256=str(self.scene_binding.scene_sha256),
                sequence_id=self._sequence_id,
                timestamp_s=time.time(),
                expected_source_depth_model=str(
                    self.scene_binding.expected_source_depth_model
                ),
                explicit_handling=True,
                perspective=bool(int(perspective) == 1),
                # Recent CoppeliaSim builds may return ``None`` for the legacy
                # ignored-channel parameters.  The same transaction already
                # obtained both image and depth buffers, so an unsupported
                # flag is treated as enabled while explicit numeric values
                # remain strict.
                rgb_enabled=(
                    True if rgb_ignored is None else bool(int(rgb_ignored) == 0)
                ),
                depth_enabled=(
                    True if depth_ignored is None else bool(int(depth_ignored) == 0)
                ),
            )
        except (AttributeError, RgbdSimContractError) as error:
            if isinstance(error, RgbdSimContractError):
                raise
            raise RgbdSimContractError(
                "RGBD_SIM_CAPTURE_INVALID", "scene binding metadata is incomplete"
            ) from error
        finally:
            self._sequence_id += 1


__all__ = ["CoppeliaRgbdCapture"]
