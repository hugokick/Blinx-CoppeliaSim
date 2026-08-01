from __future__ import annotations

import time
from typing import Any

import numpy as np

from vision_platform.cameras.base import CameraBackend
from vision_platform.coppelia import CoppeliaClientResolver
from vision_platform.errors import (
    CameraUnavailableError,
    FrameFormatError,
    FrameTimeoutError,
)
from vision_platform.models import Frame


class CoppeliaSimCamera(CameraBackend):
    def __init__(
        self,
        *,
        sensor_path: str = "/VisionLab/Camera",
        include_depth: bool = False,
        timeout_s: float = 1.0,
        host: str = "127.0.0.1",
        port: int = 23000,
        sim: Any | None = None,
        client: Any | None = None,
        resolver: CoppeliaClientResolver | None = None,
    ) -> None:
        self.sensor_path = str(sensor_path)
        self.include_depth = bool(include_depth)
        self.default_timeout_s = float(timeout_s)
        self._resolver = resolver or CoppeliaClientResolver(
            sim=sim,
            client=client,
            host=host,
            port=port,
        )
        self._sim: Any | None = None
        self._sensor_handle: int | None = None
        self._sequence_id = 0

    @property
    def is_open(self) -> bool:
        return self._sim is not None and self._sensor_handle is not None

    def open(self) -> None:
        if self.is_open:
            return
        connection = self._resolver.resolve()
        try:
            handle = connection.sim.getObject(self.sensor_path)
        except Exception as error:
            raise CameraUnavailableError(
                f"CoppeliaSim vision sensor does not exist: {self.sensor_path}",
                sensor_path=self.sensor_path,
            ) from error
        self._sim = connection.sim
        self._sensor_handle = int(handle)

    def read(self, timeout_s: float | None = None) -> Frame:
        if not self.is_open:
            self.open()
        assert self._sim is not None
        assert self._sensor_handle is not None
        timeout = self.default_timeout_s if timeout_s is None else float(timeout_s)

        get_explicit = getattr(self._sim, "getExplicitHandling", None)
        if callable(get_explicit) and int(
            get_explicit(self._sensor_handle)
        ) != 0:
            self._sim.handleVisionSensor(self._sensor_handle)
        result = self._sim.getVisionSensorImg(self._sensor_handle)
        try:
            raw, resolution = result
        except (TypeError, ValueError) as error:
            raise FrameFormatError(
                "CoppeliaSim image response must contain buffer and resolution",
                sensor_path=self.sensor_path,
            ) from error
        image_bytes = bytes(raw or b"")
        if not image_bytes:
            raise FrameTimeoutError(
                "CoppeliaSim vision sensor returned an empty frame",
                sensor_path=self.sensor_path,
                timeout_s=timeout,
            )
        width, height = self._validate_resolution(resolution)
        expected = width * height * 3
        if len(image_bytes) != expected:
            raise FrameFormatError(
                f"Vision frame has {len(image_bytes)} bytes; expected {expected}",
                sensor_path=self.sensor_path,
                resolution=[width, height],
                actual_bytes=len(image_bytes),
                expected_bytes=expected,
            )
        rgb_bottom_up = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            height,
            width,
            3,
        )
        image_bgr = np.flipud(rgb_bottom_up)[:, :, ::-1].copy()

        depth_m = None
        if self.include_depth:
            depth_result = self._sim.getVisionSensorDepth(
                self._sensor_handle,
                1,
            )
            depth_buffer = (
                depth_result[0]
                if isinstance(depth_result, (tuple, list))
                and len(depth_result) == 2
                and isinstance(depth_result[0], (bytes, bytearray, memoryview))
                else depth_result
            )
            depth_bytes = bytes(depth_buffer or b"")
            expected_depth_bytes = width * height * 4
            if len(depth_bytes) != expected_depth_bytes:
                raise FrameFormatError(
                    (
                        f"Depth frame has {len(depth_bytes)} bytes; "
                        f"expected {expected_depth_bytes}"
                    ),
                    sensor_path=self.sensor_path,
                    actual_bytes=len(depth_bytes),
                    expected_bytes=expected_depth_bytes,
                )
            depth_bottom_up = np.frombuffer(
                depth_bytes,
                dtype="<f4",
            ).reshape(height, width)
            depth_m = np.flipud(depth_bottom_up).copy()

        frame = Frame(
            image_bgr=image_bgr,
            width=width,
            height=height,
            timestamp_s=time.time(),
            source="coppeliasim",
            sequence_id=self._sequence_id,
            depth_m=depth_m,
            metadata={
                "sensor_path": self.sensor_path,
                "sensor_handle": self._sensor_handle,
                "orientation_normalized": "vertical_flip",
                "color_conversion": "RGB_to_BGR",
            },
        )
        self._sequence_id += 1
        return frame

    def close(self) -> None:
        self._sensor_handle = None
        self._sim = None
        self._resolver.close_owned()

    def _validate_resolution(self, resolution: Any) -> tuple[int, int]:
        if (
            not isinstance(resolution, (tuple, list))
            or len(resolution) != 2
        ):
            raise FrameFormatError(
                f"Invalid vision sensor resolution: {resolution!r}",
                sensor_path=self.sensor_path,
            )
        width, height = int(resolution[0]), int(resolution[1])
        if width <= 0 or height <= 0:
            raise FrameFormatError(
                f"Invalid vision sensor resolution: {[width, height]}",
                sensor_path=self.sensor_path,
            )
        return width, height
