from __future__ import annotations

from typing import Any, Mapping


class VisionPlatformError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class CameraUnavailableError(VisionPlatformError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("CAMERA_UNAVAILABLE", message, details=details)


class FrameTimeoutError(VisionPlatformError):
    def __init__(self, message: str = "Camera frame timed out", **details: Any) -> None:
        super().__init__("FRAME_TIMEOUT", message, details=details)


class FrameFormatError(VisionPlatformError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("FRAME_INVALID", message, details=details)


class CalibrationError(VisionPlatformError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "CALIBRATION_INVALID",
        **details: Any,
    ) -> None:
        super().__init__(code, message, details=details)


class DetectionError(VisionPlatformError):
    pass


class MotionSafetyError(VisionPlatformError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("TARGET_OUT_OF_WORKSPACE", message, details=details)


class SuctionError(VisionPlatformError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("SUCTION_ATTACH_FAILED", message, details=details)


class TaskCancelledError(VisionPlatformError):
    def __init__(self, message: str = "Task cancelled") -> None:
        super().__init__("TASK_CANCELLED", message)
