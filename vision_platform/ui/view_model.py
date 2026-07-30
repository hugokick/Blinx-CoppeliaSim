from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Mapping

import numpy as np

from vision_platform.models import Frame, TaskEvent


@dataclass(frozen=True)
class VisionLabSnapshot:
    connected: bool
    camera_backend: str
    robot_backend: str
    state: str
    status_text: str
    frame_bgr: np.ndarray | None
    calibration_status: str
    calibration_rms_error_mm: float | None
    calibration_max_error_mm: float | None
    detections: tuple[Mapping[str, Any], ...]
    pixel_coordinate: tuple[float, float] | None
    world_coordinate_mm: tuple[float, float, float] | None
    acceptance_status: str
    error_code: str | None
    is_running: bool
    is_paused: bool
    logs: tuple[str, ...]


class VisionLabViewModel:
    """Thread-safe state projection shared by PyQt and future simUI views."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._subscribers: list[Callable[[VisionLabSnapshot], None]] = []
        self.connected = False
        self.camera_backend = "sim"
        self.robot_backend = "sim"
        self.state = "DISCONNECTED"
        self.status_text = "未连接：请选择后端并连接实验平台"
        self.frame_bgr: np.ndarray | None = None
        self.calibration_status = "未标定"
        self.calibration_rms_error_mm: float | None = None
        self.calibration_max_error_mm: float | None = None
        self.detections: tuple[Mapping[str, Any], ...] = ()
        self.pixel_coordinate: tuple[float, float] | None = None
        self.world_coordinate_mm: tuple[float, float, float] | None = None
        self.acceptance_status = "未运行"
        self.error_code: str | None = None
        self.is_running = False
        self.is_paused = False
        self.logs: list[str] = []

    def subscribe(
        self,
        callback: Callable[[VisionLabSnapshot], None],
    ) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def snapshot(self) -> VisionLabSnapshot:
        with self._lock:
            frame = (
                self.frame_bgr.copy()
                if self.frame_bgr is not None
                else None
            )
            return VisionLabSnapshot(
                connected=self.connected,
                camera_backend=self.camera_backend,
                robot_backend=self.robot_backend,
                state=self.state,
                status_text=self.status_text,
                frame_bgr=frame,
                calibration_status=self.calibration_status,
                calibration_rms_error_mm=self.calibration_rms_error_mm,
                calibration_max_error_mm=self.calibration_max_error_mm,
                detections=tuple(dict(item) for item in self.detections),
                pixel_coordinate=self.pixel_coordinate,
                world_coordinate_mm=self.world_coordinate_mm,
                acceptance_status=self.acceptance_status,
                error_code=self.error_code,
                is_running=self.is_running,
                is_paused=self.is_paused,
                logs=tuple(self.logs),
            )

    def apply(self, event: TaskEvent) -> None:
        with self._lock:
            self.state = event.state
            self.status_text = event.message
            self.error_code = event.error_code
            self.is_running = event.state not in {"COMPLETE", "SAFE_STOP"}
            if event.state in {"COMPLETE", "SAFE_STOP"}:
                self.is_paused = False
            self.logs.append(f"[{event.state}] {event.message}")
            if event.state == "TRANSFORM":
                pixel = event.data.get("pixel")
                world_xy = event.data.get("world_xy_mm")
                world_point = event.data.get("world_point_mm")
                if pixel is not None:
                    self.pixel_coordinate = (
                        float(pixel[0]),
                        float(pixel[1]),
                    )
                if world_point is not None:
                    self.world_coordinate_mm = (
                        float(world_point[0]),
                        float(world_point[1]),
                        float(world_point[2]),
                    )
                elif world_xy is not None:
                    self.world_coordinate_mm = (
                        float(world_xy[0]),
                        float(world_xy[1]),
                        0.0,
                    )
                detection_id = event.data.get("detection_id")
                if detection_id is not None:
                    row = {
                        "detection_id": str(detection_id),
                        "color": str(event.data.get("color", "")),
                        "shape": str(event.data.get("shape", "")),
                        "confidence": event.data.get("confidence", ""),
                        "center_px": list(pixel or []),
                        "world_mm": list(
                            world_point
                            or (
                                [*world_xy, 0.0]
                                if world_xy is not None
                                else []
                            )
                        ),
                    }
                    existing = [
                        item
                        for item in self.detections
                        if item.get("detection_id") != str(detection_id)
                    ]
                    self.detections = tuple(existing + [row])
        self._notify()

    def set_connection(
        self,
        *,
        connected: bool,
        camera_backend: str,
        robot_backend: str,
        message: str | None = None,
    ) -> None:
        with self._lock:
            self.connected = bool(connected)
            self.camera_backend = str(camera_backend)
            self.robot_backend = str(robot_backend)
            self.state = "READY" if connected else "DISCONNECTED"
            self.status_text = message or (
                "实验平台已连接"
                if connected
                else "未连接：请选择后端并连接实验平台"
            )
            self.error_code = None
            self.logs.append(f"[CONNECTION] {self.status_text}")
        self._notify()

    def set_frame(self, frame: Frame | np.ndarray) -> None:
        image = (
            frame.image_bgr
            if isinstance(frame, Frame)
            else np.asarray(frame)
        )
        with self._lock:
            self.frame_bgr = image.copy()
        self._notify()

    def set_calibration(
        self,
        *,
        status: str,
        rms_error_mm: float | None,
        max_error_mm: float | None,
    ) -> None:
        with self._lock:
            self.calibration_status = str(status)
            self.calibration_rms_error_mm = (
                None if rms_error_mm is None else float(rms_error_mm)
            )
            self.calibration_max_error_mm = (
                None if max_error_mm is None else float(max_error_mm)
            )
            self.logs.append(
                (
                    f"[CALIBRATION] {status}; "
                    f"RMS={self.calibration_rms_error_mm}, "
                    f"MAX={self.calibration_max_error_mm}"
                )
            )
        self._notify()

    def set_detections(self, detections) -> None:
        with self._lock:
            self.detections = tuple(dict(item) for item in detections)
        self._notify()

    def set_running(self, running: bool, *, paused: bool = False) -> None:
        with self._lock:
            self.is_running = bool(running)
            self.is_paused = bool(paused and running)
        self._notify()

    def set_acceptance(self, status: str) -> None:
        with self._lock:
            self.acceptance_status = str(status)
            self.logs.append(f"[ACCEPTANCE] {status}")
        self._notify()

    def set_error(self, code: str, message: str) -> None:
        with self._lock:
            self.error_code = str(code)
            self.state = "ERROR"
            self.status_text = str(message)
            self.is_running = False
            self.is_paused = False
            self.logs.append(f"[ERROR:{code}] {message}")
        self._notify()

    def reset(self) -> None:
        with self._lock:
            self.state = "READY" if self.connected else "DISCONNECTED"
            self.status_text = (
                "实验平台已复位"
                if self.connected
                else "未连接：请选择后端并连接实验平台"
            )
            self.error_code = None
            self.is_running = False
            self.is_paused = False
            self.pixel_coordinate = None
            self.world_coordinate_mm = None
            self.detections = ()
            self.logs.append("[RESET] 教学界面状态已复位")
        self._notify()

    def _notify(self) -> None:
        snapshot = self.snapshot()
        with self._lock:
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            callback(snapshot)
