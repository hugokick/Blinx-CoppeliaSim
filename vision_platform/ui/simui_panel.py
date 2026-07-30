from __future__ import annotations

import queue
import sys
from typing import Any, Callable

import cv2

from vision_platform.ui.view_model import (
    VisionLabSnapshot,
    VisionLabViewModel,
)


IMAGE_ID = 1
STATUS_ID = 10
CALIBRATION_ID = 11
RECOGNITION_ID = 12
PIXEL_ID = 13
WORLD_ID = 14
ERROR_ID = 15
START_ID = 101
PAUSE_ID = 102
RESUME_ID = 103
RESET_ID = 104
CALIBRATE_ID = 105
EMERGENCY_ID = 106


_PANELS: dict[str, "SimUIPanel"] = {}


def build_simui_xml() -> str:
    """Return XML accepted by the bundled CoppeliaSim simUI plugin."""
    return """
<ui title="BL23 视觉仿真实验" closeable="true" resizable="true"
    placement="relative" position="-20,20" layout="vbox"
    style="* {font-size: 14px;}">
  <group layout="form" flat="false">
    <label text="平台状态"/>
    <label id="10" text="未连接"/>
    <label text="标定"/>
    <label id="11" text="未标定"/>
    <label text="识别结果"/>
    <label id="12" text="—" word-wrap="true"/>
    <label text="像素坐标"/>
    <label id="13" text="—"/>
    <label text="世界坐标"/>
    <label id="14" text="—"/>
    <label text="错误码"/>
    <label id="15" text="—"/>
  </group>
  <image id="1" width="320" height="240" keep-aspect-ratio="true"
         scaled-contents="true"
         style="* {background-color: #111827; border: 1px solid #334155;}"/>
  <group layout="hbox" flat="true">
    <button id="105" text="标定" on-click="_vision_lab_simui_calibrate"
            style="* {min-height: 36px; background-color: #dbeafe;}"/>
    <button id="101" text="启动" on-click="_vision_lab_simui_start"
            style="* {min-height: 36px; background-color: #fed7aa;}"/>
    <button id="102" text="暂停" on-click="_vision_lab_simui_pause"
            style="* {min-height: 36px;}"/>
    <button id="103" text="继续" on-click="_vision_lab_simui_resume"
            style="* {min-height: 36px;}"/>
    <button id="104" text="复位" on-click="_vision_lab_simui_reset"
            style="* {min-height: 36px;}"/>
    <button id="106" text="急停" on-click="_vision_lab_simui_emergency"
            style="* {min-height: 36px; color: white; background-color: #dc2626;}"/>
  </group>
</ui>
""".strip()


def _dispatch_callback(ui_handle: Any, command: str) -> None:
    panel = _PANELS.get(str(ui_handle))
    if panel is not None:
        panel.dispatch_command(command)


def _vision_lab_simui_start(ui_handle, _widget_id):
    _dispatch_callback(ui_handle, "start")


def _vision_lab_simui_pause(ui_handle, _widget_id):
    _dispatch_callback(ui_handle, "pause")


def _vision_lab_simui_resume(ui_handle, _widget_id):
    _dispatch_callback(ui_handle, "resume")


def _vision_lab_simui_reset(ui_handle, _widget_id):
    _dispatch_callback(ui_handle, "reset")


def _vision_lab_simui_calibrate(ui_handle, _widget_id):
    _dispatch_callback(ui_handle, "calibrate")


def _vision_lab_simui_emergency(ui_handle, _widget_id):
    _dispatch_callback(ui_handle, "emergency_stop")


def _register_remote_callbacks() -> None:
    main_globals = sys.modules["__main__"].__dict__
    for callback in (
        _vision_lab_simui_start,
        _vision_lab_simui_pause,
        _vision_lab_simui_resume,
        _vision_lab_simui_reset,
        _vision_lab_simui_calibrate,
        _vision_lab_simui_emergency,
    ):
        main_globals[callback.__name__] = callback


class SimUIPanel:
    """Thin simUI projection; button callbacks only enqueue commands."""

    def __init__(
        self,
        *,
        simui: Any,
        view_model: VisionLabViewModel,
        command_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.simui = simui
        self.view_model = view_model
        self._command_queue: queue.Queue[str] = queue.Queue()
        self.command_sink = command_sink or self._command_queue.put
        self.handle: Any | None = None
        self.created = False
        self.destroyed = False
        self.update_count = 0
        self._dirty = True
        self._unsubscribe = None

    @property
    def is_open(self) -> bool:
        return self.handle is not None

    def open(self) -> Any:
        if self.handle is not None:
            return self.handle
        _register_remote_callbacks()
        handle = self.simui.create(build_simui_xml())
        self.handle = handle
        _PANELS[str(handle)] = self
        self.created = True
        self.destroyed = False
        self._unsubscribe = self.view_model.subscribe(self._mark_dirty)
        self._dirty = True
        return handle

    def refresh(self, *, force: bool = True) -> bool:
        if self.handle is None:
            return False
        if not force and not self._dirty:
            return False
        snapshot = self.view_model.snapshot()
        handle = self.handle
        self.simui.setLabelText(
            handle,
            STATUS_ID,
            snapshot.status_text,
            True,
        )
        self.simui.setLabelText(
            handle,
            CALIBRATION_ID,
            self._calibration_text(snapshot),
            True,
        )
        self.simui.setLabelText(
            handle,
            RECOGNITION_ID,
            self._recognition_text(snapshot),
            True,
        )
        self.simui.setLabelText(
            handle,
            PIXEL_ID,
            self._pixel_text(snapshot),
            True,
        )
        self.simui.setLabelText(
            handle,
            WORLD_ID,
            self._world_text(snapshot),
            True,
        )
        self.simui.setLabelText(
            handle,
            ERROR_ID,
            snapshot.error_code or "—",
            True,
        )
        self.simui.setEnabled(
            handle,
            START_ID,
            not snapshot.is_running,
            True,
        )
        self.simui.setEnabled(
            handle,
            PAUSE_ID,
            snapshot.is_running and not snapshot.is_paused,
            True,
        )
        self.simui.setEnabled(
            handle,
            RESUME_ID,
            snapshot.is_running,
            True,
        )
        if snapshot.frame_bgr is not None:
            thumbnail = cv2.resize(
                snapshot.frame_bgr,
                (320, 240),
                interpolation=cv2.INTER_AREA,
            )
            rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            self.simui.setImageData(
                handle,
                IMAGE_ID,
                rgb.tobytes(),
                320,
                240,
            )
        self.update_count += 1
        self._dirty = False
        return True

    def dispatch_command(self, command: str) -> None:
        # A simUI callback runs while the remote API is servicing an external
        # call.  Only hand off a short command here; the application loop owns
        # all camera, calibration and robot work.
        self.command_sink(str(command))

    def drain_commands(self) -> list[str]:
        commands: list[str] = []
        while True:
            try:
                commands.append(self._command_queue.get_nowait())
            except queue.Empty:
                return commands

    def close(self) -> None:
        handle = self.handle
        if handle is None:
            return
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        _PANELS.pop(str(handle), None)
        self.simui.destroy(handle)
        self.handle = None
        self.destroyed = True

    def _mark_dirty(self, _snapshot: VisionLabSnapshot) -> None:
        self._dirty = True

    @staticmethod
    def _calibration_text(snapshot: VisionLabSnapshot) -> str:
        if snapshot.calibration_rms_error_mm is None:
            return snapshot.calibration_status
        return (
            f"{snapshot.calibration_status}; "
            f"RMS {snapshot.calibration_rms_error_mm:.2f} mm; "
            f"MAX {snapshot.calibration_max_error_mm:.2f} mm"
        )

    @staticmethod
    def _recognition_text(snapshot: VisionLabSnapshot) -> str:
        if not snapshot.detections:
            return "—"
        detection = snapshot.detections[-1]
        return (
            f"{detection.get('detection_id', '')} "
            f"{detection.get('color', '')}/"
            f"{detection.get('shape', '')}"
        ).strip()

    @staticmethod
    def _pixel_text(snapshot: VisionLabSnapshot) -> str:
        if snapshot.pixel_coordinate is None:
            return "—"
        return (
            f"({snapshot.pixel_coordinate[0]:.1f}, "
            f"{snapshot.pixel_coordinate[1]:.1f}) px"
        )

    @staticmethod
    def _world_text(snapshot: VisionLabSnapshot) -> str:
        if snapshot.world_coordinate_mm is None:
            return "—"
        point = snapshot.world_coordinate_mm
        return f"({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}) mm"
