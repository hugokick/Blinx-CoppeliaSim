from __future__ import annotations

import json
from types import SimpleNamespace

from vision_platform.ui.pyqt_app import VisionLabWindow


class DisconnectedApplication:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def test_pyqt_window_constructs_without_camera_or_robot_connection(qtbot):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)

    assert window.camera_backend_combo.count() == 3
    assert window.robot_backend_combo.count() == 2
    assert "未连接" in window.status_label.text()
    assert window.preview_label.minimumWidth() >= 640
    assert window.start_button.minimumHeight() >= 44
    assert window.emergency_button.minimumHeight() >= 44


def test_pyqt_window_exposes_teaching_workflow_controls(qtbot):
    window = VisionLabWindow(application=DisconnectedApplication())
    qtbot.addWidget(window)

    assert window.mode_combo.currentText() == "全自动"
    assert window.calibration_table.columnCount() == 4
    assert window.detection_table.columnCount() == 6
    assert window.pause_button.isEnabled() is False
    assert window.resume_button.isEnabled() is False
    assert "像素坐标" in window.pixel_coordinate_label.text()
    assert "世界坐标" in window.world_coordinate_label.text()


def test_pyqt_window_close_calls_application_close(qtbot):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)

    window.close()

    assert application.close_calls == 1


def test_pyqt_window_populates_calibration_table_from_report(qtbot, tmp_path):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "pixel_points": [[100, 120], [200, 120], [100, 220]],
                "world_points_mm": [[35, -70], [100, -70], [35, 70]],
            }
        ),
        encoding="utf-8",
    )
    report = SimpleNamespace(
        status="PASS",
        rms_error_mm=0.2,
        max_error_mm=0.4,
        calibration_path=calibration_path,
        report_path=tmp_path / "report.json",
    )

    window._action_finished(
        SimpleNamespace(action="calibrate", payload=report, frame=None)
    )

    assert window.calibration_table.item(0, 1).text() == "100.0"
    assert window.calibration_table.item(2, 3).text() == "(35.0, 70.0)"
