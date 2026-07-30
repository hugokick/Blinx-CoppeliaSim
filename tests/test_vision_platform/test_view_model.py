from __future__ import annotations

import numpy as np

from vision_platform.models import Frame, TaskEvent
from vision_platform.ui.view_model import VisionLabViewModel


def _event(state, message, *, error_code=None, data=None):
    return TaskEvent(
        state=state,
        message=message,
        timestamp_s=1.0,
        data=data or {},
        error_code=error_code,
    )


def test_view_model_projects_task_event_to_student_status():
    vm = VisionLabViewModel()

    vm.apply(_event("DETECT", "识别到红色正方形"))

    assert vm.state == "DETECT"
    assert vm.status_text == "识别到红色正方形"
    assert vm.snapshot().logs[-1].endswith("识别到红色正方形")


def test_view_model_projects_error_and_safe_stop_without_color_only_signal():
    vm = VisionLabViewModel()

    vm.apply(
        _event(
            "SAFE_STOP",
            "任务安全停止：目标超出工作区",
            error_code="TARGET_OUT_OF_WORKSPACE",
        )
    )

    snapshot = vm.snapshot()
    assert snapshot.error_code == "TARGET_OUT_OF_WORKSPACE"
    assert snapshot.is_running is False
    assert "安全停止" in snapshot.status_text


def test_view_model_notifies_subscribers_with_immutable_snapshot():
    vm = VisionLabViewModel()
    snapshots = []
    unsubscribe = vm.subscribe(snapshots.append)

    vm.set_connection(
        connected=True,
        camera_backend="sim",
        robot_backend="sim",
    )
    unsubscribe()
    vm.set_calibration(
        status="PASS",
        rms_error_mm=0.25,
        max_error_mm=0.40,
    )

    assert len(snapshots) == 1
    assert snapshots[0].connected is True
    assert snapshots[0].camera_backend == "sim"
    assert vm.snapshot().calibration_status == "PASS"


def test_view_model_frame_is_copied_at_snapshot_boundary():
    vm = VisionLabViewModel()
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    frame = Frame(
        image_bgr=image,
        width=3,
        height=2,
        timestamp_s=1.0,
        source="fake",
        sequence_id=0,
    )

    vm.set_frame(frame)
    snapshot = vm.snapshot()
    image[0, 0] = [1, 2, 3]

    assert snapshot.frame_bgr[0, 0].tolist() == [0, 0, 0]


def test_view_model_adds_detection_row_from_transform_event():
    vm = VisionLabViewModel()

    vm.apply(
        _event(
            "TRANSFORM",
            "像素坐标已转换",
            data={
                "detection_id": "sim-object-42",
                "color": "red",
                "shape": "square",
                "confidence": 0.98,
                "pixel": [320.0, 180.0],
                "world_point_mm": [75.0, -20.0, 20.0],
            },
        )
    )

    row = vm.snapshot().detections[0]
    assert row["detection_id"] == "sim-object-42"
    assert row["color"] == "red"
    assert row["world_mm"] == [75.0, -20.0, 20.0]
