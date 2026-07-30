from threading import Event

import numpy as np

from vision_platform.models import Detection, Frame
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.robot.tool import AttachmentEvidence
from vision_platform.tasks.classify import (
    ClassificationTask,
    DropZone,
    PlacementEvidence,
)


EXPECTED = [
    "IDLE",
    "ACQUIRE",
    "DETECT",
    "TRANSFORM",
    "APPROACH",
    "DESCEND",
    "ATTACH",
    "LIFT",
    "TRANSFER",
    "RELEASE",
    "VERIFY",
    "COMPLETE",
]


def _detection(
    detection_id="det-001",
    *,
    color="red",
    shape="square",
    confidence=0.95,
    center=(60.0, 40.0),
):
    return Detection(
        detection_id=detection_id,
        center_px=center,
        color=color,
        shape=shape,
        angle_deg=0,
        area_px=800,
        confidence=confidence,
    )


class FakeCamera:
    def __init__(self):
        self.read_calls = 0

    def read(self, timeout_s=1.0):
        self.read_calls += 1
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        return Frame(
            image_bgr=image,
            width=120,
            height=100,
            timestamp_s=float(self.read_calls),
            source="fake",
            sequence_id=self.read_calls,
        )


class FakeRecognizer:
    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, _image):
        return list(self.detections)


class IdentityCalibration:
    plane_z_mm = 20.0

    def validate_image_size(self, _size):
        return None

    def pixel_to_world(self, center_px):
        return center_px


class FakeRobot:
    def __init__(self):
        self.moves = []
        self.home_calls = 0

    def move_world(self, x, y, z, *, speed):
        self.moves.append((float(x), float(y), float(z), speed))

    def move_home(self):
        self.home_calls += 1


class FakeTool:
    def __init__(self, *, attaches=True):
        self.attaches = attaches
        self.enabled = False
        self.off_calls = 0

    def on(self):
        self.enabled = True
        return AttachmentEvidence(
            attached=self.attaches,
            object_handle=101 if self.attaches else None,
        )

    def off(self):
        self.enabled = False
        self.off_calls += 1

    def is_attached(self):
        return self.enabled and self.attaches


class FakeVerifier:
    def __init__(self, *, ok=True):
        self.ok = ok
        self.calls = []

    def verify(self, object_ref, expected_zone):
        self.calls.append((object_ref, expected_zone))
        return PlacementEvidence(
            ok=self.ok,
            expected_zone=expected_zone,
            object_ref=object_ref,
        )


def _task(*, detections=None, tool_attaches=True, cancel_event=None):
    policy = WorkspacePolicy(
        x_mm=(20, 140),
        y_mm=(-90, 90),
        z_mm=(10, 140),
        safe_z_mm=100,
    )
    tool = FakeTool(attaches=tool_attaches)
    task = ClassificationTask(
        camera=FakeCamera(),
        recognizer=FakeRecognizer(detections or [_detection()]),
        calibration=IdentityCalibration(),
        robot=FakeRobot(),
        tool=tool,
        verifier=FakeVerifier(),
        zones={
            "red": DropZone("red", (110, 50, 20)),
            "square": DropZone("square", (120, -40, 20)),
        },
        workspace=policy,
        classification_key="color",
        speed=15,
        cancel_event=cancel_event,
    )
    return task, tool


def test_successful_classification_emits_complete_state_sequence():
    task, _ = _task()

    result = task.run(max_objects=1)

    assert [event.state for event in result.events] == EXPECTED
    assert result.status == "PASS"
    assert result.metrics["objects_completed"] == 1


def test_attach_failure_turns_tool_off_and_enters_safe_stop():
    task, tool = _task(tool_attaches=False)

    result = task.run(max_objects=1)

    assert result.status == "FAIL"
    assert result.error_code == "SUCTION_ATTACH_FAILED"
    assert tool.off_calls >= 1
    assert result.events[-1].state == "SAFE_STOP"


def test_highest_confidence_detection_is_selected_deterministically():
    task, _ = _task(
        detections=[
            _detection("low", confidence=0.70, center=(40, 30)),
            _detection("high", confidence=0.99, center=(80, 10)),
        ]
    )

    result = task.run(max_objects=1)

    transformed = next(
        event for event in result.events if event.state == "TRANSFORM"
    )
    assert transformed.data["detection_id"] == "high"
    assert transformed.data["world_xy_mm"] == [80.0, 10.0]


def test_cancel_before_acquire_enters_safe_stop_with_stable_code():
    cancel = Event()
    cancel.set()
    task, tool = _task(cancel_event=cancel)

    result = task.run(max_objects=1)

    assert result.status == "CANCELLED"
    assert result.error_code == "TASK_CANCELLED"
    assert result.events[-1].state == "SAFE_STOP"
    assert tool.off_calls >= 1
