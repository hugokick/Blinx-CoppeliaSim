import numpy as np

from vision_platform.models import Detection, Frame
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.robot.tool import AttachmentEvidence
from vision_platform.tasks.classify import (
    ClassificationTask,
    DropZone,
    PlacementEvidence,
)


class SequencedCamera:
    def __init__(self, count):
        self.count = count
        self.read_calls = 0

    def read(self, timeout_s=1.0):
        self.read_calls += 1
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        return Frame(
            image_bgr=image,
            width=160,
            height=120,
            timestamp_s=float(self.read_calls),
            source="fake-rig",
            sequence_id=self.read_calls - 1,
        )


class SequencedRecognizer:
    def __init__(self):
        self.calls = 0

    def detect(self, _image):
        self.calls += 1
        if self.calls == 1:
            return [
                Detection(
                    detection_id="red-object",
                    center_px=(60, 40),
                    color="red",
                    shape="square",
                    angle_deg=0,
                    area_px=800,
                    confidence=0.98,
                    metadata={"object_handle": 101},
                )
            ]
        return [
            Detection(
                detection_id="blue-object",
                center_px=(80, 60),
                color="blue",
                shape="circle",
                angle_deg=0,
                area_px=900,
                confidence=0.97,
                metadata={"object_handle": 102},
            )
        ]


class Calibration:
    plane_z_mm = 20.0

    def validate_image_size(self, _size):
        return None

    def pixel_to_world(self, pixel):
        return (float(pixel[0]), float(pixel[1]))


class Robot:
    def __init__(self):
        self.moves = []

    def move_world(self, x, y, z, *, speed):
        self.moves.append((float(x), float(y), float(z), speed))

    def move_home(self):
        return None


class Tool:
    def __init__(self):
        self.handle = 100
        self.enabled = False
        self.on_calls = 0
        self.off_calls = 0

    def on(self):
        self.on_calls += 1
        self.handle += 1
        self.enabled = True
        return AttachmentEvidence(attached=True, object_handle=self.handle)

    def off(self):
        self.off_calls += 1
        self.enabled = False

    def is_attached(self):
        return self.enabled


class Verifier:
    def __init__(self):
        self.calls = []

    def verify(self, object_ref, expected_zone):
        self.calls.append((object_ref, expected_zone))
        return PlacementEvidence(True, expected_zone, object_ref)


def test_two_objects_use_fresh_frames_and_complete_the_same_closed_loop():
    camera = SequencedCamera(2)
    recognizer = SequencedRecognizer()
    robot = Robot()
    tool = Tool()
    verifier = Verifier()
    task = ClassificationTask(
        camera=camera,
        recognizer=recognizer,
        calibration=Calibration(),
        robot=robot,
        tool=tool,
        verifier=verifier,
        zones={
            "red": DropZone("red", (110, 50, 20)),
            "blue": DropZone("blue", (120, -40, 20)),
        },
        workspace=WorkspacePolicy(
            x_mm=(20, 140),
            y_mm=(-90, 90),
            z_mm=(10, 140),
            safe_z_mm=100,
        ),
        classification_key="color",
        speed=15,
    )

    result = task.run(max_objects=2)

    assert result.status == "PASS"
    assert result.metrics == {
        "objects_completed": 2,
        "attach_success": 2,
        "release_success": 2,
        "correct_zone": 2,
        "frames_acquired": 2,
    }
    assert camera.read_calls == 2
    assert recognizer.calls == 2
    assert tool.on_calls == tool.off_calls == 2
    assert verifier.calls == [(101, "red"), (102, "blue")]
    positions = [move[:3] for move in robot.moves]
    for previous, current in zip(positions, positions[1:]):
        if previous[:2] != current[:2]:
            assert previous[2] >= 100
            assert current[2] >= 100
    assert positions[-1][2] >= 100


def test_fake_closed_loop_never_transfers_horizontally_below_safe_z():
    camera = SequencedCamera(1)
    robot = Robot()
    task = ClassificationTask(
        camera=camera,
        recognizer=SequencedRecognizer(),
        calibration=Calibration(),
        robot=robot,
        tool=Tool(),
        verifier=Verifier(),
        zones={"red": DropZone("red", (110, 50, 20))},
        workspace=WorkspacePolicy(
            x_mm=(20, 140),
            y_mm=(-90, 90),
            z_mm=(10, 140),
            safe_z_mm=100,
        ),
        classification_key="color",
        speed=15,
    )

    result = task.run(max_objects=1)

    positions = [move[:3] for move in robot.moves]
    assert result.status == "PASS"
    for previous, current in zip(positions, positions[1:]):
        if previous[:2] != current[:2]:
            assert previous[2] >= 100
            assert current[2] >= 100
