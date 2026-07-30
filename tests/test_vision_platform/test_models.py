from pathlib import Path

import numpy as np
import pytest

from vision_platform.models import Detection, Frame, TaskEvent, TaskResult


def test_frame_rejects_dimensions_that_do_not_match_image():
    with pytest.raises(ValueError, match="dimensions"):
        Frame(
            image_bgr=np.zeros((10, 20, 3), dtype=np.uint8),
            width=21,
            height=10,
            timestamp_s=1.0,
            source="replay",
            sequence_id=1,
        )


def test_frame_rejects_non_bgr_image_shape():
    with pytest.raises(ValueError, match="BGR"):
        Frame(
            image_bgr=np.zeros((10, 20), dtype=np.uint8),
            width=20,
            height=10,
            timestamp_s=1.0,
            source="replay",
            sequence_id=1,
        )


def test_detection_exposes_pixel_center_as_float_pair():
    detection = Detection(
        detection_id="d1",
        center_px=(12.5, 30.0),
        color="red",
        shape="square",
        angle_deg=0.0,
        area_px=400.0,
        confidence=1.0,
    )
    assert detection.center_px == (12.5, 30.0)


def test_task_result_serializes_without_numpy_or_path_objects():
    result = TaskResult(
        task_id="t1",
        status="PASS",
        events=(
            TaskEvent(
                state="COMPLETE",
                message="done",
                timestamp_s=1.5,
                data={"point": np.array([1.0, 2.0])},
            ),
        ),
        metrics={"count": np.int64(1)},
        artifacts={"report": Path("artifacts/report.json")},
    )

    serialized = result.to_dict()

    assert serialized["metrics"] == {"count": 1}
    assert serialized["events"][0]["data"]["point"] == [1.0, 2.0]
    assert serialized["artifacts"]["report"] == "artifacts/report.json"
