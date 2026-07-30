import json

import cv2
import numpy as np
import pytest

from vision_platform.cameras.replay import ReplayCamera
from vision_platform.errors import CameraUnavailableError, FrameTimeoutError


def _write_image(path, color=(10, 20, 30)):
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :] = color
    assert cv2.imwrite(str(path), image)


def test_replay_camera_returns_bgr_frame_with_monotonic_sequence(tmp_path):
    image_path = tmp_path / "frame.png"
    _write_image(image_path)

    camera = ReplayCamera([image_path], loop=True)
    first = camera.read()
    second = camera.read()

    assert first.image_bgr[0, 0].tolist() == [10, 20, 30]
    assert first.source == "replay"
    assert (first.width, first.height) == (32, 24)
    assert second.sequence_id == first.sequence_id + 1


def test_non_looping_replay_reports_exhaustion(tmp_path):
    image_path = tmp_path / "frame.png"
    _write_image(image_path)
    camera = ReplayCamera([image_path], loop=False)

    camera.read()

    with pytest.raises(FrameTimeoutError, match="exhausted"):
        camera.read()


def test_missing_replay_file_reports_actionable_error(tmp_path):
    camera = ReplayCamera([tmp_path / "missing.png"])

    with pytest.raises(CameraUnavailableError, match="missing.png"):
        camera.read()


def test_manifest_paths_resolve_relative_to_manifest(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    _write_image(frame_dir / "one.png", color=(1, 2, 3))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"frames": [{"path": "frames/one.png", "label": "real"}]}),
        encoding="utf-8",
    )

    camera = ReplayCamera.from_manifest(manifest, loop=False)
    frame = camera.read()

    assert frame.image_bgr[0, 0].tolist() == [1, 2, 3]
    assert frame.metadata["label"] == "real"


def test_context_manager_opens_and_closes_camera(tmp_path):
    image_path = tmp_path / "frame.png"
    _write_image(image_path)
    camera = ReplayCamera([image_path])

    with camera as active:
        assert active.is_open is True
        active.read()

    assert camera.is_open is False


def test_replay_camera_reads_image_from_unicode_windows_path(tmp_path):
    unicode_dir = tmp_path / "实验图片"
    unicode_dir.mkdir()
    image_path = unicode_dir / "红色方块.png"
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :] = (3, 4, 5)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    image_path.write_bytes(encoded.tobytes())

    frame = ReplayCamera([image_path], loop=False).read()

    assert frame.image_bgr[0, 0].tolist() == [3, 4, 5]
