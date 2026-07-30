import sys

import cv2
import numpy as np
import pytest

from vision_platform.cameras.factory import create_camera


def test_factory_does_not_import_hikvision_for_replay(monkeypatch, tmp_path):
    image_path = tmp_path / "one.png"
    assert cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
    monkeypatch.setitem(sys.modules, "MvImport.MvCameraControl_class", None)

    camera = create_camera("replay", {"paths": [str(image_path)]})

    assert camera.__class__.__name__ == "ReplayCamera"


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported camera backend"):
        create_camera("usb", {})


def test_factory_builds_replay_from_manifest(tmp_path):
    image_path = tmp_path / "one.png"
    assert cv2.imwrite(str(image_path), np.zeros((10, 10, 3), dtype=np.uint8))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"frames":[{"path":"one.png"}]}',
        encoding="utf-8",
    )

    camera = create_camera(
        "replay",
        {"manifest": str(manifest), "loop": False},
    )

    assert camera.read().source == "replay"
