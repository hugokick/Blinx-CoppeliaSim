import json

import numpy as np

from vision_platform.calibration.affine import AffineCalibration
from vision_platform.calibration.store import load_calibration, save_calibration


def _calibration():
    return AffineCalibration.fit(
        [(0, 0), (100, 0), (0, 100)],
        [(10, 20), (110, 20), (10, 220)],
        image_size=(640, 480),
        plane_z_mm=30,
        source="sim",
        scene_version="scene-hash",
    )


def test_calibration_json_round_trip(tmp_path):
    path = tmp_path / "calibration.json"

    save_calibration(_calibration(), path)
    loaded = load_calibration(path)

    assert np.allclose(loaded.matrix, _calibration().matrix)
    assert loaded.image_size == (640, 480)
    assert loaded.plane_z_mm == 30
    assert loaded.source == "sim"
    assert loaded.scene_version == "scene-hash"
    assert loaded.pixel_to_world((25, 50)) == (35.0, 120.0)


def test_saved_calibration_has_version_and_points(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(_calibration(), path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert len(payload["pixel_points"]) == 3
    assert len(payload["world_points_mm"]) == 3
    assert payload["matrix"] == [[1.0, 0.0, 10.0], [0.0, 2.0, 20.0]]


def test_validation_sample_count_survives_round_trip(tmp_path):
    calibration = _calibration()
    calibration.evaluate([((25, 50), (35, 120))])
    path = tmp_path / "calibration.json"

    save_calibration(calibration, path)
    loaded = load_calibration(path)

    assert loaded.metrics is not None
    assert loaded.metrics.sample_count == 1
