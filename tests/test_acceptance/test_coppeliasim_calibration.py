from __future__ import annotations

import json

import pytest

from vision_platform.acceptance import run_calibration_acceptance


@pytest.mark.coppeliasim
def test_sim_calibration_meets_teaching_error_budget(
    running_vision_scene,
    tmp_path,
):
    report = run_calibration_acceptance(
        running_vision_scene,
        output_dir=tmp_path / "calibration",
    )

    assert report.rms_error_mm <= 3.0
    assert report.max_error_mm <= 5.0
    assert report.fit_marker_count == 3
    assert report.validation_marker_count >= 1
    assert report.calibration_path.is_file()
    assert report.annotated_frame_path.is_file()
    calibration = json.loads(
        report.calibration_path.read_text(encoding="utf-8")
    )
    # The fixed top-down scene contract maps +world X toward -pixel X and
    # +world Y toward +pixel Y.  RMS alone cannot distinguish the symmetric
    # but axis-swapped rectangle correspondence.
    assert calibration["matrix"][0][0] < 0
    assert calibration["matrix"][1][1] > 0
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
