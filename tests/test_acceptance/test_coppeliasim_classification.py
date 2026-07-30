from __future__ import annotations

import json

import pytest

from vision_platform.acceptance import run_classification_acceptance


@pytest.mark.coppeliasim
def test_six_objects_are_sorted_into_expected_zones(
    running_vision_scene,
    tmp_path,
):
    report = run_classification_acceptance(
        scene=running_vision_scene,
        object_count=6,
        output_dir=tmp_path / "classification",
    )

    result = report.task_result
    assert result.status == "PASS"
    assert result.metrics["objects_completed"] == 6
    assert result.metrics["attach_success"] == 6
    assert result.metrics["release_success"] == 6
    assert result.metrics["correct_zone"] == 6
    assert report.events_path.is_file()
    assert len(report.keyframe_paths) >= 12
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["pending_hardware"]
