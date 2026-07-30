from __future__ import annotations

from pathlib import Path

import pytest

from simulation.training_scenes.verify_scene import (
    LOGISTICS_GROUPS,
    verify_training_scene,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.coppeliasim
@pytest.mark.parametrize(
    ("directory", "capture_count"),
    [
        ("simulation/robot_basics", 1),
        ("simulation/logistics_lab", 3),
    ],
)
def test_training_scene_loads_paths_captures_and_reloads(
    directory,
    capture_count,
    tmp_path,
):
    report_path = tmp_path / f"{Path(directory).name}.json"
    report = verify_training_scene(
        ROOT / directory,
        host="127.0.0.1",
        port=23000,
        output=report_path,
    )

    assert report["status"] == "PASS"
    assert report["hardware_status"] == "PENDING_HARDWARE"
    assert report["reload_verified"] is True
    assert len(report["required_paths"]) == report["contract"][
        "required_path_count"
    ]
    assert len(report["captures"]) == capture_count
    assert report_path.is_file()
    for capture in report["captures"]:
        image = report_path.parent / capture["path"]
        assert image.is_file()
        assert image.stat().st_size > 0
        assert capture["size"] == [640, 480]
        assert len(capture["sha256"]) == 64

    if directory.endswith("logistics_lab"):
        assert [item["group"] for item in report["captures"]] == list(
            LOGISTICS_GROUPS
        )
        assert len({item["sha256"] for item in report["captures"]}) == 3
        for capture in report["captures"]:
            positions = capture["group_positions_m"]
            assert abs(positions[capture["group"]][2]) < 1e-6
            assert all(
                position[2] <= -2.0
                for group, position in positions.items()
                if group != capture["group"]
            )
