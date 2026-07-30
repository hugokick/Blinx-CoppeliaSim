from __future__ import annotations

import json
from pathlib import Path

from simulation.training_scenes.scene_contract import validate_scene_contract


ROOT = Path(__file__).resolve().parents[2]
SCENES = (
    ROOT / "simulation" / "robot_basics",
    ROOT / "simulation" / "logistics_lab",
)


def test_formal_training_scene_contracts_pass():
    reports = [
        validate_scene_contract(
            directory / "scene_spec.json",
            directory / "scene_manifest.json",
            project_root=ROOT,
        )
        for directory in SCENES
    ]

    assert [item["scene_id"] for item in reports] == [
        "robot-basics",
        "logistics-lab",
    ]
    assert all(item["status"] == "PASS" for item in reports)


def test_training_scenes_have_distinct_outputs_and_roots():
    specs = [
        json.loads(
            (directory / "scene_spec.json").read_text(encoding="utf-8")
        )
        for directory in SCENES
    ]

    assert specs[0]["output"] != specs[1]["output"]
    assert specs[0]["root_path"] == "/RobotBasics"
    assert specs[1]["root_path"] == "/LogisticsLab"
    assert all(
        item["template"] == "simulation/vision_lab/BL23_vision_lab.ttt"
        for item in specs
    )
    assert all(item["output"] != item["template"] for item in specs)
    assert all(item["remove_paths"] == ["/VisionLab"] for item in specs)


def test_logistics_scene_declares_three_isolated_task_groups():
    spec = json.loads(
        (SCENES[1] / "scene_spec.json").read_text(encoding="utf-8")
    )

    assert tuple(spec["tasks"]) == ("Stack", "Digits", "Classes")
    assert len(spec["tasks"]["Stack"]["objects"]) == 6
    assert [item["digit"] for item in spec["tasks"]["Digits"]["objects"]] == [
        1,
        2,
        3,
    ]
    assert len(spec["tasks"]["Classes"]["objects"]) == 4
