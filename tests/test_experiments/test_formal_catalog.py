from __future__ import annotations

import json
from pathlib import Path

from vision_platform.experiments.catalog import ExperimentCatalog


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config" / "experiments"
EXPERIMENT_IDS = ("R1-01", "R1-02", "R1-05", "R1-06", "R1-07")


def _payload(experiment_id):
    return json.loads(
        (CONFIG_DIR / f"{experiment_id}.json").read_text(encoding="utf-8")
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_keys(nested)


def test_first_batch_catalog_has_exact_order_and_hardware_boundary():
    catalog = json.loads(
        (CONFIG_DIR / "catalog.json").read_text(encoding="utf-8")
    )
    names = catalog["experiments"]
    payloads = [_payload(Path(name).stem) for name in names]

    assert names == [f"{experiment_id}.json" for experiment_id in EXPERIMENT_IDS]
    assert [item["experiment_id"] for item in payloads] == list(EXPERIMENT_IDS)
    assert all(
        item["hardware_status"] == "PENDING_HARDWARE" for item in payloads
    )


def test_first_batch_experiments_use_two_independent_scenes():
    scenes = {
        experiment_id: _payload(experiment_id)["scene"]
        for experiment_id in EXPERIMENT_IDS
    }

    assert scenes["R1-01"] == "simulation/robot_basics/BL23_robot_basics.ttt"
    assert scenes["R1-02"] == scenes["R1-01"]
    assert scenes["R1-05"] == "simulation/logistics_lab/BL23_logistics_lab.ttt"
    assert scenes["R1-06"] == scenes["R1-05"]
    assert scenes["R1-07"] == scenes["R1-05"]
    assert all("BL23_vision_lab.ttt" not in value for value in scenes.values())


def test_first_batch_declares_required_capabilities_without_unknown_or_dangerous_ones():
    known = {
        "robot.home",
        "robot.pose",
        "robot.move_world",
        "tool.suction",
        "camera.rgb",
        "experiment.info",
        "scene.probe",
    }
    dangerous = {
        "remote_api.client",
        "sim.raw",
        "robot.real",
        "camera.hikvision_sdk",
        "system.command",
        "subprocess.run",
    }
    motion = {
        "robot.home",
        "robot.pose",
        "robot.move_world",
        "experiment.info",
        "scene.probe",
    }
    vision = motion | {"tool.suction", "camera.rgb"}
    required = {
        "R1-01": motion,
        "R1-02": motion,
        "R1-05": vision,
        "R1-06": vision,
        "R1-07": vision,
    }

    for experiment_id, required_capabilities in required.items():
        declared = set(_payload(experiment_id)["capabilities"])
        assert required_capabilities <= declared
        assert declared <= known
        assert declared.isdisjoint(dangerous)


def test_first_batch_task_parameters_match_course_contracts():
    stack = _payload("R1-05")["public_parameters"]["stack_slots_mm"]
    assert len(stack) == 6
    assert {point[1] for point in stack} == {-45, 45}
    assert {point[2] for point in stack} == {20, 38, 56}
    assert len({(point[1], point[2]) for point in stack}) == 6

    digits = _payload("R1-06")["public_parameters"]
    assert digits["order"] == [1, 2, 3]
    assert digits["reverse_order"] == [3, 2, 1]

    classes = _payload("R1-07")
    class_ids = classes["public_parameters"]["classes"]
    assert len(class_ids) >= 4
    assert len(set(class_ids)) == len(class_ids)
    assert set(class_ids) == set(classes["public_parameters"]["drop_poses_mm"])
    assert "水果" not in classes["title"]


def test_formal_catalog_loads_through_strict_runtime_parser():
    catalog = ExperimentCatalog.load(
        CONFIG_DIR / "catalog.json",
        project_root=ROOT,
    )

    assert catalog.ids == EXPERIMENT_IDS


def test_each_experiment_declares_existing_assets_and_acceptance_checks():
    for experiment_id in EXPERIMENT_IDS:
        experiment = _payload(experiment_id)
        assert experiment["acceptance"]["automated_checks"]
        assert experiment["acceptance"]["human_checks"]
        assert "camera.rgb" in experiment["capabilities"] or experiment_id in {
            "R1-01",
            "R1-02",
        }
        for field in ("scene", "scene_manifest", "student_template", "guide"):
            assert (ROOT / experiment[field]).is_file()


def test_catalog_contains_no_grading_or_score_fields():
    banned_exact = {"grade", "grades", "grading", "rubric", "points", "marks"}
    banned_fragments = ("score", "评分", "分数")
    for experiment_id in EXPERIMENT_IDS:
        keys = (str(key).lower() for key in _all_keys(_payload(experiment_id)))
        assert all(
            key not in banned_exact
            and not any(fragment in key for fragment in banned_fragments)
            for key in keys
        )


def test_retained_files_contains_first_batch_scene_curriculum_assets():
    retained = {
        line.strip()
        for line in (ROOT / "RETAINED_FILES.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }
    required = {
        "config/experiments/R1-01.json",
        "config/experiments/R1-02.json",
        "config/experiments/R1-05.json",
        "config/experiments/R1-06.json",
        "config/experiments/R1-07.json",
        "config/experiments/catalog.json",
        "docs/experiments/R1-01.md",
        "docs/experiments/R1-02.md",
        "docs/experiments/R1-05.md",
        "docs/experiments/R1-06.md",
        "docs/experiments/R1-07.md",
        "simulation/logistics_lab/BL23_logistics_lab.ttt",
        "simulation/logistics_lab/__init__.py",
        "simulation/logistics_lab/assets/labels/digits/1.png",
        "simulation/logistics_lab/assets/labels/digits/2.png",
        "simulation/logistics_lab/assets/labels/digits/3.png",
        "simulation/logistics_lab/assets/labels/manifest.json",
        "simulation/logistics_lab/scene_manifest.json",
        "simulation/logistics_lab/scene_spec.json",
        "simulation/robot_basics/BL23_robot_basics.ttt",
        "simulation/robot_basics/__init__.py",
        "simulation/robot_basics/scene_manifest.json",
        "simulation/robot_basics/scene_spec.json",
        "simulation/training_scenes/__init__.py",
        "simulation/training_scenes/build_scene.py",
        "simulation/training_scenes/generate_labels.py",
        "simulation/training_scenes/scene_contract.py",
        "simulation/training_scenes/verify_scene.py",
        "student_programs/__init__.py",
        "student_programs/templates/__init__.py",
        "student_programs/templates/r1_01_robot_basics.py",
        "student_programs/templates/r1_02_teach_points.py",
        "student_programs/templates/r1_05_visual_stacking.py",
        "student_programs/templates/r1_06_digit_sort.py",
        "student_programs/templates/r1_07_component_sort.py",
        "student_programs/templates/r1_common.py",
        "tests/test_acceptance/test_coppeliasim_training_scenes.py",
        "tests/test_acceptance/test_experiment_guides.py",
        "tests/test_experiments/test_formal_catalog.py",
        "tests/test_experiments/test_student_templates.py",
        "tests/test_simulation/test_formal_training_scenes.py",
        "tests/test_simulation/test_training_labels.py",
        "tests/test_simulation/test_training_scene_contract.py",
    }

    assert required <= retained
    assert all((ROOT / path).is_file() for path in required)
