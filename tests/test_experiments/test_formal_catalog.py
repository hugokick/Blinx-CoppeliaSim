from __future__ import annotations

import json
from pathlib import Path


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
    banned = {"grade", "grades", "grading", "score", "scores", "auto_score"}
    for experiment_id in EXPERIMENT_IDS:
        assert banned.isdisjoint(_all_keys(_payload(experiment_id)))
