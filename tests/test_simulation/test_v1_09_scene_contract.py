from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT / "simulation" / "vision_defect_sorting_lab"
SPEC_PATH = SCENE_DIR / "scene_spec.json"
PROFILES_PATH = SCENE_DIR / "profiles.json"
GROUND_TRUTH_PATH = SCENE_DIR / "acceptance_ground_truth.json"


def test_v1_09_scene_contract_binds_independent_scene_and_fixed_frame() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["scene_id"] == "vision-defect-sorting-lab"
    assert spec["template"] == "simulation/vision_lab/BL23_vision_lab.ttt"
    assert spec["output"] == "simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt"
    assert spec["root_path"] == "/VisionDefectSortingLab"
    assert spec["camera"]["path"] == "/VisionDefectSortingLab/CameraRig/Camera"
    assert spec["profiles"] == "simulation/vision_defect_sorting_lab/profiles.json"
    assert spec["defect_assets_manifest"] == "simulation/vision_defect_sorting_lab/defect_assets_manifest.json"
    assert spec["port"] == 23010
    assert "/VisionDefectSortingLab/CameraRig/Camera" in spec["required_paths"]
    assert "/VisionDefectSortingLab/Slots/slot_dimension" in spec["required_paths"]
    assert "acceptance_ground_truth" not in spec


def test_v1_09_has_six_neutral_parts_and_dedicated_slots() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert [part["alias"] for part in spec["parts"]] == [f"part_{letter}" for letter in "abcdef"]
    assert [part["asset_id"] for part in spec["parts"]] == [f"entry_{letter}" for letter in "abcdef"]
    assert [slot["alias"] for slot in spec["slots"]] == [
        "slot_qualified",
        "slot_missing",
        "slot_hole",
        "slot_foreign",
        "slot_broken",
        "slot_dimension",
    ]
    assert [slot["decision"] for slot in spec["slots"]] == [
        "qualified",
        "missing",
        "hole",
        "foreign",
        "broken",
        "dimension",
    ]
    assert all(part["size_mm"] == [28, 28, 16] for part in spec["parts"])


def test_fixed_rois_are_positive_in_frame_and_non_overlapping() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    rois = spec["rois"]
    assert set(rois) == {"reference", "entry_a", "entry_b", "entry_c", "entry_d", "entry_e", "entry_f"}
    rectangles = []
    for roi in rois.values():
        x, y, width, height = roi
        assert x >= 0 and y >= 0 and width > 0 and height > 0
        assert x + width <= 1024 and y + height <= 1024
        rectangles.append((x, y, width, height))
    for index, (x, y, width, height) in enumerate(rectangles):
        for other_x, other_y, other_width, other_height in rectangles[index + 1 :]:
            assert x + width <= other_x or other_x + other_width <= x or y + height <= other_y or other_y + other_height <= y


def test_profile_is_standard_1024_and_ground_truth_is_acceptance_only() -> None:
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    assert profiles["baseline_profile_id"] == "standard"
    assert profiles["profiles"][0]["profile_id"] == "standard"
    assert profiles["profiles"][0]["resolution"] == [1024, 1024]
    assert ground_truth["scene_id"] == "V1-09"
    assert ground_truth["source"] == "acceptance-only"
    assert ground_truth["decisions"] == {
        "entry_a": "qualified",
        "entry_b": "missing",
        "entry_c": "hole",
        "entry_d": "foreign",
        "entry_e": "broken",
        "entry_f": "dimension",
    }
