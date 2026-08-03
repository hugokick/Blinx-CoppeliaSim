from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT / "simulation" / "vision_defect_sorting_lab"
SPEC_PATH = SCENE_DIR / "scene_spec.json"
PROFILES_PATH = SCENE_DIR / "profiles.json"
MANIFEST_PATH = SCENE_DIR / "scene_manifest.json"
GROUND_TRUTH_PATH = SCENE_DIR / "acceptance_ground_truth.json"

REFERENCE = {"alias": "reference", "asset_id": "reference", "position_mm": [85, -57, 18], "size_mm": [28, 28, 16]}
PART_POSITIONS = {
    "part_a": [140, -16, 18],
    "part_b": [85, -16, 18],
    "part_c": [30, -16, 18],
    "part_d": [140, 38, 18],
    "part_e": [85, 38, 18],
    "part_f": [30, 38, 18],
}
SLOT_POSITIONS = {
    "qualified": [132, -93, 22],
    "missing": [85, -93, 22],
    "hole": [38, -93, 22],
    "foreign": [132, 75, 22],
    "broken": [85, 75, 22],
    "dimension": [38, 75, 22],
}

CAMERA_RIG_POSITION_M = [0.085, 0.0, 0.492]
CALIBRATION_SCALE_MM_PER_PX = 0.16000295944756412
CALIBRATION_MATRIX = [
    [-CALIBRATION_SCALE_MM_PER_PX, 0.0, 166.84151375742906],
    [0.0, CALIBRATION_SCALE_MM_PER_PX, -81.84151375742906],
]


def _project_world_to_pixel(matrix: list[list[float]], position_mm: list[int]) -> tuple[float, float]:
    x_mm, y_mm, _z_mm = position_mm
    u_px = (float(x_mm) - float(matrix[0][2])) / float(matrix[0][0])
    v_px = (float(y_mm) - float(matrix[1][2])) / float(matrix[1][1])
    return u_px, v_px


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


def test_revised_geometry_matches_the_fixed_roi_contract() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["reference"] == REFERENCE
    assert {part["alias"]: part["position_mm"] for part in spec["parts"]} == PART_POSITIONS
    assert {slot["decision"]: slot["position_mm"] for slot in spec["slots"]} == SLOT_POSITIONS
    assert spec["workspace"] == {
        "alias": "Workspace",
        "center_mm": [85, 0, 5],
        "size_mm": [150, 220, 10],
    }


def test_inspection_faces_keep_eight_pixel_roi_margin_and_slots_do_not_overlap() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    positions = {"reference": spec["reference"]["position_mm"]}
    positions.update({part["asset_id"]: part["position_mm"] for part in spec["parts"]})
    scale_x = abs(float(spec["calibration_matrix"][0][0]))
    scale_y = abs(float(spec["calibration_matrix"][1][1]))
    face_half_x_px = 12.0 / scale_x
    face_half_y_px = 12.0 / scale_y
    for entry_id, position in positions.items():
        center_u, center_v = _project_world_to_pixel(spec["calibration_matrix"], position)
        x, y, width, height = spec["rois"][entry_id]
        margins = (
            center_u - face_half_x_px - x,
            x + width - (center_u + face_half_x_px),
            center_v - face_half_y_px - y,
            y + height - (center_v + face_half_y_px),
        )
        assert min(margins) >= 8.0, (entry_id, margins)

    centers = [tuple(slot["position_mm"][:2]) for slot in spec["slots"]]
    for index, (x_mm, y_mm) in enumerate(centers):
        assert 10.0 <= x_mm - 17.0 and x_mm + 17.0 <= 160.0
        assert -110.0 <= y_mm - 17.0 and y_mm + 17.0 <= 110.0
        assert 20.0 <= x_mm <= 155.0 and -95.0 <= y_mm <= 95.0
        for other_x, other_y in centers[index + 1 :]:
            assert abs(x_mm - other_x) >= 34.0 or abs(y_mm - other_y) >= 34.0


def test_rendering_repair_camera_is_bounded_and_recalibrated() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["camera"]["rig_position_m"] == CAMERA_RIG_POSITION_M
    assert spec["camera"]["orientation_deg"] == [180, 0, 0]
    assert abs(0.500 - spec["camera"]["rig_position_m"][2]) <= 0.010

    distance_mm = 492.0 - float(spec["camera"]["surface_face_plane_z_mm"])
    scale = 2.0 * distance_mm * math.tan(math.radians(20.0) / 2.0) / 1024.0
    assert scale == CALIBRATION_SCALE_MM_PER_PX
    assert spec["calibration_matrix"] == CALIBRATION_MATRIX

    positions = {"reference": spec["reference"]["position_mm"]}
    positions.update({part["asset_id"]: part["position_mm"] for part in spec["parts"]})
    face_half_x_px = 12.0 / CALIBRATION_SCALE_MM_PER_PX
    face_half_y_px = 12.0 / CALIBRATION_SCALE_MM_PER_PX
    local_centers: dict[str, tuple[float, float]] = {}
    for entry_id, position in positions.items():
        center_u, center_v = _project_world_to_pixel(spec["calibration_matrix"], position)
        x, y, width, height = spec["rois"][entry_id]
        local_centers[entry_id] = (center_u - x, center_v - y)
        margins = (
            center_u - face_half_x_px - x,
            x + width - (center_u + face_half_x_px),
            center_v - face_half_y_px - y,
            y + height - (center_v + face_half_y_px),
        )
        assert min(margins) >= 8.0, (entry_id, margins)

    reference_local = local_centers["reference"]
    for entry_id in (f"entry_{letter}" for letter in "abcdef"):
        candidate_local = local_centers[entry_id]
        assert abs(candidate_local[0] - reference_local[0]) <= 8.0
        assert abs(candidate_local[1] - reference_local[1]) <= 8.0


def test_v1_09_profile_scene_and_manifest_share_camera_calibration() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    standard = next(
        profile for profile in profiles["profiles"] if profile["profile_id"] == "standard"
    )

    assert standard["camera_rig_z_m"] == 0.492
    assert spec["camera"]["rig_position_m"][2] == standard["camera_rig_z_m"]
    assert manifest["defect_sorting"]["calibration_matrix"] == spec["calibration_matrix"]


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
