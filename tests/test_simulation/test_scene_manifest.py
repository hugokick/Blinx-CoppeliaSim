import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VISION_SCENE = ROOT / "simulation" / "vision_lab"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_scene_spec_declares_required_paths_and_position_only_contract():
    spec = _json(VISION_SCENE / "scene_spec.json")
    required_paths = {
        "/BLX_base_link",
        "/BLX_joint1",
        "/BLX_joint2",
        "/BLX_joint3",
        "/BLX_joint4",
        "/BLX_joint5",
        "/BLX_joint6",
        "/BLX_tool_suction",
        "/VisionLab/Camera",
        "/VisionLab/Workspace",
        "/VisionLab/Pickables",
        "/VisionLab/Zones",
    }

    assert required_paths <= set(spec["required_paths"])
    assert spec["robot_contract"]["coordinate_mode"] == "position_only"
    assert spec["robot_contract"]["orientation_ignored"] is True
    assert spec["robot_contract"]["base_flipped"] is False
    assert spec["robot_contract"]["visual_body"] == "openr6_reference"
    assert spec["robot_contract"]["end_effector"] == "custom_suction"
    assert spec["robot_visuals"]["bl23_step_runtime_usage"] == "excluded"
    assert spec["robot_visuals"]["scale_corrections"] == {
        "/BLX_link6": 0.1,
    }
    assert spec["robot_visuals"]["teaching_ready_pose"]["joint_angles_deg"] == [
        -59.98,
        -47.68,
        15.26,
        111.09,
        -204.93,
        -96.1,
    ]


def test_scene_has_three_teaching_markers_an_extra_validation_point_and_six_objects():
    spec = _json(VISION_SCENE / "scene_spec.json")

    teaching = [
        marker for marker in spec["calibration_markers"] if marker["role"] == "fit"
    ]
    validation = [
        marker
        for marker in spec["calibration_markers"]
        if marker["role"] == "validation"
    ]
    assert len(teaching) >= 3
    assert len(validation) >= 1
    assert len(spec["objects"]) >= 6
    assert len({item["path"] for item in spec["objects"]}) == len(spec["objects"])
    assert all(
        item["path"].startswith("/VisionLab/Pickables/")
        for item in spec["objects"]
    )
    assert {"red", "green", "blue", "yellow"} <= set(spec["zones"])


def test_generated_scene_manifest_is_bound_to_spec_and_step_hash():
    source = _json(VISION_SCENE / "source_manifest.json")
    scene = _json(VISION_SCENE / "scene_manifest.json")
    scene_path = VISION_SCENE / scene["scene"]["path"]

    assert scene["source_step_sha256"] == source["source_step"]["sha256"]
    assert scene["scene_spec_sha256"]
    assert scene_path.is_file()
    assert scene["scene"]["sha256"]
    assert scene["scene"]["size_bytes"] == scene_path.stat().st_size
    assert scene["protected_assets_unchanged"] is True


def test_scene_records_connected_openr6_reference_body_and_custom_suction():
    scene = _json(VISION_SCENE / "scene_manifest.json")
    visuals = {
        item["segment"]: item for item in scene["robot_visuals"]
    }

    assert scene["robot_visual_strategy"] == {
        "body": "openr6_reference",
        "end_effector": "custom_suction",
        "bl23_step_runtime_usage": "excluded",
    }
    assert scene["reference_assembly"]["archive_sha256"] == (
        "524f968f1ce19dc3fc731d2fd3fbabc83631d1506fae94a938a3725c70c5d174"
    )
    assert scene["teaching_ready_pose"]["tcp_expected_mm"] == [
        100,
        60,
        100,
    ]
    assert scene["teaching_ready_pose"]["applied"] is True
    for segment in ("base", "link1", "link2", "link3", "link4", "link5"):
        assert visuals[segment]["alignment_basis"]["mode"] == (
            "openr6_embedded_reference"
        )
        assert visuals[segment]["visibility_layer"] != 0
        assert visuals[segment]["source_mesh_sha256"]

    assert visuals["link6"]["alignment_basis"] == {
        "mode": "openr6_embedded_reference_with_scale_correction",
        "scale_correction": 0.1,
    }
    assert max(visuals["link6"]["bbox_size_m"]) < 0.05
    assert visuals["link6"]["visibility_layer"] != 0

    for index in range(1, 7):
        visual = visuals[f"link{index}"]
        assert visual["parent_path"].endswith(f"/BLX_joint{index}")
        relative_pose = visual["pose_relative_parent"]
        assert max(abs(value) for value in relative_pose[:3]) < 1e-8
        assert max(abs(value) for value in relative_pose[3:6]) < 1e-8
        assert abs(abs(relative_pose[6]) - 1.0) < 1e-8

    tool = visuals["tool"]
    assert tool["alignment_basis"]["mode"] == "custom_suction_primitives_at_tcp"
    assert tool["tcp_path"] == "/BLX_tool_suction"
    assert tool["original_end_effector_reused"] is False
    assert tool["component_paths"] == [
        "/BLX_tool_suction/SuctionAdapter",
        "/BLX_tool_suction/SuctionStem",
        "/BLX_tool_suction/SuctionCup",
    ]
    assert not any(
        item["path"].startswith("/BL23_visual_")
        for item in scene["robot_visuals"]
    )


def test_runtime_verification_includes_geometry_gates_and_three_robot_views():
    report = _json(VISION_SCENE / "runtime_verification.json")

    assert report["verdict"] == "PASS"
    for gate in (
        "link6_scale_corrected",
        "all_reference_body_visuals_visible",
        "no_bl23_runtime_visuals",
        "custom_suction_parented_to_tcp",
        "teaching_ready_pose_persisted",
        "three_robot_views_non_blank",
    ):
        assert report["gates"][gate] is True

    assert report["assembly_geometry"]["link6_bbox_max_m"] < 0.05
    assert report["assembly_geometry"]["bl23_visual_handles"] == []
    assert len(report["assembly_geometry"]["suction_components"]) == 3
    assert report["teaching_ready_pose"]["max_joint_error_deg"] < 0.01

    views = report["robot_views"]
    assert set(views) == {"iso", "front", "side"}
    for name, view in views.items():
        assert view["path"].startswith(
            "simulation/vision_lab/evidence/robot-views/"
        ), name
        path = ROOT / view["path"]
        assert path.is_file(), name
        assert _sha256(path) == view["sha256"], name
        assert view["non_blank"] is True
