import json
from pathlib import Path

from simulation.vision_lab import verify_scene


ROOT = Path(__file__).resolve().parents[2]
VISION_SCENE = ROOT / "simulation" / "vision_lab"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_runtime_verification_is_bound_to_scene_and_passes_all_gates():
    manifest = _json(VISION_SCENE / "scene_manifest.json")
    report = _json(VISION_SCENE / "runtime_verification.json")

    assert report["scene_sha256"] == manifest["scene"]["sha256"]
    assert report["verdict"] == "PASS"
    assert report["position_only_contract"] is True
    assert report["base_flipped"] is False
    assert set(manifest["required_paths"]) <= set(report["required_paths_found"])
    assert len(report["joint_motion"]) == 6
    assert all(item["follows_joint"] for item in report["joint_motion"])
    assert report["simulation_cycle"]["started"] is True
    assert report["simulation_cycle"]["stopped"] is True
    assert report["camera_frame"]["width"] == 640
    assert report["camera_frame"]["height"] == 480
    assert report["camera_frame"]["sha256"]
    assert set(report["camera_frame"]["visible_teaching_colors"]) == {
        "red",
        "green",
        "blue",
        "yellow",
    }


def test_output_evidence_paths_support_locations_outside_project(tmp_path):
    helper = getattr(verify_scene, "_report_path", None)
    assert callable(helper)

    inside = ROOT / "artifacts" / "vision_lab" / "frame.png"
    outside = tmp_path / "frame.png"

    assert helper(inside) == "artifacts/vision_lab/frame.png"
    assert helper(outside) == outside.resolve().as_posix()
