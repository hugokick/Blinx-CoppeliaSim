from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "simulation" / "vision_code_routing_lab"


def test_v1_07_scene_release_is_self_consistent() -> None:
    spec = json.loads((LAB / "scene_spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((LAB / "scene_manifest.json").read_text(encoding="utf-8"))
    scene = LAB / "BL23_vision_code_routing_lab.ttt"
    assets = LAB / "code_assets_manifest.json"
    profiles = LAB / "profiles.json"
    assert spec["scene_id"] == manifest["scene_id"] == "vision-code-routing-lab"
    assert manifest["scene"]["path"] == scene.relative_to(ROOT).as_posix()
    assert manifest["scene"]["sha256"] == hashlib.sha256(scene.read_bytes()).hexdigest()
    assert manifest["code_assets_manifest"]["sha256"] == hashlib.sha256(assets.read_bytes()).hexdigest()
    assert manifest["profile_catalog"] == {
        "path": "simulation/vision_code_routing_lab/profiles.json",
        "sha256": hashlib.sha256(profiles.read_bytes()).hexdigest(),
    }
    assert manifest["required_paths"] == spec["required_paths"]
    assert manifest["protected_assets_unchanged"] is True
    assert manifest["reset_contract"] == {"strategy": "scene_reload", "tool_off": True, "robot_home": True}
    assert manifest["code_routing"]["calibration_plane_z_mm"] == 27.4
    assert manifest["code_routing"]["calibration_matrix"] == [
        [-0.1627580685211339, 0.0, 168.25075204856],
        [0.0, 0.1627580685211339, -83.25075204855999],
    ]
    assert manifest["code_routing"]["route_slots_mm"] == {
        "route_red": [[116, -60, 22], [128, -60, 22]],
        "route_blue": [[116, 60, 22], [128, 60, 22]],
    }


def test_scene_parts_bins_and_slots_are_unique_and_bounded() -> None:
    spec = json.loads((LAB / "scene_spec.json").read_text(encoding="utf-8"))
    part_aliases = [part["alias"] for part in spec["parts"]]
    asset_ids = [part["code_asset_id"] for part in spec["parts"]]
    slots = [tuple(slot["position_mm"]) for bin_spec in spec["bins"] for slot in bin_spec["slots"]]
    assert part_aliases == ["part_a", "part_b", "part_c", "part_d"]
    assert len(set(asset_ids)) == 4
    assert len(slots) == len(set(slots)) == 4
    assert all(20 <= x <= 140 and -90 <= y <= 90 and 10 <= z <= 140 for x, y, z in slots)
