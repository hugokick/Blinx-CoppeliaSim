from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCENE_MANIFEST = ROOT / "simulation/vision_quality_lab/scene_manifest.json"


def test_quality_scene_binds_verified_v1_06_template_catalog() -> None:
    scene_manifest = json.loads(SCENE_MANIFEST.read_text(encoding="utf-8"))
    binding = scene_manifest["template_catalog"]
    catalog_path = ROOT / binding["path"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert binding["path"] == "simulation/vision_quality_lab/templates/manifest.json"
    assert binding["sha256"] == hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    assert catalog["template_id"] == "v1_06_red_rectangle"
    assert catalog["template_version"] == "1.0.0"
    assert catalog["sha256"] == hashlib.sha256(
        (ROOT / catalog["asset_path"]).read_bytes()
    ).hexdigest()
    assert scene_manifest["scene"]["path"] == (
        "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt"
    )


def test_v1_06_binding_does_not_change_protected_scene_binary() -> None:
    scene = ROOT / "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt"
    assert hashlib.sha256(scene.read_bytes()).hexdigest() == (
        "ce4c0189bb486caa2bfbd557d0b5148775d846a80e07983423f1b29d8d9eb310"
    )

