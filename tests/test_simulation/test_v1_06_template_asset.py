from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

from tools.vision_lab.generate_v1_06_template import generate


ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "simulation/vision_quality_lab/templates/v1_06_reference.png"
MANIFEST = ROOT / "simulation/vision_quality_lab/templates/manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_template_manifest_and_asset_are_self_consistent() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["template_id"] == "v1_06_red_rectangle"
    assert manifest["template_version"] == "1.0.0"
    assert manifest["method"] == "TM_CCOEFF_NORMED"
    assert manifest["threshold"] == 0.72
    assert manifest["search_roi_px"] == [0, 0, 512, 512]
    assert manifest["asset_path"] == "simulation/vision_quality_lab/templates/v1_06_reference.png"
    assert manifest["source"].startswith("deterministic")
    assert Path(manifest["asset_path"]).is_absolute() is False
    assert _sha256(TEMPLATE) == manifest["sha256"]
    image = cv2.imread(str(TEMPLATE), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape == (40, 67, 3)
    assert image.dtype.name == "uint8"


def test_generator_is_reproducible_without_external_input(tmp_path: Path) -> None:
    first = generate(tmp_path / "first")
    second = generate(tmp_path / "second")

    assert _sha256(first.asset_path) == _sha256(second.asset_path)
    assert first.manifest == second.manifest
    assert first.manifest["source"].startswith("deterministic")
