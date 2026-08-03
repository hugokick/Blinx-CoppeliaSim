from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_platform.experiments.defect_assets import (
    DefectAssetError,
    DefectAssets,
    load_defect_assets,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "simulation" / "vision_defect_sorting_lab" / "defect_assets_manifest.json"


def test_committed_defect_assets_load_hash_bound_immutable_images() -> None:
    assets = load_defect_assets(MANIFEST)
    assert isinstance(assets, DefectAssets)
    assert assets.scene_id == "V1-09"
    assert assets.seed == 20260803
    assert tuple(assets.images) == ("reference", "entry_a", "entry_b", "entry_c", "entry_d", "entry_e", "entry_f")
    assert all(image.shape == (256, 256, 3) and image.dtype == np.uint8 for image in assets.images.values())
    assert all(not image.flags.writeable for image in assets.images.values())
    assert len(assets.manifest_sha256) == 64


def test_defect_assets_reject_manifest_path_escape_and_duplicate(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in payload["assets"]:
        destination = tmp_path / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((MANIFEST.parent / record["path"]).read_bytes())
    path = tmp_path / "defect_assets_manifest.json"
    payload["assets"][0]["path"] = "../outside.png"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DefectAssetError) as exc:
        load_defect_assets(path)
    assert exc.value.code == "DEFECT_SORT_ASSET_INVALID"

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["assets"][1]["path"] = payload["assets"][0]["path"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DefectAssetError) as exc:
        load_defect_assets(path)
    assert exc.value.code == "DEFECT_SORT_ASSET_INVALID"


def test_defect_assets_reject_byte_tamper_before_decode(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in payload["assets"]:
        destination = tmp_path / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((MANIFEST.parent / record["path"]).read_bytes())
    tampered = tmp_path / payload["assets"][0]["path"]
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    path = tmp_path / "defect_assets_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DefectAssetError) as exc:
        load_defect_assets(path)
    assert exc.value.code == "DEFECT_SORT_ASSET_INVALID"
