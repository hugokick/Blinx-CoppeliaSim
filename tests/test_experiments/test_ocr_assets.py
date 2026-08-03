from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_platform.experiments.ocr_assets import (
    OcrAssetError,
    OcrTrainingAssets,
    load_ocr_assets,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "simulation/vision_ocr_sorting_lab/ocr_assets_manifest.json"


def test_committed_manifest_loads_hash_bound_training_and_labels() -> None:
    assets = load_ocr_assets(MANIFEST, expected_scene_id="V1-08")
    assert isinstance(assets, OcrTrainingAssets)
    assert assets.training_parameters["method"] == "knn"
    assert assets.training_parameters["seed"] == 20260802
    assert assets.training_parameters["test_fraction"] == 0.25
    assert tuple(assets.samples) == ("1", "2", "A", "B")
    assert all(len(assets.samples[glyph]) >= 8 for glyph in assets.samples)
    assert tuple(assets.labels) == ("A1", "A2", "B1", "B2")
    assert all(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3 for items in assets.samples.values() for image in items)
    assert all(not image.flags.writeable for items in assets.samples.values() for image in items)


def test_training_parameters_are_recursively_immutable() -> None:
    assets = load_ocr_assets(MANIFEST)
    with pytest.raises(TypeError):
        assets.training_parameters["image_size_px"][0] = 128
    with pytest.raises(TypeError):
        assets.training_parameters["nested"] = "not-allowed"


def test_loader_rejects_path_escape_and_duplicate_paths(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    first = manifest["training"][0]
    first["path"] = "../outside.png"
    path = tmp_path / "ocr_assets_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OcrAssetError) as exc:
        load_ocr_assets(path)
    assert exc.value.code == "OCR_ASSET_PATH_INVALID"

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["training"][1]["path"] = manifest["training"][0]["path"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OcrAssetError) as exc:
        load_ocr_assets(path)
    assert exc.value.code == "OCR_ASSET_DUPLICATE"


def test_loader_rejects_tampered_hash_shape_and_non_regular_file(tmp_path: Path) -> None:
    source_root = MANIFEST.parent
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    copied_root = tmp_path / "assets"
    copied_root.mkdir()
    for record in manifest["training"] + manifest["labels"]:
        destination = copied_root / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / record["path"]).read_bytes())
    path = copied_root / "ocr_assets_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    tampered = copied_root / manifest["training"][0]["path"]
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    with pytest.raises(OcrAssetError) as exc:
        load_ocr_assets(path)
    assert exc.value.code == "OCR_ASSET_HASH_MISMATCH"

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    tampered.write_bytes(bytes(encoded))
    record = manifest["training"][0]
    record["sha256"] = hashlib.sha256(bytes(encoded)).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OcrAssetError) as exc:
        load_ocr_assets(path)
    assert exc.value.code == "OCR_ASSET_SHAPE_MISMATCH"


def test_loader_rejects_unknown_scene_and_manifest_parameters(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    path = tmp_path / "ocr_assets_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OcrAssetError) as exc:
        load_ocr_assets(path, expected_scene_id="V1-09")
    assert exc.value.code == "OCR_ASSET_SCENE_MISMATCH"

    manifest["training_parameters"]["method"] = "svm"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OcrAssetError) as exc:
        load_ocr_assets(path)
    assert exc.value.code == "OCR_ASSET_CONFIG_INVALID"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("seed", 20260802.0),
    ],
)
def test_loader_rejects_bool_or_float_schema_and_seed_before_asset_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    import vision_platform.experiments.ocr_assets as module

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest[field] = value
    path = tmp_path / "ocr_assets_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_require_regular_file",
        lambda *_args, **_kwargs: pytest.fail("invalid manifest metadata must fail before asset reads"),
    )
    with pytest.raises(OcrAssetError) as exc:
        load_ocr_assets(path)
    assert exc.value.code == "OCR_ASSET_CONFIG_INVALID"


def test_loader_rejects_symlink_assets(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    copied_root = tmp_path / "assets"
    copied_root.mkdir()
    record = manifest["training"][0]
    target = copied_root / "real.png"
    target.write_bytes((MANIFEST.parent / record["path"]).read_bytes())
    link = copied_root / record["path"]
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable on this Windows worker")
    for item in manifest["training"][1:] + manifest["labels"]:
        destination = copied_root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((MANIFEST.parent / item["path"]).read_bytes())
    path = copied_root / "ocr_assets_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OcrAssetError) as exc:
        load_ocr_assets(path)
    assert exc.value.code == "OCR_ASSET_FILE_INVALID"
