from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import cv2
import pytest

from tools.vision_lab.generate_ocr_assets import generate


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "simulation" / "vision_ocr_sorting_lab"
MANIFEST = ASSET_ROOT / "ocr_assets_manifest.json"
EXPECTED_TOP_LEVEL = {
    "schema_version",
    "generator",
    "generator_version",
    "seed",
    "alphabet",
    "training_parameters",
    "allowed_scene_ids",
    "training",
    "labels",
}
EXPECTED_GLYPHS = {"1", "2", "A", "B"}
EXPECTED_IDENTIFIERS = {"A1", "A2", "B1", "B2"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_assets(manifest: dict) -> list[dict]:
    return [*manifest["training"], *manifest["labels"]]


def _safe_asset_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    assert not path.is_absolute()
    assert ".." not in path.parts
    assert path.as_posix() == relative
    resolved = (root / path).resolve()
    assert resolved.is_relative_to(root.resolve())
    return resolved


def test_manifest_binds_original_glyphs_and_scene_labels() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest) == EXPECTED_TOP_LEVEL
    assert manifest["schema_version"] == 1
    assert manifest["generator"] == "tools.vision_lab.generate_ocr_assets"
    assert manifest["seed"] == 20260802
    assert manifest["alphabet"] == ["1", "2", "A", "B"]
    assert manifest["allowed_scene_ids"] == ["V1-08"]

    parameters = manifest["training_parameters"]
    assert parameters == {
        "method": "knn",
        "test_fraction": 0.25,
        "seed": 20260802,
        "variants_per_glyph": 12,
        "bitmap_size_px": [5, 7],
        "image_size_px": [64, 96],
        "channels": 3,
    }
    assert {item["glyph"] for item in manifest["training"]} == EXPECTED_GLYPHS
    assert all(
        sum(item["glyph"] == glyph for item in manifest["training"]) >= 8
        for glyph in EXPECTED_GLYPHS
    )
    assert {item["identifier"] for item in manifest["labels"]} == EXPECTED_IDENTIFIERS
    assert len(manifest["labels"]) == 4

    paths: list[str] = []
    hashes: list[str] = []
    for item in _all_assets(manifest):
        assert set(item) == {
            "glyph" if "glyph" in item else "identifier",
            "path",
            "purpose",
            "size_px",
            "channels",
            "sha256",
        }
        path = _safe_asset_path(ASSET_ROOT, item["path"])
        assert path.is_file()
        assert not path.is_symlink()
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image is not None
        assert image.shape == (96, 64, 3)
        assert image.dtype.name == "uint8"
        assert item["size_px"] == [64, 96]
        assert item["channels"] == 3
        assert _sha256(path) == item["sha256"]
        paths.append(item["path"])
        hashes.append(item["sha256"])
    assert len(paths) == len(set(paths))
    assert len(hashes) == len(set(hashes))
    assert [item["path"] for item in manifest["training"]] == sorted(
        item["path"] for item in manifest["training"]
    )
    assert [item["path"] for item in manifest["labels"]] == sorted(
        item["path"] for item in manifest["labels"]
    )


def test_generator_is_byte_reproducible_and_check_mode_is_read_only(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    generate(first_root)
    generate(second_root)

    def files(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert files(first_root) == files(second_root)
    before = files(first_root)
    generate(first_root, check=True)
    assert files(first_root) == before

    changed = first_root / "training" / "A" / "01.png"
    changed.write_bytes(changed.read_bytes()[:-1] + b"0")
    with pytest.raises(RuntimeError, match="drift"):
        generate(first_root, check=True)


def test_generator_check_mode_rejects_missing_and_manifest_tampering(tmp_path: Path) -> None:
    output = tmp_path / "assets"
    generate(output)
    missing = output / "labels" / "A1.png"
    missing.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        generate(output, check=True)

    generate(output)
    manifest_path = output / "ocr_assets_manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b'"seed": 20260802', b'"seed": 9'))
    with pytest.raises(RuntimeError, match="manifest"):
        generate(output, check=True)


def test_generator_source_uses_only_repository_owned_deterministic_inputs() -> None:
    source_path = ROOT / "tools" / "vision_lab" / "generate_ocr_assets.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
    )
    assert imports - {"__future__"} <= {
        "argparse",
        "hashlib",
        "json",
        "pathlib",
        "random",
        "sys",
        "cv2",
        "numpy",
    }
    text = source_path.read_text(encoding="utf-8").lower()
    assert "http://" not in text
    assert "https://" not in text
    assert "socket" not in text
    assert "subprocess" not in text
    assert "torch" not in text
    assert "tensorflow" not in text
    assert "onnx" not in text


def test_formal_assets_are_usable_by_the_existing_ocr_kernel() -> None:
    from vision_platform.vision2d.ocr import recognize_text, train_glyph_classifier

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples: dict[str, list] = {glyph: [] for glyph in manifest["alphabet"]}
    for item in manifest["training"]:
        image = cv2.imread(str(ASSET_ROOT / item["path"]), cv2.IMREAD_COLOR)
        samples[item["glyph"]].append(image)
    model = train_glyph_classifier(samples, method="knn", test_fraction=0.25, seed=20260802)
    assert model.report.held_out_accuracy >= 0.95
    for item in manifest["labels"]:
        image = cv2.imread(str(ASSET_ROOT / item["path"]), cv2.IMREAD_COLOR)
        result = recognize_text(image, model, expected_text=item["identifier"])
        assert result.status == "PASS"
        assert result.text == item["identifier"]
