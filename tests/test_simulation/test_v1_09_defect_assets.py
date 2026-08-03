from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

from tools.vision_lab.generate_v1_09_defect_assets import (
    EXPECTED_ASSETS,
    MANIFEST_NAME,
    generate,
)
from vision_platform.vision2d.defect_detection import detect_surface_defects


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "simulation" / "vision_defect_sorting_lab"
MANIFEST = LAB / MANIFEST_NAME


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_binds_project_original_pngs_and_is_canonical() -> None:
    payload = _load_manifest()
    assert payload["scene_id"] == "V1-09"
    assert payload["seed"] == 20260803
    assert payload["source"] == "project-original-generated"
    assert payload["generator"] == "tools.vision_lab.generate_v1_09_defect_assets"
    assert isinstance(payload["generator_version"], str)

    manifest_bytes = MANIFEST.read_bytes()
    assert b"\r\n" not in manifest_bytes
    canonical = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert manifest_bytes == canonical

    records = payload["assets"]
    assert isinstance(records, list)
    assert [record["asset_id"] for record in records] == list(EXPECTED_ASSETS)
    assert {record["asset_id"]: record["path"] for record in records} == EXPECTED_ASSETS

    required_keys = {
        "asset_id",
        "generator",
        "generator_version",
        "path",
        "purpose",
        "sha256",
        "size_px",
        "source",
    }
    for record in records:
        assert set(record) == required_keys
        relative = Path(record["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = LAB / relative
        assert path.parent == LAB / "assets"
        assert path.is_file()
        assert not path.is_symlink()
        assert record["size_px"] == [256, 256]
        digest = record["sha256"]
        assert isinstance(digest, str)
        assert digest == digest.lower()
        assert len(digest) == 64
        assert digest == _sha256(path)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image is not None
        assert image.shape == (256, 256, 3)
        assert image.dtype.name == "uint8"

    assert sorted(path.name for path in (LAB / "assets").iterdir()) == sorted(
        Path(value).name for value in EXPECTED_ASSETS.values()
    )


def test_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(first)
    generate(second)

    relative_paths = [*EXPECTED_ASSETS.values(), MANIFEST_NAME]
    for relative in relative_paths:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_real_kernel_proves_the_six_candidate_semantics() -> None:
    manifest = _load_manifest()
    paths = {
        record["asset_id"]: LAB / record["path"]
        for record in manifest["assets"]
    }
    reference = cv2.imread(str(paths["reference"]), cv2.IMREAD_COLOR)
    assert reference is not None

    expected = {
        "entry_a": "qualified",
        "entry_b": "missing",
        "entry_c": "hole",
        "entry_d": "foreign",
        "entry_e": "broken",
        "entry_f": "dimension",
    }
    for asset_id, defect_type in expected.items():
        candidate = cv2.imread(str(paths[asset_id]), cv2.IMREAD_COLOR)
        assert candidate is not None
        result = detect_surface_defects(reference, candidate)
        if defect_type == "qualified":
            assert result.status == "PASS"
            assert result.failure_code is None
            assert result.defects == ()
        else:
            assert result.status == "PARTIAL"
            assert result.failure_code == "DEFECTS_FOUND"
            assert [finding.defect_type for finding in result.defects] == [defect_type]
