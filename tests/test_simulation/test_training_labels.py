from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import pytest

import simulation.training_scenes.generate_labels as labels_module
from simulation.training_scenes.generate_labels import generate_digit_labels


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_LABEL_ROOT = (
    PROJECT_ROOT / "simulation" / "logistics_lab" / "assets" / "labels"
)


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _manifest_matches_current_files(root: Path) -> bool:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            return False
    return True


def _staging_paths(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.name.startswith((".label-stage-", ".manifest-stage-"))
    ]


def test_digit_labels_are_deterministic_and_manifested(tmp_path):
    first = generate_digit_labels(tmp_path)
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((tmp_path / "digits").glob("*.png"))
    }
    second = generate_digit_labels(tmp_path)
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((tmp_path / "digits").glob("*.png"))
    }

    assert first_hashes == second_hashes
    assert sorted(first_hashes) == ["1.png", "2.png", "3.png"]
    assert first["generator"] == "simulation.training_scenes.generate_labels"
    assert first["license"] == "project-original-generated"
    assert first == second
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == first
    assert {
        item["path"]: item["sha256"] for item in manifest["files"]
    } == {
        f"digits/{name}": digest for name, digest in first_hashes.items()
    }
    for path in (tmp_path / "digits").glob("*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert image.shape == (96, 64)
        assert image.min() == 0
        assert image.max() == 255


def test_formal_digit_labels_are_byte_reproducible(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first_manifest = generate_digit_labels(first_root)
    second_manifest = generate_digit_labels(second_root)
    formal_bytes = _file_bytes(FORMAL_LABEL_ROOT)
    first_bytes = _file_bytes(first_root)
    second_bytes = _file_bytes(second_root)

    expected_paths = {
        "digits/1.png",
        "digits/2.png",
        "digits/3.png",
        "manifest.json",
    }
    assert set(formal_bytes) == expected_paths
    assert first_bytes == second_bytes == formal_bytes
    assert {
        path: hashlib.sha256(payload).hexdigest()
        for path, payload in first_bytes.items()
    } == {
        path: hashlib.sha256(payload).hexdigest()
        for path, payload in formal_bytes.items()
    }
    assert first_manifest == second_manifest
    assert first_manifest["generator"] == (
        "simulation.training_scenes.generate_labels"
    )
    assert first_manifest["license"] == "project-original-generated"
    for name in ("1.png", "2.png", "3.png"):
        image = cv2.imread(
            str(FORMAL_LABEL_ROOT / "digits" / name),
            cv2.IMREAD_GRAYSCALE,
        )
        assert image is not None
        assert image.shape == (96, 64)
        assert image.min() == 0
        assert image.max() == 255


def test_staging_write_failure_preserves_last_good_labels(
    tmp_path,
    monkeypatch,
):
    generate_digit_labels(tmp_path)
    before = _file_bytes(tmp_path)
    original_imwrite = labels_module.cv2.imwrite
    call_count = 0

    def partially_write_then_fail(path, image):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            Path(path).write_bytes(b"partial-stage")
            return False
        return original_imwrite(path, image)

    monkeypatch.setattr(
        labels_module.cv2,
        "imwrite",
        partially_write_then_fail,
    )

    with pytest.raises(RuntimeError):
        generate_digit_labels(tmp_path)

    assert _file_bytes(tmp_path) == before
    assert _manifest_matches_current_files(tmp_path)
    assert _staging_paths(tmp_path) == []


def test_publish_failure_never_leaves_a_stale_manifest(
    tmp_path,
    monkeypatch,
):
    generate_digit_labels(tmp_path)
    second_target = (tmp_path / "digits" / "2.png").resolve()
    original_replace = Path.replace

    def fail_second_publish(path, target):
        if Path(target).resolve() == second_target:
            raise OSError("simulated label publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_publish)

    with pytest.raises(RuntimeError):
        generate_digit_labels(tmp_path)

    manifest_path = tmp_path / "manifest.json"
    assert (
        not manifest_path.exists()
        or _manifest_matches_current_files(tmp_path)
    )
    assert all(
        (tmp_path / "digits" / f"{value}.png").is_file()
        for value in (1, 2, 3)
    )
    assert _staging_paths(tmp_path) == []


def test_generation_preserves_undeclared_files(tmp_path):
    stale = tmp_path / "digits" / "teacher-note.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"keep me")

    generate_digit_labels(tmp_path)

    assert stale.read_bytes() == b"keep me"
