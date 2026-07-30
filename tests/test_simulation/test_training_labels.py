from __future__ import annotations

import hashlib
import json

import cv2

from simulation.training_scenes.generate_labels import generate_digit_labels


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
