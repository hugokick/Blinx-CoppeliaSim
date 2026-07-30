from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vision_platform.coppelia_scene import stage_scene_for_coppeliasim


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ascii_scene_path_is_used_without_copy(tmp_path):
    scene = tmp_path / "scene.ttt"
    scene.write_bytes(b"scene-data")

    staged = stage_scene_for_coppeliasim(
        scene,
        staging_root=tmp_path / "staging",
    )

    assert staged == scene.resolve()
    assert not (tmp_path / "staging").exists()


def test_unicode_scene_is_copied_to_hash_named_ascii_path(tmp_path):
    scene = tmp_path / "视觉场景" / "机械臂.ttt"
    scene.parent.mkdir()
    scene.write_bytes(b"scene-data")
    staging_root = tmp_path / "ascii-staging"

    staged = stage_scene_for_coppeliasim(
        scene,
        staging_root=staging_root,
    )

    assert staged.parent == staging_root.resolve()
    assert staged.name == f"{_sha256(scene)}.ttt"
    assert str(staged).isascii()
    assert staged.read_bytes() == scene.read_bytes()


def test_unicode_scene_staging_repairs_a_corrupt_cached_copy(tmp_path):
    scene = tmp_path / "视觉场景" / "机械臂.ttt"
    scene.parent.mkdir()
    scene.write_bytes(b"expected-scene")
    staging_root = tmp_path / "ascii-staging"
    staged = stage_scene_for_coppeliasim(
        scene,
        staging_root=staging_root,
    )
    staged.write_bytes(b"corrupt")

    repaired = stage_scene_for_coppeliasim(
        scene,
        staging_root=staging_root,
    )

    assert repaired == staged
    assert repaired.read_bytes() == b"expected-scene"


def test_explicit_non_ascii_staging_root_is_rejected(tmp_path):
    scene = tmp_path / "视觉场景" / "机械臂.ttt"
    scene.parent.mkdir()
    scene.write_bytes(b"scene-data")

    with pytest.raises(ValueError, match="ASCII"):
        stage_scene_for_coppeliasim(
            scene,
            staging_root=tmp_path / "临时",
        )
