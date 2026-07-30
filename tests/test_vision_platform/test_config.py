import json
from pathlib import Path

import pytest

from vision_platform.config import load_config


def _write_config(path: Path, **overrides):
    payload = {
        "camera_backend": "replay",
        "robot_backend": "sim",
        "coppeliasim": {
            "root": "E:/CoppeliaSim",
            "host": "127.0.0.1",
            "port": 23000,
            "scene": "simulation/vision_lab/BL23_vision_lab.ttt",
        },
        "camera": {"replay": {"manifest": "config/replay_manifest.json"}},
        "workspace": {
            "x_mm": [20, 140],
            "y_mm": [-90, 90],
            "z_mm": [10, 140],
            "safe_z_mm": 100,
        },
        "calibration": {"plane_z_mm": 20},
        "recognition": {"min_area_ratio": 0.002},
        "task": {"classification_key": "color"},
        "ui": {"preview_width": 640},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_environment_overrides_default_json(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    _write_config(config_file)
    monkeypatch.setenv("VISION_BACKEND", "sim")
    monkeypatch.setenv("ROBOT_BACKEND", "real")
    monkeypatch.setenv("COPPELIA_PORT", "24000")

    cfg = load_config(config_file)

    assert cfg.camera_backend == "sim"
    assert cfg.robot_backend == "real"
    assert cfg.coppelia_port == 24000


def test_relative_scene_and_manifest_paths_resolve_from_project_root(tmp_path):
    config_file = tmp_path / "config.json"
    _write_config(config_file)

    cfg = load_config(config_file, project_root=tmp_path)

    assert cfg.coppelia_scene == (
        tmp_path / "simulation/vision_lab/BL23_vision_lab.ttt"
    ).resolve()
    assert cfg.camera_options["replay"]["manifest"] == str(
        (tmp_path / "config/replay_manifest.json").resolve()
    )


def test_invalid_backend_is_rejected(tmp_path):
    config_file = tmp_path / "config.json"
    _write_config(config_file, camera_backend="usb")

    with pytest.raises(ValueError, match="camera backend"):
        load_config(config_file)


def test_default_config_is_loadable():
    cfg = load_config()
    assert cfg.camera_backend in {"sim", "replay", "hik"}
    assert cfg.robot_backend in {"sim", "real"}
    assert cfg.workspace.safe_z_mm > cfg.workspace.z_mm[0]
