from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.rgbd_lab.build_rgbd_scene import (
    D1_PORT,
    PROJECT_ROOT,
    validate_build_arguments,
)


def test_builder_is_bound_to_dedicated_spec_and_port() -> None:
    spec = PROJECT_ROOT / "simulation" / "rgbd_lab" / "scene_spec.json"
    assert D1_PORT == 23009
    assert validate_build_arguments(spec_path=spec, port=23009) is None


def test_builder_rejects_other_ports_and_specs() -> None:
    spec = PROJECT_ROOT / "simulation" / "rgbd_lab" / "scene_spec.json"
    with pytest.raises(ValueError, match="23009"):
        validate_build_arguments(spec_path=spec, port=23008)
    with pytest.raises(ValueError, match="dedicated"):
        validate_build_arguments(
            spec_path=PROJECT_ROOT / "simulation" / "vision_lab" / "scene_spec.json",
            port=23009,
        )


def test_builder_spec_is_standalone_and_has_no_runtime_output_root() -> None:
    spec_path = PROJECT_ROOT / "simulation" / "rgbd_lab" / "scene_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert set(spec) >= {"formal_scene_template", "sensor", "required_paths"}
    assert "artifacts" not in spec["output_relative"]
    assert "evidence" not in spec["output_relative"]
