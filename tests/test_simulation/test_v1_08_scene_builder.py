from __future__ import annotations

import json
from pathlib import Path

import pytest

import simulation.training_scenes.build_scene as scene_builder
from tools.vision_lab import build_v1_08_scene


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "simulation" / "vision_ocr_sorting_lab" / "scene_spec.json"


def test_python_wrapper_passes_only_canonical_v1_08_spec_and_port(monkeypatch):
    calls = []

    def fake_build_scene(**kwargs):
        calls.append(kwargs)
        return {"status": "PENDING_COPPELIASIM"}

    monkeypatch.setattr(build_v1_08_scene, "build_scene", fake_build_scene)
    assert build_v1_08_scene.main() == 0
    assert calls == [
        {
            "spec_path": SPEC,
            "host": "127.0.0.1",
            "port": 23008,
        }
    ]


def test_python_wrapper_rejects_port_override(monkeypatch):
    source = (ROOT / "tools" / "vision_lab" / "build_v1_08_scene.py").read_text(
        encoding="utf-8"
    )
    assert "23008" in source
    assert "sys.argv" not in source
    assert "--port" not in source


def test_powershell_wrapper_composes_owned_launch_and_cleanup_contract():
    path = ROOT / "tools" / "vision_lab" / "build_v1_08_scene.ps1"
    source = path.read_text(encoding="utf-8")
    assert "launch_coppeliasim.ps1" in source
    assert "process_ownership.ps1" in source
    assert "build_v1_08_scene.py" in source
    assert "23008" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "finally" in source
    assert "-Code" not in source
    assert "Invoke-Expression" not in source
    assert "-Output" not in source


def test_v1_08_spec_is_strictly_validated_before_remote_connection(monkeypatch):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    formal = scene_builder._FORMAL_SCENES[
        "simulation/vision_ocr_sorting_lab/scene_spec.json"
    ]
    scene_builder._validate_spec(spec, formal)

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("remote connection must follow spec validation")

    monkeypatch.setattr(scene_builder, "RemoteAPIClient", ForbiddenClient)
    with pytest.raises(AssertionError, match="remote connection"):
        scene_builder.build_scene(spec_path=SPEC, host="127.0.0.1", port=23008)
