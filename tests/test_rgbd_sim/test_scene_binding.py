from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vision_platform.rgbd_sim.errors import RgbdSimContractError
from vision_platform.rgbd_sim.scene_binding import load_scene_binding


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "simulation" / "rgbd_lab" / "scene_manifest.json"


def _write_manifest(tmp_path: Path, mutate) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    scene = root / "simulation" / "rgbd_lab" / "BL23_rgbd_lab.ttt"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"scene")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["scene"]["path"] = "simulation/rgbd_lab/BL23_rgbd_lab.ttt"
    payload["scene"]["sha256"] = hashlib.sha256(scene.read_bytes()).hexdigest()
    payload["scene"]["size_bytes"] = scene.stat().st_size
    mutate(payload)
    manifest = root / "simulation" / "rgbd_lab" / "scene_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_load_scene_binding_validates_manifest_and_scene_hash() -> None:
    binding = load_scene_binding(MANIFEST, repository_root=ROOT)

    assert binding.sensor_path == "/RgbdLab/CameraRig/RgbdSensor"
    assert binding.resolution == (256, 256)
    assert binding.expected_source_depth_model == "optical_z"
    scene = ROOT / "simulation" / "rgbd_lab" / "BL23_rgbd_lab.ttt"
    assert binding.scene_sha256 == hashlib.sha256(scene.read_bytes()).hexdigest()
    assert set(binding.rois) == {"near_block", "far_block", "step_low", "step_high"}
    assert len(binding.anchors) == 3


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda payload: payload["scene"].update(path="../../outside.ttt"), "RGBD_SIM_BINDING_PATH_INVALID"),
        (lambda payload: payload["validation_rois"].update(extra=[0, 0, 1, 1]), "RGBD_SIM_BINDING_ROI_INVALID"),
        (lambda payload: payload["validation_rois"].update(near_block=[0, 0, 20, 20], far_block=[10, 10, 30, 30]), "RGBD_SIM_BINDING_ROI_INVALID"),
        (lambda payload: payload["validation_rois"].update(near_block=[0, 0, 257, 20]), "RGBD_SIM_BINDING_ROI_INVALID"),
        (lambda payload: payload["scene"].update(sha256="0" * 64), "RGBD_SIM_BINDING_SCENE_INVALID"),
        (lambda payload: payload.update(unexpected=True), "RGBD_SIM_BINDING_MANIFEST_INVALID"),
        (lambda payload: payload["template"].update(unexpected=True), "RGBD_SIM_BINDING_MANIFEST_INVALID"),
        (lambda payload: payload["sensor"].update(path="/RgbdLab/CameraRig/OtherSensor"), "RGBD_SIM_BINDING_SENSOR_INVALID"),
        (lambda payload: payload["probe_anchors"][0]["position_m"].__setitem__(0, float("nan")), "RGBD_SIM_BINDING_ANCHOR_INVALID"),
    ],
)
def test_manifest_security_and_roi_contract(tmp_path: Path, mutate, code: str) -> None:
    manifest = _write_manifest(tmp_path, mutate)

    with pytest.raises(RgbdSimContractError) as exc_info:
        load_scene_binding(manifest, repository_root=tmp_path / "repo")
    assert exc_info.value.code == code


def test_manifest_rejects_symlink_scene(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    manifest = _write_manifest(tmp_path, lambda payload: None)
    scene = root / "simulation" / "rgbd_lab" / "BL23_rgbd_lab.ttt"
    real = root / "real.ttt"
    real.write_bytes(scene.read_bytes())
    scene.unlink()
    try:
        scene.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink privilege unavailable: {exc}")

    with pytest.raises(RgbdSimContractError) as exc_info:
        load_scene_binding(manifest, repository_root=root)
    assert exc_info.value.code == "RGBD_SIM_BINDING_PATH_INVALID"
