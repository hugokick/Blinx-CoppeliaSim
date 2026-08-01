from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest
import vision_platform.vision_quality.catalog as catalog_module

from vision_platform.vision_quality import (
    AppliedVisionProfile,
    VisionProfile,
    load_profile_catalog,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "baseline_profile_id": "standard",
        "sensor_path": "/VisionQualityLab/CameraRig/Camera",
        "camera_rig_path": "/VisionQualityLab/CameraRig",
        "key_light_path": "/VisionQualityLab/Lighting/KeyLight",
        "fill_light_path": "/VisionQualityLab/Lighting/FillLight",
        "near_clip_m": 0.05,
        "far_clip_m": 2.0,
        "profiles": [
            {
                "profile_id": "standard",
                "label": "标准视图",
                "resolution": [512, 512],
                "perspective_angle_deg": 60,
                "camera_rig_z_m": 0.70,
                "key_diffuse_rgb": [0.8, 0.8, 0.8],
                "fill_diffuse_rgb": [0.35, 0.35, 0.35],
            },
            {
                "profile_id": "wide_dim",
                "label": "Wide and dim",
                "resolution": [256, 256],
                "perspective_angle_deg": 75,
                "camera_rig_z_m": 0.80,
                "key_diffuse_rgb": [0.35, 0.35, 0.35],
                "fill_diffuse_rgb": [0.15, 0.15, 0.15],
            },
        ],
    }


def _write_catalog(tmp_path, payload=None):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(_payload() if payload is None else payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_loads_fixed_immutable_catalog(tmp_path):
    catalog = load_profile_catalog(_write_catalog(tmp_path))

    assert catalog.baseline_profile_id == "standard"
    assert catalog.sensor_path == "/VisionQualityLab/CameraRig/Camera"
    assert catalog.camera_rig_path == "/VisionQualityLab/CameraRig"
    assert catalog.key_light_path == "/VisionQualityLab/Lighting/KeyLight"
    assert catalog.fill_light_path == "/VisionQualityLab/Lighting/FillLight"
    assert catalog.near_clip_m == 0.05
    assert catalog.far_clip_m == 2.0
    assert catalog.profile_ids == ("standard", "wide_dim")
    assert catalog.require("wide_dim").resolution == (256, 256)
    assert catalog.require("standard").label == "标准视图"
    assert catalog.require("standard").key_diffuse_rgb == (0.8, 0.8, 0.8)
    with pytest.raises(FrozenInstanceError):
        catalog.near_clip_m = 0.1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"extra": True}), "unexpected fields"),
        (lambda payload: payload.update({"schema_version": 2}), "schema_version"),
        (lambda payload: payload.pop("baseline_profile_id"), "baseline_profile_id"),
        (lambda payload: payload["profiles"][0].update({"resolution": [64, 512]}), "resolution"),
        (lambda payload: payload["profiles"][0].update({"perspective_angle_deg": 91}), "perspective_angle_deg"),
        (lambda payload: payload["profiles"][0].update({"camera_rig_z_m": 1.2}), "camera_rig_z_m"),
        (lambda payload: payload["profiles"][0].update({"key_diffuse_rgb": [1.1, 0.8, 0.8]}), "key_diffuse_rgb"),
        (lambda payload: payload["profiles"].append(payload["profiles"][0].copy()), "duplicate profile_id"),
    ],
)
def test_rejects_unsafe_or_ambiguous_mutations(tmp_path, mutate, message):
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_profile_catalog(_write_catalog(tmp_path, payload))


def test_rejects_changed_sensor_path(tmp_path):
    payload = _payload()
    payload["sensor_path"] = "/student/chosen/path"

    with pytest.raises(ValueError, match="sensor_path"):
        load_profile_catalog(_write_catalog(tmp_path, payload))


def test_rejects_non_object_catalog_root(tmp_path):
    with pytest.raises(ValueError, match="catalog"):
        load_profile_catalog(_write_catalog(tmp_path, []))


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ('"near_clip_m": 0.05', '"near_clip_m": 0.05, "near_clip_m": 0.10', "near_clip_m"),
        ('"label": "标准视图"', '"label": "标准视图", "label": "重复标签"', "label"),
    ],
)
def test_rejects_duplicate_json_object_keys_at_every_level(tmp_path, needle, replacement, message):
    path = tmp_path / "duplicate-keys.json"
    path.write_text(
        json.dumps(_payload(), ensure_ascii=False).replace(needle, replacement, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_profile_catalog(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["profiles"][0].update({"unexpected": 1}), "profile"),
        (lambda payload: payload.update({"schema_version": True}), "schema_version"),
        (lambda payload: payload.update({"near_clip_m": True}), "near_clip_m"),
        (lambda payload: payload.update({"near_clip_m": float("nan")}), "non-finite"),
        (lambda payload: payload.update({"far_clip_m": float("inf")}), "non-finite"),
        (lambda payload: payload.update({"near_clip_m": 2.0, "far_clip_m": 1.0}), "near_clip_m"),
        (lambda payload: payload.update({"profiles": []}), "profiles"),
        (lambda payload: payload.update({"profiles": {}}), "profiles"),
        (lambda payload: payload.update({"baseline_profile_id": "missing"}), "baseline_profile_id"),
        (lambda payload: payload["profiles"][0].update({"profile_id": "Bad-ID"}), "profile_id"),
        (lambda payload: payload["profiles"][0].update({"label": "   "}), "label"),
        (lambda payload: payload["profiles"][0].update({"resolution": [True, 512]}), "resolution"),
        (lambda payload: payload["profiles"][0].update({"perspective_angle_deg": float("nan")}), "non-finite"),
        (lambda payload: payload["profiles"][0].update({"camera_rig_z_m": False}), "camera_rig_z_m"),
        (lambda payload: payload["profiles"][0].update({"key_diffuse_rgb": [True, 0.8, 0.8]}), "key_diffuse_rgb"),
        (lambda payload: payload["profiles"][0].update({"fill_diffuse_rgb": [0.1, 0.2, float("inf")]}), "non-finite"),
    ],
)
def test_rejects_malformed_or_nonfinite_values(tmp_path, mutate, message):
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_profile_catalog(_write_catalog(tmp_path, payload))


def test_tuples_are_isolated_from_caller_mutation_and_require_is_exact(tmp_path):
    payload = _payload()
    catalog = load_profile_catalog(_write_catalog(tmp_path, payload))
    payload["profiles"][0]["resolution"][0] = 128
    payload["profiles"][0]["key_diffuse_rgb"][0] = 0.0

    profile = catalog.require("standard")
    assert profile.resolution == (512, 512)
    assert profile.key_diffuse_rgb == (0.8, 0.8, 0.8)
    with pytest.raises(KeyError, match="unknown vision profile: missing"):
        catalog.require("missing")


def test_accepts_integer_rgb_components_and_copies_them_as_floats(tmp_path):
    payload = _payload()
    payload["profiles"][0]["key_diffuse_rgb"] = [0, 1, 0]

    catalog = load_profile_catalog(_write_catalog(tmp_path, payload))

    assert catalog.require("standard").key_diffuse_rgb == (0.0, 1.0, 0.0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"near_clip_m": 10**400}), "near_clip_m"),
        (lambda payload: payload["profiles"][0].update({"key_diffuse_rgb": [10**400, 0.8, 0.8]}), "key_diffuse_rgb"),
    ],
)
def test_rejects_huge_integer_values_with_stable_value_errors(tmp_path, mutate, message):
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_profile_catalog(_write_catalog(tmp_path, payload))


def test_writes_and_loads_real_utf8_label_text(tmp_path):
    path = _write_catalog(tmp_path)

    assert "标准视图" in path.read_text(encoding="utf-8")
    assert load_profile_catalog(path).require("standard").label == "标准视图"


def test_applied_profile_public_dict_uses_json_native_values():
    profile = VisionProfile(
        "standard", "Standard", (512, 512), 60, 0.7, (0.8, 0.8, 0.8), (0.35, 0.35, 0.35)
    )
    applied = AppliedVisionProfile(
        profile.profile_id,
        profile.resolution,
        profile.perspective_angle_deg,
        profile.camera_rig_z_m,
        profile.key_diffuse_rgb,
        profile.fill_diffuse_rgb,
    )

    assert applied.to_public_dict() == {
        "profile_id": "standard", "resolution": [512, 512],
        "perspective_angle_deg": 60, "camera_rig_z_m": 0.7,
        "key_diffuse_rgb": [0.8, 0.8, 0.8], "fill_diffuse_rgb": [0.35, 0.35, 0.35],
    }


def test_catalog_bytes_loader_matches_path_loader(tmp_path):
    path = _write_catalog(tmp_path)

    from_bytes = catalog_module.load_profile_catalog_bytes(path.read_bytes())

    assert from_bytes == load_profile_catalog(path)
    assert from_bytes.profile_ids == ("standard", "wide_dim")


@pytest.mark.parametrize(
    ("content", "error_type", "message"),
    [
        (b"\xff", UnicodeDecodeError, None),
        (
            b'{"schema_version":1,"schema_version":1}',
            ValueError,
            "schema_version",
        ),
        (b'{"near_clip_m":NaN}', ValueError, "non-finite"),
    ],
)
def test_catalog_bytes_loader_uses_strict_shared_json_parser(
    content,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        catalog_module.load_profile_catalog_bytes(content)


def test_path_loader_reads_exactly_once_then_delegates(tmp_path, monkeypatch):
    path = _write_catalog(tmp_path)
    content = path.read_bytes()
    read_calls = []
    delegated = []
    sentinel = object()

    def read_bytes_once(self):
        read_calls.append(self)
        return content

    def load_bytes(value):
        delegated.append(value)
        return sentinel

    monkeypatch.setattr(Path, "read_bytes", read_bytes_once)
    monkeypatch.setattr(
        catalog_module,
        "load_profile_catalog_bytes",
        load_bytes,
        raising=False,
    )

    assert load_profile_catalog(path) is sentinel
    assert read_calls == [path.resolve()]
    assert delegated == [content]
