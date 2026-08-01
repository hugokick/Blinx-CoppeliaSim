from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.vision_quality.models import (
    VisionImageLayer,
    VisionResultBundle,
)
from vision_platform.vision_quality.results import (
    make_result_bundle,
    result_bundle_to_dict,
)


def _image(width: int = 32, height: int = 24) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _bundle(**overrides):
    arguments = {
        "bundle_id": "frame-000001",
        "experiment_id": "V1-01",
        "source_snapshot_id": "frame-000001",
        "status": "PASS",
        "layers": {"raw": ("原图", _image())},
        "result": {"observed": True, "count": 1},
        "profile": {"profile_id": "standard", "resolution": [32, 24]},
    }
    arguments.update(overrides)
    return make_result_bundle(**arguments)


def _records(**overrides):
    records = {
        "raw": {
            "path": "frames/frame-000001.png",
            "sha256": "0" * 64,
            "width": 32,
            "height": 24,
        }
    }
    records.update(overrides)
    return records


def test_bundle_preserves_named_read_only_bgr_layers_and_json_result():
    raw = _image()
    annotated = raw.copy()
    annotated[2:5, 3:8] = (0, 255, 0)

    bundle = _bundle(
        layers={
            "raw": ("原图", raw),
            "annotated": ("标注图", annotated),
        }
    )

    assert tuple(layer.layer_id for layer in bundle.layers) == (
        "raw",
        "annotated",
    )
    assert bundle.layers[0].image_bgr.flags.writeable is False
    with pytest.raises(ValueError):
        bundle.layers[0].image_bgr.setflags(write=True)
    assert bundle.hardware_status == "PENDING_HARDWARE"
    payload = result_bundle_to_dict(
        bundle,
        layer_records={
            "raw": _records()["raw"],
            "annotated": {
                "path": "frames/frame-000001-annotated.png",
                "sha256": "1" * 64,
                "width": 32,
                "height": 24,
            },
        },
    )
    assert payload["result"] == {"observed": True, "count": 1}
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_bundle_rejects_invalid_images(bad_image):
    with pytest.raises((TypeError, ValueError), match="image"):
        _bundle(layers={"raw": ("原图", bad_image)})


@pytest.mark.parametrize(
    "field,value",
    [
        ("result", {"bad": float("nan")}),
        ("result", {"bad": float("inf")}),
        ("result", {"bad": object()}),
        ("result", {1: "not a string key"}),
        ("profile", {"bad": float("-inf")}),
    ],
)
def test_bundle_rejects_non_json_or_nonfinite_mappings(field, value):
    with pytest.raises((TypeError, ValueError)):
        _bundle(**{field: value})


def test_bundle_deep_copies_and_recursively_freezes_caller_values():
    image = _image()
    result = {"nested": [{"count": 1}]}
    profile = {"profile_id": "standard", "resolution": [32, 24]}
    bundle = _bundle(
        layers={"raw": ("原图", image)}, result=result, profile=profile
    )

    image[:] = 255
    result["nested"][0]["count"] = 9
    profile["resolution"][0] = 99

    assert np.count_nonzero(bundle.layers[0].image_bgr) == 0
    assert bundle.result["nested"][0]["count"] == 1
    assert tuple(bundle.profile["resolution"]) == (32, 24)
    with pytest.raises(TypeError):
        bundle.result["new"] = True
    with pytest.raises(TypeError):
        bundle.result["nested"][0]["count"] = 2
    with pytest.raises(FrozenInstanceError):
        bundle.status = "REJECTED"


def test_direct_layer_construction_copies_image_and_validates_replace():
    image = _image()
    layer = VisionImageLayer("raw", "原图", image)
    image[:] = 255

    assert np.count_nonzero(layer.image_bgr) == 0
    assert layer.image_bgr.flags.writeable is False
    with pytest.raises(VisionPlatformError) as exc:
        replace(layer, layer_id="../escape")
    assert exc.value.code == "VISION_RESULT_BUNDLE_INVALID"
    assert isinstance(exc.value.__cause__, (TypeError, ValueError))


def test_direct_bundle_construction_copies_nested_values_and_validates_replace():
    result = {"nested": [{"count": 1}]}
    profile = {"profile_id": "standard", "resolution": [32, 24]}
    layer = VisionImageLayer("raw", "原图", _image())
    bundle = VisionResultBundle(
        1,
        "frame-000001",
        "V1-01",
        "frame-000001",
        "PASS",
        (layer,),
        result,
        profile,
    )
    result["nested"][0]["count"] = 9
    profile["resolution"][0] = 99

    assert bundle.result["nested"][0]["count"] == 1
    assert tuple(bundle.profile["resolution"]) == (32, 24)
    with pytest.raises(TypeError):
        bundle.result["nested"][0]["count"] = 2
    with pytest.raises(VisionPlatformError) as exc:
        replace(bundle, hardware_status="PASS")
    assert exc.value.code == "VISION_RESULT_BUNDLE_INVALID"
    assert isinstance(exc.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize(
    "construct",
    [
        lambda: VisionImageLayer("../raw", "原图", _image()),
        lambda: VisionImageLayer("raw", "", _image()),
        lambda: VisionImageLayer("raw", "原图", np.zeros((2, 2))),
        lambda: VisionResultBundle(
            2,
            "frame-000001",
            "V1-01",
            "frame-000001",
            "PASS",
            (VisionImageLayer("raw", "原图", _image()),),
            {},
            {},
        ),
        lambda: VisionResultBundle(
            1,
            "frame-000001",
            "V1-01",
            "frame-000001",
            "FAILED",
            (VisionImageLayer("raw", "原图", _image()),),
            {},
            {},
        ),
    ],
)
def test_direct_models_reject_invalid_values_with_stable_code(construct):
    with pytest.raises(VisionPlatformError) as exc:
        construct()
    assert exc.value.code == "VISION_RESULT_BUNDLE_INVALID"
    assert isinstance(exc.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize(
    "overrides",
    [
        {"layers": {"raw": ("bad\ud800title", _image())}},
        {"result": {"bad": "value\udfff"}},
        {"result": {"key\ud800": "value"}},
        {"profile": {"bad": "value\ud800"}},
    ],
)
def test_bundle_rejects_non_utf8_surrogates_with_stable_code(overrides):
    with pytest.raises(VisionPlatformError) as exc:
        _bundle(**overrides)
    assert exc.value.code == "VISION_RESULT_BUNDLE_INVALID"
    assert isinstance(exc.value.__cause__, (TypeError, ValueError, UnicodeError))


def test_factory_failure_has_public_bundle_code_and_export():
    with pytest.raises(VisionPlatformError) as exc:
        _bundle(status="FAILED")
    assert exc.value.code == "VISION_RESULT_BUNDLE_INVALID"
    assert isinstance(exc.value.__cause__, (TypeError, ValueError))

    import vision_platform.vision_quality as quality

    assert quality.VisionResultBundleError is type(exc.value)


@pytest.mark.parametrize(
    "status", ["PASS", "PARTIAL", "NO_TARGETS", "REJECTED"]
)
def test_bundle_accepts_only_published_statuses(status):
    assert _bundle(status=status).status == status


@pytest.mark.parametrize("status", ["", "FAILED", "pass", None, True])
def test_bundle_rejects_unpublished_statuses(status):
    with pytest.raises((TypeError, ValueError), match="status"):
        _bundle(status=status)


@pytest.mark.parametrize(
    "field,value",
    [
        ("bundle_id", "../escape"),
        ("bundle_id", "CON"),
        ("experiment_id", "V1 01"),
        ("source_snapshot_id", "frame.000001"),
        ("bundle_id", "x" * 81),
    ],
)
def test_bundle_rejects_suspicious_or_nonportable_ids(field, value):
    with pytest.raises((TypeError, ValueError), match="id"):
        _bundle(**{field: value})


def test_bundle_preserves_published_maximum_id_limits():
    bundle_id = "b" * 80
    layer_id = "l" + ("x" * 39)

    bundle = _bundle(
        bundle_id=bundle_id,
        layers={layer_id: ("最长公开层 ID", _image())},
    )

    assert bundle.bundle_id == bundle_id
    assert bundle.layers[0].layer_id == layer_id


@pytest.mark.parametrize(
    "layer_id", ["Raw", "../raw", "raw.layer", "con", "x" * 41]
)
def test_bundle_rejects_suspicious_layer_ids(layer_id):
    with pytest.raises((TypeError, ValueError), match="layer_id"):
        _bundle(layers={layer_id: ("原图", _image())})


class _DuplicateLayerMapping(Mapping[str, tuple[str, np.ndarray]]):
    def __getitem__(self, key: str) -> tuple[str, np.ndarray]:
        return ("原图", _image())

    def __iter__(self) -> Iterator[str]:
        return iter(("raw", "raw"))

    def __len__(self) -> int:
        return 2

    def items(self):
        return (("raw", self["raw"]), ("raw", self["raw"]))


def test_bundle_rejects_duplicate_layer_ids_from_custom_mapping():
    with pytest.raises(ValueError, match="duplicate layer_id"):
        _bundle(layers=_DuplicateLayerMapping())


@pytest.mark.parametrize("title", ["", "   ", None, 1])
def test_bundle_rejects_invalid_layer_titles(title):
    with pytest.raises((TypeError, ValueError), match="title"):
        _bundle(layers={"raw": (title, _image())})


def test_bundle_rejects_empty_layers_and_cycles():
    with pytest.raises(ValueError, match="layers"):
        _bundle(layers={})
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="cycle"):
        _bundle(result=cyclic)


class _PendingHardwareSubclass(str):
    pass


class _PretendsToBePendingHardware:
    def __eq__(self, other):
        return other == "PENDING_HARDWARE"


@pytest.mark.parametrize(
    "hardware_status",
    [
        "PASS",
        True,
        _PendingHardwareSubclass("PENDING_HARDWARE"),
        _PretendsToBePendingHardware(),
    ],
)
def test_bundle_rejects_non_exact_pending_hardware_status(hardware_status):
    with pytest.raises(ValueError, match="hardware_status"):
        _bundle(hardware_status=hardware_status)


def test_result_serialization_requires_exactly_one_record_per_layer():
    bundle = _bundle(
        layers={
            "raw": ("原图", _image()),
            "annotated": ("标注图", _image()),
        }
    )
    with pytest.raises(ValueError, match="record"):
        result_bundle_to_dict(bundle, layer_records=_records())
    with pytest.raises(ValueError, match="record"):
        result_bundle_to_dict(
            _bundle(),
            layer_records=_records(
                unexpected={
                    "path": "frames/unexpected.png",
                    "sha256": "2" * 64,
                    "width": 32,
                    "height": 24,
                }
            ),
        )


def test_result_serialization_rejects_directly_constructed_invalid_bundle():
    original = _bundle()
    bundle = object.__new__(VisionResultBundle)
    for field, value in original.__dict__.items():
        object.__setattr__(bundle, field, value)
    object.__setattr__(bundle, "schema_version", 2)

    with pytest.raises(VisionPlatformError, match="schema_version") as exc:
        result_bundle_to_dict(bundle, layer_records=_records())
    assert exc.value.code == "VISION_RESULT_BUNDLE_INVALID"


@pytest.mark.parametrize(
    "change",
    [
        {"path": 1},
        {"sha256": "bad"},
        {"width": True},
        {"height": 0},
    ],
)
def test_result_serialization_rejects_invalid_layer_records(change):
    record = dict(_records()["raw"])
    record.update(change)
    with pytest.raises((TypeError, ValueError), match="layer record"):
        result_bundle_to_dict(_bundle(), layer_records={"raw": record})
