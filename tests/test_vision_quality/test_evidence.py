from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import vision_platform.vision_quality.evidence as evidence_module
from vision_platform.errors import VisionPlatformError
from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.vision_quality.models import (
    VisionImageLayer,
    VisionResultBundle,
)
from vision_platform.vision_quality.evidence import (
    load_recorded_bundle,
    record_vision_bundle,
)
from vision_platform.vision_quality.results import (
    make_result_bundle,
    result_bundle_to_dict,
)


def _evidence(tmp_path) -> StudentRunEvidence:
    program = tmp_path / "student.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    return StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="vision-run",
    )


def _image(width: int = 32, height: int = 24, value: int = 0):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _bundle(*, layers=None, bundle_id="frame-000001"):
    return make_result_bundle(
        bundle_id=bundle_id,
        experiment_id="V1-01",
        source_snapshot_id="frame-000001",
        status="PASS",
        layers=layers or {"raw": ("原图", _image())},
        result={"capture": "ok"},
        profile={"profile_id": "standard", "resolution": [32, 24]},
    )


def _raw_record(evidence, image=None):
    selected = _image() if image is None else image
    return evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=_png(selected),
        metadata={"width": 32, "height": 24, "source": "coppeliasim"},
    )


def _record_one(tmp_path):
    evidence = _evidence(tmp_path)
    raw_record = _raw_record(evidence)
    artifact = record_vision_bundle(
        evidence,
        _bundle(),
        existing_layer_records={"raw": raw_record},
    )
    return evidence, artifact


def _record_two(tmp_path):
    evidence = _evidence(tmp_path)
    raw = _image()
    raw_record = _raw_record(evidence, raw)
    artifact = record_vision_bundle(
        evidence,
        _bundle(
            layers={
                "raw": ("原图", raw),
                "annotated": ("标注图", _image(value=127)),
            }
        ),
        existing_layer_records={"raw": raw_record},
    )
    return evidence, artifact


def test_record_bundle_reuses_existing_raw_snapshot_and_loads_safely(
    tmp_path,
):
    evidence = _evidence(tmp_path)
    image = _image()
    png = _png(image)
    raw_record = _raw_record(evidence, image)

    artifact = record_vision_bundle(
        evidence,
        _bundle(layers={"raw": ("原图", image)}),
        existing_layer_records={"raw": raw_record},
    )

    assert artifact == "vision-bundle-frame-000001.json"
    assert len(list((evidence.directory / "frames").glob("*.png"))) == 1
    loaded = load_recorded_bundle(evidence.directory, artifact)
    assert loaded["bundle_id"] == "frame-000001"
    assert loaded["layers"][0]["sha256"] == hashlib.sha256(png).hexdigest()
    assert loaded["_resolved_layer_paths"] == {
        "raw": (evidence.directory / "frames/frame-000001.png").resolve()
    }


def test_record_bundle_writes_additional_layers_before_json(tmp_path):
    evidence = _evidence(tmp_path)
    raw = _image()
    raw_record = _raw_record(evidence, raw)
    annotated = _image(value=127)

    artifact = record_vision_bundle(
        evidence,
        _bundle(
            layers={"raw": ("原图", raw), "annotated": ("标注图", annotated)}
        ),
        existing_layer_records={"raw": raw_record},
    )

    payload = json.loads(
        (evidence.directory / artifact).read_text(encoding="utf-8")
    )
    assert [layer["layer_id"] for layer in payload["layers"]] == [
        "raw",
        "annotated",
    ]
    assert payload["layers"][1]["path"] == (
        "frames/frame-000001-annotated.png"
    )
    assert (evidence.directory / payload["layers"][1]["path"]).is_file()
    assert len(list((evidence.directory / "frames").glob("*.png"))) == 2


def test_appended_layer_keeps_logical_snapshot_id_within_limit(tmp_path):
    evidence = _evidence(tmp_path)
    bundle = _bundle(layers={"annotated": ("标注图", _image())})

    artifact = record_vision_bundle(evidence, bundle)

    payload = json.loads(
        (evidence.directory / artifact).read_text(encoding="utf-8")
    )
    assert payload["layers"][0]["path"] == (
        "frames/frame-000001-annotated.png"
    )


def test_overlength_appended_layer_uses_deterministic_snapshot_id(tmp_path):
    evidence = _evidence(tmp_path)
    bundle_id = "b" * 61
    layer_id = "l" + ("x" * 39)
    logical_id = f"{bundle_id}-{layer_id}"
    assert len(logical_id) > 80
    expected_id = "vision-" + hashlib.sha256(
        logical_id.encode("ascii")
    ).hexdigest()
    bundle = _bundle(
        bundle_id=bundle_id,
        layers={layer_id: ("长 ID 图层", _image())},
    )

    artifact = record_vision_bundle(evidence, bundle)

    payload = json.loads(
        (evidence.directory / artifact).read_text(encoding="utf-8")
    )
    assert artifact == f"vision-bundle-{bundle_id}.json"
    assert len(expected_id) == 71
    assert payload["layers"][0]["path"] == f"frames/{expected_id}.png"


def test_longest_bundle_id_uses_limited_artifact_and_loads(tmp_path):
    evidence = _evidence(tmp_path)
    raw = _image()
    raw_record = _raw_record(evidence, raw)
    bundle_id = "b" * 80
    expected_artifact = (
        "vision-bundle-_"
        + hashlib.sha256(bundle_id.encode("ascii")).hexdigest()[:60]
        + ".json"
    )
    before_pngs = sorted((evidence.directory / "frames").glob("*.png"))

    artifact = record_vision_bundle(
        evidence,
        _bundle(bundle_id=bundle_id, layers={"raw": ("原图", raw)}),
        existing_layer_records={"raw": raw_record},
    )

    assert artifact == expected_artifact
    assert len(Path(artifact).stem) == 75
    assert len(artifact) == 80
    assert Path(artifact).match("vision-bundle-*.json")
    assert load_recorded_bundle(evidence.directory, artifact)["bundle_id"] == (
        bundle_id
    )
    assert sorted((evidence.directory / "frames").glob("*.png")) == (
        before_pngs
    )


def test_existing_limited_artifact_is_rejected_before_appended_png(tmp_path):
    evidence = _evidence(tmp_path)
    bundle_id = "b" * 80
    expected_artifact = (
        "vision-bundle-_"
        + hashlib.sha256(bundle_id.encode("ascii")).hexdigest()[:60]
        + ".json"
    )
    evidence.record_json_artifact(expected_artifact, {})
    before_pngs = list((evidence.directory / "frames").glob("*.png"))

    with pytest.raises(VisionPlatformError, match="already exists") as exc:
        record_vision_bundle(
            evidence,
            _bundle(
                bundle_id=bundle_id,
                layers={"annotated": ("标注图", _image())},
            ),
        )
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, FileExistsError)

    assert list((evidence.directory / "frames").glob("*.png")) == before_pngs


def test_long_artifact_and_old_digest_short_id_coexist_without_alias(tmp_path):
    evidence = _evidence(tmp_path)
    long_id = "b" * 80
    digest = hashlib.sha256(long_id.encode("ascii")).hexdigest()
    old_alias = digest[:61]
    long_artifact = record_vision_bundle(
        evidence,
        _bundle(
            bundle_id=long_id,
            layers={"raw": ("原图", _image())},
        ),
    )
    expected_long = f"vision-bundle-_{digest[:60]}.json"
    old_artifact = f"vision-bundle-{old_alias}.json"
    assert long_artifact == expected_long

    (evidence.directory / old_artifact).write_bytes(
        (evidence.directory / long_artifact).read_bytes()
    )
    with pytest.raises(VisionPlatformError) as exc:
        load_recorded_bundle(evidence.directory, old_artifact)
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    (evidence.directory / old_artifact).unlink()

    short_artifact = record_vision_bundle(
        evidence,
        _bundle(
            bundle_id=old_alias,
            layers={"raw": ("原图", _image(value=1))},
        ),
    )
    assert short_artifact == old_artifact
    assert load_recorded_bundle(evidence.directory, long_artifact)[
        "bundle_id"
    ] == long_id
    assert load_recorded_bundle(evidence.directory, short_artifact)[
        "bundle_id"
    ] == old_alias


def test_preexisting_later_snapshot_target_rejects_before_first_write(
    tmp_path
):
    evidence = _evidence(tmp_path)
    frames = evidence.directory / "frames"
    frames.mkdir()
    later = frames / "frame-000001-annotated.png"
    later.write_bytes(b"preexisting")
    before = sorted(frames.glob("*.png"))
    bundle = _bundle(
        layers={
            "raw": ("原图", _image()),
            "annotated": ("标注图", _image(value=127)),
        }
    )

    with pytest.raises(VisionPlatformError) as exc:
        record_vision_bundle(evidence, bundle)

    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, FileExistsError)
    assert sorted(frames.glob("*.png")) == before


def test_unserializable_final_payload_rejects_before_snapshot_write(tmp_path):
    evidence = _evidence(tmp_path)
    bundle = make_result_bundle(
        bundle_id="frame-000001",
        experiment_id="V1-01",
        source_snapshot_id="frame-000001",
        status="PASS",
        layers={"raw": ("原图", _image())},
        result={"huge_integer": 10**5000},
        profile={"profile_id": "standard"},
    )

    with pytest.raises(VisionPlatformError) as exc:
        record_vision_bundle(evidence, bundle)

    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, ValueError)
    assert list(evidence.directory.glob("frames/*.png")) == []


def test_record_revalidates_incoming_bundle_before_any_write(tmp_path):
    evidence = _evidence(tmp_path)
    valid = _bundle()
    unsafe = object.__new__(VisionResultBundle)
    for field, value in valid.__dict__.items():
        object.__setattr__(unsafe, field, value)
    object.__setattr__(unsafe, "status", "FAILED")

    with pytest.raises(VisionPlatformError) as exc:
        record_vision_bundle(evidence, unsafe)

    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert list(evidence.directory.glob("frames/*.png")) == []


def test_record_failure_has_public_evidence_code_and_export(tmp_path):
    evidence = _evidence(tmp_path)
    bundle = _bundle(layers={"raw": ("原图", _image())})
    target = evidence.directory / "frames/frame-000001-raw.png"
    target.parent.mkdir()
    target.write_bytes(b"exists")

    with pytest.raises(VisionPlatformError) as exc:
        record_vision_bundle(evidence, bundle)

    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, FileExistsError)

    import vision_platform.vision_quality as quality

    assert quality.VisionResultEvidenceError is type(exc.value)


def test_appended_snapshot_id_collision_is_rejected_before_any_write(
    tmp_path, monkeypatch
):
    evidence = _evidence(tmp_path)
    bundle_id = "b" * 61
    first_layer_id = "a" + ("x" * 39)
    second_layer_id = "b" + ("x" * 39)
    bundle = _bundle(
        bundle_id=bundle_id,
        layers={
            first_layer_id: ("第一层", _image()),
            second_layer_id: ("第二层", _image(value=1)),
        },
    )
    snapshot_called = False
    artifact_called = False

    class _FixedDigest:
        def hexdigest(self):
            return "a" * 64

    monkeypatch.setattr(
        evidence_module.hashlib, "sha256", lambda _payload=b"": _FixedDigest()
    )

    def reject_snapshot(**_kwargs):
        nonlocal snapshot_called
        snapshot_called = True
        raise AssertionError("snapshot write must not be attempted")

    def reject_artifact(_name, _payload):
        nonlocal artifact_called
        artifact_called = True
        raise AssertionError("artifact write must not be attempted")

    monkeypatch.setattr(evidence, "record_snapshot", reject_snapshot)
    monkeypatch.setattr(evidence, "record_json_artifact", reject_artifact)

    with pytest.raises(ValueError, match="snapshot ID collision"):
        record_vision_bundle(evidence, bundle)

    assert snapshot_called is False
    assert artifact_called is False


def test_appended_snapshot_id_cannot_collide_with_reused_raw_id(
    tmp_path, monkeypatch
):
    evidence = _evidence(tmp_path)
    raw = _image()
    source_snapshot_id = "frame-annotated"
    raw_record = evidence.record_snapshot(
        snapshot_id=source_snapshot_id,
        png_bytes=_png(raw),
        metadata={"width": 32, "height": 24},
    )
    bundle = make_result_bundle(
        bundle_id="frame",
        experiment_id="V1-01",
        source_snapshot_id=source_snapshot_id,
        status="PASS",
        layers={
            "raw": ("原图", raw),
            "annotated": ("标注图", _image(value=127)),
        },
        result={"capture": "ok"},
        profile={"profile_id": "standard", "resolution": [32, 24]},
    )
    snapshot_called = False
    real_snapshot = evidence.record_snapshot

    def track_snapshot(**kwargs):
        nonlocal snapshot_called
        snapshot_called = True
        return real_snapshot(**kwargs)

    monkeypatch.setattr(evidence, "record_snapshot", track_snapshot)

    with pytest.raises(ValueError, match="snapshot ID collision"):
        record_vision_bundle(
            evidence,
            bundle,
            existing_layer_records={"raw": raw_record},
        )

    assert snapshot_called is False
    assert list((evidence.directory / "frames").glob("*.png")) == [
        evidence.directory / f"frames/{source_snapshot_id}.png"
    ]


def test_record_existing_raw_uses_raw_layer_dimensions_when_not_first(
    tmp_path,
):
    evidence = _evidence(tmp_path)
    raw = _image()
    raw_record = _raw_record(evidence, raw)
    bundle = _bundle(
        layers={
            "annotated": ("标注图", _image(width=12, height=10)),
            "raw": ("原图", raw),
        }
    )

    artifact = record_vision_bundle(
        evidence, bundle, existing_layer_records={"raw": raw_record}
    )

    assert load_recorded_bundle(evidence.directory, artifact)["layers"][1][
        "layer_id"
    ] == "raw"


@pytest.mark.parametrize(
    "change",
    [
        {"path": "frames/other.png"},
        {"snapshot_id": "other"},
        {"sha256": "0" * 64},
        {"width": 31},
        {"height": 23},
    ],
)
def test_invalid_existing_raw_is_rejected_before_earlier_layer_write(
    tmp_path, change
):
    evidence = _evidence(tmp_path)
    raw = _image()
    raw_record = _raw_record(evidence, raw)
    raw_record.update(change)
    before_pngs = sorted((evidence.directory / "frames").glob("*.png"))
    bundle = _bundle(
        layers={
            "annotated": ("标注图", _image(value=127)),
            "raw": ("原图", raw),
        }
    )

    with pytest.raises(ValueError, match="existing.*raw"):
        record_vision_bundle(
            evidence,
            bundle,
            existing_layer_records={"raw": raw_record},
        )

    assert sorted((evidence.directory / "frames").glob("*.png")) == (
        before_pngs
    )


def test_existing_raw_without_bundle_raw_is_rejected_before_any_write(
    tmp_path, monkeypatch
):
    evidence = _evidence(tmp_path)
    raw_record = _raw_record(evidence)
    snapshot_called = False
    artifact_called = False
    real_snapshot = evidence.record_snapshot
    real_artifact = evidence.record_json_artifact

    def track_snapshot(**kwargs):
        nonlocal snapshot_called
        snapshot_called = True
        return real_snapshot(**kwargs)

    def track_artifact(name, payload):
        nonlocal artifact_called
        artifact_called = True
        return real_artifact(name, payload)

    monkeypatch.setattr(evidence, "record_snapshot", track_snapshot)
    monkeypatch.setattr(evidence, "record_json_artifact", track_artifact)

    with pytest.raises(ValueError, match="existing raw.*raw layer"):
        record_vision_bundle(
            evidence,
            _bundle(layers={"annotated": ("标注图", _image())}),
            existing_layer_records={"raw": raw_record},
        )

    assert snapshot_called is False
    assert artifact_called is False


@pytest.mark.parametrize(
    "change",
    [
        {"path": "frames/other.png"},
        {"snapshot_id": "other"},
        {"sha256": "0" * 64},
        {"width": 31},
        {"height": 23},
    ],
)
def test_record_rejects_inexact_existing_raw_record(tmp_path, change):
    evidence = _evidence(tmp_path)
    record = _raw_record(evidence)
    record.update(change)

    with pytest.raises(ValueError, match="existing.*raw"):
        record_vision_bundle(
            evidence, _bundle(), existing_layer_records={"raw": record}
        )

    assert not (evidence.directory / "vision-bundle-frame-000001.json").exists()


def test_record_rejects_changed_existing_raw_png(tmp_path):
    evidence = _evidence(tmp_path)
    record = _raw_record(evidence)
    (evidence.directory / record["path"]).write_bytes(_png(_image(value=255)))

    with pytest.raises(ValueError, match="existing.*raw"):
        record_vision_bundle(
            evidence, _bundle(), existing_layer_records={"raw": record}
        )

    assert not (evidence.directory / "vision-bundle-frame-000001.json").exists()


def test_duplicate_artifact_is_rejected_without_extra_frames(tmp_path):
    evidence = _evidence(tmp_path)
    record = _raw_record(evidence)
    record_vision_bundle(
        evidence, _bundle(), existing_layer_records={"raw": record}
    )
    before = sorted((evidence.directory / "frames").iterdir())

    with pytest.raises(VisionPlatformError, match="already exists") as exc:
        record_vision_bundle(
            evidence, _bundle(), existing_layer_records={"raw": record}
        )
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, FileExistsError)

    assert sorted((evidence.directory / "frames").iterdir()) == before


def test_png_encode_failure_leaves_no_bundle_or_new_snapshot(
    tmp_path, monkeypatch
):
    evidence = _evidence(tmp_path)
    raw_record = _raw_record(evidence)
    before = sorted((evidence.directory / "frames").iterdir())
    real_encode = cv2.imencode

    def fail_annotated(extension, image):
        if int(image[0, 0, 0]) == 127:
            return False, None
        return real_encode(extension, image)

    monkeypatch.setattr(cv2, "imencode", fail_annotated)
    bundle = _bundle(
        layers={
            "raw": ("原图", _image()),
            "annotated": ("标注图", _image(value=127)),
        }
    )

    with pytest.raises(ValueError, match="PNG"):
        record_vision_bundle(
            evidence, bundle, existing_layer_records={"raw": raw_record}
        )

    assert sorted((evidence.directory / "frames").iterdir()) == before
    assert not (evidence.directory / "vision-bundle-frame-000001.json").exists()


def test_snapshot_write_failure_never_leaves_dangling_bundle(
    tmp_path, monkeypatch
):
    evidence = _evidence(tmp_path)
    raw_record = _raw_record(evidence)
    real_record_snapshot = evidence.record_snapshot

    def fail_snapshot(**kwargs):
        if kwargs["snapshot_id"].endswith("annotated"):
            raise OSError("injected snapshot failure")
        return real_record_snapshot(**kwargs)

    monkeypatch.setattr(evidence, "record_snapshot", fail_snapshot)

    with pytest.raises(VisionPlatformError, match="snapshot failure") as exc:
        record_vision_bundle(
            evidence,
            _bundle(
                layers={
                    "raw": ("原图", _image()),
                    "annotated": ("标注图", _image(value=127)),
                }
            ),
            existing_layer_records={"raw": raw_record},
        )
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, OSError)

    assert not (evidence.directory / "vision-bundle-frame-000001.json").exists()


def test_json_write_failure_occurs_after_layers_and_leaves_no_bundle(
    tmp_path, monkeypatch
):
    evidence = _evidence(tmp_path)
    raw_record = _raw_record(evidence)

    def fail_json(name, payload):
        assert (evidence.directory / "frames/frame-000001-annotated.png").is_file()
        raise OSError("injected JSON failure")

    monkeypatch.setattr(evidence, "record_json_artifact", fail_json)
    with pytest.raises(VisionPlatformError, match="JSON failure") as exc:
        record_vision_bundle(
            evidence,
            _bundle(
                layers={
                    "raw": ("原图", _image()),
                    "annotated": ("标注图", _image(value=127)),
                }
            ),
            existing_layer_records={"raw": raw_record},
        )
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, OSError)
    assert not (evidence.directory / "vision-bundle-frame-000001.json").exists()


@pytest.mark.parametrize(
    "artifact_name",
    [
        "../vision-bundle-frame-000001.json",
        "nested/vision-bundle-frame-000001.json",
        str(Path("C:/outside/vision-bundle-frame-000001.json")),
    ],
)
def test_loader_rejects_artifact_path_escape(tmp_path, artifact_name):
    evidence, _ = _record_one(tmp_path)
    with pytest.raises(ValueError, match="inside evidence directory"):
        load_recorded_bundle(evidence.directory, artifact_name)


def test_loader_rejects_artifact_name_over_public_limit(tmp_path):
    evidence, artifact = _record_one(tmp_path)
    too_long = "vision-bundle-" + ("a" * 62) + ".json"
    assert len(Path(too_long).stem) == 76
    assert len(too_long) == 81
    (evidence.directory / too_long).write_bytes(
        (evidence.directory / artifact).read_bytes()
    )

    with pytest.raises(ValueError, match="inside evidence directory"):
        load_recorded_bundle(evidence.directory, too_long)


def test_loader_rejects_artifact_name_not_derived_from_bundle(tmp_path):
    evidence, artifact = _record_one(tmp_path)
    renamed = "vision-bundle-other.json"
    (evidence.directory / renamed).write_bytes(
        (evidence.directory / artifact).read_bytes()
    )

    with pytest.raises(ValueError, match="artifact name"):
        load_recorded_bundle(evidence.directory, renamed)


def _rewrite_artifact(evidence, artifact, mutate):
    path = evidence.directory / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def test_loader_translates_nested_bundle_error_to_evidence_code(tmp_path):
    evidence, artifact = _record_one(tmp_path)
    _rewrite_artifact(
        evidence,
        artifact,
        lambda payload: payload.update({"status": "FAILED"}),
    )

    with pytest.raises(VisionPlatformError) as exc:
        load_recorded_bundle(evidence.directory, artifact)

    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, VisionPlatformError)
    assert exc.value.__cause__.code == "VISION_RESULT_BUNDLE_INVALID"


def test_loader_rejects_same_run_arbitrary_layer_path(tmp_path):
    evidence, artifact = _record_two(tmp_path)
    payload = json.loads(
        (evidence.directory / artifact).read_text(encoding="utf-8")
    )
    annotated = payload["layers"][1]
    source = evidence.directory / annotated["path"]
    arbitrary = evidence.directory / "frames/arbitrary.png"
    arbitrary.write_bytes(source.read_bytes())
    _rewrite_artifact(
        evidence,
        artifact,
        lambda value: value["layers"][1].update(
            {"path": "frames/arbitrary.png"}
        ),
    )

    with pytest.raises(VisionPlatformError) as exc:
        load_recorded_bundle(evidence.directory, artifact)
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"


def test_loader_rejects_path_reused_by_two_layers(tmp_path):
    evidence, artifact = _record_two(tmp_path)
    payload = json.loads(
        (evidence.directory / artifact).read_text(encoding="utf-8")
    )
    raw = payload["layers"][0]

    def repeat_path(value):
        value["layers"][1].update(
            {
                "path": raw["path"],
                "sha256": raw["sha256"],
                "width": raw["width"],
                "height": raw["height"],
            }
        )

    _rewrite_artifact(evidence, artifact, repeat_path)
    with pytest.raises(VisionPlatformError) as exc:
        load_recorded_bundle(evidence.directory, artifact)
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"


def test_loader_rejects_two_paths_for_the_same_physical_png(tmp_path):
    evidence = _evidence(tmp_path)
    image = _image()
    png = _png(image)
    digest = hashlib.sha256(png).hexdigest()
    frames = evidence.directory / "frames"
    frames.mkdir()
    source_path = frames / "FRAME-ANNOTATED.png"
    appended_path = frames / "frame-annotated.png"
    source_path.write_bytes(png)
    if not appended_path.exists():
        os.link(source_path, appended_path)
    assert os.path.samefile(source_path, appended_path)

    bundle = make_result_bundle(
        bundle_id="frame",
        experiment_id="V1-01",
        source_snapshot_id="FRAME-ANNOTATED",
        status="PASS",
        layers={
            "raw": ("原图", image),
            "annotated": ("标注图", image),
        },
        result={"capture": "ok"},
        profile={"profile_id": "standard"},
    )
    payload = result_bundle_to_dict(
        bundle,
        layer_records={
            "raw": {
                "path": "frames/FRAME-ANNOTATED.png",
                "sha256": digest,
                "width": 32,
                "height": 24,
            },
            "annotated": {
                "path": "frames/frame-annotated.png",
                "sha256": digest,
                "width": 32,
                "height": 24,
            },
        },
    )
    artifact = evidence.record_json_artifact(
        "vision-bundle-frame.json", payload
    )

    with pytest.raises(VisionPlatformError) as exc:
        load_recorded_bundle(evidence.directory, artifact)
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, ValueError)


class _ReadSpy:
    def __init__(self, stream, reads):
        self._stream = stream
        self._reads = reads

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def read(self, size=-1):
        self._reads.append(size)
        return self._stream.read(size)


def _spy_on_png_reads(monkeypatch):
    reads = []
    real_open = Path.open

    def tracked_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        if path.suffix.lower() == ".png" and "b" in (
            args[0] if args else kwargs.get("mode", "r")
        ):
            return _ReadSpy(stream, reads)
        return stream

    monkeypatch.setattr(Path, "open", tracked_open)
    return reads


@pytest.mark.parametrize("operation", ["load", "existing_raw"])
def test_png_files_use_one_explicit_bounded_read(
    tmp_path, monkeypatch, operation
):
    evidence, artifact = _record_one(tmp_path)
    raw_path = evidence.directory / "frames/frame-000001.png"
    limit = raw_path.stat().st_size + 7
    monkeypatch.setattr(evidence_module, "_MAX_PNG_BYTES", limit)
    reads = _spy_on_png_reads(monkeypatch)

    if operation == "load":
        load_recorded_bundle(evidence.directory, artifact)
    else:
        (evidence.directory / artifact).unlink()
        raw_record = {
            "snapshot_id": "frame-000001",
            "path": "frames/frame-000001.png",
            "sha256": hashlib.sha256(_png(_image())).hexdigest(),
            "width": 32,
            "height": 24,
        }
        record_vision_bundle(
            evidence,
            _bundle(),
            existing_layer_records={"raw": raw_record},
        )

    assert reads == [limit + 1]


@pytest.mark.parametrize("operation", ["load", "existing_raw"])
def test_oversize_png_is_rejected_from_fstat_before_any_read(
    tmp_path, monkeypatch, operation
):
    evidence, artifact = _record_one(tmp_path)
    raw_path = evidence.directory / "frames/frame-000001.png"
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b"XX")
    monkeypatch.setattr(
        evidence_module, "_MAX_PNG_BYTES", len(original) + 1
    )
    reads = _spy_on_png_reads(monkeypatch)

    with pytest.raises(VisionPlatformError) as exc:
        if operation == "load":
            load_recorded_bundle(evidence.directory, artifact)
        else:
            (evidence.directory / artifact).unlink()
            raw_record = {
                "snapshot_id": "frame-000001",
                "path": "frames/frame-000001.png",
                "sha256": hashlib.sha256(_png(_image())).hexdigest(),
                "width": 32,
                "height": 24,
            }
            record_vision_bundle(
                evidence,
                _bundle(),
                existing_layer_records={"raw": raw_record},
            )

    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert isinstance(exc.value.__cause__, ValueError)
    assert reads == []


def _invalid_png_bytes(case: str) -> bytes:
    if case == "grayscale":
        ok, encoded = cv2.imencode(
            ".png", np.zeros((24, 32), dtype=np.uint8)
        )
        assert ok
        return encoded.tobytes()
    if case == "alpha":
        ok, encoded = cv2.imencode(
            ".png", np.zeros((24, 32, 4), dtype=np.uint8)
        )
        assert ok
        return encoded.tobytes()
    if case == "oversized":
        valid = _png(_image())
        return valid + (b"\0" * ((64 * 1024 * 1024 + 1) - len(valid)))

    payload = bytearray(_png(_image()))
    if case == "truncated":
        return bytes(payload[:20])
    if case == "ihdr_length":
        payload[8:12] = (12).to_bytes(4, "big")
    elif case == "ihdr_type":
        payload[12:16] = b"IDAT"
    elif case == "width_zero":
        payload[16:20] = (0).to_bytes(4, "big")
    elif case == "height_zero":
        payload[20:24] = (0).to_bytes(4, "big")
    elif case == "width_limit":
        payload[16:20] = (4097).to_bytes(4, "big")
    elif case == "height_pixel_limit":
        payload[20:24] = (4097).to_bytes(4, "big")
    elif case == "bit_depth":
        payload[24] = 16
    elif case == "compression":
        payload[26] = 1
    elif case == "filter":
        payload[27] = 1
    elif case == "interlace":
        payload[28] = 1
    else:
        raise AssertionError(f"unknown PNG case: {case}")
    return bytes(payload)


@pytest.mark.parametrize(
    "case",
    [
        "oversized",
        "grayscale",
        "alpha",
        "truncated",
        "ihdr_length",
        "ihdr_type",
        "width_zero",
        "height_zero",
        "width_limit",
        "height_pixel_limit",
        "bit_depth",
        "compression",
        "filter",
        "interlace",
    ],
)
def test_loader_rejects_png_header_violation_before_decode(
    tmp_path, monkeypatch, case
):
    evidence, artifact = _record_one(tmp_path)
    payload = json.loads(
        (evidence.directory / artifact).read_text(encoding="utf-8")
    )
    layer = payload["layers"][0]
    frame = evidence.directory / layer["path"]
    invalid = _invalid_png_bytes(case)
    frame.write_bytes(invalid)
    _rewrite_artifact(
        evidence,
        artifact,
        lambda value: value["layers"][0].update(
            {"sha256": hashlib.sha256(invalid).hexdigest()}
        ),
    )
    decode_called = False

    def reject_decode(*_args, **_kwargs):
        nonlocal decode_called
        decode_called = True
        raise AssertionError("OpenCV decode must not run")

    monkeypatch.setattr(cv2, "imdecode", reject_decode)
    with pytest.raises(VisionPlatformError) as exc:
        load_recorded_bundle(evidence.directory, artifact)
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert decode_called is False


def test_record_validates_encoded_png_header_before_write(tmp_path, monkeypatch):
    evidence = _evidence(tmp_path)
    invalid = np.frombuffer(_invalid_png_bytes("grayscale"), dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", lambda *_args: (True, invalid))
    with pytest.raises(VisionPlatformError) as exc:
        record_vision_bundle(evidence, _bundle())
    assert exc.value.code == "VISION_RESULT_EVIDENCE_INVALID"
    assert list(evidence.directory.glob("frames/*.png")) == []


@pytest.mark.parametrize(
    "shape,dimension_limit",
    [
        ((1, 4097, 3), 4096),
        ((4097, 1, 3), 4096),
        ((4096, 4097, 3), 5000),
    ],
)
def test_png_shape_limits_are_rejected_before_encoding(
    monkeypatch, shape, dimension_limit
):
    image = np.lib.stride_tricks.as_strided(
        np.zeros((1, 1, 3), dtype=np.uint8),
        shape=shape,
        strides=(0, 0, 1),
        writeable=False,
    )
    called = False

    def reject_encode(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("cv2.imencode must not be called")

    monkeypatch.setattr(
        evidence_module, "_MAX_PNG_DIMENSION", dimension_limit
    )
    monkeypatch.setattr(cv2, "imencode", reject_encode)

    with pytest.raises(ValueError, match="dimensions"):
        evidence_module._png_bytes(image, layer_id="raw")
    assert called is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["layers"][0].update(
            {"path": "../outside.png"}
        ),
        lambda payload: payload["layers"][0].update(
            {"path": "C:/outside.png"}
        ),
        lambda payload: payload["layers"][0].update(
            {"path": "frames/../../outside.png"}
        ),
    ],
)
def test_loader_rejects_layer_path_escape(tmp_path, mutate):
    evidence, artifact = _record_one(tmp_path)
    _rewrite_artifact(evidence, artifact, mutate)
    with pytest.raises(ValueError, match="inside evidence directory"):
        load_recorded_bundle(evidence.directory, artifact)


def test_loader_rejects_symlink_escape(tmp_path, monkeypatch):
    evidence, artifact = _record_one(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png(_image()))
    link = evidence.directory / "frames/linked.png"
    try:
        link.symlink_to(outside)
    except OSError:
        link.write_bytes(outside.read_bytes())
        real_resolve = Path.resolve
        outside_resolved = real_resolve(outside, strict=True)

        def simulate_symlink_escape(path, strict=False):
            if path == link:
                return outside_resolved
            return real_resolve(path, strict=strict)

        monkeypatch.setattr(Path, "resolve", simulate_symlink_escape)
    _rewrite_artifact(
        evidence,
        artifact,
        lambda payload: payload["layers"][0].update(
            {
                "path": "frames/linked.png",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        ),
    )

    with pytest.raises(ValueError, match="inside evidence directory"):
        load_recorded_bundle(evidence.directory, artifact)


def test_loader_rejects_noncanonical_dot_layer_path(tmp_path):
    evidence, artifact = _record_one(tmp_path)
    _rewrite_artifact(
        evidence,
        artifact,
        lambda payload: payload["layers"][0].update(
            {"path": "./frames/frame-000001.png"}
        ),
    )

    with pytest.raises(ValueError, match="inside evidence directory"):
        load_recorded_bundle(evidence.directory, artifact)


def test_loader_rejects_jpeg_bytes_before_opencv_decode(
    tmp_path, monkeypatch
):
    evidence, artifact = _record_one(tmp_path)
    frame = evidence.directory / "frames/frame-000001.png"
    ok, encoded = cv2.imencode(".jpg", _image())
    assert ok
    jpeg = encoded.tobytes()
    assert jpeg.startswith(b"\xff\xd8")
    frame.write_bytes(jpeg)
    _rewrite_artifact(
        evidence,
        artifact,
        lambda payload: payload["layers"][0].update(
            {"sha256": hashlib.sha256(jpeg).hexdigest()}
        ),
    )
    decode_called = False
    real_decode = cv2.imdecode

    def track_decode(*args, **kwargs):
        nonlocal decode_called
        decode_called = True
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecode", track_decode)
    with pytest.raises(ValueError, match="PNG signature"):
        load_recorded_bundle(evidence.directory, artifact)
    assert decode_called is False


@pytest.mark.parametrize(
    "damage",
    ["missing", "hash", "dimensions", "invalid_png"],
)
def test_loader_rejects_missing_or_changed_layer_files(tmp_path, damage):
    evidence, artifact = _record_one(tmp_path)
    frame = evidence.directory / "frames/frame-000001.png"
    if damage == "missing":
        frame.unlink()
    elif damage == "hash":
        frame.write_bytes(_png(_image(value=255)))
    elif damage == "dimensions":
        changed = _image(width=31)
        frame.write_bytes(_png(changed))
        _rewrite_artifact(
            evidence,
            artifact,
            lambda payload: payload["layers"][0].update(
                {"sha256": hashlib.sha256(frame.read_bytes()).hexdigest()}
            ),
        )
    else:
        frame.write_bytes(b"not png")
        _rewrite_artifact(
            evidence,
            artifact,
            lambda payload: payload["layers"][0].update(
                {"sha256": hashlib.sha256(frame.read_bytes()).hexdigest()}
            ),
        )

    with pytest.raises(ValueError, match="layer"):
        load_recorded_bundle(evidence.directory, artifact)


def test_loader_rejects_invalid_json_constants(tmp_path):
    evidence, artifact = _record_one(tmp_path)
    path = evidence.directory / artifact
    text = path.read_text(encoding="utf-8").replace(
        '"result": {', '"result": {"bad": NaN, '
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        load_recorded_bundle(evidence.directory, artifact)


def test_loader_contains_opencv_decode_errors(tmp_path, monkeypatch):
    evidence, artifact = _record_one(tmp_path)

    def fail_decode(*_args, **_kwargs):
        raise cv2.error("injected decoder failure")

    monkeypatch.setattr(cv2, "imdecode", fail_decode)
    with pytest.raises(ValueError, match="layer.*PNG"):
        load_recorded_bundle(evidence.directory, artifact)


@pytest.mark.parametrize("target", ["top", "layer"])
def test_loader_rejects_duplicate_json_keys_at_every_layer(tmp_path, target):
    evidence, artifact = _record_one(tmp_path)
    path = evidence.directory / artifact
    text = path.read_text(encoding="utf-8")
    if target == "top":
        text = text.replace(
            '"schema_version": 1,',
            '"schema_version": 1, "schema_version": 1,',
            1,
        )
    else:
        text = text.replace(
            '"layer_id": "raw",',
            '"layer_id": "raw", "layer_id": "raw",',
            1,
        )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_recorded_bundle(evidence.directory, artifact)


def test_loader_rejects_duplicate_layer_ids(tmp_path):
    evidence, artifact = _record_one(tmp_path)

    def duplicate(payload):
        payload["layers"].append(dict(payload["layers"][0]))

    _rewrite_artifact(evidence, artifact, duplicate)
    with pytest.raises(ValueError, match="duplicate layer_id"):
        load_recorded_bundle(evidence.directory, artifact)


def test_loader_returns_a_fresh_copied_mapping(tmp_path):
    evidence, artifact = _record_one(tmp_path)
    loaded = load_recorded_bundle(evidence.directory, artifact)
    loaded["result"]["capture"] = "changed"
    loaded["layers"][0]["title"] = "changed"
    loaded["_resolved_layer_paths"]["raw"] = Path("outside")

    reloaded = load_recorded_bundle(evidence.directory, artifact)
    assert reloaded["result"] == {"capture": "ok"}
    assert reloaded["layers"][0]["title"] == "原图"
    assert reloaded["_resolved_layer_paths"]["raw"] != Path("outside")
