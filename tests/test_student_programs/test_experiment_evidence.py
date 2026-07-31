from __future__ import annotations

import hashlib
import json
import queue
import threading

import pytest

import vision_platform.student.evidence as evidence_module
from vision_platform.student.evidence import StudentRunEvidence


def _program(tmp_path):
    program = tmp_path / "student.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    return program


def _finalize(evidence: StudentRunEvidence):
    return evidence.finalize(
        status="PASS",
        command_count=1,
        last_pose_mm=(100, 0, 120),
        safety_violation_count=0,
        error=None,
        cleanup_errors=[],
    )


def test_evidence_records_experiment_and_snapshot(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="run-r1-05",
        run_metadata={
            "experiment_id": "R1-05",
            "experiment_version": "2.2.0",
            "scene_sha256": "a" * 64,
            "hardware_status": "PENDING_HARDWARE",
        },
    )

    png_bytes = b"\x89PNG\r\n\x1a\npayload"
    record = evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=png_bytes,
        metadata={
            "width": 640,
            "height": 480,
            "source": "coppeliasim",
            "sequence_id": 1,
        },
    )
    summary_path = _finalize(evidence)

    assert record["path"] == "frames/frame-000001.png"
    assert record["sha256"] == hashlib.sha256(png_bytes).hexdigest()
    assert (evidence.directory / record["path"]).read_bytes() == png_bytes
    manifest = json.loads(
        (evidence.directory / "manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshots = [
        json.loads(line)
        for line in (evidence.directory / "snapshots.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["experiment_id"] == "R1-05"
    assert summary["experiment_id"] == "R1-05"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    assert snapshots == [record]
    assert "payload" not in snapshots[0]


def test_snapshot_id_cannot_escape_evidence_directory(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="run-safe-path",
        run_metadata={"hardware_status": "PENDING_HARDWARE"},
    )

    with pytest.raises(ValueError, match="snapshot_id"):
        evidence.record_snapshot(
            snapshot_id="../escape",
            png_bytes=b"png",
            metadata={},
        )


def test_run_metadata_is_copied_and_cannot_forge_core_evidence(tmp_path):
    metadata = {
        "experiment_id": "R1-05",
        "run_id": "forged-run",
        "status": "FAILED",
        "source_sha256": "0" * 64,
        "robot_backend": "real",
    }
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="authentic-run",
        run_metadata=metadata,
    )
    metadata["experiment_id"] = "mutated-after-create"

    manifest = json.loads(
        (evidence.directory / "manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(_finalize(evidence).read_text(encoding="utf-8"))

    assert evidence.run_metadata["experiment_id"] == "R1-05"
    assert manifest["run_id"] == "authentic-run"
    assert manifest["source_sha256"] == evidence.source_sha256
    assert manifest["robot_backend"] == "sim"
    assert summary["run_id"] == "authentic-run"
    assert summary["status"] == "PASS"
    assert summary["source_sha256"] == evidence.source_sha256
    assert summary["robot_backend"] == "sim"


@pytest.mark.parametrize("hardware_status", ["PASS", "REAL_VERIFIED", None])
def test_hardware_status_cannot_be_overstated(tmp_path, hardware_status):
    with pytest.raises(ValueError, match="hardware_status"):
        StudentRunEvidence.create(
            output_root=tmp_path / "runs",
            program_path=_program(tmp_path),
            robot_backend="sim",
            run_id="hardware-boundary",
            run_metadata={"hardware_status": hardware_status},
        )


def test_snapshot_metadata_cannot_forge_identity_path_or_digest(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="snapshot-integrity",
    )
    png_bytes = b"\x89PNG\r\n\x1a\nauthentic"

    record = evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=png_bytes,
        metadata={
            "snapshot_id": "forged",
            "path": "../../escape.png",
            "sha256": "0" * 64,
            "width": 640,
        },
    )

    assert record["snapshot_id"] == "frame-000001"
    assert record["path"] == "frames/frame-000001.png"
    assert record["sha256"] == hashlib.sha256(png_bytes).hexdigest()
    assert record["width"] == 640
    assert (evidence.directory / record["path"]).read_bytes() == png_bytes


def test_finalized_evidence_rejects_snapshot_without_creating_frame(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="sealed-run",
    )
    _finalize(evidence)

    with pytest.raises(RuntimeError, match="finalized"):
        evidence.record_snapshot(
            snapshot_id="too-late",
            png_bytes=b"\x89PNG\r\n\x1a\nlate",
            metadata={},
        )

    assert not (evidence.directory / "frames" / "too-late.png").exists()
    assert not (evidence.directory / "snapshots.jsonl").exists()


def test_unserializable_snapshot_metadata_leaves_no_binary_or_jsonl(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="invalid-metadata",
    )

    with pytest.raises(TypeError):
        evidence.record_snapshot(
            snapshot_id="frame-000001",
            png_bytes=b"\x89PNG\r\n\x1a\ninvalid",
            metadata={"not_json": object()},
        )

    assert not (evidence.directory / "frames" / "frame-000001.png").exists()
    assert not (evidence.directory / "snapshots.jsonl").exists()


def test_snapshot_ids_are_unique_within_one_run(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="unique-snapshots",
    )
    first = b"\x89PNG\r\n\x1a\nfirst"
    evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=first,
        metadata={},
    )

    with pytest.raises(FileExistsError):
        evidence.record_snapshot(
            snapshot_id="frame-000001",
            png_bytes=b"\x89PNG\r\n\x1a\nsecond",
            metadata={},
        )

    assert (evidence.directory / "frames" / "frame-000001.png").read_bytes() == first
    assert len(
        (evidence.directory / "snapshots.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1


def test_run_metadata_is_deeply_copied_and_canonicalized(tmp_path):
    metadata = {
        "experiment_id": "R1-05",
        "context": {
            "labels": ["box", "pallet"],
            "workspace_mm": (300, 200, 150),
        },
    }
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="canonical-run",
        run_metadata=metadata,
    )
    metadata["context"]["labels"].append("mutated-source")
    metadata["context"]["workspace_mm"] = (0, 0, 0)
    evidence.run_metadata["context"]["labels"].append("mutated-public")

    manifest = json.loads(
        (evidence.directory / "manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(_finalize(evidence).read_text(encoding="utf-8"))

    expected = {
        "labels": ["box", "pallet"],
        "workspace_mm": [300, 200, 150],
    }
    assert manifest["context"] == expected
    assert summary["context"] == expected
    assert evidence.run_metadata["context"]["workspace_mm"] == [300, 200, 150]


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {"payload": b"binary"},
        {"payload": object()},
        {"confidence": float("nan")},
        {1: "non-string-key"},
    ],
)
def test_invalid_run_metadata_leaves_no_partial_run_directory(
    tmp_path, invalid_metadata
):
    output_root = tmp_path / "runs"

    with pytest.raises((TypeError, ValueError)):
        StudentRunEvidence.create(
            output_root=output_root,
            program_path=_program(tmp_path),
            robot_backend="sim",
            run_id="invalid-run-metadata",
            run_metadata=invalid_metadata,
        )

    assert not output_root.exists()


def test_snapshot_metadata_is_deeply_copied_and_canonicalized(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="canonical-snapshot",
    )
    metadata = {
        "dimensions": {"size": (640, 480)},
        "labels": ["box"],
    }

    record = evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=b"\x89PNG\r\n\x1a\ncanonical",
        metadata=metadata,
    )
    metadata["dimensions"]["size"] = (1, 1)
    metadata["labels"].append("mutated-source")

    expected_dimensions = {"size": [640, 480]}
    assert record["dimensions"] == expected_dimensions
    assert record["labels"] == ["box"]
    record["dimensions"]["size"].append(999)
    persisted = json.loads(
        (evidence.directory / "snapshots.jsonl").read_text(encoding="utf-8")
    )
    assert persisted["dimensions"] == expected_dimensions
    assert persisted["labels"] == ["box"]


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {"payload": b"binary"},
        {"confidence": float("nan")},
        {1: "non-string-key"},
    ],
)
def test_invalid_snapshot_metadata_leaves_no_frame_or_jsonl(
    tmp_path, invalid_metadata
):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="invalid-snapshot-metadata",
    )

    with pytest.raises((TypeError, ValueError)):
        evidence.record_snapshot(
            snapshot_id="frame-000001",
            png_bytes=b"\x89PNG\r\n\x1a\ninvalid",
            metadata=invalid_metadata,
        )

    assert not (evidence.directory / "frames").exists()
    assert not (evidence.directory / "snapshots.jsonl").exists()


@pytest.mark.parametrize(
    "snapshot_id",
    [
        "",
        "-frame",
        "_frame",
        "帧000001",
        "a" * 81,
        "CON",
        "con",
        "PRN",
        "aux",
        "NUL",
        "COM1",
        "com9",
        "LPT1",
        "lpt9",
    ],
)
def test_snapshot_id_enforces_portable_ascii_windows_name(snapshot_id, tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="portable-snapshot-id",
    )

    with pytest.raises(ValueError, match="snapshot_id"):
        evidence.record_snapshot(
            snapshot_id=snapshot_id,
            png_bytes=b"\x89PNG\r\n\x1a\nportable",
            metadata={},
        )

    assert not (evidence.directory / "frames").exists()
    assert not (evidence.directory / "snapshots.jsonl").exists()


def test_task7_snapshot_id_remains_compatible(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="task7-compatible-id",
    )

    record = evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=b"\x89PNG\r\n\x1a\ntask7",
        metadata={},
    )

    assert record["path"] == "frames/frame-000001.png"


@pytest.mark.parametrize("non_finite", [float("inf"), float("-inf")])
def test_non_finite_run_metadata_leaves_no_partial_directory(
    tmp_path, non_finite
):
    output_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="finite"):
        StudentRunEvidence.create(
            output_root=output_root,
            program_path=_program(tmp_path),
            robot_backend="sim",
            run_id="non-finite-run",
            run_metadata={"confidence": non_finite},
        )

    assert not output_root.exists()


@pytest.mark.parametrize("container_type", ["mapping", "list"])
def test_cyclic_run_metadata_is_rejected_before_directory_creation(
    tmp_path, container_type
):
    cyclic = {} if container_type == "mapping" else []
    if container_type == "mapping":
        cyclic["self"] = cyclic
    else:
        cyclic.append(cyclic)
    output_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="cycles"):
        StudentRunEvidence.create(
            output_root=output_root,
            program_path=_program(tmp_path),
            robot_backend="sim",
            run_id="cyclic-run",
            run_metadata={"cyclic": cyclic},
        )

    assert not output_root.exists()


@pytest.mark.parametrize("container_type", ["mapping", "list"])
def test_cyclic_snapshot_metadata_is_rejected_before_frame_creation(
    tmp_path, container_type
):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="cyclic-snapshot",
    )
    cyclic = {} if container_type == "mapping" else []
    if container_type == "mapping":
        cyclic["self"] = cyclic
    else:
        cyclic.append(cyclic)

    with pytest.raises(ValueError, match="cycles"):
        evidence.record_snapshot(
            snapshot_id="frame-000001",
            png_bytes=b"\x89PNG\r\n\x1a\ncyclic",
            metadata={"cyclic": cyclic},
        )

    assert not (evidence.directory / "frames").exists()
    assert not (evidence.directory / "snapshots.jsonl").exists()


def test_shared_metadata_reference_becomes_independent_json_branches(tmp_path):
    shared = {"size": (640, 480), "labels": ["box"]}
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="shared-reference",
        run_metadata={"left": shared, "right": shared},
    )

    left = evidence.run_metadata["left"]
    right = evidence.run_metadata["right"]
    assert left == right == {"size": [640, 480], "labels": ["box"]}
    assert left is not right
    assert left["size"] is not right["size"]
    assert left["labels"] is not right["labels"]
    left["labels"].append("left-only")
    assert right["labels"] == ["box"]


def test_snapshot_jsonl_failure_rolls_back_png_without_polluting_log(
    tmp_path, monkeypatch
):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="snapshot-rollback",
    )
    evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=b"\x89PNG\r\n\x1a\nfirst",
        metadata={"sequence_id": 1},
    )
    snapshots_path = evidence.directory / "snapshots.jsonl"
    snapshots_before = snapshots_path.read_bytes()
    real_atomic_write = evidence_module._atomic_write_bytes

    def fail_snapshot_log(path, payload):
        if path == snapshots_path:
            raise OSError("injected snapshots.jsonl failure")
        real_atomic_write(path, payload)

    monkeypatch.setattr(evidence_module, "_atomic_write_bytes", fail_snapshot_log)

    with pytest.raises(OSError, match="snapshots.jsonl"):
        evidence.record_snapshot(
            snapshot_id="frame-000002",
            png_bytes=b"\x89PNG\r\n\x1a\nsecond",
            metadata={"sequence_id": 2},
        )

    assert snapshots_path.read_bytes() == snapshots_before
    assert not (evidence.directory / "frames" / "frame-000002.png").exists()
    assert (evidence.directory / "frames" / "frame-000001.png").is_file()


def test_concurrent_duplicate_snapshot_id_has_exactly_one_winner(tmp_path):
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id="concurrent-snapshot",
    )
    barrier = threading.Barrier(2)
    outcomes: queue.Queue = queue.Queue()

    def capture(payload: bytes) -> None:
        barrier.wait(timeout=2)
        try:
            outcomes.put(
                ("ok", evidence.record_snapshot(
                    snapshot_id="frame-000001",
                    png_bytes=payload,
                    metadata={},
                ))
            )
        except Exception as error:
            outcomes.put(("error", error))

    payloads = [
        b"\x89PNG\r\n\x1a\nfirst-thread",
        b"\x89PNG\r\n\x1a\nsecond-thread",
    ]
    threads = [threading.Thread(target=capture, args=(item,)) for item in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    results = [outcomes.get_nowait(), outcomes.get_nowait()]
    successes = [item for item in results if item[0] == "ok"]
    errors = [item[1] for item in results if item[0] == "error"]
    assert len(successes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], FileExistsError)
    persisted = evidence.directory / "frames" / "frame-000001.png"
    assert persisted.read_bytes() in payloads
    assert len(
        (evidence.directory / "snapshots.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1
