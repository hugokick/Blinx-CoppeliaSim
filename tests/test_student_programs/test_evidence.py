from __future__ import annotations

import hashlib
import json
import queue
import threading
from pathlib import Path

import pytest

import vision_platform.student.evidence as evidence_module
from vision_platform.student.evidence import StudentRunEvidence


class _FaultingAtomicStream:
    def __init__(self, stream, failure: str) -> None:
        self._stream = stream
        self._failure = failure

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def write(self, payload: bytes) -> int:
        if self._failure == "short_write":
            partial = max(1, len(payload) // 2)
            return self._stream.write(payload[:partial])
        if self._failure in {"write_error", "write_close_error"}:
            partial = max(1, len(payload) // 2)
            self._stream.write(payload[:partial])
            raise OSError("injected write failure")
        return self._stream.write(payload)

    def flush(self) -> None:
        self._stream.flush()
        if self._failure == "flush_error":
            raise OSError("injected flush failure")

    def close(self) -> None:
        self._stream.close()
        if self._failure in {"close_error", "write_close_error"}:
            raise OSError("injected close failure")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def _program(tmp_path: Path, source: bytes | None = None) -> Path:
    program = tmp_path / "my_task.py"
    program.write_bytes(
        source
        if source is not None
        else "def main(ctx):\n    pass\n".encode("utf-8")
    )
    return program


def _create_evidence(
    tmp_path: Path,
    *,
    run_id: str | None = "run-fixed",
) -> StudentRunEvidence:
    return StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=_program(tmp_path),
        robot_backend="sim",
        run_id=run_id,
    )


def _finalize(evidence: StudentRunEvidence) -> Path:
    return evidence.finalize(
        status="PASS",
        command_count=1,
        last_pose_mm=(100, 20, 120),
        safety_violation_count=0,
        error=None,
        cleanup_errors=[],
    )


def _is_atomic_temp(path: Path, target: Path) -> bool:
    return (
        path.parent == target.parent
        and path.name.startswith(f".{target.stem}-")
        and path.name.endswith(".tmp")
    )


def _atomic_temps(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.stem}-*.tmp"))


def _inject_atomic_failure(
    monkeypatch,
    target: Path,
    failure: str,
    *,
    cleanup_error: bool = False,
) -> None:
    original_open = Path.open
    original_replace = Path.replace
    original_unlink = Path.unlink

    def injected_open(path: Path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if _is_atomic_temp(path, target) and mode == "xb":
            return _FaultingAtomicStream(stream, failure)
        return stream

    def injected_replace(path: Path, destination):
        if (
            _is_atomic_temp(path, target)
            and Path(destination) == target
            and failure == "replace_error"
        ):
            raise OSError("injected replace failure")
        return original_replace(path, destination)

    def injected_unlink(path: Path, *args, **kwargs):
        if _is_atomic_temp(path, target) and cleanup_error:
            raise PermissionError("injected temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", injected_open)
    monkeypatch.setattr(Path, "replace", injected_replace)
    monkeypatch.setattr(Path, "unlink", injected_unlink)


def _capture_thread_result(results, name: str, action) -> None:
    try:
        results.put((name, "ok", action()))
    except Exception as error:
        results.put((name, "error", error))


def test_evidence_captures_source_and_writes_complete_summary(tmp_path):
    program = _program(tmp_path)
    source_bytes = program.read_bytes()
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="run-fixed",
    )

    evidence.record_command(
        {
            "timestamp": "student-forged",
            "command_id": "000001",
            "name": "robot.home",
            "args": {},
        }
    )
    evidence.record_event("RUNNING", "程序开始", phase="student")
    summary_path = _finalize(evidence)

    digest = hashlib.sha256(source_bytes).hexdigest()
    assert evidence.directory == (tmp_path / "runs" / "run-fixed").resolve()
    assert evidence.program_path == program.resolve()
    assert evidence.source_path == evidence.directory / "source.py"
    assert evidence.source_sha256 == digest
    assert evidence.robot_backend == "sim"
    assert evidence.run_id == "run-fixed"
    assert evidence.started_at
    assert evidence.started_monotonic > 0
    assert evidence.source_path.read_bytes() == source_bytes
    assert (
        (evidence.directory / "source.sha256")
        .read_text(encoding="utf-8")
        .strip()
        == digest
    )

    manifest = json.loads(
        (evidence.directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == {
        "schema_version": 1,
        "run_id": "run-fixed",
        "program_path": str(program.resolve()),
        "source_sha256": digest,
        "robot_backend": "sim",
        "hardware_status": "PENDING_HARDWARE",
    }

    command_bytes = (evidence.directory / "commands.jsonl").read_bytes()
    event_bytes = (evidence.directory / "events.jsonl").read_bytes()
    assert command_bytes.endswith(b"\n")
    assert event_bytes.endswith(b"\n")
    assert b"\r\n" not in command_bytes
    assert b"\r\n" not in event_bytes
    command = json.loads(command_bytes)
    event = json.loads(event_bytes)
    assert command["timestamp"] != "student-forged"
    assert command["command_id"] == "000001"
    assert event["state"] == "RUNNING"
    assert event["message"] == "程序开始"
    assert event["details"] == {"phase": "student"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary) == {
        "schema_version",
        "run_id",
        "status",
        "program_path",
        "source_sha256",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "command_count",
        "last_pose_mm",
        "safety_violation_count",
        "error",
        "cleanup_errors",
        "robot_backend",
        "hardware_status",
    }
    assert summary["schema_version"] == 1
    assert summary["run_id"] == "run-fixed"
    assert summary["status"] == "PASS"
    assert summary["program_path"] == str(program.resolve())
    assert summary["source_sha256"] == digest
    assert summary["started_at"] == evidence.started_at
    assert summary["finished_at"]
    assert summary["elapsed_seconds"] >= 0
    assert summary["command_count"] == 1
    assert summary["last_pose_mm"] == [100, 20, 120]
    assert summary["safety_violation_count"] == 0
    assert summary["error"] is None
    assert summary["cleanup_errors"] == []
    assert summary["robot_backend"] == "sim"
    assert summary["hardware_status"] == "PENDING_HARDWARE"


def test_create_reads_source_once_and_snapshot_survives_source_change(
    tmp_path, monkeypatch
):
    source_at_create = (
        "# 学生程序\r\ndef main(ctx):\r\n    ctx.log('开始')\r\n"
    ).encode("utf-8")
    program = _program(tmp_path, source_at_create)
    original_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def counting_read_bytes(path: Path) -> bytes:
        if path.resolve() == program.resolve():
            reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="immutable-source",
    )
    program.write_text(
        "def main(ctx):\n    ctx.log('changed')\n",
        encoding="utf-8",
    )

    snapshot = evidence.source_path.read_bytes()
    assert len(reads) == 1
    assert snapshot == source_at_create
    assert evidence.source_sha256 == hashlib.sha256(snapshot).hexdigest()
    assert (
        evidence.directory / "source.sha256"
    ).read_text(encoding="utf-8").strip() == hashlib.sha256(snapshot).hexdigest()


def test_create_immediately_creates_empty_jsonl_files(tmp_path):
    evidence = _create_evidence(tmp_path)

    assert (evidence.directory / "commands.jsonl").read_bytes() == b""
    assert (evidence.directory / "events.jsonl").read_bytes() == b""


def test_jsonl_serialization_errors_do_not_append_partial_lines(tmp_path):
    evidence = _create_evidence(tmp_path)
    commands_path = evidence.directory / "commands.jsonl"
    events_path = evidence.directory / "events.jsonl"
    evidence.record_command({"command_id": "000001", "value": 1})
    evidence.record_event("RUNNING", "valid")
    commands_before = commands_path.read_bytes()
    events_before = events_path.read_bytes()

    with pytest.raises(ValueError):
        evidence.record_command(
            {"command_id": "000002", "value": float("nan")}
        )
    with pytest.raises(TypeError):
        evidence.record_event("RUNNING", "invalid", payload=object())

    assert commands_path.read_bytes() == commands_before
    assert events_path.read_bytes() == events_before
    assert len(commands_before.splitlines()) == 1
    assert len(events_before.splitlines()) == 1


@pytest.mark.parametrize(
    "failure",
    [
        "short_write",
        "write_error",
        "flush_error",
        "close_error",
        "replace_error",
    ],
)
def test_jsonl_atomic_io_failures_leave_target_unchanged(
    tmp_path, monkeypatch, failure
):
    evidence = _create_evidence(tmp_path)
    commands_path = evidence.directory / "commands.jsonl"
    evidence.record_command({"command_id": "000001", "value": "baseline"})
    before = commands_path.read_bytes()
    _inject_atomic_failure(monkeypatch, commands_path, failure)

    with pytest.raises(OSError):
        evidence.record_command(
            {"command_id": "000002", "value": "must stay atomic"}
        )

    assert commands_path.read_bytes() == before
    assert _atomic_temps(commands_path) == []


def test_jsonl_cleanup_failure_keeps_target_and_chains_write_error(
    tmp_path, monkeypatch
):
    evidence = _create_evidence(tmp_path)
    commands_path = evidence.directory / "commands.jsonl"
    evidence.record_command({"command_id": "000001", "value": "baseline"})
    before = commands_path.read_bytes()
    _inject_atomic_failure(
        monkeypatch,
        commands_path,
        "write_error",
        cleanup_error=True,
    )

    with pytest.raises(OSError, match="injected write failure") as exc:
        evidence.record_command({"command_id": "000002"})

    assert isinstance(exc.value.__cause__, PermissionError)
    assert "temporary cleanup" in str(exc.value.__cause__)
    assert commands_path.read_bytes() == before
    assert len(_atomic_temps(commands_path)) == 1


def test_double_cleanup_failure_preserves_original_without_exception_group(
    tmp_path, monkeypatch
):
    evidence = _create_evidence(tmp_path)
    commands_path = evidence.directory / "commands.jsonl"
    before = commands_path.read_bytes()
    _inject_atomic_failure(
        monkeypatch,
        commands_path,
        "write_close_error",
        cleanup_error=True,
    )

    with pytest.raises(OSError, match="injected write failure") as exc:
        evidence.record_command({"command_id": "000001"})

    cause = exc.value.__cause__
    assert cause is not None
    assert not isinstance(cause, NameError)
    recovery_errors = getattr(cause, "errors", ())
    assert len(recovery_errors) == 2
    assert "injected close failure" in str(recovery_errors[0])
    assert "temporary cleanup" in str(recovery_errors[1])
    assert commands_path.read_bytes() == before


@pytest.mark.parametrize(
    "failure",
    [
        "short_write",
        "write_error",
        "flush_error",
        "close_error",
        "replace_error",
    ],
)
def test_summary_atomic_failure_cleans_temp_and_allows_retry(
    tmp_path, monkeypatch, failure
):
    evidence = _create_evidence(tmp_path)
    summary_path = evidence.directory / "summary.json"
    _inject_atomic_failure(monkeypatch, summary_path, failure)

    with pytest.raises(OSError):
        _finalize(evidence)

    assert not summary_path.exists()
    assert _atomic_temps(summary_path) == []
    monkeypatch.undo()
    assert _finalize(evidence) == summary_path
    json.loads(summary_path.read_text(encoding="utf-8"))


def test_failed_finalize_is_atomic_and_can_be_retried(tmp_path):
    evidence = _create_evidence(tmp_path)
    summary_path = evidence.directory / "summary.json"

    with pytest.raises(ValueError):
        evidence.finalize(
            status="FAILED",
            command_count=0,
            last_pose_mm=(100, 20, float("nan")),
            safety_violation_count=0,
            error="invalid pose",
            cleanup_errors=[],
        )

    assert not summary_path.exists()
    assert _finalize(evidence) == summary_path
    json.loads(summary_path.read_text(encoding="utf-8"))


def test_successful_finalize_seals_evidence(tmp_path):
    evidence = _create_evidence(tmp_path)
    summary_path = _finalize(evidence)
    summary_before = summary_path.read_bytes()
    commands_before = (evidence.directory / "commands.jsonl").read_bytes()
    events_before = (evidence.directory / "events.jsonl").read_bytes()

    with pytest.raises(RuntimeError, match="finalized"):
        _finalize(evidence)
    with pytest.raises(RuntimeError, match="finalized"):
        evidence.record_command({"command_id": "000002"})
    with pytest.raises(RuntimeError, match="finalized"):
        evidence.record_event("RUNNING", "too late")

    assert summary_path.read_bytes() == summary_before
    assert (
        evidence.directory / "commands.jsonl"
    ).read_bytes() == commands_before
    assert (evidence.directory / "events.jsonl").read_bytes() == events_before


def test_concurrent_finalize_writes_summary_exactly_once(
    tmp_path, monkeypatch
):
    evidence = _create_evidence(tmp_path)
    real_atomic_write = evidence_module._atomic_write_bytes
    first_writer_entered = threading.Event()
    second_writer_entered = threading.Event()
    release_first_writer = threading.Event()
    second_thread_started = threading.Event()
    call_lock = threading.Lock()
    atomic_call_count = 0

    def blocking_atomic_write(path: Path, payload: bytes) -> None:
        nonlocal atomic_call_count
        with call_lock:
            atomic_call_count += 1
            call_number = atomic_call_count
        if call_number == 1:
            first_writer_entered.set()
            if not release_first_writer.wait(5):
                raise TimeoutError("first atomic writer was not released")
        else:
            second_writer_entered.set()
        real_atomic_write(path, payload)

    monkeypatch.setattr(
        evidence_module,
        "_atomic_write_bytes",
        blocking_atomic_write,
    )
    results: queue.Queue = queue.Queue()
    first = threading.Thread(
        target=_capture_thread_result,
        args=(results, "first", lambda: _finalize(evidence)),
    )

    def run_second_finalize() -> None:
        second_thread_started.set()
        _capture_thread_result(
            results,
            "second",
            lambda: _finalize(evidence),
        )

    second = threading.Thread(target=run_second_finalize)
    first.start()
    assert first_writer_entered.wait(2)
    second.start()
    assert second_thread_started.wait(2)
    second_reached_writer_early = second_writer_entered.wait(0.5)
    release_first_writer.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_reached_writer_early is False
    outcomes = [results.get_nowait(), results.get_nowait()]
    assert sum(outcome[1] == "ok" for outcome in outcomes) == 1
    errors = [outcome[2] for outcome in outcomes if outcome[1] == "error"]
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "finalized" in str(errors[0])
    assert atomic_call_count == 1
    assert (evidence.directory / "summary.json").is_file()


def test_finalize_race_prevents_append_after_summary(
    tmp_path, monkeypatch
):
    evidence = _create_evidence(tmp_path)
    commands_path = evidence.directory / "commands.jsonl"
    commands_before = commands_path.read_bytes()
    real_atomic_write = evidence_module._atomic_write_bytes
    finalize_writer_entered = threading.Event()
    append_writer_entered = threading.Event()
    release_finalize_writer = threading.Event()
    append_thread_started = threading.Event()

    def blocking_atomic_write(path: Path, payload: bytes) -> None:
        if path.name == "summary.json":
            finalize_writer_entered.set()
            if not release_finalize_writer.wait(5):
                raise TimeoutError("finalize writer was not released")
        elif path.name == "commands.jsonl":
            append_writer_entered.set()
        real_atomic_write(path, payload)

    monkeypatch.setattr(
        evidence_module,
        "_atomic_write_bytes",
        blocking_atomic_write,
    )
    results: queue.Queue = queue.Queue()
    finalize_thread = threading.Thread(
        target=_capture_thread_result,
        args=(results, "finalize", lambda: _finalize(evidence)),
    )

    def run_append() -> None:
        append_thread_started.set()
        _capture_thread_result(
            results,
            "append",
            lambda: evidence.record_command({"command_id": "too-late"}),
        )

    append_thread = threading.Thread(target=run_append)
    finalize_thread.start()
    assert finalize_writer_entered.wait(2)
    append_thread.start()
    assert append_thread_started.wait(2)
    append_reached_writer_early = append_writer_entered.wait(0.5)
    release_finalize_writer.set()
    finalize_thread.join(5)
    append_thread.join(5)

    assert not finalize_thread.is_alive()
    assert not append_thread.is_alive()
    assert append_reached_writer_early is False
    outcomes = [results.get_nowait(), results.get_nowait()]
    finalize_outcome = next(item for item in outcomes if item[0] == "finalize")
    append_outcome = next(item for item in outcomes if item[0] == "append")
    assert finalize_outcome[1] == "ok"
    assert append_outcome[1] == "error"
    assert isinstance(append_outcome[2], RuntimeError)
    assert "finalized" in str(append_outcome[2])
    assert (evidence.directory / "summary.json").is_file()
    assert commands_path.read_bytes() == commands_before


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../escape",
        r"..\escape",
        "/absolute",
        r"\rooted",
        "nested/name",
        r"nested\name",
        "C:drive-relative",
        "bad:name",
        "trailing.",
        "trailing ",
    ],
)
def test_run_id_rejects_unsafe_directory_names(tmp_path, run_id):
    program = _program(tmp_path)

    with pytest.raises(ValueError, match="run_id"):
        StudentRunEvidence.create(
            output_root=tmp_path / "runs",
            program_path=program,
            robot_backend="sim",
            run_id=run_id,
        )


def test_run_id_collision_is_rejected_and_defaults_are_unique(tmp_path):
    program = _program(tmp_path)
    output_root = tmp_path / "runs"
    first_fixed = StudentRunEvidence.create(
        output_root=output_root,
        program_path=program,
        robot_backend="sim",
        run_id="run-fixed",
    )

    with pytest.raises(FileExistsError):
        StudentRunEvidence.create(
            output_root=output_root,
            program_path=program,
            robot_backend="sim",
            run_id="run-fixed",
        )

    first_default = StudentRunEvidence.create(
        output_root=output_root,
        program_path=program,
        robot_backend="sim",
    )
    second_default = StudentRunEvidence.create(
        output_root=output_root,
        program_path=program,
        robot_backend="sim",
    )
    assert first_fixed.directory.is_dir()
    assert first_default.run_id != second_default.run_id
    assert first_default.directory != second_default.directory
    assert first_default.directory.is_dir()
    assert second_default.directory.is_dir()
