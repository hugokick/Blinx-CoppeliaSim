from __future__ import annotations

import multiprocessing
import os
import sys
from multiprocessing import Pipe
from pathlib import Path
from threading import Thread
from time import monotonic

import pytest

from vision_platform.student.protocol import ResponseMessage
from vision_platform.student.runner import _child_entry
from vision_platform.student.validator import validate_program
from vision_platform.student.worker import run_student_worker


class RecordingConnection:
    def __init__(
        self,
        *,
        status: str = "PASS",
        error: dict | None = None,
    ) -> None:
        self.status = status
        self.error = error
        self.sent: list[dict] = []
        self.closed = False

    def send(self, payload: dict) -> None:
        self.sent.append(payload)

    def recv(self) -> dict:
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"],
            status=self.status,
            value=None,
            error=self.error,
        ).to_dict()

    def close(self) -> None:
        self.closed = True


def _child_entry_without_stderr(*args) -> None:
    sys.stderr = None
    _child_entry(*args)


def _registered_modules_for(path: Path) -> list[str]:
    selected = str(path.resolve())
    return [
        name
        for name, module in sys.modules.items()
        if getattr(module, "__file__", None) == selected
    ]


def test_worker_imports_program_and_reports_completion(tmp_path: Path) -> None:
    program = tmp_path / "program.py"
    program.write_text(
        "def main(ctx):\n"
        "    ctx.log('worker-ok')\n"
        "    ctx.robot.home()\n",
        encoding="utf-8",
    )
    parent, child = Pipe(duplex=True)
    result: dict = {}

    def worker() -> None:
        result.update(run_student_worker(program, child))

    thread = Thread(target=worker, daemon=True)
    thread.start()
    names: list[str] = []
    deadline = monotonic() + 2.0
    timed_out = False
    try:
        while thread.is_alive() and monotonic() < deadline:
            if parent.poll(0.05):
                try:
                    command = parent.recv()
                except EOFError:
                    break
                names.append(command["name"])
                parent.send(
                    ResponseMessage(
                        command_id=command["command_id"],
                        status="PASS",
                        value=None,
                        error=None,
                    ).to_dict()
                )
        thread.join(timeout=max(0.0, deadline - monotonic()))
        timed_out = thread.is_alive()
    finally:
        parent.close()
        child.close()
        thread.join(timeout=0.2)

    assert timed_out is False
    assert result["status"] == "PASS"
    assert names == ["context.log", "robot.home"]


def test_worker_reports_uncaught_exception(tmp_path: Path) -> None:
    program = tmp_path / "broken.py"
    program.write_text(
        "def main(ctx):\n    raise ValueError('student-error')\n",
        encoding="utf-8",
    )
    parent, child = Pipe(duplex=True)

    result = run_student_worker(program, child)
    parent.close()

    assert result["status"] == "FAIL"
    assert result["error"]["code"] == "STUDENT_PROGRAM_FAILED"
    assert result["error"]["type"] == "ValueError"
    assert "student-error" in result["error"]["message"]
    assert "ValueError: student-error" in result["error"]["traceback"]


def test_worker_routes_module_and_student_stdout_to_stderr(
    tmp_path: Path,
    capsys,
) -> None:
    program = tmp_path / "prints.py"
    program.write_text(
        "print('module-diagnostic')\n"
        "def main(ctx):\n"
        "    print('student-diagnostic')\n",
        encoding="utf-8",
    )
    connection = RecordingConnection()

    result = run_student_worker(program, connection)

    captured = capsys.readouterr()
    assert result == {"status": "PASS", "error": None}
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "module-diagnostic",
        "student-diagnostic",
    ]


def test_worker_discards_student_stdout_when_gui_has_no_stderr(
    tmp_path: Path,
    capsys,
) -> None:
    program = tmp_path / "gui_prints.py"
    program.write_text(
        "def main(ctx):\n"
        "    print('student-diagnostic')\n",
        encoding="utf-8",
    )
    connection = RecordingConnection()
    original_stderr = sys.stderr
    try:
        sys.stderr = None
        result = run_student_worker(program, connection)
    finally:
        sys.stderr = original_stderr

    captured = capsys.readouterr()
    assert result == {"status": "PASS", "error": None}
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("stderr_available", [True, False])
def test_child_routes_late_non_daemon_thread_for_process_lifetime(
    tmp_path: Path,
    capfd,
    stderr_available: bool,
) -> None:
    release = tmp_path / "release-late-thread"
    program = tmp_path / "late_thread.py"
    program.write_text(
        "import threading\n"
        "import time\n"
        "from pathlib import Path\n"
        f"RELEASE = Path({str(release)!r})\n"
        "\n"
        "def late_print():\n"
        "    while not RELEASE.exists():\n"
        "        time.sleep(0.005)\n"
        "    print('late-student-thread', flush=True)\n"
        "\n"
        "def main(ctx):\n"
        "    thread = threading.Thread(target=late_print, daemon=False)\n"
        "    thread.start()\n",
        encoding="utf-8",
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    result_queue = context.Queue(maxsize=1)
    target = (
        _child_entry
        if stderr_available
        else _child_entry_without_stderr
    )
    process = context.Process(
        target=target,
        args=(
            str(program),
            program.read_bytes(),
            child,
            result_queue,
        ),
    )
    process.start()
    child.close()
    try:
        result = result_queue.get(timeout=5)
        release.write_text("release", encoding="utf-8")
        process.join(timeout=5)
        assert process.is_alive() is False
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        parent.close()
        result_queue.close()
        result_queue.join_thread()

    captured = capfd.readouterr()
    assert result == {"status": "PASS", "error": None}
    assert captured.out == ""
    if stderr_available:
        assert "late-student-thread" in captured.err
    else:
        assert captured.err == ""


def test_worker_routes_loud_exception_stringification_to_stderr(
    tmp_path: Path,
    capsys,
) -> None:
    program = tmp_path / "loud_exception.py"
    program.write_text(
        "class LoudError(Exception):\n"
        "    def __str__(self):\n"
        "        print('loud-exception-string')\n"
        "        return 'student-error'\n"
        "\n"
        "def main(ctx):\n"
        "    raise LoudError()\n",
        encoding="utf-8",
    )

    result = run_student_worker(program, RecordingConnection())

    captured = capsys.readouterr()
    assert result["status"] == "FAIL"
    assert result["error"]["message"] == "student-error"
    assert captured.out == ""
    assert "loud-exception-string" in captured.err


def test_worker_discards_loud_exception_stringification_without_stderr(
    tmp_path: Path,
    capsys,
) -> None:
    program = tmp_path / "loud_exception_gui.py"
    program.write_text(
        "class LoudError(Exception):\n"
        "    def __str__(self):\n"
        "        print('loud-exception-string')\n"
        "        return 'student-error'\n"
        "\n"
        "def main(ctx):\n"
        "    raise LoudError()\n",
        encoding="utf-8",
    )
    original_stderr = sys.stderr
    try:
        sys.stderr = None
        result = run_student_worker(program, RecordingConnection())
    finally:
        sys.stderr = original_stderr

    captured = capsys.readouterr()
    assert result["status"] == "FAIL"
    assert result["error"]["message"] == "student-error"
    assert captured.out == ""
    assert captured.err == ""


def test_worker_executes_fresh_source_with_same_mtime_and_size(
    tmp_path: Path,
) -> None:
    program = tmp_path / "program.py"
    source_a = "def main(ctx):\n    ctx.log('version-a')\n"
    source_b = "def main(ctx):\n    ctx.log('version-b')\n"
    assert len(source_a.encode("utf-8")) == len(source_b.encode("utf-8"))
    fixed_ns = 1_700_000_000_000_000_000
    messages: list[str] = []

    for source in (source_a, source_b):
        program.write_text(source, encoding="utf-8")
        os.utime(program, ns=(fixed_ns, fixed_ns))
        connection = RecordingConnection()

        result = run_student_worker(program, connection)

        assert result == {"status": "PASS", "error": None}
        assert connection.closed is True
        messages.append(connection.sent[0]["args"]["message"])

    assert messages == ["version-a", "version-b"]
    assert not (tmp_path / "__pycache__").exists()


def test_worker_supports_valid_dataclass_and_uses_fresh_module_names(
    tmp_path: Path,
) -> None:
    program = tmp_path / "dataclass_program.py"
    program.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class Packet:\n"
        "    value: int\n"
        "\n"
        "def main(ctx):\n"
        "    ctx.log(__name__ + '|' + str(Packet(7)))\n",
        encoding="utf-8",
    )
    assert validate_program(program).ok is True
    logged: list[str] = []

    for _ in range(2):
        connection = RecordingConnection()

        result = run_student_worker(program, connection)

        assert result == {"status": "PASS", "error": None}
        logged.append(connection.sent[0]["args"]["message"])
        assert _registered_modules_for(program) == []

    module_names = [message.split("|", 1)[0] for message in logged]
    assert module_names[0] != module_names[1]
    assert all(name.isidentifier() for name in module_names)
    assert [message.split("|", 1)[1] for message in logged] == [
        "Packet(value=7)",
        "Packet(value=7)",
    ]


def test_worker_cleans_module_registration_when_exec_fails(
    tmp_path: Path,
) -> None:
    program = tmp_path / "import_failure.py"
    program.write_text(
        "raise RuntimeError('import-error')\n"
        "def main(ctx):\n"
        "    pass\n",
        encoding="utf-8",
    )
    connection = RecordingConnection()

    result = run_student_worker(program, connection)

    assert result["status"] == "FAIL"
    assert "import-error" in result["error"]["message"]
    assert _registered_modules_for(program) == []


def test_worker_maps_cancelled_response_to_standard_cancellation(
    tmp_path: Path,
) -> None:
    program = tmp_path / "cancelled_response.py"
    program.write_text(
        "def main(ctx):\n    ctx.log('cancelled')\n",
        encoding="utf-8",
    )
    connection = RecordingConnection(
        status="CANCELLED",
        error={"code": "STOP_REQUESTED", "message": "控制器请求停止"},
    )

    result = run_student_worker(program, connection)

    assert result == {
        "status": "CANCELLED",
        "error": {
            "code": "STUDENT_PROGRAM_CANCELLED",
            "message": "学生程序通信已取消",
        },
    }
    assert connection.closed is True


class CancellingConnection:
    def __init__(self) -> None:
        self.closed = False

    def send(self, payload: dict) -> None:
        raise BrokenPipeError

    def recv(self) -> dict:
        raise AssertionError("recv must not be reached")

    def close(self) -> None:
        self.closed = True


def test_worker_reports_cancelled_connection_and_closes_it(tmp_path: Path) -> None:
    program = tmp_path / "cancelled.py"
    program.write_text(
        "def main(ctx):\n    ctx.log('cancelled')\n",
        encoding="utf-8",
    )
    connection = CancellingConnection()

    result = run_student_worker(program, connection)

    assert result == {
        "status": "CANCELLED",
        "error": {
            "code": "STUDENT_PROGRAM_CANCELLED",
            "message": "学生程序通信已取消",
        },
    }
    assert connection.closed is True


def test_worker_compiles_supplied_captured_bytes_not_mutated_disk_source(
    tmp_path: Path,
) -> None:
    program = tmp_path / "captured.py"
    captured = (
        b"def main(ctx):\n"
        b"    ctx.log('captured-version')\n"
    )
    program.write_text(
        "def main(ctx):\n"
        "    ctx.log('mutated-disk-version')\n",
        encoding="utf-8",
    )
    connection = RecordingConnection()

    result = run_student_worker(
        program,
        connection,
        source_bytes=captured,
    )

    assert result == {"status": "PASS", "error": None}
    assert connection.sent[0]["args"]["message"] == "captured-version"
    assert connection.closed is True
