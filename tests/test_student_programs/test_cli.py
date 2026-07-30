from __future__ import annotations

import json
import runpy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import vision_platform.application as application_module
import vision_platform.cli as cli_module
import vision_platform.config as config_module
import vision_platform.session as session_module
import vision_platform.student.runner as runner_module
from vision_platform.cli import build_parser, main
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.student.safety import (
    StudentExecutionPolicy,
    StudentMotionGuard,
)
from vision_platform.student.validator import (
    ValidationIssue,
    ValidationResult,
    validate_program,
)


def _strict_json(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
    )
    return json.loads(
        encoded,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )


def _one_json_line(capsys):
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    return json.loads(
        lines[0],
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )


def _student_config():
    return {
        "speed_range": [2, 24],
        "max_runtime_s": 12,
        "max_commands": 80,
        "command_timeout_s": 3,
        "max_sleep_s": 2,
        "tool_on_max_z_mm": 30,
    }


class _CliSocket:
    def __init__(self) -> None:
        self.RCVTIMEO = 5000
        self.SNDTIMEO = 7000
        self.close_calls: list[int | None] = []

    def close(self, linger=None) -> None:
        self.close_calls.append(linger)


class _CliContext:
    def __init__(self) -> None:
        self.term_calls = 0

    def term(self) -> None:
        self.term_calls += 1


class _CliRobot:
    def __init__(self) -> None:
        self.pose_mm = (100.0, 20.0, 120.0)
        self.close_calls = 0

    def current_world_pose(self):
        return self.pose_mm

    def move_world(self, x, y, z, *, speed):
        del speed
        self.pose_mm = (float(x), float(y), float(z))

    def move_home(self):
        self.pose_mm = (100.0, 20.0, 120.0)

    def close(self):
        self.close_calls += 1


class _CliTool:
    def __init__(
        self,
        *,
        cleanup_entered: threading.Event,
        cleanup_release: threading.Event | None,
        late_stdout: str | None = None,
    ) -> None:
        self.cleanup_entered = cleanup_entered
        self.cleanup_release = cleanup_release
        self.late_stdout = late_stdout
        self.off_calls = 0

    def on(self):
        return None

    def off(self):
        self.off_calls += 1
        if self.cleanup_release is not None:
            self.cleanup_entered.set()
            assert self.cleanup_release.wait(timeout=5)
            if self.late_stdout is not None:
                print(self.late_stdout, flush=True)


def _install_real_controller_runtime(
    monkeypatch,
    tmp_path,
    *,
    block_cleanup=False,
    force_wait_error=False,
    runtime_stdout=None,
    late_cleanup_stdout=None,
):
    cleanup_entered = threading.Event()
    cleanup_release = threading.Event() if block_cleanup else None
    state = SimpleNamespace(
        applications=[],
        sessions=[],
        controllers=[],
        cleanup_entered=cleanup_entered,
        cleanup_release=cleanup_release,
    )
    student = _student_config()
    student.update(
        {
            "max_runtime_s": 0.5,
            "command_timeout_s": 0.05,
        }
    )
    config = SimpleNamespace(
        student=student,
        coppelia_scene=tmp_path / "scene.ttt",
    )

    class ActualLikeApplication:
        def __init__(self) -> None:
            self.config = SimpleNamespace(robot_backend="sim")
            self.workspace = WorkspacePolicy(
                x_mm=(20, 140),
                y_mm=(-90, 90),
                z_mm=(10, 140),
                safe_z_mm=100,
            )
            self.robot = _CliRobot()
            self.tool = _CliTool(
                cleanup_entered=cleanup_entered,
                cleanup_release=cleanup_release,
                late_stdout=late_cleanup_stdout,
            )
            self.camera = SimpleNamespace(close=lambda: None)
            self.socket = _CliSocket()
            self.context = _CliContext()
            self.client = SimpleNamespace(
                timeout=600.0,
                sendCnt=2,
                socket=self.socket,
                context=self.context,
            )
            self.sim = object()
            self.close_calls = 0
            self.close_quarantined_calls = 0
            self._closed = False

        @classmethod
        def from_config(cls, received):
            assert received is config
            application = cls()
            state.applications.append(application)
            return application

        def load_and_start_scene(self, scene):
            assert scene == config.coppelia_scene

        def open(self):
            if runtime_stdout is not None:
                print(runtime_stdout)

        def _close_transport(self):
            client = self.client
            if client is None:
                return
            client.socket.close(linger=0)
            client.context.term()
            self.client = None
            self.sim = None
            self._closed = True

        def close(self):
            if self._closed:
                return
            self.close_calls += 1
            self.tool.off()
            self.robot.close()
            self._close_transport()

        def close_quarantined(self):
            if self._closed:
                return
            self.close_quarantined_calls += 1
            self.robot.close()
            self._close_transport()

    base_session = session_module.VisionLabSession

    class RecordingSession(base_session):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            state.sessions.append(self)

    base_controller = runner_module.StudentProgramController

    class RecordingController(base_controller):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._inject_wait_error = force_wait_error
            state.controllers.append(self)

        def wait(self, timeout_s=None):
            if self._inject_wait_error:
                self._inject_wait_error = False
                raise TimeoutError("injected cli wait timeout")
            return super().wait(timeout_s)

        def cancel(self):
            super().cancel()
            if cleanup_release is not None:
                assert cleanup_entered.wait(timeout=2)

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        application_module,
        "VisionLabApplication",
        ActualLikeApplication,
    )
    monkeypatch.setattr(
        session_module,
        "VisionLabSession",
        RecordingSession,
    )
    monkeypatch.setattr(
        runner_module,
        "StudentProgramController",
        RecordingController,
    )
    return state


def _wait_until(predicate, timeout_s=2):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.005)
    raise AssertionError("condition did not become true before timeout")


def _install_fake_runtime(
    monkeypatch,
    tmp_path,
    *,
    validation,
    result=None,
    fail_stage=None,
    quiescent=True,
    runtime_stdout=None,
):
    state = SimpleNamespace(
        operations=[],
        applications=[],
        sessions=[],
        controllers=[],
        load_config_calls=[],
    )
    config = SimpleNamespace(
        student=_student_config(),
        coppelia_scene=tmp_path / "configured scene.ttt",
    )

    def fake_load_config(
        config_path,
        *,
        project_root,
        environ,
    ):
        state.load_config_calls.append(
            {
                "config_path": config_path,
                "project_root": project_root,
                "environ": dict(environ),
            }
        )
        if fail_stage == "config":
            raise RuntimeError("config exploded")
        return config

    class FakeApplication:
        def __init__(self):
            self.close_calls = 0
            self.close_quarantined_calls = 0

        @classmethod
        def from_config(cls, received):
            assert received is config
            if fail_stage == "application":
                raise RuntimeError("application exploded")
            application = cls()
            state.applications.append(application)
            state.operations.append("application.create")
            return application

        def load_and_start_scene(self, scene):
            state.operations.append(("application.load_scene", scene))
            if fail_stage == "load_scene":
                raise RuntimeError("scene exploded")

        def open(self):
            state.operations.append("application.open")
            if runtime_stdout is not None:
                print(runtime_stdout)
            if fail_stage == "open":
                raise RuntimeError("open exploded")

        def close(self):
            self.close_calls += 1
            state.operations.append("application.close")

        def close_quarantined(self):
            self.close_quarantined_calls += 1
            state.operations.append("application.close_quarantined")

    class FakeSession:
        def __init__(self, *, application, factory):
            self.application = application
            self.factory = factory
            self.close_calls = 0
            self.close_quarantined_calls = 0
            state.sessions.append(self)

        def close(self):
            self.close_calls += 1
            self.application.close()

        def close_quarantined(self):
            self.close_quarantined_calls += 1
            self.application.close_quarantined()

    class FakeController:
        def __init__(
            self,
            *,
            session,
            execution_policy,
            output_root,
        ):
            self.session = session
            self.policy = execution_policy
            self.output_root = output_root
            self.process_is_alive = False
            self.cancel_calls = 0
            state.controllers.append(self)

        def load(self, program):
            state.operations.append(("controller.load", program))
            if fail_stage == "load":
                raise RuntimeError("load exploded")

        def validate(self):
            state.operations.append("controller.validate")
            return validation

        def start(self):
            state.operations.append("controller.start")
            if fail_stage == "start":
                raise RuntimeError("start exploded")

        def wait(self, *, timeout_s):
            state.operations.append(("controller.wait", timeout_s))
            if fail_stage == "wait":
                raise RuntimeError("wait exploded")
            return result

        def cancel(self):
            self.cancel_calls += 1
            state.operations.append("controller.cancel")

        def wait_for_quiescence(self, timeout_s):
            state.operations.append(
                ("controller.wait_for_quiescence", timeout_s)
            )
            return quiescent

    monkeypatch.setattr(config_module, "load_config", fake_load_config)
    monkeypatch.setattr(
        application_module,
        "VisionLabApplication",
        FakeApplication,
    )
    monkeypatch.setattr(session_module, "VisionLabSession", FakeSession)
    monkeypatch.setattr(
        runner_module,
        "StudentProgramController",
        FakeController,
    )
    return state, config


def test_student_validate_parser():
    args = build_parser().parse_args(
        ["student-validate", "--program", "student_programs/my_task.py"]
    )

    assert args.command == "student-validate"


def test_student_run_parser_accepts_only_sim():
    parser = build_parser()
    args = parser.parse_args(
        [
            "student-run",
            "--program",
            "student_programs/my_task.py",
            "--robot",
            "sim",
            "--output",
            "artifacts/vision_lab/student-runs",
        ]
    )

    assert args.robot == "sim"
    with pytest.raises(SystemExit) as rejected:
        parser.parse_args(
            [
                "student-run",
                "--program",
                "student_programs/my_task.py",
                "--robot",
                "real",
            ]
        )
    assert rejected.value.code == 2


@pytest.mark.parametrize(
    ("source", "expected_status", "expected_exit"),
    [
        ("def main(ctx):\n    ctx.log('ok')\n", "PASS", 0),
        ("def main(ctx)\n    pass\n", "FAIL", 2),
    ],
)
def test_student_validate_prints_one_strict_json_line(
    tmp_path,
    capsys,
    source,
    expected_status,
    expected_exit,
):
    program = tmp_path / "学生程序.py"
    program.write_text(source, encoding="utf-8")

    exit_code = main(["student-validate", "--program", str(program)])

    payload = _one_json_line(capsys)
    assert exit_code == expected_exit
    assert payload["status"] == expected_status
    assert payload["program"] == str(program.resolve())
    assert bool(payload["issues"]) is (expected_status == "FAIL")
    _strict_json(payload)


def test_student_run_success_forces_sim_and_closes_session_once(
    tmp_path,
    monkeypatch,
    capsys,
):
    project_root = tmp_path / "project with spaces"
    project_root.mkdir()
    program = tmp_path / "student program.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    summary = tmp_path / "runs" / "summary.json"
    evidence = tmp_path / "runs"
    validation = ValidationResult(program.resolve(), True, ())
    result = SimpleNamespace(
        status="PASS",
        summary_path=summary,
        evidence_dir=evidence,
        error=None,
    )
    state, config = _install_fake_runtime(
        monkeypatch,
        tmp_path,
        validation=validation,
        result=result,
    )
    monkeypatch.setattr(cli_module, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("ROBOT_BACKEND", "real")
    monkeypatch.setenv("VISION_BACKEND", "hik")

    exit_code = main(
        [
            "student-run",
            "--program",
            str(program),
            "--config",
            "config/custom.json",
            "--robot",
            "sim",
            "--scene",
            "simulation/vision lab.ttt",
            "--host",
            "127.0.0.9",
            "--port",
            "23111",
            "--output",
            str(tmp_path / "output with spaces"),
        ]
    )

    payload = _one_json_line(capsys)
    load_call = state.load_config_calls[0]
    environ = load_call["environ"]
    controller = state.controllers[0]
    assert exit_code == 0
    assert payload == {
        "status": "PASS",
        "summary": str(summary),
        "evidence": str(evidence),
        "error": None,
    }
    assert load_call["config_path"] == "config/custom.json"
    assert load_call["project_root"] == project_root
    assert environ["ROBOT_BACKEND"] == "sim"
    assert environ["VISION_BACKEND"] == "sim"
    assert environ["COPPELIA_HOST"] == "127.0.0.9"
    assert environ["COPPELIA_PORT"] == "23111"
    assert environ["COPPELIA_SCENE"] == str(
        (project_root / "simulation/vision lab.ttt").resolve()
    )
    assert controller.policy.min_speed == 2
    assert controller.policy.max_speed == 24
    assert controller.policy.max_runtime_s == 12
    assert controller.policy.max_commands == 80
    assert controller.policy.command_timeout_s == 3
    assert controller.policy.max_sleep_s == 2
    assert controller.policy.tool_on_max_z_mm == 30
    assert state.operations == [
        "application.create",
        ("application.load_scene", config.coppelia_scene),
        "application.open",
        ("controller.load", str(program)),
        "controller.validate",
        "controller.start",
        ("controller.wait", 20.0),
        ("controller.wait_for_quiescence", 3.25),
        "application.close",
    ]
    assert state.sessions[0].close_calls == 1
    assert state.sessions[0].close_quarantined_calls == 0
    assert state.applications[0].close_calls == 1


def test_student_run_routes_runtime_stdout_to_stderr_and_emits_one_json(
    tmp_path,
    monkeypatch,
    capsys,
):
    program = tmp_path / "prints.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    validation = ValidationResult(program.resolve(), True, ())
    result = SimpleNamespace(
        status="PASS",
        summary_path=None,
        evidence_dir=None,
        error=None,
    )
    _install_fake_runtime(
        monkeypatch,
        tmp_path,
        validation=validation,
        result=result,
        runtime_stdout="[SIM] backend ready",
    )

    exit_code = main(["student-run", "--program", str(program)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err.splitlines() == ["[SIM] backend ready"]
    lines = captured.out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "PASS"


def test_student_run_defers_transport_close_while_cleanup_action_is_alive(
    tmp_path,
    monkeypatch,
    capsys,
):
    program = tmp_path / "cleanup_blocks.py"
    program.write_text(
        "def main(ctx):\n"
        "    raise ValueError('student failure')\n",
        encoding="utf-8",
    )
    state = _install_real_controller_runtime(
        monkeypatch,
        tmp_path,
        block_cleanup=True,
    )
    original_stdout = sys.stdout

    try:
        exit_code = main(["student-run", "--program", str(program)])
        payload = _one_json_line(capsys)
        application = state.applications[0]
        controller = state.controllers[0]
        session = state.sessions[0]
        socket = application.client.socket
        context = application.client.context
        result = controller.wait(timeout_s=0)
        sealed_summary = result.summary_path.read_bytes()
        sealed_error = result.error

        assert exit_code == 1
        assert payload["status"] == "FAIL"
        assert payload["error"]["code"] == (
            "STUDENT_SESSION_CLOSE_DEFERRED"
        )
        assert payload["error"]["details"]["original_error"]["code"] == (
            "STUDENT_BACKEND_COMMAND_STUCK"
        )
        assert state.cleanup_entered.is_set()
        assert controller.wait_for_quiescence(0) is False
        assert application.close_calls == 0
        assert application.close_quarantined_calls == 0
        assert socket.close_calls == []
        assert context.term_calls == 0
    finally:
        state.cleanup_release.set()
        if state.controllers:
            state.controllers[0].wait_for_quiescence(2)
            _wait_until(lambda: sys.stdout is original_stdout)

    assert controller.wait_for_quiescence(2) is True
    assert controller.wait(timeout_s=0).error == sealed_error
    assert result.summary_path.read_bytes() == sealed_summary
    session.close_quarantined()
    assert application.close_quarantined_calls == 1
    assert socket.close_calls == [0]
    assert context.term_calls == 1


def test_student_run_cancel_defers_close_until_cleanup_is_quiescent(
    tmp_path,
    monkeypatch,
    capsys,
):
    program = tmp_path / "cancel_blocks.py"
    program.write_text(
        "def main(ctx):\n"
        "    while True:\n"
        "        pass\n",
        encoding="utf-8",
    )
    state = _install_real_controller_runtime(
        monkeypatch,
        tmp_path,
        block_cleanup=True,
        force_wait_error=True,
    )
    original_stdout = sys.stdout

    try:
        exit_code = main(["student-run", "--program", str(program)])
        payload = _one_json_line(capsys)
        application = state.applications[0]
        controller = state.controllers[0]
        session = state.sessions[0]
        socket = application.client.socket
        context = application.client.context

        assert exit_code == 1
        assert payload["status"] == "FAIL"
        assert payload["error"]["code"] == (
            "STUDENT_SESSION_CLOSE_DEFERRED"
        )
        assert payload["error"]["details"]["original_error"]["code"] == (
            "STUDENT_RUN_FAILED"
        )
        assert state.cleanup_entered.is_set()
        assert controller.wait_for_quiescence(0) is False
        assert application.close_calls == 0
        assert application.close_quarantined_calls == 0
        assert socket.close_calls == []
        assert context.term_calls == 0
    finally:
        state.cleanup_release.set()
        if state.controllers:
            state.controllers[0].wait_for_quiescence(2)
            _wait_until(lambda: sys.stdout is original_stdout)

    assert controller.wait_for_quiescence(2) is True
    terminal = controller.wait(timeout_s=2)
    assert terminal.status == "FAILED"
    assert terminal.error["code"] == "STUDENT_BACKEND_COMMAND_STUCK"
    session.close_quarantined()
    assert application.close_quarantined_calls == 1
    assert socket.close_calls == [0]
    assert context.term_calls == 1


@pytest.mark.parametrize("stderr_available", [True, False])
def test_student_run_deferred_late_backend_stdout_never_follows_json(
    tmp_path,
    monkeypatch,
    capsys,
    stderr_available,
):
    program = tmp_path / "cleanup_prints_late.py"
    program.write_text(
        "def main(ctx):\n"
        "    raise ValueError('student failure')\n",
        encoding="utf-8",
    )
    state = _install_real_controller_runtime(
        monkeypatch,
        tmp_path,
        block_cleanup=True,
        late_cleanup_stdout="late-backend-diagnostic",
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        if not stderr_available:
            sys.stderr = None
        exit_code = main(["student-run", "--program", str(program)])
    finally:
        try:
            state.cleanup_release.set()
            if state.controllers:
                state.controllers[0].wait_for_quiescence(2)
                _wait_until(lambda: sys.stdout is original_stdout)
        finally:
            sys.stderr = original_stderr

    controller = state.controllers[0]
    application = state.applications[0]
    session = state.sessions[0]
    assert controller.wait_for_quiescence(2) is True
    session.close_quarantined()

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert exit_code == 1
    assert payload["error"]["code"] == (
        "STUDENT_SESSION_CLOSE_DEFERRED"
    )
    if stderr_available:
        assert "late-backend-diagnostic" in captured.err
    else:
        assert "late-backend-diagnostic" not in captured.err
    assert application.socket.close_calls == [0]
    assert application.context.term_calls == 1


def test_student_run_normal_cancel_waits_for_cleanup_then_closes_once(
    tmp_path,
    monkeypatch,
    capsys,
):
    program = tmp_path / "cancel_normal.py"
    program.write_text(
        "def main(ctx):\n"
        "    while True:\n"
        "        pass\n",
        encoding="utf-8",
    )
    state = _install_real_controller_runtime(
        monkeypatch,
        tmp_path,
        force_wait_error=True,
    )

    exit_code = main(["student-run", "--program", str(program)])

    payload = _one_json_line(capsys)
    application = state.applications[0]
    controller = state.controllers[0]
    assert exit_code == 1
    assert payload["error"]["code"] == "STUDENT_RUN_FAILED"
    assert controller.wait_for_quiescence(0) is True
    assert controller.wait(timeout_s=0).status == "CANCELLED"
    assert application.close_calls == 0
    assert application.close_quarantined_calls == 1
    assert application.socket.close_calls == [0]
    assert application.context.term_calls == 1


def test_student_run_real_worker_keeps_stdout_as_single_json_line(
    tmp_path,
    monkeypatch,
    capfd,
):
    program = tmp_path / "student_prints.py"
    program.write_text(
        "def main(ctx):\n"
        "    print('student-diagnostic')\n",
        encoding="utf-8",
    )
    _install_real_controller_runtime(
        monkeypatch,
        tmp_path,
        runtime_stdout="[SIM] backend ready",
    )

    exit_code = main(["student-run", "--program", str(program)])

    captured = capfd.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert exit_code == 0, payload
    assert payload["status"] == "PASS"
    assert "[SIM] backend ready" in captured.err
    assert "student-diagnostic" in captured.err


def test_student_run_invalid_returns_two_and_closes_session_once(
    tmp_path,
    monkeypatch,
    capsys,
):
    program = tmp_path / "invalid.py"
    issue = ValidationIssue("MAIN_MISSING", "必须定义 main(ctx)")
    validation = ValidationResult(program.resolve(), False, (issue,))
    state, _ = _install_fake_runtime(
        monkeypatch,
        tmp_path,
        validation=validation,
    )

    exit_code = main(["student-run", "--program", str(program)])

    payload = _one_json_line(capsys)
    assert exit_code == 2
    assert payload["status"] == "FAIL"
    assert payload["issues"] == [
        {
            "code": "MAIN_MISSING",
            "message": "必须定义 main(ctx)",
            "line": None,
            "column": None,
        }
    ]
    assert payload["summary"] is None
    assert payload["evidence"] is None
    assert payload["error"] is None
    assert "controller.start" not in state.operations
    assert state.sessions[0].close_calls == 1
    assert state.sessions[0].close_quarantined_calls == 0
    assert state.applications[0].close_calls == 1
    _strict_json(payload)


def test_student_run_quarantined_failure_uses_safe_close_once(
    tmp_path,
    monkeypatch,
    capsys,
):
    program = tmp_path / "quarantined.py"
    validation = ValidationResult(program.resolve(), True, ())
    result = SimpleNamespace(
        status="FAILED",
        summary_path=None,
        evidence_dir=tmp_path / "evidence",
        error={
            "code": "STUDENT_BACKEND_COMMAND_STUCK",
            "message": "backend stuck",
            "details": {
                "quarantined": True,
                "connection_unusable": True,
            },
        },
    )
    state, _ = _install_fake_runtime(
        monkeypatch,
        tmp_path,
        validation=validation,
        result=result,
    )

    exit_code = main(["student-run", "--program", str(program)])

    payload = _one_json_line(capsys)
    assert exit_code == 1
    assert payload["status"] == "FAILED"
    assert payload["summary"] is None
    assert payload["evidence"] == str(tmp_path / "evidence")
    assert payload["error"] == result.error
    assert state.sessions[0].close_calls == 0
    assert state.sessions[0].close_quarantined_calls == 1
    assert state.applications[0].close_calls == 0
    assert state.applications[0].close_quarantined_calls == 1
    _strict_json(payload)


def test_student_run_exception_is_json_and_fails_closed_once(
    tmp_path,
    monkeypatch,
    capsys,
):
    program = tmp_path / "raises.py"
    validation = ValidationResult(program.resolve(), True, ())
    state, _ = _install_fake_runtime(
        monkeypatch,
        tmp_path,
        validation=validation,
        fail_stage="wait",
    )

    exit_code = main(["student-run", "--program", str(program)])

    payload = _one_json_line(capsys)
    assert exit_code == 1
    assert payload["status"] == "FAIL"
    assert payload["summary"] is None
    assert payload["evidence"] is None
    assert payload["error"]["code"] == "STUDENT_RUN_FAILED"
    assert payload["error"]["message"] == "wait exploded"
    assert payload["error"]["type"] == "builtins.RuntimeError"
    assert state.sessions[0].close_calls == 0
    assert state.sessions[0].close_quarantined_calls == 1
    assert state.applications[0].close_calls == 0
    assert state.applications[0].close_quarantined_calls == 1
    _strict_json(payload)


@pytest.mark.parametrize(
    "name",
    ["basic_motion.py", "pick_and_place.py"],
)
def test_student_template_is_valid_and_obeys_motion_guard(name):
    path = (
        cli_module.PROJECT_ROOT
        / "student_programs"
        / "templates"
        / name
    )
    validation = validate_program(path)
    assert validation.ok, validation.issues

    workspace = WorkspacePolicy(
        x_mm=(20, 140),
        y_mm=(-90, 90),
        z_mm=(10, 140),
        safe_z_mm=100,
    )
    policy = StudentExecutionPolicy(
        min_speed=1,
        max_speed=30,
        max_runtime_s=60,
        max_commands=200,
        command_timeout_s=10,
        max_sleep_s=5,
        tool_on_max_z_mm=35,
    )
    guard = StudentMotionGuard(workspace=workspace, policy=policy)

    class Robot:
        def __init__(self):
            self.pose_mm = (100.0, 60.0, 100.0)
            self.moves = []

        def home(self):
            self.pose_mm = (100.0, 60.0, 100.0)

        def move_world(self, x, y, z, *, speed):
            target = guard.validate_move(
                self.pose_mm,
                (x, y, z),
                speed=speed,
            )
            self.moves.append(target)
            self.pose_mm = target

        def pose(self):
            return self.pose_mm

    class Tool:
        def __init__(self, robot):
            self.robot = robot
            self.enabled = False

        def on(self):
            guard.validate_tool_on(self.robot.pose())
            self.enabled = True

        def off(self):
            self.enabled = False

    robot = Robot()
    tool = Tool(robot)
    context = SimpleNamespace(
        robot=robot,
        tool=tool,
        log=lambda _message: None,
        sleep=guard.validate_sleep,
        checkpoint=lambda _label: None,
    )

    runpy.run_path(str(path))["main"](context)

    assert robot.moves
    assert tool.enabled is False
