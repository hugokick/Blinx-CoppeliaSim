from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QThread, Qt, qInstallMessageHandler
from PyQt5.QtWidgets import QFileDialog

import vision_platform.ui.student_program_panel as panel_module
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.session import VisionLabSession
from vision_platform.student.protocol import RunState
from vision_platform.student.runner import StudentProgramController
from vision_platform.student.safety import StudentExecutionPolicy
from vision_platform.student.validator import (
    ValidationIssue,
    ValidationResult,
    validate_program,
)
from vision_platform.ui.student_program_panel import StudentProgramPanel


class ContractRobot:
    def __init__(self):
        self.pose = (100.0, 20.0, 120.0)

    def current_world_pose(self):
        return self.pose

    def move_world(self, x, y, z, *, speed):
        del speed
        self.pose = (float(x), float(y), float(z))

    def move_home(self):
        self.pose = (100.0, 20.0, 120.0)

    def close(self):
        pass


class ContractTool:
    def on(self):
        pass

    def off(self):
        pass


class ContractApplication:
    def __init__(self):
        self.config = SimpleNamespace(robot_backend="sim")
        self.workspace = WorkspacePolicy(
            x_mm=(20.0, 140.0),
            y_mm=(-90.0, 90.0),
            z_mm=(10.0, 140.0),
            safe_z_mm=100.0,
        )
        self.robot = ContractRobot()
        self.tool = ContractTool()

    def load_and_start_scene(self):
        pass

    def open(self):
        pass

    def close(self):
        pass


def make_real_terminal_controller(tmp_path):
    application = ContractApplication()
    session = VisionLabSession(
        application=application,
        factory=ContractApplication,
    )
    program = tmp_path / "original.py"
    program.write_text(
        "def main(ctx):\n    ctx.log('original')\n",
        encoding="utf-8",
    )
    controller = StudentProgramController(
        session=session,
        execution_policy=StudentExecutionPolicy(
            min_speed=1,
            max_speed=30,
            max_runtime_s=2,
            max_commands=20,
            command_timeout_s=1,
            max_sleep_s=0.2,
            tool_on_max_z_mm=35,
        ),
        output_root=tmp_path / "runs",
    )
    controller.load(program)
    assert controller.validate().ok is True
    controller.start()
    assert controller.wait(timeout_s=5).status == "PASS"
    return controller, session, program.resolve()


class FakeController:
    def __init__(
        self,
        *,
        cancel_entered=None,
        cancel_release=None,
        reset_entered=None,
        reset_release=None,
        cancel_failures=0,
    ):
        self.state = RunState.EMPTY
        self.handlers = []
        self.unsubscribe_calls = 0
        self.cancel_calls = 0
        self.resume_calls = 0
        self.reset_calls = 0
        self.load_calls = []
        self.validate_calls = []
        self.loaded_path = None
        self.validation_result = None
        self.cancel_entered = cancel_entered
        self.cancel_release = cancel_release
        self.reset_entered = reset_entered
        self.reset_release = reset_release
        self.cancel_failures = cancel_failures
        self.backend_active = False

    def subscribe(self, handler):
        self.handlers.append(handler)

        def unsubscribe():
            if handler in self.handlers:
                self.handlers.remove(handler)
                self.unsubscribe_calls += 1

        return unsubscribe

    def emit_state(self, state):
        self.state = state
        if state in {RunState.RUNNING, RunState.PAUSED}:
            self.backend_active = True
        elif state in {
            RunState.PASSED,
            RunState.FAILED,
            RunState.CANCELLED,
        }:
            self.backend_active = False
        self.emit_snapshot(
            SimpleNamespace(
                state=state,
                current_command=None,
                command_count=0,
                elapsed_seconds=0.0,
                error=None,
                evidence_dir=None,
                tcp_mm=None,
            )
        )

    def emit_snapshot(self, snapshot):
        self.state = snapshot.state
        for handler in tuple(self.handlers):
            handler(snapshot)

    def snapshot(
        self,
        state,
        *,
        current_command=None,
        command_count=0,
        elapsed_seconds=0.0,
        error=None,
        evidence_dir=None,
        tcp_mm=None,
    ):
        return SimpleNamespace(
            state=state,
            current_command=current_command,
            command_count=command_count,
            elapsed_seconds=elapsed_seconds,
            error=error,
            evidence_dir=evidence_dir,
            tcp_mm=tcp_mm,
        )

    def load(self, path):
        self.load_calls.append(Path(path))
        if self.state in {
            RunState.RUNNING,
            RunState.PAUSED,
            RunState.PASSED,
            RunState.FAILED,
            RunState.CANCELLED,
            RunState.RESETTING,
        }:
            raise RuntimeError(
                "load requires an idle controller; reset terminal runs"
            )
        self.loaded_path = Path(path).resolve()
        self.emit_state(RunState.LOADED)
        return self.loaded_path

    def validate(self, program_path=None):
        self.validate_calls.append(
            None if program_path is None else Path(program_path)
        )
        if self.validation_result is None:
            selected = Path(program_path or self.loaded_path).resolve()
            result = validate_program(selected)
        else:
            result = self.validation_result
        if result.ok:
            self.emit_state(RunState.VALIDATED)
        else:
            self.emit_state(RunState.LOADED)
        return result

    def start(self, *, paused=False):
        self.backend_active = True
        self.emit_state(
            RunState.PAUSED if paused else RunState.RUNNING
        )

    def pause(self):
        self.emit_state(RunState.PAUSED)

    def resume(self):
        self.resume_calls += 1
        self.emit_state(RunState.RUNNING)

    def step(self):
        self.emit_state(RunState.PAUSED)

    def cancel(self):
        self.cancel_calls += 1
        if self.cancel_entered is not None:
            self.cancel_entered.set()
        if self.cancel_release is not None:
            assert self.cancel_release.wait(timeout=2)
        if self.cancel_failures:
            self.cancel_failures -= 1
            raise RuntimeError("cancel failed")
        self.backend_active = False
        self.emit_state(RunState.CANCELLED)

    def reset(self):
        self.reset_calls += 1
        if self.reset_entered is not None:
            self.reset_entered.set()
        if self.reset_release is not None:
            assert self.reset_release.wait(timeout=2)
        if self.state not in {
            RunState.PASSED,
            RunState.FAILED,
            RunState.CANCELLED,
        }:
            raise RuntimeError("reset requires a terminal run state")
        self.emit_state(RunState.RESETTING)
        self.emit_state(RunState.VALIDATED)
        return object()

    def wait_for_quiescence(self, timeout_s):
        assert timeout_s == 0
        return not self.backend_active


def assert_click_returns_before_release(
    qtbot,
    button,
    entered,
    release,
):
    click_returned = Event()
    observations = []

    def release_after_click() -> None:
        assert entered.wait(timeout=2)
        observations.append(click_returned.wait(timeout=0.25))
        release.set()

    gate = Thread(target=release_after_click, daemon=True)
    gate.start()
    qtbot.mouseClick(button, Qt.LeftButton)
    click_returned.set()
    gate.join(timeout=2)

    assert gate.is_alive() is False
    assert observations == [True]


def test_panel_exposes_file_and_run_controls(qtbot):
    panel = StudentProgramPanel(controller=FakeController())
    qtbot.addWidget(panel)

    assert panel.open_button.text() == "打开程序"
    assert panel.validate_button.text() == "检查代码"
    assert panel.run_button.text() == "运行"
    assert panel.pause_button.text() == "暂停"
    assert panel.resume_button.text() == "继续"
    assert panel.step_button.text() == "下一步"
    assert panel.stop_button.text() == "停止"
    assert panel.reset_button.text() == "复位场景"


def test_panel_button_states_follow_runner_state(qtbot):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)

    controller.emit_state(RunState.PAUSED)

    assert panel.step_button.isEnabled() is True
    assert panel.resume_button.isEnabled() is True
    assert panel.run_button.isEnabled() is False
    assert panel.stop_button.isEnabled() is True

    qtbot.mouseClick(panel.resume_button, Qt.LeftButton)

    assert controller.resume_calls == 1
    assert controller.state is RunState.RUNNING


def test_open_file_loads_utf8_source(qtbot, tmp_path, monkeypatch):
    program = tmp_path / "student.py"
    program.write_text(
        "def main(ctx):\n    ctx.robot.home()\n",
        encoding="utf-8",
    )
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(program), "Python (*.py)"),
    )

    qtbot.mouseClick(panel.open_button, Qt.LeftButton)

    assert "def main(ctx):" in panel.editor.toPlainText()
    assert controller.loaded_path == program


def test_validation_issues_render_line_and_code(qtbot, tmp_path):
    program = tmp_path / "student.py"
    program.write_text(
        "def main(ctx)\n    pass\n",
        encoding="utf-8",
    )
    controller = FakeController()
    controller.loaded_path = program
    controller.validation_result = ValidationResult(
        path=program,
        ok=False,
        issues=(
            ValidationIssue(
                code="SYNTAX_ERROR",
                message="expected ':'",
                line=3,
                column=10,
            ),
        ),
    )
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    panel.editor.setPlainText(
        program.read_text(encoding="utf-8")
    )
    panel.program_path = program
    controller.emit_state(RunState.LOADED)

    qtbot.mouseClick(panel.validate_button, Qt.LeftButton)

    assert "SYNTAX_ERROR" in panel.console.toPlainText()
    assert "第 3 行" in panel.console.toPlainText()


def test_close_cancels_running_controller_without_blocking(qtbot):
    entered = Event()
    release = Event()
    controller = FakeController()
    controller.cancel_entered = entered
    controller.cancel_release = release
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    controller.emit_state(RunState.RUNNING)
    panel.show()
    close_returned = Event()
    observations = []

    def release_after_close() -> None:
        assert entered.wait(timeout=2)
        observations.append(close_returned.wait(timeout=0.25))
        release.set()

    gate = Thread(target=release_after_close, daemon=True)
    gate.start()

    closed = panel.close()
    close_returned.set()
    gate.join(timeout=2)

    assert observations == [True]
    assert closed is False
    assert panel.isVisible() is True
    assert controller.cancel_calls == 1
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )

    assert panel.close() is True

    assert controller.handlers == []


@pytest.mark.parametrize(
    ("state", "enabled"),
    [
        (
            RunState.EMPTY,
            (True, False, False, False, False, False, False, False),
        ),
        (
            RunState.LOADED,
            (True, True, False, False, False, False, False, False),
        ),
        (
            RunState.VALIDATED,
            (True, True, True, False, False, False, False, False),
        ),
        (
            RunState.RUNNING,
            (False, False, False, True, False, False, True, False),
        ),
        (
            RunState.PAUSED,
            (False, False, False, False, True, True, True, False),
        ),
        (
            RunState.PASSED,
            (True, True, False, False, False, False, False, True),
        ),
        (
            RunState.FAILED,
            (True, True, False, False, False, False, False, True),
        ),
        (
            RunState.CANCELLED,
            (True, True, False, False, False, False, False, True),
        ),
        (
            RunState.RESETTING,
            (False, False, False, False, False, False, False, False),
        ),
    ],
)
def test_panel_applies_complete_button_state_matrix(
    qtbot,
    state,
    enabled,
):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)

    controller.emit_state(state)

    (
        file_enabled,
        validate,
        run,
        pause,
        resume,
        step,
        stop,
        reset,
    ) = enabled
    assert panel.open_button.isEnabled() is file_enabled
    assert panel.save_button.isEnabled() is file_enabled
    assert panel.save_as_button.isEnabled() is file_enabled
    assert panel.editor.isReadOnly() is (not file_enabled)
    assert panel.validate_button.isEnabled() is validate
    assert panel.run_button.isEnabled() is run
    assert panel.pause_button.isEnabled() is pause
    assert panel.resume_button.isEnabled() is resume
    assert panel.step_button.isEnabled() is step
    assert panel.stop_button.isEnabled() is stop
    assert panel.reset_button.isEnabled() is reset


def test_save_and_save_as_write_utf8_with_py_suffix(
    qtbot,
    tmp_path,
    monkeypatch,
):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    source = "def main(ctx):\n    ctx.log('中文步骤')\n"
    panel.editor.setPlainText(source)
    selected = tmp_path / "学生程序"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(selected), "Python (*.py)"),
    )

    qtbot.mouseClick(panel.save_as_button, Qt.LeftButton)

    expected = selected.with_suffix(".py").resolve()
    assert panel.program_path == expected
    assert expected.read_text(encoding="utf-8") == source
    assert controller.loaded_path == expected

    revised = source.replace("中文步骤", "保存成功")
    panel.editor.setPlainText(revised)
    qtbot.mouseClick(panel.save_button, Qt.LeftButton)
    assert expected.read_text(encoding="utf-8") == revised


def test_failed_save_as_preserves_existing_identity_and_retries(
    qtbot,
    tmp_path,
    monkeypatch,
):
    old_path = (tmp_path / "old.py").resolve()
    old_source = "def main(ctx):\n    ctx.log('old')\n"
    old_path.write_text(old_source, encoding="utf-8")
    failing_candidate = tmp_path / "blocked.py"
    failing_candidate.mkdir()
    retry_candidate = (tmp_path / "retry.py").resolve()
    responses = iter(
        (
            (str(failing_candidate), "Python (*.py)"),
            (str(retry_candidate), "Python (*.py)"),
        )
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: next(responses),
    )
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    panel.program_path = old_path
    panel.path_label.setText(str(old_path))
    panel._pending_program_path = old_path
    edited_source = "def main(ctx):\n    ctx.log('edited')\n"
    panel.editor.setPlainText(edited_source)
    controller.emit_state(RunState.PASSED)

    first_result = panel._save_as()
    after_failure = (
        panel.program_path,
        panel.path_label.text(),
        panel.pending_program_path,
        panel.editor.toPlainText(),
        old_path.read_text(encoding="utf-8"),
    )
    second_result = panel._save_as()

    assert first_result is False
    assert after_failure == (
        old_path,
        str(old_path),
        old_path,
        edited_source,
        old_source,
    )
    assert "STUDENT_FILE_SAVE_FAILED" in panel.console.toPlainText()
    assert second_result is True
    assert panel.program_path == retry_candidate
    assert panel.path_label.text() == str(retry_candidate)
    assert panel.pending_program_path == retry_candidate
    assert panel.editor.toPlainText() == edited_source
    assert retry_candidate.read_text(encoding="utf-8") == edited_source
    assert old_path.read_text(encoding="utf-8") == old_source


def test_failed_first_save_as_preserves_empty_identity_and_retries(
    qtbot,
    tmp_path,
    monkeypatch,
):
    failing_candidate = tmp_path / "blocked.py"
    failing_candidate.mkdir()
    retry_without_suffix = tmp_path / "first_success"
    retry_candidate = retry_without_suffix.with_suffix(".py").resolve()
    responses = iter(
        (
            (str(failing_candidate), "Python (*.py)"),
            (str(retry_without_suffix), "Python (*.py)"),
        )
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: next(responses),
    )
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    initial_label = panel.path_label.text()
    source = "def main(ctx):\n    ctx.log('first save')\n"
    panel.editor.setPlainText(source)

    first_result = panel._save_as()
    after_failure = (
        panel.program_path,
        panel.path_label.text(),
        panel.pending_program_path,
        panel.editor.toPlainText(),
    )
    second_result = panel._save_as()

    assert first_result is False
    assert after_failure == (None, initial_label, None, source)
    assert "STUDENT_FILE_SAVE_FAILED" in panel.console.toPlainText()
    assert second_result is True
    assert panel.program_path == retry_candidate
    assert panel.path_label.text() == str(retry_candidate)
    assert panel.pending_program_path is None
    assert panel.editor.toPlainText() == source
    assert retry_candidate.read_text(encoding="utf-8") == source
    assert controller.loaded_path == retry_candidate


def test_terminal_save_and_static_validation_do_not_call_controller_load(
    qtbot,
    tmp_path,
):
    program = tmp_path / "terminal_edit.py"
    program.write_text(
        "def main(ctx):\n    ctx.log('old')\n",
        encoding="utf-8",
    )
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    panel.program_path = program.resolve()
    panel.editor.setPlainText(
        "def main(ctx):\n    ctx.log('终态新内容')\n"
    )
    controller.emit_state(RunState.PASSED)

    qtbot.mouseClick(panel.save_button, Qt.LeftButton)
    qtbot.mouseClick(panel.validate_button, Qt.LeftButton)

    assert "终态新内容" in program.read_text(encoding="utf-8")
    assert controller.load_calls == []
    assert controller.validate_calls == []
    assert panel.pending_program_path == program.resolve()
    assert "已保存" in panel.console.toPlainText()
    assert "代码检查：PASS" in panel.console.toPlainText()


def test_terminal_open_stages_program_without_controller_load(
    qtbot,
    tmp_path,
    monkeypatch,
):
    program = tmp_path / "staged.py"
    program.write_text(
        "def main(ctx):\n    ctx.robot.home()\n",
        encoding="utf-8",
    )
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    controller.emit_state(RunState.FAILED)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(program), "Python (*.py)"),
    )

    qtbot.mouseClick(panel.open_button, Qt.LeftButton)

    assert panel.program_path == program.resolve()
    assert panel.pending_program_path == program.resolve()
    assert "ctx.robot.home()" in panel.editor.toPlainText()
    assert controller.load_calls == []


@pytest.mark.parametrize(
    ("source", "expected_state", "run_enabled", "expected_status"),
    [
        (
            "def main(ctx):\n    ctx.log('new')\n",
            RunState.VALIDATED,
            True,
            "PASS",
        ),
        (
            "def main(ctx)\n    pass\n",
            RunState.LOADED,
            False,
            "FAIL",
        ),
    ],
)
def test_terminal_reset_synchronizes_and_revalidates_staged_source(
    qtbot,
    tmp_path,
    source,
    expected_state,
    run_enabled,
    expected_status,
):
    program = tmp_path / "staged_after_terminal.py"
    program.write_text(source, encoding="utf-8")
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    panel.program_path = program.resolve()
    panel.editor.setPlainText(source)
    controller.emit_state(RunState.CANCELLED)

    qtbot.mouseClick(panel.save_button, Qt.LeftButton)
    qtbot.mouseClick(panel.validate_button, Qt.LeftButton)
    qtbot.mouseClick(panel.reset_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )

    assert controller.reset_calls == 1
    assert controller.load_calls == [program.resolve()]
    assert controller.validate_calls == [program.resolve()]
    assert controller.state is expected_state
    assert panel.run_button.isEnabled() is run_enabled
    assert panel.pending_program_path is None
    assert f"代码检查：{expected_status}" in panel.console.toPlainText()


@pytest.mark.parametrize(
    ("source", "expected_state", "run_enabled"),
    [
        (
            "def main(ctx):\n    ctx.log('真实契约')\n",
            RunState.VALIDATED,
            True,
        ),
        (
            "def main(ctx)\n    pass\n",
            RunState.LOADED,
            False,
        ),
    ],
)
def test_real_controller_terminal_stage_reset_never_reuses_stale_validation(
    qtbot,
    tmp_path,
    monkeypatch,
    source,
    expected_state,
    run_enabled,
):
    controller, session, original = make_real_terminal_controller(
        tmp_path
    )
    staged = tmp_path / "real_staged.py"
    staged.write_text(source, encoding="utf-8")
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(staged), "Python (*.py)"),
    )

    qtbot.mouseClick(panel.open_button, Qt.LeftButton)
    panel.editor.setPlainText(source)
    assert panel._save() is True
    qtbot.mouseClick(panel.validate_button, Qt.LeftButton)

    assert staged.read_text(encoding="utf-8") == source
    assert controller.state is RunState.PASSED
    assert controller._program_path == original
    assert panel.pending_program_path == staged.resolve()

    qtbot.mouseClick(panel.reset_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=5000,
    )

    assert controller._program_path == staged.resolve()
    assert controller.state is expected_state
    assert panel.run_button.isEnabled() is run_enabled
    assert panel.pending_program_path is None
    assert controller.process_is_alive is False

    assert panel.close() is True
    session.close()


def test_stop_click_returns_while_cancel_is_blocked(qtbot):
    entered = Event()
    release = Event()
    controller = FakeController(
        cancel_entered=entered,
        cancel_release=release,
    )
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    controller.emit_state(RunState.RUNNING)

    assert_click_returns_before_release(
        qtbot,
        panel.stop_button,
        entered,
        release,
    )
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )

    assert controller.cancel_calls == 1
    assert controller.state is RunState.CANCELLED


def test_reset_click_returns_while_scene_reset_is_blocked(qtbot):
    entered = Event()
    release = Event()
    controller = FakeController(
        reset_entered=entered,
        reset_release=release,
    )
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    controller.emit_state(RunState.PASSED)

    assert_click_returns_before_release(
        qtbot,
        panel.reset_button,
        entered,
        release,
    )
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )

    assert controller.reset_calls == 1
    assert controller.state is RunState.VALIDATED


def test_cancel_request_is_idempotent_and_failure_allows_retry(qtbot):
    first_entered = Event()
    first_release = Event()
    controller = FakeController(
        cancel_entered=first_entered,
        cancel_release=first_release,
        cancel_failures=1,
    )
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    controller.emit_state(RunState.RUNNING)

    assert panel.request_stop() is True
    assert panel.request_stop() is False
    assert first_entered.wait(timeout=2)
    first_release.set()
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )
    assert "STUDENT_CANCEL_FAILED" in panel.console.toPlainText()

    controller.cancel_entered = None
    controller.cancel_release = None
    assert panel.request_stop() is True
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )

    assert controller.cancel_calls == 2
    assert controller.state is RunState.CANCELLED


@pytest.mark.parametrize("_retry_iteration", range(12))
def test_stop_thread_start_failure_restores_state_and_allows_retry(
    qtbot,
    monkeypatch,
    _retry_iteration,
):
    del _retry_iteration
    created_threads = []
    qt_messages = []

    class FailFirstStartThread(QThread):
        start_calls = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_threads.append(self)

        def start(self, *args, **kwargs):
            type(self).start_calls += 1
            if type(self).start_calls == 1:
                raise RuntimeError("deterministic thread start failure")
            return super().start(*args, **kwargs)

    monkeypatch.setattr(panel_module, "QThread", FailFirstStartThread)
    previous_handler = qInstallMessageHandler(
        lambda _kind, _context, message: qt_messages.append(message)
    )
    try:
        controller = FakeController()
        panel = StudentProgramPanel(controller=controller)
        qtbot.addWidget(panel)
        controller.emit_state(RunState.RUNNING)

        first_result = panel.request_stop()
        after_failure = (
            panel.editor.isReadOnly(),
            panel.open_button.isEnabled(),
            panel.save_button.isEnabled(),
            panel.save_as_button.isEnabled(),
            panel.validate_button.isEnabled(),
            panel.run_button.isEnabled(),
            panel.pause_button.isEnabled(),
            panel.resume_button.isEnabled(),
            panel.step_button.isEnabled(),
            panel.stop_button.isEnabled(),
            panel.reset_button.isEnabled(),
        )
        error_visible = (
            "[STUDENT_CANCEL_FAILED] "
            "deterministic thread start failure"
            in panel.console.toPlainText()
        )
        cancel_flag_reset = not panel._cancel_requested_for_run
        operation_cleared = not panel.operation_in_progress

        second_result = panel.request_stop()
        qtbot.waitUntil(
            lambda: panel.operation_in_progress is False,
            timeout=2000,
        )
        qtbot.waitUntil(
            lambda: all(
                sip.isdeleted(thread) for thread in created_threads
            ),
            timeout=2000,
        )
    finally:
        qInstallMessageHandler(previous_handler)

    assert first_result is False
    assert operation_cleared is True
    assert cancel_flag_reset is True
    assert error_visible is True
    assert after_failure == (
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
    )
    assert second_result is True
    assert controller.cancel_calls == 1
    assert controller.state is RunState.CANCELLED
    assert not any(
        "QThread" in message or "QObject" in message
        for message in qt_messages
    )


def test_cancel_coordination_resets_for_new_run_after_scene_reset(qtbot):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    controller.emit_state(RunState.RUNNING)

    assert panel.request_stop() is True
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )
    assert controller.cancel_calls == 1

    assert panel.request_reset() is True
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )
    controller.start()

    assert panel.request_stop() is True
    qtbot.waitUntil(
        lambda: panel.operation_in_progress is False,
        timeout=2000,
    )
    assert controller.cancel_calls == 2


def test_background_snapshot_uses_signal_bridge_and_renders_all_fields(
    qtbot,
    tmp_path,
):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    evidence = tmp_path / "evidence"
    snapshot = controller.snapshot(
        RunState.RUNNING,
        current_command="robot.move_world",
        command_count=7,
        elapsed_seconds=1.26,
        error={
            "code": "STUDENT_TEST_ERROR",
            "message": "可读错误",
        },
        evidence_dir=evidence,
        tcp_mm=(100.0, -20.5, 35.25),
    )

    thread = Thread(
        target=lambda: controller.emit_snapshot(snapshot),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=2)
    assert thread.is_alive() is False
    qtbot.waitUntil(
        lambda: panel.state_label.text() == "RUNNING",
        timeout=2000,
    )

    assert "robot.move_world" in panel.command_label.text()
    assert "命令数：7" in panel.metrics_label.text()
    assert "运行时间：1.3 s" in panel.metrics_label.text()
    assert str(evidence) in panel.evidence_label.text()
    assert (
        panel.tcp_label.text()
        == "TCP：X=100.0 mm　Y=-20.5 mm　Z=35.2 mm"
    )
    assert "STUDENT_TEST_ERROR" in panel.console.toPlainText()
    assert "可读错误" in panel.console.toPlainText()


def test_state_label_is_a_distinguishable_dynamic_badge(qtbot):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)

    controller.emit_state(RunState.RUNNING)
    running_style = panel.state_label.styleSheet()

    assert panel.state_label.objectName() == "studentStateBadge"
    assert panel.state_label.property("runState") == "RUNNING"
    assert running_style

    controller.emit_state(RunState.FAILED)

    assert panel.state_label.property("runState") == "FAILED"
    assert panel.state_label.styleSheet() != running_style


def test_close_releases_subscription_idempotently(qtbot):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    assert len(controller.handlers) == 1

    panel.close()
    panel.close()
    panel.release_subscription()

    assert controller.handlers == []
    assert controller.unsubscribe_calls == 1


def test_async_stop_close_cleanup_is_race_stable(qtbot):
    for _ in range(12):
        entered = Event()
        release = Event()
        controller = FakeController(
            cancel_entered=entered,
            cancel_release=release,
        )
        panel = StudentProgramPanel(controller=controller)
        qtbot.addWidget(panel)
        controller.emit_state(RunState.RUNNING)
        panel.show()
        assert hasattr(panel, "operation_in_progress")

        qtbot.mouseClick(panel.stop_button, Qt.LeftButton)
        assert entered.wait(timeout=2)
        assert panel.close() is False
        assert controller.cancel_calls == 1
        assert controller.handlers

        release.set()
        qtbot.waitUntil(
            lambda: panel.operation_in_progress is False,
            timeout=2000,
        )
        assert panel.close() is True
        assert controller.handlers == []


def test_ui_file_error_is_rendered_without_escaping_qt_slot(
    qtbot,
    tmp_path,
    monkeypatch,
):
    invalid = tmp_path / "invalid.py"
    invalid.write_bytes(b"\xff\xfe")
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(invalid), "Python (*.py)"),
    )

    qtbot.mouseClick(panel.open_button, Qt.LeftButton)

    assert "STUDENT_FILE_OPEN_FAILED" in panel.console.toPlainText()
    assert controller.loaded_path is None
