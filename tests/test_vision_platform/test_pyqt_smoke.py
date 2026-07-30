from __future__ import annotations

import json
from threading import Event, Thread
from types import SimpleNamespace

import vision_platform.ui.pyqt_app as pyqt_app
from vision_platform.ui.pyqt_app import VisionLabWindow


class FakeEventBus:
    def __init__(
        self,
        *,
        unsubscribe_failures=0,
        subscribe_entered=None,
        allow_subscribe=None,
    ):
        self.handlers = []
        self.unsubscribe_calls = 0
        self.unsubscribe_attempts = 0
        self.unsubscribe_failures = unsubscribe_failures
        self.subscribe_entered = subscribe_entered
        self.allow_subscribe = allow_subscribe

    def subscribe(self, handler):
        if self.subscribe_entered is not None:
            self.subscribe_entered.set()
        if self.allow_subscribe is not None:
            assert self.allow_subscribe.wait(timeout=2.0)
        self.handlers.append(handler)
        unsubscribed = False

        def unsubscribe():
            nonlocal unsubscribed
            if unsubscribed:
                return
            self.unsubscribe_attempts += 1
            if self.unsubscribe_failures:
                self.unsubscribe_failures -= 1
                raise RuntimeError("event unsubscribe failed")
            unsubscribed = True
            self.unsubscribe_calls += 1
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe


class FakeTool:
    def __init__(self):
        self.off_calls = 0

    def off(self):
        self.off_calls += 1


class DisconnectedApplication:
    def __init__(self, *, with_event_bus=False, event_bus=None):
        self.close_calls = 0
        self.event_bus = (
            event_bus
            if event_bus is not None
            else FakeEventBus() if with_event_bus else None
        )
        self.tool = FakeTool()

    def close(self):
        self.close_calls += 1


class FakeSession:
    def __init__(
        self,
        application,
        *,
        unsubscribe_failures=0,
        close_failures=0,
    ):
        self.application = application
        self.handlers = []
        self.close_calls = 0
        self.unsubscribe_attempts = 0
        self.unsubscribe_failures = unsubscribe_failures
        self.close_failures = close_failures

    def subscribe(self, handler):
        self.handlers.append(handler)
        unsubscribed = False

        def unsubscribe():
            nonlocal unsubscribed
            if unsubscribed:
                return
            self.unsubscribe_attempts += 1
            if self.unsubscribe_failures:
                self.unsubscribe_failures -= 1
                raise RuntimeError("session unsubscribe failed")
            unsubscribed = True
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe

    def replace(self, application):
        self.application = application
        for handler in tuple(self.handlers):
            handler(application)

    def close(self):
        self.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("session close failed")
        self.application.close()


class FakeCloseEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def operation_controls(window):
    return (
        window.camera_backend_combo,
        window.robot_backend_combo,
        window.mode_combo,
        window.connect_button,
        window.calibrate_button,
        window.start_button,
        window.pause_button,
        window.resume_button,
        window.reset_button,
        window.emergency_button,
    )


def test_pyqt_window_constructs_without_camera_or_robot_connection(qtbot):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)

    assert window.camera_backend_combo.count() == 3
    assert window.robot_backend_combo.count() == 2
    assert "未连接" in window.status_label.text()
    assert window.preview_label.minimumWidth() >= 640
    assert window.start_button.minimumHeight() >= 44
    assert window.emergency_button.minimumHeight() >= 44


def test_pyqt_window_exposes_teaching_workflow_controls(qtbot):
    window = VisionLabWindow(application=DisconnectedApplication())
    qtbot.addWidget(window)

    assert window.mode_combo.currentText() == "全自动"
    assert window.calibration_table.columnCount() == 4
    assert window.detection_table.columnCount() == 6
    assert window.pause_button.isEnabled() is False
    assert window.resume_button.isEnabled() is False
    assert "像素坐标" in window.pixel_coordinate_label.text()
    assert "世界坐标" in window.world_coordinate_label.text()


def test_pyqt_window_close_calls_application_close(qtbot):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)

    window.close()
    window.close()

    assert application.close_calls == 1


def test_pyqt_window_populates_calibration_table_from_report(qtbot, tmp_path):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "pixel_points": [[100, 120], [200, 120], [100, 220]],
                "world_points_mm": [[35, -70], [100, -70], [35, 70]],
            }
        ),
        encoding="utf-8",
    )
    report = SimpleNamespace(
        status="PASS",
        rms_error_mm=0.2,
        max_error_mm=0.4,
        calibration_path=calibration_path,
        report_path=tmp_path / "report.json",
    )

    window._action_finished(
        SimpleNamespace(action="calibrate", payload=report, frame=None)
    )

    assert window.calibration_table.item(0, 1).text() == "100.0"
    assert window.calibration_table.item(2, 3).text() == "(35.0, 70.0)"


def test_pyqt_window_rebinds_replaced_application_event_bus(qtbot):
    first = DisconnectedApplication(with_event_bus=True)
    second = DisconnectedApplication(with_event_bus=True)
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)

    assert len(first.event_bus.handlers) == 1

    session.replace(second)

    assert window.application is second
    assert first.event_bus.handlers == []
    assert first.event_bus.unsubscribe_calls == 1
    assert len(second.event_bus.handlers) == 1


def test_pyqt_window_session_close_is_used_exactly_once(qtbot):
    first = DisconnectedApplication()
    second = DisconnectedApplication()
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)
    session.replace(second)

    window.close()
    window.close()

    assert session.handlers == []
    assert session.close_calls == 1
    assert first.close_calls == 0
    assert second.close_calls == 1


def test_pyqt_new_worker_reads_current_session_application(
    qtbot,
    monkeypatch,
):
    first = DisconnectedApplication()
    second = DisconnectedApplication()
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)
    session.application = second
    created = []

    class FakeSignal:
        def connect(self, _handler):
            pass

    class FakeThread:
        def __init__(self, _parent):
            self.started = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            pass

        def quit(self):
            pass

        def wait(self, _timeout):
            return True

    class FakeWorker:
        def __init__(self, **kwargs):
            self.application = kwargs["application"]
            self.finished = FakeSignal()
            self.failed = FakeSignal()
            created.append(self)

        def moveToThread(self, _thread):
            pass

        def run(self):
            pass

        def deleteLater(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(pyqt_app, "QThread", FakeThread)
    monkeypatch.setattr(pyqt_app, "_ActionWorker", FakeWorker)

    window._start_action("reset")

    assert created[0].application is second
    window.close()


def test_pyqt_emergency_stop_reads_current_session_application(qtbot):
    first = DisconnectedApplication()
    second = DisconnectedApplication()
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)
    session.application = second

    window._emergency_stop()

    assert first.tool.off_calls == 0
    assert second.tool.off_calls == 1


def test_pyqt_close_serializes_with_in_flight_application_replacement(qtbot):
    subscribe_entered = Event()
    allow_subscribe = Event()
    first = DisconnectedApplication(with_event_bus=True)
    second_bus = FakeEventBus(
        subscribe_entered=subscribe_entered,
        allow_subscribe=allow_subscribe,
    )
    second = DisconnectedApplication(event_bus=second_bus)
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)
    errors = []

    def replace() -> None:
        try:
            session.replace(second)
        except BaseException as error:
            errors.append(error)

    close_event = FakeCloseEvent()

    def close() -> None:
        try:
            window.closeEvent(close_event)
        except BaseException as error:
            errors.append(error)

    replace_thread = Thread(target=replace)
    replace_thread.start()
    assert subscribe_entered.wait(timeout=2.0)
    close_thread = Thread(target=close)
    close_thread.start()
    allow_subscribe.set()
    replace_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)

    assert not replace_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert close_event.accepted is True
    assert close_event.ignored is False
    assert second_bus.handlers == []
    assert window._event_unsubscribe is None
    assert session.close_calls == 1
    assert second.close_calls == 1


def test_pyqt_close_retries_failed_unsubscribes_without_reclosing_owner(
    qtbot,
    monkeypatch,
):
    event_bus = FakeEventBus(unsubscribe_failures=1)
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application, unsubscribe_failures=1)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    original_view_unsubscribe = window._view_unsubscribe
    view_unsubscribe_attempts = 0

    def flaky_view_unsubscribe() -> None:
        nonlocal view_unsubscribe_attempts
        view_unsubscribe_attempts += 1
        if view_unsubscribe_attempts == 1:
            raise RuntimeError("view unsubscribe failed")
        original_view_unsubscribe()

    window._view_unsubscribe = flaky_view_unsubscribe
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert session.close_calls == 1
    assert application.close_calls == 1
    assert session.unsubscribe_attempts == 1
    assert event_bus.unsubscribe_attempts == 1
    assert view_unsubscribe_attempts == 1
    assert window.view_model.snapshot().error_code == "UI_CLOSE_FAILED"
    assert "UI_CLOSE_FAILED" in window.error_label.text()
    assert all(
        control.isEnabled() is False
        for control in operation_controls(window)
    )

    def reject_worker_creation(**_kwargs):
        raise AssertionError("closing window created a worker")

    monkeypatch.setattr(pyqt_app, "_ActionWorker", reject_worker_creation)
    window._start_action("reset")
    assert window._worker is None
    assert window._thread is None
    window._thread_finished()
    assert all(
        control.isEnabled() is False
        for control in operation_controls(window)
    )

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert application.close_calls == 1
    assert session.unsubscribe_attempts == 2
    assert event_bus.unsubscribe_attempts == 2
    assert view_unsubscribe_attempts == 2
    assert session.handlers == []
    assert event_bus.handlers == []
    assert window._session_unsubscribe is None
    assert window._event_unsubscribe is None
    assert window._view_unsubscribe is None
    assert window._close_complete is True


def test_pyqt_owner_close_error_is_visible_and_does_not_block_process_exit(
    qtbot,
):
    event_bus = FakeEventBus()
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application, close_failures=1)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.accepted is True
    assert first_event.ignored is False
    assert session.close_calls == 1
    assert application.close_calls == 0
    assert session.handlers == []
    assert event_bus.handlers == []
    assert window._session_unsubscribe is None
    assert window._event_unsubscribe is None
    assert window._view_unsubscribe is None
    assert window.view_model.snapshot().error_code == "UI_CLOSE_FAILED"
    assert "UI_CLOSE_FAILED" in window.error_label.text()
    assert window._close_complete is True
    for control in operation_controls(window):
        assert control.isEnabled() is False

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert application.close_calls == 0
    assert window._close_complete is True


def test_pyqt_owner_close_error_has_priority_over_unsubscribe_error(qtbot):
    application = DisconnectedApplication()
    session = FakeSession(
        application,
        unsubscribe_failures=1,
        close_failures=1,
    )
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    snapshot = window.view_model.snapshot()
    assert snapshot.error_code == "UI_CLOSE_FAILED"
    assert "session close failed" in snapshot.status_text
    assert "session unsubscribe failed" not in snapshot.status_text
    assert session.close_calls == 1

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert window._close_complete is True


def test_pyqt_event_unsubscribe_error_is_visible_before_view_unsubscribe(
    qtbot,
):
    event_bus = FakeEventBus(unsubscribe_failures=1)
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert "UI_CLOSE_FAILED" in window.error_label.text()
    assert window.view_model.snapshot().error_code == "UI_CLOSE_FAILED"
    assert window._view_unsubscribe is None
    assert session.close_calls == 1

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert event_bus.handlers == []


def test_pyqt_view_unsubscribe_error_remains_visible_and_retryable(qtbot):
    application = DisconnectedApplication()
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    original_view_unsubscribe = window._view_unsubscribe
    attempts = 0

    def unsubscribe_then_fail_once() -> None:
        nonlocal attempts
        attempts += 1
        original_view_unsubscribe()
        if attempts == 1:
            raise RuntimeError("view unsubscribe failed after removal")

    window._view_unsubscribe = unsubscribe_then_fail_once
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert "UI_CLOSE_FAILED" in window.error_label.text()
    assert window.view_model.snapshot().error_code == "UI_CLOSE_FAILED"
    assert window._view_unsubscribe is unsubscribe_then_fail_once
    assert session.close_calls == 1

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert attempts == 2
    assert session.close_calls == 1
    assert window._view_unsubscribe is None


def test_pyqt_close_wait_timeout_preserves_owner_and_subscriptions_for_retry(
    qtbot,
):
    event_bus = FakeEventBus()
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)

    class FakeWorker:
        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    class FakeThread:
        def __init__(self):
            self.quit_calls = 0
            self.wait_results = [False, True]

        def quit(self):
            self.quit_calls += 1

        def wait(self, _timeout):
            return self.wait_results.pop(0)

    worker = FakeWorker()
    thread = FakeThread()
    window._worker = worker
    window._thread = thread
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()
    enabled_before = [
        control.isEnabled()
        for control in operation_controls(window)
    ]

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert window.view_model.snapshot().error_code == "UI_THREAD_STOP_TIMEOUT"
    assert session.close_calls == 0
    assert application.close_calls == 0
    assert len(session.handlers) == 1
    assert len(event_bus.handlers) == 1
    assert window._session_unsubscribe is not None
    assert window._event_unsubscribe is not None
    assert window._view_unsubscribe is not None
    assert window._closing is False
    assert [
        control.isEnabled()
        for control in operation_controls(window)
    ] == enabled_before

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert application.close_calls == 1
    assert worker.cancel_calls == 2
    assert thread.quit_calls == 2
