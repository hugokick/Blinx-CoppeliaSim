from __future__ import annotations

from threading import Event, Thread

import pytest

from vision_platform.session import VisionLabSession


class FakeApplication:
    def __init__(
        self,
        number: int,
        *,
        operations: list[str] | None = None,
        fail_load: Exception | None = None,
        fail_open: Exception | None = None,
        fail_close: Exception | None = None,
    ) -> None:
        self.number = number
        self.operations = operations if operations is not None else []
        self.fail_load = fail_load
        self.fail_open = fail_open
        self.fail_close = fail_close
        self.close_calls = 0
        self.close_quarantined_calls = 0
        self.load_calls = 0
        self.open_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.operations.append(f"{self.number}.close")
        if self.fail_close is not None:
            raise self.fail_close

    def close_quarantined(self) -> None:
        self.close_quarantined_calls += 1
        self.operations.append(f"{self.number}.close_quarantined")
        if self.fail_close is not None:
            raise self.fail_close

    def load_and_start_scene(self, scene_path: object | None = None) -> None:
        self.load_calls += 1
        suffix = "" if scene_path is None else f":{scene_path}"
        self.operations.append(f"{self.number}.load{suffix}")
        if self.fail_load is not None:
            raise self.fail_load

    def open(self) -> None:
        self.open_calls += 1
        self.operations.append(f"{self.number}.open")
        if self.fail_open is not None:
            raise self.fail_open


def test_session_reset_closes_old_and_opens_fresh_application() -> None:
    operations: list[str] = []
    created: list[FakeApplication] = []

    def factory() -> FakeApplication:
        app = FakeApplication(len(created) + 1, operations=operations)
        created.append(app)
        return app

    first = factory()
    session = VisionLabSession(application=first, factory=factory)

    replacement = session.reset_simulation()

    assert operations == ["1.close", "2.load", "2.open"]
    assert replacement is created[1]
    assert session.application is replacement


def test_session_replace_closes_old_then_loads_opens_validates_and_publishes(
) -> None:
    operations: list[str] = []
    old = FakeApplication(1, operations=operations)
    replacement = FakeApplication(2, operations=operations)
    session = VisionLabSession(application=old, factory=lambda: old)
    published: list[FakeApplication] = []
    def publish(application: FakeApplication) -> None:
        operations.append(f"{application.number}.publish")
        published.append(application)

    session.subscribe(publish)

    result = session.replace_application(
        lambda: replacement,
        scene_path="new-scene.ttt",
        validate=lambda application: operations.append(
            f"{application.number}.validate"
        ),
    )

    assert result is replacement
    assert session.application is replacement
    assert published == [replacement]
    assert operations == [
        "1.close",
        "2.load:new-scene.ttt",
        "2.open",
        "2.validate",
        "2.publish",
    ]


@pytest.mark.parametrize("stage", ["factory", "load", "open", "validate"])
def test_session_replace_failure_keeps_old_reference_cleans_new_and_does_not_publish(
    stage: str,
) -> None:
    old = FakeApplication(1)
    replacement = FakeApplication(
        2,
        fail_load=RuntimeError("load failed") if stage == "load" else None,
        fail_open=RuntimeError("open failed") if stage == "open" else None,
    )

    def factory() -> FakeApplication:
        if stage == "factory":
            raise RuntimeError("factory failed")
        return replacement

    session = VisionLabSession(application=old, factory=lambda: old)
    published: list[FakeApplication] = []
    session.subscribe(published.append)

    def validate(_application: FakeApplication) -> None:
        if stage == "validate":
            raise RuntimeError("validate failed")

    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        session.replace_application(
            factory,
            scene_path="new-scene.ttt",
            validate=validate,
        )

    assert session.application is old
    assert old.close_calls == 1
    assert replacement.close_calls == (0 if stage == "factory" else 1)
    assert published == []


def test_failed_replace_preserves_previous_active_factory_for_reset() -> None:
    old = FakeApplication(1)
    active_replacements = [FakeApplication(2), FakeApplication(3)]
    active_factory_calls = 0

    def initial_factory() -> FakeApplication:
        raise AssertionError("initial factory must not be reused")

    def active_factory() -> FakeApplication:
        nonlocal active_factory_calls
        application = active_replacements[active_factory_calls]
        active_factory_calls += 1
        return application

    def failing_factory() -> FakeApplication:
        raise RuntimeError("replacement factory failed")

    session = VisionLabSession(application=old, factory=initial_factory)
    session.replace_application(active_factory, scene_path="active-scene.ttt")

    with pytest.raises(RuntimeError, match="replacement factory failed"):
        session.replace_application(
            failing_factory,
            scene_path="failed-scene.ttt",
        )

    replacement = session.reset_simulation()

    assert replacement is active_replacements[1]
    assert session.application is active_replacements[1]
    assert active_factory_calls == 2


def test_quarantined_reset_never_calls_old_normal_close_or_backend_rpc() -> None:
    operations: list[str] = []
    created: list[FakeApplication] = []

    def factory() -> FakeApplication:
        app = FakeApplication(len(created) + 1, operations=operations)
        created.append(app)
        return app

    first = factory()
    session = VisionLabSession(application=first, factory=factory)

    replacement = session.reset_simulation_quarantined()

    assert operations == [
        "1.close_quarantined",
        "2.load",
        "2.open",
    ]
    assert first.close_calls == 0
    assert first.close_quarantined_calls == 1
    assert replacement is created[1]
    assert session.application is replacement
    session.close()
    assert first.close_quarantined_calls == 1


def test_session_notifies_all_subscribers_after_replacement() -> None:
    created: list[FakeApplication] = []

    def factory() -> FakeApplication:
        app = FakeApplication(len(created) + 1)
        created.append(app)
        return app

    session = VisionLabSession(application=factory(), factory=factory)
    received: list[tuple[str, FakeApplication]] = []
    session.subscribe(lambda app: received.append(("first", app)))
    session.subscribe(lambda app: received.append(("second", app)))

    replacement = session.reset_simulation()

    assert received == [("first", replacement), ("second", replacement)]


def test_session_unsubscribe_is_idempotent() -> None:
    created: list[FakeApplication] = []

    def factory() -> FakeApplication:
        app = FakeApplication(len(created) + 1)
        created.append(app)
        return app

    session = VisionLabSession(application=factory(), factory=factory)
    received: list[FakeApplication] = []
    unsubscribe = session.subscribe(received.append)

    unsubscribe()
    unsubscribe()
    session.reset_simulation()

    assert received == []


def test_session_close_is_idempotent_and_rejects_new_work() -> None:
    application = FakeApplication(1)
    session = VisionLabSession(
        application=application,
        factory=lambda: FakeApplication(2),
    )

    session.close()
    session.close()

    assert application.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.reset_simulation()
    with pytest.raises(RuntimeError, match="closed"):
        session.subscribe(lambda _application: None)


def test_session_normal_close_propagates_error_and_remains_idempotent() -> None:
    close_error = RuntimeError("application-close-failed")
    application = FakeApplication(1, fail_close=close_error)
    session = VisionLabSession(
        application=application,
        factory=lambda: FakeApplication(2),
    )

    with pytest.raises(RuntimeError, match="application-close-failed"):
        session.close()
    session.close()

    assert application.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.reset_simulation()


def test_quarantined_close_never_calls_normal_application_close() -> None:
    application = FakeApplication(1)
    session = VisionLabSession(
        application=application,
        factory=lambda: FakeApplication(2),
    )

    session.close_quarantined()
    session.close_quarantined()
    session.close()

    assert application.close_calls == 0
    assert application.close_quarantined_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.reset_simulation()


@pytest.mark.parametrize("stage", ["factory", "load", "open"])
def test_session_reset_failure_keeps_old_application_and_cleans_replacement(
    stage: str,
) -> None:
    old = FakeApplication(1)
    replacement = FakeApplication(
        2,
        fail_load=RuntimeError("load failed") if stage == "load" else None,
        fail_open=RuntimeError("open failed") if stage == "open" else None,
    )

    def factory() -> FakeApplication:
        if stage == "factory":
            raise RuntimeError("factory failed")
        return replacement

    session = VisionLabSession(application=old, factory=factory)

    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        session.reset_simulation()

    assert session.application is old
    assert old.close_calls == 1
    assert replacement.close_calls == (0 if stage == "factory" else 1)
    session.close()
    assert old.close_calls == 1


def test_session_reset_preserves_primary_failure_when_cleanup_fails() -> None:
    old = FakeApplication(1)
    primary_error = RuntimeError("load failed")
    cleanup_error = RuntimeError("cleanup failed")
    replacement = FakeApplication(
        2,
        fail_load=primary_error,
        fail_close=cleanup_error,
    )
    factory_calls = 0

    def factory() -> FakeApplication:
        nonlocal factory_calls
        factory_calls += 1
        return replacement

    session = VisionLabSession(application=old, factory=factory)

    with pytest.raises(RuntimeError, match="load failed") as captured:
        session.reset_simulation()

    assert captured.value is primary_error
    assert captured.value.__cause__ is cleanup_error
    assert session.application is old
    assert factory_calls == 1
    assert replacement.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.reset_simulation()
    with pytest.raises(RuntimeError, match="closed"):
        session.subscribe(lambda _application: None)
    session.close()
    assert factory_calls == 1
    assert old.close_calls == 1
    assert replacement.close_calls == 1


def test_session_becomes_fail_closed_when_old_application_close_fails() -> None:
    old = FakeApplication(1, fail_close=RuntimeError("close failed"))
    created: list[FakeApplication] = []

    def factory() -> FakeApplication:
        application = FakeApplication(2)
        created.append(application)
        return application

    session = VisionLabSession(application=old, factory=factory)

    with pytest.raises(RuntimeError, match="close failed"):
        session.reset_simulation()

    assert created == []
    assert session.application is old
    assert old.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.reset_simulation()
    with pytest.raises(RuntimeError, match="closed"):
        session.subscribe(lambda _application: None)
    session.close()
    assert created == []
    assert old.close_calls == 1


def test_session_commits_replacement_and_notifies_remaining_handlers_on_error(
) -> None:
    old = FakeApplication(1)
    replacement = FakeApplication(2)
    session = VisionLabSession(
        application=old,
        factory=lambda: replacement,
    )
    received: list[FakeApplication] = []

    def failing_handler(_application: FakeApplication) -> None:
        raise RuntimeError("subscriber failed")

    session.subscribe(failing_handler)
    session.subscribe(received.append)

    with pytest.raises(RuntimeError, match="subscriber failed"):
        session.reset_simulation()

    assert session.application is replacement
    assert received == [replacement]


def test_session_notifies_without_holding_state_lock() -> None:
    old = FakeApplication(1)
    replacement = FakeApplication(2)
    session = VisionLabSession(
        application=old,
        factory=lambda: replacement,
    )
    observed: list[FakeApplication] = []

    def handler(_application: FakeApplication) -> None:
        reader = Thread(target=lambda: observed.append(session.application))
        reader.start()
        reader.join(timeout=1.0)
        assert not reader.is_alive()

    session.subscribe(handler)

    session.reset_simulation()

    assert observed == [replacement]


@pytest.mark.parametrize("method_name", ["reset_simulation", "close"])
def test_session_rejects_lifecycle_reentry_from_notification_callback(
    method_name: str,
) -> None:
    created: list[FakeApplication] = []

    def factory() -> FakeApplication:
        application = FakeApplication(len(created) + 1)
        created.append(application)
        return application

    session = VisionLabSession(application=factory(), factory=factory)
    attempts = 0

    def handler(_application: FakeApplication) -> None:
        nonlocal attempts
        if attempts:
            return
        attempts += 1
        with pytest.raises(RuntimeError, match="notification callback"):
            getattr(session, method_name)()

    session.subscribe(handler)

    replacement = session.reset_simulation()

    assert attempts == 1
    assert len(created) == 2
    assert created[1] is replacement
    assert session.application is replacement
    assert replacement.close_calls == 0
    session.close()
    assert replacement.close_calls == 1


def test_session_close_waits_for_reset_notification_to_finish() -> None:
    old = FakeApplication(1)
    replacement = FakeApplication(2)
    session = VisionLabSession(
        application=old,
        factory=lambda: replacement,
    )
    notification_entered = Event()
    allow_notification = Event()
    close_started = Event()
    close_done = Event()
    operations: list[str] = []
    errors: list[BaseException] = []

    def handler(_application: FakeApplication) -> None:
        operations.append("handler.enter")
        notification_entered.set()
        assert allow_notification.wait(timeout=2.0)
        operations.append("handler.exit")

    def reset() -> None:
        try:
            session.reset_simulation()
        except BaseException as error:
            errors.append(error)

    def close() -> None:
        close_started.set()
        try:
            session.close()
        except BaseException as error:
            errors.append(error)
        finally:
            close_done.set()
            operations.append("close.done")

    session.subscribe(handler)
    reset_thread = Thread(target=reset)
    reset_thread.start()
    assert notification_entered.wait(timeout=2.0)

    close_thread = Thread(target=close)
    close_thread.start()
    assert close_started.wait(timeout=2.0)
    closed_before_notification_finished = close_done.wait(timeout=0.2)
    allow_notification.set()
    reset_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)

    assert closed_before_notification_finished is False
    assert not reset_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert operations == ["handler.enter", "handler.exit", "close.done"]
    assert replacement.close_calls == 1


def test_session_serializes_reset_notifications_in_application_order() -> None:
    created: list[FakeApplication] = []

    def factory() -> FakeApplication:
        application = FakeApplication(len(created) + 1)
        created.append(application)
        return application

    session = VisionLabSession(application=factory(), factory=factory)
    first_notification_entered = Event()
    allow_first_notification = Event()
    second_reset_started = Event()
    second_reset_done = Event()
    received: list[FakeApplication] = []
    errors: list[BaseException] = []

    def handler(application: FakeApplication) -> None:
        if application.number == 2:
            first_notification_entered.set()
            assert allow_first_notification.wait(timeout=2.0)
        received.append(application)

    def reset(
        started: Event | None = None,
        done: Event | None = None,
    ) -> None:
        if started is not None:
            started.set()
        try:
            session.reset_simulation()
        except BaseException as error:
            errors.append(error)
        finally:
            if done is not None:
                done.set()

    session.subscribe(handler)
    first_reset = Thread(target=reset)
    first_reset.start()
    assert first_notification_entered.wait(timeout=2.0)

    second_reset = Thread(
        target=reset,
        args=(second_reset_started, second_reset_done),
    )
    second_reset.start()
    assert second_reset_started.wait(timeout=2.0)
    second_finished_before_first_notification = second_reset_done.wait(
        timeout=0.2
    )
    allow_first_notification.set()
    first_reset.join(timeout=2.0)
    second_reset.join(timeout=2.0)

    assert second_finished_before_first_notification is False
    assert not first_reset.is_alive()
    assert not second_reset.is_alive()
    assert errors == []
    assert received == [created[1], created[2]]
    assert session.application is created[2]
    assert created[1].close_calls == 1


def test_session_serializes_reset_and_close_without_double_closing() -> None:
    close_entered = Event()
    allow_close = Event()
    operations: list[str] = []

    class BlockingApplication(FakeApplication):
        def close(self) -> None:
            super().close()
            close_entered.set()
            assert allow_close.wait(timeout=2.0)

    old = BlockingApplication(1, operations=operations)
    replacement = FakeApplication(2, operations=operations)
    session = VisionLabSession(
        application=old,
        factory=lambda: replacement,
    )
    errors: list[BaseException] = []

    def reset() -> None:
        try:
            session.reset_simulation()
        except BaseException as error:
            errors.append(error)

    def close() -> None:
        try:
            session.close()
        except BaseException as error:
            errors.append(error)

    reset_thread = Thread(target=reset)
    reset_thread.start()
    assert close_entered.wait(timeout=2.0)

    close_thread = Thread(target=close)
    close_thread.start()
    assert close_thread.is_alive()

    allow_close.set()
    reset_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)

    assert not reset_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert operations == [
        "1.close",
        "2.load",
        "2.open",
        "2.close",
    ]
    assert old.close_calls == 1
    assert replacement.close_calls == 1
