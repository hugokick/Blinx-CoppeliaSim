from __future__ import annotations

from threading import RLock, local
from typing import Any, Callable


ApplicationHandler = Callable[[Any], None]


class VisionLabSession:
    """Own the replaceable application used by the teaching UI."""

    def __init__(
        self,
        *,
        application: Any,
        factory: Callable[[], Any],
    ) -> None:
        self._application = application
        self._factory = factory
        self._handlers: list[ApplicationHandler] = []
        self._closed_applications: list[Any] = []
        self._closed = False
        self._state_lock = RLock()
        self._operation_lock = RLock()
        self._callback_state = local()

    @property
    def application(self) -> Any:
        with self._state_lock:
            return self._application

    def subscribe(self, handler: ApplicationHandler) -> Callable[[], None]:
        with self._state_lock:
            self._require_open()
            self._handlers.append(handler)
        unsubscribed = False

        def unsubscribe() -> None:
            nonlocal unsubscribed
            with self._state_lock:
                if unsubscribed:
                    return
                unsubscribed = True
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return unsubscribe

    def reset_simulation(self) -> Any:
        return self._reset_simulation(quarantined=False)

    def reset_simulation_quarantined(self) -> Any:
        """Replace an unusable backend without issuing RPCs on the old app."""
        return self._reset_simulation(quarantined=True)

    def replace_application(
        self,
        factory: Callable[[], Any],
        *,
        scene_path: Any,
        validate: Callable[[Any], None] | None = None,
    ) -> Any:
        """Replace the application after the new scene passes validation."""
        self._reject_lifecycle_reentry()
        with self._operation_lock:
            with self._state_lock:
                self._require_open()
                previous = self._application

            try:
                self._close_application_once(previous)
            except BaseException:
                with self._state_lock:
                    self._closed = True
                    self._handlers.clear()
                raise

            replacement = None
            try:
                replacement = factory()
                replacement.load_and_start_scene(scene_path)
                replacement.open()
                if validate is not None:
                    validate(replacement)
            except BaseException as primary_error:
                cleanup_error: BaseException | None = None
                if replacement is not None:
                    try:
                        self._close_application_once(replacement)
                    except BaseException as error:
                        cleanup_error = error
                if cleanup_error is not None:
                    with self._state_lock:
                        self._closed = True
                        self._handlers.clear()
                    raise primary_error from cleanup_error
                raise

            with self._state_lock:
                self._application = replacement
                self._factory = factory
                handlers = tuple(self._handlers)

            self._notify_handlers(replacement, handlers)
            return replacement

    def _reset_simulation(self, *, quarantined: bool) -> Any:
        self._reject_lifecycle_reentry()
        with self._operation_lock:
            with self._state_lock:
                self._require_open()
                old = self._application

            try:
                if quarantined:
                    self._close_application_quarantined_once(old)
                else:
                    self._close_application_once(old)
            except Exception:
                with self._state_lock:
                    self._closed = True
                    self._handlers.clear()
                raise
            replacement = None
            try:
                replacement = self._factory()
                replacement.load_and_start_scene()
                replacement.open()
            except Exception as primary_error:
                cleanup_error: Exception | None = None
                if replacement is not None:
                    try:
                        self._close_application_once(replacement)
                    except Exception as error:
                        cleanup_error = error
                if cleanup_error is not None:
                    with self._state_lock:
                        self._closed = True
                        self._handlers.clear()
                    raise primary_error from cleanup_error
                raise

            with self._state_lock:
                self._application = replacement
                handlers = tuple(self._handlers)

            self._notify_handlers(replacement, handlers)
            return replacement

    def close(self) -> None:
        self._close(quarantined=False)

    def close_quarantined(self) -> None:
        """Close local resources without issuing RPCs on an unusable app."""
        self._close(quarantined=True)

    def _close(self, *, quarantined: bool) -> None:
        self._reject_lifecycle_reentry()
        with self._operation_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                application = self._application
                self._handlers.clear()
            if quarantined:
                self._close_application_quarantined_once(application)
            else:
                self._close_application_once(application)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("VisionLabSession is closed")

    def _reject_lifecycle_reentry(self) -> None:
        if getattr(self._callback_state, "active", False):
            raise RuntimeError(
                "VisionLabSession lifecycle operations are not allowed "
                "from a notification callback"
            )

    def _notify_handlers(
        self,
        application: Any,
        handlers: tuple[ApplicationHandler, ...],
    ) -> None:
        first_error: Exception | None = None
        self._callback_state.active = True
        try:
            for handler in handlers:
                try:
                    handler(application)
                except Exception as error:
                    if first_error is None:
                        first_error = error
        finally:
            self._callback_state.active = False
        if first_error is not None:
            raise first_error

    def _close_application_once(self, application: Any) -> None:
        if any(item is application for item in self._closed_applications):
            return
        self._closed_applications.append(application)
        application.close()

    def _close_application_quarantined_once(self, application: Any) -> None:
        if any(item is application for item in self._closed_applications):
            return
        self._closed_applications.append(application)
        application.close_quarantined()
