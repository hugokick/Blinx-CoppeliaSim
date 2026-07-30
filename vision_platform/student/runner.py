from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import Any, Callable

from vision_platform.errors import VisionPlatformError
from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.student.protocol import (
    CommandMessage,
    ResponseMessage,
    RunState,
)
from vision_platform.student.safety import (
    StudentExecutionPolicy,
    StudentMotionGuard,
)
from vision_platform.student.validator import (
    ValidationResult,
    validate_program,
    validate_program_bytes,
)
from vision_platform.student.worker import (
    _route_student_stdout_for_process_lifetime,
    run_student_worker,
)


_ACTIVE_STATES = frozenset({RunState.RUNNING, RunState.PAUSED})
_TERMINAL_STATES = frozenset(
    {RunState.PASSED, RunState.FAILED, RunState.CANCELLED}
)
_POLL_SECONDS = 0.01
_ACTION_STOP_GRACE_SECONDS = 0.2
_CLEANUP_ACTION_GRACE_SECONDS = 0.05
_CHILD_EXIT_GRACE_SECONDS = 0.05
_CHILD_JOIN_SECONDS = 0.5
_RESULT_GRACE_SECONDS = 0.25
_RUN_THREAD_JOIN_SECONDS = 0.25
_PROTOCOL_ERROR_CODES = frozenset(
    {
        "COMMAND_NOT_ALLOWED",
        "PROTOCOL_KIND_INVALID",
        "PROTOCOL_VERSION_UNSUPPORTED",
    }
)


class _FrozenDict(dict):
    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("frozen mapping cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


@dataclass(frozen=True)
class StudentRunResult:
    status: str
    summary_path: Path | None
    evidence_dir: Path | None
    error: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.error is not None:
            object.__setattr__(
                self,
                "error",
                _freeze_json(_json_safe(self.error)),
            )


@dataclass(frozen=True)
class StudentRunSnapshot:
    state: RunState
    current_command: str | None
    command_count: int
    elapsed_seconds: float
    error: Mapping[str, Any] | None
    evidence_dir: Path | None
    tcp_mm: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.error is not None:
            object.__setattr__(
                self,
                "error",
                _freeze_json(_json_safe(self.error)),
            )
        if self.tcp_mm is not None:
            values = tuple(float(value) for value in self.tcp_mm)
            if len(values) != 3:
                raise ValueError("tcp_mm must contain X, Y and Z")
            if not all(isfinite(value) for value in values):
                raise ValueError("tcp_mm values must be finite")
            object.__setattr__(
                self,
                "tcp_mm",
                (values[0], values[1], values[2]),
            )


class _StopCommandLoop(Exception):
    pass


class _BackendActionStuck(Exception):
    def __init__(self, command_name: str) -> None:
        self.command_name = command_name
        super().__init__(
            f"backend action did not stop: {command_name}"
        )


@dataclass(frozen=True)
class _TimeoutSetting:
    target: Any
    attribute: str
    stage: str
    original: Any
    is_socket_option: bool


def _child_entry(
    source_path: str,
    source_bytes: bytes,
    connection: Any,
    result_queue: Any,
) -> None:
    """Pickle-safe spawn target; the parent never enters student code."""
    _route_student_stdout_for_process_lifetime()
    try:
        result = run_student_worker(
            source_path,
            connection,
            source_bytes=source_bytes,
        )
    except BaseException as error:
        result = {
            "status": "FAIL",
            "error": {
                "code": "STUDENT_WORKER_CRASHED",
                "message": _safe_text(error),
                "type": _safe_type_name(error),
            },
        }
    try:
        result_queue.put(result)
    except BaseException:
        pass
    finally:
        try:
            connection.close()
        except BaseException:
            pass


def _safe_text(value: Any, *, limit: int = 2000) -> str:
    try:
        text = str(value)
    except BaseException:
        return "<unprintable>"
    if type(text) is not str:
        return "<invalid text>"
    return text[:limit]


def _safe_type_name(value: Any) -> str:
    try:
        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"[:200]
    except BaseException:
        return "<unknown type>"


def _json_safe(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    if depth > 8:
        return "<maximum depth exceeded>"
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        return value[:10_000]
    if type(value) is int:
        if value.bit_length() > 4096:
            return "<integer too large>"
        return value
    if type(value) is float:
        return value if isfinite(value) else _safe_text(value)
    if isinstance(value, Path):
        return str(value)

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return "<recursive value>"
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            try:
                items = tuple(value.items())
            except BaseException:
                return "<unreadable mapping>"
            for key, item in items[:200]:
                output[_safe_text(key, limit=200)] = _json_safe(
                    item,
                    depth=depth + 1,
                    seen=seen,
                )
            return output
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            try:
                items = tuple(value)
            except BaseException:
                return "<unreadable sequence>"
            return [
                _json_safe(item, depth=depth + 1, seen=seen)
                for item in items[:200]
            ]
        try:
            rendered = repr(value)
        except BaseException:
            rendered = "<unprintable>"
        if type(rendered) is not str:
            rendered = "<invalid repr>"
        return rendered[:2000]
    finally:
        seen.discard(identity)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict(
            {
                str(key): _freeze_json(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _seal_terminal_payload(
    error: Mapping[str, Any] | None,
    cleanup_errors: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    sealed_error = _json_safe(error) if error is not None else None
    if sealed_error is not None and not isinstance(sealed_error, Mapping):
        sealed_error = _error(
            "STUDENT_TERMINAL_ERROR_INVALID",
            "Terminal error payload is not a JSON object",
            details={"original_error": sealed_error},
        )
    sealed_cleanup_errors = _json_safe(cleanup_errors)
    if isinstance(sealed_error, Mapping):
        sealed_details = sealed_error.get("details")
        embedded_cleanup_errors = (
            sealed_details.get("cleanup_errors")
            if isinstance(sealed_details, Mapping)
            else None
        )
        if isinstance(embedded_cleanup_errors, list):
            # Quarantined errors expose cleanup details inside the public
            # error payload. Reuse that already depth-bounded value for the
            # top-level evidence field so every terminal consumer observes
            # exactly one canonical cleanup tree.
            sealed_cleanup_errors = embedded_cleanup_errors
    if not isinstance(sealed_cleanup_errors, list):
        sealed_cleanup_errors = [
            {
                "stage": "terminal.cleanup_errors",
                "error": _error(
                    "STUDENT_TERMINAL_ERROR_INVALID",
                    "Cleanup error payload is not a JSON array",
                    details={
                        "original_cleanup_errors": (
                            sealed_cleanup_errors
                        )
                    },
                ),
            }
        ]
    payload = {
        "error": sealed_error,
        "cleanup_errors": sealed_cleanup_errors,
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return (
        dict(sealed_error) if sealed_error is not None else None,
        sealed_cleanup_errors,
    )


def _error(
    code: str,
    message: str,
    *,
    details: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": _safe_text(code, limit=200),
        "message": _safe_text(message),
    }
    if details is not None:
        payload["details"] = _json_safe(details)
    for name, value in extra.items():
        payload[name] = _json_safe(value)
    # This is a final defensive gate for evidence serialization.
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload


def _exception_error(
    error: BaseException,
    *,
    code: str = "STUDENT_COMMAND_FAILED",
) -> dict[str, Any]:
    if isinstance(error, VisionPlatformError):
        return _error(
            error.code,
            _safe_text(error),
            details=error.details,
        )
    return _error(
        code,
        _safe_text(error),
        type=_safe_type_name(error),
    )


def _protocol_exception_error(error: BaseException) -> dict[str, Any]:
    message = _safe_text(error)
    candidate = message.split(":", 1)[0]
    code = (
        candidate
        if candidate in _PROTOCOL_ERROR_CODES
        else "STUDENT_PROTOCOL_ERROR"
    )
    return _error(code, message, type=_safe_type_name(error))


def _error_with_cleanup_details(
    error: Mapping[str, Any] | None,
    cleanup_errors: Sequence[Mapping[str, Any]],
    *,
    quarantined: bool,
) -> dict[str, Any]:
    if error is None:
        payload = _error(
            "STUDENT_BACKEND_COMMAND_STUCK",
            "底层机器人命令无法安全终止",
        )
    else:
        safe = _json_safe(error)
        payload = dict(safe) if isinstance(safe, Mapping) else _error(
            "STUDENT_BACKEND_COMMAND_STUCK",
            "底层机器人命令无法安全终止",
            details={"original_error": safe},
        )
    existing = payload.get("details")
    details = (
        dict(existing)
        if isinstance(existing, Mapping)
        else {}
    )
    details["cleanup_errors"] = _json_safe(cleanup_errors)
    details["quarantined"] = quarantined
    payload["details"] = details
    return payload


def _error_with_quarantine_details(
    error: Mapping[str, Any],
    *,
    reason: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    safe = _json_safe(error)
    payload = dict(safe) if isinstance(safe, Mapping) else _error(
        "STUDENT_BACKEND_CONNECTION_QUARANTINED",
        "CoppeliaSim backend connection is unusable",
    )
    existing = payload.get("details")
    details = dict(existing) if isinstance(existing, Mapping) else {}
    details["quarantined"] = True
    details["connection_unusable"] = True
    if reason is not None and reason != error:
        details["quarantine_reason"] = _json_safe(reason)
    payload["details"] = details
    return payload


def _is_transport_timeout(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    eagain_values = {errno.EAGAIN, errno.EWOULDBLOCK}
    while pending:
        candidate = pending.pop()
        identity = id(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        if getattr(candidate, "errno", None) in eagain_values:
            return True
        candidate_type = type(candidate)
        if (
            candidate_type.__name__ == "Again"
            and candidate_type.__module__.split(".", 1)[0] == "zmq"
        ):
            return True
        for related in (
            getattr(candidate, "__cause__", None),
            getattr(candidate, "__context__", None),
        ):
            if isinstance(related, BaseException):
                pending.append(related)
    return False


class StudentProgramController:
    def __init__(
        self,
        *,
        session: Any,
        execution_policy: StudentExecutionPolicy,
        output_root: str | Path,
    ) -> None:
        self._session = session
        self._policy = execution_policy
        self._output_root = Path(output_root).expanduser().resolve()
        self._context = multiprocessing.get_context("spawn")

        self._condition = threading.Condition(threading.RLock())
        self._process_lock = threading.RLock()
        self._completion_lock = threading.Lock()
        self._client_timeout_lock = threading.RLock()
        self._done = threading.Event()
        self._handlers: list[Callable[[StudentRunSnapshot], None]] = []

        self._state = RunState.EMPTY
        self._program_path: Path | None = None
        self._validation: ValidationResult | None = None
        self._application = session.application
        self._guard = self._new_guard(self._application)

        self._evidence: StudentRunEvidence | None = None
        self._result: StudentRunResult | None = None
        self._error: dict[str, Any] | None = None
        self._cleanup_errors: list[dict[str, Any]] = []
        self._last_pose: tuple[float, float, float] | None = None
        self._safety_violation_count = 0
        self._command_count = 0
        self._current_command: str | None = None
        self._started_monotonic: float | None = None
        self._active_command_started: float | None = None
        self._active_command_name: str | None = None
        self._step_permits = 0
        self._starting = False
        self._terminalizing = False
        self._stop_requested = False
        self._requested_status: str | None = None
        self._requested_error: dict[str, Any] | None = None

        self._parent_connection: Any = None
        self._process: Any = None
        self._result_queue: Any = None
        self._command_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._backend_action_thread: threading.Thread | None = None
        self._backend_action_name: str | None = None
        self._backend_quarantined = False
        self._backend_quarantine_error: dict[str, Any] | None = None
        self._client_timeout_settings: list[_TimeoutSetting] = []
        self._client_timeout_bounded = False

    @property
    def state(self) -> RunState:
        with self._condition:
            return self._state

    @property
    def command_count(self) -> int:
        with self._condition:
            return self._command_count

    @property
    def process_is_alive(self) -> bool:
        with self._process_lock:
            process = self._process
            if process is None:
                return False
            try:
                return bool(process.is_alive())
            except (AssertionError, ValueError):
                return False

    def subscribe(
        self,
        handler: Callable[[StudentRunSnapshot], None],
    ) -> Callable[[], None]:
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._condition:
            self._handlers.append(handler)
        unsubscribed = False

        def unsubscribe() -> None:
            nonlocal unsubscribed
            with self._condition:
                if unsubscribed:
                    return
                unsubscribed = True
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return unsubscribe

    def load(self, program_path: str | Path) -> Path:
        selected = Path(program_path).expanduser().resolve()
        with self._condition:
            if (
                self._state in _ACTIVE_STATES
                or self._state in _TERMINAL_STATES
                or self._state is RunState.RESETTING
                or self._starting
            ):
                raise RuntimeError(
                    "load requires an idle controller; reset terminal runs"
                )
            self._program_path = selected
            self._validation = None
            self._state = RunState.LOADED
            self._error = None
        self._emit_snapshot()
        return selected

    def validate(
        self,
        program_path: str | Path | None = None,
    ) -> ValidationResult:
        if program_path is not None:
            selected = Path(program_path).expanduser().resolve()
            with self._condition:
                current = self._program_path
            if current != selected:
                self.load(selected)

        with self._condition:
            if self._state not in {RunState.LOADED, RunState.VALIDATED}:
                raise RuntimeError("validate requires LOADED state")
            selected = self._program_path
        assert selected is not None
        result = validate_program(selected)
        with self._condition:
            if self._state not in {RunState.LOADED, RunState.VALIDATED}:
                raise RuntimeError("controller state changed during validation")
            self._validation = result
            self._state = (
                RunState.VALIDATED if result.ok else RunState.LOADED
            )
            self._error = None
        self._emit_snapshot()
        return result

    def start(self, paused: bool = False) -> None:
        if type(paused) is not bool:
            raise TypeError("paused must be a boolean")
        with self._condition:
            if self._state is not RunState.VALIDATED or self._starting:
                raise RuntimeError("start requires VALIDATED state")
            self._application = self._session.application
            backend = getattr(
                getattr(self._application, "config", None),
                "robot_backend",
                None,
            )
            if backend != "sim":
                raise RuntimeError("REAL_BACKEND_NOT_AUTHORIZED")
            self._guard = self._new_guard(self._application)
            selected = self._program_path
            self._starting = True
        assert selected is not None

        try:
            evidence = StudentRunEvidence.create(
                output_root=self._output_root,
                program_path=selected,
                robot_backend="sim",
            )
        except BaseException as create_error:
            error = _exception_error(
                create_error,
                code="STUDENT_EVIDENCE_FAILED",
            )
            with self._condition:
                self._starting = False
                self._state = RunState.FAILED
                self._error = error
                self._result = StudentRunResult(
                    "FAILED",
                    None,
                    None,
                    error,
                )
                self._done.set()
                self._condition.notify_all()
            self._emit_snapshot()
            return

        with self._condition:
            self._prepare_run_locked(evidence)
        try:
            self._bound_client_timeout()
        except BaseException as timeout_guard_error:
            error = self._quarantine_backend(
                _exception_error(
                    timeout_guard_error,
                    code="STUDENT_TRANSPORT_TIMEOUT_GUARD_FAILED",
                )
            )
            with self._condition:
                self._starting = False
            self._complete_run("FAILED", error)
            return

        try:
            captured_source = evidence.source_path.read_bytes()
        except BaseException as capture_error:
            with self._condition:
                self._starting = False
            self._complete_run(
                "FAILED",
                _exception_error(
                    capture_error,
                    code="STUDENT_EVIDENCE_FAILED",
                ),
            )
            return

        captured_digest = hashlib.sha256(captured_source).hexdigest()
        if captured_digest != evidence.source_sha256:
            with self._condition:
                self._starting = False
            self._complete_run(
                "FAILED",
                _error(
                    "STUDENT_EVIDENCE_HASH_MISMATCH",
                    "学生运行快照与证据哈希不一致",
                    details={
                        "expected_sha256": evidence.source_sha256,
                        "captured_sha256": captured_digest,
                    },
                ),
            )
            return

        snapshot_validation = validate_program_bytes(
            captured_source,
            evidence.source_path,
        )
        if not snapshot_validation.ok:
            invalid_error = _error(
                "STUDENT_SNAPSHOT_INVALID",
                "运行快照未通过静态检查",
                details={
                    "issues": [
                        {
                            "code": issue.code,
                            "message": issue.message,
                            "line": issue.line,
                            "column": issue.column,
                        }
                        for issue in snapshot_validation.issues
                    ]
                },
            )
            with self._condition:
                self._starting = False
            self._complete_run("FAILED", invalid_error)
            return

        initial_state = RunState.PAUSED if paused else RunState.RUNNING
        try:
            evidence.record_event(
                initial_state.value,
                "学生程序开始",
                paused=paused,
            )
        except BaseException as event_error:
            with self._condition:
                self._starting = False
            self._complete_run(
                "FAILED",
                _exception_error(
                    event_error,
                    code="STUDENT_EVIDENCE_FAILED",
                ),
            )
            return

        parent_connection = None
        child_connection = None
        result_queue = None
        process = None
        try:
            parent_connection, child_connection = self._context.Pipe(
                duplex=True
            )
            result_queue = self._context.Queue(maxsize=1)
            process = self._context.Process(
                target=_child_entry,
                args=(
                    str(evidence.source_path),
                    captured_source,
                    child_connection,
                    result_queue,
                ),
                name=f"StudentProgram-{evidence.run_id}",
            )
            process.start()
            child_connection.close()
            child_connection = None
        except BaseException as spawn_error:
            if process is not None:
                try:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=_CHILD_JOIN_SECONDS)
                except (AssertionError, OSError, ValueError):
                    pass
            for resource in (child_connection, parent_connection):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException:
                        pass
            if result_queue is not None:
                try:
                    result_queue.close()
                    result_queue.join_thread()
                except BaseException:
                    pass
            with self._condition:
                self._starting = False
            self._complete_run(
                "FAILED",
                _exception_error(
                    spawn_error,
                    code="STUDENT_WORKER_START_FAILED",
                ),
            )
            return

        with self._process_lock:
            self._parent_connection = parent_connection
            self._result_queue = result_queue
            self._process = process
        with self._condition:
            self._state = initial_state
            self._started_monotonic = time.monotonic()
            self._starting = False
            self._condition.notify_all()

        self._command_thread = threading.Thread(
            target=self._command_loop,
            name=f"StudentCommandLoop-{evidence.run_id}",
            daemon=True,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name=f"StudentWatchdog-{evidence.run_id}",
            daemon=True,
        )
        try:
            self._command_thread.start()
            self._watchdog_thread.start()
        except BaseException as thread_error:
            self._request_stop(
                "FAILED",
                _exception_error(
                    thread_error,
                    code="STUDENT_CONTROLLER_THREAD_FAILED",
                ),
            )
            self._reap_child(force=True)
            if not self._command_thread.is_alive():
                self._complete_run(
                    "FAILED",
                    self._requested_error,
                )
        self._emit_snapshot()

    def pause(self) -> None:
        with self._condition:
            if self._state is not RunState.RUNNING:
                raise RuntimeError("pause requires RUNNING state")
            self._state = RunState.PAUSED
            self._step_permits = 0
            self._condition.notify_all()
        self._record_control_event("PAUSED", "学生程序暂停")
        self._emit_snapshot()

    def resume(self) -> None:
        with self._condition:
            if self._state is not RunState.PAUSED:
                raise RuntimeError("resume requires PAUSED state")
            self._state = RunState.RUNNING
            self._step_permits = 0
            self._condition.notify_all()
        self._record_control_event("RUNNING", "学生程序继续")
        self._emit_snapshot()

    def step(self) -> None:
        with self._condition:
            if self._state is not RunState.PAUSED:
                raise RuntimeError("step requires PAUSED state")
            self._step_permits += 1
            self._condition.notify_all()
        self._record_control_event("PAUSED", "放行一条学生命令")
        self._emit_snapshot()

    def cancel(self) -> None:
        with self._condition:
            if self._state in _TERMINAL_STATES:
                return
            if self._state not in _ACTIVE_STATES:
                raise RuntimeError("cancel requires RUNNING or PAUSED state")
            already_requested = self._stop_requested
        if already_requested:
            return
        cancellation = _error(
            "STUDENT_PROGRAM_CANCELLED",
            "学生程序已由控制器停止",
        )
        try:
            evidence = self._evidence
            if evidence is not None:
                evidence.record_event(
                    "CANCELLED",
                    "控制器请求停止学生程序",
                )
        except BaseException as event_error:
            self._request_stop(
                "FAILED",
                _exception_error(
                    event_error,
                    code="STUDENT_EVIDENCE_FAILED",
                ),
            )
        else:
            self._request_stop("CANCELLED", cancellation)
        self._reap_child(force=True)
        self._emit_snapshot()

    def wait(self, timeout_s: float | None = None) -> StudentRunResult:
        if timeout_s is not None:
            if type(timeout_s) not in {int, float}:
                raise TypeError("timeout_s must be a number or None")
            timeout = float(timeout_s)
            if not isfinite(timeout) or timeout < 0:
                raise ValueError("timeout_s must be finite and non-negative")
        else:
            timeout = None
        with self._condition:
            if (
                self._result is None
                and self._state
                in {RunState.EMPTY, RunState.LOADED, RunState.VALIDATED}
                and not self._starting
            ):
                raise RuntimeError("wait requires a started run")
        if not self._done.wait(timeout):
            raise TimeoutError("student program did not finish before timeout")
        with self._condition:
            result = self._result
        if result is None:
            raise RuntimeError("student run completed without a result")
        return result

    def wait_for_quiescence(self, timeout_s: float) -> bool:
        """Wait until no run-owned process or thread can touch the backend."""
        if type(timeout_s) not in {int, float}:
            raise TypeError("timeout_s must be a number")
        timeout = float(timeout_s)
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_s must be finite and non-negative")

        deadline = time.monotonic() + timeout
        current = threading.current_thread()
        while True:
            with self._condition:
                state = self._state
                starting = self._starting
                terminalizing = self._terminalizing
                threads = (
                    self._command_thread,
                    self._watchdog_thread,
                    self._backend_action_thread,
                )
            process_alive = self.process_is_alive
            threads_alive = tuple(
                thread
                for thread in threads
                if thread is not None and thread.is_alive()
            )
            lifecycle_active = (
                state in _ACTIVE_STATES
                or state is RunState.RESETTING
                or terminalizing
            )
            if (
                not starting
                and not lifecycle_active
                and not process_alive
                and not threads_alive
            ):
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            for thread in threads_alive:
                if thread is current:
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                thread.join(timeout=min(_POLL_SECONDS, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            with self._condition:
                self._condition.wait(
                    timeout=min(_POLL_SECONDS, remaining)
                )

    def reset(self) -> Any:
        with self._condition:
            if self._state not in _TERMINAL_STATES:
                raise RuntimeError("reset requires a terminal run state")
            if self.process_is_alive:
                raise RuntimeError("cannot reset while worker is alive")
        if not self._join_run_threads():
            raise RuntimeError(
                "prior student run threads are still active"
            )
        if self._backend_action_is_alive():
            raise RuntimeError(
                "STUDENT_BACKEND_COMMAND_STUCK: "
                "reset is blocked by quarantined backend action"
            )
        with self._condition:
            quarantined_reset = self._backend_quarantined
        if quarantined_reset:
            self._discard_client_timeout_state()
        else:
            timeout_restore_error = self._restore_client_timeout()
            if timeout_restore_error is not None:
                self._quarantine_backend(
                    _error(
                        "STUDENT_CLIENT_TIMEOUT_RESTORE_FAILED",
                        "Unable to restore CoppeliaSim transport timeouts "
                        "before reset",
                        details={
                            "cleanup_errors": [timeout_restore_error],
                        },
                    )
                )
                self._discard_client_timeout_state()
                quarantined_reset = True
        with self._condition:
            if self._backend_quarantined:
                quarantined_reset = True
        if quarantined_reset:
            reset_quarantined = getattr(
                self._session,
                "reset_simulation_quarantined",
                None,
            )
            if not callable(reset_quarantined):
                raise RuntimeError(
                    "QUARANTINED_RESET_UNAVAILABLE: "
                    "session does not support safe quarantined reset"
                )
        with self._condition:
            if self._state not in _TERMINAL_STATES:
                raise RuntimeError("controller state changed during reset")
            if self.process_is_alive:
                raise RuntimeError("cannot reset while worker is alive")
            self._state = RunState.RESETTING
            self._condition.notify_all()
        self._emit_snapshot()

        try:
            if quarantined_reset:
                application = reset_quarantined()
            else:
                application = self._session.reset_simulation()
            guard = self._new_guard(application)
        except BaseException as reset_error:
            error = _exception_error(
                reset_error,
                code="STUDENT_RESET_FAILED",
            )
            with self._condition:
                self._state = RunState.FAILED
                self._error = error
                self._condition.notify_all()
            self._emit_snapshot()
            raise

        self._close_ipc()
        with self._condition:
            self._application = application
            self._guard = guard
            self._evidence = None
            self._result = None
            self._error = None
            self._cleanup_errors = []
            self._last_pose = None
            self._safety_violation_count = 0
            self._command_count = 0
            self._current_command = None
            self._started_monotonic = None
            self._active_command_started = None
            self._active_command_name = None
            self._step_permits = 0
            self._terminalizing = False
            self._stop_requested = False
            self._requested_status = None
            self._requested_error = None
            self._process = None
            self._command_thread = None
            self._watchdog_thread = None
            self._backend_action_thread = None
            self._backend_action_name = None
            self._backend_quarantined = False
            self._backend_quarantine_error = None
            self._done.clear()
            self._state = RunState.VALIDATED
            self._condition.notify_all()
        self._emit_snapshot()
        return application

    def _backend_action_is_alive(self) -> bool:
        with self._condition:
            thread = self._backend_action_thread
        return bool(thread is not None and thread.is_alive())

    def _join_run_threads(self) -> bool:
        current = threading.current_thread()
        with self._condition:
            threads = (
                self._command_thread,
                self._watchdog_thread,
            )
        if any(thread is current for thread in threads):
            return False
        for thread in threads:
            if (
                thread is not None
                and thread.is_alive()
            ):
                thread.join(timeout=_RUN_THREAD_JOIN_SECONDS)
        return not any(
            thread is not None
            and thread.is_alive()
            for thread in threads
        )

    def _bound_client_timeout(self) -> None:
        with self._client_timeout_lock:
            if self._client_timeout_bounded:
                return
            client = getattr(self._application, "client", None)
            if client is None:
                return

            settings_and_values: list[tuple[_TimeoutSetting, Any]] = []

            def capture(
                target: Any,
                attribute: str,
                *,
                stage: str,
                limit: float | int,
                is_socket_option: bool,
            ) -> None:
                try:
                    original = getattr(target, attribute)
                except AttributeError:
                    return
                if (
                    type(original) in {int, float}
                    and isfinite(float(original))
                    and float(original) >= 0
                ):
                    bounded = min(float(original), float(limit))
                else:
                    bounded = limit
                if is_socket_option:
                    bounded = int(bounded)
                settings_and_values.append(
                    (
                        _TimeoutSetting(
                            target=target,
                            attribute=attribute,
                            stage=stage,
                            original=original,
                            is_socket_option=is_socket_option,
                        ),
                        bounded,
                    )
                )

            try:
                capture(
                    client,
                    "timeout",
                    stage="client.timeout",
                    limit=self._policy.command_timeout_s,
                    is_socket_option=False,
                )
                try:
                    socket = getattr(client, "socket")
                except AttributeError:
                    socket = None
                if socket is not None:
                    socket_timeout_ms = max(
                        1,
                        int(ceil(self._policy.command_timeout_s * 1000)),
                    )
                    for attribute in ("RCVTIMEO", "SNDTIMEO"):
                        capture(
                            socket,
                            attribute,
                            stage=f"client.socket.{attribute}",
                            limit=socket_timeout_ms,
                            is_socket_option=True,
                        )
            except BaseException as capture_error:
                raise VisionPlatformError(
                    "STUDENT_TRANSPORT_TIMEOUT_GUARD_FAILED",
                    "Unable to inspect CoppeliaSim transport timeout settings",
                    details={
                        "stage": "capture",
                        "error": _exception_error(capture_error),
                    },
                ) from capture_error

            if not settings_and_values:
                return

            attempted: list[_TimeoutSetting] = []
            try:
                for setting, bounded in settings_and_values:
                    attempted.append(setting)
                    setattr(setting.target, setting.attribute, bounded)
            except BaseException as setting_error:
                pending: list[_TimeoutSetting] = []
                rollback_errors: list[dict[str, Any]] = []
                for setting in reversed(attempted):
                    try:
                        setattr(
                            setting.target,
                            setting.attribute,
                            setting.original,
                        )
                    except BaseException as rollback_error:
                        pending.append(setting)
                        rollback_errors.append(
                            {
                                "stage": setting.stage,
                                "error": _exception_error(rollback_error),
                            }
                        )
                self._client_timeout_settings = list(reversed(pending))
                self._client_timeout_bounded = bool(pending)
                raise VisionPlatformError(
                    "STUDENT_TRANSPORT_TIMEOUT_GUARD_FAILED",
                    "Unable to bound CoppeliaSim transport timeout settings",
                    details={
                        "stage": attempted[-1].stage,
                        "error": _exception_error(setting_error),
                        "rollback_errors": rollback_errors,
                    },
                ) from setting_error

            self._client_timeout_settings = [
                setting for setting, _ in settings_and_values
            ]
            self._client_timeout_bounded = True

    def _restore_client_timeout(
        self,
    ) -> dict[str, Any] | None:
        action_active = self._backend_action_is_alive()
        with self._client_timeout_lock:
            if not self._client_timeout_bounded:
                return None
            pending: list[_TimeoutSetting] = []
            errors: list[dict[str, Any]] = []
            for setting in reversed(self._client_timeout_settings):
                if setting.is_socket_option and action_active:
                    pending.append(setting)
                    continue
                try:
                    setattr(
                        setting.target,
                        setting.attribute,
                        setting.original,
                    )
                except BaseException as restore_error:
                    pending.append(setting)
                    errors.append(
                        {
                            "stage": setting.stage,
                            "error": _exception_error(
                                restore_error,
                                code="STUDENT_CLEANUP_FAILED",
                            ),
                        }
                    )
            self._client_timeout_settings = list(reversed(pending))
            self._client_timeout_bounded = bool(pending)
            if not errors:
                return None
            return {
                "stage": "client.transport_timeout.restore",
                "error": _error(
                    "STUDENT_CLEANUP_FAILED",
                    "Unable to restore CoppeliaSim transport timeouts",
                    details={"errors": errors},
                ),
            }

    def _discard_client_timeout_state(self) -> None:
        with self._client_timeout_lock:
            self._client_timeout_settings = []
            self._client_timeout_bounded = False

    def _record_timeout_restore_failure(
        self,
        cleanup_errors: list[dict[str, Any]],
        restore_error: Mapping[str, Any],
    ) -> dict[str, Any]:
        safe_restore_error = _json_safe(restore_error)
        cleanup_errors.append(dict(safe_restore_error))
        quarantine_error = self._quarantine_backend(
            _error(
                "STUDENT_CLIENT_TIMEOUT_RESTORE_FAILED",
                "Unable to restore CoppeliaSim transport timeout settings",
                details={"restore_error": safe_restore_error},
            )
        )
        if not any(
            item.get("stage") == "backend.connection"
            for item in cleanup_errors
        ):
            cleanup_errors.append(
                {
                    "stage": "backend.connection",
                    "error": quarantine_error,
                }
            )
        return quarantine_error

    def _quarantine_backend(
        self,
        error: Mapping[str, Any],
    ) -> dict[str, Any]:
        quarantine_error = _error_with_quarantine_details(error)
        with self._condition:
            self._backend_quarantined = True
            self._backend_quarantine_error = quarantine_error
            self._last_pose = None
            if (
                self._stop_requested
                and self._requested_error is not None
                and self._requested_error != quarantine_error
            ):
                self._requested_error = _error_with_quarantine_details(
                    self._requested_error,
                    reason=quarantine_error,
                )
            self._condition.notify_all()
        return quarantine_error

    def _new_guard(self, application: Any) -> StudentMotionGuard:
        return StudentMotionGuard(
            workspace=application.workspace,
            policy=self._policy,
        )

    def _prepare_run_locked(self, evidence: StudentRunEvidence) -> None:
        self._evidence = evidence
        self._result = None
        self._error = None
        self._cleanup_errors = []
        self._last_pose = None
        self._safety_violation_count = 0
        self._command_count = 0
        self._current_command = None
        self._started_monotonic = evidence.started_monotonic
        self._active_command_started = None
        self._active_command_name = None
        self._step_permits = 0
        self._terminalizing = False
        self._stop_requested = False
        self._requested_status = None
        self._requested_error = None
        self._backend_action_thread = None
        self._backend_action_name = None
        self._backend_quarantined = False
        self._backend_quarantine_error = None
        self._done.clear()

    def _snapshot_locked(self) -> StudentRunSnapshot:
        elapsed = (
            max(0.0, time.monotonic() - self._started_monotonic)
            if self._started_monotonic is not None
            else 0.0
        )
        return StudentRunSnapshot(
            state=self._state,
            current_command=self._current_command,
            command_count=self._command_count,
            elapsed_seconds=elapsed,
            error=self._error,
            evidence_dir=(
                self._evidence.directory
                if self._evidence is not None
                else None
            ),
            tcp_mm=self._last_pose,
        )

    def _emit_snapshot(self) -> None:
        with self._condition:
            snapshot = self._snapshot_locked()
            handlers = tuple(self._handlers)
        for handler in handlers:
            try:
                handler(snapshot)
            except BaseException:
                pass

    def _record_control_event(self, state: str, message: str) -> None:
        evidence = self._evidence
        if evidence is None:
            return
        try:
            evidence.record_event(state, message)
        except BaseException as event_error:
            self._request_stop(
                "FAILED",
                _exception_error(
                    event_error,
                    code="STUDENT_EVIDENCE_FAILED",
                ),
            )
            self._reap_child(force=True)

    def _elapsed(self) -> float:
        with self._condition:
            started = self._started_monotonic
        if started is None:
            return 0.0
        return max(0.0, time.monotonic() - started)

    def _deadline_error(
        self,
        *,
        now: float,
        command_started: float | None,
        command_name: str | None,
    ) -> dict[str, Any] | None:
        with self._condition:
            run_started = self._started_monotonic
        candidates: list[tuple[float, dict[str, Any]]] = []
        if run_started is not None:
            candidates.append(
                (
                    run_started + self._policy.max_runtime_s,
                    _error(
                        "STUDENT_RUNTIME_TIMEOUT",
                        "学生程序超过总运行时间限制",
                        details={
                            "max_runtime_s": self._policy.max_runtime_s
                        },
                    ),
                )
            )
        if command_started is not None:
            candidates.append(
                (
                    command_started
                    + self._policy.command_timeout_s,
                    _error(
                        "STUDENT_COMMAND_TIMEOUT",
                        "学生命令超过单条等待时间限制",
                        details={
                            "command": command_name,
                            "command_timeout_s": (
                                self._policy.command_timeout_s
                            ),
                        },
                    ),
                )
            )
        expired = [
            candidate
            for candidate in candidates
            if now >= candidate[0]
        ]
        if not expired:
            return None
        return min(expired, key=lambda candidate: candidate[0])[1]

    def _synchronous_deadline_outcome(
        self,
        command_started: float | None,
    ) -> tuple[str, dict[str, Any] | None] | None:
        requested = self._requested_outcome()
        if requested is not None:
            return requested
        with self._condition:
            command_name = self._active_command_name
        error = self._deadline_error(
            now=time.monotonic(),
            command_started=command_started,
            command_name=command_name,
        )
        if error is None:
            return None
        self._request_stop("FAILED", error)
        self._reap_child(force=True)
        return self._requested_outcome() or ("FAILED", error)

    def _request_stop(
        self,
        status: str,
        error: Mapping[str, Any] | None,
    ) -> None:
        safe_error = (
            _json_safe(error) if error is not None else None
        )
        with self._condition:
            if self._result is not None or self._terminalizing:
                return
            if not self._stop_requested:
                self._requested_status = status
                self._requested_error = safe_error
            self._stop_requested = True
            self._condition.notify_all()

    def _requested_outcome(self) -> tuple[str, dict[str, Any] | None] | None:
        with self._condition:
            if not self._stop_requested:
                return None
            return (
                self._requested_status or "FAILED",
                self._requested_error,
            )

    def _watchdog_loop(self) -> None:
        while not self._done.wait(_POLL_SECONDS):
            with self._condition:
                if self._state not in _ACTIVE_STATES:
                    if self._result is not None:
                        return
                    continue
                if self._stop_requested:
                    return
                now = time.monotonic()
                command_started = self._active_command_started
                command_name = self._active_command_name
            error = self._deadline_error(
                now=now,
                command_started=command_started,
                command_name=command_name,
            )
            if error is not None:
                self._request_stop("FAILED", error)
                self._reap_child(force=True)
                return

    def _command_loop(self) -> None:
        outcome: tuple[str, dict[str, Any] | None] | None = None
        connection_closed = False
        try:
            while outcome is None:
                outcome = self._requested_outcome()
                if outcome is not None:
                    break

                payload = None
                connection = self._parent_connection
                if connection is not None and not connection_closed:
                    try:
                        if connection.poll(0.02):
                            payload = connection.recv()
                    except (EOFError, BrokenPipeError, OSError):
                        connection_closed = True

                if payload is not None:
                    outcome = self._handle_command_payload(payload)
                    if outcome is not None:
                        break

                worker_result = self._take_worker_result()
                if worker_result is not None:
                    outcome = self._worker_outcome(worker_result)
                    break

                outcome = self._requested_outcome()
                if outcome is not None:
                    break

                worker_alive = self.process_is_alive
                if connection_closed and worker_alive:
                    # run_student_worker closes its command pipe before the
                    # child entry publishes the dedicated final result.
                    # Keep waiting while the watchdog still bounds runtime.
                    self._done.wait(_POLL_SECONDS)
                    continue

                if not worker_alive:
                    worker_result = self._take_worker_result(
                        timeout_s=_RESULT_GRACE_SECONDS
                    )
                    if worker_result is not None:
                        outcome = self._worker_outcome(worker_result)
                    else:
                        exitcode = self._process_exitcode()
                        outcome = (
                            "FAILED",
                            _error(
                                "STUDENT_WORKER_RESULT_MISSING",
                                "学生子进程退出但没有返回结果",
                                exitcode=exitcode,
                            ),
                        )
                    break
        except BaseException as loop_error:
            outcome = (
                "FAILED",
                _exception_error(
                    loop_error,
                    code="STUDENT_CONTROLLER_FAILED",
                ),
            )

        if outcome is None:
            outcome = (
                "FAILED",
                _error(
                    "STUDENT_CONTROLLER_FAILED",
                    "学生命令循环没有产生结果",
                ),
            )
        self._complete_run(*outcome)

    def _await_command_permission(self) -> None:
        with self._condition:
            while True:
                if self._stop_requested:
                    raise _StopCommandLoop
                if self._state is RunState.RUNNING:
                    return
                if (
                    self._state is RunState.PAUSED
                    and self._step_permits > 0
                ):
                    self._step_permits -= 1
                    return
                if self._state is not RunState.PAUSED:
                    raise _StopCommandLoop
                self._condition.wait(timeout=0.05)

    def _handle_command_payload(
        self,
        payload: Any,
    ) -> tuple[str, dict[str, Any] | None] | None:
        try:
            command = CommandMessage.from_dict(payload)
        except BaseException as protocol_error:
            return (
                "FAILED",
                _protocol_exception_error(protocol_error),
            )

        try:
            self._await_command_permission()
        except _StopCommandLoop:
            return self._requested_outcome()

        if self._elapsed() >= self._policy.max_runtime_s:
            error = _error(
                "STUDENT_RUNTIME_TIMEOUT",
                "学生程序超过总运行时间限制",
                details={"max_runtime_s": self._policy.max_runtime_s},
            )
            self._send_failure(command.command_id, error)
            return "FAILED", error

        with self._condition:
            if self._command_count >= self._policy.max_commands:
                error = _error(
                    "STUDENT_COMMAND_LIMIT_EXCEEDED",
                    "学生程序超过命令数量限制",
                    details={"max_commands": self._policy.max_commands},
                )
            else:
                error = None
                self._command_count += 1
                self._current_command = command.name
                self._active_command_name = command.name
                command_started = time.monotonic()
                self._active_command_started = command_started
        if error is not None:
            self._send_failure(command.command_id, error)
            return "FAILED", error
        self._emit_snapshot()

        try:
            evidence = self._evidence
            if evidence is None:
                raise RuntimeError("student evidence is unavailable")
            evidence.record_command(
                {
                    "command_id": command.command_id,
                    "name": command.name,
                    "args": _json_safe(command.args),
                }
            )
        except BaseException as evidence_error:
            error = _exception_error(
                evidence_error,
                code="STUDENT_EVIDENCE_FAILED",
            )
            selected = self._requested_outcome() or ("FAILED", error)
            self._send_failure(
                command.command_id,
                selected[1],
                cancelled=selected[0] == "CANCELLED",
            )
            self._clear_active_command()
            return selected

        requested = self._synchronous_deadline_outcome(
            command_started
        )
        if requested is not None:
            self._clear_active_command()
            return requested

        try:
            value = self._dispatch_bounded(
                command,
                command_started=command_started,
            )
        except _BackendActionStuck as stuck_error:
            selected = self._requested_outcome()
            if selected is None:
                error = _error(
                    "STUDENT_BACKEND_COMMAND_STUCK",
                    "底层机器人命令无法在超时后终止",
                    details={"command": stuck_error.command_name},
                )
                self._request_stop("FAILED", error)
                self._reap_child(force=True)
                selected = ("FAILED", error)
            self._clear_active_command()
            return selected
        except VisionPlatformError as command_error:
            error = _exception_error(command_error)
            with self._condition:
                self._safety_violation_count += 1
            selected = self._requested_outcome() or ("FAILED", error)
            self._send_failure(
                command.command_id,
                selected[1],
                cancelled=selected[0] == "CANCELLED",
            )
            self._clear_active_command()
            return selected
        except _StopCommandLoop:
            requested = self._requested_outcome()
            if requested is None:
                requested = (
                    "CANCELLED",
                    _error(
                        "STUDENT_PROGRAM_CANCELLED",
                        "学生程序已停止",
                    ),
                )
            self._send_failure(
                command.command_id,
                requested[1],
                cancelled=requested[0] == "CANCELLED",
            )
            self._clear_active_command()
            return requested
        except BaseException as command_error:
            if _is_transport_timeout(command_error):
                transport_error = self._quarantine_backend(
                    _error(
                        "STUDENT_BACKEND_TRANSPORT_TIMEOUT",
                        "CoppeliaSim transport timed out; "
                        "the REQ connection cannot be reused",
                        details={
                            "command": command.name,
                            "cause": _exception_error(command_error),
                        },
                    )
                )
                requested = self._requested_outcome()
                if requested is None:
                    self._request_stop("FAILED", transport_error)
                    requested = self._requested_outcome()
                self._reap_child(force=True)
                self._clear_active_command()
                return requested or ("FAILED", transport_error)
            error = _exception_error(command_error)
            selected = self._requested_outcome() or ("FAILED", error)
            self._send_failure(
                command.command_id,
                selected[1],
                cancelled=selected[0] == "CANCELLED",
            )
            self._clear_active_command()
            return selected

        requested = self._synchronous_deadline_outcome(
            command_started
        )
        self._clear_active_command()
        if requested is not None:
            return requested
        try:
            self._send_response(
                ResponseMessage(
                    command_id=command.command_id,
                    status="PASS",
                    value=_json_safe(value),
                    error=None,
                )
            )
        except BaseException as send_error:
            return (
                "FAILED",
                _exception_error(
                    send_error,
                    code="STUDENT_PROTOCOL_IO_ERROR",
                ),
            )
        return None

    def _clear_active_command(self) -> None:
        with self._condition:
            self._active_command_started = None
            self._active_command_name = None
            self._current_command = None
            self._condition.notify_all()
        self._emit_snapshot()

    def _send_failure(
        self,
        command_id: str,
        error: Mapping[str, Any] | None,
        *,
        cancelled: bool = False,
    ) -> None:
        if error is None:
            error = _error("COMMAND_FAILED", "学生命令失败")
        try:
            self._send_response(
                ResponseMessage(
                    command_id=command_id,
                    status="CANCELLED" if cancelled else "FAIL",
                    value=None,
                    error=error,
                )
            )
        except BaseException:
            pass

    def _send_response(self, response: ResponseMessage) -> None:
        connection = self._parent_connection
        if connection is None:
            raise BrokenPipeError("student command connection is closed")
        connection.send(response.to_dict())

    def _dispatch_bounded(
        self,
        command: CommandMessage,
        *,
        command_started: float,
    ) -> Any:
        completed = threading.Event()
        result: dict[str, Any] = {}

        def invoke() -> None:
            try:
                result["value"] = self._dispatch(command)
            except BaseException as action_error:
                result["error"] = action_error
            finally:
                completed.set()

        action_thread = threading.Thread(
            target=invoke,
            name=f"StudentBackendAction-{command.name}",
            daemon=True,
        )
        with self._condition:
            self._backend_action_thread = action_thread
            self._backend_action_name = command.name
        action_thread.start()

        stop_seen_at: float | None = None
        while not completed.wait(_POLL_SECONDS):
            requested = self._synchronous_deadline_outcome(
                command_started
            )
            if requested is None:
                stop_seen_at = None
                continue
            if stop_seen_at is None:
                stop_seen_at = time.monotonic()
            if (
                time.monotonic() - stop_seen_at
                >= _ACTION_STOP_GRACE_SECONDS
            ):
                with self._condition:
                    self._backend_quarantined = True
                raise _BackendActionStuck(command.name)

        action_thread.join(timeout=_POLL_SECONDS)
        if (
            "error" in result
            and _is_transport_timeout(result["error"])
        ):
            raise result["error"]
        requested = self._synchronous_deadline_outcome(command_started)
        if requested is not None:
            raise _StopCommandLoop
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _dispatch_cleanup_bounded(
        self,
        stage: str,
        action: Callable[[], Any],
    ) -> Any:
        completed = threading.Event()
        result: dict[str, Any] = {}

        def invoke() -> None:
            try:
                result["value"] = action()
            except BaseException as action_error:
                result["error"] = action_error
            finally:
                completed.set()

        action_thread = threading.Thread(
            target=invoke,
            name=f"StudentCleanupAction-{stage}",
            daemon=True,
        )
        with self._condition:
            self._backend_action_thread = action_thread
            self._backend_action_name = f"cleanup:{stage}"
        action_thread.start()

        deadline = (
            time.monotonic()
            + self._policy.command_timeout_s
            + _CLEANUP_ACTION_GRACE_SECONDS
        )
        while not completed.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if completed.is_set():
                    break
                raise _BackendActionStuck(stage)
            completed.wait(min(_POLL_SECONDS, remaining))

        action_thread.join(timeout=_POLL_SECONDS)
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _dispatch(self, command: CommandMessage) -> Any:
        dispatch: dict[str, Callable[[Mapping[str, Any]], Any]] = {
            "context.log": self._command_log,
            "context.sleep": self._command_sleep,
            "context.checkpoint": self._command_checkpoint,
            "robot.home": self._command_home,
            "robot.move_world": self._command_move_world,
            "robot.pose": self._command_pose,
            "tool.on": self._command_tool_on,
            "tool.off": self._command_tool_off,
        }
        return dispatch[command.name](command.args)

    @staticmethod
    def _require_args(
        args: Mapping[str, Any],
        expected: tuple[str, ...],
    ) -> None:
        try:
            keys = tuple(args.keys())
        except BaseException as error:
            raise VisionPlatformError(
                "STUDENT_COMMAND_ARGUMENTS_INVALID",
                "学生命令参数无法读取",
                details={"error": _safe_text(error)},
            ) from None
        if len(keys) != len(expected) or set(keys) != set(expected):
            raise VisionPlatformError(
                "STUDENT_COMMAND_ARGUMENTS_INVALID",
                "学生命令参数名称或数量不正确",
                details={
                    "expected": list(expected),
                    "received": [_safe_text(key, limit=200) for key in keys],
                },
            )

    def _command_log(self, args: Mapping[str, Any]) -> None:
        self._require_args(args, ("message",))
        message = args["message"]
        if type(message) is not str:
            raise VisionPlatformError(
                "STUDENT_COMMAND_ARGUMENTS_INVALID",
                "context.log 的 message 必须是字符串",
                details={"field": "message"},
            )
        self._guard.validate_log(message)

    def _command_checkpoint(self, args: Mapping[str, Any]) -> None:
        self._require_args(args, ("label",))
        label = args["label"]
        if type(label) is not str:
            raise VisionPlatformError(
                "STUDENT_COMMAND_ARGUMENTS_INVALID",
                "context.checkpoint 的 label 必须是字符串",
                details={"field": "label"},
            )
        self._guard.validate_log(label)

    def _command_sleep(self, args: Mapping[str, Any]) -> None:
        self._require_args(args, ("seconds",))
        seconds = self._guard.validate_sleep(args["seconds"])
        deadline = time.monotonic() + seconds
        while True:
            with self._condition:
                if self._stop_requested:
                    raise _StopCommandLoop
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._done.wait(min(0.05, remaining))

    def _command_home(self, args: Mapping[str, Any]) -> None:
        self._require_args(args, ())
        self._raise_if_stopping()
        self._application.robot.move_home()
        self._raise_if_stopping()
        self._last_pose = self._read_pose()

    def _command_move_world(self, args: Mapping[str, Any]) -> None:
        self._require_args(
            args,
            ("x_mm", "y_mm", "z_mm", "speed"),
        )
        current = self._read_pose()
        target = self._guard.validate_move(
            current,
            (args["x_mm"], args["y_mm"], args["z_mm"]),
            speed=args["speed"],
        )
        self._raise_if_stopping()
        self._application.robot.move_world(
            target[0],
            target[1],
            target[2],
            speed=float(args["speed"]),
        )
        self._last_pose = target

    def _command_pose(self, args: Mapping[str, Any]) -> list[float]:
        self._require_args(args, ())
        return list(self._read_pose())

    def _command_tool_on(self, args: Mapping[str, Any]) -> None:
        self._require_args(args, ())
        pose = self._read_pose()
        self._guard.validate_tool_on(pose)
        self._raise_if_stopping()
        self._application.tool.on()

    def _command_tool_off(self, args: Mapping[str, Any]) -> None:
        self._require_args(args, ())
        self._raise_if_stopping()
        self._application.tool.off()

    def _raise_if_stopping(self) -> None:
        with self._condition:
            if self._stop_requested:
                raise _StopCommandLoop

    def _read_pose(self) -> tuple[float, float, float]:
        try:
            raw = self._application.robot.current_world_pose()
            if isinstance(raw, (str, bytes, bytearray)):
                raise TypeError
            values = tuple(raw)
            if len(values) != 3:
                raise ValueError
            if any(type(value) not in {int, float} for value in values):
                raise TypeError
            pose = tuple(float(value) for value in values)
            if not all(isfinite(value) for value in pose):
                raise ValueError
        except BaseException as pose_error:
            raise VisionPlatformError(
                "ROBOT_POSE_INVALID",
                "机器人返回的位姿必须包含三个有限数值",
                details={
                    "type": _safe_type_name(
                        raw if "raw" in locals() else pose_error
                    )
                },
            ) from None
        result = (pose[0], pose[1], pose[2])
        self._last_pose = result
        return result

    def _take_worker_result(
        self,
        *,
        timeout_s: float = 0.0,
    ) -> Any | None:
        result_queue = self._result_queue
        if result_queue is None:
            return None
        try:
            if timeout_s > 0:
                return result_queue.get(timeout=timeout_s)
            return result_queue.get_nowait()
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            return None

    def _worker_outcome(
        self,
        payload: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        if not isinstance(payload, Mapping):
            return (
                "FAILED",
                _error(
                    "STUDENT_WORKER_RESULT_INVALID",
                    "学生子进程返回了无效结果",
                ),
            )
        status = payload.get("status")
        worker_error = payload.get("error")
        if status == "PASS" and worker_error is None:
            return "PASS", None
        if status == "FAIL" and isinstance(worker_error, Mapping):
            safe_worker_error = _json_safe(worker_error)
            if "code" not in safe_worker_error:
                safe_worker_error["code"] = "STUDENT_PROGRAM_FAILED"
            if "message" not in safe_worker_error:
                safe_worker_error["message"] = "学生程序执行失败"
            return "FAILED", safe_worker_error
        if status == "CANCELLED":
            return (
                "CANCELLED",
                _error(
                    "STUDENT_PROGRAM_CANCELLED",
                    "学生程序通信已取消",
                ),
            )
        return (
            "FAILED",
            _error(
                "STUDENT_WORKER_RESULT_INVALID",
                "学生子进程返回了无效结果",
                details={"result": payload},
            ),
        )

    def _process_exitcode(self) -> int | None:
        with self._process_lock:
            process = self._process
            if process is None:
                return None
            try:
                process.join(timeout=0.05)
                return process.exitcode
            except (AssertionError, ValueError):
                return None

    def _reap_child(self, *, force: bool) -> bool:
        with self._process_lock:
            process = self._process
            if process is None:
                return True
            try:
                alive = process.is_alive()
            except (AssertionError, ValueError):
                return True
            if alive:
                try:
                    process.join(timeout=_CHILD_EXIT_GRACE_SECONDS)
                except (AssertionError, ValueError):
                    return True
            try:
                alive = process.is_alive()
            except (AssertionError, ValueError):
                return False
            if alive and force:
                try:
                    process.terminate()
                except (AttributeError, OSError, ValueError):
                    pass
                try:
                    process.join(timeout=_CHILD_JOIN_SECONDS)
                except (AssertionError, ValueError):
                    return True
                try:
                    alive = process.is_alive()
                except (AssertionError, ValueError):
                    return False
            if alive and force:
                kill = getattr(process, "kill", None)
                if callable(kill):
                    try:
                        kill()
                    except (OSError, ValueError):
                        pass
                    try:
                        process.join(timeout=_CHILD_JOIN_SECONDS)
                    except (AssertionError, ValueError):
                        pass
                try:
                    alive = process.is_alive()
                except (AssertionError, ValueError):
                    alive = False
            return not alive

    def _complete_run(
        self,
        status: str,
        error: Mapping[str, Any] | None,
    ) -> None:
        if not self._completion_lock.acquire(blocking=False):
            return
        try:
            if status == "PASS":
                deadline_outcome = self._synchronous_deadline_outcome(
                    None
                )
                if deadline_outcome is not None:
                    status, error = deadline_outcome
            with self._condition:
                if self._result is not None:
                    return
                if self._stop_requested:
                    status = self._requested_status or "FAILED"
                    error = self._requested_error
                self._terminalizing = True
                self._stop_requested = True
                self._condition.notify_all()

            # A worker may return a final result while leaving a non-daemon
            # student thread behind. Once the result is received, no more
            # commands are valid, so bounded force-reaping is safe for every
            # terminal status.
            child_stopped = self._reap_child(force=True)
            final_status = status
            final_error = (
                _json_safe(error) if error is not None else None
            )
            if not child_stopped:
                final_status = "FAILED"
                final_error = _error(
                    "STUDENT_WORKER_REAP_FAILED",
                    "无法安全回收学生子进程",
                    details={"original_error": final_error},
                )

            cleanup_errors: list[dict[str, Any]] = []
            with self._condition:
                action_thread = self._backend_action_thread
                action_name = self._backend_action_name
                backend_quarantined = self._backend_quarantined
                quarantine_error = self._backend_quarantine_error
            action_stuck = bool(
                action_thread is not None and action_thread.is_alive()
            )
            if action_stuck:
                stuck_error = _error_with_quarantine_details(
                    _error(
                        "STUDENT_BACKEND_COMMAND_STUCK",
                        "底层机器人命令超时后仍未返回，已进入隔离状态",
                        details={"command": action_name},
                    )
                )
                with self._condition:
                    self._backend_quarantined = True
                    self._backend_quarantine_error = stuck_error
                    self._last_pose = None
                backend_quarantined = True
                quarantine_error = stuck_error
                stuck_cleanup_error = {
                    "stage": "backend.command",
                    "error": stuck_error,
                }
                cleanup_errors.append(stuck_cleanup_error)
                final_status = "FAILED"
                if final_error is None:
                    final_error = stuck_error
                else:
                    final_error = _error_with_quarantine_details(
                        final_error,
                        reason=stuck_error,
                    )
                final_error = _error_with_cleanup_details(
                    final_error,
                    cleanup_errors,
                    quarantined=True,
                )
            elif backend_quarantined:
                if quarantine_error is None:
                    quarantine_error = _error_with_quarantine_details(
                        _error(
                            "STUDENT_BACKEND_CONNECTION_QUARANTINED",
                            "CoppeliaSim backend connection is unusable",
                        )
                    )
                cleanup_errors.append(
                    {
                        "stage": "backend.connection",
                        "error": quarantine_error,
                    }
                )
                final_status = "FAILED"
                if final_error is None:
                    final_error = quarantine_error
                else:
                    final_error = _error_with_quarantine_details(
                        final_error,
                        reason=quarantine_error,
                    )
                final_error = _error_with_cleanup_details(
                    final_error,
                    cleanup_errors,
                    quarantined=True,
                )
            elif final_status != "PASS" and child_stopped:
                cleanup_errors = self._cleanup()
            with self._condition:
                post_cleanup_quarantined = self._backend_quarantined
                post_cleanup_error = self._backend_quarantine_error
            if post_cleanup_quarantined and not backend_quarantined:
                backend_quarantined = True
                quarantine_error = post_cleanup_error
                if quarantine_error is None:
                    quarantine_error = _error_with_quarantine_details(
                        _error(
                            "STUDENT_BACKEND_CONNECTION_QUARANTINED",
                            "CoppeliaSim backend connection is unusable",
                        )
                    )
                if not any(
                    item.get("stage") == "backend.connection"
                    for item in cleanup_errors
                ):
                    cleanup_errors.append(
                        {
                            "stage": "backend.connection",
                            "error": quarantine_error,
                        }
                    )
                final_status = "FAILED"
                if quarantine_error.get("code") == (
                    "STUDENT_BACKEND_COMMAND_STUCK"
                ):
                    stuck_error = dict(_json_safe(quarantine_error))
                    stuck_details = dict(
                        stuck_error.get("details")
                        if isinstance(
                            stuck_error.get("details"),
                            Mapping,
                        )
                        else {}
                    )
                    if final_error is not None:
                        stuck_details["original_error"] = _json_safe(
                            final_error
                        )
                    stuck_error["details"] = stuck_details
                    final_error = stuck_error
                elif final_error is None:
                    final_error = quarantine_error
                else:
                    final_error = _error_with_quarantine_details(
                        final_error,
                        reason=quarantine_error,
                    )
                final_error = _error_with_cleanup_details(
                    final_error,
                    cleanup_errors,
                    quarantined=True,
                )
            timeout_restore_error = self._restore_client_timeout()
            if timeout_restore_error is not None:
                restore_quarantine_error = (
                    self._record_timeout_restore_failure(
                        cleanup_errors,
                        timeout_restore_error,
                    )
                )
                backend_quarantined = True
                if final_status == "PASS":
                    final_status = "FAILED"
                    final_error = restore_quarantine_error
                elif final_error is None:
                    final_error = restore_quarantine_error
                else:
                    final_error = _error_with_quarantine_details(
                        final_error,
                        reason=restore_quarantine_error,
                    )
                final_error = _error_with_cleanup_details(
                    final_error,
                    cleanup_errors,
                    quarantined=True,
                )
            final_error, cleanup_errors = _seal_terminal_payload(
                final_error,
                cleanup_errors,
            )
            self._cleanup_errors = cleanup_errors

            summary_path: Path | None = None
            evidence = self._evidence
            if evidence is not None:
                try:
                    summary_path = evidence.finalize(
                        status=final_status,
                        command_count=self._command_count,
                        last_pose_mm=(
                            None
                            if backend_quarantined
                            else self._last_pose
                        ),
                        safety_violation_count=(
                            self._safety_violation_count
                        ),
                        error=final_error,
                        cleanup_errors=cleanup_errors,
                    )
                except BaseException as finalize_error:
                    prior_status = final_status
                    prior_error = final_error
                    final_status = "FAILED"
                    summary_path = None
                    if prior_status == "PASS" and child_stopped:
                        try:
                            self._bound_client_timeout()
                        except BaseException as rebind_error:
                            quarantine_error = self._quarantine_backend(
                                _exception_error(
                                    rebind_error,
                                    code=(
                                        "STUDENT_TRANSPORT_TIMEOUT_"
                                        "GUARD_FAILED"
                                    ),
                                )
                            )
                            cleanup_errors.extend(
                                (
                                    {
                                        "stage": (
                                            "client.transport_timeout.rebind"
                                        ),
                                        "error": quarantine_error,
                                    },
                                    {
                                        "stage": "backend.connection",
                                        "error": quarantine_error,
                                    },
                                )
                            )
                        else:
                            cleanup_errors.extend(self._cleanup())
                        finally:
                            final_restore_error = (
                                self._restore_client_timeout()
                            )
                            if final_restore_error is not None:
                                self._record_timeout_restore_failure(
                                    cleanup_errors,
                                    final_restore_error,
                                )
                        self._cleanup_errors = cleanup_errors
                    final_error = _error(
                        "STUDENT_EVIDENCE_FAILED",
                        _safe_text(finalize_error),
                        details={
                            "stage": "finalize",
                            "original_status": prior_status,
                            "original_error": prior_error,
                            "cleanup_errors": cleanup_errors,
                        },
                        type=_safe_type_name(finalize_error),
                    )
                    with self._condition:
                        finalize_quarantined = self._backend_quarantined
                        finalize_quarantine_error = (
                            self._backend_quarantine_error
                        )
                    if finalize_quarantined:
                        backend_quarantined = True
                        if finalize_quarantine_error is None:
                            finalize_quarantine_error = (
                                _error_with_quarantine_details(
                                    _error(
                                        "STUDENT_BACKEND_CONNECTION_QUARANTINED",
                                        "CoppeliaSim backend connection "
                                        "is unusable",
                                    )
                                )
                            )
                        if not any(
                            item.get("stage") == "backend.connection"
                            for item in cleanup_errors
                        ):
                            cleanup_errors.append(
                                {
                                    "stage": "backend.connection",
                                    "error": finalize_quarantine_error,
                                }
                            )
                        final_error = _error_with_quarantine_details(
                            final_error,
                            reason=finalize_quarantine_error,
                        )
                        final_error = _error_with_cleanup_details(
                            final_error,
                            cleanup_errors,
                            quarantined=True,
                        )
                        self._cleanup_errors = cleanup_errors

            final_error, cleanup_errors = _seal_terminal_payload(
                final_error,
                cleanup_errors,
            )
            self._cleanup_errors = cleanup_errors
            self._close_ipc()
            terminal_state = {
                "PASS": RunState.PASSED,
                "FAILED": RunState.FAILED,
                "CANCELLED": RunState.CANCELLED,
            }.get(final_status, RunState.FAILED)
            if terminal_state is RunState.FAILED and final_status != "FAILED":
                final_status = "FAILED"
            result = StudentRunResult(
                status=final_status,
                summary_path=summary_path,
                evidence_dir=(
                    evidence.directory if evidence is not None else None
                ),
                error=final_error,
            )
            with self._condition:
                self._state = terminal_state
                self._error = result.error
                self._result = result
                self._current_command = None
                self._active_command_name = None
                self._active_command_started = None
                self._terminalizing = False
                self._done.set()
                self._condition.notify_all()
            self._emit_snapshot()
        finally:
            self._completion_lock.release()

    def _cleanup(self) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []

        def record_quarantine(
            stage: str,
            quarantine_error: Mapping[str, Any],
        ) -> None:
            errors.extend(
                (
                    {
                        "stage": stage,
                        "error": _json_safe(quarantine_error),
                    },
                    {
                        "stage": "backend.connection",
                        "error": _json_safe(quarantine_error),
                    },
                )
            )

        def record_error(
            stage: str,
            cleanup_error: BaseException,
        ) -> bool:
            if _is_transport_timeout(cleanup_error):
                quarantine_error = self._quarantine_backend(
                    _error(
                        "STUDENT_BACKEND_TRANSPORT_TIMEOUT",
                        "CoppeliaSim transport timed out during cleanup; "
                        "the REQ connection cannot be reused",
                        details={
                            "stage": stage,
                            "cause": _exception_error(cleanup_error),
                        },
                    )
                )
                record_quarantine(stage, quarantine_error)
                return True
            errors.append(
                {
                    "stage": stage,
                    "error": _exception_error(
                        cleanup_error,
                        code="STUDENT_CLEANUP_FAILED",
                    ),
                }
            )
            return False

        def capture(
            stage: str,
            action: Callable[[], Any],
        ) -> tuple[bool, bool, Any]:
            try:
                value = self._dispatch_cleanup_bounded(stage, action)
            except _BackendActionStuck:
                quarantine_error = self._quarantine_backend(
                    _error(
                        "STUDENT_BACKEND_COMMAND_STUCK",
                        "Cleanup backend action exceeded its bounded "
                        "deadline and remains active",
                        details={
                            "stage": stage,
                            "command_timeout_s": (
                                self._policy.command_timeout_s
                            ),
                            "cleanup_grace_s": (
                                _CLEANUP_ACTION_GRACE_SECONDS
                            ),
                        },
                    )
                )
                record_quarantine(stage, quarantine_error)
                return True, False, None
            except BaseException as cleanup_error:
                return (
                    record_error(stage, cleanup_error),
                    False,
                    None,
                )
            return False, True, value

        stop, _, _ = capture("tool.off", self._application.tool.off)
        if stop:
            return errors

        stop, pose_succeeded, pose_value = capture(
            "robot.current_world_pose",
            self._read_pose,
        )
        if stop:
            return errors
        pose: tuple[float, float, float] | None = (
            pose_value if pose_succeeded else None
        )

        if (
            pose is not None
            and pose[2] < self._application.workspace.safe_z_mm
        ):
            safe_target = (
                pose[0],
                pose[1],
                float(self._application.workspace.safe_z_mm),
            )

            try:
                target = self._guard.validate_move(
                    pose,
                    safe_target,
                    speed=self._policy.min_speed,
                )
            except BaseException as guard_error:
                if record_error("robot.safe_lift", guard_error):
                    return errors
            else:
                def safe_lift() -> None:
                    self._application.robot.move_world(
                        target[0],
                        target[1],
                        target[2],
                        speed=self._policy.min_speed,
                    )

                stop, lift_succeeded, _ = capture(
                    "robot.safe_lift",
                    safe_lift,
                )
                if stop:
                    return errors
                if lift_succeeded:
                    self._last_pose = target

        capture("robot.move_home", self._application.robot.move_home)
        return errors

    def _close_ipc(self) -> None:
        with self._process_lock:
            connection = self._parent_connection
            result_queue = self._result_queue
            process = self._process
            self._parent_connection = None
            self._result_queue = None
        if connection is not None:
            try:
                connection.close()
            except BaseException:
                pass
        if result_queue is not None:
            try:
                result_queue.close()
            except BaseException:
                pass
            try:
                result_queue.join_thread()
            except BaseException:
                pass
        if process is not None:
            with self._process_lock:
                try:
                    if (
                        self._process is process
                        and not process.is_alive()
                    ):
                        process.close()
                        self._process = None
                except (
                    AttributeError,
                    AssertionError,
                    OSError,
                    ValueError,
                ):
                    pass
