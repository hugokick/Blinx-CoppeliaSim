from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "CON",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
    "NUL",
    "PRN",
}
_WINDOWS_INVALID_NAME_CHARACTERS = frozenset('<>:"/\\|?*')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: Any, *, indent: int | None = None) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
    )
    return (text + "\n").encode("utf-8")


class _EvidenceCleanupError(Exception):
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = tuple(errors)
        super().__init__(
            f"{len(self.errors)} student evidence cleanup errors"
        )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(
        f".{path.stem}-{uuid.uuid4().hex}.tmp"
    )
    stream = None
    try:
        stream = temporary.open("xb", buffering=0)
        written = stream.write(payload)
        if written != len(payload):
            raise OSError(
                f"incomplete evidence write: {written}/{len(payload)}"
            )
        stream.flush()
        stream.close()
        temporary.replace(path)
    except Exception as original_error:
        cleanup_errors: list[Exception] = []
        if stream is not None:
            try:
                if not stream.closed:
                    stream.close()
            except Exception as close_error:
                cleanup_errors.append(close_error)
        try:
            temporary.unlink(missing_ok=True)
        except Exception as unlink_error:
            cleanup_errors.append(unlink_error)

        if cleanup_errors:
            cause: Exception
            if len(cleanup_errors) == 1:
                cause = cleanup_errors[0]
            else:
                cause = _EvidenceCleanupError(cleanup_errors)
            raise original_error from cause
        raise


def _validate_run_id(run_id: str) -> str:
    if type(run_id) is not str or not run_id or not run_id.strip():
        raise ValueError("run_id must be a non-empty directory name")
    if run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one safe directory name")
    candidate = Path(run_id)
    if candidate.is_absolute() or candidate.drive or candidate.anchor:
        raise ValueError("run_id must not be an absolute or drive path")
    if (
        any(
            character in _WINDOWS_INVALID_NAME_CHARACTERS
            or ord(character) < 32
            for character in run_id
        )
        or run_id.endswith((" ", "."))
    ):
        raise ValueError("run_id contains unsafe filename characters")
    if run_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("run_id uses a reserved filename")
    return run_id


def _safe_program_stem(stem: str) -> str:
    safe = "".join(
        "_"
        if character in _WINDOWS_INVALID_NAME_CHARACTERS
        or ord(character) < 32
        else character
        for character in stem
    ).strip(" .")
    safe = safe[:80].rstrip(" .")
    if (
        not safe
        or safe in {".", ".."}
        or safe.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        return "program"
    return safe


@dataclass
class StudentRunEvidence:
    directory: Path
    program_path: Path
    source_sha256: str
    robot_backend: str
    run_id: str
    started_at: str
    started_monotonic: float
    _finalized: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _lock: Any = field(
        default_factory=RLock,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def source_path(self) -> Path:
        return self.directory / "source.py"

    @classmethod
    def create(
        cls,
        *,
        output_root: str | Path,
        program_path: str | Path,
        robot_backend: str,
        run_id: str | None = None,
    ) -> StudentRunEvidence:
        selected = Path(program_path).expanduser().resolve()
        source = selected.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if run_id is None:
            selected_run_id = (
                datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                + "-"
                + _safe_program_stem(selected.stem)
                + "-"
                + uuid.uuid4().hex[:8]
            )
        else:
            selected_run_id = _validate_run_id(run_id)

        directory = (
            Path(output_root).expanduser().resolve() / selected_run_id
        )
        started_at = _now()
        started_monotonic = time.monotonic()
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "source.py").write_bytes(source)
        (directory / "source.sha256").write_bytes(
            (digest + "\n").encode("utf-8")
        )
        manifest = {
            "schema_version": 1,
            "run_id": selected_run_id,
            "program_path": str(selected),
            "source_sha256": digest,
            "robot_backend": robot_backend,
            "hardware_status": "PENDING_HARDWARE",
        }
        (directory / "manifest.json").write_bytes(
            _json_bytes(manifest, indent=2)
        )
        (directory / "commands.jsonl").write_bytes(b"")
        (directory / "events.jsonl").write_bytes(b"")
        return cls(
            directory=directory,
            program_path=selected,
            source_sha256=digest,
            robot_backend=robot_backend,
            run_id=selected_run_id,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("student run evidence is already finalized")

    def _append(self, name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._ensure_open()
            serialized = _json_bytes(payload)
            path = self.directory / name
            existing = path.read_bytes()
            _atomic_write_bytes(path, existing + serialized)

    def record_command(self, payload: dict[str, Any]) -> None:
        entry = dict(payload)
        entry["timestamp"] = _now()
        self._append("commands.jsonl", entry)

    def record_event(
        self,
        state: str,
        message: str,
        **details: Any,
    ) -> None:
        self._append(
            "events.jsonl",
            {
                "timestamp": _now(),
                "state": state,
                "message": message,
                "details": details,
            },
        )

    def finalize(
        self,
        *,
        status: str,
        command_count: int,
        last_pose_mm,
        safety_violation_count: int,
        error,
        cleanup_errors,
    ) -> Path:
        with self._lock:
            self._ensure_open()
            path = self.directory / "summary.json"
            payload = {
                "schema_version": 1,
                "run_id": self.run_id,
                "status": status,
                "program_path": str(self.program_path),
                "source_sha256": self.source_sha256,
                "started_at": self.started_at,
                "finished_at": _now(),
                "elapsed_seconds": round(
                    time.monotonic() - self.started_monotonic,
                    3,
                ),
                "command_count": int(command_count),
                "last_pose_mm": (
                    list(last_pose_mm)
                    if last_pose_mm is not None
                    else None
                ),
                "safety_violation_count": int(safety_violation_count),
                "error": error,
                "cleanup_errors": list(cleanup_errors),
                "robot_backend": self.robot_backend,
                "hardware_status": "PENDING_HARDWARE",
            }
            serialized = _json_bytes(payload, indent=2)
            _atomic_write_bytes(path, serialized)
            self._finalized = True
            return path
