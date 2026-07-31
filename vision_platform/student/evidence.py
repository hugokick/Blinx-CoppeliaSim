from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping


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
_RUN_METADATA_RESERVED_FIELDS = frozenset(
    {
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
)
_SNAPSHOT_RESERVED_FIELDS = frozenset({"snapshot_id", "path", "sha256"})
_SNAPSHOT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


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


def _canonicalize_json(
    value: Any,
    *,
    path: str = "$",
    active_containers: set[int] | None = None,
) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return value

    active = active_containers if active_containers is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} must not contain cycles")
        active.add(identity)
        try:
            canonical: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} keys must be strings")
                canonical[key] = _canonicalize_json(
                    item,
                    path=f"{path}.{key}",
                    active_containers=active,
                )
            return canonical
        finally:
            active.remove(identity)

    if type(value) in {list, tuple}:
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} must not contain cycles")
        active.add(identity)
        try:
            return [
                _canonicalize_json(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)

    raise TypeError(
        f"{path} must contain only JSON-native values, not "
        f"{type(value).__name__}"
    )


def _canonicalize_metadata(
    metadata: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name} must be a mapping")
    canonical = _canonicalize_json(metadata, path=name)
    assert isinstance(canonical, dict)
    return canonical


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
    run_metadata: dict[str, Any]
    _run_metadata_snapshot: dict[str, Any] = field(
        repr=False,
        compare=False,
    )
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
        run_metadata: Mapping[str, Any] | None = None,
    ) -> StudentRunEvidence:
        metadata = _canonicalize_metadata(
            {} if run_metadata is None else run_metadata,
            name="run_metadata",
        )
        if (
            metadata.get("hardware_status", "PENDING_HARDWARE")
            != "PENDING_HARDWARE"
        ):
            raise ValueError("hardware_status must remain PENDING_HARDWARE")
        metadata["hardware_status"] = "PENDING_HARDWARE"
        public_metadata = _canonicalize_metadata(
            metadata,
            name="run_metadata",
        )
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
            **{
                key: value
                for key, value in metadata.items()
                if key not in _RUN_METADATA_RESERVED_FIELDS
            },
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
            run_metadata=public_metadata,
            _run_metadata_snapshot=metadata,
        )

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("student run evidence is already finalized")

    def _append(self, name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._ensure_open()
            serialized = _json_bytes(payload)
            path = self.directory / name
            existing = path.read_bytes() if path.exists() else b""
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

    def record_snapshot(
        self,
        *,
        snapshot_id: str,
        png_bytes: bytes,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            type(snapshot_id) is not str
            or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None
            or snapshot_id.upper() in _WINDOWS_RESERVED_NAMES
        ):
            raise ValueError(
                "snapshot_id must be a portable ASCII name using 1-80 "
                "letters, digits, '-' or '_'"
            )
        if type(png_bytes) is not bytes or not png_bytes:
            raise ValueError("png_bytes must not be empty")
        relative = Path("frames") / f"{snapshot_id}.png"
        target = self.directory / relative
        canonical_metadata = _canonicalize_metadata(
            metadata,
            name="snapshot metadata",
        )
        record = {
            **{
                key: value
                for key, value in canonical_metadata.items()
                if key not in _SNAPSHOT_RESERVED_FIELDS
            },
            "snapshot_id": snapshot_id,
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(png_bytes).hexdigest(),
        }
        _json_bytes(record)
        with self._lock:
            self._ensure_open()
            frames = self.directory / "frames"
            frames.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(
                    f"snapshot_id already exists: {snapshot_id}"
                )
            _atomic_write_bytes(target, png_bytes)
            try:
                self._append("snapshots.jsonl", record)
            except Exception as original_error:
                try:
                    target.unlink(missing_ok=True)
                except Exception as cleanup_error:
                    raise original_error from cleanup_error
                raise
        return record

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
                **{
                    key: value
                    for key, value in self._run_metadata_snapshot.items()
                    if key not in _RUN_METADATA_RESERVED_FIELDS
                },
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
