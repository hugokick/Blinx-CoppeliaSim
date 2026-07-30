from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from vision_platform.cameras.base import CameraBackend
from vision_platform.errors import CameraUnavailableError, FrameTimeoutError
from vision_platform.image_io import read_bgr
from vision_platform.models import Frame


@dataclass(frozen=True)
class _ReplayEntry:
    path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ReplayCamera(CameraBackend):
    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        loop: bool = True,
        metadata: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        path_list = [Path(path).expanduser().resolve() for path in paths]
        metadata_list = list(metadata or [{} for _ in path_list])
        if len(metadata_list) != len(path_list):
            raise ValueError("metadata count must match replay path count")
        self._entries = [
            _ReplayEntry(path=path, metadata=dict(item))
            for path, item in zip(path_list, metadata_list)
        ]
        self._loop = bool(loop)
        self._is_open = False
        self._index = 0
        self._sequence_id = 0

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        loop: bool | None = None,
    ) -> "ReplayCamera":
        manifest = Path(manifest_path).expanduser().resolve()
        if not manifest.is_file():
            raise CameraUnavailableError(
                f"Replay manifest does not exist: {manifest}",
                manifest=str(manifest),
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        entries = payload.get("frames", [])
        paths: list[Path] = []
        metadata: list[Mapping[str, Any]] = []
        for entry in entries:
            if isinstance(entry, str):
                relative = entry
                item_metadata: dict[str, Any] = {}
            elif isinstance(entry, dict) and "path" in entry:
                relative = str(entry["path"])
                item_metadata = {
                    key: value for key, value in entry.items() if key != "path"
                }
            else:
                raise ValueError("Each replay frame must be a path or object with path")
            path = Path(relative).expanduser()
            if not path.is_absolute():
                path = manifest.parent / path
            paths.append(path.resolve())
            metadata.append(item_metadata)
        selected_loop = payload.get("loop", True) if loop is None else loop
        return cls(paths, loop=selected_loop, metadata=metadata)

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        if not self._entries:
            raise CameraUnavailableError("Replay camera has no configured frames")
        self._index = 0
        self._is_open = True

    def read(self, timeout_s: float = 1.0) -> Frame:
        del timeout_s
        if not self._is_open:
            self.open()
        if self._index >= len(self._entries):
            if not self._loop:
                raise FrameTimeoutError("Replay sequence is exhausted")
            self._index = 0

        entry = self._entries[self._index]
        image = read_bgr(entry.path)
        if image is None:
            raise CameraUnavailableError(
                f"Replay frame could not be read: {entry.path}",
                path=str(entry.path),
            )

        height, width = image.shape[:2]
        metadata = dict(entry.metadata)
        metadata["path"] = str(entry.path)
        frame = Frame(
            image_bgr=image,
            width=width,
            height=height,
            timestamp_s=time.time(),
            source="replay",
            sequence_id=self._sequence_id,
            metadata=metadata,
        )
        self._index += 1
        self._sequence_id += 1
        return frame

    def close(self) -> None:
        self._is_open = False
