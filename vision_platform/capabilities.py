from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)


def probe_coppeliasim(root: str | Path) -> Capability:
    root_path = Path(root).expanduser().resolve()
    executable = root_path / "coppeliaSim.exe"
    zmq_source = (
        root_path
        / "programming"
        / "zmqRemoteApi"
        / "clients"
        / "python"
        / "src"
    )
    if not executable.is_file():
        return Capability(
            "coppeliasim",
            False,
            f"coppeliaSim.exe was not found under {root_path}",
        )
    if not zmq_source.is_dir():
        return Capability(
            "coppeliasim",
            False,
            f"CoppeliaSim ZMQ Python client was not found under {root_path}",
        )
    return Capability(
        "coppeliasim",
        True,
        "CoppeliaSim executable and ZMQ client are available",
        {
            "root": str(root_path),
            "executable": str(executable),
            "zmq_source": str(zmq_source),
        },
    )


def probe_hikvision(search_roots: Iterable[str | Path]) -> Capability:
    runtime_patterns = (
        Path("Development/MVS/Runtime/Win64_x64"),
        Path("MVS/Runtime/Win64_x64"),
        Path("Runtime/Win64_x64"),
        Path("opt/MVS/lib/64"),
    )
    library_names = ("MvCameraControl.dll", "libMvCameraControl.so")
    searched: list[str] = []
    for root in search_roots:
        root_path = Path(root).expanduser().resolve()
        for relative in runtime_patterns:
            runtime = root_path / relative
            searched.append(str(runtime))
            if any((runtime / name).is_file() for name in library_names):
                return Capability(
                    "hikvision",
                    True,
                    "Hikvision MVS runtime is available",
                    {"runtime": str(runtime.resolve())},
                )
    return Capability(
        "hikvision",
        False,
        "Hikvision MVS runtime was not found; set HIK_MVS_ROOT on a hardware machine",
        {"searched": searched},
    )


def probe_replay(manifest: str | Path) -> Capability:
    path = Path(manifest).expanduser().resolve()
    if not path.is_file():
        return Capability(
            "replay",
            False,
            f"Replay manifest was not found: {path}",
        )
    return Capability(
        "replay",
        True,
        "Replay manifest is available",
        {"manifest": str(path)},
    )
