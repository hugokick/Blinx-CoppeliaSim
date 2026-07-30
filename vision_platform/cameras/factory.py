from __future__ import annotations

from typing import Any, Mapping

from vision_platform.cameras.base import CameraBackend


def create_camera(
    backend: str,
    options: Mapping[str, Any] | None = None,
    **dependencies: Any,
) -> CameraBackend:
    name = backend.strip().lower()
    selected = dict(options or {})
    if name == "replay":
        from vision_platform.cameras.replay import ReplayCamera

        if selected.get("manifest"):
            return ReplayCamera.from_manifest(
                selected["manifest"],
                loop=selected.get("loop"),
            )
        return ReplayCamera(
            selected.get("paths", []),
            loop=selected.get("loop", True),
        )
    if name == "sim":
        from vision_platform.cameras.coppeliasim import CoppeliaSimCamera

        return CoppeliaSimCamera(
            sim=dependencies.get("sim"),
            client=dependencies.get("client"),
            **selected,
        )
    if name == "hik":
        from vision_platform.cameras.hikvision import HikvisionCamera

        return HikvisionCamera(**selected)
    raise ValueError(f"Unsupported camera backend: {backend}")
