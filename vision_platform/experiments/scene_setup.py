from __future__ import annotations

from typing import Any


LOGISTICS_GROUPS = (
    "/LogisticsLab/Tasks/Stack",
    "/LogisticsLab/Tasks/Digits",
    "/LogisticsLab/Tasks/Classes",
)

_PARKED_Z = {
    "/LogisticsLab/Tasks/Stack": -2.0,
    "/LogisticsLab/Tasks/Digits": -3.0,
    "/LogisticsLab/Tasks/Classes": -4.0,
}


def validate_scene_group_path(active_path: str | None) -> str | None:
    if active_path is None:
        return None
    if active_path not in LOGISTICS_GROUPS:
        raise ValueError(f"Unknown logistics task group: {active_path}")
    return active_path


def activate_scene_group(
    sim: Any,
    *,
    active_path: str | None,
) -> None:
    active_path = validate_scene_group_path(active_path)
    if active_path is None:
        return
    handles = tuple(
        (path, int(sim.getObject(path))) for path in LOGISTICS_GROUPS
    )
    for path, handle in handles:
        position = (
            [0.0, 0.0, 0.0]
            if path == active_path
            else [0.0, 0.0, _PARKED_Z[path]]
        )
        sim.setObjectPosition(handle, position, sim.handle_world)
