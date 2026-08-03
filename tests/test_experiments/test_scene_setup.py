from __future__ import annotations

import pytest

from vision_platform.experiments.scene_setup import activate_scene_group


class FakeSim:
    handle_world = -1

    def __init__(self) -> None:
        self.handles = {
            "/LogisticsLab/Tasks/Stack": 1,
            "/LogisticsLab/Tasks/Digits": 2,
            "/LogisticsLab/Tasks/Classes": 3,
        }
        self.requested_paths: list[str] = []
        self.positions: list[tuple[int, list[float], int]] = []

    def getObject(self, path: str) -> int:
        self.requested_paths.append(path)
        return self.handles[path]

    def setObjectPosition(
        self,
        handle: int,
        position: list[float],
        relative_to: int,
    ) -> None:
        self.positions.append((handle, list(position), relative_to))


def test_activate_scene_group_moves_only_selected_group_into_workspace() -> None:
    sim = FakeSim()

    activate_scene_group(
        sim,
        active_path="/LogisticsLab/Tasks/Digits",
    )

    assert sim.positions == [
        (1, [0.0, 0.0, -2.0], -1),
        (2, [0.0, 0.0, 0.0], -1),
        (3, [0.0, 0.0, -4.0], -1),
    ]


def test_scene_without_group_needs_no_activation() -> None:
    sim = FakeSim()

    activate_scene_group(sim, active_path=None)

    assert sim.requested_paths == []
    assert sim.positions == []


def test_code_routing_scene_group_is_independently_reset_without_logistics_mutation() -> None:
    sim = FakeSim()

    activate_scene_group(sim, active_path="/VisionCodeRoutingLab")

    assert sim.requested_paths == []
    assert sim.positions == []


def test_ocr_sorting_scene_group_is_independently_reset_without_logistics_mutation() -> None:
    sim = FakeSim()

    activate_scene_group(sim, active_path="/VisionOcrSortingLab")

    assert sim.requested_paths == []
    assert sim.positions == []


def test_unknown_scene_group_fails_closed_before_mutating_scene() -> None:
    sim = FakeSim()

    with pytest.raises(ValueError, match="Unknown logistics task group"):
        activate_scene_group(
            sim,
            active_path="/LogisticsLab/Tasks/Unknown",
        )

    assert sim.requested_paths == []
    assert sim.positions == []


def test_missing_group_handle_does_not_partially_move_other_groups() -> None:
    sim = FakeSim()
    del sim.handles["/LogisticsLab/Tasks/Digits"]

    with pytest.raises(KeyError):
        activate_scene_group(
            sim,
            active_path="/LogisticsLab/Tasks/Stack",
        )

    assert sim.requested_paths == [
        "/LogisticsLab/Tasks/Stack",
        "/LogisticsLab/Tasks/Digits",
    ]
    assert sim.positions == []
