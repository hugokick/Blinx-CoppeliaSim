from __future__ import annotations

import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.code_routing import ApprovedCodeRoute, CodeRoutePlan
from vision_platform.student.code_route_guard import CodeRouteGuard


@pytest.fixture
def plan() -> CodeRoutePlan:
    entries = tuple(
        ApprovedCodeRoute(
            entry_id=f"entry_{index}",
            part_id=f"part_{index}",
            code_type="qr" if index < 2 else "ean13",
            payload=f"payload-{index}",
            route_id="route_red" if index % 2 == 0 else "route_blue",
            pick_xyz_mm=(45.0 + 20.0 * index, -45.0 + 15.0 * index, 18.0),
            drop_xyz_mm=(116.0 + 4.0 * index, -60.0 if index % 2 == 0 else 60.0, 22.0),
            confidence=0.98,
        )
        for index in range(4)
    )
    return CodeRoutePlan("a" * 64, entries, 110.0, 15.0)


def _execute_one(guard: CodeRouteGuard, entry) -> None:
    pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], 111.0)
    drop_hover = (entry.drop_xyz_mm[0], entry.drop_xyz_mm[1], 111.0)
    guard.validate_move((20.0, 0.0, 111.0), pick_hover)
    guard.validate_move(pick_hover, entry.pick_xyz_mm)
    guard.validate_tool_on(entry.pick_xyz_mm)
    guard.validate_move(entry.pick_xyz_mm, pick_hover)
    guard.validate_move(pick_hover, drop_hover)
    guard.validate_move(drop_hover, entry.drop_xyz_mm)
    guard.validate_tool_off(entry.drop_xyz_mm)
    guard.validate_move(entry.drop_xyz_mm, drop_hover)


def test_guard_allows_student_selected_order_but_only_complete_trajectories(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    for entry in reversed(plan.entries):
        _execute_one(guard, entry)
    outcome = guard.completion()
    assert outcome["completed_entry_ids"] == sorted(entry.entry_id for entry in plan.entries)
    assert outcome["all_complete"] is True
    guard.validate_home()


def test_guard_rejects_motion_before_plan() -> None:
    guard = CodeRouteGuard()
    with pytest.raises(VisionPlatformError) as captured:
        guard.validate_move((20.0, 0.0, 111.0), (45.0, -45.0, 111.0))
    assert captured.value.code == "CODE_ROUTE_GUARD_INACTIVE"


def test_guard_latches_altered_coordinate_and_stops_future_commands(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    entry = plan.entries[0]
    with pytest.raises(VisionPlatformError) as captured:
        guard.validate_move((20.0, 0.0, 111.0), (entry.pick_xyz_mm[0] + 1.0, entry.pick_xyz_mm[1], 111.0))
    assert captured.value.code == "CODE_ROUTE_SEQUENCE_INVALID"
    with pytest.raises(VisionPlatformError, match="latched"):
        guard.validate_move((20.0, 0.0, 111.0), (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], 111.0))


def test_guard_requires_safe_height_before_lateral_pick_motion(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    entry = plan.entries[0]
    pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], plan.safe_z_mm + 1.0)
    with pytest.raises(VisionPlatformError) as captured:
        guard.validate_move((20.0, 0.0, 50.0), pick_hover)
    assert captured.value.code == "CODE_ROUTE_SEQUENCE_INVALID"


def test_tool_off_is_always_permitted_but_aborts_wrong_state(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    entry = plan.entries[0]
    pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], plan.safe_z_mm + 1.0)
    guard.validate_move((20.0, 0.0, plan.safe_z_mm + 1.0), pick_hover)
    assert guard.validate_tool_off(pick_hover) is False
    assert guard.completion()["failed"] is True


def test_host_reset_clears_the_plan_and_allows_a_fresh_activation(plan) -> None:
    guard = CodeRouteGuard()
    guard.activate(plan)
    guard.reset()
    assert guard.completion() == {
        "active": False, "state": "INACTIVE", "selected_entry_id": None,
        "completed_entry_ids": [], "all_complete": False, "failed": False,
    }
    guard.activate(plan)
    assert guard.completion()["active"] is True
