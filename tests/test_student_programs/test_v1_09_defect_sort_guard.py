from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vision_platform.experiments.defect_sorting import (
    ApprovedDefectSortEntry,
    DefectSortPlan,
)
from vision_platform.student.defect_sort_guard import (
    ACTION_KINDS,
    DefectSortAction,
    DefectSortGuard,
    DefectSortGuardError,
    DefectSortState,
)


ENTRIES = (
    ("entry_a", "part_a", "qualified", "route_qualified", "slot_qualified"),
    ("entry_b", "part_b", "missing", "route_missing", "slot_missing"),
    ("entry_c", "part_c", "hole", "route_hole", "slot_hole"),
    ("entry_d", "part_d", "foreign", "route_foreign", "slot_foreign"),
    ("entry_e", "part_e", "broken", "route_broken", "slot_broken"),
    ("entry_f", "part_f", "dimension", "route_dimension", "slot_dimension"),
)
SCENE_SHA = "1" * 64
CONFIG_SHA = "2" * 64
ASSET_SHA = "3" * 64
PICK_POSITIONS = ((140.0, -16.0, 18.0), (85.0, -16.0, 18.0), (30.0, -16.0, 18.0), (140.0, 38.0, 18.0), (85.0, 38.0, 18.0), (30.0, 38.0, 18.0))
DROP_POSITIONS = ((132.0, -93.0, 22.0), (85.0, -93.0, 22.0), (38.0, -93.0, 22.0), (132.0, 75.0, 22.0), (85.0, 75.0, 22.0), (38.0, 75.0, 22.0))


@pytest.fixture
def plan() -> DefectSortPlan:
    entries = tuple(
        ApprovedDefectSortEntry(
            entry_id=entry_id,
            part_id=part_id,
            decision=decision,
            route_id=route_id,
            slot_id=slot_id,
            roi_px=(80 + (index % 3) * 336, 56 + (index // 3) * 328, 192, 192),
            pick_xyz_mm=PICK_POSITIONS[index],
            drop_xyz_mm=DROP_POSITIONS[index],
            reference_crop_sha256="a" * 64,
            candidate_crop_sha256=("b" if index == 0 else format(index + 2, "x")) * 64,
            findings_sha256="d" * 64,
            run_id="run-001",
            frame_id="frame-001",
            scene_sha256=SCENE_SHA,
            config_sha256=CONFIG_SHA,
            asset_manifest_sha256=ASSET_SHA,
        )
        for index, (entry_id, part_id, decision, route_id, slot_id) in enumerate(ENTRIES)
    )
    return DefectSortPlan(
        plan_id="e" * 64,
        entries=entries,
        run_id="run-001",
        frame_id="frame-001",
        scene_sha256=SCENE_SHA,
        config_sha256=CONFIG_SHA,
        asset_manifest_sha256=ASSET_SHA,
        image_size=(1024, 1024),
        safe_z_mm=110.0,
        speed_mm_s=15.0,
    )


def _active_guard(plan: DefectSortPlan) -> DefectSortGuard:
    guard = DefectSortGuard()
    guard.begin_analysis()
    guard.activate(plan)
    return guard


def _evidence(entry, *, evidence_id: str | None = None, run_id: str = "run-001", scene_sha: str = SCENE_SHA):
    return {
        "run_id": run_id,
        "scene_sha256": scene_sha,
        "frame_id": "frame-001",
        "plan_id": "e" * 64,
        "entry_id": entry.entry_id,
        "part_id": entry.part_id,
        "slot_id": entry.slot_id,
        "evidence_id": evidence_id or f"evidence-{entry.entry_id}",
    }


def _finish_actions(guard: DefectSortGuard, entry_id: str) -> tuple[DefectSortAction, ...]:
    actions = guard.begin_entry(entry_id)
    for action in actions:
        guard.confirm_action(action.action_id)
    return actions


def test_analysis_must_begin_before_atomic_activation(plan: DefectSortPlan) -> None:
    guard = DefectSortGuard()
    with pytest.raises(DefectSortGuardError, match="DEFECT_SORT_PLAN_NOT_ANALYZING"):
        guard.activate(plan)
    assert guard.state_enum is DefectSortState.EMPTY
    guard.begin_analysis()
    guard.activate(plan)
    assert guard.state_enum is DefectSortState.ACTIVE
    assert guard.plan_id == plan.plan_id


def test_invalid_plan_activation_is_atomic_and_analysis_failure_has_no_actions() -> None:
    guard = DefectSortGuard()
    guard.begin_analysis()
    with pytest.raises(DefectSortGuardError) as captured:
        guard.activate(object())
    assert captured.value.code == "DEFECT_SORT_PLAN_INVALID"
    assert guard.state_enum is DefectSortState.ANALYZING
    assert guard.actions == ()
    guard.fail_analysis({"code": "DEFECT_SORT_ANALYSIS_REJECTED", "message": "invalid batch"})
    assert guard.state_enum is DefectSortState.FAILED
    assert guard.actions == ()
    assert guard.plan_id is None


def test_guard_returns_immutable_eight_step_actions_without_device_access(plan: DefectSortPlan) -> None:
    guard = _active_guard(plan)
    actions = guard.begin_entry("entry_a")
    assert type(actions) is tuple
    assert tuple(action.kind for action in actions) == ACTION_KINDS
    assert all(type(action) is DefectSortAction for action in actions)
    assert actions[0].target_xyz_mm == (140.0, -16.0, 110.0)
    assert actions[1].target_xyz_mm == (140.0, -16.0, 18.0)
    assert actions[5].target_xyz_mm == (132.0, -93.0, 22.0)
    with pytest.raises(FrozenInstanceError):
        actions[0].kind = "move_drop"  # type: ignore[misc]
    with pytest.raises(TypeError):
        actions[0].target_xyz_mm[0] = 99.0  # type: ignore[index]


def test_only_one_unconsumed_entry_can_be_active(plan: DefectSortPlan) -> None:
    guard = _active_guard(plan)
    guard.begin_entry("entry_a")
    with pytest.raises(DefectSortGuardError) as captured:
        guard.begin_entry("entry_b")
    assert captured.value.code == "DEFECT_SORT_ENTRY_INVALID"
    guard.stop()
    assert guard.state_enum is DefectSortState.FAILED


def test_action_confirmation_is_ordered_and_probe_is_mandatory(plan: DefectSortPlan) -> None:
    guard = _active_guard(plan)
    actions = guard.begin_entry("entry_a")
    with pytest.raises(DefectSortGuardError) as captured:
        guard.confirm_action(actions[1].action_id)
    assert captured.value.code == "DEFECT_SORT_ACTION_INVALID"
    assert guard.state_enum is DefectSortState.FAILED

    guard.reset()
    guard.begin_analysis()
    guard.activate(plan)
    actions = guard.begin_entry("entry_a")
    for action in actions:
        guard.confirm_action(action.action_id)
    assert guard.state_enum is DefectSortState.AWAITING_PROBE
    with pytest.raises(DefectSortGuardError) as captured:
        guard.begin_entry("entry_b")
    assert captured.value.code == "DEFECT_SORT_ENTRY_INVALID"
    assert guard.state_enum is DefectSortState.AWAITING_PROBE


def test_probe_binds_exact_same_run_and_consumes_atomically(plan: DefectSortPlan) -> None:
    guard = _active_guard(plan)
    _finish_actions(guard, "entry_a")
    entry = plan.entries[0]
    bad = _evidence(entry, run_id="run-other")
    with pytest.raises(DefectSortGuardError) as captured:
        guard.confirm_entry_probe(entry.entry_id, bad)
    assert captured.value.code == "DEFECT_SORT_PROBE_INVALID"
    assert guard.state_enum is DefectSortState.FAILED

    guard.reset()
    guard.begin_analysis()
    guard.activate(plan)
    _finish_actions(guard, "entry_a")
    guard.confirm_entry_probe("entry_a", _evidence(plan.entries[0]))
    assert guard.state_enum is DefectSortState.ACTIVE
    assert guard.consumed_entry_ids == ("entry_a",)
    with pytest.raises(DefectSortGuardError) as captured:
        guard.confirm_entry_probe("entry_a", _evidence(plan.entries[0]))
    assert captured.value.code == "DEFECT_SORT_ENTRY_CONSUMED"


def test_all_six_entries_require_probe_before_complete(plan: DefectSortPlan) -> None:
    guard = _active_guard(plan)
    for entry in plan.entries:
        _finish_actions(guard, entry.entry_id)
        guard.confirm_entry_probe(entry.entry_id, _evidence(entry))
    assert guard.state_enum is DefectSortState.COMPLETE
    assert guard.consumed_entry_ids == tuple(entry.entry_id for entry in plan.entries)
    with pytest.raises(DefectSortGuardError):
        guard.begin_entry("entry_a")


def test_stop_and_failure_keep_first_error_and_controlled_reset(plan: DefectSortPlan) -> None:
    guard = _active_guard(plan)
    guard.fail({"code": "FIRST", "message": "probe failed"})
    guard.fail({"code": "SECOND", "message": "masked"})
    assert guard.state_enum is DefectSortState.FAILED
    assert guard.error["code"] == "FIRST"
    assert guard.actions == ()
    guard.stop()
    assert guard.error["code"] == "FIRST"
    guard.reset()
    assert guard.state_enum is DefectSortState.EMPTY
    assert guard.plan_id is None
    assert guard.consumed_entry_ids == ()


@pytest.mark.parametrize("field", ["run_id", "scene_sha256", "frame_id", "plan_id", "entry_id", "part_id", "slot_id", "evidence_id"])
def test_probe_rejects_missing_or_extra_fields_and_fails_closed(plan: DefectSortPlan, field: str) -> None:
    guard = _active_guard(plan)
    entry = plan.entries[0]
    _finish_actions(guard, entry.entry_id)
    missing = _evidence(entry)
    missing.pop(field)
    with pytest.raises(DefectSortGuardError) as captured:
        guard.confirm_entry_probe(entry.entry_id, missing)
    assert captured.value.code == "DEFECT_SORT_PROBE_INVALID"
    assert guard.state_enum is DefectSortState.FAILED


def test_snapshot_is_immutable_and_contains_no_devices(plan: DefectSortPlan) -> None:
    guard = _active_guard(plan)
    snapshot = guard.snapshot()
    assert snapshot["state"] == "ACTIVE"
    with pytest.raises(TypeError):
        snapshot["state"] = "FAILED"  # type: ignore[index]
