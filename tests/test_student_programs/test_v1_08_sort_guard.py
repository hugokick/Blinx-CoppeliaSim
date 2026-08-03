from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from vision_platform.experiments.ocr_sorting import (
    ApprovedOcrSortEntry,
    OcrSortPlan,
)
from vision_platform.student.ocr_sort_guard import (
    OcrSortAction,
    OcrSortGuard,
    OcrSortGuardError,
    OcrSortState,
)


@pytest.fixture
def plan() -> OcrSortPlan:
    entries = tuple(
        ApprovedOcrSortEntry(
            entry_id=entry_id,
            part_id=part_id,
            identifier=identifier,
            route_id=route_id,
            roi_px=(100 + index * 10, 120, 80, 40),
            pick_xyz_mm=pick,
            drop_xyz_mm=drop,
            confidence=0.98,
        )
        for index, (entry_id, part_id, identifier, route_id, pick, drop) in enumerate(
            (
                ("entry_a", "part_a", "A1", "route_alpha", (35.0, -55.0, 18.0), (116.0, -75.0, 22.0)),
                ("entry_b", "part_b", "A2", "route_alpha", (75.0, -55.0, 18.0), (128.0, -75.0, 22.0)),
                ("entry_c", "part_c", "B1", "route_beta", (35.0, 25.0, 18.0), (116.0, 75.0, 22.0)),
                ("entry_d", "part_d", "B2", "route_beta", (75.0, 25.0, 18.0), (128.0, 75.0, 22.0)),
            )
        )
    )
    return OcrSortPlan("a" * 64, entries, 110.0, 15.0)


def _active_guard(plan: OcrSortPlan) -> OcrSortGuard:
    guard = OcrSortGuard()
    guard.activate(plan)
    return guard


def _evidence(entry, *, run_id="run-1", scene_hash="b" * 64, evidence_id=None, slot=None):
    return {
        "run_id": run_id,
        "scene_hash": scene_hash,
        "part_id": entry.part_id,
        "route_id": entry.route_id,
        "slot_id": slot or ("slot_1" if entry.entry_id in {"entry_a", "entry_c"} else "slot_2"),
        "evidence_id": evidence_id or f"evidence-{entry.entry_id}",
    }


def _finish_actions(guard: OcrSortGuard, entry_id: str) -> tuple[OcrSortAction, ...]:
    actions = guard.begin_entry(entry_id)
    for action in actions:
        guard.confirm_action(action.action_id)
    return actions


def test_motion_is_impossible_before_atomic_plan_activation() -> None:
    guard = OcrSortGuard()
    with pytest.raises(OcrSortGuardError, match="OCR_SORT_PLAN_NOT_ACTIVE") as captured:
        guard.begin_entry("entry_a")
    assert captured.value.code == "OCR_SORT_PLAN_NOT_ACTIVE"
    assert guard.state == OcrSortState.EMPTY


def test_activation_is_atomic_and_requires_complete_ocr_plan(plan: OcrSortPlan) -> None:
    guard = OcrSortGuard()
    with pytest.raises(OcrSortGuardError) as captured:
        guard.activate(object())
    assert captured.value.code == "OCR_SORT_PLAN_INVALID"
    assert guard.state == OcrSortState.EMPTY
    incomplete = SimpleNamespace(
        plan_id="a" * 64,
        entries=plan.entries[:3],
        safe_z_mm=110.0,
        speed_mm_s=15.0,
        status="PASS",
    )
    with pytest.raises(OcrSortGuardError) as captured:
        guard.activate(incomplete)
    assert captured.value.code == "OCR_SORT_PLAN_INVALID"
    assert guard.state == OcrSortState.EMPTY


def test_guard_returns_frozen_actions_without_device_access(plan: OcrSortPlan) -> None:
    guard = _active_guard(plan)
    actions = guard.begin_entry("entry_a")
    assert type(actions) is tuple
    assert [action.kind for action in actions] == [
        "move_safe_pick",
        "move_pick",
        "tool_on",
        "move_safe_pick",
        "move_safe_drop",
        "move_drop",
        "tool_off",
        "move_safe_drop",
    ]
    assert all(type(action) is OcrSortAction for action in actions)
    assert actions[0].target_xyz_mm == (35.0, -55.0, 110.0)
    assert actions[1].target_xyz_mm == (35.0, -55.0, 18.0)
    assert actions[5].target_xyz_mm == (116.0, -75.0, 22.0)
    with pytest.raises(FrozenInstanceError):
        actions[0].kind = "move_drop"  # type: ignore[misc]
    with pytest.raises(TypeError):
        actions[0].target_xyz_mm[0] = 99.0  # type: ignore[index]


def test_action_confirmation_is_ordered_and_final_action_awaits_probe(plan: OcrSortPlan) -> None:
    guard = _active_guard(plan)
    actions = guard.begin_entry("entry_a")
    with pytest.raises(OcrSortGuardError) as captured:
        guard.confirm_action(actions[1].action_id)
    assert captured.value.code == "OCR_SORT_ACTION_INVALID"
    assert guard.state == OcrSortState.ENTRY_ACTIVE
    for action in actions[:-1]:
        guard.confirm_action(action.action_id)
    assert guard.state == OcrSortState.ENTRY_ACTIVE
    guard.confirm_action(actions[-1].action_id)
    assert guard.state == OcrSortState.ENTRY_AWAITING_PROBE
    with pytest.raises(OcrSortGuardError) as captured:
        guard.confirm_action(actions[-1].action_id)
    assert captured.value.code == "OCR_SORT_ACTION_INVALID"


def test_only_one_unconsumed_entry_can_be_active(plan: OcrSortPlan) -> None:
    guard = _active_guard(plan)
    guard.begin_entry("entry_a")
    with pytest.raises(OcrSortGuardError) as captured:
        guard.begin_entry("entry_b")
    assert captured.value.code == "OCR_SORT_ENTRY_INVALID"
    with pytest.raises(OcrSortGuardError) as captured:
        guard.begin_entry("unknown")
    assert captured.value.code == "OCR_SORT_ENTRY_INVALID"


def test_probe_requires_complete_same_run_binding_and_consumes_atomically(plan: OcrSortPlan) -> None:
    guard = _active_guard(plan)
    entry = plan.entries[0]
    _finish_actions(guard, entry.entry_id)
    with pytest.raises(OcrSortGuardError) as captured:
        guard.confirm_entry_probe(entry.entry_id, {"run_id": "run-1"})
    assert captured.value.code == "OCR_SORT_PROBE_INVALID"
    assert guard.state == OcrSortState.ENTRY_AWAITING_PROBE
    reference = _evidence(entry)
    returned = guard.confirm_entry_probe(entry.entry_id, reference)
    assert returned is None
    assert guard.state == OcrSortState.ACTIVE
    assert guard.consumed_entry_ids == ("entry_a",)
    with pytest.raises(OcrSortGuardError) as captured:
        guard.confirm_entry_probe(entry.entry_id, reference)
    assert captured.value.code == "OCR_SORT_PROBE_INVALID"

    _finish_actions(guard, plan.entries[1].entry_id)
    bad_run = _evidence(plan.entries[1], run_id="run-2")
    with pytest.raises(OcrSortGuardError) as captured:
        guard.confirm_entry_probe(plan.entries[1].entry_id, bad_run)
    assert captured.value.code == "OCR_SORT_PROBE_INVALID"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("part_id", "part_b"),
        ("route_id", "route_beta"),
        ("slot_id", "slot_2"),
    ],
)
def test_probe_rejects_mismatched_entry_binding(plan: OcrSortPlan, field: str, value: str) -> None:
    guard = _active_guard(plan)
    entry = plan.entries[0]
    _finish_actions(guard, entry.entry_id)
    reference = _evidence(entry)
    reference[field] = value
    with pytest.raises(OcrSortGuardError) as captured:
        guard.confirm_entry_probe(entry.entry_id, reference)
    assert captured.value.code == "OCR_SORT_PROBE_INVALID"
    assert guard.state == OcrSortState.ENTRY_AWAITING_PROBE


def test_fail_entry_invalidates_and_retains_first_error(plan: OcrSortPlan) -> None:
    guard = _active_guard(plan)
    guard.begin_entry("entry_a")
    guard.fail_entry({"code": "FIRST_FAILURE", "message": "probe failed"})
    assert guard.state == OcrSortState.INVALIDATED
    assert guard.error == {"code": "FIRST_FAILURE", "message": "probe failed"}
    guard.fail_entry({"code": "SECOND_FAILURE", "message": "masked"})
    assert guard.error["code"] == "FIRST_FAILURE"
    with pytest.raises(OcrSortGuardError) as captured:
        guard.begin_entry("entry_b")
    assert captured.value.code == "OCR_SORT_PLAN_INVALIDATED"


def test_stop_invalidates_and_reset_clears_the_plan(plan: OcrSortPlan) -> None:
    guard = _active_guard(plan)
    guard.stop()
    assert guard.state == OcrSortState.INVALIDATED
    assert guard.plan_id is None
    with pytest.raises(OcrSortGuardError) as captured:
        guard.begin_entry("entry_a")
    assert captured.value.code == "OCR_SORT_PLAN_INVALIDATED"

    guard.reset()
    assert guard.state == OcrSortState.EMPTY
    assert guard.plan_id is None
    with pytest.raises(OcrSortGuardError) as captured:
        guard.begin_entry("entry_a")
    assert captured.value.code == "OCR_SORT_PLAN_NOT_ACTIVE"


def test_public_snapshot_is_json_native_and_read_only(plan: OcrSortPlan) -> None:
    guard = _active_guard(plan)
    snapshot = guard.snapshot()
    json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    assert snapshot["state"] == "ACTIVE"
    assert snapshot["entries"] == ("entry_a", "entry_b", "entry_c", "entry_d")
    with pytest.raises(TypeError):
        snapshot["entries"] = ()  # type: ignore[index]
    guard.begin_entry("entry_a")
    live = guard.snapshot()
    assert live["active_entry_id"] == "entry_a"


def test_guard_module_has_no_device_or_application_dependencies() -> None:
    import vision_platform.student.ocr_sort_guard as module

    forbidden = ("robot", "tool", "sim", "gateway", "application")
    imported = set(module.__dict__)
    assert not any(
        any(token in name.lower() for token in forbidden)
        for name in imported
        if not name.startswith("__")
    )
