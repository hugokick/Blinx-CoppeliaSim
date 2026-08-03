"""Pure atomic guard for the V1-09 defect sorting plan.

The guard owns only immutable plan state and typed actions.  It deliberately
does not import a runner, robot, tool, CoppeliaSim, Qt, or subprocess module;
the private runner is the only layer allowed to turn these actions into
device calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Any

from vision_platform.experiments.defect_sorting import (
    EXPECTED_ENTRIES,
    ApprovedDefectSortEntry,
    DefectSortPlan,
)


ACTION_KINDS: tuple[str, ...] = (
    "move_safe_pick",
    "move_pick",
    "tool_on",
    "move_safe_pick",
    "move_safe_drop",
    "move_drop",
    "tool_off",
    "move_safe_drop",
)
_ACTION_KIND_SET = frozenset(ACTION_KINDS)
_EXPECTED_ENTRY_IDS = tuple(item[0] for item in EXPECTED_ENTRIES)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TEXT = re.compile(r"^[^\x00-\x1f\ud800-\udfff]{1,128}$")
_PROBE_FIELDS = frozenset(
    {
        "run_id",
        "scene_sha256",
        "frame_id",
        "plan_id",
        "entry_id",
        "part_id",
        "slot_id",
        "evidence_id",
    }
)


class DefectSortState(str, Enum):
    EMPTY = "EMPTY"
    ANALYZING = "ANALYZING"
    ACTIVE = "ACTIVE"
    ENTRY_ACTIVE = "ENTRY_ACTIVE"
    AWAITING_PROBE = "AWAITING_PROBE"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class DefectSortGuardError(ValueError):
    """Stable fail-closed error raised by the pure guard."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _raise(code: str, message: str) -> None:
    raise DefectSortGuardError(code, message)


def _finite(value: Any, name: str, code: str) -> float:
    if type(value) not in {int, float}:
        _raise(code, f"{name} must be a finite built-in number")
    try:
        normalized = float(value)
    except OverflowError:
        _raise(code, f"{name} must be a finite built-in number")
    if not math.isfinite(normalized):
        _raise(code, f"{name} must be a finite built-in number")
    return normalized


def _text(value: Any, name: str, code: str) -> str:
    if type(value) is not str or _SAFE_TEXT.fullmatch(value) is None:
        _raise(code, f"{name} is invalid")
    return value


def _point(value: Any, name: str, code: str) -> tuple[float, float, float]:
    if type(value) not in {tuple, list} or len(value) != 3:
        _raise(code, f"{name} is invalid")
    return tuple(_finite(item, f"{name}[{index}]", code) for index, item in enumerate(value))  # type: ignore[return-value]


def _digest(value: Any, name: str, code: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        _raise(code, f"{name} is invalid")
    return value


def _freeze_json(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        _raise("DEFECT_SORT_PROBE_INVALID", "evidence nesting is too deep")
    if value is None or type(value) is bool:
        return value
    if type(value) in {int, float}:
        normalized = float(value)
        if not math.isfinite(normalized):
            _raise("DEFECT_SORT_PROBE_INVALID", "evidence contains a non-finite number")
        return value
    if type(value) is str:
        if _SAFE_TEXT.fullmatch(value) is None:
            _raise("DEFECT_SORT_PROBE_INVALID", "evidence text is invalid")
        return value
    if isinstance(value, Mapping):
        if len(value) > 32 or any(type(key) is not str for key in value):
            _raise("DEFECT_SORT_PROBE_INVALID", "evidence mapping is invalid")
        return MappingProxyType({key: _freeze_json(nested, depth=depth + 1) for key, nested in value.items()})
    if type(value) in {tuple, list}:
        if len(value) > 32:
            _raise("DEFECT_SORT_PROBE_INVALID", "evidence sequence is too large")
        return tuple(_freeze_json(item, depth=depth + 1) for item in value)
    _raise("DEFECT_SORT_PROBE_INVALID", "evidence is not JSON-native")


@dataclass(frozen=True)
class DefectSortAction:
    """One immutable plan-derived action for the private runner."""

    action_id: str
    kind: str
    entry_id: str
    target_xyz_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        _text(self.action_id, "action_id", "DEFECT_SORT_ACTION_INVALID")
        if self.kind not in _ACTION_KIND_SET:
            _raise("DEFECT_SORT_ACTION_INVALID", "action kind is not approved")
        if self.entry_id not in _EXPECTED_ENTRY_IDS:
            _raise("DEFECT_SORT_ACTION_INVALID", "entry_id is not approved")
        object.__setattr__(self, "target_xyz_mm", _point(self.target_xyz_mm, "target_xyz_mm", "DEFECT_SORT_ACTION_INVALID"))

    @property
    def action_kind(self) -> str:
        return self.kind

    @property
    def target(self) -> tuple[float, float, float]:
        return self.target_xyz_mm


@dataclass(frozen=True)
class _FrozenEntry:
    entry_id: str
    part_id: str
    decision: str
    route_id: str
    slot_id: str
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]


@dataclass(frozen=True)
class _FrozenPlan:
    plan_id: str
    run_id: str
    frame_id: str
    scene_sha256: str
    entries: tuple[_FrozenEntry, ...]
    safe_z_mm: float
    speed_mm_s: float


class DefectSortGuard:
    """Atomic finite-state guard for one host-approved V1-09 plan."""

    def __init__(self) -> None:
        self._state = DefectSortState.EMPTY
        self._plan: _FrozenPlan | None = None
        self._active_entry: _FrozenEntry | None = None
        self._actions: tuple[DefectSortAction, ...] = ()
        self._action_index = 0
        self._consumed: set[str] = set()
        self._evidence: dict[str, Mapping[str, Any]] = {}
        self._error: Mapping[str, Any] | None = None

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def state_enum(self) -> DefectSortState:
        return self._state

    @property
    def plan_id(self) -> str | None:
        return None if self._plan is None else self._plan.plan_id

    @property
    def actions(self) -> tuple[DefectSortAction, ...]:
        return self._actions

    @property
    def current_action(self) -> DefectSortAction | None:
        if self._state is not DefectSortState.ENTRY_ACTIVE or self._action_index >= len(self._actions):
            return None
        return self._actions[self._action_index]

    @property
    def active_entry_id(self) -> str | None:
        return None if self._active_entry is None else self._active_entry.entry_id

    @property
    def consumed_entry_ids(self) -> tuple[str, ...]:
        if self._plan is None:
            return ()
        return tuple(entry.entry_id for entry in self._plan.entries if entry.entry_id in self._consumed)

    @property
    def error(self) -> Mapping[str, Any] | None:
        return self._error

    def begin_analysis(self) -> None:
        if self._state is not DefectSortState.EMPTY:
            _raise("DEFECT_SORT_ANALYSIS_ALREADY_ACTIVE", "analysis requires an empty guard")
        self._state = DefectSortState.ANALYZING

    def activate(self, plan: DefectSortPlan) -> None:
        if self._state is not DefectSortState.ANALYZING:
            _raise("DEFECT_SORT_PLAN_NOT_ANALYZING", "analysis is not awaiting activation")
        frozen = self._freeze_plan(plan)
        self._plan = frozen
        self._active_entry = None
        self._actions = ()
        self._action_index = 0
        self._consumed.clear()
        self._evidence.clear()
        self._error = None
        self._state = DefectSortState.ACTIVE

    def fail_analysis(self, error: Mapping[str, Any] | str | None = None) -> None:
        if self._state is not DefectSortState.ANALYZING:
            _raise("DEFECT_SORT_ANALYSIS_INVALID", "analysis is not active")
        self.fail(error or {"code": "DEFECT_SORT_ANALYSIS_REJECTED", "message": "analysis rejected"})

    def begin_entry(self, entry_id: str) -> tuple[DefectSortAction, ...]:
        if self._state is DefectSortState.EMPTY:
            _raise("DEFECT_SORT_PLAN_NOT_ACTIVE", "plan is not active")
        if self._state is DefectSortState.ANALYZING:
            _raise("DEFECT_SORT_PLAN_NOT_ACTIVE", "plan is not active")
        if self._state is DefectSortState.FAILED:
            _raise("DEFECT_SORT_PLAN_FAILED", "plan is failed")
        if self._state is DefectSortState.COMPLETE:
            _raise("DEFECT_SORT_PLAN_COMPLETE", "all entries are consumed")
        if self._state is not DefectSortState.ACTIVE or self._plan is None:
            _raise("DEFECT_SORT_ENTRY_INVALID", "another entry is already active")
        if type(entry_id) is not str or entry_id not in _EXPECTED_ENTRY_IDS:
            _raise("DEFECT_SORT_ENTRY_INVALID", "entry_id is not approved")
        if entry_id in self._consumed:
            _raise("DEFECT_SORT_ENTRY_INVALID", "entry_id has already been consumed")
        entry = next((item for item in self._plan.entries if item.entry_id == entry_id), None)
        if entry is None:
            _raise("DEFECT_SORT_ENTRY_INVALID", "entry_id is not in the active plan")
        pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], self._plan.safe_z_mm)
        drop_hover = (entry.drop_xyz_mm[0], entry.drop_xyz_mm[1], self._plan.safe_z_mm)
        targets = (
            pick_hover,
            entry.pick_xyz_mm,
            entry.pick_xyz_mm,
            pick_hover,
            drop_hover,
            entry.drop_xyz_mm,
            entry.drop_xyz_mm,
            drop_hover,
        )
        self._actions = tuple(
            DefectSortAction(
                action_id=f"{entry.entry_id}:{index}:{kind}",
                kind=kind,
                entry_id=entry.entry_id,
                target_xyz_mm=target,
            )
            for index, (kind, target) in enumerate(zip(ACTION_KINDS, targets))
        )
        self._active_entry = entry
        self._action_index = 0
        self._state = DefectSortState.ENTRY_ACTIVE
        return self._actions

    def confirm_action(self, action_id: str | DefectSortAction) -> None:
        if self._state is not DefectSortState.ENTRY_ACTIVE or self._active_entry is None:
            _raise("DEFECT_SORT_ACTION_INVALID", "no action is awaiting confirmation")
        expected = self.current_action
        if expected is None:
            _raise("DEFECT_SORT_ACTION_INVALID", "no action is awaiting confirmation")
        supplied = action_id.action_id if isinstance(action_id, DefectSortAction) else action_id
        if type(supplied) is not str or supplied != expected.action_id:
            self.fail({"code": "DEFECT_SORT_ACTION_INVALID", "message": "action confirmation is out of order"})
            _raise("DEFECT_SORT_ACTION_INVALID", "action confirmation is out of order")
        self._action_index += 1
        if self._action_index == len(self._actions):
            self._state = DefectSortState.AWAITING_PROBE

    def confirm_entry_probe(self, entry_id: str, evidence_ref: Mapping[str, Any]) -> None:
        if self._plan is not None and type(entry_id) is str and entry_id in self._consumed:
            _raise("DEFECT_SORT_ENTRY_CONSUMED", "entry_id has already been consumed")
        if self._state is not DefectSortState.AWAITING_PROBE or self._active_entry is None or self._plan is None:
            _raise("DEFECT_SORT_PROBE_INVALID", "entry probe is not expected")
        if type(entry_id) is not str or entry_id != self._active_entry.entry_id:
            self.fail({"code": "DEFECT_SORT_PROBE_INVALID", "message": "probe entry_id is not active"})
            _raise("DEFECT_SORT_PROBE_INVALID", "probe entry_id is not active")
        try:
            normalized = self._validate_probe(evidence_ref, self._active_entry, self._plan)
        except DefectSortGuardError as exc:
            self.fail({"code": exc.code, "message": str(exc).split(": ", 1)[-1][:128]})
            raise
        self._evidence[entry_id] = normalized
        self._consumed.add(entry_id)
        self._active_entry = None
        self._actions = ()
        self._action_index = 0
        self._state = DefectSortState.COMPLETE if len(self._consumed) == 6 else DefectSortState.ACTIVE

    def fail(self, error: Mapping[str, Any] | str | None = None) -> None:
        if self._state is DefectSortState.COMPLETE:
            return
        if self._error is None:
            if isinstance(error, Mapping):
                try:
                    frozen = _freeze_json(dict(error))
                    if isinstance(frozen, Mapping):
                        self._error = frozen
                    else:
                        self._error = MappingProxyType({"code": "DEFECT_SORT_FAILED", "message": "failure payload is invalid"})
                except DefectSortGuardError:
                    self._error = MappingProxyType({"code": "DEFECT_SORT_FAILED", "message": "failure payload is invalid"})
            else:
                message = "defect sorting failed" if error is None else str(error)
                self._error = MappingProxyType({"code": "DEFECT_SORT_FAILED", "message": message[:128]})
        self._plan = None
        self._active_entry = None
        self._actions = ()
        self._action_index = 0
        self._consumed.clear()
        self._evidence.clear()
        self._state = DefectSortState.FAILED

    def stop(self) -> None:
        if self._state is not DefectSortState.COMPLETE and self._state is not DefectSortState.FAILED:
            self.fail({"code": "DEFECT_SORT_STOPPED", "message": "sort execution stopped"})

    def reset(self) -> None:
        self._state = DefectSortState.EMPTY
        self._plan = None
        self._active_entry = None
        self._actions = ()
        self._action_index = 0
        self._consumed.clear()
        self._evidence.clear()
        self._error = None

    def snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "state": self._state.value,
                "plan_id": self.plan_id,
                "entries": () if self._plan is None else tuple(entry.entry_id for entry in self._plan.entries),
                "active_entry_id": self.active_entry_id,
                "current_action_id": None if self.current_action is None else self.current_action.action_id,
                "consumed_entry_ids": self.consumed_entry_ids,
                "evidence_refs": tuple(self._evidence.values()),
                "error": self._error,
            }
        )

    @staticmethod
    def _freeze_plan(plan: Any) -> _FrozenPlan:
        if not isinstance(plan, DefectSortPlan):
            _raise("DEFECT_SORT_PLAN_INVALID", "plan must be a complete DefectSortPlan")
        if plan.status != "PASS" or plan.schema_version != 1 or len(plan.entries) != 6:
            _raise("DEFECT_SORT_PLAN_INVALID", "plan status/schema is invalid")
        if tuple(entry.entry_id for entry in plan.entries) != _EXPECTED_ENTRY_IDS:
            _raise("DEFECT_SORT_PLAN_INVALID", "plan entry whitelist is incomplete")
        frozen_entries: list[_FrozenEntry] = []
        for expected, entry in zip(EXPECTED_ENTRIES, plan.entries):
            if not isinstance(entry, ApprovedDefectSortEntry):
                _raise("DEFECT_SORT_PLAN_INVALID", "plan entry type is invalid")
            if (entry.part_id, entry.route_id, entry.slot_id, entry.decision) != (expected[1], expected[3], expected[4], expected[2]):
                _raise("DEFECT_SORT_PLAN_INVALID", "plan route whitelist is invalid")
            frozen_entries.append(
                _FrozenEntry(
                    entry_id=entry.entry_id,
                    part_id=entry.part_id,
                    decision=entry.decision,
                    route_id=entry.route_id,
                    slot_id=entry.slot_id,
                    pick_xyz_mm=_point(entry.pick_xyz_mm, "pick_xyz_mm", "DEFECT_SORT_PLAN_INVALID"),
                    drop_xyz_mm=_point(entry.drop_xyz_mm, "drop_xyz_mm", "DEFECT_SORT_PLAN_INVALID"),
                )
            )
        safe_z = _finite(plan.safe_z_mm, "safe_z_mm", "DEFECT_SORT_PLAN_INVALID")
        speed = _finite(plan.speed_mm_s, "speed_mm_s", "DEFECT_SORT_PLAN_INVALID")
        if speed <= 0.0 or any(safe_z <= max(max(entry.pick_xyz_mm[2], entry.drop_xyz_mm[2]) for entry in frozen_entries) for _ in (0,)):
            _raise("DEFECT_SORT_PLAN_INVALID", "plan safety parameters are invalid")
        return _FrozenPlan(
            plan_id=_digest(plan.plan_id, "plan_id", "DEFECT_SORT_PLAN_INVALID"),
            run_id=_text(plan.run_id, "run_id", "DEFECT_SORT_PLAN_INVALID"),
            frame_id=_text(plan.frame_id, "frame_id", "DEFECT_SORT_PLAN_INVALID"),
            scene_sha256=_digest(plan.scene_sha256, "scene_sha256", "DEFECT_SORT_PLAN_INVALID"),
            entries=tuple(frozen_entries),
            safe_z_mm=safe_z,
            speed_mm_s=speed,
        )

    @staticmethod
    def _validate_probe(value: Mapping[str, Any], entry: _FrozenEntry, plan: _FrozenPlan) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or set(value) != set(_PROBE_FIELDS):
            _raise("DEFECT_SORT_PROBE_INVALID", "evidence_ref schema is invalid")
        raw = dict(value)
        for field in _PROBE_FIELDS:
            _text(raw[field], field, "DEFECT_SORT_PROBE_INVALID")
        _digest(raw["scene_sha256"], "scene_sha256", "DEFECT_SORT_PROBE_INVALID")
        _digest(raw["plan_id"], "plan_id", "DEFECT_SORT_PROBE_INVALID")
        expected = {
            "run_id": plan.run_id,
            "scene_sha256": plan.scene_sha256,
            "frame_id": plan.frame_id,
            "plan_id": plan.plan_id,
            "entry_id": entry.entry_id,
            "part_id": entry.part_id,
            "slot_id": entry.slot_id,
        }
        if any(raw[field] != expected[field] for field in expected):
            _raise("DEFECT_SORT_PROBE_INVALID", "evidence_ref does not match the active plan entry")
        return MappingProxyType({field: raw[field] for field in _PROBE_FIELDS})


__all__ = [
    "ACTION_KINDS",
    "DefectSortAction",
    "DefectSortGuard",
    "DefectSortGuardError",
    "DefectSortState",
]
