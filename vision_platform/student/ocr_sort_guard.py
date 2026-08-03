"""Pure atomic state guard for the V1-08 OCR sorting plan.

The guard is deliberately independent from the execution runtime.  It keeps a
private, immutable copy of the approved plan and only returns typed actions
which a runner may execute.  It never receives coordinates from the caller and
does not import or store any device, process, application, or gateway object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from typing import Any

from vision_platform.experiments.ocr_sorting import (
    ApprovedOcrSortEntry,
    OcrSortPlan,
    ROUTES,
)


_EXPECTED_ENTRIES = tuple(ROUTES)
_EXPECTED_ENTRY_IDS = frozenset(item[0] for item in _EXPECTED_ENTRIES)
_ACTION_KINDS = (
    "move_safe_pick",
    "move_pick",
    "tool_on",
    "move_safe_pick",
    "move_safe_drop",
    "move_drop",
    "tool_off",
    "move_safe_drop",
)
_ACTION_KIND_SET = frozenset(_ACTION_KINDS)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_PROBE_FIELDS = frozenset(
    {"run_id", "scene_hash", "part_id", "route_id", "slot_id", "evidence_id"}
)


class OcrSortState(str, Enum):
    EMPTY = "EMPTY"
    ACTIVE = "ACTIVE"
    ENTRY_ACTIVE = "ENTRY_ACTIVE"
    ENTRY_AWAITING_PROBE = "ENTRY_AWAITING_PROBE"
    COMPLETED = "COMPLETED"
    INVALIDATED = "INVALIDATED"


class OcrSortGuardError(ValueError):
    """Stable fail-closed guard error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.details = _freeze_json(dict(details or {}))
        super().__init__(f"{self.code}: {message}")


class _FrozenDict(dict[str, Any]):
    """A JSON-serializable mapping that rejects all mutation methods."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("frozen mapping cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_json(value: Any, *, depth: int = 0) -> Any:
    """Copy a bounded JSON-native value into immutable containers."""

    if depth > 4:
        raise OcrSortGuardError(
            "OCR_SORT_PROBE_INVALID", "evidence reference is too deeply nested"
        )
    if value is None or type(value) is bool or type(value) is str:
        if type(value) is str and len(value) > 128:
            raise OcrSortGuardError(
                "OCR_SORT_PROBE_INVALID", "evidence reference text is too long"
            )
        return value
    if type(value) is int:
        if value.bit_length() > 4096:
            raise OcrSortGuardError(
                "OCR_SORT_PROBE_INVALID", "evidence reference integer is too large"
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise OcrSortGuardError(
                "OCR_SORT_PROBE_INVALID", "evidence reference contains non-finite data"
            )
        return value
    if isinstance(value, Mapping):
        if len(value) > 16:
            raise OcrSortGuardError(
                "OCR_SORT_PROBE_INVALID", "evidence reference mapping is too large"
            )
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or len(key) > 64:
                raise OcrSortGuardError(
                    "OCR_SORT_PROBE_INVALID", "evidence reference key is invalid"
                )
            frozen[key] = _freeze_json(item, depth=depth + 1)
        result = _FrozenDict(frozen)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            raise OcrSortGuardError(
                "OCR_SORT_PROBE_INVALID", "evidence reference sequence is too large"
            )
        result = tuple(_freeze_json(item, depth=depth + 1) for item in value)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result
    raise OcrSortGuardError(
        "OCR_SORT_PROBE_INVALID", "evidence reference is not JSON native"
    )


def _stable_text(value: Any, name: str, *, max_length: int = 128) -> str:
    if type(value) is not str or not value or len(value) > max_length:
        raise OcrSortGuardError(
            "OCR_SORT_PROBE_INVALID", f"{name} is invalid"
        )
    if not _SAFE_TEXT.fullmatch(value):
        raise OcrSortGuardError(
            "OCR_SORT_PROBE_INVALID", f"{name} is invalid"
        )
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in {int, float}:
        raise OcrSortGuardError(
            "OCR_SORT_PLAN_INVALID", f"{name} must be a finite built-in number"
        )
    try:
        normalized = float(value)
    except OverflowError:
        raise OcrSortGuardError(
            "OCR_SORT_PLAN_INVALID", f"{name} must be a finite built-in number"
        ) from None
    if not math.isfinite(normalized):
        raise OcrSortGuardError(
            "OCR_SORT_PLAN_INVALID", f"{name} must be a finite built-in number"
        )
    return normalized


def _point(value: Any, name: str) -> tuple[float, float, float]:
    if type(value) not in {tuple, list} or len(value) != 3:
        raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", f"{name} is invalid")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


@dataclass(frozen=True)
class OcrSortAction:
    """One immutable plan-derived step consumed by the private runner."""

    action_id: str
    kind: str
    entry_id: str
    target_xyz_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise ValueError("action_id is invalid")
        if self.kind not in _ACTION_KIND_SET:
            raise ValueError("action kind is invalid")
        if type(self.entry_id) is not str or self.entry_id not in _EXPECTED_ENTRY_IDS:
            raise ValueError("entry_id is invalid")
        object.__setattr__(self, "target_xyz_mm", _point(self.target_xyz_mm, "target_xyz_mm"))

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
    identifier: str
    route_id: str
    pick_xyz_mm: tuple[float, float, float]
    drop_xyz_mm: tuple[float, float, float]
    slot_id: str


@dataclass(frozen=True)
class _FrozenPlan:
    plan_id: str
    entries: tuple[_FrozenEntry, ...]
    safe_z_mm: float
    speed_mm_s: float


class OcrSortGuard:
    """Atomic finite-state guard for one host-approved OCR sort plan."""

    def __init__(self) -> None:
        self._state = OcrSortState.EMPTY
        self._plan: _FrozenPlan | None = None
        self._active_entry: _FrozenEntry | None = None
        self._actions: tuple[OcrSortAction, ...] = ()
        self._action_index = 0
        self._consumed: set[str] = set()
        self._evidence: dict[str, _FrozenDict] = {}
        self._run_id: str | None = None
        self._scene_hash: str | None = None
        self._error: _FrozenDict | None = None

    @property
    def state(self) -> str:
        """Return the JSON-native state value, not a mutable implementation object."""

        return self._state.value

    @property
    def state_enum(self) -> OcrSortState:
        """Expose the typed state for host-side code that wants enum identity."""

        return self._state

    @property
    def plan_id(self) -> str | None:
        return None if self._plan is None else self._plan.plan_id

    @property
    def consumed_entry_ids(self) -> tuple[str, ...]:
        if self._plan is None:
            return ()
        return tuple(entry.entry_id for entry in self._plan.entries if entry.entry_id in self._consumed)

    @property
    def active_entry_id(self) -> str | None:
        return None if self._active_entry is None else self._active_entry.entry_id

    @property
    def actions(self) -> tuple[OcrSortAction, ...]:
        return self._actions

    @property
    def current_action(self) -> OcrSortAction | None:
        if self._state is not OcrSortState.ENTRY_ACTIVE:
            return None
        if self._action_index >= len(self._actions):
            return None
        return self._actions[self._action_index]

    @property
    def error(self) -> Mapping[str, Any] | None:
        return self._error

    def activate(self, plan: OcrSortPlan) -> None:
        if self._state is not OcrSortState.EMPTY:
            if self._state is OcrSortState.INVALIDATED:
                self._raise("OCR_SORT_PLAN_INVALIDATED", "sort plan is invalidated")
            self._raise("OCR_SORT_PLAN_ALREADY_ACTIVE", "sort plan is already active")
        frozen = self._freeze_plan(plan)
        self._plan = frozen
        self._active_entry = None
        self._actions = ()
        self._action_index = 0
        self._consumed.clear()
        self._evidence.clear()
        self._run_id = None
        self._scene_hash = None
        self._error = None
        self._state = OcrSortState.ACTIVE

    def begin_entry(self, entry_id: str) -> tuple[OcrSortAction, ...]:
        if self._state is OcrSortState.EMPTY:
            self._raise("OCR_SORT_PLAN_NOT_ACTIVE", "sort plan is not active")
        if self._state is OcrSortState.INVALIDATED:
            self._raise("OCR_SORT_PLAN_INVALIDATED", "sort plan is invalidated")
        if self._state is OcrSortState.COMPLETED:
            self._raise("OCR_SORT_PLAN_COMPLETED", "all sort entries are consumed")
        if self._state is not OcrSortState.ACTIVE or self._plan is None:
            self._raise("OCR_SORT_ENTRY_INVALID", "another entry is already active")
        if type(entry_id) is not str or entry_id not in {entry.entry_id for entry in self._plan.entries}:
            self._raise("OCR_SORT_ENTRY_INVALID", "entry_id is not approved")
        if entry_id in self._consumed:
            self._raise("OCR_SORT_ENTRY_INVALID", "entry_id has already been consumed")
        entry = next(item for item in self._plan.entries if item.entry_id == entry_id)
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
        actions = tuple(
            OcrSortAction(
                action_id=f"{entry.entry_id}:{index}:{kind}",
                kind=kind,
                entry_id=entry.entry_id,
                target_xyz_mm=target,
            )
            for index, (kind, target) in enumerate(zip(_ACTION_KINDS, targets))
        )
        self._active_entry = entry
        self._actions = actions
        self._action_index = 0
        self._state = OcrSortState.ENTRY_ACTIVE
        return actions

    def confirm_action(self, action_id: str | OcrSortAction) -> None:
        if self._state is not OcrSortState.ENTRY_ACTIVE or self._active_entry is None:
            self._raise("OCR_SORT_ACTION_INVALID", "no action is awaiting confirmation")
        expected = self.current_action
        if expected is None:
            self._raise("OCR_SORT_ACTION_INVALID", "no action is awaiting confirmation")
        supplied = action_id.action_id if isinstance(action_id, OcrSortAction) else action_id
        if type(supplied) is not str or supplied != expected.action_id:
            self._raise("OCR_SORT_ACTION_INVALID", "action confirmation is out of order")
        self._action_index += 1
        if self._action_index == len(self._actions):
            self._state = OcrSortState.ENTRY_AWAITING_PROBE

    def confirm_entry_probe(self, entry_id: str, evidence_ref: Mapping[str, Any]) -> None:
        if self._state is not OcrSortState.ENTRY_AWAITING_PROBE or self._active_entry is None:
            self._raise("OCR_SORT_PROBE_INVALID", "entry probe is not expected")
        if type(entry_id) is not str or entry_id != self._active_entry.entry_id:
            self._raise("OCR_SORT_PROBE_INVALID", "probe entry_id is not active")
        normalized = self._validate_probe(evidence_ref, self._active_entry)
        self._evidence[entry_id] = normalized
        self._consumed.add(entry_id)
        self._active_entry = None
        self._actions = ()
        self._action_index = 0
        if self._plan is not None and len(self._consumed) == len(self._plan.entries):
            self._state = OcrSortState.COMPLETED
        else:
            self._state = OcrSortState.ACTIVE

    def fail_entry(self, error: Mapping[str, Any] | str | None = None) -> None:
        if self._error is None:
            if isinstance(error, Mapping):
                try:
                    safe = _freeze_json(dict(error))
                except OcrSortGuardError:
                    safe = _FrozenDict(
                        {
                            "code": "OCR_SORT_ENTRY_FAILED",
                            "message": "entry failure payload is invalid",
                        }
                    )
                if not isinstance(safe, _FrozenDict):
                    safe = _FrozenDict(
                        {"code": "OCR_SORT_ENTRY_FAILED", "message": str(safe)}
                    )
            else:
                message = "entry failed" if error is None else str(error)
                safe = _FrozenDict({"code": "OCR_SORT_ENTRY_FAILED", "message": message[:128]})
            self._error = safe
        self._invalidate()

    def stop(self) -> None:
        if self._state is not OcrSortState.INVALIDATED:
            if self._error is None:
                self._error = _FrozenDict(
                    {"code": "OCR_SORT_STOPPED", "message": "sort execution stopped"}
                )
            self._invalidate()

    def reset(self) -> None:
        self._state = OcrSortState.EMPTY
        self._plan = None
        self._active_entry = None
        self._actions = ()
        self._action_index = 0
        self._consumed.clear()
        self._evidence.clear()
        self._run_id = None
        self._scene_hash = None
        self._error = None

    def snapshot(self) -> Mapping[str, Any]:
        evidence = tuple(
            _FrozenDict({"entry_id": entry_id, "reference": self._evidence[entry_id]})
            for entry_id in self.consumed_entry_ids
            if entry_id in self._evidence
        )
        return _FrozenDict(
            {
                "state": self._state.value,
                "plan_id": self.plan_id,
                "entries": () if self._plan is None else tuple(item.entry_id for item in self._plan.entries),
                "active_entry_id": self.active_entry_id,
                "current_action_id": None if self.current_action is None else self.current_action.action_id,
                "consumed_entry_ids": self.consumed_entry_ids,
                "evidence_refs": evidence,
                "error": self._error,
            }
        )

    def _raise(self, code: str, message: str) -> None:
        raise OcrSortGuardError(code, message)

    def _invalidate(self) -> None:
        self._active_entry = None
        self._actions = ()
        self._action_index = 0
        self._state = OcrSortState.INVALIDATED

    @staticmethod
    def _freeze_plan(plan: Any) -> _FrozenPlan:
        if not isinstance(plan, OcrSortPlan):
            raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "plan must be a complete OcrSortPlan")
        if type(plan.plan_id) is not str or _HEX64.fullmatch(plan.plan_id) is None:
            raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "plan_id is invalid")
        if type(plan.status) is not str or plan.status != "PASS":
            raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "plan status is not PASS")
        if type(plan.entries) is not tuple or len(plan.entries) != 4:
            raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "plan must contain four entries")
        seen: set[str] = set()
        frozen_entries: list[_FrozenEntry] = []
        by_route: dict[str, list[ApprovedOcrSortEntry]] = {}
        for entry in plan.entries:
            if not isinstance(entry, ApprovedOcrSortEntry):
                raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "plan entry type is invalid")
            if entry.entry_id in seen:
                raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "plan entries are not unique")
            seen.add(entry.entry_id)
            by_route.setdefault(entry.route_id, []).append(entry)
        if seen != _EXPECTED_ENTRY_IDS:
            raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "plan entry whitelist is incomplete")
        for expected_entry_id, expected_part_id, expected_identifier, expected_route_id in _EXPECTED_ENTRIES:
            entry = next(item for item in plan.entries if item.entry_id == expected_entry_id)
            if (entry.part_id, entry.identifier, entry.route_id) != (
                expected_part_id,
                expected_identifier,
                expected_route_id,
            ):
                raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "plan route whitelist is invalid")
        safe_z = _number(plan.safe_z_mm, "safe_z_mm")
        speed = _number(plan.speed_mm_s, "speed_mm_s")
        if speed <= 0.0:
            raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "speed_mm_s must be positive")
        for entry in plan.entries:
            pick = _point(entry.pick_xyz_mm, "pick_xyz_mm")
            drop = _point(entry.drop_xyz_mm, "drop_xyz_mm")
            if safe_z <= max(pick[2], drop[2]):
                raise OcrSortGuardError("OCR_SORT_PLAN_INVALID", "safe_z_mm is below a route point")
            frozen_entries.append(
                _FrozenEntry(
                    entry_id=entry.entry_id,
                    part_id=entry.part_id,
                    identifier=entry.identifier,
                    route_id=entry.route_id,
                    pick_xyz_mm=pick,
                    drop_xyz_mm=drop,
                    slot_id="",
                )
            )
        route_positions: dict[str, dict[str, str]] = {}
        for route_id, values in by_route.items():
            ordered = sorted(values, key=lambda item: tuple(float(value) for value in item.drop_xyz_mm))
            route_positions[route_id] = {
                item.entry_id: f"slot_{index + 1}" for index, item in enumerate(ordered)
            }
        with_slots = tuple(
            _FrozenEntry(
                entry_id=item.entry_id,
                part_id=item.part_id,
                identifier=item.identifier,
                route_id=item.route_id,
                pick_xyz_mm=item.pick_xyz_mm,
                drop_xyz_mm=item.drop_xyz_mm,
                slot_id=route_positions[item.route_id][item.entry_id],
            )
            for item in frozen_entries
        )
        return _FrozenPlan(plan.plan_id, with_slots, safe_z, speed)

    def _validate_probe(self, value: Mapping[str, Any], entry: _FrozenEntry) -> _FrozenDict:
        if not isinstance(value, Mapping):
            self._raise("OCR_SORT_PROBE_INVALID", "evidence_ref must be a mapping")
        raw = dict(value)
        if "scene_sha256" in raw and "scene_hash" not in raw:
            raw["scene_hash"] = raw.pop("scene_sha256")
        if "slot" in raw and "slot_id" not in raw:
            raw["slot_id"] = raw.pop("slot")
        if set(raw) != set(_PROBE_FIELDS):
            self._raise("OCR_SORT_PROBE_INVALID", "evidence_ref schema is invalid")
        for field in _PROBE_FIELDS:
            _stable_text(raw[field], field)
        if _HEX64.fullmatch(raw["scene_hash"]) is None:
            self._raise("OCR_SORT_PROBE_INVALID", "scene_hash is invalid")
        if raw["part_id"] != entry.part_id or raw["route_id"] != entry.route_id or raw["slot_id"] != entry.slot_id:
            self._raise("OCR_SORT_PROBE_INVALID", "evidence_ref does not match the active route")
        if self._run_id is not None and (raw["run_id"] != self._run_id or raw["scene_hash"] != self._scene_hash):
            self._raise("OCR_SORT_PROBE_INVALID", "evidence_ref is from another run or scene")
        self._run_id = raw["run_id"]
        self._scene_hash = raw["scene_hash"]
        return _FrozenDict({key: raw[key] for key in ("run_id", "scene_hash", "part_id", "route_id", "slot_id", "evidence_id")})


__all__ = [
    "OcrSortAction",
    "OcrSortGuard",
    "OcrSortGuardError",
    "OcrSortState",
]
