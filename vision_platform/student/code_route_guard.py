"""Host-only finite-state guard for V1-07 route execution."""

from __future__ import annotations

import math
from typing import Any

from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.code_routing import ApprovedCodeRoute, CodeRoutePlan


class CodeRouteGuard:
    """Allow only the finite trajectory of one host-approved route plan."""

    def __init__(self) -> None:
        self._plan: CodeRoutePlan | None = None
        self._state = "INACTIVE"
        self._selected: ApprovedCodeRoute | None = None
        self._completed: set[str] = set()
        self._failed = False

    def activate(self, plan: CodeRoutePlan) -> None:
        if self._plan is not None or not isinstance(plan, CodeRoutePlan):
            raise VisionPlatformError(
                "CODE_ROUTE_PLAN_ALREADY_ACTIVE", "路线计划已经激活或无效"
            )
        self._plan = plan
        self._state = "READY"

    def reset(self) -> None:
        self._plan = None
        self._state = "INACTIVE"
        self._selected = None
        self._completed.clear()
        self._failed = False

    @staticmethod
    def _near(left: Any, right: Any, tolerance: float) -> bool:
        try:
            left_values = tuple(float(value) for value in left)
            right_values = tuple(float(value) for value in right)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            len(left_values) == 3
            and len(right_values) == 3
            and all(math.isfinite(value) for value in left_values + right_values)
            and all(abs(a - b) <= tolerance for a, b in zip(left_values, right_values))
        )

    def _reject(self, message: str) -> None:
        self._failed = True
        raise VisionPlatformError("CODE_ROUTE_SEQUENCE_INVALID", message)

    def _required(self) -> CodeRoutePlan:
        if self._plan is None:
            raise VisionPlatformError("CODE_ROUTE_GUARD_INACTIVE", "路线计划尚未激活")
        if self._failed:
            raise VisionPlatformError(
                "CODE_ROUTE_SEQUENCE_INVALID", "route guard failure is latched"
            )
        return self._plan

    def validate_move(
        self,
        current_xyz_mm: tuple[float, float, float],
        target_xyz_mm: tuple[float, float, float],
    ) -> None:
        plan = self._required()
        if not self._near(current_xyz_mm, current_xyz_mm, 0.0) or not self._near(target_xyz_mm, target_xyz_mm, 0.0):
            self._reject("坐标必须是三个有限值")
        if self._state == "READY":
            safe_lift = (current_xyz_mm[0], current_xyz_mm[1], plan.safe_z_mm + 1.0)
            if self._near(target_xyz_mm, safe_lift, 0.25):
                return
            for entry in plan.entries:
                pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], plan.safe_z_mm + 1.0)
                if entry.entry_id not in self._completed and self._near(target_xyz_mm, pick_hover, 0.25):
                    current_hover = (current_xyz_mm[0], current_xyz_mm[1], plan.safe_z_mm + 1.0)
                    if not self._near(current_xyz_mm, current_hover, 1.0):
                        self._reject("横向移动到抓取悬停点前必须先到安全高度")
                    self._selected = entry
                    self._state = "AT_PICK_HOVER"
                    return
            self._reject("READY 只允许安全抬升或未完成构件的抓取悬停点")
        entry = self._selected
        if entry is None:
            self._reject("没有选中的路线条目")
        pick_hover = (entry.pick_xyz_mm[0], entry.pick_xyz_mm[1], plan.safe_z_mm + 1.0)
        drop_hover = (entry.drop_xyz_mm[0], entry.drop_xyz_mm[1], plan.safe_z_mm + 1.0)
        transitions = {
            "AT_PICK_HOVER": (pick_hover, entry.pick_xyz_mm, "AT_PICK"),
            "ATTACHED": (entry.pick_xyz_mm, pick_hover, "LIFTED"),
            "LIFTED": (pick_hover, drop_hover, "AT_DROP_HOVER"),
            "AT_DROP_HOVER": (drop_hover, entry.drop_xyz_mm, "AT_DROP"),
            "RELEASED": (entry.drop_xyz_mm, drop_hover, "COMPLETE"),
        }
        if self._state not in transitions:
            self._reject(f"状态 {self._state} 不允许移动")
        expected_current, expected_target, next_state = transitions[self._state]
        if not self._near(current_xyz_mm, expected_current, 1.0) or not self._near(target_xyz_mm, expected_target, 0.25):
            self._reject(f"状态 {self._state} 的当前位置或目标不匹配")
        if next_state == "COMPLETE":
            self._completed.add(entry.entry_id)
            self._selected = None
            self._state = "READY"
        else:
            self._state = next_state

    def validate_tool_on(self, pose_xyz_mm: tuple[float, float, float]) -> None:
        self._required()
        if self._state != "AT_PICK" or self._selected is None or not self._near(pose_xyz_mm, self._selected.pick_xyz_mm, 1.0):
            self._reject("吸盘只能在当前条目的抓取点开启")
        self._state = "ATTACHED"

    def validate_tool_off(self, pose_xyz_mm: tuple[float, float, float]) -> bool:
        if self._plan is None:
            return True
        if self._state == "READY" and len(self._completed) == len(self._plan.entries):
            return True
        if self._state == "AT_DROP" and self._selected is not None and self._near(pose_xyz_mm, self._selected.drop_xyz_mm, 1.0):
            self._state = "RELEASED"
            return True
        self._failed = True
        return False

    def validate_home(self) -> None:
        plan = self._required()
        if self._state != "READY" or len(self._completed) != len(plan.entries):
            self._reject("全部路线完成前禁止学生程序回零")

    def completion(self) -> dict[str, Any]:
        count = 0 if self._plan is None else len(self._plan.entries)
        return {
            "active": self._plan is not None,
            "state": self._state,
            "selected_entry_id": None if self._selected is None else self._selected.entry_id,
            "completed_entry_ids": sorted(self._completed),
            "all_complete": count > 0 and len(self._completed) == count and not self._failed,
            "failed": self._failed,
        }
