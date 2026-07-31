from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Any

from vision_platform.errors import SuctionError
from vision_platform.robot.tool import AttachmentEvidence


@dataclass(frozen=True)
class _Attachment:
    evidence: AttachmentEvidence
    original_parent: int
    original_static: int | None
    world_pose: tuple[float, ...] | None


class CoppeliaSimSuction:
    """Attach the nearest eligible scene object to the BLX suction TCP."""

    def __init__(
        self,
        sim: Any,
        *,
        tcp_path: str = "/BLX_tool_suction",
        pickables_path: str = "/VisionLab/Pickables",
        max_attach_distance_m: float = 0.035,
    ) -> None:
        if max_attach_distance_m <= 0:
            raise ValueError("max_attach_distance_m must be positive")
        self.sim = sim
        self.tcp_path = str(tcp_path)
        self.pickables_path = str(pickables_path)
        self.max_attach_distance_m = float(max_attach_distance_m)
        self._tcp_handle: int | None = None
        self._pickables_handle: int | None = None
        self._attachment: _Attachment | None = None

    def validate(self) -> None:
        """Resolve required scene handles without attaching an object."""
        self._resolve_handles()

    def on(self) -> AttachmentEvidence:
        if self._attachment is not None:
            return self._attachment.evidence
        self.validate()
        assert self._tcp_handle is not None
        assert self._pickables_handle is not None

        tcp_position = self.sim.getObjectPosition(self._tcp_handle, -1)
        candidates = self.sim.getObjectsInTree(
            self._pickables_handle,
            self.sim.handle_all,
            1,
        )
        distances: list[tuple[float, int]] = []
        for candidate in candidates:
            try:
                position = self.sim.getObjectPosition(candidate, -1)
                distances.append((float(dist(tcp_position, position)), int(candidate)))
            except Exception:
                continue
        if not distances:
            raise SuctionError(
                "No eligible objects exist under the pickables collection",
                pickables_path=self.pickables_path,
            )
        distance_m, object_handle = min(
            distances,
            key=lambda item: (item[0], item[1]),
        )
        if distance_m > self.max_attach_distance_m:
            raise SuctionError(
                (
                    f"Nearest pickable is {distance_m:.4f} m from the TCP, "
                    f"outside {self.max_attach_distance_m:.4f} m tolerance"
                ),
                nearest_distance_m=distance_m,
                max_attach_distance_m=self.max_attach_distance_m,
            )

        original_parent = int(self.sim.getObjectParent(object_handle))
        original_static = self._get_static(object_handle)
        world_pose = self._get_world_pose(object_handle)
        if original_static is not None:
            self._set_static(object_handle, 1)
        self.sim.setObjectParent(
            object_handle,
            self._tcp_handle,
            True,
        )
        evidence = AttachmentEvidence(
            attached=True,
            object_handle=object_handle,
            distance_m=distance_m,
            metadata={
                "tcp_path": self.tcp_path,
                "pickables_path": self.pickables_path,
                "original_parent": original_parent,
                "original_static": original_static,
                "world_pose_before": list(world_pose) if world_pose else None,
            },
        )
        self._attachment = _Attachment(
            evidence=evidence,
            original_parent=original_parent,
            original_static=original_static,
            world_pose=world_pose,
        )
        return evidence

    def off(self) -> None:
        attachment = self._attachment
        if attachment is None:
            return
        object_handle = attachment.evidence.object_handle
        assert object_handle is not None
        self.sim.setObjectParent(
            object_handle,
            attachment.original_parent,
            True,
        )
        if attachment.original_static is not None:
            self._set_static(object_handle, attachment.original_static)
        reset = getattr(self.sim, "resetDynamicObject", None)
        if callable(reset):
            reset(object_handle)
        self._attachment = None

    def is_attached(self) -> bool:
        return self._attachment is not None

    def _resolve_handles(self) -> None:
        if self._tcp_handle is None:
            self._tcp_handle = int(self.sim.getObject(self.tcp_path))
        if self._pickables_handle is None:
            self._pickables_handle = int(self.sim.getObject(self.pickables_path))

    def _get_static(self, object_handle: int) -> int | None:
        parameter = getattr(self.sim, "shapeintparam_static", None)
        getter = getattr(self.sim, "getObjectInt32Param", None)
        if parameter is None or not callable(getter):
            return None
        try:
            return int(getter(object_handle, parameter))
        except Exception:
            return None

    def _set_static(self, object_handle: int, value: int) -> None:
        parameter = getattr(self.sim, "shapeintparam_static", None)
        setter = getattr(self.sim, "setObjectInt32Param", None)
        if parameter is None or not callable(setter):
            return
        setter(object_handle, parameter, int(value))

    def _get_world_pose(self, object_handle: int) -> tuple[float, ...] | None:
        getter = getattr(self.sim, "getObjectPose", None)
        if not callable(getter):
            return None
        try:
            return tuple(float(value) for value in getter(object_handle, -1))
        except Exception:
            return None
