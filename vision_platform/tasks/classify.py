from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Mapping, Protocol
from uuid import uuid4

from vision_platform.errors import (
    DetectionError,
    SuctionError,
    TaskCancelledError,
    VisionPlatformError,
)
from vision_platform.events import EventBus
from vision_platform.models import Detection, TaskResult
from vision_platform.robot.safety import Point3, WorkspacePolicy
from vision_platform.tasks.state_machine import TaskStateMachine


@dataclass(frozen=True)
class DropZone:
    name: str
    position_mm: Point3
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementEvidence:
    ok: bool
    expected_zone: str
    object_ref: Any
    details: Mapping[str, Any] = field(default_factory=dict)


class PlacementVerifier(Protocol):
    def verify(
        self,
        object_ref: Any,
        expected_zone: str,
    ) -> PlacementEvidence:
        ...


class ClassificationTask:
    def __init__(
        self,
        *,
        camera: Any,
        recognizer: Any,
        calibration: Any,
        robot: Any,
        tool: Any,
        verifier: PlacementVerifier,
        zones: Mapping[str, DropZone],
        workspace: WorkspacePolicy,
        classification_key: str = "color",
        speed: float = 15,
        cancel_event: Event | None = None,
        event_bus: EventBus | None = None,
        step_waiter=None,
    ) -> None:
        if classification_key not in {"color", "shape"}:
            raise ValueError("classification_key must be color or shape")
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.camera = camera
        self.recognizer = recognizer
        self.calibration = calibration
        self.robot = robot
        self.tool = tool
        self.verifier = verifier
        self.zones = dict(zones)
        self.workspace = workspace
        self.classification_key = classification_key
        self.speed = float(speed)
        self.cancel_event = cancel_event or Event()
        self.event_bus = event_bus or EventBus()
        self.step_waiter = step_waiter

    def run(self, *, max_objects: int = 1, task_id: str | None = None) -> TaskResult:
        if max_objects < 1:
            raise ValueError("max_objects must be at least one")
        selected_task_id = task_id or f"classify-{uuid4().hex[:12]}"
        machine = TaskStateMachine(
            event_bus=self.event_bus,
            step_waiter=self.step_waiter,
        )
        metrics = {
            "objects_completed": 0,
            "attach_success": 0,
            "release_success": 0,
            "correct_zone": 0,
            "frames_acquired": 0,
        }
        processed: set[Any] = set()
        last_position: Point3 | None = None
        machine.transition("IDLE", "分类任务就绪")

        try:
            for index in range(max_objects):
                self._check_cancel()
                machine.transition(
                    "ACQUIRE",
                    f"采集第 {index + 1} 个目标图像",
                    data={"object_index": index + 1},
                )
                frame = self.camera.read()
                metrics["frames_acquired"] += 1

                self._check_cancel()
                machine.transition(
                    "DETECT",
                    "识别颜色与形状",
                    data={"sequence_id": frame.sequence_id},
                )
                detections = self.recognizer.detect(frame.image_bgr)
                detection = self._select_detection(detections, processed)

                self._check_cancel()
                self.calibration.validate_image_size((frame.width, frame.height))
                world_xy = self.calibration.pixel_to_world(detection.center_px)
                pick = self.workspace.validate(
                    (world_xy[0], world_xy[1], self.calibration.plane_z_mm)
                )
                category = str(getattr(detection, self.classification_key))
                if category not in self.zones:
                    raise VisionPlatformError(
                        "CLASSIFICATION_ZONE_MISSING",
                        f"No drop zone is configured for {category}",
                        details={"category": category},
                    )
                zone = self.zones[category]
                drop = self.workspace.validate(zone.position_mm)
                machine.transition(
                    "TRANSFORM",
                    "像素坐标已转换为机器人世界坐标",
                    data={
                        "detection_id": detection.detection_id,
                        "pixel": list(detection.center_px),
                        "world_xy_mm": [float(world_xy[0]), float(world_xy[1])],
                        "world_point_mm": [
                            float(pick[0]),
                            float(pick[1]),
                            float(pick[2]),
                        ],
                        "color": detection.color,
                        "shape": detection.shape,
                        "confidence": float(detection.confidence),
                        "category": category,
                        "zone": zone.name,
                    },
                )

                above_pick = (pick[0], pick[1], float(self.workspace.safe_z_mm))
                above_drop = (drop[0], drop[1], float(self.workspace.safe_z_mm))

                self._check_cancel()
                machine.transition(
                    "APPROACH",
                    "移动到目标上方安全高度",
                    data={"target_mm": list(above_pick)},
                )
                self._move(above_pick)
                last_position = above_pick

                self._check_cancel()
                machine.transition(
                    "DESCEND",
                    "垂直下降到抓取高度",
                    data={"target_mm": list(pick)},
                )
                self._move(pick)
                last_position = pick

                self._check_cancel()
                machine.transition("ATTACH", "启动吸盘并验证附着")
                attachment = self.tool.on()
                if not attachment.attached or not self.tool.is_attached():
                    raise SuctionError(
                        "Suction tool did not attach an eligible object",
                        detection_id=detection.detection_id,
                    )
                metrics["attach_success"] += 1

                self._check_cancel()
                machine.transition(
                    "LIFT",
                    "附着成功，垂直抬升",
                    data={"target_mm": list(above_pick)},
                )
                self._move(above_pick)
                last_position = above_pick

                self._check_cancel()
                machine.transition(
                    "TRANSFER",
                    f"搬运到分类区 {zone.name}",
                    data={
                        "safe_target_mm": list(above_drop),
                        "drop_target_mm": list(drop),
                    },
                )
                self._move(above_drop)
                last_position = above_drop
                self._move(drop)
                last_position = drop

                self._check_cancel()
                machine.transition("RELEASE", "关闭吸盘并释放对象")
                self.tool.off()
                metrics["release_success"] += 1
                self._move(above_drop)
                last_position = above_drop

                self._check_cancel()
                machine.transition(
                    "VERIFY",
                    f"验证对象是否进入分类区 {zone.name}",
                )
                object_ref = (
                    attachment.object_handle
                    if attachment.object_handle is not None
                    else detection.metadata.get("object_handle", detection)
                )
                evidence = self.verifier.verify(object_ref, zone.name)
                if not evidence.ok:
                    raise VisionPlatformError(
                        "PLACEMENT_VERIFICATION_FAILED",
                        f"Object was not verified in zone {zone.name}",
                        details=dict(evidence.details),
                    )
                metrics["correct_zone"] += 1
                metrics["objects_completed"] += 1
                processed.add(self._detection_key(detection))

            machine.transition(
                "COMPLETE",
                f"分类完成：{metrics['objects_completed']} 个对象",
                data={"metrics": dict(metrics)},
            )
            return TaskResult(
                task_id=selected_task_id,
                status="PASS",
                events=tuple(machine.events),
                metrics=metrics,
            )
        except Exception as error:
            self._safe_stop(last_position)
            code = getattr(error, "code", "TASK_FAILED")
            machine.transition(
                "SAFE_STOP",
                f"任务安全停止：{error}",
                error_code=code,
            )
            return TaskResult(
                task_id=selected_task_id,
                status=(
                    "CANCELLED"
                    if isinstance(error, TaskCancelledError)
                    else "FAIL"
                ),
                events=tuple(machine.events),
                metrics=metrics,
                error_code=code,
                error_message=str(error),
            )

    def _move(self, point: Point3) -> None:
        self.robot.move_world(*point, speed=self.speed)

    def _safe_stop(self, last_position: Point3 | None) -> None:
        try:
            self.tool.off()
        except Exception:
            pass
        if last_position is None:
            return
        safe = (
            last_position[0],
            last_position[1],
            float(self.workspace.safe_z_mm),
        )
        try:
            self.workspace.validate(safe)
            self._move(safe)
        except Exception:
            pass

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise TaskCancelledError()

    @staticmethod
    def _detection_key(detection: Detection) -> Any:
        handle = detection.metadata.get("object_handle")
        if handle is not None:
            return ("handle", handle)
        return (
            detection.color,
            detection.shape,
            round(detection.center_px[0], 1),
            round(detection.center_px[1], 1),
        )

    def _select_detection(
        self,
        detections: list[Detection],
        processed: set[Any],
    ) -> Detection:
        candidates = [
            detection
            for detection in detections
            if self._detection_key(detection) not in processed
        ]
        if not candidates:
            raise DetectionError(
                "DETECTION_EMPTY",
                "No unprocessed object was detected",
            )
        return min(
            candidates,
            key=lambda detection: (
                -detection.confidence,
                detection.detection_id,
            ),
        )
