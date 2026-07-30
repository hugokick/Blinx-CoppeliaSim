from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any, Mapping

from vision_platform.calibration.affine import AffineCalibration
from vision_platform.calibration.store import load_calibration
from vision_platform.cameras.factory import create_camera
from vision_platform.config import VisionLabConfig, load_config
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim
from vision_platform.events import EventBus
from vision_platform.models import Detection
from vision_platform.recognition.color_shape import ColorShapeRecognizer
from vision_platform.robot.adapter import RobotAdapter
from vision_platform.robot.coppeliasim_suction import CoppeliaSimSuction
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.robot.tool import BackendPumpTool
from vision_platform.tasks.classify import ClassificationTask, DropZone
from vision_platform.tasks.coppeliasim_verifier import (
    CoppeliaSimPlacementVerifier,
)


class SceneAwareRecognizer:
    """Associate image detections with unplaced CoppeliaSim objects.

    The translucent teaching zones are deliberately visible in the camera.
    After an object is released into a zone it also remains visible, so a
    plain image recognizer would select it again.  This adapter projects each
    detection into world XY, associates it with the nearest scene object and
    excludes objects whose centers are already inside any drop zone.
    """

    def __init__(
        self,
        *,
        recognizer: ColorShapeRecognizer,
        calibration: AffineCalibration,
        sim: Any,
        pickables_path: str,
        zones: Mapping[str, Mapping[str, Any]],
        max_match_distance_mm: float = 15.0,
    ) -> None:
        if max_match_distance_mm <= 0:
            raise ValueError("max_match_distance_mm must be positive")
        self.recognizer = recognizer
        self.calibration = calibration
        self.sim = sim
        self.pickables_path = str(pickables_path)
        self.zones = {
            str(name): dict(specification)
            for name, specification in zones.items()
        }
        self.max_match_distance_mm = float(max_match_distance_mm)
        self._pickables_handle: int | None = None

    def detect(self, image_bgr):
        detections = self.recognizer.detect(image_bgr)
        candidates = self._unplaced_objects()
        if not candidates:
            return []

        pairs: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            world_xy = self.calibration.pixel_to_world(detection.center_px)
            for object_handle, object_xy in candidates.items():
                distance_mm = (
                    (world_xy[0] - object_xy[0]) ** 2
                    + (world_xy[1] - object_xy[1]) ** 2
                ) ** 0.5
                if distance_mm <= self.max_match_distance_mm:
                    pairs.append(
                        (float(distance_mm), detection_index, object_handle)
                    )

        matched_detections: set[int] = set()
        matched_objects: set[int] = set()
        output: list[Detection] = []
        for distance_mm, detection_index, object_handle in sorted(pairs):
            if (
                detection_index in matched_detections
                or object_handle in matched_objects
            ):
                continue
            detection = detections[detection_index]
            metadata = dict(detection.metadata)
            metadata.update(
                {
                    "object_handle": object_handle,
                    "scene_match_distance_mm": distance_mm,
                }
            )
            output.append(
                replace(
                    detection,
                    detection_id=f"sim-object-{object_handle}",
                    metadata=metadata,
                )
            )
            matched_detections.add(detection_index)
            matched_objects.add(object_handle)
        output.sort(key=lambda item: (item.center_px[1], item.center_px[0]))
        return output

    def annotate(self, image_bgr, detections):
        return self.recognizer.annotate(image_bgr, detections)

    def _unplaced_objects(self) -> dict[int, tuple[float, float]]:
        if self._pickables_handle is None:
            self._pickables_handle = int(self.sim.getObject(self.pickables_path))
        handles = self.sim.getObjectsInTree(
            self._pickables_handle,
            self.sim.handle_all,
            1,
        )
        candidates: dict[int, tuple[float, float]] = {}
        for raw_handle in handles:
            handle = int(raw_handle)
            try:
                position = self.sim.getObjectPosition(handle, -1)
            except Exception:
                continue
            world_mm = (
                float(position[0]) * 1000.0,
                float(position[1]) * 1000.0,
            )
            if not self._inside_any_zone(world_mm):
                candidates[handle] = world_mm
        return candidates

    def _inside_any_zone(self, point_mm: tuple[float, float]) -> bool:
        for specification in self.zones.values():
            center = specification["center_mm"]
            size = specification["size_mm"]
            if (
                abs(point_mm[0] - float(center[0])) <= float(size[0]) / 2.0
                and abs(point_mm[1] - float(center[1]))
                <= float(size[1]) / 2.0
            ):
                return True
        return False


class VisionLabApplication:
    """Shared application composition root for CLI and teaching UIs."""

    def __init__(
        self,
        *,
        config: VisionLabConfig,
        camera: Any,
        base_recognizer: ColorShapeRecognizer,
        robot_backend: Any,
        robot: RobotAdapter,
        tool: Any,
        verifier: Any,
        event_bus: EventBus,
        workspace: WorkspacePolicy,
        zones: Mapping[str, DropZone],
        scene_spec: Mapping[str, Any],
        sim: Any | None,
        client: Any | None,
        owns_client: bool,
        calibration: AffineCalibration | None,
    ) -> None:
        self.config = config
        self.camera = camera
        self.base_recognizer = base_recognizer
        self.robot_backend = robot_backend
        self.robot = robot
        self.tool = tool
        self.verifier = verifier
        self.event_bus = event_bus
        self.workspace = workspace
        self.zones = dict(zones)
        self.scene_spec = dict(scene_spec)
        self.sim = sim
        self.client = client
        self.owns_client = bool(owns_client)
        self.calibration = calibration
        self.recognizer: Any = base_recognizer
        self._opened = False
        self._closed = False
        if calibration is not None:
            self.set_calibration(calibration)

    @classmethod
    def from_config(
        cls,
        config: VisionLabConfig | str | Path | None = None,
        *,
        sim: Any | None = None,
        client: Any | None = None,
        calibration: AffineCalibration | None = None,
    ) -> "VisionLabApplication":
        selected = (
            config
            if isinstance(config, VisionLabConfig)
            else load_config(config)
        )
        needs_sim = (
            selected.camera_backend == "sim"
            or selected.robot_backend == "sim"
        )
        owns_client = False
        if needs_sim and sim is None:
            if client is None:
                from coppeliasim_zmqremoteapi_client import RemoteAPIClient

                client = RemoteAPIClient(
                    host=selected.coppelia_host,
                    port=selected.coppelia_port,
                )
                owns_client = True
            sim = (
                client.require("sim")
                if hasattr(client, "require")
                else client.getObject("sim")
            )

        workspace = WorkspacePolicy(
            x_mm=selected.workspace.x_mm,
            y_mm=selected.workspace.y_mm,
            z_mm=selected.workspace.z_mm,
            safe_z_mm=selected.workspace.safe_z_mm,
        )
        scene_spec_path = (
            selected.project_root
            / "simulation"
            / "vision_lab"
            / "scene_spec.json"
        )
        scene_spec = json.loads(scene_spec_path.read_text(encoding="utf-8"))
        zones = {
            name: DropZone(
                name=name,
                position_mm=tuple(float(value) for value in value["center_mm"]),
                metadata={
                    "path": value["path"],
                    "size_mm": list(value["size_mm"]),
                },
            )
            for name, value in scene_spec["zones"].items()
        }

        camera_options = selected.camera_options.get(
            selected.camera_backend,
            {},
        )
        camera = create_camera(
            selected.camera_backend,
            camera_options,
            sim=sim,
            client=client,
        )
        recognizer = ColorShapeRecognizer(
            min_area_ratio=float(
                selected.recognition.get("min_area_ratio", 0.002)
            ),
            max_area_ratio=float(
                selected.recognition.get("max_area_ratio", 0.25)
            ),
        )

        if selected.robot_backend == "sim":
            from config.backend import get_backend_settings
            from robot_backends.coppeliasim_robot import (
                CoppeliaSimRobotBackend,
            )

            backend_settings = dict(get_backend_settings()["sim"])
            backend_settings.update(
                {
                    "host": selected.coppelia_host,
                    "port": selected.coppelia_port,
                    "scene": str(selected.coppelia_scene),
                }
            )
            teaching_ready_pose = (
                scene_spec.get("robot_visuals", {})
                .get("teaching_ready_pose")
            )
            if teaching_ready_pose is not None:
                backend_settings["home_angles"] = [
                    float(value)
                    for value in teaching_ready_pose["joint_angles_deg"]
                ]
            robot_backend = CoppeliaSimRobotBackend(
                settings=backend_settings,
                sim_client=client,
            )
            tool = CoppeliaSimSuction(sim=sim)
            verifier = CoppeliaSimPlacementVerifier(
                sim=sim,
                zone_paths={
                    name: value["path"]
                    for name, value in scene_spec["zones"].items()
                },
            )
        else:
            from robot_backends.real_robot import RealRobotBackend

            robot_backend = RealRobotBackend()
            tool = BackendPumpTool(robot_backend)
            verifier = None

        if calibration is None:
            calibration_path = cls._resolve_project_path(
                selected,
                selected.calibration.get(
                    "store",
                    "artifacts/vision_lab/calibration.json",
                ),
            )
            if calibration_path.is_file():
                calibration = load_calibration(calibration_path)

        return cls(
            config=selected,
            camera=camera,
            base_recognizer=recognizer,
            robot_backend=robot_backend,
            robot=RobotAdapter(robot_backend, policy=workspace),
            tool=tool,
            verifier=verifier,
            event_bus=EventBus(),
            workspace=workspace,
            zones=zones,
            scene_spec=scene_spec,
            sim=sim,
            client=client,
            owns_client=owns_client,
            calibration=calibration,
        )

    @staticmethod
    def _resolve_project_path(
        config: VisionLabConfig,
        value: str | Path,
    ) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config.project_root / path
        return path.resolve()

    @property
    def calibration_store_path(self) -> Path:
        return self._resolve_project_path(
            self.config,
            self.config.calibration.get(
                "store",
                "artifacts/vision_lab/calibration.json",
            ),
        )

    def open(self) -> None:
        if self._closed:
            raise RuntimeError("VisionLabApplication is closed")
        if self._opened:
            return
        if (
            self.sim is not None
            and int(self.sim.getSimulationState())
            == int(self.sim.simulation_stopped)
        ):
            self.sim.startSimulation()
            deadline = time.monotonic() + 5.0
            while (
                int(self.sim.getSimulationState())
                == int(self.sim.simulation_stopped)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            if (
                int(self.sim.getSimulationState())
                == int(self.sim.simulation_stopped)
            ):
                raise RuntimeError("CoppeliaSim simulation did not start")
            # Let the saved CameraRenderScope child script initialise its
            # render collection before the first teaching frame is requested.
            time.sleep(0.2)
        self.camera.open()
        self.robot.initialize()
        self._opened = True

    def set_calibration(self, calibration: AffineCalibration) -> None:
        self.calibration = calibration
        if self.sim is not None and self.config.camera_backend == "sim":
            self.recognizer = SceneAwareRecognizer(
                recognizer=self.base_recognizer,
                calibration=calibration,
                sim=self.sim,
                pickables_path="/VisionLab/Pickables",
                zones=self.scene_spec["zones"],
            )
        else:
            self.recognizer = self.base_recognizer

    def create_classification_task(
        self,
        *,
        cancel_event: Event | None = None,
        step_waiter=None,
    ) -> ClassificationTask:
        if self.calibration is None:
            raise RuntimeError("Classification requires a completed calibration")
        if self.verifier is None:
            raise RuntimeError(
                "Real-hardware placement verification is not configured"
            )
        return ClassificationTask(
            camera=self.camera,
            recognizer=self.recognizer,
            calibration=self.calibration,
            robot=self.robot,
            tool=self.tool,
            verifier=self.verifier,
            zones=self.zones,
            workspace=self.workspace,
            classification_key=str(
                self.config.task.get("classification_key", "color")
            ),
            speed=float(self.config.task.get("speed", 15)),
            cancel_event=cancel_event,
            event_bus=self.event_bus,
            step_waiter=step_waiter,
        )

    def load_and_start_scene(self, scene_path: str | Path | None = None) -> None:
        if self.sim is None:
            raise RuntimeError("CoppeliaSim is not connected")
        selected = Path(scene_path or self.config.coppelia_scene).resolve()
        if int(self.sim.getSimulationState()) != int(self.sim.simulation_stopped):
            self.sim.stopSimulation()
            deadline = time.monotonic() + 5.0
            while (
                int(self.sim.getSimulationState())
                != int(self.sim.simulation_stopped)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
        if int(self.sim.getSimulationState()) != int(self.sim.simulation_stopped):
            raise RuntimeError("CoppeliaSim did not stop before scene loading")
        staged = stage_scene_for_coppeliasim(selected)
        self.sim.loadScene(staged.as_posix())
        self.sim.startSimulation()
        deadline = time.monotonic() + 5.0
        while (
            int(self.sim.getSimulationState())
            == int(self.sim.simulation_stopped)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if int(self.sim.getSimulationState()) == int(self.sim.simulation_stopped):
            raise RuntimeError("CoppeliaSim scene did not start")
        time.sleep(0.2)

    def close(self) -> None:
        if self._closed:
            return
        try:
            try:
                self.tool.off()
            except Exception:
                pass
            try:
                self.camera.close()
            finally:
                self.robot.close()
        finally:
            try:
                if self.owns_client and self.client is not None:
                    close = getattr(self.client, "close", None)
                    if callable(close):
                        try:
                            close()
                        except BaseException as public_close_error:
                            try:
                                self._close_client_transport_locally(
                                    self.client
                                )
                            except BaseException as local_close_error:
                                raise public_close_error from local_close_error
                            raise
                    else:
                        self._close_client_transport_locally(self.client)
            finally:
                self.sim = None
                self.client = None
                self._opened = False
                self._closed = True

    def close_quarantined(self) -> None:
        """Close local resources without sending on an unusable REQ socket."""
        if self._closed:
            return
        try:
            try:
                self.camera.close()
            finally:
                self.robot.close()
        finally:
            try:
                if self.owns_client and self.client is not None:
                    self._close_client_transport_locally(self.client)
            finally:
                self.sim = None
                self.client = None
                self._opened = False
                self._closed = True

    @staticmethod
    def _close_client_transport_locally(client: Any) -> None:
        socket = getattr(client, "socket", None)
        context = getattr(client, "context", None)
        try:
            if socket is not None:
                close_socket = getattr(socket, "close", None)
                if callable(close_socket):
                    close_socket(linger=0)
        finally:
            if context is not None:
                terminate_context = getattr(context, "term", None)
                if callable(terminate_context):
                    terminate_context()

    def __enter__(self) -> "VisionLabApplication":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
