from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import threading
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vision_platform.student.experiment_gateway as gateway_module
import vision_platform.student.runner as runner_module
import vision_platform.vision_quality.controller as profile_controller_module
from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.errors import VisionPlatformError
from vision_platform.experiments.models import (
    ExperimentAcceptance,
    ExperimentDefinition,
    ExperimentRunContext,
)
from vision_platform.models import Frame
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.student.protocol import CommandMessage, RunState
from vision_platform.student.runner import (
    StudentProgramController,
    StudentRunResult,
    StudentRunSnapshot,
)
from vision_platform.student.safety import StudentExecutionPolicy
from vision_platform.vision_quality.models import (
    VisionProfile,
    VisionProfileCatalog,
)


class FakeRobot:
    def __init__(
        self,
        actions: list | None = None,
        *,
        move_entered: threading.Event | None = None,
        move_release: threading.Event | None = None,
        fail_move: bool = False,
        move_error: BaseException | None = None,
        fail_home: bool = False,
        move_duration_s: float = 0.0,
        pose=(100.0, 20.0, 120.0),
        timeout_probe=None,
    ):
        self.current = pose
        self.moves: list[tuple[float, float, float, float]] = []
        self.pose_calls = 0
        self.home_calls = 0
        self.actions = actions if actions is not None else []
        self.move_entered = move_entered
        self.move_release = move_release
        self.fail_move = fail_move
        self.move_error = move_error
        self.fail_home = fail_home
        self.move_duration_s = move_duration_s
        self.timeout_probe = timeout_probe
        self.timeout_observations: list[tuple[str, tuple]] = []

    def current_world_pose(self):
        self.pose_calls += 1
        self._record_timeouts("robot.current_world_pose")
        return self.current

    def move_world(self, x, y, z, *, speed):
        self._record_timeouts("robot.move_world")
        self.actions.append(("move", float(x), float(y), float(z)))
        if self.move_entered is not None:
            self.move_entered.set()
        if self.move_release is not None:
            if not self.move_release.wait(timeout=5):
                raise TimeoutError("test move release timed out")
        deadline = time.monotonic() + self.move_duration_s
        while time.monotonic() < deadline:
            threading.Event().wait(
                min(0.01, max(0.0, deadline - time.monotonic()))
            )
        if self.move_error is not None:
            raise self.move_error
        if self.fail_move:
            raise RuntimeError("backend-move-failed")
        self.current = (float(x), float(y), float(z))
        self.moves.append((*self.current, float(speed)))

    def move_home(self):
        self._record_timeouts("robot.move_home")
        self.actions.append(("home",))
        self.home_calls += 1
        if self.fail_home:
            raise RuntimeError("home-cleanup-failed")
        self.current = (100.0, 20.0, 120.0)

    def _record_timeouts(self, stage: str) -> None:
        if self.timeout_probe is not None:
            self.timeout_observations.append((stage, self.timeout_probe()))


class FakeTool:
    def __init__(
        self,
        actions: list | None = None,
        *,
        fail_off: bool = False,
        off_error: BaseException | None = None,
        alive_probe=None,
        timeout_probe=None,
        off_hook=None,
    ):
        self.on_calls = 0
        self.off_calls = 0
        self.actions = actions if actions is not None else []
        self.fail_off = fail_off
        self.off_error = off_error
        self.alive_probe = alive_probe
        self.timeout_probe = timeout_probe
        self.off_hook = off_hook
        self.cleanup_liveness: list[bool] = []
        self.timeout_observations: list[tuple] = []

    def on(self):
        self.actions.append(("tool.on",))
        self.on_calls += 1

    def off(self):
        self.actions.append(("tool.off",))
        self.off_calls += 1
        if self.timeout_probe is not None:
            self.timeout_observations.append(self.timeout_probe())
        if self.off_hook is not None:
            self.off_hook()
        if self.alive_probe is not None:
            self.cleanup_liveness.append(bool(self.alive_probe()))
        if self.off_error is not None:
            raise self.off_error
        if self.fail_off:
            raise RuntimeError("tool-cleanup-failed")


class ProtocolSocket:
    def __init__(
        self,
        *,
        rcvtimeo: int = 5000,
        sndtimeo: int = 7000,
        fail_on_set: str | None = None,
        fail_on_first_restore: str | None = None,
    ) -> None:
        self._rcvtimeo = rcvtimeo
        self._sndtimeo = sndtimeo
        self._originals = {
            "RCVTIMEO": rcvtimeo,
            "SNDTIMEO": sndtimeo,
        }
        self.fail_on_set = fail_on_set
        self.fail_on_first_restore = fail_on_first_restore
        self._restore_failed = False
        self.set_history: list[tuple[str, int, str]] = []
        self.access_history: list[tuple[str, str, int, str]] = []

    @property
    def RCVTIMEO(self) -> int:
        self.access_history.append(
            (
                "get",
                "RCVTIMEO",
                self._rcvtimeo,
                threading.current_thread().name,
            )
        )
        return self._rcvtimeo

    @RCVTIMEO.setter
    def RCVTIMEO(self, value: int) -> None:
        self.access_history.append(
            (
                "set",
                "RCVTIMEO",
                value,
                threading.current_thread().name,
            )
        )
        self.set_history.append(
            ("RCVTIMEO", value, threading.current_thread().name)
        )
        if self.fail_on_set == "RCVTIMEO":
            self.fail_on_set = None
            raise RuntimeError("rcvtimeo-set-failed")
        if (
            self.fail_on_first_restore == "RCVTIMEO"
            and not self._restore_failed
            and value == self._originals["RCVTIMEO"]
            and self._rcvtimeo != value
        ):
            self._restore_failed = True
            raise RuntimeError("rcvtimeo-restore-failed")
        self._rcvtimeo = value

    @property
    def SNDTIMEO(self) -> int:
        self.access_history.append(
            (
                "get",
                "SNDTIMEO",
                self._sndtimeo,
                threading.current_thread().name,
            )
        )
        return self._sndtimeo

    @SNDTIMEO.setter
    def SNDTIMEO(self, value: int) -> None:
        self.access_history.append(
            (
                "set",
                "SNDTIMEO",
                value,
                threading.current_thread().name,
            )
        )
        self.set_history.append(
            ("SNDTIMEO", value, threading.current_thread().name)
        )
        if self.fail_on_set == "SNDTIMEO":
            self.fail_on_set = None
            raise RuntimeError("sndtimeo-set-failed")
        if (
            self.fail_on_first_restore == "SNDTIMEO"
            and not self._restore_failed
            and value == self._originals["SNDTIMEO"]
            and self._sndtimeo != value
        ):
            self._restore_failed = True
            raise RuntimeError("sndtimeo-restore-failed")
        self._sndtimeo = value


class ProtocolFaithfulClient:
    """Models the timeout-relevant part of RemoteAPIClient._send."""

    def __init__(
        self,
        *,
        timeout: float = 600.0,
        socket: ProtocolSocket | None = None,
        send_count: int = 2,
    ) -> None:
        self.timeout = timeout
        self.socket = socket or ProtocolSocket()
        self.sendCnt = send_count

    def next_request(self) -> dict[str, float]:
        self.sendCnt += 1
        request: dict[str, float] = {}
        if self.sendCnt == 1:
            request["timeout"] = self.timeout
        return request


class ProtocolAgain(OSError):
    def __init__(self) -> None:
        super().__init__(errno.EAGAIN, "Resource temporarily unavailable")


class FakeSession:
    def __init__(
        self,
        *,
        backend="sim",
        robot: FakeRobot | None = None,
        tool: FakeTool | None = None,
        client=None,
    ):
        self.actions: list[tuple] = []
        self._backend = backend
        self._client = client
        self.application = self._application(
            robot or FakeRobot(self.actions),
            tool or FakeTool(self.actions),
        )
        self.reset_calls = 0
        self.quarantined_reset_calls = 0

    def _application(self, robot, tool):
        application = SimpleNamespace(
            robot=robot,
            tool=tool,
            workspace=WorkspacePolicy(
                x_mm=(20, 140),
                y_mm=(-90, 90),
                z_mm=(10, 140),
                safe_z_mm=100,
            ),
            config=SimpleNamespace(robot_backend=self._backend),
        )
        if self._client is not None:
            application.client = self._client
        return application

    def reset_simulation(self):
        self.reset_calls += 1
        return self._replace_application()

    def reset_simulation_quarantined(self):
        self.quarantined_reset_calls += 1
        return self._replace_application()

    def _replace_application(self):
        self.actions = []
        self.application = self._application(
            FakeRobot(self.actions),
            FakeTool(self.actions),
        )
        return self.application


class FakeCamera:
    def read(self, timeout_s):
        assert timeout_s == 2.0
        return Frame(
            image_bgr=np.full((3, 4, 3), 80, dtype=np.uint8),
            width=4,
            height=3,
            timestamp_s=1.25,
            source="coppeliasim",
            sequence_id=7,
        )


class RebuildingFakeSession(FakeSession):
    def __init__(self, *, replacement_client, **kwargs):
        super().__init__(**kwargs)
        self.replacement_client = replacement_client

    def reset_simulation(self):
        self._client = self.replacement_client
        return super().reset_simulation()

    def reset_simulation_quarantined(self):
        self.quarantined_reset_calls += 1
        self._client = self.replacement_client
        return self._replace_application()


def wait_until(predicate, timeout_s=2):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


def write_program(tmp_path: Path, source: str, *, name="student.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def policy(**overrides) -> StudentExecutionPolicy:
    values = {
        "min_speed": 1,
        "max_speed": 30,
        "max_runtime_s": 2,
        "max_commands": 200,
        "command_timeout_s": 1,
        "max_sleep_s": 0.2,
        "tool_on_max_z_mm": 35,
    }
    values.update(overrides)
    return StudentExecutionPolicy(**values)


def make_controller(
    tmp_path: Path,
    source: str,
    *,
    backend: str = "sim",
    execution_policy: StudentExecutionPolicy | None = None,
    session: FakeSession | None = None,
    experiment_context: ExperimentRunContext | None = None,
    experiment_definition: ExperimentDefinition | None = None,
    scene_manifest=None,
):
    selected_session = session or FakeSession(backend=backend)
    controller = StudentProgramController(
        session=selected_session,
        execution_policy=execution_policy or policy(),
        output_root=tmp_path / "runs",
        experiment_context=experiment_context,
        experiment_definition=experiment_definition,
        scene_manifest=scene_manifest,
    )
    program = write_program(tmp_path, source)
    controller.load(program)
    return controller, selected_session, program


def experiment_bundle(
    tmp_path: Path,
    *,
    experiment_id="R1-01",
    probe_kind="motion_observation",
    public_parameters=None,
    task_contracts=None,
):
    scene = tmp_path / "scene.ttt"
    scene.write_bytes(b"runner-test-scene")
    scene_sha256 = hashlib.sha256(scene.read_bytes()).hexdigest()
    manifest_path = tmp_path / "scene_manifest.json"
    manifest = {
        "schema_version": 1,
        "scene": {"path": str(scene), "sha256": scene_sha256},
        "task_contracts": dict(task_contracts or {}),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    parameters = dict(public_parameters or {})
    definition = ExperimentDefinition(
        experiment_id=experiment_id,
        pack_id="R1",
        title="Runner experiment",
        version="2.2.0",
        scene=scene,
        scene_manifest=manifest_path,
        student_template=tmp_path / "template.py",
        guide=tmp_path / "guide.md",
        capabilities=("scene.probe",),
        workspace={
            "x_mm": (20, 140),
            "y_mm": (-90, 90),
            "z_mm": (10, 140),
            "safe_z_mm": 100,
        },
        public_parameters=parameters,
        acceptance=ExperimentAcceptance(
            probe_kind=probe_kind,
            automated_checks=("scene_probe",),
            human_checks=("teacher_review",),
        ),
        hardware_status="PENDING_HARDWARE",
    )
    context = ExperimentRunContext(
        experiment_id=experiment_id,
        experiment_version="2.2.0",
        scene_path=scene,
        scene_sha256=scene_sha256,
        scene_manifest_path=manifest_path,
        public_parameters=parameters,
    )
    return context, definition, manifest


def profile_factory_bundle(
    tmp_path: Path,
    *,
    allowed_profile_ids=("standard", "wide_dim"),
):
    project = tmp_path / "project"
    manifest_dir = project / "config" / "experiments"
    manifest_dir.mkdir(parents=True)
    scene_dir = project / "simulation" / "vision_quality_lab"
    scene_dir.mkdir(parents=True)
    scene = scene_dir / "BL23_vision_quality_lab.ttt"
    scene.write_bytes(b"runner-profile-scene")
    scene_sha256 = hashlib.sha256(scene.read_bytes()).hexdigest()
    profile_path = scene_dir / "profiles.json"
    profile_payload = {
        "schema_version": 1,
        "baseline_profile_id": "standard",
        "sensor_path": "/VisionQualityLab/CameraRig/Camera",
        "camera_rig_path": "/VisionQualityLab/CameraRig",
        "key_light_path": "/VisionQualityLab/Lighting/KeyLight",
        "fill_light_path": "/VisionQualityLab/Lighting/FillLight",
        "near_clip_m": 0.05,
        "far_clip_m": 2.0,
        "profiles": [
            {
                "profile_id": "standard",
                "label": "standard",
                "resolution": [512, 512],
                "perspective_angle_deg": 60,
                "camera_rig_z_m": 0.7,
                "key_diffuse_rgb": [0.8, 0.8, 0.8],
                "fill_diffuse_rgb": [0.35, 0.35, 0.35],
            },
            {
                "profile_id": "wide_dim",
                "label": "wide dim",
                "resolution": [256, 256],
                "perspective_angle_deg": 75,
                "camera_rig_z_m": 0.8,
                "key_diffuse_rgb": [0.35, 0.35, 0.35],
                "fill_diffuse_rgb": [0.15, 0.15, 0.15],
            },
        ],
    }
    profile_bytes = json.dumps(
        profile_payload,
        ensure_ascii=False,
    ).encode("utf-8")
    profile_path.write_bytes(profile_bytes)
    manifest_path = manifest_dir / "V1-01.scene.json"
    manifest = {
        "schema_version": 1,
        "scene": {
            "path": str(scene),
            "sha256": scene_sha256,
        },
        "profile_catalog": {
            "path": "simulation/vision_quality_lab/profiles.json",
            "sha256": hashlib.sha256(profile_bytes).hexdigest(),
        },
        "task_contracts": {},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    parameters = {
        "baseline_profile_id": "standard",
        "allowed_profile_ids": list(allowed_profile_ids),
    }
    definition = ExperimentDefinition(
        experiment_id="V1-01",
        pack_id="V1",
        title="Vision quality profile",
        version="2.2.0",
        scene=scene,
        scene_manifest=manifest_path,
        student_template=project / "student.py",
        guide=project / "guide.md",
        capabilities=(
            "camera.rgb",
            "camera.profile",
            "lighting.profile",
            "scene.probe",
        ),
        workspace={
            "x_mm": (20, 140),
            "y_mm": (-90, 90),
            "z_mm": (10, 140),
            "safe_z_mm": 100,
        },
        public_parameters=parameters,
        acceptance=ExperimentAcceptance(
            probe_kind="motion_observation",
            automated_checks=("scene_probe",),
            human_checks=("teacher_review",),
        ),
        hardware_status="PENDING_HARDWARE",
    )
    context = ExperimentRunContext(
        experiment_id="V1-01",
        experiment_version="2.2.0",
        scene_path=scene,
        scene_sha256=scene_sha256,
        scene_manifest_path=manifest_path,
        public_parameters=parameters,
    )
    return context, definition, manifest


class FunctionalProfileSim:
    handle_world = -1
    visionintparam_resolution_x = "resolution_x"
    visionintparam_resolution_y = "resolution_y"
    visionfloatparam_perspective_angle = "perspective_angle"
    visionfloatparam_near_clipping = "near_clip"
    visionfloatparam_far_clipping = "far_clip"

    def __init__(self, *, read_error=False, mismatch=False):
        resolution = 300 if mismatch else 512
        self.ints = {
            "resolution_x": resolution,
            "resolution_y": resolution,
        }
        self.floats = {
            "perspective_angle": 1.0471975511965976,
            "near_clip": 0.05,
            "far_clip": 2.0,
        }
        self.rig_position = [0.0, 0.0, 0.7]
        self.lights = {
            3: (1, [0.8, 0.8, 0.8], [0.0, 0.0, 0.0]),
            4: (1, [0.35, 0.35, 0.35], [0.0, 0.0, 0.0]),
        }
        self.read_error = read_error
        self.get_calls = []

    def getObject(self, path):
        self.get_calls.append(path)
        return {
            "/VisionQualityLab/CameraRig/Camera": 1,
            "/VisionQualityLab/CameraRig": 2,
            "/VisionQualityLab/Lighting/KeyLight": 3,
            "/VisionQualityLab/Lighting/FillLight": 4,
        }.get(path, path)

    def _fail_read(self):
        if self.read_error:
            raise RuntimeError("profile-read-failed")

    def getObjectInt32Param(self, handle, parameter):
        self._fail_read()
        return self.ints[parameter]

    def getObjectFloatParam(self, handle, parameter):
        self._fail_read()
        return self.floats[parameter]

    def getObjectPosition(self, handle, relative_to):
        self._fail_read()
        return list(self.rig_position)

    def getLightParameters(self, handle):
        self._fail_read()
        enabled, diffuse, specular = self.lights[handle]
        return enabled, [0.0, 0.0, 0.0], list(diffuse), list(specular)

    def setObjectInt32Param(self, handle, parameter, value):
        self.ints[parameter] = value

    def setObjectFloatParam(self, handle, parameter, value):
        self.floats[parameter] = value

    def setObjectPosition(self, handle, relative_to, value):
        self.rig_position = list(value)

    def setLightParameters(
        self,
        handle,
        enabled,
        ambient,
        diffuse,
        specular,
    ):
        self.lights[handle] = (enabled, list(diffuse), list(specular))

    def getVisionSensorImg(self, handle):
        width = self.ints["resolution_x"]
        height = self.ints["resolution_y"]
        return bytes(width * height * 3), [width, height]


class ProbeSim:
    handle_world = -1

    def __init__(
        self,
        positions=None,
        *,
        actions=None,
        fail_after=None,
        probe_error_after=None,
        probe_error=None,
        block_after=None,
        block_entered=None,
        block_release=None,
        timeout_probe=None,
    ):
        self.positions = dict(positions or {})
        self.actions = actions if actions is not None else []
        self.fail_after = fail_after
        self.probe_error_after = probe_error_after
        self.probe_error = probe_error
        self.block_after = block_after
        self.block_entered = block_entered
        self.block_release = block_release
        self.timeout_probe = timeout_probe
        self.timeout_observations: list[tuple] = []
        self.get_calls = 0

    @classmethod
    def motion(cls, *, actions=None, fail_after=None, **kwargs):
        paths = [f"/BLX_joint{index}" for index in range(1, 7)]
        paths.append("/BLX_tool_suction")
        return cls(
            {path: None for path in paths},
            actions=actions,
            fail_after=fail_after,
            **kwargs,
        )

    def getObject(self, path):
        self.get_calls += 1
        self.actions.append(("probe.get", path))
        if self.timeout_probe is not None:
            self.timeout_observations.append(self.timeout_probe())
        if (
            self.block_after is not None
            and self.get_calls > self.block_after
        ):
            if self.block_entered is not None:
                self.block_entered.set()
            if self.block_release is None or not self.block_release.wait(10):
                raise TimeoutError("test probe block was not released")
        if (
            self.probe_error_after is not None
            and self.get_calls > self.probe_error_after
        ):
            raise self.probe_error or ProtocolAgain()
        if self.fail_after is not None and self.get_calls > self.fail_after:
            raise RuntimeError("injected-probe-failure")
        if path not in self.positions:
            raise RuntimeError(f"missing: {path}")
        return path

    def getObjectPosition(self, handle, relative_to):
        assert relative_to == self.handle_world
        self.actions.append(("probe.position", handle))
        return self.positions[handle]


@pytest.mark.parametrize(
    "source",
    [
        "def main(ctx):\n    ctx.camera.capture()\n",
        "def main(ctx):\n    ctx.experiment.info()\n",
    ],
)
def test_experiment_command_without_context_fails_closed(tmp_path, source):
    controller, _, _ = make_controller(
        tmp_path,
        source,
        experiment_context=None,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_CONTEXT_REQUIRED"


def test_controller_dispatches_experiment_commands_and_records_metadata(
    tmp_path,
):
    session = FakeSession()
    session.application.camera = FakeCamera()
    session.application.sim = ProbeSim.motion(actions=session.actions)
    context, definition, scene_manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        (
            "def main(ctx):\n"
            "    info = ctx.experiment.info()\n"
            "    frame = ctx.camera.capture()\n"
            "    ctx.log(info['experiment_id'])\n"
            "    ctx.log(frame.snapshot_id)\n"
        ),
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=scene_manifest,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "PASS", result.error
    assert result.evidence_dir is not None
    manifest = json.loads(
        (result.evidence_dir / "manifest.json").read_text(encoding="utf-8")
    )
    snapshots = [
        json.loads(line)
        for line in (result.evidence_dir / "snapshots.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["experiment_id"] == "R1-01"
    assert manifest["experiment_version"] == "2.2.0"
    assert manifest["scene_sha256"] == context.scene_sha256
    assert manifest["hardware_status"] == "PENDING_HARDWARE"
    assert "scene_path" not in manifest
    assert snapshots[0]["source"] == "coppeliasim"
    assert snapshots[0]["sequence_id"] == 7
    assert (result.evidence_dir / snapshots[0]["path"]).read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    summary = json.loads(
        result.summary_path.read_text(encoding="utf-8")
    )
    assert summary["scene_probe_status"] == "PASS"
    assert (result.evidence_dir / "scene-initial.json").is_file()
    assert (result.evidence_dir / "scene-final.json").is_file()


def test_profile_commands_route_only_through_experiment_gateway(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    calls = []
    controller._experiment_gateway = SimpleNamespace(
        dispatch=lambda name, args: calls.append((name, args)) or name
    )

    for index, (name, args) in enumerate(
        (
            ("camera.profile.get", {}),
            ("camera.profile.apply", {"profile_id": "wide_dim"}),
            ("camera.profile.reset", {}),
        ),
        start=1,
    ):
        value = controller._dispatch(
            CommandMessage(f"cmd-{index}", name, args)
        )
        assert value == name

    assert calls == [
        ("camera.profile.get", {}),
        ("camera.profile.apply", {"profile_id": "wide_dim"}),
        ("camera.profile.reset", {}),
    ]


def test_terminal_cleanup_resets_vision_before_tool_and_robot(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    trace = []
    controller._experiment_gateway = SimpleNamespace(
        reset_environment=lambda: trace.append("vision.reset")
    )
    controller._application.tool.off = lambda: trace.append("tool.off")
    controller._read_pose = (
        lambda: trace.append("robot.pose") or (0.0, 0.0, 100.0)
    )
    controller._application.robot.move_home = (
        lambda: trace.append("robot.home")
    )

    errors = controller._cleanup()

    assert errors == []
    assert trace == [
        "vision.reset",
        "tool.off",
        "robot.pose",
        "robot.home",
    ]


def test_profile_reset_failure_changes_pass_to_failed_and_preserves_hardware(
    tmp_path,
    monkeypatch,
):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(actions=session.actions)
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    def fail_reset(self):
        raise RuntimeError("vision-reset-failed")

    monkeypatch.setattr(
        runner_module.StudentExperimentGateway,
        "reset_environment",
        fail_reset,
        raising=False,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_CLEANUP_FAILED"
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    assert summary["status"] == "FAILED"
    assert [item["stage"] for item in summary["cleanup_errors"]] == [
        "vision.profile.reset",
    ]


def test_profile_reset_transport_failure_quarantines_before_tool(tmp_path):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )

    def fail_reset():
        try:
            raise ProtocolAgain()
        except ProtocolAgain as error:
            raise RuntimeError("VISION_PROFILE_RESET_FAILED") from error

    controller._experiment_gateway = SimpleNamespace(
        reset_environment=fail_reset
    )

    errors = controller._cleanup()

    assert [item["stage"] for item in errors] == [
        "vision.profile.reset",
        "backend.connection",
    ]
    assert controller._backend_quarantined is True
    assert session.application.tool.off_calls == 0
    assert session.application.robot.pose_calls == 0


def test_profile_reset_rollback_failure_quarantines_before_tool(tmp_path):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )

    def fail_reset():
        try:
            raise RuntimeError("VISION_PROFILE_ROLLBACK_FAILED")
        except RuntimeError as error:
            raise RuntimeError("VISION_PROFILE_RESET_FAILED") from error

    controller._experiment_gateway = SimpleNamespace(
        reset_environment=fail_reset
    )

    errors = controller._cleanup()

    assert [item["stage"] for item in errors] == [
        "vision.profile.reset",
        "backend.connection",
    ]
    assert errors[0]["error"]["code"] == (
        "STUDENT_BACKEND_CONNECTION_QUARANTINED"
    )
    assert controller._backend_quarantined is True
    assert session.application.tool.off_calls == 0
    assert session.application.robot.pose_calls == 0


def test_profile_rollback_failure_detection_walks_nested_exception_groups():
    rollback = RuntimeError("VISION_PROFILE_ROLLBACK_FAILED")
    wrapped = RuntimeError("VISION_PROFILE_RESET_FAILED")
    wrapped.__cause__ = ExceptionGroup("rollback", [rollback])

    marker = KeyboardInterrupt("student interrupt")
    marker.vision_profile_rollback_failure = (
        "VISION_PROFILE_ROLLBACK_FAILED"
    )
    noted = SystemExit("student exit")
    noted.add_note("VISION_PROFILE_ROLLBACK_FAILED during rollback")
    coded = VisionPlatformError(
        "VISION_PROFILE_ROLLBACK_FAILED",
        "rollback failed",
    )
    contextual = RuntimeError("outer")
    contextual.__context__ = BaseExceptionGroup(
        "nested",
        [marker, noted, coded],
    )

    assert runner_module._is_profile_rollback_failure(rollback) is True
    assert runner_module._is_profile_rollback_failure(wrapped) is True
    assert runner_module._is_profile_rollback_failure(marker) is True
    assert runner_module._is_profile_rollback_failure(noted) is True
    assert runner_module._is_profile_rollback_failure(coded) is True
    assert runner_module._is_profile_rollback_failure(contextual) is True
    assert (
        runner_module._is_profile_rollback_failure(
            RuntimeError("VISION_PROFILE_APPLY_FAILED")
        )
        is False
    )


def test_profile_error_code_survives_student_error_boundary():
    failure = VisionPlatformError(
        "VISION_PROFILE_NOT_ALLOWED",
        "profile is not allowed",
    )

    assert runner_module._exception_error(failure) == {
        "code": "VISION_PROFILE_NOT_ALLOWED",
        "message": "profile is not allowed",
        "details": {},
    }


def test_profile_reset_stuck_is_bounded_and_quarantined_before_tool(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        execution_policy=policy(command_timeout_s=0.05),
    )

    def block_reset():
        entered.set()
        assert release.wait(timeout=5)

    controller._experiment_gateway = SimpleNamespace(
        reset_environment=block_reset
    )
    started = time.monotonic()
    try:
        errors = controller._cleanup()

        assert entered.is_set()
        assert time.monotonic() - started < 0.4
        assert [item["stage"] for item in errors] == [
            "vision.profile.reset",
            "backend.connection",
        ]
        assert errors[0]["error"]["code"] == (
            "STUDENT_BACKEND_COMMAND_STUCK"
        )
        assert controller._backend_quarantined is True
        assert controller._backend_action_is_alive() is True
        assert session.application.tool.off_calls == 0
        assert session.application.robot.pose_calls == 0
    finally:
        release.set()
        assert controller.wait_for_quiescence(1) is True


def stack_experiment_bundle(tmp_path):
    objects = [
        {
            "alias": f"stack_{index:02d}",
            "position_mm": [40 + index * 10, -60, 20],
        }
        for index in range(1, 7)
    ]
    parameters = {
        "scene_group_path": "/LogisticsLab/Tasks/Stack",
        "stack_slots_mm": [
            [118, -45, 20],
            [118, 45, 20],
            [118, -45, 38],
            [118, 45, 38],
            [118, -45, 56],
            [118, 45, 56],
        ],
    }
    context, definition, manifest = experiment_bundle(
        tmp_path,
        experiment_id="R1-05",
        probe_kind="stack_2x3",
        public_parameters=parameters,
        task_contracts={"Stack": {"objects": objects}},
    )
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/{item['alias']}": [
            value / 1000 for value in item["position_mm"]
        ]
        for item in objects
    }
    return context, definition, manifest, positions


def test_final_probe_runs_after_safe_cleanup_and_does_not_fail_student_pass(
    tmp_path,
):
    session = FakeSession()
    context, definition, manifest, positions = stack_experiment_bundle(
        tmp_path
    )
    session.application.sim = ProbeSim(positions, actions=session.actions)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "PASS", result.error
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    final_probe = json.loads(
        (result.evidence_dir / "scene-final.json").read_text(encoding="utf-8")
    )
    assert final_probe["status"] == "FAIL"
    assert summary["status"] == "PASS"
    assert summary["scene_probe_status"] == "FAIL"
    home_index = session.actions.index(("home",))
    final_probe_index = next(
        index
        for index in range(home_index + 1, len(session.actions))
        if session.actions[index][0] == "probe.get"
    )
    assert session.actions[home_index - 1] == ("tool.off",)
    assert final_probe_index > home_index


def test_initial_probe_failure_records_evidence_and_never_attempts_spawn(
    tmp_path,
):
    session = FakeSession()
    context, definition, manifest, positions = stack_experiment_bundle(
        tmp_path
    )
    positions = {path: [0.0, 0.0, 0.0] for path in positions}
    session.application.sim = ProbeSim(positions, actions=session.actions)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_SCENE_INITIAL_INVALID"
    initial = json.loads(
        (result.evidence_dir / "scene-initial.json").read_text(encoding="utf-8")
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert initial["status"] == "FAIL"
    assert summary["error"]["code"] == "EXPERIMENT_SCENE_INITIAL_INVALID"
    assert (result.evidence_dir / "events.jsonl").read_text(encoding="utf-8") == ""


def test_initial_probe_exception_records_error_and_never_attempts_spawn(
    tmp_path,
):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(
        actions=session.actions,
        fail_after=0,
    )
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    initial = json.loads(
        (result.evidence_dir / "scene-initial.json").read_text(encoding="utf-8")
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_SCENE_INITIAL_PROBE_ERROR"
    assert initial["status"] == "ERROR"
    assert summary["error"]["code"] == "EXPERIMENT_SCENE_INITIAL_PROBE_ERROR"
    assert (result.evidence_dir / "events.jsonl").read_text(encoding="utf-8") == ""


def test_final_probe_exception_is_diagnostic_and_preserves_student_pass(
    tmp_path,
):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(
        actions=session.actions,
        fail_after=7,
    )
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "PASS", result.error
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    final_probe = json.loads(
        (result.evidence_dir / "scene-final.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"
    assert summary["scene_probe_status"] == "ERROR"
    assert final_probe["status"] == "ERROR"
    assert any(
        item["stage"] == "scene_probe.final"
        for item in summary["cleanup_errors"]
    )


def test_final_probe_exception_does_not_replace_student_failure(tmp_path):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(
        actions=session.actions,
        fail_after=7,
    )
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    raise ValueError('primary-student-failure')\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_PROGRAM_FAILED"
    assert "primary-student-failure" in result.error["message"]
    assert summary["scene_probe_status"] == "ERROR"
    assert summary["error"]["code"] == "STUDENT_PROGRAM_FAILED"
    assert any(
        item["stage"] == "scene_probe.final"
        for item in summary["cleanup_errors"]
    )


def test_failed_student_status_is_not_changed_by_final_probe_failure(tmp_path):
    session = FakeSession()
    context, definition, manifest, positions = stack_experiment_bundle(
        tmp_path
    )
    session.application.sim = ProbeSim(positions, actions=session.actions)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    raise ValueError('student-failed')\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_PROGRAM_FAILED"
    assert summary["scene_probe_status"] == "FAIL"


def test_cancelled_student_status_is_not_changed_by_final_probe(tmp_path):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(actions=session.actions)
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    while True:\n        pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True
    controller.start()
    wait_until(lambda: controller.process_is_alive)

    controller.cancel()
    result = controller.wait(timeout_s=5)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "CANCELLED"
    assert summary["status"] == "CANCELLED"
    assert summary["scene_probe_status"] == "PASS"


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing_definition",
        "context_missing",
        "experiment_id",
        "experiment_version",
        "scene_path",
        "scene_sha256",
        "scene_manifest_path",
        "manifest_payload",
    ],
)
def test_experiment_binding_mismatch_fails_before_student_spawn(
    tmp_path,
    mismatch,
):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(actions=session.actions)
    context, definition, manifest = experiment_bundle(tmp_path)
    if mismatch == "missing_definition":
        definition = None
    elif mismatch == "context_missing":
        context = None
    elif mismatch == "experiment_id":
        context = replace(context, experiment_id="R1-99")
    elif mismatch == "experiment_version":
        context = replace(context, experiment_version="9.9.9")
    elif mismatch == "scene_path":
        other = tmp_path / "other.ttt"
        other.write_bytes(b"other")
        context = replace(context, scene_path=other)
    elif mismatch == "scene_sha256":
        context = replace(context, scene_sha256="0" * 64)
    elif mismatch == "scene_manifest_path":
        other = tmp_path / "other-manifest.json"
        other.write_text("{}", encoding="utf-8")
        context = replace(context, scene_manifest_path=other)
    elif mismatch == "manifest_payload":
        manifest = json.loads(json.dumps(manifest))
        manifest["scene"]["sha256"] = "0" * 64

    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_CONTEXT_INVALID"
    assert controller.process_is_alive is False


def test_reset_builds_new_gateway_bound_to_current_application(tmp_path):
    session = FakeSession()
    first_sim = ProbeSim.motion(actions=session.actions)
    session.application.sim = first_sim
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True
    controller.start()
    assert controller.wait(timeout_s=5).status == "PASS"
    first_call_count = first_sim.get_calls

    replacement = controller.reset()
    second_sim = ProbeSim.motion(actions=session.actions)
    replacement.sim = second_sim
    first_sim.fail_after = first_sim.get_calls
    controller.start()
    second_result = controller.wait(timeout_s=5)

    assert second_result.status == "PASS", second_result.error
    assert first_sim.get_calls == first_call_count
    assert second_sim.get_calls == 14


def test_final_probe_runs_while_client_timeouts_are_still_bounded(tmp_path):
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(timeout=600.0, socket=socket)
    session = FakeSession(client=client)
    timeout_probe = lambda: (
        client.timeout,
        socket.RCVTIMEO,
        socket.SNDTIMEO,
    )
    sim = ProbeSim.motion(
        actions=session.actions,
        timeout_probe=timeout_probe,
    )
    session.application.sim = sim
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "PASS", result.error
    assert sim.get_calls == 14
    assert sim.timeout_observations == [(0.1, 100, 100)] * 14
    assert client.timeout == 600.0
    assert socket.RCVTIMEO == 5000
    assert socket.SNDTIMEO == 7000
    assert controller._backend_action_threads == {}


def test_gateway_factory_stuck_is_timeout_bounded_and_quarantined(
    tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(timeout=600.0, socket=socket)
    observations = []

    class BlockingSim(FunctionalProfileSim):
        def getObject(self, path):
            observations.append(
                (client.timeout, socket.RCVTIMEO, socket.SNDTIMEO)
            )
            entered.set()
            if not release.wait(10):
                raise TimeoutError("profile factory test release timed out")
            return super().getObject(path)

    sim = BlockingSim()
    camera = CoppeliaSimCamera(
        sensor_path="/VisionQualityLab/CameraRig/Camera",
        sim=sim,
    )
    session = FakeSession(client=client)
    session.application.config.camera_backend = "sim"
    session.application.sim = sim
    session.application.camera = camera
    context, definition, manifest = profile_factory_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.05),
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True

    starter = threading.Thread(target=controller.start)
    starter.start()
    assert entered.wait(2)
    completed_while_blocked = controller._done.wait(0.5)
    if not completed_while_blocked:
        release.set()
        starter.join(5)
    assert completed_while_blocked, "gateway factory was not execution-bounded"

    result = controller.wait(timeout_s=1)
    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_BACKEND_COMMAND_STUCK"
    assert controller._backend_quarantined is True
    assert observations == [(0.05, 50, 50)]
    assert session.application.tool.off_calls == 0
    assert session.application.robot.home_calls == 0

    release.set()
    starter.join(5)
    assert controller.wait_for_quiescence(2) is True


def test_initial_probe_stuck_is_bounded_quarantined_and_never_spawns(
    tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(timeout=600.0, socket=socket)
    session = FakeSession(client=client)
    session.application.sim = ProbeSim.motion(
        actions=session.actions,
        block_after=0,
        block_entered=entered,
        block_release=release,
    )
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.05),
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True
    starter = threading.Thread(target=controller.start)
    starter.start()
    assert entered.wait(2)
    completed_while_blocked = controller._done.wait(0.5)
    if not completed_while_blocked:
        release.set()
        starter.join(5)
    assert completed_while_blocked, "initial probe was not execution-bounded"

    result = controller.wait(timeout_s=1)
    initial = json.loads(
        (result.evidence_dir / "scene-initial.json").read_text(encoding="utf-8")
    )
    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_SCENE_INITIAL_PROBE_ERROR"
    assert initial["status"] == "ERROR"
    assert initial["error"]["code"] == "SCENE_PROBE_EXECUTION_TIMEOUT"
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert not any(
        item["stage"] == "backend.command"
        for item in summary["cleanup_errors"]
    )
    assert controller._backend_quarantined is True
    assert client.timeout == 0.05
    assert socket.RCVTIMEO == 50
    assert socket.SNDTIMEO == 50
    with pytest.raises(RuntimeError, match="STUDENT_BACKEND_COMMAND_STUCK"):
        controller.reset()

    release.set()
    starter.join(5)
    assert controller.wait_for_quiescence(2) is True
    old_application = session.application
    replacement = controller.reset()
    assert replacement is not old_application
    assert session.quarantined_reset_calls == 1


def test_final_probe_stuck_is_bounded_and_preserves_student_pass(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(timeout=600.0, socket=socket)
    session = FakeSession(client=client)
    session.application.sim = ProbeSim.motion(
        actions=session.actions,
        block_after=7,
        block_entered=entered,
        block_release=release,
    )
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.05),
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True
    controller.start()
    assert entered.wait(3)
    completed_while_blocked = controller._done.wait(0.5)
    if not completed_while_blocked:
        release.set()
        controller.wait(timeout_s=5)
    assert completed_while_blocked, "final probe was not execution-bounded"

    result = controller.wait(timeout_s=1)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    final_probe = json.loads(
        (result.evidence_dir / "scene-final.json").read_text(encoding="utf-8")
    )
    assert result.status == "PASS"
    assert result.error is None
    assert summary["status"] == "PASS"
    assert summary["error"] is None
    assert summary["scene_probe_status"] == "ERROR"
    assert final_probe["error"]["code"] == "SCENE_PROBE_EXECUTION_TIMEOUT"
    assert controller._backend_quarantined is True
    assert client.timeout == 0.05
    assert socket.RCVTIMEO == 50
    assert socket.SNDTIMEO == 50
    with pytest.raises(RuntimeError, match="STUDENT_BACKEND_COMMAND_STUCK"):
        controller.reset()

    final_probe_bytes = (result.evidence_dir / "scene-final.json").read_bytes()
    release.set()
    assert controller.wait_for_quiescence(2) is True
    assert (
        result.evidence_dir / "scene-final.json"
    ).read_bytes() == final_probe_bytes
    assert list(result.evidence_dir.glob("scene-final*.json")) == [
        result.evidence_dir / "scene-final.json"
    ]
    old_application = session.application
    replacement = controller.reset()
    assert replacement is not old_application
    assert session.quarantined_reset_calls == 1


def test_initial_probe_transport_error_is_evidenced_and_quarantined(
    tmp_path,
):
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(timeout=600.0, socket=socket)
    session = FakeSession(client=client)
    session.application.sim = ProbeSim.motion(
        actions=session.actions,
        probe_error_after=0,
        probe_error=ProtocolAgain(),
    )
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.05),
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=2)

    initial = json.loads(
        (result.evidence_dir / "scene-initial.json").read_text(encoding="utf-8")
    )
    assert result.status == "FAILED"
    assert initial["status"] == "ERROR"
    assert initial["error"]["code"] == "SCENE_PROBE_TRANSPORT_FAILED"
    assert controller._backend_quarantined is True
    assert client.timeout == 0.05
    assert socket.RCVTIMEO == 50
    assert socket.SNDTIMEO == 50
    old_application = session.application
    replacement = controller.reset()
    assert replacement is not old_application
    assert session.quarantined_reset_calls == 1


def test_final_probe_timeout_finalize_failure_never_reuses_old_backend(
    tmp_path,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(timeout=600.0, socket=socket)
    replacement_client = ProtocolFaithfulClient()
    actions: list[tuple] = []
    robot = FakeRobot(actions)
    tool = FakeTool(actions)
    session = RebuildingFakeSession(
        robot=robot,
        tool=tool,
        client=client,
        replacement_client=replacement_client,
    )
    session.application.sim = ProbeSim.motion(
        actions=session.actions,
        block_after=7,
        block_entered=entered,
        block_release=release,
    )
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.05),
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True
    at_finalize: dict[str, object] = {}

    def fail_finalize(*args, **kwargs):
        at_finalize.update(
            tool_off_calls=tool.off_calls,
            robot_pose_calls=robot.pose_calls,
            robot_home_calls=robot.home_calls,
            socket_accesses=tuple(socket.access_history),
            probe_thread=controller._backend_action_thread,
        )
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    assert entered.wait(3)
    result = controller.wait(timeout_s=2)
    probe_thread = at_finalize["probe_thread"]

    try:
        assert result.status == "FAILED"
        assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
        assert result.error["details"]["original_status"] == "PASS"
        assert result.error["details"]["original_error"] is None
        assert result.summary_path is None
        assert tool.off_calls == at_finalize["tool_off_calls"]
        assert robot.pose_calls == at_finalize["robot_pose_calls"]
        assert robot.home_calls == at_finalize["robot_home_calls"]
        assert tuple(socket.access_history) == at_finalize[
            "socket_accesses"
        ]
        assert probe_thread.is_alive() is True
        assert controller._backend_action_is_alive() is True
        assert controller._backend_action_threads.get(probe_thread) == (
            "probe:final"
        )
        final_path = result.evidence_dir / "scene-final.json"
        final_bytes = final_path.read_bytes()
        with pytest.raises(RuntimeError, match="STUDENT_BACKEND_COMMAND_STUCK"):
            controller.reset()
    finally:
        release.set()

    probe_thread.join(timeout=2)
    assert probe_thread.is_alive() is False
    assert controller.wait_for_quiescence(2) is True
    assert controller._backend_action_threads == {}
    assert final_path.read_bytes() == final_bytes
    assert list(result.evidence_dir.glob("scene-final*.json")) == [
        final_path
    ]
    old_application = session.application
    replacement = controller.reset()
    assert replacement is not old_application
    assert session.quarantined_reset_calls == 1
    assert session.reset_calls == 0
    assert socket.RCVTIMEO == 50
    assert socket.SNDTIMEO == 50
    assert session.application.client is replacement_client
    assert controller._backend_action_threads == {}


def test_final_probe_transport_finalize_failure_never_reuses_old_backend(
    tmp_path,
    monkeypatch,
):
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(timeout=600.0, socket=socket)
    replacement_client = ProtocolFaithfulClient()
    actions: list[tuple] = []
    robot = FakeRobot(actions)
    tool = FakeTool(actions)
    session = RebuildingFakeSession(
        robot=robot,
        tool=tool,
        client=client,
        replacement_client=replacement_client,
    )
    session.application.sim = ProbeSim.motion(
        actions=session.actions,
        probe_error_after=7,
        probe_error=ProtocolAgain(),
    )
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.05),
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True
    at_finalize: dict[str, object] = {}

    def fail_finalize(*args, **kwargs):
        at_finalize.update(
            tool_off_calls=tool.off_calls,
            robot_pose_calls=robot.pose_calls,
            robot_home_calls=robot.home_calls,
            socket_accesses=tuple(socket.access_history),
        )
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    result = controller.wait(timeout_s=2)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert result.error["details"]["original_status"] == "PASS"
    assert result.error["details"]["original_error"] is None
    assert result.summary_path is None
    assert tool.off_calls == at_finalize["tool_off_calls"]
    assert robot.pose_calls == at_finalize["robot_pose_calls"]
    assert robot.home_calls == at_finalize["robot_home_calls"]
    assert tuple(socket.access_history) == at_finalize["socket_accesses"]
    assert controller._backend_quarantined is True
    assert controller._backend_action_is_alive() is False
    assert controller._backend_action_threads == {}
    final_path = result.evidence_dir / "scene-final.json"
    final_bytes = final_path.read_bytes()
    final_probe = json.loads(final_bytes)
    assert final_probe["error"]["code"] == (
        "SCENE_PROBE_TRANSPORT_FAILED"
    )
    assert list(result.evidence_dir.glob("scene-final*.json")) == [
        final_path
    ]
    old_application = session.application
    replacement = controller.reset()
    assert replacement is not old_application
    assert session.quarantined_reset_calls == 1
    assert session.reset_calls == 0
    assert final_path.read_bytes() == final_bytes
    assert socket.RCVTIMEO == 50
    assert socket.SNDTIMEO == 50
    assert session.application.client is replacement_client
    assert controller._backend_action_threads == {}


_TYPE_SENSITIVE_PAIRS = [
    (True, 1),
    (False, 0),
    (1, 1.0),
    (True, 1.0),
    (False, 0.0),
]

_INVALID_IDENTITY_EQUAL_PAIRS = [
    *_TYPE_SENSITIVE_PAIRS,
    ("", ""),
    ("   ", "   "),
]


@pytest.mark.parametrize(
    "context_value,definition_value",
    _INVALID_IDENTITY_EQUAL_PAIRS,
)
@pytest.mark.parametrize(
    "identity_field",
    ["experiment_id", "experiment_version"],
)
def test_experiment_identity_requires_exact_nonempty_strings_before_spawn(
    tmp_path,
    identity_field,
    context_value,
    definition_value,
):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(actions=session.actions)
    context, definition, manifest = experiment_bundle(tmp_path)
    if identity_field == "experiment_id":
        context = replace(context, experiment_id=context_value)
        definition = replace(
            definition,
            experiment_id=definition_value,
        )
    else:
        context = replace(
            context,
            experiment_version=context_value,
        )
        definition = replace(definition, version=definition_value)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True
    controller.start()
    result = controller.wait(timeout_s=2)

    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_CONTEXT_INVALID"


@pytest.mark.parametrize("context_value,definition_value", _TYPE_SENSITIVE_PAIRS)
def test_context_definition_binding_is_type_sensitive_before_spawn(
    tmp_path,
    context_value,
    definition_value,
):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(actions=session.actions)
    context, definition, manifest = experiment_bundle(
        tmp_path,
        public_parameters={"typed_value": context_value},
    )
    definition = replace(
        definition,
        public_parameters={"typed_value": definition_value},
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True
    controller.start()
    result = controller.wait(timeout_s=2)

    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_CONTEXT_INVALID"


@pytest.mark.parametrize("passed_value,file_value", _TYPE_SENSITIVE_PAIRS)
def test_passed_and_file_manifest_binding_is_type_sensitive_before_spawn(
    tmp_path,
    passed_value,
    file_value,
):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(actions=session.actions)
    context, definition, manifest = experiment_bundle(tmp_path)
    manifest["typed_value"] = passed_value
    file_manifest = json.loads(json.dumps(manifest))
    file_manifest["typed_value"] = file_value
    definition.scene_manifest.write_text(
        json.dumps(file_manifest),
        encoding="utf-8",
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True
    controller.start()
    result = controller.wait(timeout_s=2)

    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_CONTEXT_INVALID"


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_manifest_schema_version_requires_exact_integer_one_before_spawn(
    tmp_path,
    schema_version,
):
    session = FakeSession()
    session.application.sim = ProbeSim.motion(actions=session.actions)
    context, definition, manifest = experiment_bundle(tmp_path)
    manifest["schema_version"] = schema_version
    definition.scene_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )

    class NoSpawnContext:
        def Pipe(self, *args, **kwargs):
            raise AssertionError("student spawn was attempted")

    controller._context = NoSpawnContext()
    assert controller.validate().ok is True
    controller.start()
    result = controller.wait(timeout_s=2)

    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_CONTEXT_INVALID"


def student_processes():
    return {
        child.pid
        for child in multiprocessing.active_children()
        if child.name.startswith("StudentProgram-")
    }


def strict_json(value):
    return json.loads(
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    )


def test_result_and_snapshot_are_frozen_value_objects(tmp_path):
    result = StudentRunResult("PASS", None, tmp_path, None)
    snapshot = StudentRunSnapshot(
        RunState.EMPTY,
        None,
        0,
        0.0,
        None,
        None,
    )

    with pytest.raises(FrozenInstanceError):
        result.status = "FAILED"
    with pytest.raises(FrozenInstanceError):
        snapshot.command_count = 1
    assert snapshot.tcp_mm is None


def test_snapshot_tcp_is_copied_to_an_immutable_tuple(tmp_path):
    source_tcp = [100, -20.5, 35.25]
    snapshot = StudentRunSnapshot(
        RunState.RUNNING,
        "robot.pose",
        1,
        0.1,
        None,
        tmp_path,
        source_tcp,
    )

    source_tcp[0] = 999

    assert snapshot.tcp_mm == (100.0, -20.5, 35.25)
    assert isinstance(snapshot.tcp_mm, tuple)
    with pytest.raises(FrozenInstanceError):
        snapshot.tcp_mm = None


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_snapshot_tcp_rejects_non_finite_values(
    tmp_path,
    non_finite,
):
    with pytest.raises(ValueError, match="finite"):
        StudentRunSnapshot(
            RunState.RUNNING,
            "robot.pose",
            1,
            0.1,
            None,
            tmp_path,
            (100.0, non_finite, 35.25),
        )


def test_controller_snapshots_publish_last_pose_without_ui_queries(
    tmp_path,
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.pose()\n",
    )
    snapshots: list[StudentRunSnapshot] = []
    controller.subscribe(snapshots.append)
    assert controller.validate().ok is True

    controller.start()
    assert controller.wait(timeout_s=5).status == "PASS"

    assert any(
        snapshot.tcp_mm == (100.0, 20.0, 120.0)
        for snapshot in snapshots
    )


def test_wait_for_quiescence_validates_timeout_and_accepts_idle_states(
    tmp_path,
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )

    assert controller.wait_for_quiescence(0) is True
    assert controller.validate().ok is True
    assert controller.wait_for_quiescence(0) is True

    for invalid in (None, True, "1"):
        with pytest.raises(TypeError):
            controller.wait_for_quiescence(invalid)
    for invalid in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            controller.wait_for_quiescence(invalid)


def test_valid_program_runs_commands_and_finishes_pass(tmp_path):
    controller, session, program = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 100, speed=15)\n"
        "    assert ctx.robot.pose() == (100.0, 20.0, 100.0)\n",
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "PASS"
    assert result.error is None
    assert result.evidence_dir is not None
    assert result.summary_path == result.evidence_dir / "summary.json"
    assert (result.evidence_dir / "source.py").read_bytes() == (
        program.read_bytes()
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["command_count"] == 2
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    assert session.application.robot.moves == [
        (100.0, 20.0, 100.0, 15.0)
    ]
    assert controller.process_is_alive is False


def test_invalid_program_never_starts_process(tmp_path):
    controller, _, program = make_controller(
        tmp_path,
        "def main(ctx)\n    pass\n",
    )

    assert controller.validate(program).ok is False
    with pytest.raises(RuntimeError, match="VALIDATED"):
        controller.start()
    assert controller.process_is_alive is False


def test_pause_blocks_commands_and_each_step_releases_exactly_one(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.log('one')\n"
        "    ctx.log('two')\n",
    )
    assert controller.validate().ok is True

    controller.start(paused=True)
    assert controller.state is RunState.PAUSED
    controller.step()
    wait_until(lambda: controller.command_count == 1)
    assert controller.state is RunState.PAUSED
    with pytest.raises(TimeoutError):
        controller.wait(timeout_s=0.02)
    controller.step()

    assert controller.wait(timeout_s=5).status == "PASS"
    assert controller.command_count == 2


def test_resume_clears_step_permits_and_runs_continuously(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.log('one')\n"
        "    ctx.log('two')\n"
        "    ctx.log('three')\n",
    )
    assert controller.validate().ok is True
    controller.start(paused=True)
    controller.step()
    wait_until(lambda: controller.command_count == 1)

    controller.step()
    controller.resume()

    assert controller.wait(timeout_s=5).status == "PASS"
    assert controller.command_count == 3


def test_low_horizontal_move_fails_before_backend_call(tmp_path):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 25, speed=8)\n"
        "    ctx.robot.move_world(120, 20, 25, speed=8)\n",
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "TARGET_OUT_OF_WORKSPACE"
    assert session.application.robot.moves[0] == (
        100.0,
        20.0,
        25.0,
        8.0,
    )
    assert (120.0, 20.0, 25.0, 8.0) not in (
        session.application.robot.moves
    )


def test_cancel_terminates_worker_before_ordered_cleanup(tmp_path):
    actions: list[tuple] = []
    robot = FakeRobot(actions, pose=(100.0, 20.0, 25.0))
    tool = FakeTool(actions)
    session = FakeSession(robot=robot, tool=tool)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    while True:\n"
        "        pass\n",
        session=session,
    )
    tool.alive_probe = lambda: controller.process_is_alive
    assert controller.validate().ok is True
    controller.start()
    wait_until(lambda: controller.process_is_alive)
    assert controller.wait_for_quiescence(0) is False

    controller.cancel()
    result = controller.wait(timeout_s=5)

    assert result.status == "CANCELLED"
    assert controller.wait_for_quiescence(2) is True
    assert controller.process_is_alive is False
    assert tool.cleanup_liveness == [False]
    assert actions == [
        ("tool.off",),
        ("move", 100.0, 20.0, 100.0),
        ("home",),
    ]


def test_runtime_limit_terminates_infinite_program(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    while True:\n"
        "        pass\n",
        execution_policy=policy(max_runtime_s=0.2),
    )
    assert controller.validate().ok is True
    controller.start()

    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_RUNTIME_TIMEOUT"
    assert controller.process_is_alive is False


def test_real_backend_is_rejected_before_evidence_or_process(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
        backend="real",
    )
    assert controller.validate().ok is True

    with pytest.raises(RuntimeError, match="REAL_BACKEND_NOT_AUTHORIZED"):
        controller.start()

    assert controller.process_is_alive is False
    assert not (tmp_path / "runs").exists()
    assert controller.state is RunState.VALIDATED


def test_command_limit_fails_without_accepting_extra_command(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.log('one')\n"
        "    ctx.log('two')\n",
        execution_policy=policy(max_commands=1),
    )
    assert controller.validate().ok is True
    controller.start()

    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_COMMAND_LIMIT_EXCEEDED"
    assert controller.command_count == 1


def test_sleep_can_be_cancelled_without_waiting_for_full_duration(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.sleep(0.2)\n",
    )
    assert controller.validate().ok is True
    controller.start()
    wait_until(lambda: controller.command_count == 1)
    started = time.monotonic()

    controller.cancel()
    result = controller.wait(timeout_s=2)

    assert result.status == "CANCELLED"
    assert time.monotonic() - started < 0.18


def test_cleanup_errors_do_not_overwrite_original_failure(tmp_path):
    actions: list[tuple] = []
    robot = FakeRobot(actions, fail_home=True)
    tool = FakeTool(actions, fail_off=True)
    session = FakeSession(robot=robot, tool=tool)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    raise ValueError('original-student-error')\n",
        session=session,
    )
    assert controller.validate().ok is True
    controller.start()

    result = controller.wait(timeout_s=5)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_PROGRAM_FAILED"
    assert "original-student-error" in result.error["message"]
    assert [item["stage"] for item in summary["cleanup_errors"]] == [
        "tool.off",
        "robot.move_home",
    ]


def test_command_evidence_failure_stops_before_backend_and_fails(
    tmp_path, monkeypatch
):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
    )
    assert controller.validate().ok is True

    def fail_record(*args, **kwargs):
        raise OSError("command-evidence-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "record_command",
        fail_record,
    )
    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert session.application.robot.home_calls == 1
    assert controller.command_count == 1


def test_finalize_failure_changes_pass_to_failed_without_summary(
    tmp_path, monkeypatch
):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    assert controller.validate().ok is True

    def fail_finalize(*args, **kwargs):
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert result.summary_path is None
    assert session.application.tool.off_calls == 1
    assert session.application.robot.home_calls == 1


def test_snapshot_is_revalidated_after_original_changes(tmp_path):
    controller, session, program = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
    )
    assert controller.validate().ok is True
    program.write_text("def main(ctx)\n    pass\n", encoding="utf-8")

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_SNAPSHOT_INVALID"
    assert controller.process_is_alive is False
    assert session.application.robot.home_calls == 1
    assert result.evidence_dir.joinpath("source.py").read_bytes() == (
        program.read_bytes()
    )


def test_student_exception_and_worker_exit_without_result_are_stable(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    raise LookupError('student-broke')\n",
    )
    assert controller.validate().ok is True
    controller.start()
    result = controller.wait(timeout_s=5)
    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_PROGRAM_FAILED"
    assert result.error["type"] == "LookupError"

    controller2, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    import os\n"
        "    os._exit(23)\n",
    )
    assert controller2.validate().ok is True
    controller2.start()
    result2 = controller2.wait(timeout_s=5)
    assert result2.status == "FAILED"
    assert result2.error["code"] == "STUDENT_WORKER_RESULT_MISSING"
    assert result2.error["exitcode"] == 23


def test_repeated_cancel_and_wait_are_idempotent(tmp_path):
    before = student_processes()
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    while True:\n"
        "        pass\n",
    )
    assert controller.validate().ok is True
    controller.start()
    wait_until(lambda: controller.process_is_alive)

    controller.cancel()
    controller.cancel()
    first = controller.wait(timeout_s=5)
    second = controller.wait(timeout_s=0)
    controller.cancel()

    assert first is second
    assert first.status == "CANCELLED"
    assert student_processes() == before


def test_reset_replaces_application_and_returns_to_validated(tmp_path):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    assert controller.validate().ok is True
    old_application = session.application
    controller.start()
    assert controller.wait(timeout_s=5).status == "PASS"

    replacement = controller.reset()

    assert replacement is session.application
    assert replacement is not old_application
    assert session.reset_calls == 1
    assert controller.state is RunState.VALIDATED
    controller.start()
    assert controller.wait(timeout_s=5).status == "PASS"


def test_subscriber_exceptions_do_not_break_run_and_unsubscribe_is_idempotent(
    tmp_path,
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.log('hello')\n",
    )
    snapshots: list[StudentRunSnapshot] = []
    unsubscribe_bad = controller.subscribe(
        lambda snapshot: (_ for _ in ()).throw(RuntimeError("observer-broke"))
    )
    unsubscribe_good = controller.subscribe(snapshots.append)
    assert controller.validate().ok is True

    controller.start()
    assert controller.wait(timeout_s=5).status == "PASS"
    unsubscribe_bad()
    unsubscribe_bad()
    unsubscribe_good()
    unsubscribe_good()

    assert snapshots
    assert snapshots[-1].state is RunState.PASSED
    assert snapshots[-1].command_count == 1


def test_protocol_argument_smuggling_is_rejected_before_backend(tmp_path):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx._rpc.connection.send({\n"
        "        'schema_version': 1,\n"
        "        'kind': 'command',\n"
        "        'command_id': 'evil',\n"
        "        'name': 'robot.home',\n"
        "        'args': {'unexpected': True},\n"
        "    })\n"
        "    ctx._rpc.connection.recv()\n",
    )
    assert controller.validate().ok is True
    controller.start()

    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_COMMAND_ARGUMENTS_INVALID"
    assert session.application.robot.home_calls == 1


def test_invalid_backend_pose_is_protocol_failure_not_success(tmp_path):
    robot = FakeRobot(pose=(float("nan"), 20.0, 120.0))
    session = FakeSession(robot=robot)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.pose()\n",
        session=session,
    )
    assert controller.validate().ok is True
    controller.start()

    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "ROBOT_POSE_INVALID"
    json.dumps(result.error, allow_nan=False)


def test_command_timeout_reaps_child_before_cleanup_and_backend_release(
    tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    actions: list[tuple] = []
    robot = FakeRobot(
        actions,
        move_entered=entered,
        move_release=release,
    )
    tool = FakeTool(actions)
    session = FakeSession(robot=robot, tool=tool)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 100, speed=8)\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    tool.alive_probe = lambda: controller.process_is_alive
    assert controller.validate().ok is True
    controller.start()
    assert entered.wait(timeout=2)

    wait_until(lambda: not controller.process_is_alive, timeout_s=2)
    assert ("tool.off",) not in actions
    release.set()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_COMMAND_TIMEOUT"
    assert tool.cleanup_liveness == [False]
    assert actions[-2:] == [("tool.off",), ("home",)]


def test_evidence_creation_failure_is_terminal_without_process(
    tmp_path, monkeypatch
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    assert controller.validate().ok is True

    def fail_create(*args, **kwargs):
        raise OSError("cannot-create-evidence")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "create",
        fail_create,
    )
    controller.start()
    result = controller.wait(timeout_s=1)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert result.evidence_dir is None
    assert result.summary_path is None
    assert controller.process_is_alive is False


def test_worker_is_force_reaped_after_returning_result_with_lingering_thread(
    tmp_path,
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    import threading\n"
        "    import time\n"
        "    thread = threading.Thread(target=lambda: time.sleep(2))\n"
        "    thread.start()\n",
    )
    assert controller.validate().ok is True
    controller.start()

    try:
        result = controller.wait(timeout_s=5)
        assert result.status == "PASS"
        assert controller.process_is_alive is False
    finally:
        if controller.process_is_alive:
            controller._process.terminate()
            controller._process.join(timeout=2)


def test_initial_event_evidence_failure_never_spawns_worker(
    tmp_path, monkeypatch
):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
    )
    assert controller.validate().ok is True
    before = student_processes()

    def fail_event(*args, **kwargs):
        raise OSError("event-evidence-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "record_event",
        fail_event,
    )
    controller.start()
    result = controller.wait(timeout_s=2)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert controller.process_is_alive is False
    assert student_processes() == before
    assert session.application.robot.home_calls == 1


def test_spawn_failure_is_recorded_and_does_not_escape_or_leak(
    tmp_path, monkeypatch
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    assert controller.validate().ok is True
    before = student_processes()

    class StartFailingProcess:
        def start(self):
            raise TypeError("injected process start failure")

        def is_alive(self):
            return False

        def join(self, timeout=None):
            del timeout

    context = controller._context
    monkeypatch.setattr(
        controller,
        "_context",
        SimpleNamespace(
            Pipe=context.Pipe,
            Queue=context.Queue,
            Process=lambda **_kwargs: StartFailingProcess(),
        ),
    )

    controller.start()
    result = controller.wait(timeout_s=2)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_WORKER_START_FAILED"
    assert controller.process_is_alive is False
    assert student_processes() == before


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (
            "{'schema_version': 1, 'kind': 'command', "
            "'command_id': 'evil', 'name': 'sim.call', 'args': {}}",
            "COMMAND_NOT_ALLOWED",
        ),
        (
            "{'schema_version': 1, 'kind': 'command', "
            "'command_id': 'evil', 'name': 'robot.home', 'args': []}",
            "STUDENT_PROTOCOL_ERROR",
        ),
        (
            "{'schema_version': 99, 'kind': 'command', "
            "'command_id': 'evil', 'name': 'robot.home', 'args': {}}",
            "PROTOCOL_VERSION_UNSUPPORTED",
        ),
        (
            "{'schema_version': 1, 'kind': 'event', "
            "'command_id': 'evil', 'name': 'robot.home', 'args': {}}",
            "PROTOCOL_KIND_INVALID",
        ),
    ],
)
def test_malformed_or_unknown_protocol_is_stable_and_never_dispatched(
    tmp_path,
    payload,
    expected_code,
):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        f"    ctx._rpc.connection.send({payload})\n"
        "    ctx._rpc.connection.recv()\n",
    )
    assert controller.validate().ok is True
    controller.start()

    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == expected_code
    assert session.application.robot.home_calls == 1
    json.dumps(result.error, allow_nan=False)


def test_finalize_is_attempted_exactly_once(tmp_path, monkeypatch):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    assert controller.validate().ok is True
    original = runner_module.StudentRunEvidence.finalize
    calls = 0

    def counting_finalize(self, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, **kwargs)

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        counting_finalize,
    )
    controller.start()

    first = controller.wait(timeout_s=5)
    second = controller.wait(timeout_s=0)

    assert first is second
    assert first.status == "PASS"
    assert calls == 1


def test_pipe_eof_does_not_beat_delayed_dedicated_worker_result(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    import time\n"
        "    ctx._rpc.connection.close()\n"
        "    time.sleep(0.4)\n",
    )
    assert controller.validate().ok is True
    controller.start()

    result = controller.wait(timeout_s=5)

    assert result.status == "PASS"
    assert controller.process_is_alive is False


def test_timeout_during_evidence_write_never_reaches_student_backend(
    tmp_path, monkeypatch
):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True
    original = runner_module.StudentRunEvidence.record_command

    def delayed_record(self, payload):
        original(self, payload)
        deadline = time.monotonic() + 2
        while controller.process_is_alive and time.monotonic() < deadline:
            threading.Event().wait(0.01)

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "record_command",
        delayed_record,
    )
    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_COMMAND_TIMEOUT"
    # The only home is failure cleanup; the timed-out student command did not
    # cross the parent-side backend boundary.
    assert session.application.robot.home_calls == 1


def test_command_timeout_remains_primary_when_backend_later_raises(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    robot = FakeRobot(
        move_entered=entered,
        move_release=release,
        fail_move=True,
    )
    session = FakeSession(robot=robot)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 100, speed=8)\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True
    controller.start()
    assert entered.wait(timeout=2)
    wait_until(lambda: not controller.process_is_alive, timeout_s=2)

    release.set()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_COMMAND_TIMEOUT"


def test_runner_validates_and_executes_the_same_captured_source_bytes(
    tmp_path, monkeypatch
):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
    )
    assert controller.validate().ok is True
    original_validate = runner_module.validate_program_bytes

    def validate_then_replace(payload, path):
        result = original_validate(payload, path)
        Path(path).write_text(
            "def main(ctx):\n"
            "    ctx.robot.move_world(120, 20, 120, speed=8)\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        runner_module,
        "validate_program_bytes",
        validate_then_replace,
    )
    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "PASS"
    assert session.application.robot.home_calls == 1
    assert session.application.robot.moves == []


def test_evidence_source_hash_mismatch_is_rejected_before_spawn(
    tmp_path, monkeypatch
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    assert controller.validate().ok is True
    original_create = runner_module.StudentRunEvidence.create
    before = student_processes()

    def create_then_corrupt(**kwargs):
        evidence = original_create(**kwargs)
        evidence.source_path.write_text(
            "def main(ctx):\n    ctx.robot.home()\n",
            encoding="utf-8",
        )
        return evidence

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "create",
        create_then_corrupt,
    )
    controller.start()
    result = controller.wait(timeout_s=2)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_HASH_MISMATCH"
    assert controller.process_is_alive is False
    assert student_processes() == before


def test_finalize_failure_reports_cleanup_errors_from_degraded_pass(
    tmp_path, monkeypatch
):
    actions: list[tuple] = []
    robot = FakeRobot(actions, fail_home=True)
    tool = FakeTool(actions, fail_off=True)
    session = FakeSession(robot=robot, tool=tool)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
    )
    assert controller.validate().ok is True

    def fail_finalize(*args, **kwargs):
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.summary_path is None
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    cleanup_errors = result.error["details"]["cleanup_errors"]
    assert [item["stage"] for item in cleanup_errors] == [
        "tool.off",
        "robot.move_home",
    ]


def test_result_and_snapshot_errors_are_deeply_immutable_and_json_safe(
    tmp_path,
):
    source_error = {
        "code": "BROKEN",
        "message": "broken",
        "details": {"items": [{"value": 1}]},
    }
    result = StudentRunResult("FAILED", None, tmp_path, source_error)
    snapshot = StudentRunSnapshot(
        RunState.FAILED,
        None,
        0,
        0.0,
        source_error,
        tmp_path,
    )
    source_error["code"] = "MUTATED"
    source_error["details"]["items"][0]["value"] = 99

    for frozen_error in (result.error, snapshot.error):
        assert frozen_error.get("code") == "BROKEN"
        assert frozen_error["details"]["items"][0]["value"] == 1
        with pytest.raises(TypeError):
            frozen_error["code"] = "changed"
        with pytest.raises(TypeError):
            frozen_error["details"]["items"][0]["value"] = 2
        with pytest.raises(AttributeError):
            frozen_error["details"]["items"].append({"value": 3})
        json.dumps(frozen_error, ensure_ascii=False, allow_nan=False)


def test_permanently_blocked_backend_returns_fail_closed_quarantine(
    tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    actions: list[tuple] = []
    robot = FakeRobot(
        actions,
        move_entered=entered,
        move_release=release,
    )
    tool = FakeTool(actions)
    old_socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(socket=old_socket)
    replacement_client = ProtocolFaithfulClient(
        socket=ProtocolSocket(rcvtimeo=9000, sndtimeo=11000)
    )
    session = RebuildingFakeSession(
        robot=robot,
        tool=tool,
        client=client,
        replacement_client=replacement_client,
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 100, speed=8)\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True
    controller.start()
    assert entered.wait(timeout=2)

    result = None
    try:
        result = controller.wait(timeout_s=0.75)
        assert result.status == "FAILED"
        assert result.error["code"] == "STUDENT_COMMAND_TIMEOUT"
        cleanup_errors = result.error["details"]["cleanup_errors"]
        assert cleanup_errors[0]["error"]["code"] == (
            "STUDENT_BACKEND_COMMAND_STUCK"
        )
        summary = json.loads(
            result.summary_path.read_text(encoding="utf-8")
        )
        assert summary["cleanup_errors"][0]["error"]["code"] == (
            "STUDENT_BACKEND_COMMAND_STUCK"
        )
        assert controller.process_is_alive is False
        assert client.timeout == 600.0
        assert old_socket.RCVTIMEO == 100
        assert old_socket.SNDTIMEO == 100
        assert tool.off_calls == 0
        assert robot.home_calls == 0
        with pytest.raises(RuntimeError, match="BACKEND_COMMAND_STUCK"):
            controller.reset()
    finally:
        release.set()
        if result is None:
            try:
                controller.wait(timeout_s=3)
            except (RuntimeError, TimeoutError):
                pass

    wait_until(
        lambda: not controller._backend_action_is_alive(),
        timeout_s=2,
    )
    assert tool.off_calls == 0
    assert robot.home_calls == 0
    assert controller.reset() is session.application
    assert session.quarantined_reset_calls == 1
    assert session.reset_calls == 0
    assert old_socket.RCVTIMEO == 100
    assert old_socket.SNDTIMEO == 100
    assert session.application.client is replacement_client


def test_connected_client_and_socket_timeouts_are_bounded_and_restored(
    tmp_path,
):
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(socket=socket, send_count=2)
    assert "timeout" not in client.next_request()
    session = FakeSession(client=client)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    while True:\n"
        "        pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.2001),
    )
    assert controller.validate().ok is True

    controller.start()
    assert client.timeout == 0.2001
    assert socket.RCVTIMEO == 201
    assert socket.SNDTIMEO == 201
    controller.cancel()
    assert controller.wait(timeout_s=3).status == "CANCELLED"

    assert client.timeout == 600.0
    assert socket.RCVTIMEO == 5000
    assert socket.SNDTIMEO == 7000


def test_socket_timeout_partial_set_failure_rolls_back_and_fails_closed(
    tmp_path,
):
    socket = ProtocolSocket(
        rcvtimeo=5000,
        sndtimeo=7000,
        fail_on_set="SNDTIMEO",
    )
    client = ProtocolFaithfulClient(socket=socket)
    session = FakeSession(client=client)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.2),
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_TRANSPORT_TIMEOUT_GUARD_FAILED"
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert controller.process_is_alive is False
    assert client.timeout == 600.0
    assert socket.RCVTIMEO == 5000
    assert socket.SNDTIMEO == 7000
    assert session.application.tool.off_calls == 0
    assert session.application.robot.home_calls == 0


def test_transport_timeout_quarantines_connection_and_reset_rebuilds_client(
    tmp_path,
):
    actions: list[tuple] = []
    robot = FakeRobot(actions, move_error=ProtocolAgain())
    tool = FakeTool(actions)
    old_socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    old_client = ProtocolFaithfulClient(socket=old_socket)
    replacement_client = ProtocolFaithfulClient(
        socket=ProtocolSocket(rcvtimeo=9000, sndtimeo=11000)
    )
    session = RebuildingFakeSession(
        robot=robot,
        tool=tool,
        client=old_client,
        replacement_client=replacement_client,
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 100, speed=8)\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.2),
    )
    assert controller.validate().ok is True

    started = time.monotonic()
    controller.start()
    result = controller.wait(timeout_s=2)

    assert time.monotonic() - started < 1
    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_BACKEND_TRANSPORT_TIMEOUT"
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert controller.process_is_alive is False
    assert tool.off_calls == 0
    assert robot.home_calls == 0
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["error"]["details"]["quarantined"] is True
    assert summary["error"]["details"]["connection_unusable"] is True
    assert summary["cleanup_errors"][0]["stage"] == "backend.connection"
    assert old_client.timeout == 600.0
    assert old_socket.RCVTIMEO == 5000
    assert old_socket.SNDTIMEO == 7000
    with pytest.raises(RuntimeError, match="VALIDATED"):
        controller.start()

    assert controller.reset() is session.application
    assert session.application.client is replacement_client
    assert controller.state is RunState.VALIDATED


def test_profile_command_rollback_failure_quarantines_terminal_flow(
    tmp_path,
    monkeypatch,
):
    class RollbackGateway:
        def __init__(self):
            self.dispatch_calls = []
            self.reset_calls = 0
            self.final_probe_errors = 0

        def collect_probe(self, phase):
            return {"phase": phase, "status": "PASS"}

        def record_probe_report(self, phase, report):
            assert report["phase"] == phase
            return dict(report)

        def record_probe_error(self, phase, **kwargs):
            assert phase == "final"
            self.final_probe_errors += 1
            return {"phase": phase, "status": "ERROR", "error": kwargs}

        def dispatch(self, name, args):
            self.dispatch_calls.append((name, dict(args)))
            if name != "camera.profile.apply":
                raise AssertionError(f"unexpected command: {name}")
            rollback = RuntimeError("VISION_PROFILE_ROLLBACK_FAILED")
            failure = RuntimeError("VISION_PROFILE_APPLY_FAILED")
            failure.__cause__ = ExceptionGroup("rollback", [rollback])
            raise failure

        def reset_environment(self):
            self.reset_calls += 1
            raise AssertionError("quarantined cleanup touched the backend")

    gateway = RollbackGateway()
    monkeypatch.setattr(
        runner_module,
        "StudentExperimentGateway",
        lambda **kwargs: gateway,
    )
    session = FakeSession()
    context, definition, manifest = experiment_bundle(
        tmp_path,
        experiment_id="V1-01",
    )
    definition = replace(
        definition,
        capabilities=(
            "camera.rgb",
            "camera.profile",
            "lighting.profile",
            "scene.probe",
        ),
    )
    controller, _, _ = make_controller(
        tmp_path,
        (
            "def main(ctx):\n"
            "    ctx.camera.apply_profile('wide_dim')\n"
        ),
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    reap_calls = []
    original_reap_child = controller._reap_child

    def record_reap(*, force):
        reap_calls.append(force)
        return original_reap_child(force=force)

    monkeypatch.setattr(controller, "_reap_child", record_reap)
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == (
        "STUDENT_BACKEND_CONNECTION_QUARANTINED"
    )
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert result.error["details"]["cause"]["message"] == (
        "VISION_PROFILE_APPLY_FAILED"
    )
    assert result.error["details"]["rollback_failure"] == (
        "VISION_PROFILE_ROLLBACK_FAILED"
    )
    assert controller._backend_quarantined is True
    assert controller._stop_requested is True
    assert controller._requested_status == "FAILED"
    assert reap_calls.count(True) >= 2
    assert controller.process_is_alive is False
    assert gateway.dispatch_calls == [
        ("camera.profile.apply", {"profile_id": "wide_dim"})
    ]
    assert gateway.reset_calls == 0
    assert gateway.final_probe_errors == 1
    assert session.application.tool.off_calls == 0
    assert session.application.robot.pose_calls == 0
    assert session.application.robot.home_calls == 0
    with pytest.raises(RuntimeError, match="隔离"):
        controller.ensure_experiment_switch_allowed()


def test_real_profile_controller_gateway_runner_quarantines_interrupt_rollback(
    tmp_path,
    monkeypatch,
):
    class RollbackFailingSim:
        handle_world = -1
        visionintparam_resolution_x = "resolution_x"
        visionintparam_resolution_y = "resolution_y"
        visionfloatparam_perspective_angle = "perspective_angle"
        visionfloatparam_near_clipping = "near_clip"
        visionfloatparam_far_clipping = "far_clip"

        def __init__(self):
            self.ints = {"resolution_x": 512, "resolution_y": 512}
            self.floats = {
                "perspective_angle": 1.0471975511965976,
                "near_clip": 0.05,
                "far_clip": 2.0,
            }
            self.rig_position = [0.0, 0.0, 0.7]
            self.lights = {
                "/VisionQualityLab/Lighting/KeyLight": (
                    1,
                    [0.8, 0.8, 0.8],
                    [0.0, 0.0, 0.0],
                ),
                "/VisionQualityLab/Lighting/FillLight": (
                    1,
                    [0.35, 0.35, 0.35],
                    [0.0, 0.0, 0.0],
                ),
            }
            self.interrupt = KeyboardInterrupt("camera interrupted")
            self.rollback_started = False
            self.rollback_failure_injected = False
            self.set_attempts = 0
            self.vision_reads = 0

        def getObject(self, path):
            if path == "/VisionQualityLab/CameraRig/Camera":
                return 1
            return path

        def getObjectInt32Param(self, handle, parameter):
            return self.ints[parameter]

        def getObjectFloatParam(self, handle, parameter):
            return self.floats[parameter]

        def getObjectPosition(self, handle, relative_to):
            return list(self.rig_position)

        def getLightParameters(self, handle):
            enabled, diffuse, specular = self.lights[handle]
            return enabled, [0.0, 0.0, 0.0], list(diffuse), list(specular)

        def setObjectInt32Param(self, handle, parameter, value):
            self.set_attempts += 1
            self.ints[parameter] = value

        def setObjectFloatParam(self, handle, parameter, value):
            self.set_attempts += 1
            self.floats[parameter] = value

        def setObjectPosition(self, handle, relative_to, value):
            self.set_attempts += 1
            self.rig_position = list(value)

        def setLightParameters(
            self,
            handle,
            enabled,
            ambient,
            diffuse,
            specular,
        ):
            self.set_attempts += 1
            if self.rollback_started and not self.rollback_failure_injected:
                self.rollback_failure_injected = True
                raise RuntimeError("rollback-set-failed")
            self.lights[handle] = (
                enabled,
                list(diffuse),
                list(specular),
            )

        def getVisionSensorImg(self, handle):
            self.vision_reads += 1
            self.rollback_started = True
            raise self.interrupt

    sim = RollbackFailingSim()
    camera = CoppeliaSimCamera(
        sensor_path="/VisionQualityLab/CameraRig/Camera",
        sim=sim,
    )
    catalog = VisionProfileCatalog(
        baseline_profile_id="standard",
        sensor_path="/VisionQualityLab/CameraRig/Camera",
        camera_rig_path="/VisionQualityLab/CameraRig",
        key_light_path="/VisionQualityLab/Lighting/KeyLight",
        fill_light_path="/VisionQualityLab/Lighting/FillLight",
        near_clip_m=0.05,
        far_clip_m=2.0,
        profiles=(
            VisionProfile(
                "standard",
                "standard",
                (512, 512),
                60,
                0.7,
                (0.8, 0.8, 0.8),
                (0.35, 0.35, 0.35),
            ),
            VisionProfile(
                "wide_dim",
                "wide dim",
                (256, 256),
                75,
                0.8,
                (0.35, 0.35, 0.35),
                (0.15, 0.15, 0.15),
            ),
        ),
    )
    real_profile_controller = profile_controller_module.VisionProfileController(
        sim=sim,
        camera=camera,
        catalog=catalog,
        allowed_profile_ids=("standard", "wide_dim"),
    )
    monkeypatch.setattr(
        gateway_module,
        "controller_for_experiment",
        lambda application, definition, manifest: real_profile_controller,
    )

    session = FakeSession()
    session.application.config.camera_backend = "sim"
    session.application.sim = sim
    session.application.camera = camera
    parameters = {
        "baseline_profile_id": "standard",
        "allowed_profile_ids": ["standard", "wide_dim"],
    }
    context, definition, manifest = experiment_bundle(
        tmp_path,
        experiment_id="V1-01",
        public_parameters=parameters,
    )
    definition = replace(
        definition,
        capabilities=(
            "camera.rgb",
            "camera.profile",
            "lighting.profile",
            "scene.probe",
        ),
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.camera.apply_profile('wide_dim')\n",
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == (
        "STUDENT_BACKEND_CONNECTION_QUARANTINED"
    )
    assert controller._backend_quarantined is True
    assert sim.interrupt.vision_profile_rollback_failure == (
        "VISION_PROFILE_ROLLBACK_FAILED"
    )
    assert sim.rollback_failure_injected is True
    assert sim.vision_reads == 1
    assert sim.set_attempts == 16
    assert session.application.tool.off_calls == 0
    assert session.application.robot.pose_calls == 0
    assert session.application.robot.home_calls == 0


@pytest.mark.parametrize(
    (
        "source",
        "expected_code",
        "allowed_profile_ids",
        "sim_options",
    ),
    [
        (
            "def main(ctx):\n    ctx.camera.apply_profile('missing')\n",
            "VISION_PROFILE_ID_INVALID",
            ("standard", "wide_dim"),
            {},
        ),
        (
            "def main(ctx):\n    ctx.camera.apply_profile('wide_dim')\n",
            "VISION_PROFILE_NOT_ALLOWED",
            ("standard",),
            {},
        ),
        (
            "def main(ctx):\n    ctx.camera.apply_profile('wide_dim')\n",
            "VISION_PROFILE_APPLY_FAILED",
            ("standard", "wide_dim"),
            {"read_error": True},
        ),
        (
            "def main(ctx):\n    ctx.camera.reset_profile()\n",
            "VISION_PROFILE_RESET_FAILED",
            ("standard", "wide_dim"),
            {"read_error": True},
        ),
        (
            "def main(ctx):\n    ctx.camera.get_profile()\n",
            "VISION_PROFILE_BACKEND_UNAVAILABLE",
            ("standard", "wide_dim"),
            {"read_error": True},
        ),
        (
            "def main(ctx):\n    ctx.camera.get_profile()\n",
            "VISION_PROFILE_READBACK_MISMATCH",
            ("standard", "wide_dim"),
            {"mismatch": True},
        ),
    ],
)
def test_real_profile_errors_keep_codes_through_gateway_and_runner(
    tmp_path,
    source,
    expected_code,
    allowed_profile_ids,
    sim_options,
):
    sim = FunctionalProfileSim(**sim_options)
    camera = CoppeliaSimCamera(
        sensor_path="/VisionQualityLab/CameraRig/Camera",
        sim=sim,
    )
    session = FakeSession()
    session.application.config.camera_backend = "sim"
    session.application.sim = sim
    session.application.camera = camera
    context, definition, manifest = profile_factory_bundle(
        tmp_path,
        allowed_profile_ids=allowed_profile_ids,
    )
    controller, _, _ = make_controller(
        tmp_path,
        source,
        session=session,
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == expected_code
    assert type(controller._experiment_gateway) is (
        gateway_module.StudentExperimentGateway
    )
    assert isinstance(
        controller._experiment_gateway._profile_controller,
        profile_controller_module.VisionProfileController,
    )


def test_non_transport_backend_failure_restores_socket_timeouts(tmp_path):
    actions: list[tuple] = []
    robot = FakeRobot(actions, fail_move=True)
    tool = FakeTool(actions)
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(socket=socket)
    session = FakeSession(robot=robot, tool=tool, client=client)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 100, speed=8)\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.2),
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_COMMAND_FAILED"
    assert client.timeout == 600.0
    assert socket.RCVTIMEO == 5000
    assert socket.SNDTIMEO == 7000


def test_cleanup_tool_transport_timeout_quarantines_and_stops_old_connection(
    tmp_path,
):
    actions: list[tuple] = []
    robot = FakeRobot(actions, pose=(100.0, 20.0, 25.0))
    tool = FakeTool(actions, off_error=ProtocolAgain())
    old_socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    old_client = ProtocolFaithfulClient(socket=old_socket)
    replacement_client = ProtocolFaithfulClient()
    session = RebuildingFakeSession(
        robot=robot,
        tool=tool,
        client=old_client,
        replacement_client=replacement_client,
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    raise ValueError('original-student-error')\n",
        session=session,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_PROGRAM_FAILED"
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert controller.process_is_alive is False
    assert controller._backend_quarantined is True
    assert tool.off_calls == 1
    assert robot.pose_calls == 0
    assert robot.moves == []
    assert robot.home_calls == 0
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["last_pose_mm"] is None
    assert summary["error"]["details"]["quarantined"] is True
    assert summary["error"]["details"]["connection_unusable"] is True
    assert [item["stage"] for item in summary["cleanup_errors"]] == [
        "tool.off",
        "backend.connection",
    ]
    assert old_client.timeout == 600.0
    assert old_socket.RCVTIMEO == 5000
    assert old_socket.SNDTIMEO == 7000

    assert controller.reset() is session.application
    assert session.application.client is replacement_client
    assert controller.state is RunState.VALIDATED


def test_quarantined_reset_fails_closed_without_safe_session_contract(
    tmp_path,
):
    actions: list[tuple] = []
    robot = FakeRobot(actions)
    tool = FakeTool(actions, off_error=ProtocolAgain())
    session = FakeSession(robot=robot, tool=tool)
    session.reset_simulation_quarantined = None
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    raise ValueError('original-student-error')\n",
        session=session,
    )
    assert controller.validate().ok is True
    controller.start()
    result = controller.wait(timeout_s=3)
    assert result.error["details"]["quarantined"] is True

    with pytest.raises(
        RuntimeError,
        match="QUARANTINED_RESET_UNAVAILABLE",
    ):
        controller.reset()

    assert session.reset_calls == 0
    assert controller.state is RunState.FAILED


def test_cleanup_safe_lift_transport_timeout_skips_home(tmp_path):
    actions: list[tuple] = []
    robot = FakeRobot(
        actions,
        pose=(100.0, 20.0, 25.0),
        move_error=ProtocolAgain(),
    )
    tool = FakeTool(actions)
    session = FakeSession(robot=robot, tool=tool)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    raise ValueError('original-student-error')\n",
        session=session,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert controller.process_is_alive is False
    assert tool.off_calls == 1
    assert robot.pose_calls == 1
    assert len(robot.moves) == 0
    assert robot.home_calls == 0
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert [item["stage"] for item in summary["cleanup_errors"]] == [
        "robot.safe_lift",
        "backend.connection",
    ]


def test_permanently_blocked_cleanup_is_bounded_and_quarantined(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def block_cleanup():
        entered.set()
        assert release.wait(timeout=5)

    actions: list[tuple] = []
    robot = FakeRobot(actions)
    tool = FakeTool(actions, off_hook=block_cleanup)
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(socket=socket)
    replacement_client = ProtocolFaithfulClient()
    session = RebuildingFakeSession(
        robot=robot,
        tool=tool,
        client=client,
        replacement_client=replacement_client,
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    raise ValueError('original-student-error')\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.05),
    )
    assert controller.validate().ok is True
    controller.start()
    assert entered.wait(timeout=2)
    wait_started = time.monotonic()

    result = None
    sealed_summary = None
    sealed_error = None
    try:
        result = controller.wait(timeout_s=0.6)
        assert time.monotonic() - wait_started < 0.4
        assert result.status == "FAILED"
        assert result.error["code"] == "STUDENT_BACKEND_COMMAND_STUCK"
        assert result.error["details"]["quarantined"] is True
        assert result.error["details"]["connection_unusable"] is True
        assert controller.process_is_alive is False
        assert controller.wait_for_quiescence(0) is False
        assert controller._backend_action_is_alive() is True
        assert controller._backend_action_thread.daemon is True
        assert controller._backend_action_thread.name.startswith(
            "StudentCleanupAction-"
        )
        assert tool.off_calls == 1
        assert robot.pose_calls == 0
        assert robot.home_calls == 0
        assert client.timeout == 600.0
        assert socket.RCVTIMEO == 50
        assert socket.SNDTIMEO == 50
        summary = json.loads(
            result.summary_path.read_text(encoding="utf-8")
        )
        assert summary["error"]["code"] == (
            "STUDENT_BACKEND_COMMAND_STUCK"
        )
        assert [item["stage"] for item in summary["cleanup_errors"]] == [
            "tool.off",
            "backend.connection",
        ]
        sealed_summary = result.summary_path.read_bytes()
        sealed_error = json.dumps(result.error, sort_keys=True)
        with pytest.raises(RuntimeError, match="BACKEND_COMMAND_STUCK"):
            controller.reset()
    finally:
        release.set()
        if result is None:
            try:
                controller.wait(timeout_s=3)
            except (RuntimeError, TimeoutError):
                pass

    wait_until(
        lambda: not controller._backend_action_is_alive(),
        timeout_s=2,
    )
    assert controller.wait_for_quiescence(2) is True
    assert result.summary_path.read_bytes() == sealed_summary
    assert json.dumps(result.error, sort_keys=True) == sealed_error
    assert robot.pose_calls == 0
    assert robot.home_calls == 0

    controller.reset()
    assert session.quarantined_reset_calls == 1
    assert session.reset_calls == 0
    assert socket.RCVTIMEO == 50
    assert socket.SNDTIMEO == 50
    assert session.application.client is replacement_client


def test_finalize_failure_cleanup_transport_timeout_is_quarantined(
    tmp_path,
    monkeypatch,
):
    actions: list[tuple] = []
    robot = FakeRobot(actions, pose=(100.0, 20.0, 25.0))
    tool = FakeTool(actions, off_error=ProtocolAgain())
    session = FakeSession(robot=robot, tool=tool)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
    )
    assert controller.validate().ok is True

    def fail_finalize(*args, **kwargs):
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.summary_path is None
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert [item["stage"] for item in result.error["details"][
        "cleanup_errors"
    ]] == ["tool.off", "backend.connection"]
    assert controller.process_is_alive is False
    assert controller._backend_quarantined is True
    assert tool.off_calls == 1
    assert robot.pose_calls == 0
    assert robot.home_calls == 0


def test_finalize_failure_cleanup_runs_with_transport_timeouts_rebound(
    tmp_path,
    monkeypatch,
):
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(socket=socket)

    def timeout_probe():
        return (client.timeout, socket.RCVTIMEO, socket.SNDTIMEO)

    actions: list[tuple] = []
    robot = FakeRobot(actions, timeout_probe=timeout_probe)
    tool = FakeTool(actions, timeout_probe=timeout_probe)
    session = FakeSession(robot=robot, tool=tool, client=client)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True

    def fail_finalize(*args, **kwargs):
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    result = controller.wait(timeout_s=3)

    bounded = (0.1, 100, 100)
    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert tool.timeout_observations == [bounded]
    assert robot.timeout_observations == [
        ("robot.current_world_pose", bounded),
        ("robot.move_home", bounded),
    ]
    assert client.timeout == 600.0
    assert socket.RCVTIMEO == 5000
    assert socket.SNDTIMEO == 7000


def test_finalize_failure_delayed_transport_timeout_is_bounded(
    tmp_path,
    monkeypatch,
):
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(socket=socket)
    delay: dict[str, float] = {}

    def timeout_probe():
        return (client.timeout, socket.RCVTIMEO, socket.SNDTIMEO)

    def wait_for_transport_timeout():
        started = time.monotonic()
        threading.Event().wait(min(socket.RCVTIMEO / 1000.0, 0.35))
        delay["elapsed"] = time.monotonic() - started

    actions: list[tuple] = []
    robot = FakeRobot(actions)
    tool = FakeTool(
        actions,
        timeout_probe=timeout_probe,
        off_hook=wait_for_transport_timeout,
        off_error=ProtocolAgain(),
    )
    session = FakeSession(robot=robot, tool=tool, client=client)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True

    def fail_finalize(*args, **kwargs):
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert tool.timeout_observations == [(0.1, 100, 100)]
    assert delay["elapsed"] < 0.2
    assert robot.pose_calls == 0
    assert robot.home_calls == 0
    assert client.timeout == 600.0
    assert socket.RCVTIMEO == 5000
    assert socket.SNDTIMEO == 7000


def test_finalize_failure_timeout_rebind_failure_skips_backend_cleanup(
    tmp_path,
    monkeypatch,
):
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(socket=socket)
    actions: list[tuple] = []
    robot = FakeRobot(actions)
    tool = FakeTool(actions)
    session = FakeSession(robot=robot, tool=tool, client=client)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True

    def fail_finalize(*args, **kwargs):
        socket.fail_on_set = "RCVTIMEO"
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert [item["stage"] for item in result.error["details"][
        "cleanup_errors"
    ]] == [
        "client.transport_timeout.rebind",
        "backend.connection",
    ]
    assert tool.off_calls == 0
    assert robot.pose_calls == 0
    assert robot.home_calls == 0
    assert client.timeout == 600.0
    assert socket.RCVTIMEO == 5000
    assert socket.SNDTIMEO == 7000


def test_finalize_failure_second_timeout_restore_error_is_reported(
    tmp_path,
    monkeypatch,
):
    socket = ProtocolSocket(rcvtimeo=5000, sndtimeo=7000)
    client = ProtocolFaithfulClient(socket=socket)
    replacement_client = ProtocolFaithfulClient()

    def fail_next_restore():
        socket.fail_on_set = "RCVTIMEO"

    actions: list[tuple] = []
    robot = FakeRobot(actions)
    tool = FakeTool(actions, off_hook=fail_next_restore)
    session = RebuildingFakeSession(
        robot=robot,
        tool=tool,
        client=client,
        replacement_client=replacement_client,
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True
    snapshots: list[StudentRunSnapshot] = []
    controller.subscribe(snapshots.append)

    def fail_finalize(*args, **kwargs):
        raise OSError("summary-write-failed")

    monkeypatch.setattr(
        runner_module.StudentRunEvidence,
        "finalize",
        fail_finalize,
    )
    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_EVIDENCE_FAILED"
    assert "client.transport_timeout.restore" in [
        item["stage"]
        for item in result.error["details"]["cleanup_errors"]
    ]
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    result_error = strict_json(result.error)
    assert result.summary_path is None
    assert strict_json(snapshots[-1].error) == result_error
    assert strict_json(controller._cleanup_errors) == (
        result_error["details"]["cleanup_errors"]
    )
    assert socket.RCVTIMEO == 100

    controller.reset()
    assert session.quarantined_reset_calls == 1
    assert session.reset_calls == 0
    assert socket.RCVTIMEO == 100
    assert session.application.client is replacement_client


def test_first_timeout_restore_failure_never_runs_unprotected_cleanup(
    tmp_path,
):
    socket = ProtocolSocket(
        rcvtimeo=5000,
        sndtimeo=7000,
        fail_on_first_restore="RCVTIMEO",
    )
    client = ProtocolFaithfulClient(socket=socket)
    replacement_client = ProtocolFaithfulClient()
    actions: list[tuple] = []
    robot = FakeRobot(actions)
    tool = FakeTool(actions)
    session = RebuildingFakeSession(
        robot=robot,
        tool=tool,
        client=client,
        replacement_client=replacement_client,
    )
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True
    snapshots: list[StudentRunSnapshot] = []
    controller.subscribe(snapshots.append)

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_CLIENT_TIMEOUT_RESTORE_FAILED"
    assert result.error["details"]["quarantined"] is True
    assert result.error["details"]["connection_unusable"] is True
    assert [item["stage"] for item in result.error["details"][
        "cleanup_errors"
    ]] == [
        "client.transport_timeout.restore",
        "backend.connection",
    ]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    result_error = strict_json(result.error)
    assert result_error == summary["error"]
    assert result_error["details"]["cleanup_errors"] == (
        summary["cleanup_errors"]
    )
    assert strict_json(controller._cleanup_errors) == (
        summary["cleanup_errors"]
    )
    assert strict_json(snapshots[-1].error) == result_error
    assert summary["error"]["details"]["quarantined"] is True
    assert summary["error"]["details"]["connection_unusable"] is True
    assert tool.off_calls == 0
    assert robot.pose_calls == 0
    assert robot.home_calls == 0
    assert socket.RCVTIMEO == 100

    controller.reset()
    assert session.quarantined_reset_calls == 1
    assert session.reset_calls == 0
    assert tool.off_calls == 0
    assert socket.RCVTIMEO == 100
    assert session.application.client is replacement_client


def test_reset_rejects_until_prior_run_threads_have_exited(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    callback_entered = threading.Event()
    callback_release = threading.Event()

    def blocking_terminal_callback(snapshot):
        if snapshot.state is RunState.PASSED:
            callback_entered.set()
            callback_release.wait(timeout=5)

    unsubscribe = controller.subscribe(blocking_terminal_callback)
    assert controller.validate().ok is True
    controller.start()
    assert controller.wait(timeout_s=5).status == "PASS"
    assert callback_entered.wait(timeout=2)

    try:
        with pytest.raises(RuntimeError, match="threads are still active"):
            controller.reset()
        assert controller.state is RunState.PASSED
    finally:
        callback_release.set()
        unsubscribe()

    wait_until(
        lambda: not controller._command_thread.is_alive(),
        timeout_s=2,
    )
    controller.reset()
    assert controller.state is RunState.VALIDATED


def test_reset_is_rejected_from_the_finishing_command_thread(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    controller._state = RunState.PASSED
    controller._command_thread = threading.current_thread()

    with pytest.raises(RuntimeError, match="threads are still active"):
        controller.reset()

    assert controller.state is RunState.PASSED


def test_rapid_pass_reset_generations_do_not_interfere(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    assert controller.validate().ok is True

    for index in range(5):
        controller.start()
        assert controller.wait(timeout_s=5).status == "PASS"
        if index < 4:
            controller.reset()
            assert controller.state is RunState.VALIDATED


def test_requested_terminal_outcome_wins_over_racing_worker_pass(
    tmp_path, monkeypatch
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    assert controller.validate().ok is True
    original = controller._worker_outcome

    def request_cancel_then_report(payload):
        outcome = original(payload)
        controller._request_stop(
            "CANCELLED",
            {
                "code": "STUDENT_PROGRAM_CANCELLED",
                "message": "barrier cancellation",
            },
        )
        return outcome

    monkeypatch.setattr(
        controller,
        "_worker_outcome",
        request_cancel_then_report,
    )
    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "CANCELLED"
    assert result.error["code"] == "STUDENT_PROGRAM_CANCELLED"


def test_total_runtime_is_rechecked_synchronously_without_watchdog(
    tmp_path, monkeypatch
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    import time\n"
        "    time.sleep(0.15)\n",
        execution_policy=policy(max_runtime_s=0.1),
    )
    assert controller.validate().ok is True
    monkeypatch.setattr(controller, "_watchdog_loop", lambda: None)

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_RUNTIME_TIMEOUT"


def test_command_timeout_is_rechecked_synchronously_without_watchdog(
    tmp_path, monkeypatch
):
    robot = FakeRobot(move_duration_s=0.15)
    session = FakeSession(robot=robot)
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 100, speed=8)\n",
        session=session,
        execution_policy=policy(command_timeout_s=0.1),
    )
    assert controller.validate().ok is True
    monkeypatch.setattr(controller, "_watchdog_loop", lambda: None)

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_COMMAND_TIMEOUT"


def test_reap_gives_cooperative_exit_grace_before_terminate(tmp_path):
    class GracefulProcess:
        def __init__(self):
            self.alive = True
            self.terminate_calls = 0
            self.join_timeouts = []

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)
            self.alive = False

        def terminate(self):
            self.terminate_calls += 1
            self.alive = False

    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    process = GracefulProcess()
    controller._process = process

    assert controller._reap_child(force=True) is True
    assert process.join_timeouts[0] > 0
    assert process.terminate_calls == 0


def test_bind_experiment_clears_previous_program_when_idle(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
    )
    manifest = {"scene_id": "robot-basics", "nested": {"value": 1}}

    controller.bind_experiment(
        context=SimpleNamespace(experiment_id="R1-01"),
        definition=SimpleNamespace(experiment_id="R1-01"),
        scene_manifest=manifest,
    )
    manifest["nested"]["value"] = 99

    assert controller.state is RunState.EMPTY
    assert controller.program_path is None
    assert controller._validation is None
    assert controller._result is None
    assert controller._evidence is None
    assert controller._experiment_gateway is None
    assert controller._scene_manifest["nested"]["value"] == 1


def test_bind_experiment_atomically_loads_optional_program_when_idle(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
    )
    template = tmp_path / "catalog-template.py"
    template.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    context = SimpleNamespace(experiment_id="R1-05")
    definition = SimpleNamespace(experiment_id="R1-05")
    snapshots = []
    controller.subscribe(snapshots.append)

    controller.bind_experiment(
        context=context,
        definition=definition,
        scene_manifest={"scene_id": "visual-positioning"},
        program_path=template,
    )

    assert controller._experiment_context is context
    assert controller._experiment_definition is definition
    assert controller.program_path == template.resolve()
    assert controller.state is RunState.LOADED
    assert [snapshot.state for snapshot in snapshots] == [RunState.LOADED]


def test_bind_experiment_rejects_changed_sealed_manifest_before_mutation(
    tmp_path,
):
    from vision_platform.student.experiment_gateway import FileBytesSeal

    context, definition, manifest = experiment_bundle(tmp_path)
    controller, _, original_program = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    original_context = controller._experiment_context
    original_manifest = controller._scene_manifest
    seal = FileBytesSeal.capture(definition.scene_manifest)
    replacement = dict(manifest)
    replacement["handoff_generation"] = "changed"
    definition.scene_manifest.write_text(
        json.dumps(replacement),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="changed"):
        controller.bind_experiment(
            context=context,
            definition=definition,
            scene_manifest=manifest,
            program_path=definition.student_template,
            scene_manifest_seal=seal,
        )

    assert controller._experiment_context is original_context
    assert controller._scene_manifest is original_manifest
    assert controller.program_path == original_program.resolve()
    assert controller.state is RunState.LOADED


def test_experiment_binding_snapshot_restores_one_coherent_idle_state(tmp_path):
    context, definition, manifest = experiment_bundle(tmp_path)
    controller, session, original_program = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
        experiment_context=context,
        experiment_definition=definition,
        scene_manifest=manifest,
    )
    snapshot = controller.capture_experiment_snapshot()
    replacement_program = tmp_path / "replacement.py"
    replacement_program.write_text(
        "def main(ctx):\n    ctx.robot.home()\n",
        encoding="utf-8",
    )
    controller.bind_experiment(
        context=SimpleNamespace(experiment_id="R1-05"),
        definition=SimpleNamespace(experiment_id="R1-05"),
        scene_manifest={"generation": "replacement"},
        program_path=replacement_program,
    )
    replacement_application = session._replace_application()
    restored_context = SimpleNamespace(experiment_id="R1-01")
    snapshots = []
    controller.subscribe(snapshots.append)

    controller.restore_experiment_snapshot(
        snapshot,
        restored_context=restored_context,
    )

    assert controller._experiment_context is restored_context
    assert controller._experiment_definition is definition
    assert controller._scene_manifest == manifest
    assert controller.program_path == original_program.resolve()
    assert controller.state is RunState.LOADED
    assert controller._application is replacement_application
    assert [item.state for item in snapshots] == [RunState.LOADED]


def test_experiment_binding_quarantine_clears_binding_and_blocks_later_apis(
    tmp_path,
):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def main(ctx):\n    pass\n", encoding="utf-8")

    controller.quarantine_experiment_binding(
        RuntimeError("scene-rollback-failed")
    )

    assert controller.experiment_binding_quarantined is True
    assert controller.program_path is None
    assert controller._experiment_context is None
    assert controller._experiment_definition is None
    assert controller._scene_manifest is None
    assert controller.state is RunState.EMPTY
    with pytest.raises(RuntimeError, match="隔离"):
        controller.load(candidate)
    with pytest.raises(RuntimeError, match="隔离"):
        controller.bind_experiment(
            context=SimpleNamespace(experiment_id="R1-05"),
            definition=SimpleNamespace(experiment_id="R1-05"),
            scene_manifest={},
            program_path=candidate,
        )


@pytest.mark.parametrize(
    "state",
    (RunState.RUNNING, RunState.PAUSED, RunState.RESETTING),
)
def test_bind_experiment_is_rejected_before_mutation_while_active(
    tmp_path,
    state,
):
    controller, _, program = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    old_context = controller._experiment_context
    with controller._condition:
        controller._state = state

    with pytest.raises(RuntimeError, match="运行期间"):
        controller.bind_experiment(
            context=SimpleNamespace(experiment_id="R1-01"),
            definition=SimpleNamespace(experiment_id="R1-01"),
            scene_manifest={"scene_id": "robot-basics"},
        )

    assert controller.program_path == program.resolve()
    assert controller._experiment_context is old_context


def test_bind_experiment_rejects_a_still_live_backend_action(tmp_path):
    controller, _, program = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    current = threading.current_thread()
    with controller._condition:
        controller._backend_action_threads[current] = "late-action"

    try:
        with pytest.raises(RuntimeError, match="运行期间"):
            controller.bind_experiment(
                context=SimpleNamespace(experiment_id="R1-01"),
                definition=SimpleNamespace(experiment_id="R1-01"),
                scene_manifest={"scene_id": "robot-basics"},
            )
    finally:
        with controller._condition:
            controller._backend_action_threads.pop(current, None)

    assert controller.program_path == program.resolve()


def test_bind_experiment_preserves_backend_quarantine_until_safe_reset(
    tmp_path,
):
    controller, _, program = make_controller(
        tmp_path,
        "def main(ctx):\n    pass\n",
    )
    with controller._condition:
        controller._backend_quarantined = True
        controller._backend_quarantine_error = {
            "code": "STUDENT_BACKEND_CONNECTION_QUARANTINED"
        }

    with pytest.raises(RuntimeError, match="隔离"):
        controller.bind_experiment(
            context=SimpleNamespace(experiment_id="R1-01"),
            definition=SimpleNamespace(experiment_id="R1-01"),
            scene_manifest={"scene_id": "robot-basics"},
        )

    assert controller.program_path == program.resolve()
    assert controller._backend_quarantined is True
    assert controller._backend_quarantine_error == {
        "code": "STUDENT_BACKEND_CONNECTION_QUARANTINED"
    }
