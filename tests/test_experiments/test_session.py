from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from types import MappingProxyType
from typing import Callable

import pytest

from vision_platform import application as application_module
from vision_platform.application import VisionLabApplication
from vision_platform.config import VisionLabConfig, load_config
from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.experiments import session as experiment_session_module
from vision_platform.experiments.session import ExperimentSession
from vision_platform.robot.coppeliasim_suction import CoppeliaSimSuction
from vision_platform.session import VisionLabSession


class FakeApplication:
    def __init__(
        self,
        config: VisionLabConfig,
        *,
        operations: list[str] | None = None,
        camera_available: bool = True,
        fail_load: Exception | None = None,
        fail_open: Exception | None = None,
        sim: object | None = None,
    ) -> None:
        self.config = config
        self.operations = operations if operations is not None else []
        self.camera = object() if camera_available else None
        self.robot = object()
        self.tool = object()
        self.sim = sim if sim is not None else object()
        self.loaded: list[Path] = []
        self.open_calls = 0
        self.close_calls = 0
        self.fail_load = fail_load
        self.fail_open = fail_open

    def load_and_start_scene(self, path: Path | None = None) -> None:
        self.loaded.append(path)
        self.operations.append("replacement.load")
        if self.fail_load is not None:
            raise self.fail_load

    def open(self) -> None:
        self.open_calls += 1
        self.operations.append("replacement.open")
        if self.fail_open is not None:
            raise self.fail_open

    def close(self) -> None:
        self.close_calls += 1
        self.operations.append("replacement.close")


class OldApplication:
    def __init__(self, config: VisionLabConfig, operations: list[str]) -> None:
        self.config = config
        self.operations = operations
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.operations.append("old.close")


class FakeLogisticsSim:
    handle_world = -1

    def __init__(self, operations: list[str]) -> None:
        self.operations = operations
        self.handles = {
            "/LogisticsLab/Tasks/Stack": 1,
            "/LogisticsLab/Tasks/Digits": 2,
            "/LogisticsLab/Tasks/Classes": 3,
        }
        self.positions: list[tuple[int, list[float], int]] = []

    def getObject(self, path: str) -> int:
        self.operations.append(f"activate.get-{path.rsplit('/', 1)[-1]}")
        return self.handles[path]

    def setObjectPosition(
        self,
        handle: int,
        position: list[float],
        relative_to: int,
    ) -> None:
        self.positions.append((handle, list(position), relative_to))
        self.operations.append(f"activate.set-{handle}")


class ReloadingLogisticsSim(FakeLogisticsSim):
    simulation_stopped = 0

    def __init__(self, operations: list[str]) -> None:
        super().__init__(operations)
        self.state = self.simulation_stopped
        self.world_positions: dict[int, list[float]] = {}
        self._restore_saved_stack_layout()

    def getSimulationState(self) -> int:
        return self.state

    def stopSimulation(self) -> None:
        self.state = self.simulation_stopped

    def loadScene(self, _path: str) -> None:
        self.operations.append("replacement.load")
        self._restore_saved_stack_layout()

    def startSimulation(self) -> None:
        self.state = 1

    def setObjectPosition(
        self,
        handle: int,
        position: list[float],
        relative_to: int,
    ) -> None:
        super().setObjectPosition(handle, position, relative_to)
        self.world_positions[handle] = list(position)

    def _restore_saved_stack_layout(self) -> None:
        self.world_positions = {
            1: [0.0, 0.0, 0.0],
            2: [0.0, 0.0, -3.0],
            3: [0.0, 0.0, -4.0],
        }


class ReloadingApplication(FakeApplication):
    load_and_start_scene = VisionLabApplication.load_and_start_scene
    activate_configured_scene_group = (
        VisionLabApplication.activate_configured_scene_group
    )


class MissingPickablesSim:
    def __init__(self) -> None:
        self.requested_paths: list[str] = []

    def getObject(self, path: str) -> int:
        self.requested_paths.append(path)
        if path == "/BLX_tool_suction":
            return 10
        raise KeyError(path)


class InterruptBeforePublicationSession(VisionLabSession):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.interrupt_next: BaseException | None = None

    def replace_application(
        self,
        factory: Callable[[], object],
        *,
        scene_path: object,
        validate: Callable[[object], None] | None = None,
    ):
        interrupt = self.interrupt_next
        self.interrupt_next = None
        if interrupt is None:
            return super().replace_application(
                factory,
                scene_path=scene_path,
                validate=validate,
            )

        def validate_then_interrupt(application: object) -> None:
            if validate is not None:
                validate(application)
            raise interrupt

        return super().replace_application(
            factory,
            scene_path=scene_path,
            validate=validate_then_interrupt,
        )


class ReturnGateVisionSession(VisionLabSession):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.first_published = Event()
        self.second_published = Event()
        self.release_first = Event()
        self.published_by_thread: dict[str, FakeApplication] = {}
        self._publication_count = 0
        self._publication_count_lock = Lock()

    def replace_application(self, *args, **kwargs):
        application = super().replace_application(*args, **kwargs)
        self.published_by_thread[current_thread().name] = application
        with self._publication_count_lock:
            self._publication_count += 1
            publication_count = self._publication_count
        if publication_count == 1:
            self.first_published.set()
            if not self.release_first.wait(timeout=2.0):
                raise AssertionError("first selection was not released")
        else:
            self.second_published.set()
        return application


def _catalog(
    tmp_path: Path,
    *,
    declared_hash: str | None = None,
    include_pickables: bool = True,
    scene_group_path: str | None = None,
    require_suction: bool = False,
) -> ExperimentCatalog:
    scene = tmp_path / "scene.ttt"
    scene.write_bytes(b"formal-scene")
    manifest = tmp_path / "scene_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene": {
                    "path": "scene.ttt",
                    "sha256": declared_hash
                    or hashlib.sha256(scene.read_bytes()).hexdigest(),
                },
                "required_paths": ["/BLX_base_link"],
            }
        ),
        encoding="utf-8",
    )
    guide = tmp_path / "guide.md"
    guide.write_text("# guide\n", encoding="utf-8")
    template = tmp_path / "template.py"
    template.write_text("def main(ctx):\n    ctx.robot.home()\n", encoding="utf-8")
    public_parameters: dict[str, object] = {
        "camera_path": "/RobotBasics/Camera",
    }
    if include_pickables:
        public_parameters["pickables_path"] = "/RobotBasics/Pickables"
    if scene_group_path is not None:
        public_parameters["scene_group_path"] = scene_group_path
    definition = {
        "schema_version": 1,
        "experiment_id": "R1-01",
        "pack_id": "R1",
        "title": "机械臂基础",
        "version": "2.2.0",
        "scene": "scene.ttt",
        "scene_manifest": "scene_manifest.json",
        "student_template": "template.py",
        "guide": "guide.md",
        "capabilities": [
            "robot.home",
            "camera.rgb",
            "scene.probe",
        ]
        + (["tool.suction"] if require_suction else []),
        "workspace": {
            "x_mm": [20, 140],
            "y_mm": [-90, 90],
            "z_mm": [10, 140],
            "safe_z_mm": 100,
        },
        "public_parameters": public_parameters,
        "acceptance": {
            "probe_kind": "motion_observation",
            "automated_checks": ["required_paths"],
            "human_checks": ["观察机器人"],
        },
        "hardware_status": "PENDING_HARDWARE",
    }
    experiments = tmp_path / "config" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "R1-01.json").write_text(
        json.dumps(definition, ensure_ascii=False),
        encoding="utf-8",
    )
    (experiments / "catalog.json").write_text(
        json.dumps({"schema_version": 1, "experiments": ["R1-01.json"]}),
        encoding="utf-8",
    )
    return ExperimentCatalog.load(
        experiments / "catalog.json",
        project_root=tmp_path,
    )


def _two_experiment_catalog(tmp_path: Path) -> ExperimentCatalog:
    _catalog(tmp_path)
    scene = tmp_path / "scene-b.ttt"
    scene.write_bytes(b"formal-scene-b")
    manifest = tmp_path / "scene_manifest-b.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene": {
                    "path": "scene-b.ttt",
                    "sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                },
                "required_paths": ["/BLX_base_link"],
            }
        ),
        encoding="utf-8",
    )
    guide = tmp_path / "guide-b.md"
    guide.write_text("# guide b\n", encoding="utf-8")
    template = tmp_path / "template-b.py"
    template.write_text("def main(ctx):\n    ctx.robot.home()\n", encoding="utf-8")
    experiments = tmp_path / "config" / "experiments"
    first_definition = json.loads(
        (experiments / "R1-01.json").read_text(encoding="utf-8")
    )
    second_definition = {
        **first_definition,
        "experiment_id": "R1-02",
        "title": "示教与运动",
        "scene": "scene-b.ttt",
        "scene_manifest": "scene_manifest-b.json",
        "student_template": "template-b.py",
        "guide": "guide-b.md",
        "public_parameters": {
            **first_definition["public_parameters"],
            "camera_path": "/RobotBasicsB/Camera",
        },
    }
    (experiments / "R1-02.json").write_text(
        json.dumps(second_definition, ensure_ascii=False),
        encoding="utf-8",
    )
    (experiments / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": ["R1-01.json", "R1-02.json"],
            }
        ),
        encoding="utf-8",
    )
    return ExperimentCatalog.load(
        experiments / "catalog.json",
        project_root=tmp_path,
    )


def _base_config(project_root: Path) -> VisionLabConfig:
    return replace(load_config(environ={}), project_root=project_root.resolve())


def _session(
    tmp_path: Path,
    *,
    catalog: ExperimentCatalog | None = None,
    student_is_idle=lambda: True,
    camera_available: bool = True,
):
    operations: list[str] = []
    config = _base_config(tmp_path)
    first = OldApplication(config, operations)
    vision_session = VisionLabSession(
        application=first,
        factory=lambda: first,
    )
    created: list[FakeApplication] = []

    def factory(selected: VisionLabConfig) -> FakeApplication:
        operations.append("replacement.construct")
        application = FakeApplication(
            selected,
            operations=operations,
            camera_available=camera_available,
        )
        created.append(application)
        return application

    session = ExperimentSession(
        catalog=catalog or _catalog(tmp_path),
        vision_session=vision_session,
        base_config=config,
        application_factory=factory,
        student_is_idle=student_is_idle,
    )
    return session, vision_session, first, created, operations


def test_select_replaces_application_only_after_validation_and_returns_context(
    tmp_path: Path,
) -> None:
    session, vision_session, first, created, operations = _session(tmp_path)
    published: list[FakeApplication] = []
    vision_session.subscribe(published.append)

    context = session.select("R1-01")

    assert operations == [
        "old.close",
        "replacement.construct",
        "replacement.load",
        "replacement.open",
    ]
    assert first.close_calls == 1
    assert created[0].loaded == [context.scene_path]
    assert created[0].open_calls == 1
    assert isinstance(created[0].config, VisionLabConfig)
    assert created[0].config.camera_backend == "sim"
    assert created[0].config.robot_backend == "sim"
    assert created[0].config.coppelia_scene == context.scene_path
    assert created[0].config.camera_options["sim"]["sensor_path"] == (
        "/RobotBasics/Camera"
    )
    assert created[0].config.task["pickables_path"] == (
        "/RobotBasics/Pickables"
    )
    assert created[0].config.workspace.x_mm == (20.0, 140.0)
    assert published == [created[0]]
    assert vision_session.application is created[0]
    assert session.current is context
    assert context.experiment_id == "R1-01"
    assert context.scene_sha256 == hashlib.sha256(b"formal-scene").hexdigest()
    assert context.hardware_status == "PENDING_HARDWARE"
    assert isinstance(context.public_parameters, MappingProxyType)
    with pytest.raises(TypeError):
        context.public_parameters["camera_path"] = "/unsafe"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        context.hardware_status = "PASS"  # type: ignore[misc]


def test_snapshot_restores_previous_experiment_with_a_fresh_application(
    tmp_path: Path,
) -> None:
    session, vision_session, _, created, _ = _session(
        tmp_path,
        catalog=_two_experiment_catalog(tmp_path),
    )
    previous = session.select("R1-01")
    snapshot = session.capture_snapshot()
    selected = session.select("R1-02")

    restored = session.restore_snapshot(snapshot)

    assert selected.experiment_id == "R1-02"
    assert restored is not previous
    assert restored.experiment_id == "R1-01"
    assert session.current is restored
    assert vision_session.application is created[-1]
    assert created[-1].loaded == [restored.scene_path]


def test_snapshot_before_first_selection_restores_a_fresh_base_application(
    tmp_path: Path,
) -> None:
    session, vision_session, first, created, _ = _session(tmp_path)
    snapshot = session.capture_snapshot()
    session.select("R1-01")

    restored = session.restore_snapshot(snapshot)

    assert restored is None
    assert session.current is None
    assert vision_session.application is created[-1]
    assert vision_session.application is not first
    assert created[-1].config == session.base_config
    assert created[-1].loaded == [session.base_config.coppelia_scene]


def test_select_activates_group_before_capability_hash_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []
    config = _base_config(tmp_path)
    initial = OldApplication(config, operations)
    vision_session = VisionLabSession(application=initial, factory=lambda: initial)
    sim = FakeLogisticsSim(operations)
    created: list[FakeApplication] = []

    def factory(selected: VisionLabConfig) -> FakeApplication:
        operations.append("replacement.construct")
        application = FakeApplication(
            selected,
            operations=operations,
            sim=sim,
        )
        created.append(application)
        return application

    real_check = experiment_session_module.check_capabilities
    real_sha256 = experiment_session_module._sha256

    def tracked_check(application, required):
        operations.append("capabilities")
        return real_check(application, required)

    def tracked_sha256(path):
        operations.append("hash")
        return real_sha256(path)

    monkeypatch.setattr(
        experiment_session_module,
        "check_capabilities",
        tracked_check,
    )
    monkeypatch.setattr(experiment_session_module, "_sha256", tracked_sha256)
    session = ExperimentSession(
        catalog=_catalog(
            tmp_path,
            scene_group_path="/LogisticsLab/Tasks/Digits",
        ),
        vision_session=vision_session,
        base_config=config,
        application_factory=factory,
        student_is_idle=lambda: True,
    )
    vision_session.subscribe(lambda _application: operations.append("published"))

    session.select("R1-01")

    assert sim.positions == [
        (1, [0.0, 0.0, -2.0], -1),
        (2, [0.0, 0.0, 0.0], -1),
        (3, [0.0, 0.0, -4.0], -1),
    ]
    assert operations.index("activate.set-3") < operations.index("capabilities")
    assert operations.index("capabilities") < operations.index("hash")
    assert operations.index("hash") < operations.index("published")
    assert vision_session.application is created[0]


def test_unknown_scene_group_is_rejected_before_replacement(
    tmp_path: Path,
) -> None:
    operations: list[str] = []
    config = _base_config(tmp_path)
    initial = OldApplication(config, operations)
    vision_session = VisionLabSession(application=initial, factory=lambda: initial)
    candidate_sim = FakeLogisticsSim(operations)
    created: list[FakeApplication] = []

    def factory(selected: VisionLabConfig) -> FakeApplication:
        application = FakeApplication(
            selected,
            operations=operations,
            sim=candidate_sim,
        )
        created.append(application)
        return application

    session = ExperimentSession(
        catalog=_catalog(
            tmp_path,
            scene_group_path="/LogisticsLab/Tasks/Unknown",
        ),
        vision_session=vision_session,
        base_config=config,
        application_factory=factory,
        student_is_idle=lambda: True,
    )
    previous_context = object()
    session.current = previous_context  # type: ignore[assignment]
    published: list[FakeApplication] = []
    vision_session.subscribe(published.append)

    with pytest.raises(ValueError, match="Unknown logistics task group"):
        session.select("R1-01")

    assert created == []
    assert initial.close_calls == 0
    assert published == []
    assert vision_session.application is initial
    assert session.current is previous_context
    assert candidate_sim.positions == []


def test_missing_pickables_fails_capability_before_publication(
    tmp_path: Path,
) -> None:
    operations: list[str] = []
    config = _base_config(tmp_path)
    initial = OldApplication(config, operations)
    vision_session = VisionLabSession(application=initial, factory=lambda: initial)
    sim = MissingPickablesSim()
    created: list[FakeApplication] = []

    def factory(selected: VisionLabConfig) -> FakeApplication:
        application = FakeApplication(
            selected,
            operations=operations,
            sim=sim,
        )
        application.tool = CoppeliaSimSuction(
            sim,
            pickables_path=str(selected.task["pickables_path"]),
        )
        created.append(application)
        return application

    session = ExperimentSession(
        catalog=_catalog(tmp_path, require_suction=True),
        vision_session=vision_session,
        base_config=config,
        application_factory=factory,
        student_is_idle=lambda: True,
    )
    published: list[FakeApplication] = []
    vision_session.subscribe(published.append)

    with pytest.raises(
        RuntimeError,
        match="tool.suction: 吸盘 TCP 或可抓取集合不可用",
    ):
        session.select("R1-01")

    assert sim.requested_paths == [
        "/BLX_tool_suction",
        "/RobotBasics/Pickables",
    ]
    assert created[0].close_calls == 1
    assert published == []
    assert vision_session.application is initial
    assert session.current is None


def test_select_preserves_default_pickables_when_experiment_omits_it(
    tmp_path: Path,
) -> None:
    base = _base_config(tmp_path)
    base = replace(
        base,
        task={**base.task, "pickables_path": "/VisionLab/Pickables"},
    )
    operations: list[str] = []
    old = OldApplication(base, operations)
    vision_session = VisionLabSession(application=old, factory=lambda: old)
    created: list[FakeApplication] = []
    session = ExperimentSession(
        catalog=_catalog(tmp_path, include_pickables=False),
        vision_session=vision_session,
        base_config=base,
        application_factory=lambda config: created.append(
            FakeApplication(config, operations=operations)
        )
        or created[-1],
        student_is_idle=lambda: True,
    )

    session.select("R1-01")

    assert created[0].config.task["pickables_path"] == "/VisionLab/Pickables"


def test_select_refuses_to_switch_while_student_is_active(tmp_path: Path) -> None:
    session, vision_session, first, created, operations = _session(
        tmp_path,
        student_is_idle=lambda: False,
    )
    published: list[FakeApplication] = []
    vision_session.subscribe(published.append)

    with pytest.raises(RuntimeError, match="学生程序"):
        session.select("R1-01")

    assert operations == []
    assert first.close_calls == 0
    assert created == []
    assert published == []
    assert vision_session.application is first
    assert session.current is None

def test_missing_capability_closes_new_application_without_publishing(
    tmp_path: Path,
) -> None:
    session, vision_session, first, created, operations = _session(
        tmp_path,
        camera_available=False,
    )
    published: list[FakeApplication] = []
    vision_session.subscribe(published.append)

    with pytest.raises(RuntimeError, match=r"实验能力不可用.*camera\.rgb"):
        session.select("R1-01")

    assert operations == [
        "old.close",
        "replacement.construct",
        "replacement.load",
        "replacement.open",
        "replacement.close",
    ]
    assert created[0].close_calls == 1
    assert published == []
    assert vision_session.application is first
    assert session.current is None


def test_scene_hash_mismatch_closes_new_application_without_publishing(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path, declared_hash="0" * 64)
    session, vision_session, first, created, _operations = _session(
        tmp_path,
        catalog=catalog,
    )
    published: list[FakeApplication] = []
    vision_session.subscribe(published.append)

    with pytest.raises(RuntimeError, match="场景哈希不匹配"):
        session.select("R1-01")

    assert created[0].close_calls == 1
    assert published == []
    assert vision_session.application is first
    assert session.current is None


@pytest.mark.parametrize("stage", ["factory", "load", "open", "validate"])
def test_failed_reselection_invalidates_previous_context_before_publication(
    tmp_path: Path,
    stage: str,
) -> None:
    catalog = _catalog(tmp_path)
    operations: list[str] = []
    config = _base_config(tmp_path)
    initial = OldApplication(config, operations)
    vision_session = VisionLabSession(
        application=initial,
        factory=lambda: initial,
    )
    created: list[FakeApplication] = []
    factory_calls = 0

    def factory(selected: VisionLabConfig) -> FakeApplication:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2 and stage == "factory":
            raise RuntimeError("factory failed")
        application = FakeApplication(
            selected,
            operations=operations,
            camera_available=not (factory_calls == 2 and stage == "validate"),
            fail_load=(
                RuntimeError("load failed")
                if factory_calls == 2 and stage == "load"
                else None
            ),
            fail_open=(
                RuntimeError("open failed")
                if factory_calls == 2 and stage == "open"
                else None
            ),
        )
        created.append(application)
        return application

    session = ExperimentSession(
        catalog=catalog,
        vision_session=vision_session,
        base_config=config,
        application_factory=factory,
        student_is_idle=lambda: True,
    )
    published: list[FakeApplication] = []
    vision_session.subscribe(published.append)
    previous_context = session.select("R1-01")

    expected_error = (
        r"实验能力不可用.*camera\.rgb"
        if stage == "validate"
        else f"{stage} failed"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        session.select("R1-01")

    assert previous_context is not None
    assert session.current is None
    assert vision_session.application is created[0]
    assert created[0].close_calls == 1
    if stage == "factory":
        assert len(created) == 1
    else:
        assert created[1].close_calls == 1
    assert published == [created[0]]


def test_subscriber_error_after_publication_commits_new_context(
    tmp_path: Path,
) -> None:
    session, vision_session, _first, created, _operations = _session(tmp_path)
    previous_context = session.select("R1-01")
    received: list[FakeApplication] = []

    def fail_after_publication(_application: FakeApplication) -> None:
        raise RuntimeError("subscriber failed")

    vision_session.subscribe(fail_after_publication)
    vision_session.subscribe(received.append)

    with pytest.raises(RuntimeError, match="subscriber failed"):
        session.select("R1-01")

    assert session.current is not None
    assert session.current is not previous_context
    assert session.current.experiment_id == "R1-01"
    assert session.current.hardware_status == "PENDING_HARDWARE"
    assert vision_session.application is created[1]
    assert created[0].close_calls == 1
    assert created[1].close_calls == 0
    assert received == [created[1]]


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_after_validation_before_publication_does_not_commit_context(
    tmp_path: Path,
    interrupt_type: type[BaseException],
) -> None:
    operations: list[str] = []
    config = _base_config(tmp_path)
    initial = OldApplication(config, operations)
    vision_session = InterruptBeforePublicationSession(
        application=initial,
        factory=lambda: initial,
    )
    created: list[FakeApplication] = []

    def factory(selected: VisionLabConfig) -> FakeApplication:
        application = FakeApplication(selected, operations=operations)
        created.append(application)
        return application

    session = ExperimentSession(
        catalog=_catalog(tmp_path),
        vision_session=vision_session,
        base_config=config,
        application_factory=factory,
        student_is_idle=lambda: True,
    )
    published: list[FakeApplication] = []
    vision_session.subscribe(published.append)
    previous_context = session.select("R1-01")
    vision_session.interrupt_next = interrupt_type()

    with pytest.raises(interrupt_type):
        session.select("R1-01")

    assert previous_context is not None
    assert session.current is None
    assert vision_session.application is created[0]
    assert created[0].close_calls == 1
    assert created[1].close_calls == 1
    assert published == [created[0]]


def _run_concurrent_selections(tmp_path: Path):
    operations: list[str] = []
    config = _base_config(tmp_path)
    initial = OldApplication(config, operations)
    vision_session = ReturnGateVisionSession(
        application=initial,
        factory=lambda: initial,
    )
    created: list[FakeApplication] = []

    def factory(selected: VisionLabConfig) -> FakeApplication:
        application = FakeApplication(selected, operations=operations)
        created.append(application)
        return application

    session = ExperimentSession(
        catalog=_two_experiment_catalog(tmp_path),
        vision_session=vision_session,
        base_config=config,
        application_factory=factory,
        student_is_idle=lambda: True,
    )
    results = {}
    errors: list[BaseException] = []
    second_started = Event()

    def select(experiment_id: str) -> None:
        if current_thread().name == "select-b":
            second_started.set()
        try:
            results[current_thread().name] = session.select(experiment_id)
        except BaseException as error:
            errors.append(error)

    first = Thread(target=select, args=("R1-01",), name="select-a")
    second = Thread(target=select, args=("R1-02",), name="select-b")
    first.start()
    assert vision_session.first_published.wait(timeout=2.0)
    second.start()
    assert second_started.wait(timeout=2.0)
    second_overtook_context_commit = vision_session.second_published.wait(
        timeout=0.25
    )
    vision_session.release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    return (
        session,
        vision_session,
        results,
        second_overtook_context_commit,
    )


def test_concurrent_selects_commit_each_context_before_next_publication(
    tmp_path: Path,
) -> None:
    (
        _session_result,
        vision_session,
        results,
        second_overtook_context_commit,
    ) = _run_concurrent_selections(tmp_path)

    assert second_overtook_context_commit is False
    for thread_name, context in results.items():
        application = vision_session.published_by_thread[thread_name]
        assert context.scene_path == application.config.coppelia_scene


def test_concurrent_selects_leave_final_application_and_context_aligned(
    tmp_path: Path,
) -> None:
    session, vision_session, results, _overtook = _run_concurrent_selections(
        tmp_path
    )

    assert session.current is results["select-b"]
    assert session.current.scene_path == (
        vision_session.application.config.coppelia_scene
    )
    assert session.current.experiment_id == "R1-02"


def test_shared_reset_reuses_selected_experiment_factory_and_context(
    tmp_path: Path,
) -> None:
    session, vision_session, _first, created, _operations = _session(tmp_path)
    context = session.select("R1-01")
    selected_config = created[0].config

    replacement = vision_session.reset_simulation()

    assert replacement is created[1]
    assert replacement.config == selected_config
    assert replacement.config.coppelia_scene == context.scene_path
    assert replacement.loaded == [None]
    assert vision_session.application is replacement
    assert session.current is context


def test_shared_reset_reactivates_selected_nondefault_scene_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application_module.time, "sleep", lambda _seconds: None)
    operations: list[str] = []
    config = _base_config(tmp_path)
    initial = OldApplication(config, operations)
    vision_session = VisionLabSession(application=initial, factory=lambda: initial)
    created: list[ReloadingApplication] = []

    def factory(selected: VisionLabConfig) -> ReloadingApplication:
        application = ReloadingApplication(
            selected,
            operations=operations,
            sim=ReloadingLogisticsSim(operations),
        )
        created.append(application)
        return application

    session = ExperimentSession(
        catalog=_catalog(
            tmp_path,
            scene_group_path="/LogisticsLab/Tasks/Digits",
        ),
        vision_session=vision_session,
        base_config=config,
        application_factory=factory,
        student_is_idle=lambda: True,
    )
    context = session.select("R1-01")

    assert len(created[0].sim.positions) == 3

    replacement = vision_session.reset_simulation()

    assert replacement is created[1]
    assert replacement.config.task["scene_group_path"] == (
        "/LogisticsLab/Tasks/Digits"
    )
    assert replacement.sim.world_positions == {
        1: [0.0, 0.0, -2.0],
        2: [0.0, 0.0, 0.0],
        3: [0.0, 0.0, -4.0],
    }
    assert len(replacement.sim.positions) == 3
    assert session.current is context
    assert vision_session.application is replacement


def _rewrite_manifest(catalog: ExperimentCatalog, **scene_updates) -> None:
    definition = catalog.require("R1-01")
    payload = json.loads(definition.scene_manifest.read_text(encoding="utf-8"))
    payload["scene"].update(scene_updates)
    definition.scene_manifest.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_select_rejects_scene_manifest_schema_mismatch(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    definition = catalog.require("R1-01")
    payload = json.loads(definition.scene_manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    definition.scene_manifest.write_text(json.dumps(payload), encoding="utf-8")
    session, _vision, _first, created, _operations = _session(
        tmp_path,
        catalog=catalog,
    )

    with pytest.raises(ValueError, match="schema_version"):
        session.select("R1-01")

    assert created[0].close_calls == 1
    assert session.current is None


def test_select_rejects_scene_manifest_path_mismatch(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    other_scene = tmp_path / "other-scene.ttt"
    other_scene.write_bytes(b"other-scene")
    _rewrite_manifest(catalog, path="other-scene.ttt")
    session, _vision, _first, created, _operations = _session(
        tmp_path,
        catalog=catalog,
    )

    with pytest.raises(ValueError, match="scene.path.*experiment scene"):
        session.select("R1-01")

    assert created[0].close_calls == 1
    assert session.current is None


def test_select_rejects_scene_manifest_path_escape(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    _rewrite_manifest(catalog, path="../outside-scene.ttt")
    session, _vision, _first, created, _operations = _session(
        tmp_path,
        catalog=catalog,
    )

    with pytest.raises(ValueError, match="scene.path.*inside project"):
        session.select("R1-01")

    assert created[0].close_calls == 1
    assert session.current is None
