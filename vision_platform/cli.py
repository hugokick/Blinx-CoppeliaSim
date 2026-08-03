from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from vision_platform.image_io import read_bgr, write_image
from vision_platform.recognition.color_shape import ColorShapeRecognizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STDOUT_ROUTING_LOCK = threading.Lock()


class _RuntimeStdoutRouter:
    """Keep runtime diagnostics off the dedicated CLI JSON stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._restored = False
        self._json_stream: Any = None
        self._previous_stdout: Any = None
        self._diagnostic_stream: Any = None
        self._owned_diagnostic_stream: Any = None

    def start(self) -> None:
        _STDOUT_ROUTING_LOCK.acquire()
        try:
            with self._lock:
                if self._started:
                    raise RuntimeError("stdout router already started")
                self._started = True
                self._previous_stdout = sys.stdout
                self._json_stream = sys.stdout
                diagnostic_stream = sys.stderr
                if diagnostic_stream is None:
                    diagnostic_stream = open(
                        os.devnull,
                        "w",
                        encoding="utf-8",
                    )
                    self._owned_diagnostic_stream = diagnostic_stream
                self._diagnostic_stream = diagnostic_stream
                sys.stdout = diagnostic_stream
        except BaseException:
            _STDOUT_ROUTING_LOCK.release()
            raise

    def write_json(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            if not self._started or self._restored:
                raise RuntimeError("stdout router is not active")
            stream = self._json_stream
        _print_json(payload, stream=stream)

    def restore(self) -> None:
        owned_stream = None
        release_lock = False
        with self._lock:
            if not self._started or self._restored:
                return
            if sys.stdout is self._diagnostic_stream:
                sys.stdout = self._previous_stdout
            owned_stream = self._owned_diagnostic_stream
            self._owned_diagnostic_stream = None
            self._restored = True
            release_lock = True
        try:
            if owned_stream is not None and sys.stdout is not owned_stream:
                owned_stream.close()
        finally:
            if release_lock:
                _STDOUT_ROUTING_LOCK.release()

    def restore_when_quiescent(
        self,
        controller: Any,
        *,
        on_quiescent: Any = None,
    ) -> None:
        thread = threading.Thread(
            target=self._wait_and_restore,
            args=(controller, on_quiescent),
            name="StudentCliStdoutRestorer",
            daemon=True,
        )
        thread.start()

    def _wait_and_restore(
        self,
        controller: Any,
        on_quiescent: Any,
    ) -> None:
        try:
            while True:
                try:
                    if controller.wait_for_quiescence(0.25):
                        break
                except BaseException as barrier_error:
                    self._write_diagnostic(
                        "Deferred CLI quiescence check failed",
                        barrier_error,
                    )
                    return
            if on_quiescent is not None:
                try:
                    on_quiescent()
                except BaseException as close_error:
                    self._write_diagnostic(
                        "Deferred CLI resource close failed",
                        close_error,
                    )
        finally:
            self.restore()

    def _write_diagnostic(
        self,
        prefix: str,
        error: BaseException,
    ) -> None:
        try:
            stream = self._diagnostic_stream
            if stream is not None:
                print(
                    f"{prefix}: {error}",
                    file=stream,
                    flush=True,
                )
        except BaseException:
            # A diagnostic stream failure must not strand stdout or the lock.
            pass


def _print_json(
    payload: Mapping[str, Any],
    *,
    stream: Any = None,
) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    if stream is None:
        print(encoded)
        return
    print(
        encoded,
        file=stream,
        flush=True,
    )


def _exception_payload(
    error: BaseException,
    *,
    code: str,
) -> dict[str, str]:
    return {
        "code": code,
        "message": str(error)[:2000],
        "type": (
            f"{type(error).__module__}.{type(error).__qualname__}"
        )[:200],
    }


def _path_or_none(path: Any) -> str | None:
    return None if path is None else str(path)


def _connection_unusable(error: Any) -> bool:
    if not isinstance(error, Mapping):
        return False
    details = error.get("details")
    return bool(
        isinstance(details, Mapping)
        and details.get("connection_unusable") is True
    )


def _detection_dict(detection) -> dict:
    return {
        "detection_id": detection.detection_id,
        "center_px": list(detection.center_px),
        "color": detection.color,
        "shape": detection.shape,
        "angle_deg": detection.angle_deg,
        "area_px": detection.area_px,
        "confidence": detection.confidence,
    }


def _recognize(args: argparse.Namespace) -> int:
    source = Path(args.image).expanduser().resolve()
    image = read_bgr(source)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {source}")
    recognizer = ColorShapeRecognizer(
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
    )
    detections = recognizer.detect(image)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": str(source),
        "image_size": [image.shape[1], image.shape[0]],
        "detections": [_detection_dict(item) for item in detections],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.annotated:
        annotated_path = Path(args.annotated).expanduser().resolve()
        annotated = recognizer.annotate(image, detections)
        if not write_image(annotated_path, annotated):
            raise RuntimeError(f"Annotated image could not be written: {annotated_path}")
    return 0


def _accept(args: argparse.Namespace) -> int:
    from vision_platform.acceptance import run_full_acceptance
    from vision_platform.application import VisionLabApplication
    from vision_platform.config import load_config

    environ = dict(os.environ)
    environ["VISION_BACKEND"] = args.camera
    environ["ROBOT_BACKEND"] = args.robot
    if args.host:
        environ["COPPELIA_HOST"] = args.host
    if args.port is not None:
        environ["COPPELIA_PORT"] = str(args.port)
    if args.scene:
        scene_path = Path(args.scene).expanduser()
        if not scene_path.is_absolute():
            scene_path = PROJECT_ROOT / scene_path
        environ["COPPELIA_SCENE"] = str(scene_path.resolve())
    config = load_config(
        args.config,
        project_root=PROJECT_ROOT,
        environ=environ,
    )
    output = Path(args.output).expanduser().resolve()
    application = VisionLabApplication.from_config(config)
    try:
        if (
            not args.reuse_scene
            and (
                config.camera_backend == "sim"
                or config.robot_backend == "sim"
            )
        ):
            application.load_and_start_scene(config.coppelia_scene)
        acceptance_path = run_full_acceptance(
            application,
            output_dir=output,
            object_count=args.object_count,
        )
    finally:
        application.close()
    payload = json.loads(acceptance_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "acceptance": str(acceptance_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


def _simui_smoke(args: argparse.Namespace) -> int:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
    from vision_platform.ui.simui_panel import SimUIPanel
    from vision_platform.ui.view_model import VisionLabViewModel

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    panel = None
    started_here = False
    report = {
        "schema_version": 1,
        "status": "FAIL",
        "created": False,
        "update_count": 0,
        "destroyed": False,
        "duration_s": float(args.duration),
    }
    try:
        client = RemoteAPIClient(host=args.host, port=args.port)
        sim = client.require("sim")
        simui = client.require("simUI")
        if int(sim.getSimulationState()) == int(sim.simulation_stopped):
            sim.startSimulation()
            started_here = True
            deadline = time.monotonic() + 5.0
            while (
                int(sim.getSimulationState())
                == int(sim.simulation_stopped)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            time.sleep(0.2)
        view_model = VisionLabViewModel()
        view_model.set_connection(
            connected=True,
            camera_backend="sim",
            robot_backend="sim",
            message="CoppeliaSim 内嵌教学面板已连接",
        )
        camera = CoppeliaSimCamera(sim=sim, client=client)
        camera.open()
        view_model.set_frame(camera.read(timeout_s=5.0))
        panel = SimUIPanel(simui=simui, view_model=view_model)
        panel.open()
        report["created"] = panel.created
        deadline = time.monotonic() + float(args.duration)
        while time.monotonic() < deadline:
            panel.refresh(force=True)
            sim.handleExtCalls()
            time.sleep(0.05)
        panel.close()
        report["update_count"] = panel.update_count
        report["destroyed"] = panel.destroyed
        report["status"] = (
            "PASS"
            if report["created"]
            and report["update_count"] >= 1
            and report["destroyed"]
            else "FAIL"
        )
    except Exception as error:
        report["error"] = str(error)
        if panel is not None:
            try:
                panel.close()
            except Exception as close_error:
                report["close_error"] = str(close_error)
            report["update_count"] = panel.update_count
            report["destroyed"] = panel.destroyed
    finally:
        if started_here:
            try:
                sim.stopSimulation()
            except Exception:
                pass
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


def _student_validate(args: argparse.Namespace) -> int:
    from vision_platform.student.validator import validate_program

    try:
        result = validate_program(args.program)
    except Exception as error:
        payload = {
            "status": "FAIL",
            "program": str(Path(args.program).expanduser().resolve()),
            "issues": [],
            "error": _exception_payload(
                error,
                code="STUDENT_VALIDATION_FAILED",
            ),
        }
        _print_json(payload)
        return 1
    payload = {
        "status": "PASS" if result.ok else "FAIL",
        "program": str(result.path),
        "issues": [asdict(issue) for issue in result.issues],
    }
    _print_json(payload)
    return 0 if result.ok else 2


def _execute_student_run(
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any], Any | None]:
    from vision_platform.application import VisionLabApplication
    from vision_platform.config import load_config
    from vision_platform.session import VisionLabSession
    from vision_platform.student.runner import StudentProgramController
    from vision_platform.student.safety import StudentExecutionPolicy

    application = None
    session = None
    controller = None
    policy = None
    close_quarantined = False
    deferred_controller = None
    exit_code = 1
    payload: dict[str, Any] = {
        "status": "FAIL",
        "summary": None,
        "evidence": None,
        "error": None,
    }
    try:
        environ = dict(os.environ)
        environ["ROBOT_BACKEND"] = "sim"
        environ["VISION_BACKEND"] = "sim"
        environ["COPPELIA_HOST"] = args.host
        environ["COPPELIA_PORT"] = str(args.port)
        if args.scene:
            scene = Path(args.scene).expanduser()
            if not scene.is_absolute():
                scene = PROJECT_ROOT / scene
            environ["COPPELIA_SCENE"] = str(scene.resolve())
        config = load_config(
            args.config,
            project_root=PROJECT_ROOT,
            environ=environ,
        )

        def factory():
            return VisionLabApplication.from_config(config)

        application = factory()
        session = VisionLabSession(
            application=application,
            factory=factory,
        )
        student = config.student
        speed_range = student["speed_range"]
        policy = StudentExecutionPolicy(
            min_speed=float(speed_range[0]),
            max_speed=float(speed_range[1]),
            max_runtime_s=float(student["max_runtime_s"]),
            max_commands=int(student["max_commands"]),
            command_timeout_s=float(student["command_timeout_s"]),
            max_sleep_s=float(student["max_sleep_s"]),
            tool_on_max_z_mm=float(student["tool_on_max_z_mm"]),
        )
        controller = StudentProgramController(
            session=session,
            execution_policy=policy,
            output_root=args.output,
        )
        application.load_and_start_scene(config.coppelia_scene)
        application.open()
        controller.load(args.program)
        validation = controller.validate()
        if not validation.ok:
            payload = {
                "status": "FAIL",
                "issues": [
                    asdict(issue) for issue in validation.issues
                ],
                "summary": None,
                "evidence": None,
                "error": None,
            }
            exit_code = 2
        else:
            controller.start()
            result = controller.wait(
                timeout_s=(
                    policy.max_runtime_s
                    + policy.command_timeout_s
                    + 5
                )
            )
            close_quarantined = _connection_unusable(result.error)
            payload = {
                "status": result.status,
                "summary": _path_or_none(result.summary_path),
                "evidence": _path_or_none(result.evidence_dir),
                "error": result.error,
            }
            exit_code = 0 if result.status == "PASS" else 1
    except Exception as error:
        close_quarantined = session is not None or application is not None
        payload = {
            "status": "FAIL",
            "summary": None,
            "evidence": None,
            "error": _exception_payload(
                error,
                code="STUDENT_RUN_FAILED",
            ),
        }
        exit_code = 1
    finally:
        if controller is not None and controller.process_is_alive:
            close_quarantined = True
            try:
                controller.cancel()
            except Exception as cancel_error:
                payload["error"] = {
                    **_exception_payload(
                        cancel_error,
                        code="STUDENT_CANCEL_FAILED",
                    ),
                    "details": {
                        "original_error": payload.get("error"),
                    },
                }
                payload["status"] = "FAIL"
                exit_code = 1
        quiescent = True
        quiescence_error = None
        if controller is not None:
            timeout_s = (
                float(policy.command_timeout_s) + 0.25
                if policy is not None
                else 0.25
            )
            try:
                quiescent = controller.wait_for_quiescence(timeout_s)
            except Exception as barrier_error:
                quiescent = False
                quiescence_error = _exception_payload(
                    barrier_error,
                    code="STUDENT_QUIESCENCE_WAIT_FAILED",
                )
        if not quiescent:
            deferred_controller = controller
            details = {"original_error": payload.get("error")}
            if quiescence_error is not None:
                details["quiescence_error"] = quiescence_error
            payload = {
                "status": "FAIL",
                "summary": payload.get("summary"),
                "evidence": payload.get("evidence"),
                "error": {
                    "code": "STUDENT_SESSION_CLOSE_DEFERRED",
                    "message": (
                        "Student run resources are still active; backend "
                        "transport close was deferred to process exit"
                    ),
                    "details": details,
                },
            }
            exit_code = 1
        else:
            try:
                if session is not None:
                    if close_quarantined:
                        session.close_quarantined()
                    else:
                        session.close()
                elif application is not None:
                    if close_quarantined:
                        application.close_quarantined()
                    else:
                        application.close()
            except Exception as close_error:
                payload = {
                    "status": "FAIL",
                    "summary": payload.get("summary"),
                    "evidence": payload.get("evidence"),
                    "error": {
                        **_exception_payload(
                            close_error,
                            code="STUDENT_SESSION_CLOSE_FAILED",
                        ),
                        "details": {
                            "original_error": payload.get("error"),
                        },
                    },
                }
                exit_code = 1
    return exit_code, payload, deferred_controller


def _student_run(args: argparse.Namespace) -> int:
    router = _RuntimeStdoutRouter()
    router.start()
    deferred_controller = None
    try:
        (
            exit_code,
            payload,
            deferred_controller,
        ) = _execute_student_run(args)
        router.write_json(payload)
    finally:
        if deferred_controller is None:
            router.restore()
        else:
            router.restore_when_quiescent(deferred_controller)
    return exit_code


def _experiment_catalog():
    from vision_platform.experiments.catalog import ExperimentCatalog

    return ExperimentCatalog.load(
        PROJECT_ROOT / "config" / "experiments" / "catalog.json",
        project_root=PROJECT_ROOT,
    )


def _experiment_failure_payload(
    *,
    experiment_id: str | None,
    error: BaseException,
    code: str,
) -> dict[str, Any]:
    return {
        "status": "FAIL",
        "experiment_id": experiment_id,
        "scene_probe_status": None,
        "summary": None,
        "evidence": None,
        "error": _exception_payload(error, code=code),
        "hardware_status": "PENDING_HARDWARE",
    }


def _append_experiment_cleanup_error(
    payload: dict[str, Any],
    error: BaseException,
    *,
    code: str,
) -> None:
    cleanup_error = _exception_payload(error, code=code)
    primary = payload.get("error")
    if not isinstance(primary, Mapping):
        payload["status"] = "FAIL"
        payload["error"] = cleanup_error
        return

    merged = dict(primary)
    raw_details = merged.get("details")
    details = dict(raw_details) if isinstance(raw_details, Mapping) else {}
    raw_cleanup_errors = details.get("cleanup_errors")
    cleanup_errors = (
        list(raw_cleanup_errors)
        if isinstance(raw_cleanup_errors, (list, tuple))
        else []
    )
    cleanup_errors.append(cleanup_error)
    details["cleanup_errors"] = cleanup_errors
    merged["details"] = details
    payload["error"] = merged


def _close_experiment_resources(
    session: Any | None,
    application: Any | None,
    *,
    quarantined: bool,
) -> None:
    target = session if session is not None else application
    if target is None:
        return
    if quarantined:
        target.close_quarantined()
    else:
        target.close()


def _experiment_list(args: argparse.Namespace) -> int:
    del args
    payload = {
        "status": "PASS",
        "experiments": [
            {
                "experiment_id": item.experiment_id,
                "title": item.title,
                "version": item.version,
                "capabilities": list(item.capabilities),
                "hardware_status": item.hardware_status,
            }
            for item in _experiment_catalog().definitions
        ],
    }
    _print_json(payload)
    return 0


def _experiment_show(args: argparse.Namespace) -> int:
    try:
        item = _experiment_catalog().require(args.experiment)
        payload = {
            "status": "PASS",
            "experiment_id": item.experiment_id,
            "title": item.title,
            "version": item.version,
            "scene": item.scene.relative_to(PROJECT_ROOT).as_posix(),
            "student_template": item.student_template.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "guide": item.guide.relative_to(PROJECT_ROOT).as_posix(),
            "capabilities": list(item.capabilities),
            "automated_checks": list(item.acceptance.automated_checks),
            "human_checks": list(item.acceptance.human_checks),
            "hardware_status": item.hardware_status,
        }
    except KeyError as error:
        payload = _experiment_failure_payload(
            experiment_id=args.experiment,
            error=error,
            code="EXPERIMENT_NOT_FOUND",
        )
        _print_json(payload)
        return 2
    _print_json(payload)
    return 0


def _resolve_experiment_program(
    raw_program: str | None,
    default_program: Path,
) -> Path:
    if raw_program:
        selected = Path(raw_program).expanduser()
        if not selected.is_absolute():
            selected = PROJECT_ROOT / selected
    else:
        selected = default_program
    selected = selected.resolve()
    if not selected.is_file():
        raise ValueError(f"Student program was not found: {selected}")
    if selected.suffix.lower() != ".py":
        raise ValueError(f"Student program must be a .py file: {selected}")
    return selected


def _resolve_experiment_port(experiment_id: str, raw_port: Any) -> int:
    """Resolve the endpoint while keeping formal labs on owned ports."""

    owned_ports = {"V1-08": 23008, "V1-09": 23010}
    default_port = owned_ports.get(experiment_id, 23000)
    if raw_port is None or raw_port == "":
        return default_port
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if experiment_id in owned_ports and port != owned_ports[experiment_id]:
        raise ValueError(
            f"{experiment_id} requires dedicated CoppeliaSim port "
            f"{owned_ports[experiment_id]}"
        )
    return port


def _execute_experiment_run(
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any], Any | None]:
    from vision_platform.application import VisionLabApplication
    from vision_platform.config import load_config
    from vision_platform.experiments.session import ExperimentSession
    from vision_platform.session import VisionLabSession
    from vision_platform.student.runner import StudentProgramController
    from vision_platform.student.safety import StudentExecutionPolicy

    experiment_id = args.experiment
    try:
        catalog = _experiment_catalog()
        definition = catalog.require(experiment_id)
    except KeyError as error:
        return (
            2,
            _experiment_failure_payload(
                experiment_id=experiment_id,
                error=error,
                code="EXPERIMENT_NOT_FOUND",
            ),
            None,
        )
    except Exception as error:
        return (
            1,
            _experiment_failure_payload(
                experiment_id=experiment_id,
                error=error,
                code="EXPERIMENT_CATALOG_FAILED",
            ),
            None,
        )

    try:
        port = _resolve_experiment_port(experiment_id, args.port)
        program = _resolve_experiment_program(
            args.program,
            definition.student_template,
        )
    except (OSError, TypeError, ValueError) as error:
        return (
            2,
            _experiment_failure_payload(
                experiment_id=experiment_id,
                error=error,
                code="EXPERIMENT_ARGUMENT_INVALID",
            ),
            None,
        )

    application = None
    vision_session = None
    controller = None
    policy = None
    close_quarantined = False
    deferred_resources = None
    handled_primary_error: Exception | None = None
    exit_code = 1
    payload: dict[str, Any] = {
        "status": "FAIL",
        "experiment_id": experiment_id,
        "scene_probe_status": None,
        "summary": None,
        "evidence": None,
        "error": None,
        "hardware_status": "PENDING_HARDWARE",
    }
    try:
        environ = dict(os.environ)
        environ["ROBOT_BACKEND"] = "sim"
        environ["VISION_BACKEND"] = "sim"
        environ["COPPELIA_HOST"] = args.host
        environ["COPPELIA_PORT"] = str(port)
        base_config = load_config(
            args.config,
            project_root=PROJECT_ROOT,
            environ=environ,
        )
        application = VisionLabApplication.from_config(base_config)
        vision_session = VisionLabSession(
            application=application,
            factory=lambda: VisionLabApplication.from_config(base_config),
        )
        experiment_session = ExperimentSession(
            catalog=catalog,
            vision_session=vision_session,
            base_config=base_config,
            application_factory=VisionLabApplication.from_config,
            student_is_idle=lambda: (
                controller is None or not controller.process_is_alive
            ),
        )
        context = experiment_session.select(experiment_id)
        selected_config = vision_session.application.config
        student = selected_config.student
        speed_range = student["speed_range"]
        policy = StudentExecutionPolicy(
            min_speed=float(speed_range[0]),
            max_speed=float(speed_range[1]),
            max_runtime_s=float(student["max_runtime_s"]),
            max_commands=int(student["max_commands"]),
            command_timeout_s=float(student["command_timeout_s"]),
            max_sleep_s=float(student["max_sleep_s"]),
            tool_on_max_z_mm=float(student["tool_on_max_z_mm"]),
        )
        scene_manifest = json.loads(
            definition.scene_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(scene_manifest, dict):
            raise ValueError("scene manifest must be a JSON object")
        controller = StudentProgramController(
            session=vision_session,
            execution_policy=policy,
            output_root=args.output,
            experiment_context=context,
            experiment_definition=definition,
            scene_manifest=scene_manifest,
        )
        controller.load(program)
        validation = controller.validate()
        if not validation.ok:
            payload["issues"] = [
                asdict(issue) for issue in validation.issues
            ]
            exit_code = 2
        else:
            controller.start()
            result = controller.wait(
                timeout_s=(
                    policy.max_runtime_s
                    + policy.command_timeout_s
                    + 5
                )
            )
            close_quarantined = _connection_unusable(result.error)
            summary = {}
            if result.summary_path is not None:
                summary = json.loads(
                    result.summary_path.read_text(encoding="utf-8")
                )
                if not isinstance(summary, dict):
                    raise ValueError("student run summary must be a JSON object")
            payload = {
                "status": result.status,
                "experiment_id": definition.experiment_id,
                "scene_probe_status": summary.get("scene_probe_status"),
                "summary": _path_or_none(result.summary_path),
                "evidence": _path_or_none(result.evidence_dir),
                "error": result.error,
                "hardware_status": "PENDING_HARDWARE",
            }
            exit_code = 0 if result.status == "PASS" else 1
    except Exception as error:
        handled_primary_error = error
        payload = _experiment_failure_payload(
            experiment_id=experiment_id,
            error=error,
            code="EXPERIMENT_RUN_FAILED",
        )
        exit_code = 1
    finally:
        active_error = sys.exception()
        pending_cleanup_base_error: BaseException | None = None

        def record_cleanup_error(
            error: BaseException,
            *,
            code: str,
        ) -> None:
            nonlocal exit_code, pending_cleanup_base_error
            primary_result_failed = (
                exit_code != 0
                or payload.get("status") != "PASS"
                or payload.get("error") is not None
            )
            primary_base_error = (
                active_error
                or handled_primary_error
                or pending_cleanup_base_error
            )
            if primary_base_error is not None or primary_result_failed:
                if (
                    active_error is None
                    and pending_cleanup_base_error is None
                    and (
                        handled_primary_error is not None
                        or primary_result_failed
                    )
                ):
                    _append_experiment_cleanup_error(
                        payload,
                        error,
                        code=code,
                    )
                    if exit_code == 0:
                        exit_code = 1
                diagnostic = _exception_payload(error, code=code)
                note = (
                    f"{diagnostic['code']}: {diagnostic['type']}: "
                    f"{diagnostic['message']}"
                )
                if primary_base_error is not None:
                    try:
                        primary_base_error.add_note(note)
                    except BaseException:
                        pass
                try:
                    if sys.stderr is not None:
                        print(
                            f"Experiment cleanup diagnostic: {note}",
                            file=sys.stderr,
                            flush=True,
                        )
                except BaseException:
                    pass
                return
            if isinstance(error, Exception):
                _append_experiment_cleanup_error(
                    payload,
                    error,
                    code=code,
                )
                exit_code = 1
                return
            pending_cleanup_base_error = error

        process_alive = False
        if controller is not None:
            try:
                process_alive = controller.process_is_alive
            except BaseException as liveness_error:
                close_quarantined = True
                record_cleanup_error(
                    liveness_error,
                    code="EXPERIMENT_PROCESS_LIVENESS_FAILED",
                )
        if process_alive:
            try:
                controller.cancel()
            except BaseException as cancel_error:
                close_quarantined = True
                record_cleanup_error(
                    cancel_error,
                    code="EXPERIMENT_CANCEL_FAILED",
                )

        quiescent = True
        if controller is not None:
            timeout_s = (
                float(policy.command_timeout_s) + 0.25
                if policy is not None
                else 0.25
            )
            try:
                quiescent = controller.wait_for_quiescence(timeout_s)
            except BaseException as barrier_error:
                quiescent = False
                close_quarantined = True
                record_cleanup_error(
                    barrier_error,
                    code="EXPERIMENT_QUIESCENCE_WAIT_FAILED",
                )

        if not quiescent:
            record_cleanup_error(
                RuntimeError(
                    "experiment resources remain active; "
                    "session close deferred"
                ),
                code="EXPERIMENT_SESSION_CLOSE_DEFERRED",
            )
            if active_error is None and pending_cleanup_base_error is None:
                deferred_resources = (
                    controller,
                    lambda: _close_experiment_resources(
                        vision_session,
                        application,
                        quarantined=close_quarantined,
                    ),
                )
        else:
            try:
                _close_experiment_resources(
                    vision_session,
                    application,
                    quarantined=close_quarantined,
                )
            except BaseException as close_error:
                record_cleanup_error(
                    close_error,
                    code="EXPERIMENT_SESSION_CLOSE_FAILED",
                )

        primary_failure_recorded = (
            active_error is not None
            or handled_primary_error is not None
            or exit_code != 0
            or payload.get("status") != "PASS"
            or payload.get("error") is not None
        )
        if (
            not primary_failure_recorded
            and pending_cleanup_base_error is not None
        ):
            raise pending_cleanup_base_error

    return exit_code, payload, deferred_resources


def _experiment_run(args: argparse.Namespace) -> int:
    router = _RuntimeStdoutRouter()
    router.start()
    deferred_resources = None
    try:
        exit_code, payload, deferred_resources = _execute_experiment_run(args)
        router.write_json(payload)
    finally:
        if deferred_resources is None:
            router.restore()
        else:
            controller, close_resources = deferred_resources
            router.restore_when_quiescent(
                controller,
                on_quiescent=close_resources,
            )
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    recognize = subparsers.add_parser(
        "recognize",
        help="Recognize colors and shapes in one image",
    )
    recognize.add_argument("--image", required=True)
    recognize.add_argument("--output", required=True)
    recognize.add_argument("--annotated")
    recognize.add_argument("--min-area-ratio", type=float, default=0.002)
    recognize.add_argument("--max-area-ratio", type=float, default=0.40)
    recognize.set_defaults(handler=_recognize)

    accept = subparsers.add_parser(
        "accept",
        help="Run automated calibration and sorting acceptance",
    )
    accept.add_argument("--config")
    accept.add_argument(
        "--robot",
        choices=("sim", "real"),
        default="sim",
    )
    accept.add_argument(
        "--camera",
        choices=("sim", "replay", "hik"),
        default="sim",
    )
    accept.add_argument("--scene")
    accept.add_argument("--host", default="127.0.0.1")
    accept.add_argument("--port", type=int, default=23000)
    accept.add_argument("--object-count", type=int, default=6)
    accept.add_argument("--output", required=True)
    accept.add_argument(
        "--reuse-scene",
        action="store_true",
        help="Use the scene already loaded in CoppeliaSim",
    )
    accept.set_defaults(handler=_accept)

    simui_smoke = subparsers.add_parser(
        "simui-smoke",
        help="Create, update and destroy the CoppeliaSim teaching panel",
    )
    simui_smoke.add_argument("--host", default="127.0.0.1")
    simui_smoke.add_argument("--port", type=int, default=23000)
    simui_smoke.add_argument("--duration", type=float, default=5.0)
    simui_smoke.add_argument("--output", required=True)
    simui_smoke.set_defaults(handler=_simui_smoke)

    student_validate = subparsers.add_parser(
        "student-validate",
        help="Validate one student Python program",
    )
    student_validate.add_argument("--program", required=True)
    student_validate.set_defaults(handler=_student_validate)

    student_run = subparsers.add_parser(
        "student-run",
        help="Run one student program through the guarded simulator",
    )
    student_run.add_argument("--program", required=True)
    student_run.add_argument("--config")
    student_run.add_argument("--robot", choices=("sim",), default="sim")
    student_run.add_argument("--scene")
    student_run.add_argument("--host", default="127.0.0.1")
    student_run.add_argument("--port", type=int, default=23000)
    student_run.add_argument(
        "--output",
        default="artifacts/vision_lab/student-runs",
    )
    student_run.set_defaults(handler=_student_run)

    experiment_list = subparsers.add_parser(
        "experiment-list",
        help="List formal CoppeliaSim curriculum experiments",
    )
    experiment_list.set_defaults(handler=_experiment_list)

    experiment_show = subparsers.add_parser(
        "experiment-show",
        help="Show one formal curriculum experiment",
    )
    experiment_show.add_argument("--experiment", required=True)
    experiment_show.set_defaults(handler=_experiment_show)

    experiment_run = subparsers.add_parser(
        "experiment-run",
        help="Run one formal experiment with the guarded student runner",
    )
    experiment_run.add_argument("--experiment", required=True)
    experiment_run.add_argument("--program")
    experiment_run.add_argument("--config")
    experiment_run.add_argument("--host", default="127.0.0.1")
    experiment_run.add_argument(
        "--port",
        default=None,
        help="CoppeliaSim port (V1-08=23008, V1-09=23010; other labs default to 23000)",
    )
    experiment_run.add_argument(
        "--output",
        default="artifacts/vision_lab/experiment-runs",
    )
    experiment_run.set_defaults(handler=_experiment_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
