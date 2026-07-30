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

    def restore_when_quiescent(self, controller: Any) -> None:
        thread = threading.Thread(
            target=self._wait_and_restore,
            args=(controller,),
            name="StudentCliStdoutRestorer",
            daemon=True,
        )
        thread.start()

    def _wait_and_restore(self, controller: Any) -> None:
        while True:
            try:
                if controller.wait_for_quiescence(0.25):
                    self.restore()
                    return
            except BaseException:
                # Unknown liveness must remain fail-closed until process exit.
                return


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
