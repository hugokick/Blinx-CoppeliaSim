from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

from vision_platform.image_io import read_bgr, write_image
from vision_platform.recognition.color_shape import ColorShapeRecognizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
