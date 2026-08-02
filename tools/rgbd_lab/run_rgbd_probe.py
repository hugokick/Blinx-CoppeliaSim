"""Bounded development-only RGB-D probe CLI for the D1-01 lab scene."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vision_platform.rgbd_sim import (
    CoppeliaRgbdCapture,
    RgbdSimContractError,
    build_probe_report,
    colorize_depth,
    copy_bgr_preview,
    load_scene_binding,
    normalize_source_capture,
    observe_source_depth_model,
    probe_report_to_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = PROJECT_ROOT / "simulation" / "rgbd_lab" / "scene_manifest.json"
DEFAULT_SENSOR_PATH = "/RgbdLab/CameraRig/RgbdSensor"
DEFAULT_HOST = "127.0.0.1"
FIXED_PORT = 23009
_OUTPUT_NAMES = ("rgb.png", "depth.png", "report.json", "error.json")


@dataclass(frozen=True)
class CliOptions:
    scene: Path
    output_dir: Path
    sensor_path: str
    host: str
    port: int
    timeout_s: float
    overwrite: bool


def _int_arg(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    return result


def _float_arg(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated D1-01 RGB-D probe")
    parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="scene manifest under the repository")
    parser.add_argument("--sensor-path", default=DEFAULT_SENSOR_PATH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=_int_arg, default=FIXED_PORT)
    parser.add_argument("--timeout-s", type=_float_arg, default=15.0)
    parser.add_argument("--output-dir", required=True, help="explicit output directory outside the repository")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing probe files")
    return parser


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path) -> None:
    current = path
    existing: list[Path] = []
    while current != current.parent:
        existing.append(current)
        if current.exists():
            break
        current = current.parent
    for item in existing:
        if item.is_symlink():
            raise RgbdSimContractError("RGBD_SIM_CLI_OUTPUT_INVALID", "output path may not use symlinks")


def validate_options(args: argparse.Namespace, *, repository_root: Path = PROJECT_ROOT) -> CliOptions:
    root = Path(repository_root).resolve()
    if type(args.port) is not int or args.port != FIXED_PORT:
        raise RgbdSimContractError("RGBD_SIM_CLI_PORT_INVALID", "D1-01 CLI port is fixed at 23009")
    if type(args.timeout_s) not in {int, float} or not math.isfinite(float(args.timeout_s)) or not 0.1 <= float(args.timeout_s) <= 120.0:
        raise RgbdSimContractError("RGBD_SIM_CLI_TIMEOUT_INVALID", "timeout must be between 0.1 and 120 seconds")
    if type(args.sensor_path) is not str or args.sensor_path != DEFAULT_SENSOR_PATH:
        raise RgbdSimContractError("RGBD_SIM_CLI_SENSOR_INVALID", "sensor path is not the D1-01 sensor")
    if type(args.host) is not str or not args.host or args.host not in {"127.0.0.1", "localhost"}:
        raise RgbdSimContractError("RGBD_SIM_CLI_HOST_INVALID", "only the local simulator host is allowed")
    scene_input = Path(args.scene)
    if not scene_input.is_absolute():
        scene_input = root / scene_input
    scene = scene_input.resolve(strict=False)
    if not _inside(scene, root) or scene.suffix.lower() != ".json" or not scene.is_file():
        raise RgbdSimContractError("RGBD_SIM_CLI_PATH_INVALID", "scene manifest must be an existing repository file")
    output_input = Path(args.output_dir)
    if not output_input.is_absolute():
        output_input = Path.cwd() / output_input
    output_dir = output_input.resolve(strict=False)
    if _inside(output_dir, root):
        raise RgbdSimContractError("RGBD_SIM_CLI_OUTPUT_INVALID", "output directory must be outside the repository")
    _reject_symlink_components(output_dir)
    if type(args.overwrite) is not bool:
        raise RgbdSimContractError("RGBD_SIM_CLI_OUTPUT_INVALID", "overwrite must be a boolean flag")
    return CliOptions(
        scene=scene,
        output_dir=output_dir,
        sensor_path=args.sensor_path,
        host=args.host,
        port=args.port,
        timeout_s=float(args.timeout_s),
        overwrite=args.overwrite,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output(options: CliOptions) -> None:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    if not options.overwrite:
        existing = [name for name in _OUTPUT_NAMES if (options.output_dir / name).exists()]
        if existing:
            raise RgbdSimContractError(
                "RGBD_SIM_CLI_OUTPUT_EXISTS", f"probe output already exists: {', '.join(existing)}"
            )


def _write_failure(output_dir: Path, code: str, message: str) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "error.json",
            {"status": "FAIL", "code": str(code), "message": str(message)[:240]},
        )
    except Exception:
        return


def run_probe(options: CliOptions, *, repository_root: Path = PROJECT_ROOT) -> None:
    import cv2

    binding = load_scene_binding(options.scene, repository_root=repository_root)
    if binding.sensor_path != options.sensor_path:
        raise RgbdSimContractError("RGBD_SIM_CLI_SENSOR_INVALID", "scene sensor does not match requested sensor")
    capture_adapter = CoppeliaRgbdCapture(
        sensor_path=options.sensor_path,
        scene_binding=binding,
        host=options.host,
        port=options.port,
    )
    try:
        source = capture_adapter.read_source()
        observed = observe_source_depth_model(source, binding.anchors)
        capture = normalize_source_capture(source, observed)
        report = build_probe_report(capture, binding)
        if report.status != "PASS":
            raise RgbdSimContractError(
                report.failure_code or "RGBD_SIM_CLI_PROBE_FAILED",
                report.message or "RGB-D probe did not pass",
            )
        if not cv2.imwrite(str(options.output_dir / "rgb.png"), copy_bgr_preview(capture.frame.image_bgr)):
            raise RgbdSimContractError("RGBD_SIM_CLI_OUTPUT_WRITE", "RGB preview could not be written")
        if not cv2.imwrite(
            str(options.output_dir / "depth.png"),
            colorize_depth(
                capture.frame.depth_m,
                minimum_m=binding.near_clip_m,
                maximum_m=binding.far_clip_m,
            ),
        ):
            raise RgbdSimContractError("RGBD_SIM_CLI_OUTPUT_WRITE", "depth preview could not be written")
        _write_json(options.output_dir / "report.json", probe_report_to_dict(report))
    finally:
        capture_adapter.close()


def main(argv: Sequence[str] | None = None, *, repository_root: Path = PROJECT_ROOT) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code)
    try:
        options = validate_options(args, repository_root=repository_root)
        _prepare_output(options)
        run_probe(options, repository_root=Path(repository_root).resolve())
        return 0
    except RgbdSimContractError as exc:
        output = Path(args.output_dir) if isinstance(getattr(args, "output_dir", None), str) else None
        if output is not None and not output.is_absolute():
            output = Path.cwd() / output
        if output is not None and not _inside(output.resolve(strict=False), Path(repository_root).resolve()):
            _write_failure(output, exc.code, str(exc))
        return 1
    except Exception as exc:
        output = Path(args.output_dir) if isinstance(getattr(args, "output_dir", None), str) else None
        if output is not None and not output.is_absolute():
            output = Path.cwd() / output
        if output is not None and not _inside(output.resolve(strict=False), Path(repository_root).resolve()):
            _write_failure(output, "RGBD_SIM_CLI_RUNTIME_ERROR", str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CliOptions", "build_parser", "main", "run_probe", "validate_options"]
