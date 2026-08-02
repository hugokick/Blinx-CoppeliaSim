from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

from vision_platform.rgbd.models import RgbdFrame
from vision_platform.rgbd_sim.capture import CoppeliaRgbdCapture
from vision_platform.rgbd_sim.intrinsics import derive_intrinsics
from vision_platform.rgbd_sim.models import RgbdSourceCapture, RgbdSensorMetadata
from vision_platform.rgbd_sim.errors import RgbdSimContractError
from tools.rgbd_lab import run_rgbd_probe as cli


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "simulation" / "rgbd_lab" / "scene_manifest.json"


def _source() -> RgbdSourceCapture:
    metadata = RgbdSensorMetadata(
        sensor_path="/RgbdLab/CameraRig/RgbdSensor",
        resolution=(256, 256),
        near_clip_m=0.05,
        far_clip_m=5.0,
        perspective_angle_rad=math.radians(60.0),
        scene_path="simulation/rgbd_lab/BL23_rgbd_lab.ttt",
        scene_sha256="4ad8f920dde25d21a0973cca336e53c1f5c73a26b72292cb471ba95181cc6f40",
        sequence_id=3,
        timestamp_s=1.0,
        expected_source_depth_model="optical_z",
    )
    depth = np.full((256, 256), 2.475, dtype=np.float32)
    depth[88:128, 72:112] = 1.8
    depth[96:128, 144:184] = 2.8
    depth[150:184, 72:112] = 2.4
    depth[142:184, 144:184] = 1.9
    return RgbdSourceCapture(metadata, np.zeros((256, 256, 3), dtype=np.uint8), depth)


class _FakeCapture:
    def __init__(self, **kwargs) -> None:
        assert kwargs["port"] == 23009
        self.kwargs = kwargs

    def read_source(self) -> RgbdSourceCapture:
        return _source()

    def close(self) -> None:
        return None


def test_cli_parser_rejects_unknown_and_non_numeric_options() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "x", "--unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "x", "--port", "True"])


def test_validate_options_rejects_path_escape_tracked_output_and_wrong_port(tmp_path: Path) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    options = cli.validate_options(args, repository_root=ROOT)
    assert options.port == 23009

    args = parser.parse_args(["--scene", "..\\outside.json", "--output-dir", str(tmp_path)])
    with pytest.raises(RgbdSimContractError) as exc_info:
        cli.validate_options(args, repository_root=ROOT)
    assert exc_info.value.code == "RGBD_SIM_CLI_PATH_INVALID"

    args = parser.parse_args(["--output-dir", str(ROOT / "artifacts")])
    with pytest.raises(RgbdSimContractError) as exc_info:
        cli.validate_options(args, repository_root=ROOT)
    assert exc_info.value.code == "RGBD_SIM_CLI_OUTPUT_INVALID"

    args = parser.parse_args(["--output-dir", str(tmp_path), "--port", "23008"])
    with pytest.raises(RgbdSimContractError) as exc_info:
        cli.validate_options(args, repository_root=ROOT)
    assert exc_info.value.code == "RGBD_SIM_CLI_PORT_INVALID"


def test_cli_success_writes_bounded_rgb_depth_and_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "CoppeliaRgbdCapture", _FakeCapture)
    output_dir = tmp_path / "probe-output"

    result = cli.main(
        ["--scene", str(MANIFEST), "--output-dir", str(output_dir)],
        repository_root=ROOT,
    )

    assert result == 0
    assert (output_dir / "rgb.png").is_file()
    assert (output_dir / "depth.png").is_file()
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["observed_source_depth_model"] == "optical_z"
    assert report["output_depth_model"] == "optical_z"
    assert "image_bgr" not in json.dumps(report)


def test_cli_refuses_overwrite_without_explicit_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "CoppeliaRgbdCapture", _FakeCapture)
    output_dir = tmp_path / "probe-output"
    output_dir.mkdir()
    (output_dir / "report.json").write_text("{}", encoding="utf-8")

    result = cli.main(
        ["--scene", str(MANIFEST), "--output-dir", str(output_dir)],
        repository_root=ROOT,
    )

    assert result != 0
    assert json.loads((output_dir / "error.json").read_text(encoding="utf-8"))["code"] == "RGBD_SIM_CLI_OUTPUT_EXISTS"


def test_cli_failure_writes_bounded_error(tmp_path: Path, monkeypatch) -> None:
    class FailingCapture(_FakeCapture):
        def read_source(self):
            raise RuntimeError("sim unavailable")

    monkeypatch.setattr(cli, "CoppeliaRgbdCapture", FailingCapture)
    output_dir = tmp_path / "probe-output"
    result = cli.main(
        ["--scene", str(MANIFEST), "--output-dir", str(output_dir), "--overwrite"],
        repository_root=ROOT,
    )

    assert result != 0
    error = json.loads((output_dir / "error.json").read_text(encoding="utf-8"))
    assert error["status"] == "FAIL"
    assert error["code"] == "RGBD_SIM_CLI_RUNTIME_ERROR"
    assert len(error["message"]) <= 240
