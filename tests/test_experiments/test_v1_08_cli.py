from __future__ import annotations

from pathlib import Path

from vision_platform.cli import build_parser


ROOT = Path(__file__).resolve().parents[2]


def test_v1_08_cli_defaults_to_and_rejects_non_dedicated_port():
    args = build_parser().parse_args(
        ["experiment-run", "--experiment", "V1-08"]
    )
    assert args.experiment == "V1-08"
    assert args.port is None


def test_v1_08_is_selected_by_catalog_id_without_arbitrary_scene_or_command():
    args = build_parser().parse_args(
        [
            "experiment-run",
            "--experiment",
            "V1-08",
            "--program",
            "student_programs/templates/v1_08_ocr_sorting.py",
        ]
    )
    assert args.program.endswith("v1_08_ocr_sorting.py")
    assert not hasattr(args, "scene")
    assert not hasattr(args, "command_text")
    assert not hasattr(args, "eval")


def test_v1_08_powershell_launcher_has_fixed_port_and_owned_status_propagation():
    source = (ROOT / "tools" / "vision_lab" / "run_experiment.ps1").read_text(
        encoding="utf-8"
    )
    assert "'V1-08'" in source
    assert "23008" in source
    assert "LASTEXITCODE" in source
    assert "Invoke-Expression" not in source
    assert "Start-Process" not in source
