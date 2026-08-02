from __future__ import annotations

import ast
import json
from pathlib import Path

from vision_platform.experiments.catalog import ExperimentCatalog


ROOT = Path(__file__).parents[2]


def test_v1_06_is_a_formal_read_only_template_matching_experiment() -> None:
    catalog = ExperimentCatalog.load(
        ROOT / "config/experiments/catalog.json",
        project_root=ROOT,
    )
    definition = catalog.require("V1-06")

    assert definition.scene == (ROOT / "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt").resolve()
    assert definition.capabilities == (
        "camera.rgb",
        "camera.profile",
        "lighting.profile",
        "vision2d.template_matching",
        "experiment.info",
        "scene.probe",
    )
    assert definition.public_parameters["template_id"] == "v1_06_red_rectangle"
    assert definition.public_parameters["threshold"] == 0.72
    assert definition.public_parameters["search_roi_px"] == (0, 0, 512, 512)
    assert definition.hardware_status == "PENDING_HARDWARE"
    assert "grade" not in definition.public_parameters
    assert "score" not in definition.public_parameters


def test_v1_06_template_and_wrappers_are_safe_and_syntax_valid() -> None:
    config = json.loads(
        (ROOT / "config/experiments/V1-06.json").read_text(encoding="utf-8")
    )
    template_path = ROOT / config["student_template"]
    source = template_path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "template_match" in source
    assert "robot" not in source
    assert "tool" not in source

    wrapper = (ROOT / "tools/vision_lab/run_experiment.ps1").read_text(encoding="utf-8")
    assert "'V1-06'" in wrapper
    assert "23005" not in wrapper
    dedicated = (ROOT / "tools/vision_lab/run_v1_06_template_matching.ps1").read_text(encoding="utf-8")
    assert "'V1-06'" in dedicated
    assert "[int]$Port = 23005" in dedicated
