from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vision_platform.vision_quality.evidence import load_recorded_bundle


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_IDS = ("V1-03", "V1-04", "V1-05")
EXPECTED_LAYERS = (
    "raw",
    "roi-input",
    "foreground-mask",
    "cleaned-mask",
    "annotated",
)


def _require_positive_finite(*values: object) -> None:
    assert all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
        for value in values
    )


def _assert_topic_contract(experiment_id: str, targets: list[dict]) -> None:
    if experiment_id == "V1-03":
        for target in targets:
            assert len(target["center_px"]) == 2
            assert len(target["rotated_box_px"]) == 4
            assert all(
                math.isfinite(float(value))
                for value in target["center_px"]
            )
            assert all(
                len(point) == 2
                and all(math.isfinite(float(value)) for value in point)
                for point in target["rotated_box_px"]
            )
            if target["shape"] == "circle":
                assert target["angle_deg"] is None
                assert "ANGLE_UNDEFINED_FOR_CIRCLE" in target["quality_flags"]
            else:
                assert math.isfinite(float(target["angle_deg"]))
        return
    if experiment_id == "V1-04":
        for target in targets:
            _require_positive_finite(
                target["perimeter_px"],
                target["area_px2"],
                target["perimeter_mm"],
                target["area_mm2"],
            )
        return

    assert {target["shape"] for target in targets} == {
        "circle",
        "rectangle",
        "triangle",
    }
    assert {target["color"] for target in targets} == {
        "blue",
        "red",
        "green",
    }
    for target in targets:
        assert target["color"] != "unknown"
        assert target["shape"] != "unknown"
        assert target["vertex_count"] >= 3
        assert len(target["contour_px"]) >= 3


@pytest.mark.coppeliasim
@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_v1_03_to_v1_05_student_templates_record_topic_evidence(
    experiment_id,
    tmp_path,
    request,
):
    host = (
        request.config.getoption("--coppelia-host")
        or os.environ.get("COPPELIA_HOST")
        or "127.0.0.1"
    )
    configured_port = request.config.getoption("--coppelia-port")
    port = (
        configured_port
        if configured_port is not None
        else int(os.environ.get("COPPELIA_PORT", "23005"))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision_platform.cli",
            "experiment-run",
            "--experiment",
            experiment_id,
            "--host",
            host,
            "--port",
            str(port),
            "--output",
            str(tmp_path / "runs"),
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
        check=False,
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["status"] == "PASS"
    assert payload["hardware_status"] == "PENDING_HARDWARE"

    evidence_dir = Path(payload["evidence"]).resolve()
    summary = json.loads(Path(payload["summary"]).read_text(encoding="utf-8"))
    commands = [
        json.loads(line)
        for line in (evidence_dir / "commands.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert summary["scene_probe_status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    assert sum(item["name"] == "vision2d.analyze" for item in commands) == 1
    assert sum(item["name"] == "camera.capture" for item in commands) == 0
    assert commands[-1]["name"] == "camera.profile.reset"

    final_probe = json.loads(
        (evidence_dir / "scene-final.json").read_text(encoding="utf-8")
    )
    assert final_probe["status"] == "PASS"
    assert final_probe["profile_id"] == "standard"
    assert final_probe["hardware_status"] == "PENDING_HARDWARE"

    artifacts = sorted(evidence_dir.glob("vision-bundle-*.json"))
    assert len(artifacts) == 1
    loaded = load_recorded_bundle(evidence_dir, artifacts[0].name)
    assert loaded["experiment_id"] == experiment_id
    assert loaded["status"] == "PASS"
    assert loaded["hardware_status"] == "PENDING_HARDWARE"
    assert loaded["profile"]["profile_id"] == "standard"
    assert tuple(layer["layer_id"] for layer in loaded["layers"]) == EXPECTED_LAYERS
    assert tuple(loaded["_resolved_layer_paths"]) == EXPECTED_LAYERS
    assert all(path.is_file() for path in loaded["_resolved_layer_paths"].values())

    targets = loaded["result"]["targets"]
    assert len(targets) == 3
    _assert_topic_contract(experiment_id, targets)
