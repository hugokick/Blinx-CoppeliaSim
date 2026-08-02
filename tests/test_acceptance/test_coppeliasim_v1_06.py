from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vision_platform.vision_quality.evidence import load_recorded_bundle


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.coppeliasim
def test_v1_06_student_template_records_template_evidence(tmp_path, request):
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
            "V1-06",
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
    assert sum(item["name"] == "vision2d.template_match" for item in commands) == 1
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
    assert loaded["experiment_id"] == "V1-06"
    assert loaded["status"] in {"PASS", "PARTIAL"}
    assert loaded["hardware_status"] == "PENDING_HARDWARE"
    assert loaded["profile"]["template"]["template_id"] == "v1_06_red_rectangle"
    assert [layer["layer_id"] for layer in loaded["layers"]] == [
        "raw",
        "template",
        "annotated",
    ]
    assert len(loaded["_resolved_layer_paths"]) == 3
    assert all(path.is_file() for path in loaded["_resolved_layer_paths"].values())
    result = loaded["result"]
    assert result["schema_version"] == 1
    assert result["template_id"] == "v1_06_red_rectangle"
    assert result["method"] == "TM_CCOEFF_NORMED"
    assert result["matched"] is True
    assert result["bbox_px"] is not None
    assert result["center_px"] is not None
