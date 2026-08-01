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
def test_v1_01_student_template_records_three_profiles_and_resets(
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
    output_root = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision_platform.cli",
            "experiment-run",
            "--experiment",
            "V1-01",
            "--host",
            host,
            "--port",
            str(port),
            "--output",
            str(output_root),
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
    summary_path = Path(payload["summary"]).resolve()
    evidence_dir = Path(payload["evidence"]).resolve()
    assert output_root.resolve() in evidence_dir.parents
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    commands = [
        json.loads(line)
        for line in (evidence_dir / "commands.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    bundles = sorted(evidence_dir.glob("vision-bundle-*.json"))
    assert [
        item["args"].get("profile_id")
        for item in commands
        if item["name"] == "camera.profile.apply"
    ] == ["standard", "wide_dim", "detail_bright"]
    assert sum(item["name"] == "camera.capture" for item in commands) == 3
    assert commands[-1]["name"] == "camera.profile.reset"
    assert len(bundles) == 3
    assert summary["scene_probe_status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"

    final_probe = json.loads(
        (evidence_dir / "scene-final.json").read_text(encoding="utf-8")
    )
    assert final_probe["status"] == "PASS"
    assert final_probe["profile_id"] == "standard"
    assert final_probe["hardware_status"] == "PENDING_HARDWARE"

    expected_profiles = ["standard", "wide_dim", "detail_bright"]
    for artifact, expected_profile in zip(bundles, expected_profiles):
        loaded = load_recorded_bundle(evidence_dir, artifact.name)
        assert loaded["hardware_status"] == "PENDING_HARDWARE"
        assert loaded["profile"]["profile_id"] == expected_profile
        assert len(loaded["layers"]) == 1
        layer = loaded["layers"][0]
        assert layer["layer_id"] == "raw"
        assert [layer["width"], layer["height"]] == loaded["profile"][
            "resolution"
        ]
        assert loaded["_resolved_layer_paths"]["raw"].is_file()
