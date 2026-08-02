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
def test_v1_07_student_template_completes_guarded_code_routes(tmp_path, request) -> None:
    host = request.config.getoption("--coppelia-host") or os.environ.get("COPPELIA_HOST") or "127.0.0.1"
    configured_port = request.config.getoption("--coppelia-port")
    port = configured_port if configured_port is not None else int(os.environ.get("COPPELIA_PORT", "23007"))
    completed = subprocess.run(
        [
            sys.executable, "-m", "vision_platform.cli", "experiment-run",
            "--experiment", "V1-07", "--host", host, "--port", str(port),
            "--output", str(tmp_path / "runs"),
        ],
        cwd=ROOT, text=True, encoding="utf-8", capture_output=True,
        timeout=300, check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["status"] == "PASS"
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    evidence_dir = Path(payload["evidence"]).resolve()
    summary = json.loads(Path(payload["summary"]).read_text(encoding="utf-8"))
    commands = [
        json.loads(line)
        for line in (evidence_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    names = [item["name"] for item in commands]
    plan_index = names.index("vision2d.code_routes")
    first_motion = min(names.index("robot.move_world"), names.index("tool.on"))
    assert plan_index < first_motion
    assert names.count("vision2d.code_routes") == 1
    assert names.count("camera.capture") == 0
    assert names[-1] == "camera.profile.reset"
    assert summary["scene_probe_status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    final_probe = json.loads((evidence_dir / "scene-final.json").read_text(encoding="utf-8"))
    assert final_probe["status"] == "PASS"
    assert final_probe["matched"] == final_probe["expected"] == 4
    assert final_probe["final_occupancy"] == {
        "route_blue": ["part_b", "part_d"],
        "route_red": ["part_a", "part_c"],
    }
    artifacts = sorted(evidence_dir.glob("vision-bundle-*.json"))
    assert len(artifacts) == 2
    final_artifact = next(path for path in artifacts if "zfinal" in path.name)
    bundle = load_recorded_bundle(evidence_dir, final_artifact.name)
    result = bundle["result"]
    assert bundle["experiment_id"] == "V1-07"
    assert bundle["status"] == "PASS"
    assert [layer["layer_id"] for layer in bundle["layers"]] == ["raw", "annotated"]
    assert len(result["entries"]) == 4
    assert len(result["completed_entry_ids"]) == 4
    assert result["final_occupancy"] == final_probe["final_occupancy"]
    assert result["human_acceptance"] == "PENDING_HUMAN_ACCEPTANCE"
    assert result["hardware_status"] == "PENDING_HARDWARE"
    assert result["snapshot_id"] == bundle["source_snapshot_id"]
    assert all(event["plan_id"] == result["plan_id"] for event in result["motion_events"])
