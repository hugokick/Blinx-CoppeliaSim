from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.experiments.defect_assets import load_defect_assets
from vision_platform.experiments.defect_sorting import DefectSortError


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "simulation" / "vision_defect_sorting_lab" / "BL23_vision_defect_sorting_lab.ttt"
MANIFEST = ROOT / "simulation" / "vision_defect_sorting_lab" / "scene_manifest.json"
PORT = 23010
WRAPPER = ROOT / "tools" / "vision_lab" / "run_v1_09_surface_defects.ps1"


def _endpoint(request) -> tuple[str, int]:
    host = (
        request.config.getoption("--coppelia-host")
        or os.environ.get("COPPELIA_HOST")
        or "127.0.0.1"
    )
    configured = request.config.getoption("--coppelia-port")
    port = configured if configured is not None else int(os.environ.get("COPPELIA_PORT", PORT))
    if port != PORT:
        pytest.fail(f"V1-09 online acceptance requires dedicated port {PORT}, got {port}")
    return host, port


def test_v1_09_definition_uses_only_the_independent_scene() -> None:
    definition = ExperimentCatalog.load(
        ROOT / "config" / "experiments" / "catalog.json",
        project_root=ROOT,
    ).require("V1-09")
    assert definition.scene == SCENE.resolve()
    assert definition.scene.name == "BL23_vision_defect_sorting_lab.ttt"
    assert not str(definition.scene).endswith("BL23_vision_lab.ttt")


def test_v1_09_release_manifest_is_hash_bound_and_published() -> None:
    assert SCENE.is_file(), f"missing V1-09 scene: {SCENE}"
    assert MANIFEST.is_file(), f"missing V1-09 scene manifest: {MANIFEST}"
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scene = payload["scene"]
    assert scene["path"] == "simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt"
    assert scene["sha256"] == hashlib.sha256(SCENE.read_bytes()).hexdigest()
    assert scene["size_bytes"] == SCENE.stat().st_size
    assert payload["required_paths"]
    assert payload["profile_catalog"]["sha256"] == hashlib.sha256(
        (ROOT / payload["profile_catalog"]["path"]).read_bytes()
    ).hexdigest()
    assert payload["defect_assets_manifest"]["sha256"] == hashlib.sha256(
        (ROOT / payload["defect_assets_manifest"]["path"]).read_bytes()
    ).hexdigest()
    assert payload["scene_id"] == "vision-defect-sorting-lab"


def test_v1_09_tampered_asset_fails_before_plan_generation(tmp_path: Path) -> None:
    source = ROOT / "simulation" / "vision_defect_sorting_lab" / "defect_assets_manifest.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    root = tmp_path / "assets"
    root.mkdir()
    for record in payload["assets"]:
        destination = root / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source.parent / record["path"]).read_bytes())
    tampered = root / payload["assets"][0]["path"]
    tampered.write_bytes(tampered.read_bytes() + b"tampered")
    manifest = root / "defect_assets_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="DEFECT_SORT"):
        load_defect_assets(manifest)


def test_v1_09_online_wrapper_uses_owned_launcher_and_exact_scope() -> None:
    assert WRAPPER.is_file(), f"missing V1-09 online wrapper: {WRAPPER}"
    source = WRAPPER.read_text(encoding="utf-8")
    assert "launch_coppeliasim.ps1" in source
    assert "process_ownership.ps1" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "23010" in source
    assert "BL23_vision_defect_sorting_lab.ttt" in source
    assert "test_coppeliasim_v1_09.py" in source
    assert "Stop-Process -Name" not in source
    assert "taskkill" not in source.lower()


@pytest.mark.coppeliasim
def test_v1_09_online_defect_sorting_is_complete(tmp_path: Path, request) -> None:
    host, port = _endpoint(request)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError as error:
        pytest.fail(
            f"No owned CoppeliaSim listener at {host}:{port}; run the V1-09 wrapper first: {error}"
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision_platform.cli",
            "experiment-run",
            "--experiment",
            "V1-09",
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
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["status"] == "PASS"
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    evidence_dir = Path(payload["evidence"]).resolve()
    summary = json.loads(Path(payload["summary"]).read_text(encoding="utf-8"))
    assert summary["scene_probe_status"] == "PASS"
    final_probe = json.loads((evidence_dir / "scene-final.json").read_text(encoding="utf-8"))
    assert final_probe["status"] == "PASS"
    assert final_probe["matched"] == final_probe["expected"] == 6
    assert final_probe["same_run_evidence"] is True
    assert final_probe["robot_home"] is True
    assert final_probe["tool_off"] is True
    assert len(final_probe["entry_evidence"]) == 6
    commands = [
        json.loads(line)
        for line in (evidence_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    names = [item["name"] for item in commands]
    assert names.count("vision2d.surface_defects") == 1
    assert names.count("vision2d.defect_sort_entry") == 6
    assert not any(name.startswith("robot.") or name.startswith("tool.") for name in names)
    final_evidence = evidence_dir / "v1-09-defect-final-evidence.json"
    assert final_evidence.is_file()
    payload = json.loads(final_evidence.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["port"] == 23010
    plan_entries = payload["plan"]["entries"]
    assert [entry["decision"] for entry in plan_entries] == [
        "qualified", "missing", "hole", "foreign", "broken", "dimension"
    ]
    assert [entry["pick_xyz_mm"] for entry in plan_entries] == [
        [140.0, -16.0, 18.0], [85.0, -16.0, 18.0], [30.0, -16.0, 18.0],
        [140.0, 38.0, 18.0], [85.0, 38.0, 18.0], [30.0, 38.0, 18.0],
    ]
    assert [entry["drop_xyz_mm"] for entry in plan_entries] == [
        [132.0, -93.0, 22.0], [85.0, -93.0, 22.0], [38.0, -93.0, 22.0],
        [132.0, 75.0, 22.0], [85.0, 75.0, 22.0], [38.0, 75.0, 22.0],
    ]
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    assert payload["human_acceptance"] == "PENDING_HUMAN_ACCEPTANCE"
