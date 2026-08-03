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
from vision_platform.experiments.ocr_assets import OcrAssetError, load_ocr_assets
from vision_platform.experiments.ocr_sorting import OcrSortError


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "simulation" / "vision_ocr_sorting_lab" / "BL23_vision_ocr_sorting_lab.ttt"
PORT = 23008
WRAPPER = ROOT / "tools" / "vision_lab" / "run_v1_08_ocr_sorting.ps1"


def _endpoint(request) -> tuple[str, int]:
    host = (
        request.config.getoption("--coppelia-host")
        or os.environ.get("COPPELIA_HOST")
        or "127.0.0.1"
    )
    configured = request.config.getoption("--coppelia-port")
    port = configured if configured is not None else int(os.environ.get("COPPELIA_PORT", PORT))
    if port != PORT:
        pytest.fail(f"V1-08 online acceptance requires dedicated port {PORT}, got {port}")
    return host, port


def test_v1_08_definition_uses_only_the_independent_scene() -> None:
    definition = ExperimentCatalog.load(
        ROOT / "config" / "experiments" / "catalog.json",
        project_root=ROOT,
    ).require("V1-08")
    assert definition.scene == SCENE.resolve()
    assert definition.scene.name == "BL23_vision_ocr_sorting_lab.ttt"
    assert not str(definition.scene).endswith("BL23_vision_lab.ttt")


def test_v1_08_tampered_asset_hash_fails_before_plan_generation(tmp_path: Path) -> None:
    source_manifest = ROOT / "simulation" / "vision_ocr_sorting_lab" / "ocr_assets_manifest.json"
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest_root = tmp_path / "assets"
    manifest_root.mkdir()
    for record in payload["training"] + payload["labels"]:
        destination = manifest_root / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_manifest.parent / record["path"]).read_bytes())
    tampered = manifest_root / payload["training"][0]["path"]
    tampered.write_bytes(tampered.read_bytes() + b"tampered")
    manifest = manifest_root / "ocr_assets_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OcrAssetError) as captured:
        load_ocr_assets(manifest, expected_scene_id="V1-08")
    assert captured.value.code == "OCR_ASSET_HASH_MISMATCH"


def test_v1_08_incomplete_recognition_fails_before_sort_plan() -> None:
    # The public plan builder is the motion-independent boundary: incomplete
    # recognition cannot produce the frozen plan that activates the guard.
    from tests.test_experiments.test_ocr_sorting import _analysis, _config

    analysis = _analysis()
    incomplete = type(analysis)(
        training_report=analysis.training_report,
        observations=analysis.observations[:3],
        image_size=analysis.image_size,
    )
    with pytest.raises(OcrSortError) as captured:
        from vision_platform.experiments.ocr_sorting import build_ocr_sort_plan

        build_ocr_sort_plan(_config(), incomplete, scene_part_ids={"part_a", "part_b", "part_c", "part_d"})
    assert captured.value.code == "OCR_SORT_RESULT_INVALID"


def test_v1_08_online_wrapper_uses_owned_launcher_and_exact_scope() -> None:
    assert WRAPPER.is_file(), f"missing V1-08 online wrapper: {WRAPPER}"
    source = WRAPPER.read_text(encoding="utf-8")
    assert "launch_coppeliasim.ps1" in source
    assert "process_ownership.ps1" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "23008" in source
    assert "BL23_vision_ocr_sorting_lab.ttt" in source
    assert '"tests/test_acceptance/test_coppeliasim_v1_08.py"' in source
    assert '"-m", "coppeliasim"' in source
    assert "Stop-Process -Name" not in source
    assert "taskkill" not in source.lower()


@pytest.mark.coppeliasim
def test_v1_08_online_ocr_sorting_is_complete(tmp_path: Path, request) -> None:
    host, port = _endpoint(request)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError as error:
        pytest.fail(
            f"No owned CoppeliaSim listener at {host}:{port}; run the V1-08 wrapper first: {error}"
        )

    output_root = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision_platform.cli",
            "experiment-run",
            "--experiment",
            "V1-08",
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
        timeout=300,
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
    assert summary["scene_probe_status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"

    commands = [
        json.loads(line)
        for line in (evidence_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    names = [item["name"] for item in commands]
    assert names[0] == "vision2d.ocr_sorting"
    assert names.count("vision2d.ocr_sorting") == 1
    assert names.count("vision2d.ocr_sort_entry") == 4
    assert not any(name.startswith("robot.") or name.startswith("tool.") for name in names)

    final_probe = json.loads((evidence_dir / "scene-final.json").read_text(encoding="utf-8"))
    assert final_probe["status"] == "PASS"
    assert final_probe["matched"] == final_probe["expected"] == 4
    assert final_probe["same_run_evidence"] is True
    assert final_probe["robot_home"] is True
    assert final_probe["tool_off"] is True
    assert len(final_probe["entry_evidence"]) == 4
    assert all(item["snapshot_id"] == final_probe["snapshot_id"] for item in final_probe["entry_evidence"])

    entry_artifacts = sorted(evidence_dir.glob("ocr-entry-*.json"))
    assert len(entry_artifacts) == 4
    bundles = sorted(evidence_dir.glob("vision-bundle-*.json"))
    assert len(bundles) == 1
    bundle = json.loads(bundles[0].read_text(encoding="utf-8"))
    assert bundle["experiment_id"] == "V1-08"
    assert bundle["status"] == "PASS"
    assert len(bundle["result"]["results"]) == 4
    assert bundle["result"]["training"]["held_out_accuracy"] >= 0.95
    assert bundle["hardware_status"] == "PENDING_HARDWARE"

