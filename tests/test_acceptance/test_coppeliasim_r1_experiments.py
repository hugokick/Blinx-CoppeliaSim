from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from vision_platform.experiments.catalog import ExperimentCatalog


ROOT = Path(__file__).resolve().parents[2]
R1_EXPERIMENTS = ("R1-01", "R1-02", "R1-05", "R1-06", "R1-07")
SNAPSHOT_EXPERIMENTS = frozenset({"R1-05", "R1-06", "R1-07"})
FORMAL_CATALOG = ExperimentCatalog.load(
    ROOT / "config/experiments/catalog.json",
    project_root=ROOT,
)


def _single_stdout_object(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected exactly one JSON stdout line, got {lines!r}"
    payload = json.loads(lines[0])
    assert isinstance(payload, dict), "stdout JSON must be an object"
    return payload


def _inside_output(raw: Any, output_root: Path) -> Path:
    assert isinstance(raw, str) and raw.strip(), "evidence path must be non-empty"
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    root = output_root.resolve()
    assert candidate == root or root in candidate.parents, (
        f"evidence path escaped pytest output root: {candidate}"
    )
    return candidate


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"JSON object required: {path}"
    return payload


def _command_records(
    commands: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "command_id": f"{index:06d}",
            "name": name,
            "args": args,
            "timestamp": "2026-08-01T00:00:00+00:00",
        }
        for index, (name, args) in enumerate(commands, start=1)
    ]


def _expected_motion_records(experiment_id: str) -> list[dict[str, Any]]:
    definition = FORMAL_CATALOG.require(experiment_id)
    commands: list[tuple[str, dict[str, Any]]] = [
        ("experiment.info", {}),
    ]
    if experiment_id == "R1-01":
        observation = definition.public_parameters["observation_pose_mm"]
        commands.extend(
            [
                ("context.log", {"message": "start"}),
                ("robot.home", {}),
                ("robot.pose", {}),
                ("context.log", {"message": "home"}),
                (
                    "robot.move_world",
                    {
                        "x_mm": observation[0],
                        "y_mm": observation[1],
                        "z_mm": observation[2],
                        "speed": 12,
                    },
                ),
                ("context.checkpoint", {"label": "观察六个关节和 TCP"}),
                ("robot.home", {}),
                ("context.log", {"message": "done"}),
            ]
        )
    elif experiment_id == "R1-02":
        commands.append(("robot.home", {}))
        for index, point in enumerate(
            definition.public_parameters["teach_points_mm"],
            start=1,
        ):
            commands.extend(
                [
                    (
                        "robot.move_world",
                        {
                            "x_mm": point[0],
                            "y_mm": point[1],
                            "z_mm": point[2],
                            "speed": 12,
                        },
                    ),
                    ("robot.pose", {}),
                    ("context.log", {"message": f"point {index}"}),
                    ("context.checkpoint", {"label": f"示教点 {index}"}),
                ]
            )
        commands.append(("robot.home", {}))
    else:
        raise AssertionError(f"motion fixture is not defined for {experiment_id}")
    return _command_records(commands)


def _write_formal_evidence_fixture(
    tmp_path: Path,
    experiment_id: str,
    commands: list[dict[str, Any]],
    *,
    source: bytes | None = None,
    program_path: Path | None = None,
    source_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    definition = FORMAL_CATALOG.require(experiment_id)
    evidence_dir = tmp_path / experiment_id
    evidence_dir.mkdir()
    selected_source = (
        definition.student_template.read_bytes() if source is None else source
    )
    selected_path = (
        definition.student_template if program_path is None else program_path
    )
    selected_sha = (
        hashlib.sha256(selected_source).hexdigest()
        if source_sha256 is None
        else source_sha256
    )
    (evidence_dir / "source.py").write_bytes(selected_source)
    (evidence_dir / "source.sha256").write_text(
        selected_sha + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "commands.jsonl").write_text(
        "".join(
            json.dumps(command, ensure_ascii=False) + "\n"
            for command in commands
        ),
        encoding="utf-8",
    )
    summary = {
        "program_path": str(selected_path.resolve()),
        "source_sha256": selected_sha,
    }
    manifest = dict(summary)
    return evidence_dir, summary, manifest


def _assert_formal_run_evidence(
    experiment_id: str,
    evidence_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    definition = FORMAL_CATALOG.require(experiment_id)
    formal_template = definition.student_template.resolve()
    for label, payload in (("summary", summary), ("manifest", manifest)):
        program_path = payload.get("program_path")
        assert isinstance(program_path, str), (
            f"{label} program_path must bind to formal template"
        )
        assert Path(program_path).expanduser().resolve() == formal_template, (
            f"{label} program_path must bind to formal template"
        )

    formal_source = formal_template.read_bytes()
    formal_sha = hashlib.sha256(formal_source).hexdigest()
    assert (evidence_dir / "source.py").read_bytes() == formal_source, (
        "source.py must match formal template"
    )
    assert summary.get("source_sha256") == formal_sha, (
        "summary source SHA must match formal template"
    )
    assert manifest.get("source_sha256") == formal_sha, (
        "manifest source SHA must match formal template"
    )
    assert (evidence_dir / "source.sha256").read_text(
        encoding="utf-8"
    ).strip() == formal_sha, "source.sha256 must match formal template"

    commands: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        (evidence_dir / "commands.jsonl").read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        command = json.loads(line)
        assert isinstance(command, dict), (
            f"commands.jsonl line {line_number} must be an object"
        )
        assert set(command) == {"command_id", "name", "args", "timestamp"}, (
            f"commands.jsonl line {line_number} fields mismatch"
        )
        assert (
            isinstance(command["command_id"], str) and command["command_id"]
        ), (
            f"commands.jsonl line {line_number} command_id is invalid"
        )
        assert isinstance(command["name"], str) and command["name"], (
            f"commands.jsonl line {line_number} name is invalid"
        )
        assert isinstance(command["args"], dict), (
            f"commands.jsonl line {line_number} args must be an object"
        )
        assert isinstance(command["timestamp"], str) and command["timestamp"], (
            f"commands.jsonl line {line_number} timestamp is invalid"
        )
        commands.append(command)

    if experiment_id == "R1-01":
        _assert_r1_01_motion_trace(commands, definition.public_parameters)
    elif experiment_id == "R1-02":
        _assert_r1_02_motion_trace(commands, definition.public_parameters)


def _assert_r1_01_motion_trace(
    commands: list[dict[str, Any]],
    public_parameters: Mapping[str, Any],
) -> None:
    names = {
        "robot.home",
        "robot.pose",
        "robot.move_world",
        "context.checkpoint",
    }
    trace = [command for command in commands if command["name"] in names]
    assert [command["name"] for command in trace] == [
        "robot.home",
        "robot.pose",
        "robot.move_world",
        "context.checkpoint",
        "robot.home",
    ], "R1-01 motion trace must open and end home with pose, move and checkpoint"
    observation = public_parameters["observation_pose_mm"]
    assert trace[0]["args"] == {} and trace[-1]["args"] == {}, (
        "R1-01 motion trace home commands must not have arguments"
    )
    assert trace[1]["args"] == {}, (
        "R1-01 motion trace pose must have no arguments"
    )
    assert trace[2]["args"] == {
        "x_mm": observation[0],
        "y_mm": observation[1],
        "z_mm": observation[2],
        "speed": 12,
    }, "R1-01 motion trace must use the formal observation pose at speed 12"
    assert trace[3]["args"] == {"label": "观察六个关节和 TCP"}, (
        "R1-01 motion trace checkpoint is invalid"
    )


def _assert_r1_02_motion_trace(
    commands: list[dict[str, Any]],
    public_parameters: Mapping[str, Any],
) -> None:
    names = {
        "robot.home",
        "robot.pose",
        "robot.move_world",
        "context.checkpoint",
    }
    trace = [command for command in commands if command["name"] in names]
    expected_names = ["robot.home"]
    for _ in range(4):
        expected_names.extend(
            ["robot.move_world", "robot.pose", "context.checkpoint"]
        )
    expected_names.append("robot.home")
    assert [command["name"] for command in trace] == expected_names, (
        "R1-02 trajectory must open and end home with four move/pose/checkpoint stages"
    )
    assert trace[0]["args"] == {} and trace[-1]["args"] == {}, (
        "R1-02 trajectory home commands must not have arguments"
    )
    points = public_parameters["teach_points_mm"]
    assert isinstance(points, (list, tuple)) and len(points) == 4, (
        "R1-02 trajectory requires four formal teach points"
    )
    for index, point in enumerate(points, start=1):
        offset = 1 + ((index - 1) * 3)
        move, pose, checkpoint = trace[offset : offset + 3]
        assert move["args"] == {
            "x_mm": point[0],
            "y_mm": point[1],
            "z_mm": point[2],
            "speed": 12,
        }, f"R1-02 trajectory teach point {index} is invalid"
        assert pose["args"] == {}, (
            f"R1-02 trajectory pose {index} must have no arguments"
        )
        assert checkpoint["args"] == {"label": f"示教点 {index}"}, (
            f"R1-02 trajectory checkpoint {index} is invalid"
        )


def _utf8_subprocess_env() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _subprocess_diagnostics(
    completed: subprocess.CompletedProcess[str],
) -> str:
    return (completed.stderr or "") + (completed.stdout or "")


def _endpoint(request) -> tuple[str, int]:
    host = (
        request.config.getoption("--coppelia-host")
        or os.environ.get("COPPELIA_HOST")
        or "127.0.0.1"
    )
    configured_port = request.config.getoption("--coppelia-port")
    port = (
        configured_port
        if configured_port is not None
        else int(os.environ.get("COPPELIA_PORT", "23000"))
    )
    return str(host), int(port)


def test_stdout_contract_rejects_diagnostics_and_non_object_json():
    with pytest.raises(AssertionError, match="exactly one"):
        _single_stdout_object('{"status":"PASS"}\ndiagnostic\n')
    with pytest.raises(AssertionError, match="object"):
        _single_stdout_object("[]\n")


def test_evidence_paths_cannot_escape_pytest_output_root(tmp_path):
    inside = tmp_path / "runs" / "one" / "summary.json"
    assert _inside_output(str(inside), tmp_path / "runs") == inside.resolve()
    with pytest.raises(AssertionError, match="escaped"):
        _inside_output(str(tmp_path / "outside.json"), tmp_path / "runs")


def test_experiment_run_subprocess_env_forces_utf8_without_mutating_parent(
    monkeypatch,
):
    monkeypatch.setenv("PYTHONIOENCODING", "cp936")
    monkeypatch.setenv("TASK15_ENV_SENTINEL", "preserved")

    child_env = _utf8_subprocess_env()

    assert child_env["PYTHONIOENCODING"] == "utf-8"
    assert child_env["TASK15_ENV_SENTINEL"] == "preserved"
    assert os.environ["PYTHONIOENCODING"] == "cp936"


def test_subprocess_diagnostics_tolerates_missing_captured_stream():
    completed = subprocess.CompletedProcess(
        args=["python"],
        returncode=1,
        stdout=None,
        stderr="original stderr",
    )

    assert _subprocess_diagnostics(completed) == "original stderr"


@pytest.mark.parametrize("experiment_id", ("R1-01", "R1-02"))
def test_formal_evidence_helper_accepts_expected_motion_trajectory(
    tmp_path,
    experiment_id,
):
    evidence_dir, summary, manifest = _write_formal_evidence_fixture(
        tmp_path,
        experiment_id,
        _expected_motion_records(experiment_id),
    )

    _assert_formal_run_evidence(
        experiment_id,
        evidence_dir,
        summary,
        manifest,
    )


@pytest.mark.parametrize(
    "corruption",
    ("program-path", "source", "source-sha"),
)
def test_formal_evidence_helper_rejects_wrong_template_binding(
    tmp_path,
    corruption,
):
    definition = FORMAL_CATALOG.require("R1-01")
    kwargs: dict[str, Any] = {}
    if corruption == "program-path":
        kwargs["program_path"] = (
            ROOT / "student_programs/templates/basic_motion.py"
        )
    elif corruption == "source":
        kwargs["source"] = b"def main(ctx):\n    pass\n"
    else:
        assert corruption == "source-sha"
        kwargs["source_sha256"] = "0" * 64
    evidence_dir, summary, manifest = _write_formal_evidence_fixture(
        tmp_path,
        definition.experiment_id,
        _expected_motion_records(definition.experiment_id),
        **kwargs,
    )

    with pytest.raises(AssertionError, match="formal template"):
        _assert_formal_run_evidence(
            definition.experiment_id,
            evidence_dir,
            summary,
            manifest,
        )


def test_formal_evidence_helper_rejects_noop_motion_trace(tmp_path):
    evidence_dir, summary, manifest = _write_formal_evidence_fixture(
        tmp_path,
        "R1-01",
        _command_records(
            [
                ("experiment.info", {}),
                ("robot.home", {}),
                ("robot.home", {}),
            ]
        ),
    )

    with pytest.raises(AssertionError, match="R1-01 motion trace"):
        _assert_formal_run_evidence(
            "R1-01",
            evidence_dir,
            summary,
            manifest,
        )


def test_formal_evidence_helper_rejects_wrong_teach_point_trajectory(tmp_path):
    commands = _expected_motion_records("R1-02")
    first_move = next(
        command for command in commands if command["name"] == "robot.move_world"
    )
    first_move["args"]["x_mm"] += 1
    evidence_dir, summary, manifest = _write_formal_evidence_fixture(
        tmp_path,
        "R1-02",
        commands,
    )

    with pytest.raises(AssertionError, match="R1-02 trajectory"):
        _assert_formal_run_evidence(
            "R1-02",
            evidence_dir,
            summary,
            manifest,
        )


@pytest.mark.coppeliasim
@pytest.mark.parametrize("experiment_id", R1_EXPERIMENTS)
def test_r1_experiment_runs_through_guarded_student_process(
    experiment_id,
    tmp_path,
    request,
):
    host, port = _endpoint(request)
    output_root = tmp_path / "runs"
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
            str(output_root),
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=_utf8_subprocess_env(),
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, _subprocess_diagnostics(completed)
    payload = _single_stdout_object(completed.stdout)

    assert payload["status"] == "PASS"
    assert payload["experiment_id"] == experiment_id
    assert payload["scene_probe_status"] == "PASS"
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    assert payload["error"] is None

    summary_path = _inside_output(payload["summary"], output_root)
    evidence_dir = _inside_output(payload["evidence"], output_root)
    assert summary_path.is_file()
    assert evidence_dir.is_dir()
    assert summary_path == evidence_dir / "summary.json"

    summary = _read_object(summary_path)
    manifest = _read_object(evidence_dir / "manifest.json")
    initial = _read_object(evidence_dir / "scene-initial.json")
    final = _read_object(evidence_dir / "scene-final.json")
    assert summary["status"] == "PASS"
    assert summary["experiment_id"] == experiment_id
    assert summary["scene_probe_status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    assert summary["error"] is None
    assert summary["cleanup_errors"] == []
    assert manifest["experiment_id"] == experiment_id
    assert manifest["hardware_status"] == "PENDING_HARDWARE"
    assert manifest["robot_backend"] == "sim"
    _assert_formal_run_evidence(
        experiment_id,
        evidence_dir,
        summary,
        manifest,
    )
    assert summary["source_sha256"] == manifest["source_sha256"]
    assert (evidence_dir / "source.sha256").read_text(
        encoding="utf-8"
    ).strip() == summary["source_sha256"]
    assert hashlib.sha256((evidence_dir / "source.py").read_bytes()).hexdigest() == (
        summary["source_sha256"]
    )
    for phase, probe in (("initial", initial), ("final", final)):
        assert probe["experiment_id"] == experiment_id
        assert probe["phase"] == phase
        assert probe["status"] == "PASS"
        assert probe["hardware_status"] == "PENDING_HARDWARE"

    snapshots_path = evidence_dir / "snapshots.jsonl"
    if experiment_id in SNAPSHOT_EXPERIMENTS:
        assert snapshots_path.is_file()
        lines = [
            line
            for line in snapshots_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 1
        snapshot = json.loads(lines[0])
        assert snapshot["width"] == 640
        assert snapshot["height"] == 480
        assert snapshot["source"] == "coppeliasim"
        frame_path = _inside_output(
            str(evidence_dir / snapshot["path"]),
            output_root,
        )
        assert evidence_dir in frame_path.parents
        assert frame_path.is_file()
        assert snapshot["sha256"] == hashlib.sha256(
            frame_path.read_bytes()
        ).hexdigest()
