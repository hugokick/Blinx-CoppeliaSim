from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.rgbd_lab.build_rgbd_scene import (
    D1_PORT,
    PROJECT_ROOT,
    validate_build_arguments,
)
from tests.test_rgbd_sim import coppeliasim_process as lifecycle


ROOT = Path(__file__).resolve().parents[2]
BUILDER_SCRIPT = ROOT / "tools" / "rgbd_lab" / "build_rgbd_scene.ps1"
LISTENER_IDENTITY = ROOT / "tools" / "rgbd_lab" / "listener_identity.ps1"


def test_builder_is_bound_to_dedicated_spec_and_port() -> None:
    spec = PROJECT_ROOT / "simulation" / "rgbd_lab" / "scene_spec.json"
    assert D1_PORT == 23009
    assert validate_build_arguments(spec_path=spec, port=23009) is None


def test_builder_rejects_other_ports_and_specs() -> None:
    spec = PROJECT_ROOT / "simulation" / "rgbd_lab" / "scene_spec.json"
    with pytest.raises(ValueError, match="23009"):
        validate_build_arguments(spec_path=spec, port=23008)
    with pytest.raises(ValueError, match="dedicated"):
        validate_build_arguments(
            spec_path=PROJECT_ROOT / "simulation" / "vision_lab" / "scene_spec.json",
            port=23009,
        )


def test_builder_spec_is_standalone_and_has_no_runtime_output_root() -> None:
    spec_path = PROJECT_ROOT / "simulation" / "rgbd_lab" / "scene_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert set(spec) >= {"formal_scene_template", "sensor", "required_paths"}
    assert "artifacts" not in spec["output_relative"]
    assert "evidence" not in spec["output_relative"]


def test_builder_rechecks_listener_identity_after_readiness_before_build() -> None:
    source = BUILDER_SCRIPT.read_text(encoding="utf-8")
    identity_source = LISTENER_IDENTITY.read_text(encoding="utf-8")
    readiness = source.index("vision_platform.coppeliasim_readiness")
    listener_check = source.index("Assert-RgbdOwnedListener", readiness)
    builder = source.index("tools.rgbd_lab.build_rgbd_scene", listener_check)
    assert listener_check < builder
    assert "ProcessPath" in identity_source
    assert "ProcessStartTimeUtcTicks" in identity_source


def _run_listener_identity_scenario(scenario: str) -> dict[str, str]:
    helper = str(LISTENER_IDENTITY).replace("'", "''")
    script = rf"""
. '{helper}'
$fakeProcess = [pscustomobject]@{{
    Id = 23009
    Path = 'C:\CoppeliaSim\coppeliaSim.exe'
    StartTime = [datetime]'2026-08-03T09:00:00'
}}
$replacementProcess = [pscustomobject]@{{
    Id = 23009
    Path = 'C:\Other\unexpected.exe'
    StartTime = [datetime]'2026-08-03T09:01:00'
}}
$expected = Get-RgbdProcessIdentity -Process $fakeProcess
$listenerAction = {{
    [pscustomobject]@{{ OwningProcess = 23009 }}
}}
$processAction = {{
    param($requestedId)
    if ('{scenario}' -eq 'replacement') {{ return $replacementProcess }}
    return $fakeProcess
}}
try {{
    $actual = Assert-RgbdOwnedListener `
        -ExpectedIdentity $expected `
        -GetListenerAction $listenerAction `
        -GetProcessAction $processAction
    [pscustomobject]@{{ status = 'PASS'; pid = $actual.ProcessId }} |
        ConvertTo-Json -Compress
}} catch {{
    [pscustomobject]@{{ status = 'ERROR'; error = $_.Exception.Message }} |
        ConvertTo-Json -Compress
}}
"""
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_builder_rejects_replacement_listener_identity() -> None:
    result = _run_listener_identity_scenario("replacement")
    assert result["status"] == "ERROR"
    assert "identity mismatch" in result["error"].lower()


def test_builder_accepts_the_recorded_listener_identity() -> None:
    result = _run_listener_identity_scenario("owned")
    assert result == {"status": "PASS", "pid": 23009}


class _FakeProcess:
    pid = 1234

    def __init__(self, *, exited: bool = False) -> None:
        self.exited = exited
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return 0 if self.exited else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.exited = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired("fake-coppeliasim", timeout)
        self.exited = True
        return 0


def _owned_process(process: _FakeProcess) -> lifecycle.OwnedCoppeliaSim:
    return lifecycle.OwnedCoppeliaSim(
        process=process,
        executable=Path("coppeliaSim.exe"),
        scene=Path("scene.ttt"),
        host="127.0.0.1",
        port=23009,
    )


def test_stop_cleans_owned_process_before_reporting_replacement_listener(monkeypatch) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(lifecycle, "_listener_pid", lambda port: 999)

    with pytest.raises(RuntimeError, match=r"replacement listener PID 999.*1234"):
        _owned_process(process).stop(timeout_s=0.01)

    assert process.terminated is True
    assert process.killed is True


def test_stop_is_idempotent_for_already_exited_owned_process(monkeypatch) -> None:
    process = _FakeProcess(exited=True)
    monkeypatch.setattr(lifecycle, "_listener_pid", lambda port: 999)

    _owned_process(process).stop(timeout_s=0.01)

    assert process.terminated is False
    assert process.killed is False
