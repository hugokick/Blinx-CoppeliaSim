from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "tools" / "vision_lab" / "process_ownership.ps1"
LAUNCHER = ROOT / "tools" / "vision_lab" / "launch_coppeliasim.ps1"
ACCEPTANCE = ROOT / "tools" / "vision_lab" / "run_acceptance.ps1"
POWERSHELL = "powershell.exe"


def _run_cleanup_scenario(scenario: str) -> dict:
    helper = str(HELPER).replace("'", "''")
    script = rf"""
. '{helper}'
$script:Scenario = '{scenario}'
$script:GetCalls = 0
$script:StopCalls = 0
$script:WaitResult = $true
$script:Fake = [pscustomobject]@{{
    Id = 4242
    Path = 'C:\CoppeliaSim\coppeliaSim.exe'
    Marker = 'verified'
    StartTime = [datetime]'2026-07-31T09:00:00'
}}
$script:Reused = [pscustomobject]@{{
    Id = 4242
    Path = 'C:\CoppeliaSim\coppeliaSim.exe'
    Marker = 'reused'
    StartTime = [datetime]'2026-07-31T09:01:00'
}}
$script:ExpectedStartTimeUtcTicks = [long](
    $script:Fake.StartTime.ToUniversalTime().Ticks
)
$script:Fake | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value {{
    param($TimeoutMilliseconds)
    return $script:WaitResult
}}
if ($script:Scenario -eq 'timeout') {{
    $script:WaitResult = $false
}}
if ($script:Scenario -eq 'path_mismatch') {{
    $script:Fake.Path = 'C:\Other\unexpected.exe'
}}
if ($script:Scenario -eq 'start_time_unreadable') {{
    $script:Fake.StartTime = $null
}}
$GetProcessAction = {{
    param($RequestedId)
    $script:GetCalls += 1
    if ($script:Scenario -eq 'already_exited') {{
        return $null
    }}
    if (
        $script:Scenario -eq 'success' -and
        $script:GetCalls -gt 1
    ) {{
        return $null
    }}
    if (
        $script:Scenario -eq 'pid_reused' -and
        $script:GetCalls -gt 1
    ) {{
        return $script:Reused
    }}
    if ($script:Scenario -eq 'replacement_before_cleanup') {{
        return $script:Reused
    }}
    return $script:Fake
}}
$StopProcessAction = {{
    param($VerifiedProcess)
    $script:StopCalls += 1
    $script:StoppedMarker = $VerifiedProcess.Marker
    $script:StoppedWasVerified = [object]::ReferenceEquals(
        $VerifiedProcess,
        $script:Fake
    )
    $script:StoppedWasReused = [object]::ReferenceEquals(
        $VerifiedProcess,
        $script:Reused
    )
}}
$ErrorMessage = $null
try {{
    Stop-ExactOwnedProcess `
        -ProcessId 4242 `
        -ProcessPath 'C:\CoppeliaSim\coppeliaSim.exe' `
        -ProcessStartTimeUtcTicks $script:ExpectedStartTimeUtcTicks `
        -TimeoutMilliseconds 5 `
        -GetProcessAction $GetProcessAction `
        -StopProcessAction $StopProcessAction
}} catch {{
    $ErrorMessage = $_.Exception.Message
}}
[pscustomobject]@{{
    scenario = $script:Scenario
    get_calls = $script:GetCalls
    stop_calls = $script:StopCalls
    stopped_marker = $script:StoppedMarker
    stopped_was_verified = $script:StoppedWasVerified
    stopped_was_reused = $script:StoppedWasReused
    error = $ErrorMessage
}} | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _run_launcher_identity_capture_failure(tmp_path: Path) -> dict:
    fake_root = tmp_path / "CoppeliaSim"
    fake_root.mkdir()
    (fake_root / "coppeliaSim.exe").write_bytes(b"")
    launcher = str(LAUNCHER).replace("'", "''")
    root = str(fake_root).replace("'", "''")
    scene = str(
        ROOT / "simulation" / "vision_lab" / "BL23_vision_lab.ttt"
    ).replace("'", "''")
    script = rf"""
$global:StartCalls = 0
$global:StopCalls = 0
$global:WaitCalls = 0
$global:GetProcessCalls = 0
$global:FakeStarted = [pscustomobject]@{{
    Id = 4242
    Marker = 'raw-start-process-object'
    Path = '{root}\coppeliaSim.exe'
    StartTime = $null
}}
$global:FakeStarted | Add-Member `
    -MemberType ScriptMethod `
    -Name WaitForExit `
    -Value {{
        param($TimeoutMilliseconds)
        $global:WaitCalls += 1
        return $true
    }}
function global:Get-NetTCPConnection {{
    param($LocalPort, $State, $ErrorAction)
    return $null
}}
function global:Start-Process {{
    param(
        $FilePath,
        $ArgumentList,
        $WorkingDirectory,
        [switch]$PassThru,
        $WindowStyle
    )
    $global:StartCalls += 1
    return $global:FakeStarted
}}
function global:Get-Process {{
    param($Id, $ErrorAction)
    $global:GetProcessCalls += 1
    throw 'cleanup must not query by unverified PID'
}}
function global:Stop-Process {{
    param($InputObject, [switch]$Force, $ErrorAction)
    $global:StopCalls += 1
    $global:StoppedWasRawObject = [object]::ReferenceEquals(
        $InputObject,
        $global:FakeStarted
    )
}}
$ErrorMessage = $null
try {{
    & '{launcher}' `
        -CoppeliaRoot '{root}' `
        -Scene '{scene}' `
        -Port 23999 `
        -Hidden
}} catch {{
    $ErrorMessage = $_.Exception.Message
}}
[pscustomobject]@{{
    start_calls = $global:StartCalls
    stop_calls = $global:StopCalls
    wait_calls = $global:WaitCalls
    get_process_calls = $global:GetProcessCalls
    stopped_was_raw_object = $global:StoppedWasRawObject
    error = $ErrorMessage
}} | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _run_acceptance_environment_failure(
    tmp_path: Path,
    initial_value: str | None,
) -> dict:
    acceptance = str(ACCEPTANCE).replace("'", "''")
    output_dir = str(
        tmp_path
        / (
            "acceptance-with-caller-value"
            if initial_value is not None
            else "acceptance-without-caller-value"
        )
    ).replace("'", "''")
    missing_root = str(tmp_path / "missing-coppeliasim").replace("'", "''")
    if initial_value is None:
        initialise = (
            "Remove-Item Env:QT_QPA_PLATFORM "
            "-ErrorAction SilentlyContinue"
        )
    else:
        escaped_value = initial_value.replace("'", "''")
        initialise = f"$env:QT_QPA_PLATFORM = '{escaped_value}'"
    script = rf"""
$ErrorActionPreference = 'Stop'
$BeforeLocation = (Get-Location).Path
{initialise}
$Caught = $false
$CaughtMessage = $null
try {{
    . '{acceptance}' `
        -OutputDir '{output_dir}' `
        -CoppeliaRoot '{missing_root}' `
        -Port 23997
}} catch {{
    $Caught = $true
    $CaughtMessage = $_.Exception.Message
}}
$VariableExists = Test-Path Env:QT_QPA_PLATFORM
$VariableValue = if ($VariableExists) {{
    $env:QT_QPA_PLATFORM
}} else {{
    $null
}}
$SummaryStatus = $null
$SummaryPath = Join-Path '{output_dir}' 'acceptance-summary.json'
if (Test-Path -LiteralPath $SummaryPath -PathType Leaf) {{
    $SummaryStatus = (
        Get-Content -LiteralPath $SummaryPath -Raw -Encoding UTF8 |
        ConvertFrom-Json
    ).status
}}
[pscustomobject]@{{
    caught = $Caught
    caught_message = $CaughtMessage
    variable_exists = $VariableExists
    variable_value = $VariableValue
    location_restored = ((Get-Location).Path -eq $BeforeLocation)
    summary_status = $SummaryStatus
}} | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("scenario", ("already_exited", "success"))
def test_exact_owned_cleanup_accepts_normal_terminal_states(scenario: str):
    result = _run_cleanup_scenario(scenario)

    assert result["error"] is None
    assert result["stop_calls"] == (0 if scenario == "already_exited" else 1)
    if scenario == "success":
        assert result["stopped_marker"] == "verified"
        assert result["stopped_was_verified"] is True
        assert result["stopped_was_reused"] is False


@pytest.mark.parametrize("scenario", ("timeout", "survivor"))
def test_exact_owned_cleanup_fails_on_timeout_or_same_path_survivor(
    scenario: str,
):
    result = _run_cleanup_scenario(scenario)

    assert "did not exit" in result["error"]
    assert result["stop_calls"] == 1
    assert result["get_calls"] >= 2


@pytest.mark.parametrize(
    "scenario",
    ("path_mismatch", "replacement_before_cleanup"),
)
def test_exact_owned_cleanup_fails_closed_on_initial_identity_mismatch(
    scenario: str,
):
    result = _run_cleanup_scenario(scenario)

    assert "identity mismatch" in result["error"]
    assert result["stop_calls"] == 0


def test_exact_owned_cleanup_fails_closed_when_start_time_is_unreadable():
    result = _run_cleanup_scenario("start_time_unreadable")

    assert "start time" in result["error"].lower()
    assert result["stop_calls"] == 0


def test_pid_reuse_never_stops_the_replacement_process_object():
    result = _run_cleanup_scenario("pid_reused")

    assert result["error"] is None
    assert result["stop_calls"] == 1
    assert result["stopped_marker"] == "verified"
    assert result["stopped_was_verified"] is True
    assert result["stopped_was_reused"] is False
    assert result["get_calls"] == 2


def test_exact_owned_cleanup_uses_verified_process_object_not_pid():
    source = HELPER.read_text(encoding="utf-8")

    assert "[long]$ProcessStartTimeUtcTicks" in source
    assert "StartTime.ToUniversalTime().Ticks" in source
    assert "Stop-Process -InputObject $VerifiedProcess" in source
    assert "& $StopProcessAction $OwnedProcess" in source
    assert "& $StopProcessAction ([int]$ProcessId)" not in source


def test_launcher_cleans_raw_started_object_when_identity_capture_fails(
    tmp_path: Path,
):
    result = _run_launcher_identity_capture_failure(tmp_path)

    assert "Could not capture CoppeliaSim process identity" in result["error"]
    assert result["start_calls"] == 1
    assert result["stop_calls"] == 1
    assert result["wait_calls"] == 1
    assert result["get_process_calls"] == 0
    assert result["stopped_was_raw_object"] is True


def test_launcher_canonicalizes_dotted_root_before_process_identity_use():
    source = (
        ROOT / "tools" / "vision_lab" / "launch_coppeliasim.ps1"
    ).read_text(encoding="utf-8")

    root_resolution = source.index(
        "$CoppeliaRoot = (Resolve-Path -LiteralPath $CoppeliaRoot).Path"
    )
    executable_join = source.index(
        '$Executable = Join-Path $CoppeliaRoot "coppeliaSim.exe"'
    )
    start_parameters = source.index("$StartParameters = @{")
    assert root_resolution < executable_join < start_parameters
    assert (
        "$Executable = (Resolve-Path -LiteralPath $Executable).Path"
        in source
    )
    identity_capture = source.index("$LaunchProcessStartTimeUtcTicks")
    readiness = source.index(
        "& $Python -m vision_platform.coppeliasim_readiness"
    )
    assert identity_capture < readiness
    assert "ProcessStartTimeUtcTicks = $LaunchProcessStartTimeUtcTicks" in source
    catch_tail = source.split("} catch {", 1)[1]
    assert "-ProcessId $LaunchProcessId" in catch_tail
    assert "-ProcessPath $LaunchProcessPath" in catch_tail
    assert (
        "-ProcessStartTimeUtcTicks $LaunchProcessStartTimeUtcTicks"
        in catch_tail
    )
    assert "Stop-StartedProcessObject" in catch_tail
    assert "-StartedProcess $StartedProcess" in catch_tail
    assert "Stop-Process -Id" not in catch_tail


def test_all_coppeliasim_launchers_share_exact_owned_cleanup_helper():
    assert HELPER.is_file()
    for relative in (
        "tools/vision_lab/launch_coppeliasim.ps1",
        "tools/vision_lab/run_acceptance.ps1",
        "tools/vision_lab/run_student_program.ps1",
        "tools/vision_lab/run_pyqt.ps1",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "process_ownership.ps1" in source
        assert "Stop-ExactOwnedProcess" in source
        assert "ProcessStartTimeUtcTicks" in source
        assert "Stop-Process -Name" not in source
        assert ".WaitForExit(" not in source


def test_acceptance_cleanup_failure_precedes_and_controls_summary_status(
    tmp_path: Path,
):
    source = (
        ROOT / "tools" / "vision_lab" / "run_acceptance.ps1"
    ).read_text(encoding="utf-8")

    cleanup_position = source.rindex("Stop-ExactOwnedProcess")
    status_position = source.index("$OverallStatus")
    summary_position = source.index("$Summary =")
    write_position = source.index("Set-Content -LiteralPath $SummaryPath")
    assert cleanup_position < status_position < summary_position < write_position
    cleanup_tail = source[cleanup_position:status_position]
    assert "catch {" in cleanup_tail
    assert "$FailureMessage" in cleanup_tail
    assert "cleanup failed" in cleanup_tail.lower()
    assert "throw $FailureMessage" in source
    assert "exit 1" not in source

    for initial_value in (None, "caller-platform"):
        result = _run_acceptance_environment_failure(
            tmp_path,
            initial_value,
        )
        assert result["caught"] is True
        assert result["caught_message"]
        assert result["location_restored"] is True
        assert result["summary_status"] == "FAIL"
        assert result["variable_exists"] is (initial_value is not None)
        assert result["variable_value"] == initial_value
