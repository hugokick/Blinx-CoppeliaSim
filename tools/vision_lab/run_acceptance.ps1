[CmdletBinding()]
param(
    [string]$OutputDir = "artifacts\vision_lab\final",
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) {
        $env:COPPELIASIM_ROOT
    } else {
        "E:\CoppeliaSim"
    }),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
$Scene = Join-Path `
    $ProjectRoot `
    "simulation\vision_lab\BL23_vision_lab.ttt"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$SummaryPath = Join-Path $OutputDir "acceptance-summary.json"
$Steps = [ordered]@{}
$FailureMessage = $null
$OwnedProcessId = $null

# Required live gates are intentionally explicit for delivery auditing:
# python -m vision_platform.cli accept
# python -m vision_platform.cli simui-smoke
# python -m pytest ... -m coppeliasim
# python -m simulation.vision_lab.verify_scene
# Hardware remains PENDING_HARDWARE.

function Invoke-CheckedPython {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $StartedAt = Get-Date
    & $Python @Arguments
    $ExitCode = $LASTEXITCODE
    $Steps[$Name] = [ordered]@{
        status = $(if ($ExitCode -eq 0) { "PASS" } else { "FAIL" })
        exit_code = $ExitCode
        elapsed_seconds = [math]::Round(
            ((Get-Date) - $StartedAt).TotalSeconds,
            3
        )
        command = "python " + ($Arguments -join " ")
    }
    if ($ExitCode -ne 0) {
        throw "Step '$Name' failed with exit code $ExitCode"
    }
}

$ListenerBefore = Get-NetTCPConnection `
    -LocalPort $Port `
    -State Listen `
    -ErrorAction SilentlyContinue |
    Select-Object -First 1

Push-Location $ProjectRoot
try {
    $Launch = & (Join-Path $PSScriptRoot "launch_coppeliasim.ps1") `
        -CoppeliaRoot $CoppeliaRoot `
        -Scene $Scene `
        -HostAddress $HostAddress `
        -Port $Port `
        -Hidden
    $Launch | Format-Table -AutoSize | Out-Host
    if (-not $ListenerBefore -and $Launch.StartedByScript) {
        $OwnedProcessId = [int]$Launch.ProcessId
    }

    Invoke-CheckedPython -Name "closed_loop_acceptance" -Arguments @(
        "-m", "vision_platform.cli", "accept",
        "--camera", "sim",
        "--robot", "sim",
        "--scene", $Scene,
        "--host", $HostAddress,
        "--port", [string]$Port,
        "--object-count", "6",
        "--output", (Join-Path $OutputDir "closed-loop")
    )

    Invoke-CheckedPython -Name "simui_lifecycle" -Arguments @(
        "-m", "vision_platform.cli", "simui-smoke",
        "--host", $HostAddress,
        "--port", [string]$Port,
        "--duration", "2",
        "--output", (Join-Path $OutputDir "simui-smoke.json")
    )

    Invoke-CheckedPython -Name "live_coppeliasim_tests" -Arguments @(
        "-m", "pytest",
        "tests/test_acceptance/test_coppeliasim_calibration.py",
        "tests/test_acceptance/test_coppeliasim_classification.py",
        "-m", "coppeliasim",
        "--junitxml", (Join-Path $OutputDir "coppeliasim-tests.xml"),
        "-q"
    )

    $PreviousQtPlatform = $env:QT_QPA_PLATFORM
    $env:QT_QPA_PLATFORM = "offscreen"
    try {
        Invoke-CheckedPython -Name "pyqt_offscreen_smoke" -Arguments @(
            "-m", "pytest",
            "tests/test_vision_platform/test_pyqt_smoke.py",
            "--junitxml", (Join-Path $OutputDir "pyqt-smoke.xml"),
            "-q"
        )
    } finally {
        $env:QT_QPA_PLATFORM = $PreviousQtPlatform
    }

    Invoke-CheckedPython -Name "full_static_test_suite" -Arguments @(
        "-m", "pytest",
        "--junitxml", (Join-Path $OutputDir "pytest-full.xml"),
        "-q"
    )

    # Run the scene verifier last because it owns a complete
    # stop/start/stop cycle and produces the final immutable render evidence.
    Invoke-CheckedPython -Name "scene_runtime_verification" -Arguments @(
        "-m", "simulation.vision_lab.verify_scene",
        "--host", $HostAddress,
        "--port", [string]$Port,
        "--report", (Join-Path $OutputDir "scene-runtime.json"),
        "--frame", (Join-Path $OutputDir "scene-camera.png"),
        "--robot-views-dir", (Join-Path $OutputDir "robot-views")
    )
} catch {
    $FailureMessage = $_.Exception.Message
} finally {
    Pop-Location
    $OverallStatus = if ($FailureMessage) { "FAIL" } else { "PASS" }
    $Summary = [ordered]@{
        schema_version = 1
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        status = $OverallStatus
        automated_scope = "CoppeliaSim and replay-independent software gates"
        steps = $Steps
        failure = $FailureMessage
        hardware_status = "PENDING_HARDWARE"
        pending_hardware = @(
            "Hikvision camera enumeration and MVS acquisition",
            "real three-point calibration",
            "emergency stop",
            "low-speed dry run",
            "single-object physical grasp",
            "multi-object physical classification"
        )
        note = (
            "PENDING_HARDWARE items are not counted as automated PASS. " +
            "Use the hardware-transfer checklist before any lab claim."
        )
    }
    $Summary |
        ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $SummaryPath -Encoding UTF8
    if ($OwnedProcessId) {
        $OwnedProcess = Get-Process `
            -Id $OwnedProcessId `
            -ErrorAction SilentlyContinue
        if (
            $OwnedProcess -and
            $OwnedProcess.Path -eq (Join-Path $CoppeliaRoot "coppeliaSim.exe")
        ) {
            Stop-Process -Id $OwnedProcessId -Force
        }
    }
}

Write-Host "Acceptance summary: $SummaryPath"
if ($FailureMessage) {
    Write-Error $FailureMessage
    exit 1
}
