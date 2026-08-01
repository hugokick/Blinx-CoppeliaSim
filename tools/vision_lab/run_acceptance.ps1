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
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$SummaryPath = Join-Path $OutputDir "acceptance-summary.json"
$V22JUnitPath = Join-Path $OutputDir "v2-2-first-batch.xml"
$OwnedEvidencePaths = @($SummaryPath, $V22JUnitPath)
$InvalidEvidencePaths = @()
foreach ($EvidencePath in $OwnedEvidencePaths) {
    if (Test-Path -LiteralPath $EvidencePath) {
        if (Test-Path -LiteralPath $EvidencePath -PathType Leaf) {
            Remove-Item -LiteralPath $EvidencePath -Force
        } else {
            $InvalidEvidencePaths += $EvidencePath
        }
    }
}
if ($InvalidEvidencePaths.Count -gt 0) {
    throw (
        "Acceptance evidence target is not a file: " +
        ($InvalidEvidencePaths -join ", ")
    )
}

. (Join-Path $PSScriptRoot "process_ownership.ps1")
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
$Scene = Join-Path `
    $ProjectRoot `
    "simulation\vision_lab\BL23_vision_lab.ttt"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}
$Steps = [ordered]@{}
$FailureMessage = $null
$OwnedProcessId = $null
$OwnedProcessPath = $null
$OwnedProcessStartTimeUtcTicks = $null
$CallerQtPlatform = $env:QT_QPA_PLATFORM
$V22FirstBatchGatePassed = $false
$V22FirstBatchStatus = "NOT_RUN"

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

function Assert-JUnitNoSkips {
    param(
        [string]$Name,
        [string]$Path,
        [int]$ExpectedTests = 0
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        $Steps[$Name]["status"] = "FAIL"
        throw "Step '$Name' did not produce JUnit evidence: $Path"
    }
    [xml]$Report = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    $Suites = @($Report.SelectNodes("//testsuite"))
    if ($Suites.Count -eq 0) {
        $Steps[$Name]["status"] = "FAIL"
        throw "Step '$Name' produced invalid JUnit evidence: $Path"
    }
    [int]$Tests = 0
    [int]$Skipped = 0
    [int]$Failures = 0
    [int]$Errors = 0
    foreach ($Suite in $Suites) {
        $Tests += [int]$Suite.tests
        $Skipped += [int]$Suite.skipped
        $Failures += [int]$Suite.failures
        $Errors += [int]$Suite.errors
    }
    $Steps[$Name]["tests"] = $Tests
    $Steps[$Name]["skipped"] = $Skipped
    $Steps[$Name]["failures"] = $Failures
    $Steps[$Name]["errors"] = $Errors
    $WrongTestCount = (
        $ExpectedTests -gt 0 -and $Tests -ne $ExpectedTests
    )
    if (
        $Tests -lt 1 -or
        $WrongTestCount -or
        $Skipped -ne 0 -or
        $Failures -ne 0 -or
        $Errors -ne 0
    ) {
        $Steps[$Name]["status"] = "FAIL"
        throw (
            "Step '$Name' requires the exact executed test count with " +
            "zero skips, failures and errors; tests=$Tests " +
            "expected=$ExpectedTests skipped=$Skipped " +
            "failures=$Failures errors=$Errors"
        )
    }
}

function Invoke-AcceptanceWorkflow {
    Push-Location $ProjectRoot
    try {
    $Launch = & (Join-Path $PSScriptRoot "launch_coppeliasim.ps1") `
        -CoppeliaRoot $CoppeliaRoot `
        -Scene $Scene `
        -HostAddress $HostAddress `
        -Port $Port `
        -Hidden
    $Launch | Format-Table -AutoSize | Out-Host
    if ($Launch.StartedByScript) {
        $OwnedProcessId = [int]$Launch.ProcessId
        $OwnedProcessPath = [string]$Launch.ProcessPath
        $OwnedProcessStartTimeUtcTicks = `
            [long]$Launch.ProcessStartTimeUtcTicks
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
        "--coppelia-host", $HostAddress,
        "--coppelia-port", [string]$Port,
        "--junitxml", (Join-Path $OutputDir "coppeliasim-tests.xml"),
        "-q"
    )

    Invoke-CheckedPython -Name "student_program_online" -Arguments @(
        "-m", "pytest",
        "tests/test_acceptance/test_coppeliasim_student_program.py",
        "-m", "coppeliasim",
        "--coppelia-host", $HostAddress,
        "--coppelia-port", [string]$Port,
        "--junitxml", (Join-Path $OutputDir "student-program.xml"),
        "-q"
    )
    Assert-JUnitNoSkips `
        -Name "student_program_online" `
        -Path (Join-Path $OutputDir "student-program.xml")

    Invoke-CheckedPython -Name "v2_2_first_batch_online" -Arguments @(
        "-m", "pytest",
        "tests/test_acceptance/test_coppeliasim_training_scenes.py",
        "tests/test_acceptance/test_coppeliasim_r1_experiments.py",
        "-m", "coppeliasim",
        "--coppelia-host", $HostAddress,
        "--coppelia-port", [string]$Port,
        "--junitxml", $V22JUnitPath,
        "-q"
    )
    Assert-JUnitNoSkips `
        -Name "v2_2_first_batch_online" `
        -Path $V22JUnitPath `
        -ExpectedTests 7
    $V22FirstBatchGatePassed = $true

    $env:QT_QPA_PLATFORM = "offscreen"
    try {
        Invoke-CheckedPython -Name "pyqt_offscreen_smoke" -Arguments @(
            "-m", "pytest",
            "tests/test_vision_platform/test_pyqt_smoke.py",
            "--junitxml", (Join-Path $OutputDir "pyqt-smoke.xml"),
            "-q"
        )

        Invoke-CheckedPython -Name "full_static_test_suite" -Arguments @(
            "-m", "pytest",
            "--junitxml", (Join-Path $OutputDir "pytest-full.xml"),
            "-q"
        )
    } finally {
        $env:QT_QPA_PLATFORM = $null
    }

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
        if ($OwnedProcessId) {
            try {
                Stop-ExactOwnedProcess `
                    -ProcessId $OwnedProcessId `
                    -ProcessPath $OwnedProcessPath `
                    -ProcessStartTimeUtcTicks `
                        $OwnedProcessStartTimeUtcTicks
            } catch {
                $CleanupFailure = (
                    "Owned CoppeliaSim cleanup failed: " +
                    $_.Exception.Message
                )
                if ($FailureMessage) {
                    $FailureMessage = (
                        $FailureMessage + [Environment]::NewLine +
                        $CleanupFailure
                    )
                } else {
                    $FailureMessage = $CleanupFailure
                }
            }
        }
        $OverallStatus = if ($FailureMessage) { "FAIL" } else { "PASS" }
        if ($FailureMessage) {
            $V22FirstBatchStatus = if (
                $V22FirstBatchGatePassed -or
                $Steps.Contains("v2_2_first_batch_online")
            ) { "FAIL" } else { "NOT_RUN" }
        } elseif ($V22FirstBatchGatePassed) {
            $V22FirstBatchStatus = "PASS"
        } else {
            $V22FirstBatchStatus = "FAIL"
        }
        $Summary = [ordered]@{
            schema_version = 1
            generated_at = (Get-Date).ToUniversalTime().ToString("o")
            status = $OverallStatus
            automated_scope = (
                "CoppeliaSim and replay-independent software gates"
            )
            steps = $Steps
            failure = $FailureMessage
            v2_2_first_batch = $V22FirstBatchStatus
            r1_experiments = @(
                "R1-01",
                "R1-02",
                "R1-05",
                "R1-06",
                "R1-07"
            )
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
    }

    Write-Host "Acceptance summary: $SummaryPath"
    if ($FailureMessage) {
        throw $FailureMessage
    }
}

try {
    $env:QT_QPA_PLATFORM = $null
    Invoke-AcceptanceWorkflow
} finally {
    $env:QT_QPA_PLATFORM = $CallerQtPlatform
}
