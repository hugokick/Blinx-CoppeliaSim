[CmdletBinding()]
param(
    [string]$OutputDir = "artifacts\vision_lab\v1-08-online",
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) {
        $env:COPPELIASIM_ROOT
    } else {
        "E:\CoppeliaSim"
    }),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23008
)

$ErrorActionPreference = "Stop"
if ($Port -ne 23008) {
    throw "V1-08 online acceptance requires dedicated port 23008."
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$SummaryPath = Join-Path $OutputDir "v1-08-online-summary.json"
$JUnitPath = Join-Path $OutputDir "v1-08-online.xml"
$Scene = Join-Path $ProjectRoot "simulation\vision_ocr_sorting_lab\BL23_vision_ocr_sorting_lab.ttt"
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}
if (-not (Test-Path -LiteralPath $Scene -PathType Leaf)) {
    throw "V1-08 scene is missing: $Scene"
}

. (Join-Path $PSScriptRoot "process_ownership.ps1")

$Steps = [ordered]@{}
$FailureMessage = $null
$OwnedProcessId = $null
$OwnedProcessPath = $null
$OwnedProcessStartTimeUtcTicks = $null

function Invoke-CheckedPython {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $started = Get-Date
    & $Python @Arguments
    $exitCode = $LASTEXITCODE
    $Steps[$Name] = [ordered]@{
        status = $(if ($exitCode -eq 0) { "PASS" } else { "FAIL" })
        exit_code = $exitCode
        elapsed_seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
        command = "python " + ($Arguments -join " ")
    }
    if ($exitCode -ne 0) {
        throw "Step '$Name' failed with exit code $exitCode"
    }
}

function Read-JUnitCounts {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Online acceptance did not produce JUnit evidence: $Path"
    }
    [xml]$report = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    $suites = @($report.SelectNodes("//testsuite"))
    if ($suites.Count -eq 0) {
        throw "Online acceptance produced invalid JUnit evidence: $Path"
    }
    $tests = 0
    $skipped = 0
    $failures = 0
    $errors = 0
    foreach ($suite in $suites) {
        $tests += [int]$suite.tests
        $skipped += [int]$suite.skipped
        $failures += [int]$suite.failures
        $errors += [int]$suite.errors
    }
    $Steps["online_coppeliasim"] = [ordered]@{
        status = $(if ($skipped -eq 0 -and $failures -eq 0 -and $errors -eq 0) { "PASS" } else { "FAIL" })
        tests = $tests
        skipped = $skipped
        failures = $failures
        errors = $errors
        junit = $Path
    }
    if ($tests -lt 1 -or $skipped -ne 0 -or $failures -ne 0 -or $errors -ne 0) {
        throw "V1-08 online acceptance requires tests>0 and zero skipped/failures/errors; tests=$tests skipped=$skipped failures=$failures errors=$errors"
    }
}

try {
    Push-Location $ProjectRoot
    try {
        $launch = & (Join-Path $PSScriptRoot "launch_coppeliasim.ps1") `
            -CoppeliaRoot $CoppeliaRoot `
            -Scene $Scene `
            -HostAddress $HostAddress `
            -Port 23008 `
            -Hidden
        if (-not $launch -or $null -eq $launch.ProcessId) {
            throw "launch_coppeliasim.ps1 did not return an owned process identity"
        }
        if ($launch.StartedByScript) {
            $OwnedProcessId = [int]$launch.ProcessId
            $OwnedProcessPath = [string]$launch.ProcessPath
            $OwnedProcessStartTimeUtcTicks = [long]$launch.ProcessStartTimeUtcTicks
        }
        $Steps["launch"] = [ordered]@{
            status = "PASS"
            port = 23008
            scene = $Scene
            started_by_script = [bool]$launch.StartedByScript
            process_id = [int]$launch.ProcessId
        }

        $env:COPPELIA_HOST = $HostAddress
        $env:COPPELIA_PORT = "23008"
        $env:PYTHONUTF8 = "1"
        $env:PYTHONIOENCODING = "utf-8"
        Invoke-CheckedPython -Name "online_coppeliasim" -Arguments @(
            "-m", "pytest",
            "tests/test_acceptance/test_coppeliasim_v1_08.py",
            "-m", "coppeliasim",
            "--coppelia-host", $HostAddress,
            "--coppelia-port", "23008",
            "--junitxml", $JUnitPath,
            "-q"
        )
        Read-JUnitCounts -Path $JUnitPath
    } finally {
        Pop-Location
    }
} catch {
    $FailureMessage = $_.Exception.Message
} finally {
    if ($null -ne $OwnedProcessId) {
        try {
            Stop-ExactOwnedProcess `
                -ProcessId $OwnedProcessId `
                -ProcessPath $OwnedProcessPath `
                -ProcessStartTimeUtcTicks $OwnedProcessStartTimeUtcTicks
        } catch {
            $cleanup = "Owned CoppeliaSim cleanup failed: $($_.Exception.Message)"
            $FailureMessage = if ($FailureMessage) {
                "$FailureMessage`n$cleanup"
            } else {
                $cleanup
            }
        }
    }
    $summary = [ordered]@{
        schema_version = 1
        status = $(if ($FailureMessage) { "FAIL" } else { "PASS" })
        port = 23008
        scene = $Scene
        steps = $Steps
        failure = $FailureMessage
        human_acceptance = "PENDING_HUMAN_ACCEPTANCE"
        hardware_status = "PENDING_HARDWARE"
    }
    $summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
}

Write-Host "V1-08 online acceptance summary: $SummaryPath"
if ($FailureMessage) {
    throw $FailureMessage
}
