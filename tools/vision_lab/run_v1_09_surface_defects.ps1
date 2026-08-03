[CmdletBinding()]
param(
    [string]$OutputDir = "artifacts\vision_lab\v1-09-online",
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) {
        $env:COPPELIASIM_ROOT
    } else {
        "E:\CoppeliaSim"
    }),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23010
)

$ErrorActionPreference = "Stop"
if ($Port -ne 23010) {
    throw "V1-09 online acceptance requires dedicated port 23010."
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$RuntimeEvidenceDir = Join-Path $OutputDir "runtime-evidence"
New-Item -ItemType Directory -Force -Path $RuntimeEvidenceDir | Out-Null
$RuntimeEvidenceDir = (Resolve-Path -LiteralPath $RuntimeEvidenceDir).Path
$RuntimeEvidenceEnvName = "V1_09_RUNTIME_EVIDENCE_DIR"
$PreviousRuntimeEvidenceDir = [Environment]::GetEnvironmentVariable(
    $RuntimeEvidenceEnvName,
    "Process"
)
$SummaryPath = Join-Path $OutputDir "v1-09-online-summary.json"
$JUnitPath = Join-Path $OutputDir "v1-09-online.xml"
$Scene = Join-Path $ProjectRoot "simulation\vision_defect_sorting_lab\BL23_vision_defect_sorting_lab.ttt"
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}
if (-not (Test-Path -LiteralPath $Scene -PathType Leaf)) {
    throw "V1-09 scene is missing: $Scene"
}

. (Join-Path $PSScriptRoot "process_ownership.ps1")

$Steps = [ordered]@{}
$FailureMessage = $null
$OwnedProcessId = $null
$OwnedProcessPath = $null
$OwnedProcessStartTimeUtcTicks = $null

function Read-JUnitCounts {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Online acceptance did not produce JUnit evidence: $Path"
    }
    [xml]$Report = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    $Suites = @($Report.SelectNodes("//testsuite"))
    if ($Suites.Count -eq 0) {
        throw "Online acceptance produced invalid JUnit evidence: $Path"
    }
    $Tests = 0
    $Skipped = 0
    $Failures = 0
    $Errors = 0
    foreach ($Suite in $Suites) {
        $Tests += [int]$Suite.tests
        $Skipped += [int]$Suite.skipped
        $Failures += [int]$Suite.failures
        $Errors += [int]$Suite.errors
    }
    $Steps["online_coppeliasim"] = [ordered]@{
        status = $(if ($Skipped -eq 0 -and $Failures -eq 0 -and $Errors -eq 0) { "PASS" } else { "FAIL" })
        tests = $Tests
        skipped = $Skipped
        failures = $Failures
        errors = $Errors
        junit = $Path
    }
    if ($Tests -lt 1 -or $Skipped -ne 0 -or $Failures -ne 0 -or $Errors -ne 0) {
        throw "V1-09 online acceptance requires tests>0 and zero skipped/failures/errors; tests=$Tests skipped=$Skipped failures=$Failures errors=$Errors"
    }
}

function Confirm-RuntimeEvidence {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "V1-09 runtime evidence root is missing: $Path"
    }
    $Runs = @(Get-ChildItem -LiteralPath $Path -Directory)
    if ($Runs.Count -ne 1) {
        throw "V1-09 runtime evidence requires exactly one run directory; found $($Runs.Count): $Path"
    }
    $Run = $Runs[0]
    $RequiredFiles = @(
        "summary.json",
        "scene-initial.json",
        "scene-final.json",
        "v1-09-defect-final-evidence.json",
        "commands.jsonl",
        "events.jsonl",
        "frames\frame-000001.png"
    )
    foreach ($RelativePath in $RequiredFiles) {
        $RequiredPath = Join-Path $Run.FullName $RelativePath
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
            throw "V1-09 runtime evidence is missing required file: $RequiredPath"
        }
    }
    $Bundles = @(Get-ChildItem -LiteralPath $Run.FullName -Recurse -File -Filter "vision-bundle-*.json")
    if ($Bundles.Count -ne 1) {
        throw "V1-09 runtime evidence requires exactly one vision bundle; found $($Bundles.Count)"
    }
    $Probes = @(Get-ChildItem -LiteralPath $Run.FullName -Recurse -File -Filter "defect-entry-*.json")
    if ($Probes.Count -ne 12) {
        throw "V1-09 runtime evidence requires exactly twelve entry probes; found $($Probes.Count)"
    }
    $Pngs = @(Get-ChildItem -LiteralPath $Run.FullName -Recurse -File -Filter "*.png")
    if ($Pngs.Count -lt 15) {
        throw "V1-09 runtime evidence requires at least fifteen PNG files; found $($Pngs.Count)"
    }
    $FileCount = @(Get-ChildItem -LiteralPath $Run.FullName -Recurse -File).Count
    $Steps["runtime_evidence"] = [ordered]@{
        status = "PASS"
        root = $Path
        run_id = $Run.Name
        file_count = $FileCount
    }
}

try {
    Push-Location $ProjectRoot
    try {
        [Environment]::SetEnvironmentVariable(
            $RuntimeEvidenceEnvName,
            $RuntimeEvidenceDir,
            "Process"
        )
        $Launch = & (Join-Path $PSScriptRoot "launch_coppeliasim.ps1") `
            -CoppeliaRoot $CoppeliaRoot `
            -Scene $Scene `
            -HostAddress $HostAddress `
            -Port 23010 `
            -Hidden
        if (-not $Launch -or $null -eq $Launch.ProcessId) {
            throw "launch_coppeliasim.ps1 did not return an owned process identity"
        }
        if ($Launch.StartedByScript) {
            $OwnedProcessId = [int]$Launch.ProcessId
            $OwnedProcessPath = [string]$Launch.ProcessPath
            $OwnedProcessStartTimeUtcTicks = [long]$Launch.ProcessStartTimeUtcTicks
        }
        $Steps["launch"] = [ordered]@{
            status = "PASS"
            port = 23010
            scene = $Scene
            started_by_script = [bool]$Launch.StartedByScript
            process_id = [int]$Launch.ProcessId
            process_path = [string]$Launch.ProcessPath
            process_start_time_utc_ticks = [long]$Launch.ProcessStartTimeUtcTicks
        }

        $env:COPPELIA_HOST = $HostAddress
        $env:COPPELIA_PORT = "23010"
        $env:PYTHONUTF8 = "1"
        $env:PYTHONIOENCODING = "utf-8"
        $Started = Get-Date
        & $Python -m pytest tests/test_acceptance/test_coppeliasim_v1_09.py `
            -m coppeliasim `
            --coppelia-host $HostAddress `
            --coppelia-port 23010 `
            --junitxml $JUnitPath `
            -q
        $ExitCode = $LASTEXITCODE
        $Steps["online_command"] = [ordered]@{
            status = $(if ($ExitCode -eq 0) { "PASS" } else { "FAIL" })
            exit_code = $ExitCode
            elapsed_seconds = [math]::Round(((Get-Date) - $Started).TotalSeconds, 3)
            command = "python -m pytest tests/test_acceptance/test_coppeliasim_v1_09.py -m coppeliasim --coppelia-host $HostAddress --coppelia-port 23010 --junitxml $JUnitPath -q"
        }
        if ($ExitCode -ne 0) {
            throw "V1-09 online acceptance failed with exit code $ExitCode"
        }
        Read-JUnitCounts -Path $JUnitPath
        Confirm-RuntimeEvidence -Path $RuntimeEvidenceDir
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
            $Cleanup = "Owned CoppeliaSim cleanup failed: $($_.Exception.Message)"
            $FailureMessage = if ($FailureMessage) { "$FailureMessage`n$Cleanup" } else { $Cleanup }
        }
    }
    try {
        [Environment]::SetEnvironmentVariable(
            $RuntimeEvidenceEnvName,
            $PreviousRuntimeEvidenceDir,
            "Process"
        )
    } catch {
        $Cleanup = "Runtime evidence environment cleanup failed: $($_.Exception.Message)"
        $FailureMessage = if ($FailureMessage) { "$FailureMessage`n$Cleanup" } else { $Cleanup }
    }
    $Summary = [ordered]@{
        schema_version = 1
        status = $(if ($FailureMessage) { "FAIL" } else { "PASS" })
        port = 23010
        scene = $Scene
        steps = $Steps
        failure = $FailureMessage
        human_acceptance = "PENDING_HUMAN_ACCEPTANCE"
        hardware_status = "PENDING_HARDWARE"
    }
    $Summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
}

Write-Host "V1-09 online acceptance summary: $SummaryPath"
if ($FailureMessage) {
    throw $FailureMessage
}
