[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Program,
    [string]$OutputDir = "artifacts\vision_lab\student-runs",
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
. (Join-Path $PSScriptRoot "process_ownership.ps1")
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}

if (-not [System.IO.Path]::IsPathRooted($Program)) {
    $Program = Join-Path $ProjectRoot $Program
}
if (-not (Test-Path -LiteralPath $Program -PathType Leaf)) {
    throw "Student program was not found: $Program"
}
$Program = (Resolve-Path -LiteralPath $Program).Path
if ([System.IO.Path]::GetExtension($Program) -ne ".py") {
    throw "Student program must be a .py file: $Program"
}

if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Scene = Join-Path $ProjectRoot "simulation\vision_lab\BL23_vision_lab.ttt"
$OwnedProcessId = $null
$OwnedProcessPath = $null
$OwnedProcessStartTimeUtcTicks = $null
$StudentExitCode = 1
try {
    $Launch = & (Join-Path $PSScriptRoot "launch_coppeliasim.ps1") `
        -CoppeliaRoot $CoppeliaRoot `
        -Scene $Scene `
        -HostAddress $HostAddress `
        -Port $Port
    $Launch | Format-Table -AutoSize | Out-Host
    if ($Launch.StartedByScript) {
        $OwnedProcessId = [int]$Launch.ProcessId
        $OwnedProcessPath = [string]$Launch.ProcessPath
        $OwnedProcessStartTimeUtcTicks = `
            [long]$Launch.ProcessStartTimeUtcTicks
    }

    $env:COPPELIA_HOST = $HostAddress
    $env:COPPELIA_PORT = [string]$Port
    Push-Location -LiteralPath $ProjectRoot
    try {
        & $Python -m vision_platform.cli student-run `
            --program $Program `
            --robot sim `
            --scene $Scene `
            --host $HostAddress `
            --port $Port `
            --output $OutputDir
        $StudentExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    if ($OwnedProcessId) {
        Stop-ExactOwnedProcess `
            -ProcessId $OwnedProcessId `
            -ProcessPath $OwnedProcessPath `
            -ProcessStartTimeUtcTicks $OwnedProcessStartTimeUtcTicks
    }
}
exit $StudentExitCode
