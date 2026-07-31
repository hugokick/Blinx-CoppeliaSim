[CmdletBinding()]
param(
    [ValidateSet("sim", "replay", "hik")]
    [string]$Camera = "sim",
    [ValidateSet("sim", "real")]
    [string]$Robot = "sim",
    [string]$Config,
    [string]$OutputDir = "artifacts\vision_lab\pyqt-session",
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) {
        $env:COPPELIASIM_ROOT
    } else {
        "E:\CoppeliaSim"
    }),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23000,
    [switch]$HiddenSimulator
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. (Join-Path $PSScriptRoot "process_ownership.ps1")
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw (
        "Python environment is missing: $Python. " +
        "Create it and install requirements-vision.txt first."
    )
}

if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Scene = Join-Path `
    $ProjectRoot `
    "simulation\vision_lab\BL23_vision_lab.ttt"
$OwnedProcessId = $null
$OwnedProcessPath = $null
$OwnedProcessStartTimeUtcTicks = $null
$AppExitCode = 1
try {
    if ($Camera -eq "sim" -or $Robot -eq "sim") {
        $LaunchArguments = @{
            CoppeliaRoot = $CoppeliaRoot
            HostAddress = $HostAddress
            Port = $Port
            Scene = $Scene
        }
        if ($HiddenSimulator) {
            $LaunchArguments.Hidden = $true
        }
        $Launch = & (
            Join-Path $PSScriptRoot "launch_coppeliasim.ps1"
        ) @LaunchArguments
        $Launch | Format-Table -AutoSize | Out-Host
        if ($Launch.StartedByScript) {
            $OwnedProcessId = [int]$Launch.ProcessId
            $OwnedProcessPath = [string]$Launch.ProcessPath
            $OwnedProcessStartTimeUtcTicks = `
                [long]$Launch.ProcessStartTimeUtcTicks
        }
    }

    $env:COPPELIA_HOST = $HostAddress
    $env:COPPELIA_PORT = [string]$Port
    $env:COPPELIA_SCENE = $Scene

    $Arguments = @(
        "-m"
        "vision_platform.ui.pyqt_app"
        "--camera"
        $Camera
        "--robot"
        $Robot
        "--output"
        $OutputDir
    )
    if ($Config) {
        $Arguments += @("--config", $Config)
    }

    Push-Location $ProjectRoot
    try {
        & $Python @Arguments
        $AppExitCode = $LASTEXITCODE
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
exit $AppExitCode
