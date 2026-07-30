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

if ($Camera -eq "sim" -or $Robot -eq "sim") {
    $LaunchArguments = @{
        CoppeliaRoot = $CoppeliaRoot
        HostAddress = $HostAddress
        Port = $Port
    }
    if ($HiddenSimulator) {
        $LaunchArguments.Hidden = $true
    }
    & (Join-Path $PSScriptRoot "launch_coppeliasim.ps1") @LaunchArguments |
        Format-Table -AutoSize |
        Out-Host
}

$env:COPPELIA_HOST = $HostAddress
$env:COPPELIA_PORT = [string]$Port
$env:COPPELIA_SCENE = Join-Path `
    $ProjectRoot `
    "simulation\vision_lab\BL23_vision_lab.ttt"

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
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
