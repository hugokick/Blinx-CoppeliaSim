[CmdletBinding()]
param(
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) {
        $env:COPPELIASIM_ROOT
    } else {
        "E:\CoppeliaSim"
    }),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23008,
    [int]$TimeoutSeconds = 30,
    [switch]$Hidden
)

$ErrorActionPreference = "Stop"
if ($Port -ne 23008) {
    throw "V1-08 scene builder requires dedicated port 23008."
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Launch = Join-Path $PSScriptRoot "launch_coppeliasim.ps1"
. (Join-Path $PSScriptRoot "process_ownership.ps1")
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
$Scene = Join-Path $ProjectRoot "simulation\vision_lab\BL23_vision_lab.ttt"
$Builder = Join-Path $PSScriptRoot "build_v1_08_scene.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}

$Owned = $null
$ProcessId = $null
$ProcessPath = $null
$ProcessStartTimeUtcTicks = $null
try {
    $LaunchArguments = @{
        CoppeliaRoot = $CoppeliaRoot
        Scene = $Scene
        HostAddress = $HostAddress
        Port = 23008
        TimeoutSeconds = $TimeoutSeconds
    }
    if ($Hidden) {
        $LaunchArguments.Hidden = $true
    }
    $Owned = & $Launch @LaunchArguments
    if ($null -eq $Owned) {
        throw "CoppeliaSim launch wrapper returned no process identity."
    }
    $ProcessId = [int]$Owned.ProcessId
    $ProcessPath = [string]$Owned.ProcessPath
    $ProcessStartTimeUtcTicks = [long]$Owned.ProcessStartTimeUtcTicks
    Push-Location -LiteralPath $ProjectRoot
    try {
        & $Python $Builder
        if ($LASTEXITCODE -ne 0) {
            throw "V1-08 Python scene builder failed with exit code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
} finally {
    if ($null -ne $Owned -and [bool]$Owned.StartedByScript) {
        Stop-ExactOwnedProcess `
            -ProcessId $ProcessId `
            -ProcessPath $ProcessPath `
            -ProcessStartTimeUtcTicks $ProcessStartTimeUtcTicks
    }
}
