[CmdletBinding()]
param(
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) { $env:COPPELIASIM_ROOT } else { "E:\CoppeliaSim" }),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23009,
    [int]$TimeoutSeconds = 30,
    [switch]$Hidden
)

$ErrorActionPreference = "Stop"
if ($Port -ne 23009) { throw "D1-01 is bound to port 23009" }
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
$Executable = (Resolve-Path (Join-Path $CoppeliaRoot "coppeliaSim.exe")).Path
$Scene = (Resolve-Path (Join-Path $ProjectRoot "simulation\vision_lab\BL23_vision_lab.ttt")).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python environment is missing: $Python" }
. (Join-Path $ProjectRoot "tools\rgbd_lab\listener_identity.ps1")
. (Join-Path $ProjectRoot "tools\vision_lab\process_ownership.ps1")

function Get-PortListener {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}

$Existing = Get-PortListener
if ($Existing) { throw "Dedicated D1-01 port 23009 is already occupied; refusing to clean it" }
$arguments = @("-GzmqRemoteApi.rpcPort=$Port", ('"' + $Scene + '"'))
$start = @{ FilePath = $Executable; ArgumentList = $arguments; WorkingDirectory = $CoppeliaRoot; PassThru = $true }
if ($Hidden) { $start.WindowStyle = "Hidden" }
$process = $null
$identity = $null
try {
    $process = Start-Process @start
    $identity = Get-RgbdProcessIdentity -Process $process
    Push-Location -LiteralPath $ProjectRoot
    try {
        & $Python -m vision_platform.coppeliasim_readiness --host $HostAddress --port $Port --scene $Scene --timeout $TimeoutSeconds
        if ($LASTEXITCODE -ne 0) { throw "CoppeliaSim readiness failed on 23009" }
        Assert-RgbdOwnedListener `
            -ExpectedIdentity $identity `
            -GetListenerAction { Get-PortListener } `
            -GetProcessAction {
                param($requestedId)
                Get-Process -Id $requestedId -ErrorAction Stop
            }
        & $Python -m tools.rgbd_lab.build_rgbd_scene --host $HostAddress --port $Port
        if ($LASTEXITCODE -ne 0) { throw "D1-01 scene build failed" }
    } finally {
        Pop-Location
    }
} finally {
    if ($null -ne $identity) {
        $current = Get-Process -Id $identity.ProcessId -ErrorAction SilentlyContinue
        if ($current) {
            $currentPath = [System.IO.Path]::GetFullPath([string]$current.Path)
            $currentTicks = [long]$current.StartTime.ToUniversalTime().Ticks
            if ($currentPath -ne $identity.ProcessPath -or $currentTicks -ne $identity.ProcessStartTimeUtcTicks) {
                throw "D1-01 cleanup identity mismatch; refusing to stop replacement process"
            }
            Stop-ExactOwnedProcess `
                -ProcessId $identity.ProcessId `
                -ProcessPath $identity.ProcessPath `
                -ProcessStartTimeUtcTicks $identity.ProcessStartTimeUtcTicks
        }
    }
}
