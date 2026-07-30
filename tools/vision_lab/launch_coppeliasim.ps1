[CmdletBinding()]
param(
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) {
        $env:COPPELIASIM_ROOT
    } else {
        "E:\CoppeliaSim"
    }),
    [string]$Scene,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23000,
    [int]$TimeoutSeconds = 30,
    [switch]$Hidden
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if (-not $Scene) {
    $Scene = Join-Path $ProjectRoot "simulation\vision_lab\BL23_vision_lab.ttt"
} elseif (-not [System.IO.Path]::IsPathRooted($Scene)) {
    $Scene = Join-Path $ProjectRoot $Scene
}
$Scene = (Resolve-Path -LiteralPath $Scene).Path

$Executable = Join-Path $CoppeliaRoot "coppeliaSim.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "CoppeliaSim executable was not found: $Executable"
}

function Get-PortListener {
    Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

$ExistingListener = Get-PortListener
if ($ExistingListener) {
    $ExistingProcess = Get-Process `
        -Id $ExistingListener.OwningProcess `
        -ErrorAction Stop
    if ($ExistingProcess.Path -ne $Executable) {
        throw (
            "Port $Port is occupied by PID $($ExistingProcess.Id), " +
            "but it is not $Executable."
        )
    }
    [pscustomobject]@{
        Status = "REUSED"
        ProcessId = $ExistingProcess.Id
        Port = $Port
        Scene = $Scene
        StartedByScript = $false
    }
    return
}

$QuotedScene = '"' + $Scene + '"'
$StartArguments = @(
    "-GzmqRemoteApi.rpcPort=$Port"
    $QuotedScene
)
$StartParameters = @{
    FilePath = $Executable
    ArgumentList = $StartArguments
    WorkingDirectory = $CoppeliaRoot
    PassThru = $true
}
if ($Hidden) {
    $StartParameters.WindowStyle = "Hidden"
}

$Process = Start-Process @StartParameters
$Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$Listener = $null
do {
    Start-Sleep -Milliseconds 250
    $Listener = Get-PortListener
    if ($Process.HasExited) {
        throw (
            "CoppeliaSim exited before opening port $Port. " +
            "ExitCode=$($Process.ExitCode), Scene=$Scene"
        )
    }
} while (
    (-not $Listener -or $Listener.OwningProcess -ne $Process.Id) -and
    (Get-Date) -lt $Deadline
)

if (-not $Listener -or $Listener.OwningProcess -ne $Process.Id) {
    $ProcessState = Get-Process -Id $Process.Id -ErrorAction SilentlyContinue
    $PortState = Get-NetTCPConnection `
        -LocalPort $Port `
        -ErrorAction SilentlyContinue
    throw (
        "CoppeliaSim did not open ZMQ port $Port within " +
        "$TimeoutSeconds seconds. PID=$($Process.Id); " +
        "Responding=$($ProcessState.Responding); " +
        "PortState=$($PortState.State); Scene=$Scene"
    )
}

[pscustomobject]@{
    Status = "STARTED"
    ProcessId = $Process.Id
    Port = $Port
    Scene = $Scene
    StartedByScript = $true
}
