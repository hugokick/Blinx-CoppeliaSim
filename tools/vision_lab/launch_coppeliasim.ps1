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
. (Join-Path $PSScriptRoot "process_ownership.ps1")

if (-not (Test-Path -LiteralPath $CoppeliaRoot -PathType Container)) {
    throw "CoppeliaSim root was not found: $CoppeliaRoot"
}
$CoppeliaRoot = (Resolve-Path -LiteralPath $CoppeliaRoot).Path

if (-not $Scene) {
    $Scene = Join-Path $ProjectRoot "simulation\vision_lab\BL23_vision_lab.ttt"
} elseif (-not [System.IO.Path]::IsPathRooted($Scene)) {
    $Scene = Join-Path $ProjectRoot $Scene
}
$Scene = (Resolve-Path -LiteralPath $Scene).Path

$Executable = Join-Path $CoppeliaRoot "coppeliaSim.exe"
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "CoppeliaSim executable was not found: $Executable"
}
$Executable = (Resolve-Path -LiteralPath $Executable).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}

function Get-PortListener {
    Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Get-LaunchProcessIdentity {
    param(
        [Parameter(Mandatory = $true)]
        $Process
    )

    try {
        $IdentityPath = [System.IO.Path]::GetFullPath(
            [string]$Process.Path
        )
        $IdentityStartTimeUtcTicks = [long](
            $Process.StartTime.ToUniversalTime().Ticks
        )
    } catch {
        throw (
            "Could not capture CoppeliaSim process identity for PID " +
            "$($Process.Id): $($_.Exception.Message)"
        )
    }
    [pscustomobject]@{
        ProcessId = [int]$Process.Id
        ProcessPath = $IdentityPath
        ProcessStartTimeUtcTicks = $IdentityStartTimeUtcTicks
    }
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

$StartedProcess = $null
$LaunchProcessId = $null
$LaunchProcessPath = $null
$LaunchProcessStartTimeUtcTicks = $null
try {
    $ExistingListener = Get-PortListener
    if ($ExistingListener) {
        $ListenerProcess = Get-Process `
            -Id $ExistingListener.OwningProcess `
            -ErrorAction Stop
        $LaunchIdentity = Get-LaunchProcessIdentity -Process $ListenerProcess
        if ($LaunchIdentity.ProcessPath -ne $Executable) {
            throw (
                "Port $Port is occupied by PID $($ListenerProcess.Id), " +
                "but it is not $Executable."
            )
        }
    } else {
        $StartedProcess = Start-Process @StartParameters
        $ListenerProcess = $StartedProcess
        $LaunchIdentity = Get-LaunchProcessIdentity -Process $ListenerProcess
    }
    $LaunchProcessId = [int]$LaunchIdentity.ProcessId
    $LaunchProcessPath = [string]$LaunchIdentity.ProcessPath
    $LaunchProcessStartTimeUtcTicks = [long](
        $LaunchIdentity.ProcessStartTimeUtcTicks
    )

    Push-Location -LiteralPath $ProjectRoot
    try {
        $ReadinessOutput = & $Python -m vision_platform.coppeliasim_readiness `
            --host $HostAddress `
            --port $Port `
            --scene $Scene `
            --timeout $TimeoutSeconds
        $ReadyExitCode = $LASTEXITCODE
        $ReadinessOutput | Out-Host
    } finally {
        Pop-Location
    }
    if ($ReadyExitCode -ne 0) {
        $ExitDetail = ""
        if ($null -ne $StartedProcess -and $StartedProcess.HasExited) {
            $ExitDetail = " ExitCode=$($StartedProcess.ExitCode)."
        }
        throw (
            "CoppeliaSim readiness failed for $HostAddress`:$Port." +
            "$ExitDetail Scene=$Scene"
        )
    }

    $ReadyListener = Get-PortListener
    if (
        -not $ReadyListener -or
        $ReadyListener.OwningProcess -ne $LaunchProcessId
    ) {
        throw (
            "CoppeliaSim readiness succeeded on an unexpected listener. " +
            "ExpectedPID=$LaunchProcessId, Port=$Port"
        )
    }
    $ReadyProcess = Get-Process `
        -Id $ReadyListener.OwningProcess `
        -ErrorAction Stop
    $ReadyIdentity = Get-LaunchProcessIdentity -Process $ReadyProcess
    if (
        $ReadyIdentity.ProcessId -ne $LaunchProcessId -or
        $ReadyIdentity.ProcessPath -ne $LaunchProcessPath -or
        $ReadyIdentity.ProcessStartTimeUtcTicks -ne
            $LaunchProcessStartTimeUtcTicks
    ) {
        throw (
            "CoppeliaSim readiness succeeded on a replacement process. " +
            "ExpectedPID=$LaunchProcessId, Port=$Port"
        )
    }

    [pscustomobject]@{
        Status = $(if ($StartedProcess) { "STARTED" } else { "REUSED" })
        ProcessId = $LaunchProcessId
        ProcessPath = $LaunchProcessPath
        ProcessStartTimeUtcTicks = $LaunchProcessStartTimeUtcTicks
        Port = $Port
        Scene = $Scene
        StartedByScript = [bool]$StartedProcess
    }
} catch {
    $LaunchFailure = $_
    if ($null -ne $StartedProcess) {
        try {
            $HasCapturedLaunchIdentity = (
                $null -ne $LaunchProcessId -and
                -not [string]::IsNullOrWhiteSpace($LaunchProcessPath) -and
                $null -ne $LaunchProcessStartTimeUtcTicks
            )
            if ($HasCapturedLaunchIdentity) {
                Stop-ExactOwnedProcess `
                    -ProcessId $LaunchProcessId `
                    -ProcessPath $LaunchProcessPath `
                    -ProcessStartTimeUtcTicks $LaunchProcessStartTimeUtcTicks
            } else {
                Stop-StartedProcessObject `
                    -StartedProcess $StartedProcess
            }
        } catch {
            throw (
                "$($LaunchFailure.Exception.Message) " +
                "Owned CoppeliaSim cleanup failed: " +
                "$($_.Exception.Message)"
            )
        }
    }
    throw $LaunchFailure
}
