function Get-RgbdProcessIdentity {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Process
    )

    try {
        $path = [System.IO.Path]::GetFullPath([string]$Process.Path)
        $startTimeUtcTicks = [long]$Process.StartTime.ToUniversalTime().Ticks
        $processId = [int]$Process.Id
    } catch {
        throw (
            "Could not read RGB-D process identity for PID " +
            "$($Process.Id): $($_.Exception.Message)"
        )
    }
    if ($processId -le 0 -or [string]::IsNullOrWhiteSpace($path) -or $startTimeUtcTicks -le 0) {
        throw "RGB-D process identity is incomplete for PID $processId"
    }
    [pscustomobject]@{
        ProcessId = $processId
        ProcessPath = $path
        ProcessStartTimeUtcTicks = $startTimeUtcTicks
    }
}

function Assert-RgbdOwnedListener {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $ExpectedIdentity,
        [Parameter(Mandatory = $true)]
        [scriptblock]$GetListenerAction,
        [Parameter(Mandatory = $true)]
        [scriptblock]$GetProcessAction
    )

    $listener = & $GetListenerAction
    if ($null -eq $listener) {
        throw "RGB-D listener identity mismatch: the dedicated listener is missing"
    }
    try {
        $listenerPid = [int]$listener.OwningProcess
    } catch {
        throw "RGB-D listener identity mismatch: listener PID is invalid"
    }
    if ($listenerPid -ne [int]$ExpectedIdentity.ProcessId) {
        throw (
            "RGB-D listener identity mismatch: expected PID " +
            "$($ExpectedIdentity.ProcessId), actual PID $listenerPid"
        )
    }
    $listenerProcess = & $GetProcessAction $listenerPid
    if ($null -eq $listenerProcess) {
        throw "RGB-D listener identity mismatch: owning process is missing"
    }
    $actualIdentity = Get-RgbdProcessIdentity -Process $listenerProcess
    if (
        $actualIdentity.ProcessId -ne [int]$ExpectedIdentity.ProcessId -or
        $actualIdentity.ProcessPath -ne [string]$ExpectedIdentity.ProcessPath -or
        $actualIdentity.ProcessStartTimeUtcTicks -ne
            [long]$ExpectedIdentity.ProcessStartTimeUtcTicks
    ) {
        throw (
            "RGB-D listener identity mismatch. " +
            "Expected=PID:$($ExpectedIdentity.ProcessId) " +
            "Path:$($ExpectedIdentity.ProcessPath) " +
            "StartTimeUtcTicks:$($ExpectedIdentity.ProcessStartTimeUtcTicks); " +
            "Actual=PID:$($actualIdentity.ProcessId) " +
            "Path:$($actualIdentity.ProcessPath) " +
            "StartTimeUtcTicks:$($actualIdentity.ProcessStartTimeUtcTicks)"
        )
    }
    return $actualIdentity
}
