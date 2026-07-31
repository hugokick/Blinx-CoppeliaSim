function Stop-StartedProcessObject {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $StartedProcess,
        [int]$TimeoutMilliseconds = 10000,
        [scriptblock]$StopProcessAction = {
            param($VerifiedProcess)
            Stop-Process -InputObject $VerifiedProcess -Force `
                -ErrorAction Stop
        }
    )

    if ($TimeoutMilliseconds -lt 0) {
        throw "Started process timeout must not be negative."
    }
    if ($StartedProcess.HasExited) {
        return
    }

    & $StopProcessAction $StartedProcess
    if (-not [bool]$StartedProcess.WaitForExit($TimeoutMilliseconds)) {
        throw (
            "Started process object did not exit within " +
            "$TimeoutMilliseconds ms."
        )
    }
}

function Stop-ExactOwnedProcess {
    [CmdletBinding()]
    param(
        [Nullable[int]]$ProcessId,
        [Parameter(Mandatory = $true)]
        [string]$ProcessPath,
        [Parameter(Mandatory = $true)]
        [long]$ProcessStartTimeUtcTicks,
        [int]$TimeoutMilliseconds = 10000,
        [scriptblock]$GetProcessAction = {
            param([int]$RequestedId)
            Get-Process `
                -Id $RequestedId `
                -ErrorAction SilentlyContinue
        },
        [scriptblock]$StopProcessAction = {
            param($VerifiedProcess)
            Stop-Process -InputObject $VerifiedProcess -Force `
                -ErrorAction Stop
        }
    )

    if ($null -eq $ProcessId) {
        return
    }
    if ([string]::IsNullOrWhiteSpace($ProcessPath)) {
        throw "Owned process path is required for PID $ProcessId."
    }
    if ($ProcessStartTimeUtcTicks -le 0) {
        throw "Owned process start time is required for PID $ProcessId."
    }
    if ($TimeoutMilliseconds -lt 0) {
        throw "Owned process timeout must not be negative."
    }

    try {
        $ExpectedProcessPath = [System.IO.Path]::GetFullPath($ProcessPath)
    } catch {
        throw "Owned process path is invalid for PID $ProcessId."
    }

    $OwnedProcess = & $GetProcessAction ([int]$ProcessId)
    if ($null -eq $OwnedProcess) {
        return
    }
    try {
        $ActualProcessPath = [System.IO.Path]::GetFullPath(
            [string]$OwnedProcess.Path
        )
    } catch {
        throw "Owned process path could not be read for PID $ProcessId."
    }
    try {
        $ActualStartTimeUtcTicks = [long](
            $OwnedProcess.StartTime.ToUniversalTime().Ticks
        )
    } catch {
        throw "Owned process start time could not be read for PID $ProcessId."
    }
    if (
        [int]$OwnedProcess.Id -ne [int]$ProcessId -or
        $ActualProcessPath -ne $ExpectedProcessPath -or
        $ActualStartTimeUtcTicks -ne $ProcessStartTimeUtcTicks
    ) {
        throw (
            "Owned process identity mismatch. " +
            "Expected=PID:$ProcessId Path:$ExpectedProcessPath " +
            "StartTimeUtcTicks:$ProcessStartTimeUtcTicks; " +
            "Actual=PID:$($OwnedProcess.Id) Path:$ActualProcessPath " +
            "StartTimeUtcTicks:$ActualStartTimeUtcTicks"
        )
    }

    & $StopProcessAction $OwnedProcess
    $WaitCompleted = [bool]$OwnedProcess.WaitForExit(
        $TimeoutMilliseconds
    )
    $Survivor = & $GetProcessAction ([int]$ProcessId)
    $SameOwnedProcessSurvived = $false
    if ($null -ne $Survivor) {
        try {
            $SurvivorPath = [System.IO.Path]::GetFullPath(
                [string]$Survivor.Path
            )
        } catch {
            throw (
                "Surviving process path could not be read for PID " +
                "$ProcessId."
            )
        }
        try {
            $SurvivorStartTimeUtcTicks = [long](
                $Survivor.StartTime.ToUniversalTime().Ticks
            )
        } catch {
            throw (
                "Surviving process start time could not be read for PID " +
                "$ProcessId."
            )
        }
        $SameOwnedProcessSurvived = (
            [int]$Survivor.Id -eq [int]$ProcessId -and
            $SurvivorPath -eq $ExpectedProcessPath -and
            $SurvivorStartTimeUtcTicks -eq $ProcessStartTimeUtcTicks
        )
    }
    if (-not $WaitCompleted -or $SameOwnedProcessSurvived) {
        throw (
            "Owned process PID $ProcessId did not exit within " +
            "$TimeoutMilliseconds ms: $ExpectedProcessPath"
        )
    }
}
