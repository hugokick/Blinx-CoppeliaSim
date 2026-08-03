[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'R1-01', 'R1-02', 'R1-05', 'R1-06', 'R1-07',
        'V1-01', 'V1-02', 'V1-03', 'V1-04', 'V1-05', 'V1-06', 'V1-07', 'V1-08'
    )]
    [string]$Experiment,
    [string]$Program,
    [string]$HostName = '127.0.0.1',
    [int]$Port = 0,
    [string]$Output = 'artifacts/vision_lab/experiment-runs'
)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$SelectedPort = $Port
if ($Experiment -eq 'V1-08') {
    if ($Port -eq 0) {
        $SelectedPort = 23008
    } elseif ($Port -ne 23008) {
        throw 'V1-08 requires dedicated CoppeliaSim port 23008.'
    }
} elseif ($SelectedPort -eq 0) {
    $SelectedPort = 23000
}
$PythonWrapper = Join-Path $Project 'tools\vision_lab\python.ps1'
$Arguments = @(
    '-m', 'vision_platform.cli',
    'experiment-run',
    '--experiment', $Experiment,
    '--host', $HostName,
    '--port', [string]$SelectedPort,
    '--output', $Output
)
if ($Program) {
    $Arguments += @('--program', $Program)
}

$ExitCode = 1
Push-Location -LiteralPath $Project
try {
    & powershell.exe -ExecutionPolicy Bypass -File $PythonWrapper @Arguments
    $ExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $ExitCode
