[CmdletBinding()]
param(
    [string]$Program,
    [string]$HostName = '127.0.0.1',
    [int]$Port = 23007,
    [string]$Output = 'artifacts/vision_lab/experiment-runs'
)

$ErrorActionPreference = 'Stop'
$General = Join-Path $PSScriptRoot 'run_experiment.ps1'
$Arguments = @(
    '-Experiment', 'V1-07',
    '-HostName', $HostName,
    '-Port', [string]$Port,
    '-Output', $Output
)
if ($Program) {
    $Arguments += @('-Program', $Program)
}
& powershell.exe -ExecutionPolicy Bypass -File $General @Arguments
exit $LASTEXITCODE
