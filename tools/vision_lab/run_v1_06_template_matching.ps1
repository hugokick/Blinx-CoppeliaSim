[CmdletBinding()]
param(
    [string]$Program,
    [string]$HostName = '127.0.0.1',
    [int]$Port = 23005,
    [string]$Output = 'artifacts/vision_lab/experiment-runs'
)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$General = Join-Path $PSScriptRoot 'run_experiment.ps1'
$Arguments = @(
    '-Experiment', 'V1-06',
    '-HostName', $HostName,
    '-Port', [string]$Port,
    '-Output', $Output
)
if ($Program) {
    $Arguments += @('-Program', $Program)
}
& powershell.exe -ExecutionPolicy Bypass -File $General @Arguments
exit $LASTEXITCODE
