[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('R1-01', 'R1-02', 'R1-05', 'R1-06', 'R1-07')]
    [string]$Experiment,
    [string]$Program,
    [string]$HostName = '127.0.0.1',
    [int]$Port = 23000,
    [string]$Output = 'artifacts/vision_lab/experiment-runs'
)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PythonWrapper = Join-Path $Project 'tools\vision_lab\python.ps1'
$Arguments = @(
    '-m', 'vision_platform.cli',
    'experiment-run',
    '--experiment', $Experiment,
    '--host', $HostName,
    '--port', [string]$Port,
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
