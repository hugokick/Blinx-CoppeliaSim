[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $projectRoot '.venv-vision\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Vision environment is missing. Run tools\vision_lab\bootstrap.ps1 first."
}

& $pythonExe @PythonArgs
exit $LASTEXITCODE
