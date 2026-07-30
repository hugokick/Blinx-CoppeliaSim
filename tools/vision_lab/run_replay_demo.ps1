[CmdletBinding()]
param(
    [string]$Manifest = "config\replay_manifest.json",
    [string]$OutputDir = "artifacts\vision_lab\replay"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}
if (-not [System.IO.Path]::IsPathRooted($Manifest)) {
    $Manifest = Join-Path $ProjectRoot $Manifest
}
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$ManifestPayload = Get-Content `
    -LiteralPath $Manifest `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
$ManifestDirectory = Split-Path -Parent $Manifest
$Rows = @()
$Index = 0

Push-Location $ProjectRoot
try {
    foreach ($Frame in $ManifestPayload.frames) {
        $Index += 1
        $Image = Join-Path $ManifestDirectory $Frame.path
        $Image = (Resolve-Path -LiteralPath $Image).Path
        $Stem = "frame-{0:D2}-experiment-{1}" -f $Index, $Frame.experiment
        $DetectionJson = Join-Path $OutputDir "$Stem.json"
        $AnnotatedImage = Join-Path $OutputDir "$Stem-annotated.png"
        & $Python `
            -m vision_platform.cli recognize `
            --image $Image `
            --output $DetectionJson `
            --annotated $AnnotatedImage
        $ExitCode = $LASTEXITCODE
        if ($ExitCode -ne 0) {
            throw "Replay recognition failed for $Image (exit $ExitCode)"
        }
        $DetectionPayload = Get-Content `
            -LiteralPath $DetectionJson `
            -Raw `
            -Encoding UTF8 |
            ConvertFrom-Json
        $Rows += [pscustomobject]@{
            index = $Index
            source = $Image
            experiment = $Frame.experiment
            kind = $Frame.kind
            detection_count = @($DetectionPayload.detections).Count
            detection_json = $DetectionJson
            annotated_image = $AnnotatedImage
        }
    }
} finally {
    Pop-Location
}

$SummaryPath = Join-Path $OutputDir "replay-summary.json"
$Summary = [ordered]@{
    schema_version = 1
    status = "PASS"
    scope = "authentic-image recognition replay only"
    frames_processed = $Rows.Count
    results = $Rows
    hardware_validation = "PENDING_HARDWARE"
    note = (
        "Replay processing is not proof of Hikvision camera acquisition, " +
        "real TCP calibration, or physical grasp success."
    )
}
$Summary |
    ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath $SummaryPath -Encoding UTF8
Write-Host "Replay evidence: $SummaryPath"
