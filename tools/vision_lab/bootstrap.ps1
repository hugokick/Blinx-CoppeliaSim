[CmdletBinding()]
param(
    [string]$CoppeliaSimRoot = $env:COPPELIASIM_ROOT,
    [switch]$SkipCad
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$venvPath = Join-Path $projectRoot '.venv-vision'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & $pyLauncher.Source -3.11 -m venv $venvPath
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 virtual environment creation failed with exit code $LASTEXITCODE."
        }
    }
    else {
        $systemPython = Get-Command python -ErrorAction Stop
        $version = & $systemPython.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version.Trim() -ne '3.11') {
            throw "Python 3.11 is required; found $version. Install Python 3.11 or the py launcher."
        }
        & $systemPython.Source -m venv $venvPath
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 virtual environment creation failed with exit code $LASTEXITCODE."
        }
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE."
}
& $venvPython -m pip install -r (Join-Path $projectRoot 'requirements-vision.txt')
if ($LASTEXITCODE -ne 0) {
    throw "Vision dependency installation failed with exit code $LASTEXITCODE."
}
if (-not $SkipCad) {
    & $venvPython -m pip install -r (Join-Path $projectRoot 'requirements-vision-cad.txt')
    if ($LASTEXITCODE -ne 0) {
        throw "CAD dependency installation failed with exit code $LASTEXITCODE."
    }
}

if ([string]::IsNullOrWhiteSpace($CoppeliaSimRoot)) {
    $CoppeliaSimRoot = 'E:\CoppeliaSim'
}
$CoppeliaSimRoot = (Resolve-Path -LiteralPath $CoppeliaSimRoot).Path
$zmqClientSource = Join-Path $CoppeliaSimRoot 'programming\zmqRemoteApi\clients\python\src'
if (-not (Test-Path -LiteralPath $zmqClientSource)) {
    throw "CoppeliaSim ZMQ Python client was not found under: $CoppeliaSimRoot"
}

$sitePackages = Join-Path $venvPath 'Lib\site-packages'
$pthPath = Join-Path $sitePackages 'coppeliasim-local.pth'
Set-Content -LiteralPath $pthPath -Value $zmqClientSource -Encoding ASCII

# Qt can lose a non-ASCII virtual-environment path while resolving its
# platform plugins on Windows. A .pth startup hook sets the path from
# Python's Unicode sys.prefix before PyQt loads any Qt libraries.
$qtPthPath = Join-Path $sitePackages 'vision-lab-qt.pth'
$qtPthValue = 'import os, sys; os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.join(sys.prefix, "Lib", "site-packages", "PyQt5", "Qt5", "plugins", "platforms"))'
Set-Content -LiteralPath $qtPthPath -Value $qtPthValue -Encoding ASCII

$env:COPPELIASIM_ROOT = $CoppeliaSimRoot
& $venvPython -c @'
import cv2
import numpy
import PyQt5
import zmq
import cbor2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print('VISION_ENV_OK')
print(f'python packages: numpy={numpy.__version__}, opencv={cv2.__version__}, pyzmq={zmq.__version__}')
'@
if ($LASTEXITCODE -ne 0) {
    throw "Vision environment import verification failed with exit code $LASTEXITCODE."
}
if (-not $SkipCad) {
    & $venvPython -c "import OCP, trimesh; print('VISION_CAD_ENV_OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "CAD environment import verification failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Vision environment ready: $venvPath"
Write-Host "CoppeliaSim root: $CoppeliaSimRoot"
