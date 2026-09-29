# Tomo: set up Tomo's Python, and (with -Models) the voice, "Hey Tomo" and
# the health programme's camera.
#
# Makes Tomo's own Python environment (.venv in Tomo's folder, with uv),
# installs what Tomo needs into it, and with -Models downloads the speech
# models (Piper's voices, Vosk and Parakeet) and the pose model. Everything
# runs on this computer; this only needs the internet once. Run it again any
# time to repair or finish the setup.

param([switch]$Models)

$ErrorActionPreference = "Stop"
$tools = $PSScriptRoot
$app = Split-Path -Parent $tools
$venv = Join-Path $app ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$uv = Join-Path $tools "uv.exe"

Write-Host "Setting up Tomo. Everything runs on this computer; this downloads what it needs once." -ForegroundColor Cyan
try {
    if (-not (Test-Path $python)) {
        & $uv venv --python 3.12 $venv
        if ($LASTEXITCODE -ne 0) { throw "couldn't create the Python environment" }
    }
    & $uv pip install --python $python -r (Join-Path $app "pyproject.toml") --extra all
    if ($LASTEXITCODE -ne 0) { throw "couldn't install Tomo's Python packages" }
    if ($Models) {
        & $python (Join-Path $app "scripts\setup_models.py")
        if ($LASTEXITCODE -ne 0) { throw "some models didn't download" }
        Write-Host "Tomo's voice and camera are ready." -ForegroundColor Green
    }
    Write-Host "Tomo is set up." -ForegroundColor Green
    Start-Sleep -Seconds 3
}
catch {
    Write-Host "The setup didn't finish: $_" -ForegroundColor Yellow
    Write-Host "To try again: Start menu -> Tomo -> Set up Tomo's voice and camera."
    Read-Host "Press Enter to close"
}
