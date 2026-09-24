# Tomo: set up the voice and "Hey Tomo".
#
# Creates a private Python (with uv) holding the speech helpers — Piper for
# Tomo's voice, Vosk and Whisper for what you say — and downloads their
# models. They all run on this computer; this only needs the internet once
# (about 1 GB). Run it again any time to repair or finish the setup.

$ErrorActionPreference = "Stop"
$tools = $PSScriptRoot
$app = Split-Path -Parent $tools
$scripts = Join-Path $app "scripts"
$venv = Join-Path $scripts ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$uv = Join-Path $tools "uv.exe"

Write-Host "Setting up Tomo's voice. Speech runs on this computer; this downloads about 1 GB, once." -ForegroundColor Cyan
try {
    if (-not (Test-Path $python)) {
        & $uv venv --python 3.12 $venv
        if ($LASTEXITCODE -ne 0) { throw "couldn't create the Python environment" }
    }
    & $uv pip install --python $python -r (Join-Path $scripts "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "couldn't install the speech helpers" }
    & $python (Join-Path $scripts "setup_models.py")
    if ($LASTEXITCODE -ne 0) { throw "some speech models didn't download" }
    Write-Host "Tomo's voice is ready." -ForegroundColor Green
    Start-Sleep -Seconds 3
}
catch {
    Write-Host "Setting up the voice didn't finish: $_" -ForegroundColor Yellow
    Write-Host "Tomo still works by text. To try again: Start menu -> Tomo -> Set up Tomo's voice."
    Read-Host "Press Enter to close"
}
