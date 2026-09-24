# Tomo: download a model for Ollama to think with. It runs on this computer.
#
# Tomo works with Ollama or LM Studio. This fetches a model for Ollama —
# qwen2.5:7b by default, which is good at the tool use Tomo relies on and fits
# an 8 GB graphics card. Without Ollama it opens the download page instead.

param([string]$Model = "qwen2.5:7b")

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $candidate) { $ollama = $candidate }
}
if (-not $ollama) {
    Write-Host "Tomo thinks with a model running on this computer, through Ollama or LM Studio." -ForegroundColor Cyan
    Write-Host "Ollama isn't installed yet, so its download page opens now. After installing it, run:"
    Write-Host "    ollama pull $Model" -ForegroundColor Green
    Write-Host "(Or use LM Studio: download a model there and start its local server.)"
    Start-Process "https://ollama.com/download"
    Read-Host "Press Enter to close"
    exit 0
}

# The Ollama app normally runs its server; start one if it isn't.
& $ollama list *> $null
if ($LASTEXITCODE -ne 0) {
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}
Write-Host "Downloading $Model for Ollama (a few GB, once)..." -ForegroundColor Cyan
& $ollama pull $Model
if ($LASTEXITCODE -ne 0) {
    Write-Host "The download didn't finish. To try again: Start menu -> Tomo -> Download a model for Tomo." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
}
