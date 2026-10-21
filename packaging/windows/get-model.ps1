# Tomo: download a model for Ollama to think with. It runs on this computer.
#
# Tomo works with Ollama or LM Studio. This fetches a model for Ollama —
# Gemma 4 12B QAT by default: good at the tool use Tomo relies on, fluent in
# English and Bulgarian, and it sees images (the ones you send, and the
# characters, to choose their voices). It needs Ollama 0.30.5 or newer. Without
# Ollama it opens the download page instead.
#
# For a Gemma 4 model it also gives Ollama two settings (for this user):
# without them Ollama keeps saving Gemma's state between replies and can stall
# for minutes. LLAMA_ARG_CACHE_RAM=0 turns that cache off and
# LLAMA_ARG_CTX_CHECKPOINTS=4 keeps few checkpoints. Ollama reads them when it
# starts, so it has to be restarted once.

param([string]$Model = "gemma4:12b-it-qat")

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

if ($Model -like "gemma4*") {
    [Environment]::SetEnvironmentVariable("LLAMA_ARG_CACHE_RAM", "0", "User")
    [Environment]::SetEnvironmentVariable("LLAMA_ARG_CTX_CHECKPOINTS", "4", "User")
    $env:LLAMA_ARG_CACHE_RAM = "0"
    $env:LLAMA_ARG_CTX_CHECKPOINTS = "4"
    Write-Host "Set for Ollama: LLAMA_ARG_CACHE_RAM=0 and LLAMA_ARG_CTX_CHECKPOINTS=4 (so Gemma doesn't stall)." -ForegroundColor Cyan
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
    Write-Host "(An older Ollama can't run Gemma 4: update Ollama first.)" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
} elseif ($Model -like "gemma4*") {
    Write-Host "Done. Quit Ollama (its icon by the clock -> Quit) and start it again, so it uses the new settings." -ForegroundColor Green
    Read-Host "Press Enter to close"
}
