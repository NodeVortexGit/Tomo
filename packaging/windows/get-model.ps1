# Tomo: get Ollama and the models Tomo thinks with. They run on this computer.
#
# The installer runs this, so nothing is left to do by hand:
#   1. Ollama: installed (for this user, no admin needed) if it isn't there,
#      or updated if it's older than Tomo needs — silently, from ollama.com;
#   2. Ollama's two settings for Gemma 4: without them Ollama keeps saving
#      Gemma's state between replies and can stall for minutes
#      (LLAMA_ARG_CACHE_RAM=0 turns that cache off, LLAMA_ARG_CTX_CHECKPOINTS=4
#      keeps few checkpoints);
#   3. Ollama restarted, since it reads those settings when it starts;
#   4. the models: Gemma 4 12B QAT, Tomo's brain (tools, English and
#      Bulgarian, it sees images), and BgGPT, which checks Tomo's Bulgarian.
#
# Run it again any time (Start menu -> Tomo -> Download Tomo's models):
# what's there already is kept, so it only finishes what's missing.

# -NoModels: Ollama and its settings only (the build's test uses it).
param([string]$Model = "gemma4:12b-it-qat",
      [string]$Proofreader = "hf.co/INSAIT-Institute/BgGPT-Gemma-3-4B-IT-GGUF:Q4_K_M",
      [switch]$NoModels)

# Not "Stop": Windows PowerShell 5.1 would treat anything ollama.exe writes to
# its error stream (as it does while its server starts) as a failure. The
# steps that must succeed say so themselves (-ErrorAction Stop, throw).
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is many times faster without its progress bar
$MinimumOllama = [version]"0.30.5"
$OllamaSetupUrl = "https://ollama.com/download/OllamaSetup.exe"
$OllamaDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"

function Find-Ollama {
    $found = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $found) {
        $candidate = Join-Path $OllamaDir "ollama.exe"
        if (Test-Path $candidate) { $found = $candidate }
    }
    return $found
}

function Get-OllamaVersion([string]$exe) {
    # "ollama version is 0.40.17" (with a warning first when no server runs)
    $text = (& $exe --version 2>&1 | Out-String)
    if ($text -match "(\d+)\.(\d+)\.(\d+)") { return [version]"$($Matches[1]).$($Matches[2]).$($Matches[3])" }
    return $null
}

function Install-Ollama {
    $setup = Join-Path $env:TEMP "OllamaSetup.exe"
    Write-Host "Downloading Ollama, which runs Tomo's models (about 1 GB, once)..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $OllamaSetupUrl -OutFile $setup -UseBasicParsing -ErrorAction Stop
    Write-Host "Installing Ollama..." -ForegroundColor Cyan
    # Waits for the installer only: -Wait would also wait for what it starts
    # (Ollama's tray app, which keeps running).
    $run = Start-Process -FilePath $setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-" -PassThru -ErrorAction Stop
    $run.WaitForExit()
    Remove-Item $setup -ErrorAction SilentlyContinue
    if ($run.ExitCode -ne 0) { throw "Ollama's installer stopped with code $($run.ExitCode)" }
}

function Restart-Ollama([string]$exe) {
    # Ollama's tray app runs the server; both read the settings when they start.
    Get-Process -Name "ollama app", "ollama", "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    $app = Join-Path (Split-Path -Parent $exe) "ollama app.exe"
    if (Test-Path $app) { Start-Process -FilePath $app -ErrorAction Stop }
    else { Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden -ErrorAction Stop }
    if (Wait-Ollama $exe 15) { return }
    # The tray app didn't bring the server up: start the server itself.
    Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden -ErrorAction Stop
    if (Wait-Ollama $exe 15) { return }
    throw "Ollama didn't start"
}

function Wait-Ollama([string]$exe, [int]$tries) {
    foreach ($i in 1..$tries) {
        & $exe list *> $null
        if ($LASTEXITCODE -eq 0) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Get-Model([string]$exe, [string]$name, [string]$what) {
    Write-Host "Downloading $what ($name)..." -ForegroundColor Cyan
    & $exe pull $name | Out-Host  # shown, not returned with the answer
    return $LASTEXITCODE -eq 0
}

Write-Host "Setting up Ollama and the models Tomo thinks with. Everything runs on this computer." -ForegroundColor Cyan
try {
    $ollama = Find-Ollama
    $version = if ($ollama) { Get-OllamaVersion $ollama } else { $null }
    if (-not $ollama -or ($version -and $version -lt $MinimumOllama)) {
        if ($ollama) { Write-Host "Ollama $version is too old for Gemma 4; updating it." -ForegroundColor Cyan }
        Install-Ollama
        $ollama = Find-Ollama
        if (-not $ollama) { throw "Ollama was installed, but ollama.exe isn't in $OllamaDir" }
    }

    if ($Model -like "gemma4*") {
        [Environment]::SetEnvironmentVariable("LLAMA_ARG_CACHE_RAM", "0", "User")
        [Environment]::SetEnvironmentVariable("LLAMA_ARG_CTX_CHECKPOINTS", "4", "User")
        $env:LLAMA_ARG_CACHE_RAM = "0"
        $env:LLAMA_ARG_CTX_CHECKPOINTS = "4"
        Write-Host "Set for Ollama: LLAMA_ARG_CACHE_RAM=0 and LLAMA_ARG_CTX_CHECKPOINTS=4 (so Gemma doesn't stall)."
    }
    Restart-Ollama $ollama
    Write-Host "Ollama $(Get-OllamaVersion $ollama) is running." -ForegroundColor Green
    if ($NoModels) { exit 0 }

    if (-not (Get-Model $ollama $Model "Tomo's brain, about 7.2 GB, once")) {
        throw "$Model didn't download"
    }
    if (-not (Get-Model $ollama $Proofreader "the model that checks Tomo's Bulgarian, about 2.5 GB, once")) {
        Write-Host "The proofreading model didn't download; Tomo still checks its Bulgarian with rules and a dictionary." -ForegroundColor Yellow
    }
    Write-Host "Ollama and Tomo's models are ready." -ForegroundColor Green
    Start-Sleep -Seconds 3
    exit 0
}
catch {
    Write-Host "This step didn't finish: $_" -ForegroundColor Yellow
    Write-Host "Check the internet connection and the free disk space (about 12 GB), then try again:"
    Write-Host "Start menu -> Tomo -> Download Tomo's models."
    if ([Environment]::UserInteractive -and -not $NoModels) { Read-Host "Press Enter to close" }
    exit 1
}
