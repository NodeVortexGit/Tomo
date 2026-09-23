//! The speech bridge: text-to-speech via Edge-TTS and speech-to-text via
//! Google Cloud STT.
//!
//! Both engines have mature, well-tested Python libraries and no equally good
//! pure-Rust equivalent, so — exactly as the brief allows ("a bit of bash /
//! Python for device control") — this module shells out to two small helper
//! scripts in `scripts/` that run inside the virtualenv `install.sh` creates:
//!
//!   scripts/tts_edge.py    text  → an mp3 file   (Microsoft Edge neural TTS)
//!   scripts/stt_google.py  wav   → a transcript  (Google Cloud Speech-to-Text)
//!
//! Audio capture uses the standard ALSA/PulseAudio recorders (`parecord` /
//! `arecord`) and playback tries the common players in turn, so there is no
//! hard dependency on any one desktop's audio stack.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use tokio::process::Command;
use tokio::time::timeout;

use crate::config::Config;

#[derive(Clone)]
pub struct Speech {
    python: PathBuf,
    scripts_dir: PathBuf,
    cache_dir: PathBuf,
    voice: String,
    rate: String,
    stt_key: String,
    stt_lang: String,
}

impl Speech {
    pub fn new(cfg: &Config) -> Self {
        let cache_dir = cfg.data_dir.join("cache");
        std::fs::create_dir_all(&cache_dir).ok();
        Self {
            python: resolve_python(&cfg.scripts_dir, &cfg.data_dir),
            scripts_dir: cfg.scripts_dir.clone(),
            cache_dir,
            voice: cfg.edge_tts_voice.clone(),
            rate: cfg.edge_tts_rate.clone(),
            stt_key: cfg.google_stt_api_key.clone(),
            stt_lang: cfg.stt_language.clone(),
        }
    }

    /// Synthesize `text` with Edge-TTS into an mp3 for [`play`]. Separate from
    /// playback so the body can move its mouth exactly while the voice plays.
    pub async fn synthesize(&self, text: &str) -> Result<PathBuf> {
        let out = self.cache_dir.join("last_tts.mp3");
        let script = self.scripts_dir.join("tts_edge.py");

        let status = timeout(
            Duration::from_secs(30),
            Command::new(&self.python)
                .arg(&script)
                .arg("--voice")
                .arg(&self.voice)
                .arg("--rate")
                .arg(&self.rate)
                .arg("--out")
                .arg(&out)
                .arg("--text")
                .arg(text)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .status(),
        )
        .await
        .context("edge-tts timed out")?
        .context("failed to launch edge-tts helper")?;

        if !status.success() {
            return Err(anyhow!("edge-tts helper exited with {status}"));
        }
        Ok(out)
    }

    /// Play synthesized speech. Returns once playback ends.
    pub async fn play(&self, audio: &std::path::Path) -> Result<()> {
        play_audio(audio).await
    }

    /// Whether STT is usable (needs a Google API key).
    pub fn can_listen(&self) -> bool {
        !self.stt_key.is_empty()
    }

    /// Record `seconds` of microphone audio and transcribe it with Google STT.
    /// Returns the recognised text (may be empty if nothing was heard).
    pub async fn listen(&self, seconds: u32) -> Result<String> {
        if !self.can_listen() {
            return Err(anyhow!("Google STT is not configured (no GOOGLE_STT_API_KEY)"));
        }
        let wav = self.cache_dir.join("mic.wav");
        record_wav(&wav, seconds).await?;

        let script = self.scripts_dir.join("stt_google.py");
        let output = timeout(
            Duration::from_secs(30),
            Command::new(&self.python)
                .arg(&script)
                .arg("--key")
                .arg(&self.stt_key)
                .arg("--lang")
                .arg(&self.stt_lang)
                .arg("--audio")
                .arg(&wav)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .context("google stt timed out")?
        .context("failed to launch google stt helper")?;

        if !output.status.success() {
            return Err(anyhow!(
                "stt helper failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

/// Prefer the virtualenv Python that `install.sh` creates; fall back to the
/// system interpreter so the code still runs in development.
pub(crate) fn resolve_python(scripts_dir: &std::path::Path, data_dir: &std::path::Path) -> PathBuf {
    for candidate in [
        scripts_dir.join(".venv/bin/python3"),
        data_dir.join("venv/bin/python3"),
    ] {
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("python3")
}

/// Record a mono 16 kHz WAV (the format Google STT likes) using whatever
/// recorder is present. `parecord` (PulseAudio/PipeWire) first, then `arecord`.
async fn record_wav(path: &std::path::Path, seconds: u32) -> Result<()> {
    if which("parecord").await {
        let mut child = Command::new("parecord")
            .args(["--channels=1", "--rate=16000", "--format=s16le", "--file-format=wav"])
            .arg(path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to start parecord")?;
        // parecord has no duration flag; record for `seconds`, then SIGTERM it
        // (default `kill`, NOT SIGKILL) so it finalises the WAV header, and
        // reap it so we don't leave a zombie.
        tokio::time::sleep(Duration::from_secs(seconds as u64)).await;
        if let Some(pid) = child.id() {
            let _ = Command::new("kill").arg(pid.to_string()).status().await;
        }
        let _ = child.wait().await;
        return Ok(());
    }
    if which("arecord").await {
        let status = Command::new("arecord")
            .args(["-f", "S16_LE", "-r", "16000", "-c", "1", "-d"])
            .arg(seconds.to_string())
            .arg(path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .context("failed to run arecord")?;
        if !status.success() {
            return Err(anyhow!("arecord exited with {status}"));
        }
        return Ok(());
    }
    Err(anyhow!(
        "no microphone recorder found (install pulseaudio-utils or alsa-utils)"
    ))
}

/// Play an audio file with the first available player.
async fn play_audio(path: &std::path::Path) -> Result<()> {
    for (bin, args) in [
        ("mpv", vec!["--really-quiet", "--no-video"]),
        ("ffplay", vec!["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("cvlc", vec!["--play-and-exit", "--intf", "dummy"]),
        ("paplay", vec![]),
    ] {
        if which(bin).await {
            let status = Command::new(bin)
                .args(&args)
                .arg(path)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await;
            if let Ok(s) = status {
                if s.success() {
                    return Ok(());
                }
            }
        }
    }
    Err(anyhow!(
        "no audio player found (install mpv or ffmpeg for playback)"
    ))
}

/// Cheap `which` using the shell so we don't pull in an extra crate.
async fn which(bin: &str) -> bool {
    Command::new("bash")
        .arg("-lc")
        .arg(format!("command -v {bin}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
}
