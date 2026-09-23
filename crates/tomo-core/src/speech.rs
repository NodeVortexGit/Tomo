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
    /// Markdown and emoji are left out of the voice (see [`speakable`]).
    pub async fn synthesize(&self, text: &str) -> Result<PathBuf> {
        let text = speakable(text);
        if text.is_empty() {
            return Err(anyhow!("nothing to say out loud"));
        }
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
                .arg(&text)
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

/// A reply as it should be *heard*. The chat shows markdown and emoji fine,
/// but a voice saying "asterisk" or "party popper" out loud is not. Drops
/// markdown markers and emoji, keeps a link's words but not its address, and
/// ends every line on a pause so list items don't run together.
pub fn speakable(text: &str) -> String {
    let mut lines = Vec::new();
    for line in text.lines() {
        let line = link_words(line);
        let line = line.trim_start().trim_start_matches(['#', '>']).trim_start();
        let line = ["- ", "* ", "+ ", "• "]
            .iter()
            .find_map(|bullet| line.strip_prefix(bullet))
            .unwrap_or(line);
        let cleaned: String = line
            .chars()
            .filter_map(|c| match c {
                '*' | '`' | '~' | '|' => None,
                '_' => Some(' '),
                c if is_emoji(c) => None,
                c => Some(c),
            })
            .collect();
        // Rejoin the words; punctuation that lost its word to a dropped emoji
        // ("time ❤️!") goes back onto the word before it.
        let mut words = String::new();
        for word in cleaned.split_whitespace() {
            if !words.is_empty() && !word.starts_with(['.', ',', '!', '?', ';', ':', '…', ')']) {
                words.push(' ');
            }
            words.push_str(word);
        }
        if !words.chars().any(char::is_alphanumeric) {
            continue; // blank, a rule like "---", or nothing but emoji
        }
        if words.ends_with(['.', '!', '?', '…', ':', ';', ',']) {
            lines.push(words);
        } else {
            lines.push(words + ".");
        }
    }
    lines.join("\n")
}

/// `[words](address)` → `words`, anywhere in the line.
fn link_words(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('[') {
        let inner = &rest[open + 1..];
        let link = inner.find(']').and_then(|close| {
            let target = inner[close + 1..].strip_prefix('(')?;
            let end = target.find(')')?;
            Some((&inner[..close], &target[end + 1..]))
        });
        match link {
            Some((words, after)) => {
                out.push_str(&rest[..open]);
                out.push_str(words);
                rest = after;
            }
            None => {
                out.push_str(&rest[..=open]);
                rest = inner;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Emoji and pictographic symbols (plus the joiners and variation selectors
/// that glue them together), none of which a voice should read out.
fn is_emoji(c: char) -> bool {
    matches!(u32::from(c),
        0x1F000..=0x1FAFF   // emoticons, pictographs, flags, …
        | 0x2190..=0x21FF   // arrows
        | 0x2300..=0x23FF   // ⌚ ⏰ …
        | 0x2600..=0x27BF   // ☀ ♥ ✔ ✨ …
        | 0x2B00..=0x2BFF   // ⭐ ⬆ …
        | 0xFE00..=0xFE0F   // variation selectors
        | 0x200D            // zero-width joiner
        | 0x20E3            // keycap
        | 0xE0000..=0xE007F // tags (subdivision flags)
    )
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

#[cfg(test)]
mod tests {
    use super::speakable;

    #[test]
    fn markdown_markers_are_not_read_out() {
        assert_eq!(speakable("**Sure!** I'll `wave` now"), "Sure! I'll wave now.");
        assert_eq!(speakable("## Plan\n> quoted"), "Plan.\nquoted.");
        assert_eq!(speakable("a snake_case name"), "a snake case name.");
    }

    #[test]
    fn emoji_are_dropped() {
        assert_eq!(speakable("Hi there 👋"), "Hi there.");
        assert_eq!(speakable("Family 👨‍👩‍👧 time ❤️!"), "Family time!");
        assert_eq!(speakable("🎉🎉"), "");
    }

    #[test]
    fn list_items_each_get_a_pause() {
        assert_eq!(speakable("Here:\n- one\n* two\n\n---\n3. three"), "Here:\none.\ntwo.\n3. three.");
    }

    #[test]
    fn links_keep_their_words_only() {
        assert_eq!(speakable("See [the docs](https://example.com/x) now"), "See the docs now.");
        assert_eq!(speakable("[a] and [b](c)"), "[a] and b.");
    }
}
