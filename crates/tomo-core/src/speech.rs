//! Speech out, on this computer: Piper turns Tomo's replies into a voice.
//!
//! Piper (a small, fast neural text-to-speech model) runs in a Python helper,
//! `scripts/tts_piper.py`, started once and kept running: it loads the voice
//! once, then turns each line of text it's sent into a WAV file. Playback uses
//! what the system has — PipeWire, PulseAudio or ALSA players on Linux, the
//! built-in sound player on Windows, `afplay` on macOS. (Speech *in* is the
//! listener in wake.rs.)

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::timeout;

use crate::config::Config;
use crate::platform;

/// Loading the voice the first time can mean downloading it.
const START_TIMEOUT: Duration = Duration::from_secs(180);
const SPEAK_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub struct Speech {
    python: PathBuf,
    script: PathBuf,
    voices_dir: PathBuf,
    cache_dir: PathBuf,
    voice: String,
    speed: f32,
    helper: Arc<Mutex<Option<Helper>>>,
    count: Arc<AtomicU64>,
}

/// The running Piper helper.
struct Helper {
    _child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

impl Speech {
    pub fn new(cfg: &Config) -> Self {
        let cache_dir = cfg.data_dir.join("cache");
        std::fs::create_dir_all(&cache_dir).ok();
        Self {
            python: resolve_python(&cfg.scripts_dir, &cfg.data_dir),
            script: cfg.scripts_dir.join("tts_piper.py"),
            voices_dir: cfg.data_dir.join("voices"),
            cache_dir,
            voice: cfg.tts_voice.clone(),
            speed: cfg.tts_speed,
            helper: Arc::new(Mutex::new(None)),
            count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Synthesize `text` into a WAV file for [`play`]. Separate from playback
    /// so the body can move its mouth exactly while the voice plays. Markdown
    /// and emoji are left out of the voice (see [`speakable`]).
    pub async fn synthesize(&self, text: &str) -> Result<PathBuf> {
        let text = speakable(text);
        if text.is_empty() {
            bail!("nothing to say out loud");
        }
        let out = self.cache_dir.join(format!("speech-{}.wav", self.count.fetch_add(1, Ordering::Relaxed)));
        let mut helper = self.helper.lock().await;
        if helper.is_none() {
            *helper = Some(self.start_helper().await?);
        }
        let running = helper.as_mut().expect("just started");
        let request = json!({ "text": text, "out": out, "speed": self.speed }).to_string() + "\n";
        let answer = timeout(SPEAK_TIMEOUT, async {
            running.stdin.write_all(request.as_bytes()).await?;
            running.stdin.flush().await?;
            running.lines.next_line().await?.ok_or_else(|| anyhow!("the voice helper stopped"))
        })
        .await
        .map_err(|_| anyhow!("the voice took too long"))
        .and_then(|answer| answer);
        match answer.and_then(|line| helper_reply(&line)) {
            Ok(()) => Ok(out),
            Err(e) => {
                // Start afresh next time.
                *helper = None;
                Err(e)
            }
        }
    }

    /// Play synthesized speech, then delete it. Returns once playback ends.
    pub async fn play(&self, audio: &Path) -> Result<()> {
        let played = play_wav(audio).await;
        let _ = tokio::fs::remove_file(audio).await;
        played
    }

    async fn start_helper(&self) -> Result<Helper> {
        let mut command = Command::new(&self.python);
        command
            .arg(&self.script)
            .arg("--voice")
            .arg(&self.voice)
            .arg("--voices-dir")
            .arg(&self.voices_dir);
        let mut child = platform::quiet(&mut command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("couldn't start the voice helper (tts_piper.py)")?;
        let stdin = child.stdin.take().context("no stdin")?;
        let mut lines = BufReader::new(child.stdout.take().context("no stdout")?).lines();
        let ready = timeout(START_TIMEOUT, lines.next_line())
            .await
            .map_err(|_| anyhow!("the voice took too long to load"))??
            .ok_or_else(|| anyhow!("the voice helper exited"))?;
        helper_reply(&ready)?;
        tracing::info!("voice ready: {}", self.voice);
        Ok(Helper { _child: child, stdin, lines })
    }
}

/// The helper answers each line with `{"ok": true}` or `{"error": "…"}`.
fn helper_reply(line: &str) -> Result<()> {
    let reply: serde_json::Value = serde_json::from_str(line).context("the voice helper said something odd")?;
    match reply["error"].as_str() {
        Some(error) => bail!("{error}"),
        None => Ok(()),
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

/// Prefer the virtual environment the installer creates; fall back to the
/// system's Python so the code still runs in development.
pub(crate) fn resolve_python(scripts_dir: &Path, data_dir: &Path) -> PathBuf {
    [scripts_dir.join(".venv"), data_dir.join("venv")]
        .iter()
        .map(|venv| platform::venv_python(venv))
        .find(|python| python.exists())
        .unwrap_or_else(|| PathBuf::from(platform::system_python()))
}

/// Play a WAV file with what the system has.
async fn play_wav(path: &Path) -> Result<()> {
    if cfg!(windows) {
        // The sound player built into Windows plays WAV files, and waits.
        let file = path.display().to_string().replace('\'', "''");
        let status = platform::shell(&format!("(New-Object System.Media.SoundPlayer '{file}').PlaySync()"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await?;
        return if status.success() { Ok(()) } else { Err(anyhow!("the sound player failed")) };
    }
    let players: &[(&str, &[&str])] = &[
        ("afplay", &[]),
        ("pw-play", &[]),
        ("paplay", &[]),
        ("aplay", &["-q"]),
        ("mpv", &["--really-quiet", "--no-video"]),
        ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ];
    for (player, args) in players {
        let Some(program) = platform::which(player) else { continue };
        let status = Command::new(program)
            .args(*args)
            .arg(path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
        if status.is_ok_and(|s| s.success()) {
            return Ok(());
        }
    }
    Err(anyhow!("no audio player worked (install pipewire, pulseaudio-utils or alsa-utils)"))
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
