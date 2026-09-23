//! "Hey Tomo": the always-on wake-word listener.
//!
//! Runs `scripts/wake_word.py` as a child process — Vosk spots the phrase,
//! Whisper transcribes the request, all offline — and turns its JSON-lines
//! output into brain traffic: a spoken request becomes a
//! [`UiToBrain::UserMessage`], exactly as if typed, and the body is told while
//! Tomo is listening. The models live in `<data dir>/models` (install.sh
//! downloads them); without them the listener stays off.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::config::Config;
use crate::events::{BrainToUi, UiToBrain};

pub const VOSK_MODEL: &str = "vosk-model-small-en-us-0.15";
pub const WHISPER_MODEL: &str = "whisper-base.en";

/// Steers the running listener. Cheap to clone.
#[derive(Clone)]
pub struct Wake {
    commands: UnboundedSender<&'static str>,
}

impl Wake {
    /// Skip the wake phrase and take a request now (push-to-talk).
    pub fn listen(&self) {
        let _ = self.commands.send("listen");
    }

    /// Ignore the microphone while Tomo speaks, so she can't wake herself.
    pub fn mute(&self, muted: bool) {
        let _ = self.commands.send(if muted { "mute" } else { "unmute" });
    }
}

#[derive(Deserialize)]
struct Event {
    event: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    message: String,
}

/// Start listening, if switched on and the speech models are installed.
pub fn spawn(
    cfg: &Config,
    python: PathBuf,
    to_brain: UnboundedSender<UiToBrain>,
    ui: UnboundedSender<BrainToUi>,
) -> Option<Wake> {
    if !cfg.wake_word {
        return None;
    }
    let models = cfg.data_dir.join("models");
    let (vosk, whisper) = (models.join(VOSK_MODEL), models.join(WHISPER_MODEL));
    if !vosk.is_dir() || !whisper.is_dir() {
        tracing::warn!(
            "\"Hey Tomo\" is off: speech models missing in {} (install.sh downloads them)",
            models.display()
        );
        return None;
    }
    let script = cfg.scripts_dir.join("wake_word.py");
    let (commands, mut pending) = unbounded_channel();

    tokio::spawn(async move {
        // Restart after a crash, but give up on one that keeps failing fast
        // (no microphone, broken install).
        let mut quick_failures = 0;
        loop {
            let started = Instant::now();
            match listen(&python, &script, &vosk, &whisper, &mut pending, &to_brain, &ui).await {
                Ok(()) => return, // the brain is gone
                Err(e) => tracing::warn!("wake-word listener stopped: {e:#}"),
            }
            let _ = ui.send(BrainToUi::Listening(false));
            quick_failures = if started.elapsed() > Duration::from_secs(60) {
                1
            } else {
                quick_failures + 1
            };
            if quick_failures == 3 {
                tracing::warn!("\"Hey Tomo\" is off after repeated failures");
                return;
            }
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    });
    Some(Wake { commands })
}

/// One run of the listener process. Ends with `Ok` when the brain drops its
/// last [`Wake`], and with an error when the process fails.
async fn listen(
    python: &Path,
    script: &Path,
    vosk: &Path,
    whisper: &Path,
    pending: &mut UnboundedReceiver<&'static str>,
    to_brain: &UnboundedSender<UiToBrain>,
    ui: &UnboundedSender<BrainToUi>,
) -> Result<()> {
    let mut child = Command::new(python)
        .arg(script)
        .arg("--vosk")
        .arg(vosk)
        .arg("--whisper")
        .arg(whisper)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .context("couldn't start wake_word.py")?;
    let mut stdin = child.stdin.take().context("no stdin")?;
    let mut lines = BufReader::new(child.stdout.take().context("no stdout")?).lines();

    loop {
        tokio::select! {
            line = lines.next_line() => {
                let Some(line) = line? else { return Err(anyhow!("it exited")) };
                let Ok(event) = serde_json::from_str::<Event>(&line) else { continue };
                match event.event.as_str() {
                    "ready" => tracing::info!("listening for \"Hey Tomo\""),
                    "wake" => {
                        let _ = ui.send(BrainToUi::Listening(true));
                    }
                    "heard" => {
                        let _ = ui.send(BrainToUi::Listening(false));
                        let _ = to_brain.send(UiToBrain::UserMessage(event.text));
                    }
                    "idle" => {
                        let _ = ui.send(BrainToUi::Listening(false));
                    }
                    "error" => return Err(anyhow!(event.message)),
                    _ => {}
                }
            }
            command = pending.recv() => {
                let Some(command) = command else { return Ok(()) };
                stdin.write_all(format!("{command}\n").as_bytes()).await?;
            }
        }
    }
}
