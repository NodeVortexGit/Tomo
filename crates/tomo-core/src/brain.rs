//! The brain's event loop — the seam between the async world (AI, DB, speech,
//! shell) and the synchronous Bevy render loop.
//!
//! [`Brain::spawn`] starts a dedicated OS thread that owns a Tokio runtime and
//! runs [`run_loop`]. It hands back a [`BrainHandle`] holding two channels:
//!
//!   handle.to_brain    the app pushes [`UiToBrain`] here (user typed, mic
//!                      pressed, character imported, shutdown)
//!   handle.from_brain  the app drains [`BrainToUi`] here every frame (chat
//!                      lines, walk/emote/animate, thinking, load-character)
//!
//! Because everything the app does is a non-blocking channel op, the render
//! loop never waits on the network or the model. Speaking (TTS) is spawned as
//! its own task so a long sentence doesn't stall the next user message.
//!
//! Single source of truth: the app does NOT add the user's own line to the
//! transcript. It sends `UserMessage`; the brain persists it, echoes it back as
//! a `Chat` line, then follows with the assistant's reply. That keeps the UI,
//! the database and the model context perfectly in step.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::ai::{AiClient, ControlFlag, SharedCatalog, TRANSCRIPT_CONTEXT};
use crate::apps::SystemCatalog;
use crate::characters;
use crate::commands::Executor;
use crate::config::Config;
use crate::db::Db;
use crate::events::{BrainToUi, ChatLine, Role, UiToBrain};
use crate::speech::{resolve_python, Speech};
use crate::wake::{self, Wake};

/// How many seconds of microphone audio one voice-input press captures.
const VOICE_SECONDS: u32 = 5;
/// How often to re-scan the OS for newly installed/removed apps + toggles.
const RESCAN_INTERVAL: Duration = Duration::from_secs(120);

pub struct BrainHandle {
    pub to_brain: UnboundedSender<UiToBrain>,
    pub from_brain: UnboundedReceiver<BrainToUi>,
    /// Kept so the worker thread isn't detached; joined on drop attempts.
    _thread: std::thread::JoinHandle<()>,
}

pub struct Brain;

impl Brain {
    /// Start the brain on its own thread + runtime. Returns immediately.
    pub fn spawn(cfg: Config) -> Result<BrainHandle> {
        let (to_brain_tx, to_brain_rx) = unbounded_channel::<UiToBrain>();
        let (from_brain_tx, from_brain_rx) = unbounded_channel::<BrainToUi>();
        // The wake-word listener speaks to the brain the way the app does.
        let voice_tx = to_brain_tx.clone();

        let thread = std::thread::Builder::new()
            .name("tomo-brain".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!("failed to build brain runtime: {e}");
                        return;
                    }
                };
                rt.block_on(async move {
                    if let Err(e) = run_loop(cfg, to_brain_rx, voice_tx, from_brain_tx).await {
                        tracing::error!("brain loop ended with error: {e}");
                    }
                });
            })?;

        Ok(BrainHandle {
            to_brain: to_brain_tx,
            from_brain: from_brain_rx,
            _thread: thread,
        })
    }
}

async fn run_loop(
    cfg: Config,
    mut rx: UnboundedReceiver<UiToBrain>,
    voice_tx: UnboundedSender<UiToBrain>,
    ui: UnboundedSender<BrainToUi>,
) -> Result<()> {
    tracing::info!("brain starting: {}", cfg.redacted());

    let db = Db::open(&cfg.db_path)?;
    let executor = Executor::new(
        cfg.allow_command_execution,
        cfg.audit_log_path.clone(),
        cfg.extra_allowed_commands.clone(),
    );
    let speech = Speech::new(&cfg);
    let python = resolve_python(&cfg.scripts_dir, &cfg.data_dir);
    let wake = wake::spawn(&cfg, python, voice_tx, ui.clone());
    // Replies are spoken unless the chat's 🔊 toggle is off.
    let mut voice = true;

    // Live OS catalog (apps + toggles) and the mouse/keyboard control switch.
    let catalog: SharedCatalog = Arc::new(RwLock::new(load_cached_catalog(&db)));
    let control_allowed: ControlFlag = Arc::new(AtomicBool::new(cfg.allow_command_execution));
    let ai = AiClient::new(
        cfg.clone(),
        db.clone(),
        executor,
        catalog.clone(),
        control_allowed.clone(),
    );

    // Scan the OS now and then keep it fresh (boot scan + change detection).
    spawn_catalog_scanner(db.clone(), catalog.clone(), ui.clone());

    // Tell the app which character to show first, and what else it could be.
    match characters::startup(&db, &cfg) {
        Some(path) => {
            let _ = ui.send(BrainToUi::LoadCharacter(path));
        }
        None => tracing::warn!(
            "no character to show: put a .vrm at {} or set TOMO_CHARACTER in .env",
            cfg.character_path.display()
        ),
    }
    let _ = ui.send(BrainToUi::Characters(characters::available(&db, &cfg)));
    // Pick the conversation up where it left off: the chat shows the same
    // recent lines the model gets as context.
    for line in db.recent_messages(TRANSCRIPT_CONTEXT).unwrap_or_default() {
        let _ = ui.send(BrainToUi::Chat(line));
    }
    let _ = ui.send(BrainToUi::Status("ready".into()));

    while let Some(msg) = rx.recv().await {
        match msg {
            UiToBrain::UserMessage(text) => {
                let speech = voice.then_some(&speech);
                handle_user_text(&db, &ai, speech, wake.as_ref(), &ui, text).await;
            }
            UiToBrain::StartVoiceInput => {
                // Push-to-talk through the offline listener when it runs.
                if let Some(wake) = &wake {
                    wake.listen();
                    continue;
                }
                if !speech.can_listen() {
                    let _ = ui.send(BrainToUi::Status(
                        "voice input needs GOOGLE_STT_API_KEY in .env".into(),
                    ));
                    continue;
                }
                let _ = ui.send(BrainToUi::Status("listening…".into()));
                match speech.listen(VOICE_SECONDS).await {
                    Ok(text) if !text.trim().is_empty() => {
                        let speech = voice.then_some(&speech);
                        handle_user_text(&db, &ai, speech, None, &ui, text).await;
                    }
                    Ok(_) => {
                        let _ = ui.send(BrainToUi::Status("didn't catch that".into()));
                    }
                    Err(e) => {
                        let _ = ui.send(BrainToUi::Status(format!("mic error: {e}")));
                    }
                }
            }
            UiToBrain::ImportCharacter { path, name } => {
                match characters::activate(&db, &cfg, &path, &name) {
                    Ok(path) => {
                        let _ = ui.send(BrainToUi::LoadCharacter(path));
                        let _ = ui.send(BrainToUi::Characters(characters::available(&db, &cfg)));
                    }
                    Err(e) => {
                        tracing::warn!("couldn't switch to {}: {e}", path.display());
                        let _ = ui.send(BrainToUi::Chat(ChatLine::new(
                            Role::System,
                            format!("Couldn't load “{name}”: {e}"),
                        )));
                    }
                }
            }
            UiToBrain::ChatVisibility(_visible) => {
                // Reserved: could nudge idle behaviour when chat opens/closes.
            }
            UiToBrain::PanicStop => {
                control_allowed.store(false, Ordering::Relaxed);
                let _ = ui.send(BrainToUi::ControlMode(false));
                let _ = ui.send(BrainToUi::Status("control released (panic hotkey)".into()));
                tracing::warn!("panic stop: mouse/keyboard control disabled");
            }
            UiToBrain::SetVoice(on) => {
                voice = on;
                tracing::info!("voice replies {}", if on { "on" } else { "off" });
            }
            UiToBrain::SetControlAllowed(on) => {
                control_allowed.store(on, Ordering::Relaxed);
                let _ = ui.send(BrainToUi::Status(
                    if on { "control enabled" } else { "control disabled" }.into(),
                ));
            }
            UiToBrain::Shutdown => {
                tracing::info!("brain shutting down");
                break;
            }
        }
    }
    Ok(())
}

/// The core turn: persist + echo the user line, get a reply, persist + show +
/// speak it (unless `speech` is `None`: the voice is switched off).
/// Movement/emotion happen inside `ai.respond` via the `ui` sender.
async fn handle_user_text(
    db: &Db,
    ai: &AiClient,
    speech: Option<&Speech>,
    wake: Option<&Wake>,
    ui: &UnboundedSender<BrainToUi>,
    text: String,
) {
    // Echo the user's line to the UI immediately, but DON'T persist it yet:
    // `ai.respond` rebuilds context from the stored transcript and appends the
    // current message itself. Persisting first would make it appear twice in
    // the model's context. We store both lines together after the reply.
    let user_line = ChatLine::new(Role::User, text.clone());
    let _ = ui.send(BrainToUi::Chat(user_line.clone()));

    let reply = match ai.respond(&text, ui).await {
        Ok(r) if !r.trim().is_empty() => r,
        Ok(_) => "…".to_string(),
        Err(e) => {
            tracing::warn!("ai error: {e}");
            format!("Sorry — I hit a snag reaching my brain ({e}).")
        }
    };

    let reply_line = ChatLine::new(Role::Assistant, reply.clone());
    let _ = db.add_message(&user_line);
    let _ = db.add_message(&reply_line);
    let _ = ui.send(BrainToUi::Chat(reply_line));

    // Speak in the background so the loop is free for the next message, with
    // the wake word muted so Tomo's own voice can't trigger it.
    let Some(speech) = speech.cloned() else {
        return;
    };
    let wake = wake.cloned();
    let ui = ui.clone();
    tokio::spawn(async move {
        let audio = match speech.synthesize(&reply).await {
            Ok(audio) => audio,
            Err(e) => return tracing::debug!("tts skipped: {e}"),
        };
        if let Some(wake) = &wake {
            wake.mute(true);
        }
        let _ = ui.send(BrainToUi::Speaking(true));
        if let Err(e) = speech.play(&audio).await {
            tracing::debug!("tts playback failed: {e}");
        }
        let _ = ui.send(BrainToUi::Speaking(false));
        if let Some(wake) = &wake {
            wake.mute(false);
        }
    });
}

/// Load the last catalog scan from the DB so the AI has something to work with
/// instantly at boot, before the first fresh scan finishes.
fn load_cached_catalog(db: &Db) -> SystemCatalog {
    db.get_snapshot("catalog")
        .ok()
        .flatten()
        .and_then(|(_, json)| serde_json::from_str::<SystemCatalog>(&json).ok())
        .unwrap_or_default()
}

/// Scan the OS once now, then on a light interval — updating the shared catalog
/// and the DB cache only when the fingerprint changes ("fetch at boot, and
/// again on change"). Filesystem work runs on the blocking pool.
fn spawn_catalog_scanner(db: Db, catalog: SharedCatalog, ui: UnboundedSender<BrainToUi>) {
    tokio::spawn(async move {
        loop {
            if let Ok(fresh) = tokio::task::spawn_blocking(SystemCatalog::scan).await {
                let fp = fresh.fingerprint() as i64;
                let prev = db.get_snapshot("catalog").ok().flatten().map(|(f, _)| f);
                if prev != Some(fp) {
                    if let Ok(json) = serde_json::to_string(&fresh) {
                        let _ = db.set_snapshot("catalog", fp, &json);
                    }
                    let n = fresh.apps.len();
                    if let Ok(mut guard) = catalog.write() {
                        *guard = fresh;
                    }
                    let _ = ui.send(BrainToUi::Status(format!("catalog refreshed: {n} apps")));
                    tracing::info!("catalog refreshed: {n} apps");
                }
            }
            tokio::time::sleep(RESCAN_INTERVAL).await;
        }
    });
}
