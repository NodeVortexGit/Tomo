//! `tomo-core` — the brain of the Tomo desktop companion.
//!
//! This crate is deliberately free of any graphics / windowing dependency so
//! it builds in seconds and is fully unit-testable. It contains:
//!
//!   [`config`]   — load `.env` + XDG paths into a validated [`config::Config`]
//!   [`db`]       — SQLite long-term memory (prefs, memories, transcript, chars)
//!   [`characters`] — the `.vrm` models Tomo can appear as, and switching
//!   [`commands`] — the audited, deny-listed shell [`commands::Executor`]
//!   [`ai`]       — the local model's tool-use loop (Ollama / LM Studio), [`ai::AiClient`]
//!   [`screen`]   — screenshots for the model to look at
//!   [`speech`]   — Piper text-to-speech, [`speech::Speech`]
//!   [`wake`]     — voice input: "Hey Tomo" and push-to-talk (Vosk + Whisper)
//!   [`platform`] — what differs between Linux, Windows and macOS
//!   [`events`]   — the message types the brain and body exchange
//!   [`brain`]    — [`brain::Brain::spawn`], the async event loop + channels
//!
//! The graphical front-end (`tomo-app`) depends on this crate and only ever
//! talks to [`brain::BrainHandle`]'s two channels.

pub mod ai;
pub mod apps;
pub mod brain;
pub mod characters;
pub mod commands;
pub mod config;
pub mod db;
pub mod events;
pub mod platform;
pub mod screen;
pub mod speech;
pub mod wake;

pub use brain::{Brain, BrainHandle};
pub use config::Config;
pub use events::{BrainToUi, CharacterChoice, ChatLine, Role, UiToBrain};

/// Initialise tracing once. Safe to call from either crate's `main`.
/// Honours `RUST_LOG`; defaults to `info` (the renderer's crates quieter).
/// On Windows, where Tomo has no console, the log goes to `tomo.log` in the
/// data folder instead.
pub fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    // The body's renderer (wgpu/naga) is chatty below these levels.
    let filter = || {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,wgpu=error,wgpu_core=warn,wgpu_hal=warn,naga=warn"))
    };
    if cfg!(windows) {
        let file = config::default_data_dir().and_then(|dir| {
            std::fs::create_dir_all(&dir).ok()?;
            std::fs::File::create(dir.join("tomo.log")).ok()
        });
        if let Some(file) = file {
            let writer = std::sync::Mutex::new(file);
            let _ = fmt().with_env_filter(filter()).with_target(false).with_ansi(false).with_writer(writer).try_init();
            return;
        }
    }
    let _ = fmt().with_env_filter(filter()).with_target(false).try_init();
}
