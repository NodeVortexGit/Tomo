//! `tomo-core` — the brain of the Tomo desktop companion.
//!
//! This crate is deliberately free of any graphics / windowing dependency so
//! it builds in seconds and is fully unit-testable. It contains:
//!
//!   [`config`]   — load `.env` + XDG paths into a validated [`config::Config`]
//!   [`db`]       — SQLite long-term memory (prefs, memories, transcript, chars)
//!   [`characters`] — the `.vrm` models Tomo can appear as, and switching
//!   [`commands`] — the audited, deny-listed shell [`commands::Executor`]
//!   [`ai`]       — Claude's tool-use loop, [`ai::AiClient`]
//!   [`screen`]   — screenshots for the model to look at
//!   [`speech`]   — Edge-TTS + Google-STT [`speech::Speech`] bridge
//!   [`wake`]     — the offline "Hey Tomo" wake-word listener
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
pub mod screen;
pub mod speech;
pub mod wake;

pub use brain::{Brain, BrainHandle};
pub use config::Config;
pub use events::{BrainToUi, CharacterChoice, ChatLine, Role, UiToBrain};

/// Initialise tracing once. Safe to call from either crate's `main`.
/// Honours `RUST_LOG`; defaults to `info` (the renderer's crates quieter).
pub fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    // The body's renderer (wgpu/naga) is chatty below these levels.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,wgpu=error,wgpu_core=warn,wgpu_hal=warn,naga=warn"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}
