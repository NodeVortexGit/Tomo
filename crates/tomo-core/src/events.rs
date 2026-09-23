//! The shared vocabulary between the brain (`tomo-core`) and the body
//! (`tomo-app`).
//!
//! The two halves run on different threads: the Bevy app owns the main thread
//! (rendering must), while the AI/DB/speech work happens on a Tokio runtime.
//! They never touch each other's state directly — they only exchange the
//! messages defined here over channels. That keeps the render loop from ever
//! blocking on a network call.
//!
//! Flow:
//!   UI  -- UiToBrain  -->  brain (AI loop, DB, commands, speech)
//!   brain -- BrainToUi --> UI  (move, emote, speak, show chat text)

use serde::{Deserialize, Serialize};

/// A single line in the chat transcript, as shown to the user and stored in
/// the database. Note the AI's *command execution* never appears here — the
/// user only ever sees natural conversation, per the brief.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatLine {
    pub role: Role,
    pub text: String,
    /// Unix millis; handy for ordering and for the DB.
    pub at_ms: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    /// System / status lines (rare, e.g. "character loaded"). Not sent to model.
    System,
}

/// Messages the front-end sends to the brain.
#[derive(Debug, Clone)]
pub enum UiToBrain {
    /// The user typed (or dictated) a message.
    UserMessage(String),
    /// The mic button was pressed; brain should record + transcribe, then
    /// treat the result as a UserMessage.
    StartVoiceInput,
    /// The user dropped a new `.vrm` onto the app.
    ImportCharacter { path: std::path::PathBuf, name: String },
    /// The chat window opened/closed — lets the brain adjust idle behaviour.
    ChatVisibility(bool),
    /// The user hit the panic hotkey: release the mouse/keyboard immediately
    /// and stop any in-progress control sequence.
    PanicStop,
    /// Turn the character's ability to drive the mouse/keyboard on or off
    /// (a user setting; also flipped off by PanicStop).
    SetControlAllowed(bool),
    /// Graceful shutdown.
    Shutdown,
}

/// Messages the brain sends to the front-end.
#[derive(Debug, Clone)]
pub enum BrainToUi {
    /// Append a line to the visible transcript.
    Chat(ChatLine),
    /// Speak this text aloud (TTS). Usually paired with a Chat line.
    Speak(String),
    /// Play a facial/emotional expression by name (maps to a VRM BlendShape
    /// preset such as "happy", "sad", "surprised", "neutral").
    Emote(String),
    /// Ask the character to walk to a horizontal screen fraction in [0.0, 1.0].
    /// 0.0 = far left edge, 1.0 = far right edge. Vertical position is always
    /// the floor (the taskbar/desktop baseline) — the character never levitates.
    WalkTo(f32),
    /// The watchable path: walk the character over to a screen pixel and have
    /// it physically click there (real cursor move + click). The app shows the
    /// "control active" indicator for the duration.
    ClickAt { x: f32, y: f32, double: bool },
    /// Type text as if from the keyboard (used after focusing a field).
    TypeText(String),
    /// Show/hide the "character is driving the mouse/keyboard" indicator.
    ControlMode(bool),
    /// Play a one-off body animation clip by name ("wave", "sit", "idle").
    Animate(String),
    /// The brain is busy thinking (drives a subtle "thinking" pose, not a
    /// spinner the user has to look at).
    Thinking(bool),
    /// "Hey Tomo" was heard and Tomo is taking the spoken request (true), or
    /// is done listening (false).
    Listening(bool),
    /// Tomo's voice started (true) / stopped (false) playing — the mouth moves
    /// meanwhile.
    Speaking(bool),
    /// Load (or swap to) the VRM model at this path — used after the user
    /// imports a new character.
    LoadCharacter(std::path::PathBuf),
    /// Transient status for logs / tray, never shown as chat.
    Status(String),
}

impl ChatLine {
    pub fn new(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
            at_ms: now_ms(),
        }
    }
}

/// Milliseconds since the Unix epoch. Small helper so callers don't each
/// reinvent it.
pub fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
