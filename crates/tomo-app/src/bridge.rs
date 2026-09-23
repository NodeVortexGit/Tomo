//! The Bevy⇄brain bridge.
//!
//! [`tomo_core::BrainHandle`] gives us two channels. This module wraps them in
//! a Bevy [`Resource`] and, once per frame, drains everything the brain has
//! said and turns it into Bevy [`Event`]s that the character, movement and
//! chat systems listen for. Outgoing user actions go the other way through
//! [`Bridge::send`].
//!
//! The receiver is behind a `Mutex` only so the resource is `Sync` (Bevy
//! requires it); there is never real contention because a single system reads
//! it.

use std::path::PathBuf;
use std::sync::Mutex;

use bevy::prelude::*;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::UnboundedReceiver;

use tomo_core::events::{BrainToUi, ChatLine};
use tomo_core::{BrainHandle, UiToBrain};

#[derive(Resource)]
pub struct Bridge {
    to_brain: tokio::sync::mpsc::UnboundedSender<UiToBrain>,
    from_brain: Mutex<UnboundedReceiver<BrainToUi>>,
}

impl Bridge {
    pub fn new(handle: BrainHandle) -> Self {
        Self {
            to_brain: handle.to_brain,
            from_brain: Mutex::new(handle.from_brain),
        }
    }

    /// Send a user action to the brain. Errors only if the brain thread died.
    pub fn send(&self, msg: UiToBrain) {
        if let Err(e) = self.to_brain.send(msg) {
            warn!("brain channel closed: {e}");
        }
    }
}

// ---- Bevy events mirroring BrainToUi -------------------------------------

#[derive(Event, Debug, Clone)]
pub struct WalkToEvent(pub f32);

#[derive(Event, Debug, Clone)]
pub struct EmoteEvent(pub String);

#[derive(Event, Debug, Clone)]
pub struct AnimateEvent(pub String);

#[derive(Event, Debug, Clone)]
pub struct LoadCharacterEvent(pub PathBuf);

#[derive(Event, Debug, Clone)]
pub struct ChatAppendEvent(pub ChatLine);

#[derive(Event, Debug, Clone)]
pub struct ThinkingEvent(pub bool);

/// "Hey Tomo" was heard (true) / the spoken request is in (false).
#[derive(Event, Debug, Clone)]
pub struct ListeningEvent(pub bool);

/// Tomo's voice started (true) / stopped (false) playing.
#[derive(Event, Debug, Clone)]
pub struct SpeakingEvent(pub bool);

/// Walk to a screen pixel and physically click it.
#[derive(Event, Debug, Clone)]
pub struct ClickAtEvent {
    pub x: f32,
    pub y: f32,
    pub double: bool,
}

/// Type text via synthesized keyboard input.
#[derive(Event, Debug, Clone)]
pub struct TypeTextEvent(pub String);

/// Show/hide the "character is driving the mouse/keyboard" indicator.
#[derive(Event, Debug, Clone)]
pub struct ControlModeEvent(pub bool);

/// Registers the events and the pump system.
pub struct BridgePlugin;

impl Plugin for BridgePlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<WalkToEvent>()
            .add_event::<EmoteEvent>()
            .add_event::<AnimateEvent>()
            .add_event::<LoadCharacterEvent>()
            .add_event::<ChatAppendEvent>()
            .add_event::<ThinkingEvent>()
            .add_event::<ListeningEvent>()
            .add_event::<SpeakingEvent>()
            .add_event::<ClickAtEvent>()
            .add_event::<TypeTextEvent>()
            .add_event::<ControlModeEvent>()
            .add_systems(Update, pump_from_brain);
    }
}

/// Drain the brain→UI channel each frame and fan out to typed Bevy events.
fn pump_from_brain(
    bridge: Res<Bridge>,
    mut walk: EventWriter<WalkToEvent>,
    mut emote: EventWriter<EmoteEvent>,
    mut animate: EventWriter<AnimateEvent>,
    mut load: EventWriter<LoadCharacterEvent>,
    mut chat: EventWriter<ChatAppendEvent>,
    mut thinking: EventWriter<ThinkingEvent>,
    mut listening: EventWriter<ListeningEvent>,
    mut speaking: EventWriter<SpeakingEvent>,
    mut click: EventWriter<ClickAtEvent>,
    mut type_text: EventWriter<TypeTextEvent>,
    mut control: EventWriter<ControlModeEvent>,
) {
    let mut rx = match bridge.from_brain.lock() {
        Ok(rx) => rx,
        Err(_) => return,
    };
    loop {
        match rx.try_recv() {
            Ok(BrainToUi::WalkTo(x)) => {
                walk.send(WalkToEvent(x));
            }
            Ok(BrainToUi::Emote(e)) => {
                emote.send(EmoteEvent(e));
            }
            Ok(BrainToUi::Animate(a)) => {
                animate.send(AnimateEvent(a));
            }
            Ok(BrainToUi::LoadCharacter(p)) => {
                load.send(LoadCharacterEvent(p));
            }
            Ok(BrainToUi::Chat(line)) => {
                chat.send(ChatAppendEvent(line));
            }
            Ok(BrainToUi::Thinking(t)) => {
                thinking.send(ThinkingEvent(t));
            }
            Ok(BrainToUi::Listening(on)) => {
                listening.send(ListeningEvent(on));
            }
            Ok(BrainToUi::Speaking(on)) => {
                speaking.send(SpeakingEvent(on));
            }
            Ok(BrainToUi::ClickAt { x, y, double }) => {
                click.send(ClickAtEvent { x, y, double });
            }
            Ok(BrainToUi::TypeText(t)) => {
                type_text.send(TypeTextEvent(t));
            }
            Ok(BrainToUi::ControlMode(on)) => {
                control.send(ControlModeEvent(on));
            }
            Ok(BrainToUi::Speak(_text)) => {
                // TTS is performed inside the brain; nothing to do UI-side.
            }
            Ok(BrainToUi::Status(s)) => {
                info!("status: {s}");
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                warn!("brain disconnected");
                break;
            }
        }
    }
}
