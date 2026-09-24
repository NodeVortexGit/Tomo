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

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::UnboundedReceiver;

use tomo_core::events::{BrainToUi, CharacterChoice, ChatLine};
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

    /// A sender of its own, for work off the render thread (a file dialog).
    pub fn sender(&self) -> tokio::sync::mpsc::UnboundedSender<UiToBrain> {
        self.to_brain.clone()
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

/// The characters on offer changed (for the chat's character menu).
#[derive(Event, Debug, Clone)]
pub struct CharactersEvent(pub Vec<CharacterChoice>);

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
            .add_event::<CharactersEvent>()
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

/// Everything the brain can make happen, as Bevy events.
#[derive(SystemParam)]
struct BrainEvents<'w> {
    walk: EventWriter<'w, WalkToEvent>,
    emote: EventWriter<'w, EmoteEvent>,
    animate: EventWriter<'w, AnimateEvent>,
    load: EventWriter<'w, LoadCharacterEvent>,
    offered: EventWriter<'w, CharactersEvent>,
    chat: EventWriter<'w, ChatAppendEvent>,
    thinking: EventWriter<'w, ThinkingEvent>,
    listening: EventWriter<'w, ListeningEvent>,
    speaking: EventWriter<'w, SpeakingEvent>,
    click: EventWriter<'w, ClickAtEvent>,
    type_text: EventWriter<'w, TypeTextEvent>,
    control: EventWriter<'w, ControlModeEvent>,
}

/// Drain the brain→UI channel each frame and fan out to typed Bevy events.
fn pump_from_brain(bridge: Res<Bridge>, mut events: BrainEvents) {
    let mut rx = match bridge.from_brain.lock() {
        Ok(rx) => rx,
        Err(_) => return,
    };
    loop {
        match rx.try_recv() {
            Ok(BrainToUi::WalkTo(x)) => {
                events.walk.send(WalkToEvent(x));
            }
            Ok(BrainToUi::Emote(e)) => {
                events.emote.send(EmoteEvent(e));
            }
            Ok(BrainToUi::Animate(a)) => {
                events.animate.send(AnimateEvent(a));
            }
            Ok(BrainToUi::LoadCharacter(p)) => {
                events.load.send(LoadCharacterEvent(p));
            }
            Ok(BrainToUi::Characters(list)) => {
                events.offered.send(CharactersEvent(list));
            }
            Ok(BrainToUi::Chat(line)) => {
                events.chat.send(ChatAppendEvent(line));
            }
            Ok(BrainToUi::Thinking(t)) => {
                events.thinking.send(ThinkingEvent(t));
            }
            Ok(BrainToUi::Listening(on)) => {
                events.listening.send(ListeningEvent(on));
            }
            Ok(BrainToUi::Speaking(on)) => {
                events.speaking.send(SpeakingEvent(on));
            }
            Ok(BrainToUi::ClickAt { x, y, double }) => {
                events.click.send(ClickAtEvent { x, y, double });
            }
            Ok(BrainToUi::TypeText(t)) => {
                events.type_text.send(TypeTextEvent(t));
            }
            Ok(BrainToUi::ControlMode(on)) => {
                events.control.send(ControlModeEvent(on));
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
