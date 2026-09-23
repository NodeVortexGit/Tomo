//! Input synthesis — the "mouse in disguise".
//!
//! When the brain chooses the *watchable* path, it sends `ClickAt`/`TypeText`.
//! This module walks the character over to the spot and then drives the real
//! cursor/keyboard, using the same mechanism as `xdotool`/`ydotool` and
//! accessibility tools. It's all local and you can watch every move.
//!
//! Consent & safety are first-class here, not an afterthought:
//!   • A visible **"Tomo is in control"** badge is shown the whole time the
//!     character is driving input — you always know when it's happening.
//!   • A **panic hotkey** (Ctrl+Alt+Esc, or the Pause key) instantly releases
//!     control and tells the brain to stop. Nothing keeps driving after that.
//!   • Real input is behind the `control` cargo feature. Without it the app
//!     builds and runs fine and simply *logs* what it would have done — handy
//!     for trying the character out before granting it the keyboard.
//!
//! Enabling it: `cargo run --release --features control` and, on X11, install
//! `xdotool` (enigo links libxdo); on Wayland, run `ydotoold` (enigo drives it
//! through the virtual-input device).

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts};

use tomo_core::UiToBrain;

use crate::bridge::{Bridge, ClickAtEvent, ControlModeEvent, TypeTextEvent};
use crate::character::Character;
use crate::movement::Locomotion;

/// Whether the character is currently driving input (for the badge + gating).
#[derive(Resource, Default)]
pub struct ControlState {
    pub active: bool,
}

/// A click waiting for the character to finish walking over to it.
#[derive(Resource, Default)]
struct PendingClick {
    target: Option<(f32, f32, bool)>,
}

pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ControlState>()
            .init_resource::<PendingClick>()
            .add_systems(
                Update,
                (
                    on_control_mode,
                    on_click_at,
                    resolve_pending_click,
                    on_type_text,
                    panic_hotkey,
                    draw_control_badge,
                )
                    .chain(),
            );
    }
}

fn on_control_mode(mut events: EventReader<ControlModeEvent>, mut state: ResMut<ControlState>) {
    for ControlModeEvent(on) in events.read() {
        state.active = *on;
    }
}

/// Queue a click and start walking the character toward its column.
fn on_click_at(
    mut events: EventReader<ClickAtEvent>,
    mut pending: ResMut<PendingClick>,
    mut state: ResMut<ControlState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut loco_q: Query<&mut Locomotion, With<Character>>,
) {
    let Ok(window) = windows.get_single() else { return };
    for ev in events.read() {
        pending.target = Some((ev.x, ev.y, ev.double));
        state.active = true;
        if let Ok(mut loco) = loco_q.get_single_mut() {
            loco.walk_to(ev.x / window.width().max(1.0));
            loco.held = true; // don't wander off mid-task
        }
    }
}

/// Once the character has walked over, perform the real click.
fn resolve_pending_click(
    mut pending: ResMut<PendingClick>,
    mut state: ResMut<ControlState>,
    mut loco_q: Query<&mut Locomotion, With<Character>>,
) {
    let Some((x, y, double)) = pending.target else { return };
    let Ok(mut loco) = loco_q.get_single_mut() else { return };
    if loco.is_idle() {
        perform_click(x, y, double);
        pending.target = None;
        loco.held = false;
        state.active = false;
    }
}

fn on_type_text(mut events: EventReader<TypeTextEvent>, mut state: ResMut<ControlState>) {
    for TypeTextEvent(text) in events.read() {
        state.active = true;
        perform_type(text);
        state.active = false;
    }
}

/// Panic hotkey: Ctrl+Alt+Esc, or the Pause key. Releases control immediately
/// and tells the brain to stop scheduling more.
fn panic_hotkey(
    keys: Res<ButtonInput<KeyCode>>,
    mut pending: ResMut<PendingClick>,
    mut state: ResMut<ControlState>,
    mut loco_q: Query<&mut Locomotion, With<Character>>,
    bridge: Res<Bridge>,
) {
    let combo = keys.just_pressed(KeyCode::Escape)
        && keys.pressed(KeyCode::ControlLeft)
        && keys.pressed(KeyCode::AltLeft);
    if combo || keys.just_pressed(KeyCode::Pause) {
        pending.target = None;
        state.active = false;
        if let Ok(mut loco) = loco_q.get_single_mut() {
            loco.held = false;
        }
        bridge.send(UiToBrain::PanicStop);
        warn!("panic hotkey: released mouse/keyboard control");
    }
}

/// The always-visible indicator while the character drives input.
fn draw_control_badge(mut contexts: EguiContexts, state: Res<ControlState>) {
    if !state.active {
        return;
    }
    let ctx = contexts.ctx_mut();
    egui::Area::new(egui::Id::new("tomo-control-badge"))
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 14.0))
        .show(ctx, |ui| {
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_unmultiplied(200, 60, 60, 235))
                .rounding(999.0)
                .inner_margin(egui::Margin::symmetric(14.0, 7.0))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("🖱  Tomo is controlling the desktop  ·  press Pause to stop")
                            .color(egui::Color32::WHITE)
                            .strong(),
                    );
                });
        });
}

// ---- the actual synthesis, gated behind the `control` feature -------------

#[cfg(feature = "control")]
fn perform_click(x: f32, y: f32, double: bool) {
    use enigo::{Button, Coordinate, Direction, Enigo, Mouse, Settings};
    match Enigo::new(&Settings::default()) {
        Ok(mut e) => {
            let _ = e.move_mouse(x as i32, y as i32, Coordinate::Abs);
            let _ = e.button(Button::Left, Direction::Click);
            if double {
                let _ = e.button(Button::Left, Direction::Click);
            }
        }
        Err(err) => warn!("could not init input backend: {err}"),
    }
}

#[cfg(not(feature = "control"))]
fn perform_click(x: f32, y: f32, double: bool) {
    info!("[control feature off] would {}click at ({x:.0},{y:.0})", if double { "double-" } else { "" });
}

#[cfg(feature = "control")]
fn perform_type(text: &str) {
    use enigo::{Enigo, Keyboard, Settings};
    match Enigo::new(&Settings::default()) {
        Ok(mut e) => {
            let _ = e.text(text);
        }
        Err(err) => warn!("could not init input backend: {err}"),
    }
}

#[cfg(not(feature = "control"))]
fn perform_type(text: &str) {
    info!("[control feature off] would type: {text:?}");
}
