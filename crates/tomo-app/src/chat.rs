//! The chat interaction and its signature animation.
//!
//! The brief describes a specific choreography, and this module implements its
//! state machine end to end:
//!
//!   1. The user clicks the character.
//!   2. The character walks to the right side of the screen.        (Phase::Sliding)
//!   3. A "liquid" blob emerges from it and grows.                  (Phase::Liquid)
//!   4. Once fully out, the liquid forms the chat window.           (Phase::Open)
//!   5. Closing reverses it: the window melts back in.              (Phase::Closing)
//!      (× button, Escape, clicking the character again, or — in a regular
//!      window — clicking away; the overlay lets those clicks through.)
//!
//! The transcript, the text input, the mic (STT) button and the thinking
//! indicator all live in the window. Command execution is never shown here —
//! only natural conversation, exactly as asked.
//!
//! The liquid is drawn as a cluster of merging circles (a cheap metaball look)
//! that eases from the character's position into a rounded panel. That reads
//! well and needs no shader; `TODO(fluid)` marks where to drop in a real
//! signed-distance-field / metaball shader for a glossier effect.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts};

use crate::bridge::{Bridge, ChatAppendEvent, ListeningEvent, ThinkingEvent};
use crate::character::Character;
use crate::movement::{CharacterClicked, Locomotion};
use crate::window::{pixel_rect, Busy, InputRegion};
use tomo_core::events::{ChatLine, Role};
use tomo_core::UiToBrain;

/// Where the character parks to open the chat (screen fraction from the left).
const DOCK_FRACTION: f32 = 0.86;
/// Seconds for the liquid to fully emerge / melt back.
const LIQUID_GROW_SECS: f32 = 0.55;
const LIQUID_MELT_SECS: f32 = 0.40;
/// Chat panel size in logical pixels.
const PANEL_W: f32 = 340.0;
const PANEL_H: f32 = 460.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Closed,
    Sliding,
    Liquid,
    Open,
    Closing,
}

#[derive(Resource)]
pub struct ChatState {
    phase: Phase,
    /// 0.0 = fully retracted, 1.0 = fully formed panel.
    liquid_t: f32,
    transcript: Vec<ChatLine>,
    input: String,
    thinking: bool,
    /// "Hey Tomo" was heard; the spoken request is being taken.
    listening: bool,
    /// Toggle shown in the header (brain does the actual TTS).
    voice_replies: bool,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            phase: Phase::Closed,
            liquid_t: 0.0,
            transcript: Vec::new(),
            input: String::new(),
            thinking: false,
            listening: false,
            voice_replies: true,
        }
    }
}

pub struct ChatPlugin;

impl Plugin for ChatPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChatState>().add_systems(
            Update,
            (
                ingest_brain_events,
                toggle_on_character_click,
                close_on_click_away,
                drive_sequence,
                draw_chat_ui,
                claim_input,
            )
                .chain(),
        );
    }
}

/// Where the chat panel sits: right side, vertically centred (logical px).
fn panel_rect(window: &Window) -> Rect {
    Rect::from_center_size(
        Vec2::new(window.width() - PANEL_W * 0.5 - 24.0, window.height() * 0.5),
        Vec2::new(PANEL_W, PANEL_H),
    )
}

/// Pull chat lines + thinking/listening state coming from the brain into local
/// state. Hearing "Hey Tomo" makes her hop and opens the chat.
fn ingest_brain_events(
    mut state: ResMut<ChatState>,
    mut chat: EventReader<ChatAppendEvent>,
    mut thinking: EventReader<ThinkingEvent>,
    mut listening: EventReader<ListeningEvent>,
    mut loco_q: Query<&mut Locomotion, With<Character>>,
) {
    for ListeningEvent(on) in listening.read() {
        state.listening = *on;
        if let (true, Ok(mut loco)) = (*on, loco_q.get_single_mut()) {
            loco.jump();
            if state.phase == Phase::Closed {
                start_opening(&mut state, &mut loco);
            }
        }
    }
    for ChatAppendEvent(line) in chat.read() {
        state.transcript.push(line.clone());
        // Keep memory bounded; full history still lives in the DB.
        if state.transcript.len() > 400 {
            let drain = state.transcript.len() - 400;
            state.transcript.drain(0..drain);
        }
    }
    for ThinkingEvent(t) in thinking.read() {
        state.thinking = *t;
    }
}

/// Clicking the character starts the open sequence; clicking it again while
/// open closes the chat. (Dragging it is physics, not a click — movement.rs.)
fn toggle_on_character_click(
    mut clicks: EventReader<CharacterClicked>,
    mut state: ResMut<ChatState>,
    mut loco_q: Query<&mut Locomotion, With<Character>>,
) {
    for CharacterClicked in clicks.read() {
        match state.phase {
            Phase::Closed => {
                let Ok(mut loco) = loco_q.get_single_mut() else { continue };
                start_opening(&mut state, &mut loco);
            }
            Phase::Open => state.phase = Phase::Closing,
            _ => {}
        }
    }
}

/// The first step of opening: walk to the dock (the liquid grows once there).
fn start_opening(state: &mut ChatState, loco: &mut Locomotion) {
    state.phase = Phase::Sliding;
    loco.held = true;
    loco.walk_to(DOCK_FRACTION);
}

/// Close on Escape, or on a click away from both the chat and the character.
fn close_on_click_away(
    buttons: Res<ButtonInput<MouseButton>>,
    mut keys: EventReader<KeyboardInput>,
    windows: Query<&Window, With<PrimaryWindow>>,
    characters: Query<&Character>,
    mut state: ResMut<ChatState>,
) {
    // The key's meaning, not its code: virtual keyboards and remapped layouts
    // put other keys on Escape's physical code.
    let escape = keys
        .read()
        .any(|k| k.state.is_pressed() && k.logical_key == Key::Escape);
    if state.phase != Phase::Open {
        return;
    }
    let Ok(window) = windows.get_single() else { return };
    let clicked_away = buttons.just_pressed(MouseButton::Left)
        && window.cursor_position().is_some_and(|cursor| {
            let on_character = characters
                .iter()
                .any(|c| c.screen_rect.is_some_and(|r| r.contains(cursor)));
            !on_character && !panel_rect(window).contains(cursor)
        });
    if clicked_away || escape {
        state.phase = Phase::Closing;
    }
}

/// Advance the animation phases.
fn drive_sequence(
    time: Res<Time>,
    mut state: ResMut<ChatState>,
    mut busy: ResMut<Busy>,
    mut loco_q: Query<&mut Locomotion, With<Character>>,
) {
    let dt = time.delta_secs();
    busy.0 |= matches!(state.phase, Phase::Sliding | Phase::Liquid | Phase::Closing);
    match state.phase {
        Phase::Sliding => {
            if let Ok(loco) = loco_q.get_single() {
                if loco.is_idle() {
                    state.phase = Phase::Liquid;
                    state.liquid_t = 0.0;
                }
            }
        }
        Phase::Liquid => {
            state.liquid_t += dt / LIQUID_GROW_SECS;
            if state.liquid_t >= 1.0 {
                state.liquid_t = 1.0;
                state.phase = Phase::Open;
            }
        }
        Phase::Closing => {
            state.liquid_t -= dt / LIQUID_MELT_SECS;
            if state.liquid_t <= 0.0 {
                state.liquid_t = 0.0;
                state.phase = Phase::Closed;
                if let Ok(mut loco) = loco_q.get_single_mut() {
                    loco.held = false; // resume wandering
                }
            }
        }
        _ => {}
    }
}

/// Draw the liquid and, when open, the chat window.
fn draw_chat_ui(
    mut contexts: EguiContexts,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut state: ResMut<ChatState>,
    bridge: Res<Bridge>,
    characters: Query<&Character>,
    mut exit: EventWriter<AppExit>,
) {
    if matches!(state.phase, Phase::Closed) {
        return;
    }
    let Ok(window) = windows.get_single() else { return };
    let ctx = contexts.ctx_mut();

    // The liquid comes out of the character's middle, wherever it stands.
    let from = characters
        .get_single()
        .ok()
        .and_then(|c| c.screen_rect)
        .map(|r| r.center())
        .unwrap_or(Vec2::new(window.width() * DOCK_FRACTION, window.height() * 0.5));

    // --- the liquid, drawn on a background layer between char and panel ---
    let t = ease_out_cubic(state.liquid_t);
    draw_liquid(ctx, from, t, window);

    // --- the chat window, only once (mostly) formed ---
    if matches!(state.phase, Phase::Open) || state.liquid_t > 0.85 {
        let header = draw_window(ctx, &mut state, &bridge, t);
        if header.close {
            state.phase = Phase::Closing;
        }
        if header.quit {
            // There's no window to close in overlay mode; this is the way out.
            bridge.send(UiToBrain::Shutdown);
            exit.send(AppExit::Success);
        }
    }
}

/// Tell the overlay where the chat needs the mouse (the panel, while shown)
/// and that it wants the keyboard while open, so the text box can be typed in.
fn claim_input(
    state: Res<ChatState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut region: ResMut<InputRegion>,
) {
    let Ok(window) = windows.get_single() else { return };
    let shown = matches!(state.phase, Phase::Liquid | Phase::Open | Phase::Closing);
    region.chat = shown.then(|| pixel_rect(panel_rect(window)));
    region.keyboard = state.phase == Phase::Open;
}

/// A blob of merging circles that eases from the character into a panel shape.
fn draw_liquid(ctx: &egui::Context, from: Vec2, t: f32, window: &Window) {
    use egui::{Color32, Pos2, Shape};
    let layer = egui::LayerId::new(egui::Order::Background, egui::Id::new("tomo-liquid"));
    let painter = ctx.layer_painter(layer);

    let center = panel_rect(window).center();
    let target = Pos2::new(center.x, center.y);
    let src = Pos2::new(from.x, from.y);
    let center = src.lerp(target, t);

    // A soft, glossy teal that reads on any wallpaper.
    let fill = Color32::from_rgba_unmultiplied(64, 196, 208, (220.0 * t) as u8);

    // Metaball-ish: a few circles that spread as t grows, approximating a
    // droplet stretching into a rounded rectangle.
    // TODO(fluid): replace with an SDF metaball shader for true surface tension.
    let spread = 8.0 + 150.0 * t;
    let r = 14.0 + 60.0 * t;
    for i in 0..5 {
        let f = i as f32 / 4.0 - 0.5;
        let p = Pos2::new(center.x + f * spread * 0.4, center.y + f * spread);
        painter.add(Shape::circle_filled(p, r, fill));
    }
    // Once nearly formed, lay a rounded rect so the edge is clean under egui.
    if t > 0.6 {
        let a = ((t - 0.6) / 0.4).clamp(0.0, 1.0);
        let rect = egui::Rect::from_center_size(target, egui::vec2(PANEL_W, PANEL_H));
        painter.add(Shape::rect_filled(
            rect,
            egui::Rounding::same(24.0),
            Color32::from_rgba_unmultiplied(20, 28, 34, (235.0 * a) as u8),
        ));
    }
}

/// Header buttons the user clicked this frame.
#[derive(Default)]
struct HeaderClicks {
    close: bool,
    quit: bool,
}

/// The actual chat panel.
fn draw_window(
    ctx: &egui::Context,
    state: &mut ChatState,
    bridge: &Bridge,
    t: f32,
) -> HeaderClicks {
    use egui::{Align, Color32, Layout, RichText};

    let mut send_text: Option<String> = None;
    let mut start_voice = false;
    let mut header = HeaderClicks::default();

    egui::Window::new("tomo-chat")
        .title_bar(false)
        .resizable(false)
        .fixed_size(egui::vec2(PANEL_W - 24.0, PANEL_H - 24.0))
        .anchor(egui::Align2::RIGHT_CENTER, egui::vec2(-24.0, 0.0))
        .frame(
            egui::Frame::none()
                .fill(Color32::from_rgba_unmultiplied(20, 28, 34, (245.0 * t) as u8))
                .rounding(20.0)
                .inner_margin(egui::Margin::same(14.0)),
        )
        .show(ctx, |ui| {
            // Header
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Tomo")
                        .color(Color32::from_rgb(120, 230, 240))
                        .strong()
                        .size(18.0),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    header.close = ui.button("×").on_hover_text("Close chat").clicked();
                    header.quit = ui.button("Quit").on_hover_text("Quit Tomo").clicked();
                    ui.checkbox(&mut state.voice_replies, "🔊");
                });
            });
            ui.separator();

            // Transcript
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .max_height(PANEL_H - 140.0)
                .show(ui, |ui| {
                    for line in &state.transcript {
                        chat_bubble(ui, line);
                    }
                    if state.listening {
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new("Listening…")
                                .italics()
                                .color(Color32::from_rgb(120, 230, 240)),
                        );
                    }
                    if state.thinking {
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new("Tomo is thinking…")
                                .italics()
                                .color(Color32::from_gray(150)),
                        );
                    }
                });

            ui.separator();

            // Input row
            ui.horizontal(|ui| {
                let hint = "Say something…";
                let edit = egui::TextEdit::singleline(&mut state.input)
                    .hint_text(hint)
                    .desired_width(PANEL_W - 150.0);
                let resp = ui.add(edit);
                let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                if ui.button("Send").clicked() || entered {
                    let text = state.input.trim().to_string();
                    if !text.is_empty() {
                        send_text = Some(text);
                        state.input.clear();
                    }
                    resp.request_focus();
                }
                if ui
                    .button("Talk")
                    .on_hover_text("Push to talk (or just say \"Hey Tomo\")")
                    .clicked()
                {
                    start_voice = true;
                }
            });
        });

    if let Some(text) = send_text {
        // Optimistic local echo is skipped: the brain echoes the user line back
        // (single source of truth), so we only send.
        bridge.send(UiToBrain::UserMessage(text));
    }
    if start_voice {
        bridge.send(UiToBrain::StartVoiceInput);
    }
    header
}

fn chat_bubble(ui: &mut egui::Ui, line: &ChatLine) {
    use egui::{Align, Color32, Layout, RichText};
    let (align, color, prefix) = match line.role {
        Role::User => (Align::RIGHT, Color32::from_rgb(70, 130, 180), "You"),
        Role::Assistant => (Align::LEFT, Color32::from_rgb(38, 66, 74), "Tomo"),
        Role::System => (Align::Center, Color32::from_gray(60), "•"),
    };
    ui.with_layout(Layout::top_down(align), |ui| {
        egui::Frame::none()
            .fill(color)
            .rounding(12.0)
            .inner_margin(egui::Margin::symmetric(10.0, 6.0))
            .show(ui, |ui| {
                ui.label(RichText::new(&line.text).color(Color32::WHITE));
            });
        ui.small(RichText::new(prefix).color(Color32::from_gray(120)));
        ui.add_space(6.0);
    });
}

/// Cubic ease-out for a natural "settle".
fn ease_out_cubic(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    1.0 - (1.0 - x).powi(3)
}
