//! The surface Tomo lives on.
//!
//! The character needs a transparent surface covering the desktop, above the
//! other windows, so it can roam anywhere. main.rs picks one of two:
//!
//!   • The overlay (Wayland with layer-shell: Hyprland, Sway, KDE Plasma, …) —
//!     see overlay.rs. Not a window at all: a layer-shell surface above
//!     everything, letting clicks through except where [`InputRegion`] says.
//!   • A regular window (X11, GNOME Wayland) — built here: borderless,
//!     transparent and always-on-top, as far as each compositor allows:
//!
//! ┌───────────────────────────────────────────────────────────────────────┐
//! │ transparency        X11: ✔ works via a compositor (picom/kwin/mutter). │
//! │                     Wayland: ✔ (pre-multiplied alpha, see below).       │
//! │ always-on-top       X11: ✔ WindowLevel::AlwaysOnTop → _NET_WM_STATE.    │
//! │                     GNOME Wayland: ✘ Mutter ignores it.                 │
//! │ skip taskbar/pager  X11: ✔ skip_taskbar. Wayland: no winit support yet. │
//! │ click-through       ✘ the whole window catches the mouse — see          │
//! │                     `TODO(input-region)` below.                         │
//! └───────────────────────────────────────────────────────────────────────┘

use bevy::prelude::*;
use bevy::window::{CompositeAlphaMode, PrimaryWindow, WindowLevel, WindowResolution};

use crate::session::{DisplayServer, Session};

/// The parts of the screen Tomo wants the mouse on (logical pixels, origin
/// top-left) — the character, and the chat while it's open — and whether the
/// chat wants the keyboard. Everywhere else, clicks fall through to the
/// desktop. Only the overlay can honour this; a regular window catches the
/// mouse everywhere.
#[derive(Resource, Default, Clone, PartialEq, Eq)]
pub struct InputRegion {
    pub character: Option<IRect>,
    pub chat: Option<IRect>,
    pub keyboard: bool,
}

/// Set during a frame by anything moving or animating (walking, falling, the
/// chat unfolding, talking…). When nothing is, the overlay drops to a low frame
/// rate to save power (overlay.rs). Cleared at the start of every frame.
#[derive(Resource, Default)]
pub struct Busy(pub bool);

/// Round a rect outward to whole pixels.
pub fn pixel_rect(rect: Rect) -> IRect {
    IRect::from_corners(rect.min.floor().as_ivec2(), rect.max.ceil().as_ivec2())
}

/// Build the primary window description. In overlay mode only the rendering
/// settings (alpha mode, present mode) matter; the layer surface replaces the
/// rest.
///
/// `session` decides the alpha mode.
pub fn overlay_window(session: &Session) -> Window {
    // Pre-multiplied alpha is what both X11 compositors and Wayland need. Do
    // NOT use `Auto`: wgpu resolves it to `Opaque` whenever the surface offers
    // that (it always does), and the transparent clear colour then shows up as
    // a black box instead of transparency — the classic first bug on a new
    // setup.
    let composite_alpha_mode = match session.server {
        DisplayServer::X11 | DisplayServer::Wayland => CompositeAlphaMode::PreMultiplied,
        DisplayServer::Unknown => CompositeAlphaMode::Auto,
    };

    Window {
        title: "Tomo".into(),
        // `name` becomes the Wayland app_id / X11 WM_CLASS — lets users target
        // the window in compositor rules (e.g. an i3/Hyprland `float` rule).
        name: Some("tomo.desktop.companion".into()),
        transparent: true,
        decorations: false,
        resizable: false,
        focused: false,
        // Float above normal windows.
        window_level: WindowLevel::AlwaysOnTop,
        // Keep Tomo out of the taskbar/alt-tab (X11 + Windows honour this).
        skip_taskbar: true,
        composite_alpha_mode,
        // A large canvas the character can roam. `main.rs` resizes this to the
        // real monitor once winit reports it (see `fit_to_primary_monitor`).
        resolution: WindowResolution::new(1280.0, 800.0),
        ..default()
    }
}

/// Which winit backend to prefer. Kept as the single decision point so a
/// layer-shell implementation can slot in later (see module docs).
pub fn recommended_backend(session: &Session) -> &'static str {
    match session.server {
        DisplayServer::Wayland => "wayland",
        DisplayServer::X11 => "x11",
        DisplayServer::Unknown => "auto",
    }
}

/// Log the caveats for the detected environment so a first-time user on, say,
/// GNOME Wayland understands why always-on-top may misbehave — instead of
/// thinking the app is broken.
pub fn log_environment_notes(session: &Session) {
    info!("Detected session: {}", session.describe());
    match session.server {
        DisplayServer::X11 => {
            info!("X11: make sure a compositor is running (picom/compton) or \
                   transparency will render as solid black.");
        }
        DisplayServer::Wayland => {
            if matches!(session.desktop, crate::session::Desktop::Gnome) {
                warn!("GNOME/Mutter (Wayland) ignores always-on-top and \
                       skip-taskbar, and has no wlr-layer-shell. Tomo will run \
                       but may sit among normal windows. wlroots compositors \
                       (Hyprland, Sway) and KDE float it above everything.");
            }
        }
        DisplayServer::Unknown => {
            warn!("Could not detect the display server; using winit defaults.");
        }
    }
}

/// Resize the window to fill the primary monitor once winit knows its size.
/// Runs a few frames after startup (monitor info isn't ready at frame 0).
pub fn fit_to_primary_monitor(
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    monitors: Query<&bevy::window::Monitor>,
) {
    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };
    if let Some(monitor) = monitors.iter().next() {
        let w = monitor.physical_width as f32 / window.resolution.scale_factor();
        let h = monitor.physical_height as f32 / window.resolution.scale_factor();
        if w > 0.0 && h > 0.0 {
            window.resolution.set(w, h);
            // Anchor to the top-left so screen-fraction math in `movement.rs`
            // maps cleanly onto the whole desktop.
            window.position = WindowPosition::At(IVec2::ZERO);
        }
    }
}

// TODO(input-region): make clicks pass through a *regular window* everywhere
// except the character, as the overlay already does from [`InputRegion`].
//
// winit exposes only whole-window `set_cursor_hittest(bool)`. We want a shaped
// input region:
//
//   X11  — use x11rb/xcb to set the XShape *input* region to the character's
//          screen-space bounding box each frame:
//              xcb_shape_rectangles(conn, SO::Set, SK::Input, ClipOrdering::Unsorted,
//                                   window, 0, 0, &[char_bbox]);
//   GNOME Wayland — call wl_surface.set_input_region on winit's surface, via
//          raw-window-handle + wayland-client (the overlay's apply_requests
//          shows the calls).
//
// Until then, set `cursor_options.hit_test = false` for a fully click-through
// (but unclickable) character, or leave it `true` to make the character
// clickable at the cost of the transparent area also swallowing clicks.
