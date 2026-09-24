//! The surface Tomo lives on.
//!
//! The character needs a transparent surface covering the desktop, above the
//! other windows, so it can roam anywhere. main.rs picks one of two:
//!
//!   • The overlay (Wayland with layer-shell: Hyprland, Sway, KDE Plasma, …) —
//!     see overlay.rs. Not a window at all: a layer-shell surface above
//!     everything, letting clicks through except where [`InputRegion`] says.
//!   • A regular window (Windows, X11, GNOME Wayland) — built here:
//!     borderless, transparent and always-on-top, as far as each system allows:
//!
//! ┌───────────────────────────────────────────────────────────────────────┐
//! │ transparency        Windows: ✔ through DWM (winit).                    │
//! │                     X11: ✔ works via a compositor (picom/kwin/mutter). │
//! │                     Wayland: ✔ (pre-multiplied alpha, see below).       │
//! │ always-on-top       Windows, X11: ✔ WindowLevel::AlwaysOnTop.            │
//! │                     GNOME Wayland: ✘ Mutter ignores it.                 │
//! │ skip taskbar        Windows, X11: ✔ skip_taskbar.                        │
//! │ click-through       Windows: ✔ hit-testing follows the cursor, on only  │
//! │                     over the character and the chat (see `windows`).   │
//! │                     X11/GNOME: ✘ — see `TODO(input-region)` below.      │
//! └───────────────────────────────────────────────────────────────────────┘

use bevy::prelude::*;
use bevy::window::{CompositeAlphaMode, PrimaryWindow, WindowLevel, WindowResolution};

use crate::session::{DisplayServer, Session};

/// The parts of the screen Tomo wants the mouse on (logical pixels, origin
/// top-left) — the character, and the chat while it's open — and whether the
/// chat wants the keyboard. Everywhere else, clicks fall through to the
/// desktop. The overlay and Windows honour this; elsewhere a regular window
/// catches the mouse everywhere.
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
    //
    // Windows is the exception: its GPU drivers mostly offer only `Opaque`,
    // and DWM makes the window see-through by its alpha anyway (winit
    // enables that), so `Auto` there. macOS composites post-multiplied.
    let composite_alpha_mode = match session.server {
        DisplayServer::X11 | DisplayServer::Wayland => CompositeAlphaMode::PreMultiplied,
        DisplayServer::MacOs => CompositeAlphaMode::PostMultiplied,
        DisplayServer::Windows | DisplayServer::Unknown => CompositeAlphaMode::Auto,
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
        DisplayServer::Windows | DisplayServer::MacOs | DisplayServer::Unknown => "auto",
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
        DisplayServer::Windows => {
            info!("Windows: if Tomo's background shows black instead of the desktop, \
                   set TOMO_RENDERER=gl (or dx12) in .env, or use the \
                   \"Tomo (compatibility)\" shortcut.");
        }
        DisplayServer::MacOs => {}
        DisplayServer::Unknown => {
            warn!("Could not detect the display server; using winit defaults.");
        }
    }
}

/// In a regular window, keep Bevy's winit loop as lazy as the overlay's:
/// at the display's rate while something moves, 24 fps at rest.
pub fn pace_frames(busy: Res<Busy>, mut settings: ResMut<bevy::winit::WinitSettings>) {
    use bevy::winit::UpdateMode;
    let mode = if busy.0 {
        UpdateMode::Continuous
    } else {
        UpdateMode::reactive(std::time::Duration::from_micros(41_667))
    };
    if settings.focused_mode != mode {
        settings.focused_mode = mode;
        settings.unfocused_mode = mode;
    }
}

/// Windows: fill the work area (the screen minus the taskbar, so the taskbar
/// is her floor), and let clicks through everywhere but the character and
/// the chat.
#[cfg(windows)]
pub mod windows {
    use bevy::prelude::*;
    use bevy::window::{PrimaryWindow, WindowPosition};
    use windows_sys::Win32::Foundation::{POINT, RECT};
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetCursorPos, SystemParametersInfoW, SPI_GETWORKAREA};

    use super::InputRegion;

    /// The primary monitor's work area, in physical pixels.
    fn work_area() -> Option<RECT> {
        let mut area = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        // SAFETY: SPI_GETWORKAREA writes one RECT to the pointer given.
        let ok = unsafe { SystemParametersInfoW(SPI_GETWORKAREA, 0, (&mut area as *mut RECT).cast(), 0) };
        (ok != 0 && area.right > area.left && area.bottom > area.top).then_some(area)
    }

    /// Place and size the window over the work area, once.
    pub fn fit_to_work_area(mut done: Local<bool>, mut windows: Query<&mut Window, With<PrimaryWindow>>) {
        if *done {
            return;
        }
        let (Ok(mut window), Some(area)) = (windows.get_single_mut(), work_area()) else { return };
        window.position = WindowPosition::At(IVec2::new(area.left, area.top));
        window
            .resolution
            .set_physical_resolution((area.right - area.left) as u32, (area.bottom - area.top) as u32);
        *done = true;
    }

    /// Turn the window's hit-testing on only while the cursor is over the
    /// character or the chat: everywhere else, clicks go to the desktop.
    pub fn click_through(mut windows: Query<&mut Window, With<PrimaryWindow>>, region: Res<InputRegion>) {
        let Ok(mut window) = windows.get_single_mut() else { return };
        let mut cursor = POINT { x: 0, y: 0 };
        // SAFETY: GetCursorPos writes one POINT to the pointer given.
        if unsafe { GetCursorPos(&mut cursor) } == 0 {
            return;
        }
        let origin = match window.position {
            WindowPosition::At(position) => position.as_vec2(),
            _ => Vec2::ZERO,
        };
        let at = (Vec2::new(cursor.x as f32, cursor.y as f32) - origin) / window.resolution.scale_factor();
        let over = [region.character, region.chat]
            .into_iter()
            .flatten()
            .any(|rect| rect.as_rect().contains(at));
        if window.cursor_options.hit_test != over {
            window.cursor_options.hit_test = over;
        }
    }
}

/// Resize the window to fill the primary monitor once winit knows its size
/// (on Windows, [`windows::fit_to_work_area`] does it instead).
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
