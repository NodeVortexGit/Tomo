//! Desktop-session detection.
//!
//! Tomo has to behave on a spread of environments — KDE, XFCE, Cinnamon,
//! GNOME, Hyprland and i3, over both Wayland and X11. The *windowing tricks*
//! needed for a floating desktop character (transparency, always-on-top,
//! click-through, skip-taskbar) are requested through winit the same way
//! everywhere, but how well each is honoured depends on the compositor. This
//! module figures out where we're running so [`crate::window`] can pick the
//! best strategy and log a clear note when a compositor is known to need a
//! workaround.
//!
//! This is ordinary environment sniffing — no graphics — so it is fully
//! working and unit-tested.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayServer {
    Wayland,
    X11,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Desktop {
    Kde,
    Gnome,
    Xfce,
    Cinnamon,
    Hyprland,
    I3,
    Sway,
    Other,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub server: DisplayServer,
    pub desktop: Desktop,
    /// Raw value of XDG_CURRENT_DESKTOP, kept for logging.
    pub raw_desktop: String,
}

impl Session {
    /// Detect from the current process environment.
    pub fn detect() -> Self {
        let server = detect_server(
            std::env::var("XDG_SESSION_TYPE").ok().as_deref(),
            std::env::var("WAYLAND_DISPLAY").ok().as_deref(),
            std::env::var("DISPLAY").ok().as_deref(),
        );
        let raw_desktop = std::env::var("XDG_CURRENT_DESKTOP")
            .or_else(|_| std::env::var("DESKTOP_SESSION"))
            .unwrap_or_default();
        let desktop = detect_desktop(&raw_desktop);
        Session {
            server,
            desktop,
            raw_desktop,
        }
    }

    /// True when we're on a compositor where winit can't, on its own, place a
    /// true always-on-top desktop overlay, so we rely on extra hints /
    /// documented manual rules (see window.rs).
    pub fn needs_layer_shell(&self) -> bool {
        // On Wayland, only wlr-layer-shell gives real overlay semantics.
        // GNOME's Mutter doesn't implement it at all; wlroots compositors
        // (Hyprland, Sway) do.
        self.server == DisplayServer::Wayland
    }

    /// A one-line human summary for the log.
    pub fn describe(&self) -> String {
        format!("{} on {}", self.desktop, self.server)
    }
}

fn detect_server(
    session_type: Option<&str>,
    wayland_display: Option<&str>,
    x_display: Option<&str>,
) -> DisplayServer {
    if let Some(t) = session_type {
        match t.to_ascii_lowercase().as_str() {
            "wayland" => return DisplayServer::Wayland,
            "x11" => return DisplayServer::X11,
            _ => {}
        }
    }
    if wayland_display.map(|s| !s.is_empty()).unwrap_or(false) {
        return DisplayServer::Wayland;
    }
    if x_display.map(|s| !s.is_empty()).unwrap_or(false) {
        return DisplayServer::X11;
    }
    DisplayServer::Unknown
}

fn detect_desktop(raw: &str) -> Desktop {
    let r = raw.to_ascii_lowercase();
    // XDG_CURRENT_DESKTOP can be colon-separated (e.g. "ubuntu:GNOME").
    if r.contains("hyprland") {
        Desktop::Hyprland
    } else if r.contains("sway") {
        Desktop::Sway
    } else if r.contains("kde") || r.contains("plasma") {
        Desktop::Kde
    } else if r.contains("gnome") {
        Desktop::Gnome
    } else if r.contains("xfce") {
        Desktop::Xfce
    } else if r.contains("cinnamon") || r.contains("x-cinnamon") {
        Desktop::Cinnamon
    } else if r.contains("i3") {
        Desktop::I3
    } else {
        Desktop::Other
    }
}

impl fmt::Display for DisplayServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DisplayServer::Wayland => "Wayland",
            DisplayServer::X11 => "X11",
            DisplayServer::Unknown => "unknown display server",
        };
        f.write_str(s)
    }
}

impl fmt::Display for Desktop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Desktop::Kde => "KDE Plasma",
            Desktop::Gnome => "GNOME",
            Desktop::Xfce => "XFCE",
            Desktop::Cinnamon => "Cinnamon",
            Desktop::Hyprland => "Hyprland",
            Desktop::I3 => "i3",
            Desktop::Sway => "Sway",
            Desktop::Other => "other desktop",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_prefers_explicit_session_type() {
        assert_eq!(
            detect_server(Some("wayland"), None, Some(":0")),
            DisplayServer::Wayland
        );
        assert_eq!(
            detect_server(Some("x11"), Some("wayland-0"), None),
            DisplayServer::X11
        );
    }

    #[test]
    fn server_falls_back_to_display_vars() {
        assert_eq!(
            detect_server(None, Some("wayland-0"), None),
            DisplayServer::Wayland
        );
        assert_eq!(detect_server(None, None, Some(":0")), DisplayServer::X11);
        assert_eq!(detect_server(None, None, None), DisplayServer::Unknown);
    }

    #[test]
    fn desktop_parsing_handles_composite_values() {
        assert_eq!(detect_desktop("ubuntu:GNOME"), Desktop::Gnome);
        assert_eq!(detect_desktop("KDE"), Desktop::Kde);
        assert_eq!(detect_desktop("X-Cinnamon"), Desktop::Cinnamon);
        assert_eq!(detect_desktop("Hyprland"), Desktop::Hyprland);
        assert_eq!(detect_desktop("i3"), Desktop::I3);
        assert_eq!(detect_desktop("whatever"), Desktop::Other);
    }
}
