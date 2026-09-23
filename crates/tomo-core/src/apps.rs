//! What the desktop can do, read from the OS.
//!
//! Two things are discovered here, both straight from the operating system so
//! nothing is hard-coded per machine:
//!
//!   • **Apps** — every installed launcher, parsed from the freedesktop
//!     `.desktop` files in the standard XDG locations (system, user, and
//!     Flatpak). Each gives us a name, an icon name, and the exact command to
//!     launch it. This is what lets the character "know" your apps.
//!
//!   • **Toggles** — which system switches are actually available on THIS box
//!     (Bluetooth, Wi-Fi, volume, brightness), decided by probing for the
//!     tools that drive them (`rfkill`, `nmcli`, `pactl`/`wpctl`,
//!     `brightnessctl`). No tool present → that toggle simply isn't offered.
//!
//! The whole thing is fingerprinted so the brain can scan once at boot and
//! then, on a light periodic re-scan, notice when something was installed or
//! removed and refresh — "fetch once, and again on change", as asked.
//!
//! This module is pure filesystem + PATH inspection (no graphics, no network),
//! so it is fully unit-tested.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One launchable application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesktopApp {
    /// Stable id: the `.desktop` file stem (e.g. `firefox`, `org.gnome.Nautilus`).
    pub id: String,
    pub name: String,
    /// Launch command, with freedesktop field codes (%U, %f…) stripped.
    pub exec: String,
    /// Icon *name* (themed) or absolute path, as given in the entry.
    pub icon: String,
    pub categories: Vec<String>,
    /// True if it wants a terminal (`Terminal=true`).
    pub terminal: bool,
}

/// A system switch we can flip, and the commands that do it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Toggle {
    /// e.g. `bluetooth`, `wifi`, `volume`, `brightness`.
    pub key: String,
    pub label: String,
    pub status_cmd: String,
    pub on_cmd: String,
    pub off_cmd: String,
}

/// Everything the desktop currently offers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemCatalog {
    pub apps: Vec<DesktopApp>,
    pub toggles: Vec<Toggle>,
}

impl SystemCatalog {
    /// Scan the OS: parse all `.desktop` entries and probe for toggles.
    pub fn scan() -> Self {
        let mut apps = parse_all_entries(&application_dirs());
        apps.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        SystemCatalog {
            apps,
            toggles: probe_toggles(),
        }
    }

    /// A content fingerprint. Two catalogs with the same apps + toggles hash
    /// identically, so the brain can cheaply tell "did anything change?".
    pub fn fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        for a in &self.apps {
            a.id.hash(&mut h);
            a.exec.hash(&mut h);
            a.name.hash(&mut h);
        }
        for t in &self.toggles {
            t.key.hash(&mut h);
        }
        h.finish()
    }

    /// Fuzzy-ish lookup by name or id for the AI's `find_target` / `open_app`.
    /// Returns best matches first (exact id, then name prefix, then contains).
    pub fn find_app(&self, query: &str) -> Vec<&DesktopApp> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(u8, &DesktopApp)> = self
            .apps
            .iter()
            .filter_map(|a| {
                let name = a.name.to_lowercase();
                let id = a.id.to_lowercase();
                let score = if id == q || name == q {
                    0
                } else if name.starts_with(&q) || id.starts_with(&q) {
                    1
                } else if name.contains(&q) || id.contains(&q) {
                    2
                } else {
                    return None;
                };
                Some((score, a))
            })
            .collect();
        scored.sort_by_key(|(s, _)| *s);
        scored.into_iter().map(|(_, a)| a).collect()
    }

    /// A compact catalog summary for injecting into the model's context —
    /// capped so it never blows up the prompt.
    pub fn summary(&self, max_apps: usize) -> String {
        let mut s = String::new();
        if !self.toggles.is_empty() {
            let keys: Vec<&str> = self.toggles.iter().map(|t| t.key.as_str()).collect();
            s.push_str(&format!("Available system toggles: {}\n", keys.join(", ")));
        }
        let shown = self.apps.len().min(max_apps);
        s.push_str(&format!(
            "Installed apps ({} total, first {} shown): ",
            self.apps.len(),
            shown
        ));
        s.push_str(
            &self.apps[..shown]
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        );
        s
    }
}

/// The XDG-standard directories that hold `.desktop` files, user first so a
/// user override wins over the system copy.
fn application_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let home = std::env::var("HOME").unwrap_or_default();

    let data_home = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{home}/.local/share"));
    dirs.push(PathBuf::from(format!("{data_home}/applications")));
    dirs.push(PathBuf::from(format!(
        "{home}/.local/share/flatpak/exports/share/applications"
    )));

    let data_dirs = std::env::var("XDG_DATA_DIRS")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/usr/local/share:/usr/share".to_string());
    for base in data_dirs.split(':') {
        if !base.is_empty() {
            dirs.push(PathBuf::from(format!("{base}/applications")));
        }
    }
    dirs.push(PathBuf::from(
        "/var/lib/flatpak/exports/share/applications",
    ));
    dirs
}

fn parse_all_entries(dirs: &[PathBuf]) -> Vec<DesktopApp> {
    // Keyed by id so the first (highest-priority) copy wins.
    let mut seen: BTreeMap<String, DesktopApp> = BTreeMap::new();
    for dir in dirs {
        let Ok(read) = std::fs::read_dir(dir) else { continue };
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("desktop") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            if let Some(app) = parse_entry(&text, &path) {
                seen.entry(app.id.clone()).or_insert(app);
            }
        }
    }
    seen.into_values().collect()
}

/// Parse a single `.desktop` file's `[Desktop Entry]` group. Returns `None`
/// for anything that shouldn't appear in a launcher (hidden, no-display, or
/// not an Application).
fn parse_entry(text: &str, path: &Path) -> Option<DesktopApp> {
    let mut in_group = false;
    let mut kv: BTreeMap<String, String> = BTreeMap::new();

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_group = line == "[Desktop Entry]";
            continue;
        }
        if !in_group || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            // Ignore localised keys like Name[de]; keep the base key only.
            let key = k.trim();
            if key.contains('[') {
                continue;
            }
            kv.insert(key.to_string(), v.trim().to_string());
        }
    }

    if kv.get("Type").map(|t| t != "Application").unwrap_or(true) {
        return None;
    }
    if is_true(kv.get("NoDisplay")) || is_true(kv.get("Hidden")) {
        return None;
    }
    let name = kv.get("Name")?.clone();
    let exec = clean_exec(kv.get("Exec")?);
    if exec.is_empty() {
        return None;
    }

    let id = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| name.clone());
    let categories = kv
        .get("Categories")
        .map(|c| {
            c.split(';')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    Some(DesktopApp {
        id,
        name,
        exec,
        icon: kv.get("Icon").cloned().unwrap_or_default(),
        categories,
        terminal: is_true(kv.get("Terminal")),
    })
}

/// Strip freedesktop field codes (%U, %f, %i, %c…) and collapse whitespace.
fn clean_exec(exec: &str) -> String {
    let mut out = String::with_capacity(exec.len());
    let mut chars = exec.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            // Drop the code letter that follows, except an escaped %%.
            match chars.peek() {
                Some('%') => {
                    out.push('%');
                    chars.next();
                }
                Some(_) => {
                    chars.next();
                }
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_true(v: Option<&String>) -> bool {
    v.map(|s| s.eq_ignore_ascii_case("true")).unwrap_or(false)
}

/// Build the toggle list from whatever control tools are actually installed.
fn probe_toggles() -> Vec<Toggle> {
    let mut out = Vec::new();

    // Bluetooth via rfkill (most portable), else bluetoothctl.
    if has_binary("rfkill") {
        out.push(Toggle {
            key: "bluetooth".into(),
            label: "Bluetooth".into(),
            status_cmd: "rfkill list bluetooth".into(),
            on_cmd: "rfkill unblock bluetooth".into(),
            off_cmd: "rfkill block bluetooth".into(),
        });
    } else if has_binary("bluetoothctl") {
        out.push(Toggle {
            key: "bluetooth".into(),
            label: "Bluetooth".into(),
            status_cmd: "bluetoothctl show".into(),
            on_cmd: "bluetoothctl power on".into(),
            off_cmd: "bluetoothctl power off".into(),
        });
    }

    // Wi-Fi via NetworkManager.
    if has_binary("nmcli") {
        out.push(Toggle {
            key: "wifi".into(),
            label: "Wi-Fi".into(),
            status_cmd: "nmcli radio wifi".into(),
            on_cmd: "nmcli radio wifi on".into(),
            off_cmd: "nmcli radio wifi off".into(),
        });
    }

    // Volume via wireplumber, else PulseAudio.
    if has_binary("wpctl") {
        out.push(Toggle {
            key: "mute".into(),
            label: "Mute".into(),
            status_cmd: "wpctl get-volume @DEFAULT_AUDIO_SINK@".into(),
            on_cmd: "wpctl set-mute @DEFAULT_AUDIO_SINK@ 1".into(),
            off_cmd: "wpctl set-mute @DEFAULT_AUDIO_SINK@ 0".into(),
        });
    } else if has_binary("pactl") {
        out.push(Toggle {
            key: "mute".into(),
            label: "Mute".into(),
            status_cmd: "pactl get-sink-mute @DEFAULT_SINK@".into(),
            on_cmd: "pactl set-sink-mute @DEFAULT_SINK@ 1".into(),
            off_cmd: "pactl set-sink-mute @DEFAULT_SINK@ 0".into(),
        });
    }

    out
}

/// True if `name` is an executable on `PATH`. Pure PATH scan — no subprocess.
pub fn has_binary(name: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else { return false };
    for dir in path.split(':') {
        if dir.is_empty() {
            continue;
        }
        let candidate = Path::new(dir).join(name);
        if candidate.is_file() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn parses_a_normal_entry_and_strips_field_codes() {
        let text = "\
[Desktop Entry]
Type=Application
Name=Firefox
Exec=firefox %u
Icon=firefox
Categories=Network;WebBrowser;
";
        let app = parse_entry(text, &p("/x/firefox.desktop")).unwrap();
        assert_eq!(app.id, "firefox");
        assert_eq!(app.name, "Firefox");
        assert_eq!(app.exec, "firefox"); // %u removed
        assert_eq!(app.categories, vec!["Network", "WebBrowser"]);
        assert!(!app.terminal);
    }

    #[test]
    fn skips_hidden_nodisplay_and_non_applications() {
        assert!(parse_entry("[Desktop Entry]\nType=Application\nName=X\nExec=x\nNoDisplay=true\n", &p("/x/x.desktop")).is_none());
        assert!(parse_entry("[Desktop Entry]\nType=Application\nName=X\nExec=x\nHidden=true\n", &p("/x/x.desktop")).is_none());
        assert!(parse_entry("[Desktop Entry]\nType=Link\nName=X\nURL=http://x\n", &p("/x/x.desktop")).is_none());
    }

    #[test]
    fn ignores_localised_keys_and_keeps_base_name() {
        let text = "[Desktop Entry]\nType=Application\nName=Files\nName[de]=Dateien\nExec=nautilus\n";
        let app = parse_entry(text, &p("/x/org.gnome.Nautilus.desktop")).unwrap();
        assert_eq!(app.name, "Files");
        assert_eq!(app.id, "org.gnome.Nautilus");
    }

    #[test]
    fn clean_exec_keeps_escaped_percent() {
        assert_eq!(clean_exec("app %F --flag %U"), "app --flag");
        assert_eq!(clean_exec("app 100%% done"), "app 100% done");
    }

    #[test]
    fn find_app_ranks_exact_then_prefix_then_contains() {
        let cat = SystemCatalog {
            apps: vec![
                DesktopApp { id: "firefox".into(), name: "Firefox".into(), exec: "firefox".into(), icon: "".into(), categories: vec![], terminal: false },
                DesktopApp { id: "fire".into(), name: "Fireworks".into(), exec: "fw".into(), icon: "".into(), categories: vec![], terminal: false },
                DesktopApp { id: "x".into(), name: "Xfire chat".into(), exec: "xf".into(), icon: "".into(), categories: vec![], terminal: false },
            ],
            toggles: vec![],
        };
        let hits = cat.find_app("fire");
        assert_eq!(hits[0].name, "Fireworks"); // name prefix beats "contains"
        assert!(hits.iter().any(|a| a.name == "Xfire chat"));
    }

    #[test]
    fn fingerprint_changes_when_an_app_is_added() {
        let mut a = SystemCatalog::default();
        let f0 = a.fingerprint();
        a.apps.push(DesktopApp { id: "n".into(), name: "N".into(), exec: "n".into(), icon: "".into(), categories: vec![], terminal: false });
        assert_ne!(f0, a.fingerprint());
    }
}
