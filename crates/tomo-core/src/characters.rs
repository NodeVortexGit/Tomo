//! Which characters (`.vrm` models) Tomo can appear as, and switching
//! between them.
//!
//! The choices are every `.vrm` in the configured character's folder, the
//! data dir's `characters/` and the bundled `assets/characters/`, plus any
//! imported file the database knows.
//! Switching — from the chat's character menu, or by asking Tomo (the
//! `change_character` tool) — remembers the choice for the next start. A file
//! from anywhere else is copied into the data dir first, so it keeps working
//! if the original is moved or deleted.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::config::Config;
use crate::db::Db;
use crate::events::CharacterChoice;

/// The character to show at start: the last one chosen, else the configured
/// one, else the first on offer (a fresh install's bundled one) — whichever
/// exists. Canonical, so the app's asset loader takes it as is rather than
/// resolving it against its own asset folder.
pub(crate) fn startup(db: &Db, cfg: &Config) -> Option<PathBuf> {
    let chosen = db.active_character().ok().flatten().map(|c| PathBuf::from(c.path));
    chosen
        .into_iter()
        .chain(std::iter::once(cfg.character_path.clone()))
        .find_map(|p| p.canonicalize().ok())
        .or_else(|| available(db, cfg).into_iter().next().map(|c| c.path))
}

/// Every character on offer, sorted by name.
pub(crate) fn available(db: &Db, cfg: &Config) -> Vec<CharacterChoice> {
    let known = db.list_characters().unwrap_or_default();
    let mut files: Vec<(Option<String>, PathBuf)> =
        known.iter().map(|c| (Some(c.name.clone()), PathBuf::from(&c.path))).collect();
    for dir in folders(cfg) {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("vrm")) {
                files.push((None, path));
            }
        }
    }
    let mut seen = HashSet::new();
    let mut choices: Vec<CharacterChoice> = files
        .into_iter()
        .filter_map(|(name, path)| {
            let path = path.canonicalize().ok()?;
            if !seen.insert(path.clone()) {
                return None;
            }
            let name = name.unwrap_or_else(|| stem(&path));
            Some(CharacterChoice { name, path })
        })
        .collect();
    choices.sort_by_key(|c| c.name.to_lowercase());
    choices
}

/// Make `path` the character, remembered for next time. A file from outside
/// the character folders is copied in first. Returns the path to load.
pub(crate) fn activate(db: &Db, cfg: &Config, path: &Path, name: &str) -> Result<PathBuf> {
    let path = path.canonicalize()?;
    let in_a_folder = folders(cfg)
        .iter()
        .filter_map(|dir| dir.canonicalize().ok())
        .any(|dir| path.parent() == Some(dir.as_path()));
    let path = if in_a_folder {
        path
    } else {
        let dir = cfg.data_dir.join("characters");
        std::fs::create_dir_all(&dir)?;
        let dest = dir.join(path.file_name().unwrap_or_else(|| "character.vrm".as_ref()));
        std::fs::copy(&path, &dest)?;
        dest.canonicalize()?
    };
    db.add_character(name, &path.to_string_lossy())?;
    db.set_active_character(name)?;
    Ok(path)
}

/// A readable name for a model file: its name without the extension.
pub(crate) fn stem(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
}

/// Where characters live: the configured character's folder, the data dir's
/// `characters/` and the bundled ones.
fn folders(cfg: &Config) -> Vec<PathBuf> {
    let mut dirs = vec![cfg.data_dir.join("characters"), cfg.assets_dir.join("characters")];
    if let Some(parent) = cfg.character_path.parent() {
        dirs.push(parent.to_path_buf());
    }
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, Config, Db) {
        let tmp = tempfile::tempdir().unwrap();
        let bundled = tmp.path().join("assets/characters");
        std::fs::create_dir_all(&bundled).unwrap();
        for name in ["Tomo.vrm", "ayako.VRM", "notes.txt"] {
            std::fs::write(bundled.join(name), b"glTF").unwrap();
        }
        let mut cfg = Config::for_tests(tmp.path());
        cfg.character_path = bundled.join("Tomo.vrm");
        (tmp, cfg, Db::open_in_memory().unwrap())
    }

    #[test]
    fn it_offers_the_models_in_the_character_folders() {
        let (_tmp, cfg, db) = setup();
        let names: Vec<_> = available(&db, &cfg).into_iter().map(|c| c.name).collect();
        assert_eq!(names, ["ayako", "Tomo"]);
    }

    #[test]
    fn a_choice_is_remembered_and_shown_at_start() {
        let (_tmp, cfg, db) = setup();
        let ayako = cfg.character_path.with_file_name("ayako.VRM");
        let loaded = activate(&db, &cfg, &ayako, "ayako").unwrap();
        assert_eq!(loaded, ayako.canonicalize().unwrap(), "used in place, not copied");
        assert_eq!(startup(&db, &cfg), Some(loaded));
        assert_eq!(available(&db, &cfg).len(), 2, "no duplicate for the remembered one");
    }

    #[test]
    fn a_fresh_install_shows_a_bundled_character() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = Config::for_tests(tmp.path());
        let bundled = cfg.assets_dir.join("characters");
        std::fs::create_dir_all(&bundled).unwrap();
        std::fs::write(bundled.join("Tomo.vrm"), b"glTF").unwrap();
        let db = Db::open_in_memory().unwrap();
        assert!(!cfg.character_path.exists(), "nothing configured");
        assert_eq!(startup(&db, &cfg), Some(bundled.join("Tomo.vrm").canonicalize().unwrap()));
    }

    #[test]
    fn a_file_from_elsewhere_is_copied_in() {
        let (tmp, cfg, db) = setup();
        let download = tmp.path().join("Downloads/Rin.vrm");
        std::fs::create_dir_all(download.parent().unwrap()).unwrap();
        std::fs::write(&download, b"glTF").unwrap();
        let loaded = activate(&db, &cfg, &download, "Rin").unwrap();
        assert_eq!(loaded, cfg.data_dir.join("characters/Rin.vrm").canonicalize().unwrap());
        std::fs::remove_file(&download).unwrap();
        let names: Vec<_> = available(&db, &cfg).into_iter().map(|c| c.name).collect();
        assert_eq!(names, ["ayako", "Rin", "Tomo"]);
    }
}
