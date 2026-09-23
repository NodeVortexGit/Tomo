//! Long-term memory, backed by an embedded SQLite database.
//!
//! This is what lets the assistant "remember preferences and all sorts of
//! stuff" across restarts, and pull context back out again before answering.
//! Four tables:
//!
//!   preferences  — typed key/value the user sets or the AI infers
//!                  ("call me Alex", "prefers dark themes", "timezone=…").
//!   memories     — free-form facts the AI decides are worth keeping, each
//!                  with an importance score so recall can prioritise.
//!   messages     — the rolling chat transcript (for continuity + context).
//!   characters   — VRM models the user has imported, and which is active.
//!
//! The connection is wrapped in an `Arc<Mutex<…>>` so the struct is cheap to
//! clone and safe to share across the async tasks in the brain. Queries are
//! short; if you ever add a heavy one, wrap the call in `spawn_blocking`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

use crate::events::{now_ms, ChatLine, Role};

#[derive(Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
}

/// A remembered fact.
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: i64,
    pub kind: String,
    pub content: String,
    pub importance: i64,
    pub created_at: i64,
}

/// An imported VRM character.
#[derive(Debug, Clone)]
pub struct Character {
    pub id: i64,
    pub name: String,
    pub path: String,
    pub is_active: bool,
}

impl Db {
    /// Open (creating if needed) the database at `path` and run migrations.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening database {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        conn.pragma_update(None, "foreign_keys", "ON").ok();
        let db = Db {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.migrate()?;
        Ok(db)
    }

    /// In-memory database — used by the tests.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Db {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS preferences (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                kind       TEXT NOT NULL DEFAULT 'fact',
                content    TEXT NOT NULL,
                importance INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC, created_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                role    TEXT NOT NULL,
                content TEXT NOT NULL,
                at_ms   INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_at ON messages(at_ms);

            CREATE TABLE IF NOT EXISTS characters (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL UNIQUE,
                path      TEXT NOT NULL,
                added_at  INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0
            );

            -- Cached OS scans (the app/toggle catalog), keyed by name, with a
            -- fingerprint so a re-scan can detect changes without reparsing.
            CREATE TABLE IF NOT EXISTS snapshots (
                key         TEXT PRIMARY KEY,
                fingerprint INTEGER NOT NULL,
                json        TEXT NOT NULL,
                updated_at  INTEGER NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    // ---- preferences ------------------------------------------------------

    pub fn set_pref(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO preferences(key, value, updated_at) VALUES(?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            params![key, value, now_ms()],
        )?;
        Ok(())
    }

    pub fn get_pref(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let v = conn
            .query_row(
                "SELECT value FROM preferences WHERE key = ?1",
                params![key],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        Ok(v)
    }

    pub fn all_prefs(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT key, value FROM preferences ORDER BY key")?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ---- memories ---------------------------------------------------------

    pub fn add_memory(&self, kind: &str, content: &str, importance: i64) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memories(kind, content, importance, created_at) VALUES(?1, ?2, ?3, ?4)",
            params![kind, content, importance, now_ms()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// The `limit` most important+recent memories — the default context pull.
    pub fn top_memories(&self, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, kind, content, importance, created_at FROM memories
             ORDER BY importance DESC, created_at DESC LIMIT ?1",
        )?;
        let rows = stmt
            .query_map(params![limit as i64], map_memory)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Keyword search across memories (simple LIKE — good enough for a
    /// companion; swap for FTS5 if the store grows large).
    pub fn search_memories(&self, query: &str, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().unwrap();
        let like = format!("%{}%", query);
        let mut stmt = conn.prepare(
            "SELECT id, kind, content, importance, created_at FROM memories
             WHERE content LIKE ?1 ORDER BY importance DESC, created_at DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![like, limit as i64], map_memory)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ---- messages / transcript -------------------------------------------

    pub fn add_message(&self, line: &ChatLine) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO messages(role, content, at_ms) VALUES(?1, ?2, ?3)",
            params![role_str(line.role), line.text, line.at_ms],
        )?;
        Ok(())
    }

    /// The `limit` most recent lines, returned oldest→newest for replay.
    pub fn recent_messages(&self, limit: usize) -> Result<Vec<ChatLine>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT role, content, at_ms FROM
               (SELECT role, content, at_ms FROM messages ORDER BY at_ms DESC LIMIT ?1)
             ORDER BY at_ms ASC",
        )?;
        let rows = stmt
            .query_map(params![limit as i64], |r| {
                Ok(ChatLine {
                    role: role_from_str(&r.get::<_, String>(0)?),
                    text: r.get::<_, String>(1)?,
                    at_ms: r.get::<_, i64>(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ---- characters -------------------------------------------------------

    /// Register (or update the path of) an imported VRM and make it active.
    pub fn add_character(&self, name: &str, path: &str) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO characters(name, path, added_at, is_active) VALUES(?1, ?2, ?3, 0)
             ON CONFLICT(name) DO UPDATE SET path = excluded.path",
            params![name, path, now_ms()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn set_active_character(&self, name: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("UPDATE characters SET is_active = 0", [])?;
        conn.execute(
            "UPDATE characters SET is_active = 1 WHERE name = ?1",
            params![name],
        )?;
        Ok(())
    }

    pub fn active_character(&self) -> Result<Option<Character>> {
        let conn = self.conn.lock().unwrap();
        let c = conn
            .query_row(
                "SELECT id, name, path, is_active FROM characters WHERE is_active = 1 LIMIT 1",
                [],
                map_character,
            )
            .optional()?;
        Ok(c)
    }

    pub fn list_characters(&self) -> Result<Vec<Character>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, name, path, is_active FROM characters ORDER BY added_at DESC")?;
        let rows = stmt
            .query_map([], map_character)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ---- snapshots (cached OS catalog) -----------------------------------

    /// Store a scan under `key` with its fingerprint and JSON payload.
    pub fn set_snapshot(&self, key: &str, fingerprint: i64, json: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO snapshots(key, fingerprint, json, updated_at) VALUES(?1, ?2, ?3, ?4)
             ON CONFLICT(key) DO UPDATE SET
                fingerprint = excluded.fingerprint,
                json        = excluded.json,
                updated_at  = excluded.updated_at",
            params![key, fingerprint, json, now_ms()],
        )?;
        Ok(())
    }

    /// Return `(fingerprint, json)` for a cached scan, if present.
    pub fn get_snapshot(&self, key: &str) -> Result<Option<(i64, String)>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT fingerprint, json FROM snapshots WHERE key = ?1",
                params![key],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
            )
            .optional()?;
        Ok(row)
    }
}

fn map_memory(r: &rusqlite::Row) -> rusqlite::Result<Memory> {
    Ok(Memory {
        id: r.get(0)?,
        kind: r.get(1)?,
        content: r.get(2)?,
        importance: r.get(3)?,
        created_at: r.get(4)?,
    })
}

fn map_character(r: &rusqlite::Row) -> rusqlite::Result<Character> {
    Ok(Character {
        id: r.get(0)?,
        name: r.get(1)?,
        path: r.get(2)?,
        is_active: r.get::<_, i64>(3)? != 0,
    })
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
    }
}

fn role_from_str(s: &str) -> Role {
    match s {
        "user" => Role::User,
        "system" => Role::System,
        _ => Role::Assistant,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefs_roundtrip_and_upsert() {
        let db = Db::open_in_memory().unwrap();
        db.set_pref("name", "Alex").unwrap();
        db.set_pref("name", "Alexandra").unwrap(); // upsert, not duplicate
        assert_eq!(db.get_pref("name").unwrap().as_deref(), Some("Alexandra"));
        assert_eq!(db.all_prefs().unwrap().len(), 1);
        assert!(db.get_pref("missing").unwrap().is_none());
    }

    #[test]
    fn memories_rank_by_importance() {
        let db = Db::open_in_memory().unwrap();
        db.add_memory("fact", "likes tea", 1).unwrap();
        db.add_memory("fact", "birthday in May", 5).unwrap();
        let top = db.top_memories(10).unwrap();
        assert_eq!(top[0].content, "birthday in May");
        let hit = db.search_memories("tea", 10).unwrap();
        assert_eq!(hit.len(), 1);
    }

    #[test]
    fn messages_return_in_chronological_order() {
        let db = Db::open_in_memory().unwrap();
        for i in 0..5 {
            db.add_message(&ChatLine {
                role: Role::User,
                text: format!("m{i}"),
                at_ms: 1000 + i,
            })
            .unwrap();
        }
        let recent = db.recent_messages(3).unwrap();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].text, "m2"); // oldest of the last three
        assert_eq!(recent[2].text, "m4");
    }

    #[test]
    fn only_one_active_character() {
        let db = Db::open_in_memory().unwrap();
        db.add_character("Miku", "/a.vrm").unwrap();
        db.add_character("Rin", "/b.vrm").unwrap();
        db.set_active_character("Miku").unwrap();
        db.set_active_character("Rin").unwrap();
        assert_eq!(db.active_character().unwrap().unwrap().name, "Rin");
        assert_eq!(db.list_characters().unwrap().len(), 2);
    }
}
