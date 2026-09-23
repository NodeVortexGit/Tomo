//! Configuration loading.
//!
//! Everything secret (API keys) and everything the user is likely to tweak
//! lives in a `.env` file, exactly as the brief asked. This module reads that
//! file plus a few XDG paths and hands back a validated [`Config`] that the
//! rest of the program can rely on.
//!
//! Precedence, highest first:
//!   1. real process environment variables
//!   2. the `.env` file next to the binary / project root
//!   3. the compiled-in defaults below

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

/// The persona name the assistant answers to by default.
pub const DEFAULT_PERSONA: &str = "Tomo";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // ---- Claude (Anthropic Messages API) ------------------------------------
    /// Secret. Never logged. Read from `ANTHROPIC_API_KEY`.
    pub api_key: String,
    /// API root, `https://api.anthropic.com` unless you go through a gateway.
    pub base_url: String,
    /// e.g. `claude-sonnet-5`.
    pub model: String,
    /// How much the model thinks before answering: `low` (quick replies, the
    /// default for a chatty companion), `medium`, `high`, `xhigh` or `max`.
    pub effort: String,

    // ---- Speech -----------------------------------------------------------
    /// Edge-TTS voice id, e.g. `en-US-AriaNeural`.
    pub edge_tts_voice: String,
    /// Speaking rate for Edge-TTS, e.g. `+0%`, `-10%`.
    pub edge_tts_rate: String,
    /// Google Cloud Speech-to-Text API key (simple API-key mode). Optional:
    /// if empty, STT is disabled and the mic button is greyed out.
    pub google_stt_api_key: String,
    /// BCP-47 language for STT, e.g. `en-US`.
    pub stt_language: String,

    // ---- Character / persona ---------------------------------------------
    /// Friendly name shown in the chat header and used in the system prompt.
    pub persona_name: String,
    /// Absolute path to the `.vrm` file to load on start.
    pub character_path: PathBuf,

    // ---- Paths ------------------------------------------------------------
    /// Directory for the SQLite DB, imported characters, logs, cached audio.
    pub data_dir: PathBuf,
    /// The SQLite memory database.
    pub db_path: PathBuf,
    /// Append-only audit log of every command the AI runs.
    pub audit_log_path: PathBuf,
    /// Directory that holds the two Python speech helpers + their venv.
    pub scripts_dir: PathBuf,

    // ---- Behaviour switches ----------------------------------------------
    /// Master safety switch. When false the executor refuses everything and
    /// only logs the request. Ships `true`; flip to `false` to audit first.
    pub allow_command_execution: bool,
    /// Extra commands the user explicitly permits (comma-separated in .env).
    pub extra_allowed_commands: Vec<String>,
    /// Whether the AI may take screenshots to see what's on screen.
    pub allow_screen: bool,
    /// Whether to listen for the "Hey Tomo" wake word (recognised offline).
    pub wake_word: bool,
}

impl Config {
    /// Load `.env` (if present) then build the config from the environment.
    ///
    /// `project_root` is used only to locate the bundled `scripts/` dir and a
    /// project-local `.env`; pass the directory that contains `install.sh`.
    pub fn load(project_root: &Path) -> Result<Self> {
        // Load `.env` from the project root first, then fall back to CWD. We
        // ignore "not found" — the real environment may already hold the vars.
        // Anything else must be surfaced: dotenvy stops at the first line it
        // can't parse, silently dropping every key after it.
        // (Both calls usually hit the same file, so report only the first.)
        let problem = [
            dotenvy::from_path(project_root.join(".env")),
            dotenvy::dotenv().map(drop),
        ]
        .into_iter()
        .filter_map(Result::err)
        .find(|e| !e.not_found());
        match problem {
            // Don't echo the line itself: it may hold a secret.
            Some(dotenvy::Error::LineParse(..)) => tracing::warn!(
                ".env has a line that can't be parsed (a comment missing its `#`, or an \
                 unquoted value with spaces?) — every key after it was ignored"
            ),
            Some(e) => tracing::warn!(".env was not loaded: {e}"),
            None => {}
        }

        let dirs = ProjectDirs::from("dev", "tomo", "tomo")
            .context("could not determine a home directory for config")?;
        let data_dir = env_path("TOMO_DATA_DIR").unwrap_or_else(|| dirs.data_dir().to_path_buf());
        std::fs::create_dir_all(&data_dir)
            .with_context(|| format!("creating data dir {}", data_dir.display()))?;

        let scripts_dir =
            env_path("TOMO_SCRIPTS_DIR").unwrap_or_else(|| project_root.join("scripts"));

        let character_path = env_path("TOMO_CHARACTER")
            .unwrap_or_else(|| data_dir.join("characters").join("default.vrm"));

        let cfg = Config {
            // The OPENAI_* names are still read: Tomo used Anthropic's
            // OpenAI-compatible endpoint before moving to the Messages API.
            api_key: env_str("ANTHROPIC_API_KEY")
                .or_else(|| env_str("OPENAI_API_KEY"))
                .unwrap_or_default(),
            base_url: env_str("ANTHROPIC_BASE_URL")
                .or_else(|| env_str("OPENAI_BASE_URL"))
                .unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            model: env_str("ANTHROPIC_MODEL")
                .or_else(|| env_str("OPENAI_MODEL"))
                .unwrap_or_else(|| "claude-sonnet-5".to_string()),
            effort: env_str("TOMO_EFFORT").unwrap_or_else(|| "low".to_string()),

            edge_tts_voice: env_str("EDGE_TTS_VOICE")
                .unwrap_or_else(|| "en-US-AriaNeural".to_string()),
            edge_tts_rate: env_str("EDGE_TTS_RATE").unwrap_or_else(|| "+0%".to_string()),
            google_stt_api_key: env_str("GOOGLE_STT_API_KEY").unwrap_or_default(),
            stt_language: env_str("STT_LANGUAGE").unwrap_or_else(|| "en-US".to_string()),

            persona_name: env_str("TOMO_PERSONA").unwrap_or_else(|| DEFAULT_PERSONA.to_string()),
            character_path,

            db_path: data_dir.join("memory.sqlite3"),
            audit_log_path: data_dir.join("command-audit.log"),
            data_dir,
            scripts_dir,

            allow_command_execution: env_bool("TOMO_ALLOW_COMMANDS").unwrap_or(true),
            extra_allowed_commands: env_str("TOMO_EXTRA_ALLOWED")
                .map(|s| {
                    s.split(',')
                        .map(|p| p.trim().to_string())
                        .filter(|p| !p.is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            allow_screen: env_bool("TOMO_ALLOW_SCREEN").unwrap_or(true),
            wake_word: env_bool("TOMO_WAKE_WORD").unwrap_or(true),
        };

        Ok(cfg)
    }

    /// True when we have enough to talk to the model.
    pub fn ai_ready(&self) -> bool {
        !self.api_key.is_empty()
    }

    /// True when speech-to-text is configured.
    pub fn stt_ready(&self) -> bool {
        !self.google_stt_api_key.is_empty()
    }

    /// A redacted view safe to print in logs.
    pub fn redacted(&self) -> String {
        format!(
            "Config {{ model: {}, effort: {}, base_url: {}, voice: {}, persona: {}, ai_ready: {}, stt_ready: {}, data_dir: {} }}",
            self.model,
            self.effort,
            self.base_url,
            self.edge_tts_voice,
            self.persona_name,
            self.ai_ready(),
            self.stt_ready(),
            self.data_dir.display(),
        )
    }
}

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn env_path(key: &str) -> Option<PathBuf> {
    env_str(key).map(PathBuf::from)
}

fn env_bool(key: &str) -> Option<bool> {
    env_str(key).map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}
