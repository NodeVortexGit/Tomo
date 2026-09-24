//! Configuration loading.
//!
//! Everything the user is likely to tweak lives in a `.env` file. This module
//! reads that file plus the platform's usual folders (XDG on Linux, AppData on
//! Windows) and hands back a validated [`Config`] that the rest of the
//! program can rely on.
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
/// The Piper voice used unless `TOMO_TTS_VOICE` picks another.
pub const DEFAULT_VOICE: &str = "en_US-lessac-medium";
/// Enough for the persona, the tools and a recent stretch of conversation.
const DEFAULT_CONTEXT: u32 = 8192;

/// Which local model server thinks for Tomo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmProvider {
    /// Ollama if it's running, else LM Studio.
    Auto,
    /// Ollama (its own API, which also sets the context size).
    Ollama,
    /// LM Studio's local server.
    LmStudio,
    /// Any other server with an OpenAI-compatible API, at `llm_url`.
    OpenAiCompatible,
}

impl LlmProvider {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace([' ', '-', '_'], "").as_str() {
            "auto" | "" => Some(Self::Auto),
            "ollama" => Some(Self::Ollama),
            "lmstudio" => Some(Self::LmStudio),
            "openai" | "openaicompatible" => Some(Self::OpenAiCompatible),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // ---- The model: a local server (Ollama or LM Studio) ------------------
    pub llm_provider: LlmProvider,
    /// The server's address; empty for the provider's usual one on this
    /// machine (Ollama: http://127.0.0.1:11434, LM Studio:
    /// http://127.0.0.1:1234/v1).
    pub llm_url: String,
    /// The model to use; empty for the first local chat model the server has.
    pub llm_model: String,
    /// Only for a server that asks for one. Never logged.
    pub llm_api_key: String,
    /// How much conversation the model reads at once, in tokens (Ollama).
    pub llm_context: u32,

    // ---- Speech (all on this machine) --------------------------------------
    /// Piper voice, e.g. `en_US-lessac-medium` (see `python -m
    /// piper.download_voices`).
    pub tts_voice: String,
    /// Speaking speed: 1.0 normal, 1.2 a fifth faster.
    pub tts_speed: f32,

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
    /// Directory that holds the Python speech helpers + their venv.
    pub scripts_dir: PathBuf,
    /// The bundled assets (`<install>/assets`): characters that ship with Tomo.
    pub assets_dir: PathBuf,

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

        let data_dir = default_data_dir().context("could not determine a home directory for config")?;
        std::fs::create_dir_all(&data_dir)
            .with_context(|| format!("creating data dir {}", data_dir.display()))?;

        let scripts_dir =
            env_path("TOMO_SCRIPTS_DIR").unwrap_or_else(|| project_root.join("scripts"));

        let character_path = env_path("TOMO_CHARACTER")
            .unwrap_or_else(|| data_dir.join("characters").join("default.vrm"));

        let llm_provider = match env_str("TOMO_LLM") {
            Some(value) => LlmProvider::parse(&value).unwrap_or_else(|| {
                tracing::warn!("TOMO_LLM={value:?} isn't one of auto, ollama, lmstudio, openai; using auto");
                LlmProvider::Auto
            }),
            None => LlmProvider::Auto,
        };
        let cfg = Config {
            llm_provider,
            llm_url: env_str("TOMO_LLM_URL").unwrap_or_default(),
            llm_model: env_str("TOMO_LLM_MODEL").unwrap_or_default(),
            llm_api_key: env_str("TOMO_LLM_API_KEY").unwrap_or_default(),
            llm_context: env_str("TOMO_LLM_CONTEXT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_CONTEXT),

            tts_voice: env_str("TOMO_TTS_VOICE").unwrap_or_else(|| DEFAULT_VOICE.to_string()),
            tts_speed: env_str("TOMO_TTS_SPEED")
                .and_then(|v| v.parse().ok())
                .filter(|s: &f32| *s > 0.1 && *s < 5.0)
                .unwrap_or(1.0),

            persona_name: env_str("TOMO_PERSONA").unwrap_or_else(|| DEFAULT_PERSONA.to_string()),
            character_path,

            db_path: data_dir.join("memory.sqlite3"),
            audit_log_path: data_dir.join("command-audit.log"),
            data_dir,
            scripts_dir,
            assets_dir: project_root.join("assets"),

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

    /// Defaults with everything under `root`, and nothing read from the
    /// environment or a `.env`: for tests that must not touch the real setup.
    #[cfg(test)]
    pub(crate) fn for_tests(root: &Path) -> Self {
        let data_dir = root.join("data");
        std::fs::create_dir_all(&data_dir).expect("test data dir");
        Config {
            llm_provider: LlmProvider::Auto,
            llm_url: String::new(),
            llm_model: String::new(),
            llm_api_key: String::new(),
            llm_context: DEFAULT_CONTEXT,
            tts_voice: DEFAULT_VOICE.into(),
            tts_speed: 1.0,
            persona_name: DEFAULT_PERSONA.into(),
            character_path: data_dir.join("characters").join("default.vrm"),
            db_path: data_dir.join("memory.sqlite3"),
            audit_log_path: data_dir.join("command-audit.log"),
            data_dir,
            scripts_dir: root.join("scripts"),
            assets_dir: root.join("assets"),
            allow_command_execution: false,
            extra_allowed_commands: Vec::new(),
            allow_screen: false,
            wake_word: false,
        }
    }

    /// A redacted view safe to print in logs.
    pub fn redacted(&self) -> String {
        let or_auto = |s: &str| if s.is_empty() { "auto".to_string() } else { s.to_string() };
        format!(
            "Config {{ llm: {:?} (url: {}, model: {}), voice: {}, persona: {}, wake_word: {}, data_dir: {} }}",
            self.llm_provider,
            or_auto(&self.llm_url),
            or_auto(&self.llm_model),
            self.tts_voice,
            self.persona_name,
            self.wake_word,
            self.data_dir.display(),
        )
    }
}

/// Tomo's data folder: TOMO_DATA_DIR, else the platform's usual place —
/// `~/.local/share/tomo` on Linux, `%APPDATA%\\tomo\\tomo\\data` on Windows.
pub fn default_data_dir() -> Option<PathBuf> {
    env_path("TOMO_DATA_DIR").or_else(|| Some(ProjectDirs::from("dev", "tomo", "tomo")?.data_dir().to_path_buf()))
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
