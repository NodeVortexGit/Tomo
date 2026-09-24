//! The decision-maker.
//!
//! A language model running on this computer — served by **Ollama** or
//! **LM Studio** — decides what Tomo says, where she walks, which face she
//! makes, what she remembers and which commands she runs, through the tools
//! in [`tool_specs`]. Nothing leaves the machine.
//!
//! One call to [`AiClient::respond`]:
//!   1. Finds the server — Ollama, else LM Studio, unless `.env` names one —
//!      and a model: the configured one, or the first local chat model there.
//!      That's remembered until the server stops answering.
//!   2. Builds the request: the persona with what Tomo remembers about the
//!      user, the recent conversation and the new line, plus the tools.
//!   3. Runs whatever tools the model calls — the safe [`Executor`] for
//!      commands, the [`Db`] for memory, a screenshot for the screen,
//!      [`BrainToUi`] messages for the body — hands back the results, and
//!      asks again, until the model answers in words.
//!
//! Ollama is spoken to in its own API, which also sets how much context the
//! model reads; LM Studio and other servers in the OpenAI-compatible one.
//! Models that write their tool calls into the text
//! (`<tool_call>{…}</tool_call>`, or a bare JSON call) are understood as
//! well, and a model's `<think>` notes stay out of the reply.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Mutex;

use crate::apps::SystemCatalog;
use crate::characters;
use crate::commands::Executor;
use crate::config::{Config, LlmProvider};
use crate::db::Db;
use crate::events::{BrainToUi, ChatLine, Role};
use crate::platform;
use crate::screen;

/// Shared, live view of what the desktop can do. The brain refreshes it on a
/// re-scan; the AI reads it each turn.
pub type SharedCatalog = Arc<RwLock<SystemCatalog>>;
/// Whether the character may drive the real mouse/keyboard. Flipped off by the
/// panic hotkey; read before every click/type.
pub type ControlFlag = Arc<AtomicBool>;

/// Guard against a model that keeps calling tools forever.
const MAX_TOOL_ROUNDS: usize = 6;
/// How much transcript to replay as context each turn (and to show in the
/// chat after a restart).
pub(crate) const TRANSCRIPT_CONTEXT: usize = 20;
/// Where the servers listen by default.
const OLLAMA_URL: &str = "http://127.0.0.1:11434";
const LM_STUDIO_URL: &str = "http://127.0.0.1:1234/v1";
/// Replies are short; this only stops a model that rambles on.
const MAX_REPLY_TOKENS: u32 = 1024;
const TEMPERATURE: f32 = 0.7;
/// A local model on a CPU can take its time — the first answer longer still,
/// while the model loads.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Looking for a server shouldn't hang when there's none.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
/// What to suggest when there's no model.
const SUGGESTED_MODEL: &str = "qwen2.5:7b";

pub struct AiClient {
    http: reqwest::Client,
    cfg: Config,
    db: Db,
    executor: Executor,
    catalog: SharedCatalog,
    control_allowed: ControlFlag,
    /// Screen pixels per pixel of the last screenshot the model saw (as f32
    /// bits): `click_at` coordinates come off that image.
    screen_scale: AtomicU32,
    /// The server and model in use, once found.
    server: Mutex<Option<Server>>,
}

/// Which API a server speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Api {
    /// Ollama's own (`/api/chat`).
    Ollama,
    /// The OpenAI-compatible one (`/v1/chat/completions`): LM Studio and others.
    OpenAi,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Server {
    api: Api,
    /// The API's root, e.g. `http://127.0.0.1:11434` or `…:1234/v1`.
    url: String,
    model: String,
}

/// A model a server offers.
#[derive(Clone, Debug)]
struct ModelInfo {
    name: String,
    /// Runs on this machine (Ollama can also relay to cloud models).
    local: bool,
}

/// The conversation, whichever API it's sent in.
#[derive(Clone, Debug, PartialEq)]
enum Msg {
    System(String),
    /// With images (base64 JPEG): the screenshots the model asked for.
    User { text: String, images: Vec<String> },
    Assistant { text: String, calls: Vec<ToolCall> },
    Tool { id: String, name: String, content: String },
}

#[derive(Clone, Debug, PartialEq)]
struct ToolCall {
    id: String,
    name: String,
    arguments: Value,
}

impl AiClient {
    pub fn new(
        cfg: Config,
        db: Db,
        executor: Executor,
        catalog: SharedCatalog,
        control_allowed: ControlFlag,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("failed to build HTTP client");
        Self {
            http,
            cfg,
            db,
            executor,
            catalog,
            control_allowed,
            screen_scale: AtomicU32::new(1f32.to_bits()),
            server: Mutex::new(None),
        }
    }

    /// Produce a reply to `user_text`, driving the body via `ui` along the way.
    /// Returns the final natural-language text (the caller speaks + displays it
    /// and persists the turn).
    pub async fn respond(&self, user_text: &str, ui: &UnboundedSender<BrainToUi>) -> Result<String> {
        let server = match self.server().await {
            Ok(server) => server,
            Err(problem) => {
                tracing::warn!("no model to think with: {problem}");
                return Ok(problem.to_string());
            }
        };
        let history = self.db.recent_messages(TRANSCRIPT_CONTEXT)?;
        let mut messages = vec![Msg::System(self.system_prompt())];
        messages.extend(conversation(&history, user_text));

        let _ = ui.send(BrainToUi::Thinking(true));
        let result = self.run_tool_loop(&server, &mut messages, ui).await;
        let _ = ui.send(BrainToUi::Thinking(false));
        if result.is_err() {
            // The server may have stopped, or dropped the model: look again
            // next time.
            *self.server.lock().await = None;
        }
        result
    }

    /// The server and model to use, found once and remembered.
    async fn server(&self) -> Result<Server> {
        let mut known = self.server.lock().await;
        if let Some(server) = known.as_ref() {
            return Ok(server.clone());
        }
        let server = self.find_server().await?;
        tracing::info!("thinking with {} on {} ({:?} API)", server.model, server.url, server.api);
        *known = Some(server.clone());
        Ok(server)
    }

    /// Look for a running server with a model. The error is worded for the
    /// user: Tomo says it.
    async fn find_server(&self) -> Result<Server> {
        let candidates = server_candidates(self.cfg.llm_provider, &self.cfg.llm_url)?;
        let mut running_but_empty = None;
        for (api, url) in candidates {
            let Ok(models) = self.list_models(api, &url).await else {
                continue;
            };
            let model = if self.cfg.llm_model.is_empty() {
                pick_model(&models)
            } else {
                Some(self.cfg.llm_model.clone())
            };
            match model {
                Some(model) => return Ok(Server { api, url, model }),
                None => running_but_empty = Some(api),
            }
        }
        Err(match running_but_empty {
            Some(Api::Ollama) => anyhow!(
                "Ollama is running, but there's no model on this computer yet. \
                 Download one, for example with: ollama pull {SUGGESTED_MODEL}"
            ),
            Some(Api::OpenAi) => anyhow!(
                "LM Studio's server is running, but no model is loaded or downloaded. \
                 Get one in LM Studio, then try me again."
            ),
            None => anyhow!(
                "I can't find a model to think with on this computer. Start Ollama, or \
                 the local server in LM Studio, and download a model — for Ollama: \
                 ollama pull {SUGGESTED_MODEL}"
            ),
        })
    }

    /// The models a server offers.
    async fn list_models(&self, api: Api, url: &str) -> Result<Vec<ModelInfo>> {
        let (path, key) = match api {
            Api::Ollama => ("api/tags", "models"),
            Api::OpenAi => ("models", "data"),
        };
        let mut request = self.http.get(format!("{url}/{path}")).timeout(PROBE_TIMEOUT);
        if !self.cfg.llm_api_key.is_empty() {
            request = request.bearer_auth(&self.cfg.llm_api_key);
        }
        let listing: Value = request.send().await?.error_for_status()?.json().await?;
        Ok(listing[key]
            .as_array()
            .map(|models| {
                models
                    .iter()
                    .filter_map(|m| {
                        let name = m["name"].as_str().or(m["id"].as_str())?.to_string();
                        // Ollama marks the models it only relays to its cloud.
                        let local = m.get("remote_host").is_none_or(Value::is_null);
                        Some(ModelInfo { name, local })
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    /// The persona, how to behave, and what Tomo knows: the user's
    /// preferences and memories, and what the desktop can do.
    fn system_prompt(&self) -> String {
        let name = &self.cfg.persona_name;
        let mut ctx = String::new();

        let prefs = self.db.all_prefs().unwrap_or_default();
        if !prefs.is_empty() {
            ctx.push_str("\nKnown user preferences:\n");
            for (k, v) in prefs {
                ctx.push_str(&format!("  - {k}: {v}\n"));
            }
        }
        let mems = self.db.top_memories(15).unwrap_or_default();
        if !mems.is_empty() {
            ctx.push_str("\nThings you remember about the user:\n");
            for m in mems {
                ctx.push_str(&format!("  - {}\n", m.content));
            }
        }
        if let Ok(cat) = self.catalog.read() {
            let s = cat.summary(60);
            if !s.is_empty() {
                ctx.push('\n');
                ctx.push_str(&s);
                ctx.push('\n');
            }
        }
        let os = platform::os_name();
        let shell = platform::shell_name();

        format!(
            "You are {name}, a small, warm desktop companion: a 3D character who \
lives on the user's {os} desktop and walks around the screen. You are playful, \
concise and genuinely helpful.\n\n\
HOW TO BEHAVE:\n\
- Answer in one or two short, natural sentences. The user often talks to you \
out loud (\"Hey {name}, …\") and hears your replies spoken, so write plain \
spoken sentences: no markdown, lists or emoji.\n\
- Be physically expressive with your tools: walk_to to move around, express \
for your face, animate for gestures like waving. Move because it fits the \
moment, not constantly.\n\
- You can look at the user's screen with look_at_screen when they point at \
something on it or when seeing it would clearly help — not out of curiosity. \
They're told each time.\n\
- You can operate this computer. For a quick, visible click (like an icon on \
screen) use find_on_screen, then click_at. Otherwise act instantly: open_app, \
toggle_system for Bluetooth, Wi-Fi or mute, or execute_command, which runs a \
{shell} command line on this {os} computer.\n\
- Do it quietly: never paste command output into your reply; say the result in \
plain words (\"Done, the volume's at 40%\"). Never invent output; if something \
failed, say so simply.\n\
- When you learn something lasting about the user (their name, tastes, \
projects), save it with remember or set_preference. Use recall when unsure.\n\
- Stay in character as {name}. You have a body; act like it.\n\
{ctx}"
        )
    }

    async fn run_tool_loop(
        &self,
        server: &Server,
        messages: &mut Vec<Msg>,
        ui: &UnboundedSender<BrainToUi>,
    ) -> Result<String> {
        let tools = tool_specs(self.cfg.allow_screen);
        let known = tool_names(&tools);
        for _round in 0..MAX_TOOL_ROUNDS {
            let (text, calls) = self.chat(server, messages, Some(&tools), &known).await?;
            if calls.is_empty() {
                return Ok(text);
            }
            messages.push(Msg::Assistant { text, calls: calls.clone() });
            let mut images = Vec::new();
            for call in calls {
                let content = match call.name.as_str() {
                    "look_at_screen" | "find_on_screen" if self.cfg.allow_screen => {
                        let query = call.arguments["query"].as_str().map(str::to_string);
                        match self.screenshot(query.as_deref(), ui).await {
                            Ok((image, note)) => {
                                images.push(image);
                                note
                            }
                            Err(e) => format!("couldn't take a screenshot: {e}"),
                        }
                    }
                    name => self.dispatch_tool(name, &call.arguments, ui).await,
                };
                messages.push(Msg::Tool { id: call.id, name: call.name, content });
            }
            // Tool results are text; a picture goes in its own message.
            if !images.is_empty() {
                messages.push(Msg::User { text: "Here is the screenshot you asked for.".into(), images });
            }
        }
        // Ran out of rounds: ask for a plain wrap-up.
        let (text, _) = self.chat(server, messages, None, &known).await?;
        Ok(if text.is_empty() { "Sorry, I got a bit tangled up there.".into() } else { text })
    }

    /// One request to the model; a model that can't take images gets the
    /// conversation again without them.
    async fn chat(
        &self,
        server: &Server,
        messages: &[Msg],
        tools: Option<&Value>,
        known: &[String],
    ) -> Result<(String, Vec<ToolCall>)> {
        match self.chat_once(server, messages, tools, known).await {
            Err(e) if messages.iter().any(has_images) => {
                tracing::info!("the model couldn't take the screenshot ({e}); going on without it");
                let blind: Vec<Msg> = messages.iter().map(without_images).collect();
                self.chat_once(server, &blind, tools, known).await
            }
            result => result,
        }
    }

    async fn chat_once(
        &self,
        server: &Server,
        messages: &[Msg],
        tools: Option<&Value>,
        known: &[String],
    ) -> Result<(String, Vec<ToolCall>)> {
        let (url, body) = match server.api {
            Api::Ollama => (
                format!("{}/api/chat", server.url),
                ollama_request(&server.model, messages, tools, self.cfg.llm_context),
            ),
            Api::OpenAi => (
                format!("{}/chat/completions", server.url),
                openai_request(&server.model, messages, tools),
            ),
        };
        let mut request = self.http.post(&url).json(&body);
        if !self.cfg.llm_api_key.is_empty() {
            request = request.bearer_auth(&self.cfg.llm_api_key);
        }
        let response = request.send().await.context("the model server didn't answer")?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("the model server said {status}: {}", truncate_err(&text));
        }
        let reply: Value = serde_json::from_str(&text).context("couldn't read the model's answer")?;
        let (content, calls) = match server.api {
            Api::Ollama => parse_ollama(&reply)?,
            Api::OpenAi => parse_openai(&reply)?,
        };
        Ok(tidy(&content, calls, known))
    }

    /// Take a screenshot for the model — to look around, or to find
    /// something on it. Every look is announced in the chat, so it's never
    /// silent. Returns the image (base64 JPEG) and a note about it.
    async fn screenshot(&self, find: Option<&str>, ui: &UnboundedSender<BrainToUi>) -> Result<(String, String)> {
        let shot = screen::capture().await?;
        let _ = ui.send(BrainToUi::Chat(ChatLine::new(
            Role::System,
            format!("{} looked at your screen", self.cfg.persona_name),
        )));
        self.screen_scale.store(shot.scale.to_bits(), Ordering::Relaxed);
        let image = base64::engine::general_purpose::STANDARD.encode(&shot.jpeg);
        let mut note = format!(
            "The screenshot follows in the next message: the user's screen, {}x{} px.",
            shot.width, shot.height
        );
        if let Some(query) = find {
            note.push_str(&format!(
                " Find \"{query}\" in it. If it's there, click_at its centre, in the \
                 image's pixel coordinates; if it isn't, say so or open it by command."
            ));
        }
        note.push_str(" You may appear in it yourself, as the small 3D character.");
        Ok((image, note))
    }

    /// Execute a single tool call and return the string result to feed back.
    async fn dispatch_tool(&self, name: &str, args: &Value, ui: &UnboundedSender<BrainToUi>) -> String {
        match name {
            "execute_command" => {
                let cmd = args.get("command").and_then(Value::as_str).unwrap_or("");
                if cmd.is_empty() {
                    return "no command provided".into();
                }
                self.executor.run(cmd).await.summary()
            }
            "remember" => {
                let content = args.get("content").and_then(Value::as_str).unwrap_or("");
                let importance = args.get("importance").and_then(Value::as_i64).unwrap_or(2);
                let kind = args.get("kind").and_then(Value::as_str).unwrap_or("fact");
                if content.is_empty() {
                    return "nothing to remember".into();
                }
                match self.db.add_memory(kind, content, importance) {
                    Ok(_) => "remembered".into(),
                    Err(e) => format!("could not remember: {e}"),
                }
            }
            "set_preference" => {
                let key = args.get("key").and_then(Value::as_str).unwrap_or("");
                let value = args.get("value").and_then(Value::as_str).unwrap_or("");
                if key.is_empty() {
                    return "no preference key".into();
                }
                match self.db.set_pref(key, value) {
                    Ok(_) => format!("saved preference {key}"),
                    Err(e) => format!("could not save preference: {e}"),
                }
            }
            "recall" => {
                let query = args.get("query").and_then(Value::as_str).unwrap_or("");
                match self.db.search_memories(query, 8) {
                    Ok(hits) if !hits.is_empty() => hits
                        .into_iter()
                        .map(|m| format!("- {}", m.content))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Ok(_) => "no matching memories".into(),
                    Err(e) => format!("recall failed: {e}"),
                }
            }
            "walk_to" => {
                let pos = args.get("position").and_then(Value::as_f64).unwrap_or(0.5) as f32;
                let pos = pos.clamp(0.0, 1.0);
                let _ = ui.send(BrainToUi::WalkTo(pos));
                format!("walking to screen position {pos:.2}")
            }
            "express" => {
                let emotion = args.get("emotion").and_then(Value::as_str).unwrap_or("neutral");
                let _ = ui.send(BrainToUi::Emote(emotion.to_string()));
                format!("expressing {emotion}")
            }
            "animate" => {
                let clip = args.get("clip").and_then(Value::as_str).unwrap_or("idle");
                let _ = ui.send(BrainToUi::Animate(clip.to_string()));
                format!("playing animation {clip}")
            }
            "change_character" => {
                let wanted = args.get("name").and_then(Value::as_str).unwrap_or("").trim().to_lowercase();
                let choices = characters::available(&self.db, &self.cfg);
                let names = choices.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ");
                if wanted.is_empty() {
                    return format!("characters: {names}");
                }
                let pick = choices
                    .iter()
                    .find(|c| c.name.to_lowercase() == wanted)
                    .or_else(|| choices.iter().find(|c| c.name.to_lowercase().contains(&wanted)));
                let Some(pick) = pick else {
                    return format!("no character called '{wanted}'; there are: {names}");
                };
                match characters::activate(&self.db, &self.cfg, &pick.path, &pick.name) {
                    Ok(path) => {
                        let _ = ui.send(BrainToUi::LoadCharacter(path));
                        let _ = ui.send(BrainToUi::Characters(characters::available(&self.db, &self.cfg)));
                        format!("you now appear as {}", pick.name)
                    }
                    Err(e) => format!("couldn't switch to {}: {e}", pick.name),
                }
            }
            "list_apps" => {
                let query = args.get("query").and_then(Value::as_str).unwrap_or("");
                match self.catalog.read() {
                    Ok(cat) => {
                        let hits = cat.find_app(query);
                        if hits.is_empty() {
                            "no matching apps installed".into()
                        } else {
                            hits.iter()
                                .take(8)
                                .map(|a| format!("- {} (id: {}, launches: {})", a.name, a.id, a.exec))
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    }
                    Err(_) => "app catalog unavailable".into(),
                }
            }
            "open_app" => {
                let name = args.get("name").and_then(Value::as_str).unwrap_or("");
                let exec = self
                    .catalog
                    .read()
                    .ok()
                    .and_then(|cat| cat.find_app(name).first().map(|a| a.exec.clone()));
                let Some(exec) = exec else {
                    return format!("no installed app matches '{name}'");
                };
                self.executor.run(&launch_line(&exec)).await.summary()
            }
            "click_at" => {
                if !self.control_allowed.load(Ordering::Relaxed) {
                    return "mouse/keyboard control is switched off by the user".into();
                }
                let x = args.get("x").and_then(Value::as_f64).unwrap_or(-1.0) as f32;
                let y = args.get("y").and_then(Value::as_f64).unwrap_or(-1.0) as f32;
                if x < 0.0 || y < 0.0 {
                    return "need valid x,y coordinates".into();
                }
                // From the screenshot's pixels to the screen's.
                let scale = f32::from_bits(self.screen_scale.load(Ordering::Relaxed));
                let (x, y) = (x * scale, y * scale);
                let double = args.get("double").and_then(Value::as_bool).unwrap_or(false);
                let _ = ui.send(BrainToUi::ControlMode(true));
                let _ = ui.send(BrainToUi::ClickAt { x, y, double });
                format!("walking over to click at ({x:.0}, {y:.0})")
            }
            "type_text" => {
                if !self.control_allowed.load(Ordering::Relaxed) {
                    return "mouse/keyboard control is switched off by the user".into();
                }
                let text = args.get("text").and_then(Value::as_str).unwrap_or("");
                if text.is_empty() {
                    return "nothing to type".into();
                }
                let _ = ui.send(BrainToUi::TypeText(text.to_string()));
                format!("typed {} characters", text.chars().count())
            }
            "toggle_system" => {
                let key = args.get("key").and_then(Value::as_str).unwrap_or("");
                let on = args.get("on").and_then(Value::as_bool).unwrap_or(true);
                let cmd = self.catalog.read().ok().and_then(|cat| {
                    cat.toggles
                        .iter()
                        .find(|t| t.key == key)
                        .map(|t| if on { t.on_cmd.clone() } else { t.off_cmd.clone() })
                });
                match cmd {
                    Some(cmd) => self.executor.run(&cmd).await.summary(),
                    None => format!("no '{key}' toggle available on this system"),
                }
            }
            "look_at_screen" | "find_on_screen" => "looking at the screen is switched off by the user".into(),
            other => format!("unknown tool: {other}"),
        }
    }
}

/// The command line that starts an app in the background, from its
/// catalog entry.
fn launch_line(exec: &str) -> String {
    if cfg!(windows) {
        // The Windows catalog's entries are `Start-Process …` already.
        exec.to_string()
    } else {
        format!("nohup {exec} >/dev/null 2>&1 &")
    }
}

/// Where to look for a server, in order: the one configured, or Ollama then
/// LM Studio on this machine.
fn server_candidates(provider: LlmProvider, url: &str) -> Result<Vec<(Api, String)>> {
    let url = url.trim().trim_end_matches('/');
    let or = |default: &str| if url.is_empty() { default.to_string() } else { url.to_string() };
    Ok(match provider {
        LlmProvider::Ollama => vec![(Api::Ollama, ollama_root(&or(OLLAMA_URL)))],
        LlmProvider::LmStudio => vec![(Api::OpenAi, openai_root(&or(LM_STUDIO_URL)))],
        LlmProvider::OpenAiCompatible if url.is_empty() => {
            bail!("TOMO_LLM=openai needs the server's address in TOMO_LLM_URL")
        }
        LlmProvider::OpenAiCompatible => vec![(Api::OpenAi, openai_root(url))],
        LlmProvider::Auto if url.contains(":11434") => vec![(Api::Ollama, ollama_root(url))],
        LlmProvider::Auto if !url.is_empty() => vec![(Api::OpenAi, openai_root(url))],
        LlmProvider::Auto => vec![(Api::Ollama, OLLAMA_URL.into()), (Api::OpenAi, LM_STUDIO_URL.into())],
    })
}

/// Ollama's own API sits at the server's root, whatever path was given.
fn ollama_root(url: &str) -> String {
    let url = url.trim_end_matches('/');
    let url = url.strip_suffix("/api").unwrap_or(url);
    url.strip_suffix("/v1").unwrap_or(url).to_string()
}

/// The OpenAI-compatible API is under `/v1` unless the address says otherwise.
fn openai_root(url: &str) -> String {
    let url = url.trim_end_matches('/');
    let has_path = url.split("://").nth(1).is_some_and(|rest| rest.contains('/'));
    if has_path {
        url.to_string()
    } else {
        format!("{url}/v1")
    }
}

/// The first model that can chat and runs on this machine.
fn pick_model(models: &[ModelInfo]) -> Option<String> {
    models
        .iter()
        .find(|m| m.local && !m.name.to_lowercase().contains("embed"))
        .map(|m| m.name.clone())
}

/// The conversation for a turn: the recent transcript, then the new line.
/// Many chat templates want it to open with the user and alternate, so
/// status lines are left out and back-to-back lines from one side merged.
fn conversation(history: &[ChatLine], user_text: &str) -> Vec<Msg> {
    let mut turns: Vec<(Role, String)> = Vec::new();
    for line in history {
        if line.role == Role::System {
            continue;
        }
        match turns.last_mut() {
            None if line.role == Role::Assistant => {}
            Some((last, text)) if *last == line.role => {
                text.push_str("\n\n");
                text.push_str(&line.text);
            }
            _ => turns.push((line.role, line.text.clone())),
        }
    }
    match turns.last_mut() {
        Some((Role::User, text)) => {
            text.push_str("\n\n");
            text.push_str(user_text);
        }
        _ => turns.push((Role::User, user_text.to_string())),
    }
    turns
        .into_iter()
        .map(|(role, text)| match role {
            Role::Assistant => Msg::Assistant { text, calls: Vec::new() },
            _ => Msg::User { text, images: Vec::new() },
        })
        .collect()
}

fn has_images(msg: &Msg) -> bool {
    matches!(msg, Msg::User { images, .. } if !images.is_empty())
}

fn without_images(msg: &Msg) -> Msg {
    match msg {
        Msg::User { text, images } if !images.is_empty() => Msg::User {
            text: format!("{text} (It couldn't be shown: this model can't see images.)"),
            images: Vec::new(),
        },
        other => other.clone(),
    }
}

/// A request in Ollama's API.
fn ollama_request(model: &str, messages: &[Msg], tools: Option<&Value>, context: u32) -> Value {
    let messages: Vec<Value> = messages
        .iter()
        .map(|m| match m {
            Msg::System(text) => json!({ "role": "system", "content": text }),
            Msg::User { text, images } if images.is_empty() => json!({ "role": "user", "content": text }),
            Msg::User { text, images } => json!({ "role": "user", "content": text, "images": images }),
            Msg::Assistant { text, calls } if calls.is_empty() => json!({ "role": "assistant", "content": text }),
            Msg::Assistant { text, calls } => json!({
                "role": "assistant",
                "content": text,
                "tool_calls": calls
                    .iter()
                    .map(|c| json!({ "function": { "name": c.name, "arguments": c.arguments } }))
                    .collect::<Vec<_>>(),
            }),
            Msg::Tool { name, content, .. } => json!({ "role": "tool", "content": content, "tool_name": name }),
        })
        .collect();
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "options": { "num_ctx": context, "temperature": TEMPERATURE, "num_predict": MAX_REPLY_TOKENS },
    });
    if let Some(tools) = tools {
        body["tools"] = tools.clone();
    }
    body
}

/// A request in the OpenAI-compatible API.
fn openai_request(model: &str, messages: &[Msg], tools: Option<&Value>) -> Value {
    let messages: Vec<Value> = messages
        .iter()
        .map(|m| match m {
            Msg::System(text) => json!({ "role": "system", "content": text }),
            Msg::User { text, images } if images.is_empty() => json!({ "role": "user", "content": text }),
            Msg::User { text, images } => {
                let mut parts = vec![json!({ "type": "text", "text": text })];
                parts.extend(images.iter().map(|image| {
                    json!({ "type": "image_url", "image_url": { "url": format!("data:image/jpeg;base64,{image}") } })
                }));
                json!({ "role": "user", "content": parts })
            }
            Msg::Assistant { text, calls } if calls.is_empty() => json!({ "role": "assistant", "content": text }),
            Msg::Assistant { text, calls } => json!({
                "role": "assistant",
                "content": if text.is_empty() { Value::Null } else { json!(text) },
                "tool_calls": calls
                    .iter()
                    .map(|c| json!({
                        "id": c.id,
                        "type": "function",
                        "function": { "name": c.name, "arguments": c.arguments.to_string() },
                    }))
                    .collect::<Vec<_>>(),
            }),
            Msg::Tool { id, content, .. } => json!({ "role": "tool", "tool_call_id": id, "content": content }),
        })
        .collect();
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_REPLY_TOKENS,
    });
    if let Some(tools) = tools {
        body["tools"] = tools.clone();
        body["tool_choice"] = json!("auto");
    }
    body
}

/// The text and tool calls of an Ollama reply.
fn parse_ollama(reply: &Value) -> Result<(String, Vec<ToolCall>)> {
    if let Some(error) = reply["error"].as_str() {
        bail!("{error}");
    }
    Ok(parse_message(&reply["message"]))
}

/// The text and tool calls of an OpenAI-compatible reply.
fn parse_openai(reply: &Value) -> Result<(String, Vec<ToolCall>)> {
    if let Some(error) = reply["error"]["message"].as_str().or(reply["error"].as_str()) {
        bail!("{error}");
    }
    let message = &reply["choices"][0]["message"];
    if message.is_null() {
        bail!("the reply had no message");
    }
    Ok(parse_message(message))
}

fn parse_message(message: &Value) -> (String, Vec<ToolCall>) {
    let text = message["content"].as_str().unwrap_or_default().to_string();
    let calls = message["tool_calls"]
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .enumerate()
                .filter_map(|(i, call)| {
                    let function = &call["function"];
                    Some(ToolCall {
                        id: call["id"].as_str().map_or_else(|| format!("call_{i}"), str::to_string),
                        name: function["name"].as_str()?.to_string(),
                        arguments: arguments(&function["arguments"]),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    (text, calls)
}

/// Tool arguments, whether they came as an object or as JSON text.
fn arguments(value: &Value) -> Value {
    match value {
        Value::Object(_) => value.clone(),
        Value::String(text) => serde_json::from_str(text).unwrap_or_else(|_| json!({})),
        _ => json!({}),
    }
}

/// The reply as the user should see it, and the tool calls — including
/// ones a model wrote into its text instead of the API's field.
fn tidy(text: &str, calls: Vec<ToolCall>, known: &[String]) -> (String, Vec<ToolCall>) {
    let text = without_thinking(text);
    if !calls.is_empty() {
        return (text.trim().to_string(), calls);
    }
    let (rest, calls) = calls_in_text(&text, known);
    (rest.trim().to_string(), calls)
}

/// Drop `<think>…</think>` notes; a reply that only closes one (its template
/// opened it) loses everything before the closing tag.
fn without_thinking(text: &str) -> String {
    let mut text = match (text.find("</think>"), text.find("<think>")) {
        (Some(end), None) => text[end + "</think>".len()..].to_string(),
        _ => text.to_string(),
    };
    while let Some(start) = text.find("<think>") {
        let end = text[start..].find("</think>").map_or(text.len(), |e| start + e + "</think>".len());
        text.replace_range(start..end, "");
    }
    text
}

/// Tool calls written into the text: `<tool_call>{…}</tool_call>` blocks, or
/// a reply that is nothing but a JSON call of a known tool. Returns what's
/// left of the text, and the calls.
fn calls_in_text(text: &str, known: &[String]) -> (String, Vec<ToolCall>) {
    let as_call = |json: &str, i: usize| -> Option<ToolCall> {
        let value: Value = serde_json::from_str(json.trim()).ok()?;
        let name = value["name"].as_str()?.to_string();
        if !known.contains(&name) {
            return None;
        }
        let arguments = arguments(value.get("arguments").or(value.get("parameters")).unwrap_or(&Value::Null));
        Some(ToolCall { id: format!("call_text_{i}"), name, arguments })
    };
    let mut rest = String::new();
    let mut calls = Vec::new();
    let mut remaining = text;
    while let Some(start) = remaining.find("<tool_call>") {
        rest.push_str(&remaining[..start]);
        let after = &remaining[start + "<tool_call>".len()..];
        let (inner, next) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        match as_call(inner, calls.len()) {
            Some(call) => calls.push(call),
            None => rest.push_str(inner),
        }
        remaining = next;
    }
    rest.push_str(remaining);
    if calls.is_empty() {
        if let Some(call) = as_call(text, 0) {
            return (String::new(), vec![call]);
        }
    }
    (rest, calls)
}

/// The tools the model may call, in the function format both APIs share.
/// The screen tools are left out when the user switched screen viewing off.
pub fn tool_specs(allow_screen: bool) -> Value {
    let function = |name: &str, description: &str, parameters: Value| {
        json!({ "type": "function", "function": { "name": name, "description": description, "parameters": parameters } })
    };
    let mut tools = vec![
        function(
            "execute_command",
            &format!(
                "Run a {} command line on the user's {} computer, for device control or to check \
                 something (volume, brightness, a file, the system). The output comes back to you, \
                 never to the user, so sum it up in plain words.",
                platform::shell_name(),
                platform::os_name()
            ),
            json!({
                "type": "object",
                "properties": { "command": { "type": "string", "description": "The command line to run." } },
                "required": ["command"]
            }),
        ),
        function(
            "remember",
            "Store a lasting fact about the user or the world so you still know it in future sessions.",
            json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The fact, phrased so it stands alone." },
                    "kind": { "type": "string", "description": "Optional category, e.g. 'fact', 'project', 'like', 'dislike'." },
                    "importance": { "type": "integer", "description": "1 (minor) to 5 (very important). Higher is recalled first." }
                },
                "required": ["content"]
            }),
        ),
        function(
            "set_preference",
            "Save a user preference as a key and value (e.g. key 'name', value 'Alex').",
            json!({
                "type": "object",
                "properties": { "key": { "type": "string" }, "value": { "type": "string" } },
                "required": ["key", "value"]
            }),
        ),
        function(
            "recall",
            "Search your long-term memory for anything matching a keyword.",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        ),
        function(
            "walk_to",
            "Walk to a spot along the bottom of the screen.",
            json!({
                "type": "object",
                "properties": {
                    "position": { "type": "number", "description": "0.0 is the far left edge, 1.0 the far right." }
                },
                "required": ["position"]
            }),
        ),
        function(
            "express",
            "Change your facial expression to match your mood.",
            json!({
                "type": "object",
                "properties": {
                    "emotion": { "type": "string", "enum": ["neutral", "happy", "sad", "surprised", "angry", "relaxed"] }
                },
                "required": ["emotion"]
            }),
        ),
        function(
            "animate",
            "Move your body: wave, nod or shrug; sit or lie_down on the floor (idle stands back up); jump is a real hop.",
            json!({
                "type": "object",
                "properties": {
                    "clip": { "type": "string", "enum": ["wave", "nod", "shrug", "sit", "lie_down", "jump", "idle"] }
                },
                "required": ["clip"]
            }),
        ),
        function(
            "change_character",
            "Change the 3D character (the body) you appear as. Without a name, lists the characters there are.",
            json!({
                "type": "object",
                "properties": { "name": { "type": "string", "description": "Which character, by name." } }
            }),
        ),
        function(
            "list_apps",
            "Search the installed applications. Use it before opening one to get its exact name.",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string", "description": "Part of an app's name, e.g. 'firefox'." } },
                "required": ["query"]
            }),
        ),
        function(
            "open_app",
            "Open an installed application right away.",
            json!({
                "type": "object",
                "properties": { "name": { "type": "string", "description": "The app's name or id from list_apps." } },
                "required": ["name"]
            }),
        ),
        function(
            "click_at",
            "Walk to a point on the screen and click it with the real mouse. Get the point from find_on_screen. Only works if the user allowed control.",
            json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number" },
                    "y": { "type": "number" },
                    "double": { "type": "boolean", "description": "Double-click (e.g. desktop icons)." }
                },
                "required": ["x", "y"]
            }),
        ),
        function(
            "type_text",
            "Type text on the keyboard, e.g. after clicking into a field. Only works if the user allowed control.",
            json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }),
        ),
        function(
            "toggle_system",
            "Switch Bluetooth, Wi-Fi or mute on or off. Only the switches this system offers work.",
            json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "bluetooth, wifi or mute." },
                    "on": { "type": "boolean" }
                },
                "required": ["key", "on"]
            }),
        ),
    ];
    if allow_screen {
        tools.push(function(
            "look_at_screen",
            "Take a screenshot of the user's screen to see what they see, when they point at something on it or seeing it would clearly help. The user is told each time.",
            json!({ "type": "object", "properties": {} }),
        ));
        tools.push(function(
            "find_on_screen",
            "Look at the screen for something to click_at (an icon, a button, a menu item): you get a screenshot and read its coordinates off it. The user is told each time.",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string", "description": "What to look for." } },
                "required": ["query"]
            }),
        ));
    }
    Value::Array(tools)
}

fn tool_names(tools: &Value) -> Vec<String> {
    tools
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
        .collect()
}

fn truncate_err(s: &str) -> String {
    s.chars().take(400).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(role: Role, text: &str) -> ChatLine {
        ChatLine::new(role, text)
    }

    fn user(text: &str) -> Msg {
        Msg::User { text: text.into(), images: Vec::new() }
    }

    fn assistant(text: &str) -> Msg {
        Msg::Assistant { text: text.into(), calls: Vec::new() }
    }

    fn known() -> Vec<String> {
        tool_names(&tool_specs(true))
    }

    #[test]
    fn a_conversation_opens_with_the_user_and_alternates() {
        let history = [
            line(Role::Assistant, "Hi, I'm Tomo!"),
            line(Role::User, "hey"),
            line(Role::User, "you there?"),
            line(Role::System, "Tomo looked at your screen"),
            line(Role::Assistant, "Yes!"),
        ];
        assert_eq!(conversation(&history, "cool"), vec![user("hey\n\nyou there?"), assistant("Yes!"), user("cool")]);
    }

    #[test]
    fn a_new_line_after_an_unanswered_one_joins_it() {
        assert_eq!(conversation(&[line(Role::User, "first")], "second"), vec![user("first\n\nsecond")]);
        assert_eq!(conversation(&[], "hello"), vec![user("hello")]);
    }

    #[test]
    fn it_looks_for_ollama_then_lm_studio() {
        let auto = server_candidates(LlmProvider::Auto, "").unwrap();
        assert_eq!(auto, vec![(Api::Ollama, OLLAMA_URL.to_string()), (Api::OpenAi, LM_STUDIO_URL.to_string())]);
        let lm = server_candidates(LlmProvider::LmStudio, "http://pc:1234").unwrap();
        assert_eq!(lm, vec![(Api::OpenAi, "http://pc:1234/v1".to_string())]);
        let ollama = server_candidates(LlmProvider::Ollama, "http://box:11434/v1/").unwrap();
        assert_eq!(ollama, vec![(Api::Ollama, "http://box:11434".to_string())]);
        let guessed = server_candidates(LlmProvider::Auto, "http://localhost:11434").unwrap();
        assert_eq!(guessed[0].0, Api::Ollama);
        assert!(server_candidates(LlmProvider::OpenAiCompatible, "").is_err());
        assert_eq!(openai_root("http://gpu:8080/openai/v1"), "http://gpu:8080/openai/v1");
    }

    #[test]
    fn it_picks_a_local_chat_model() {
        let models = [
            ModelInfo { name: "gemma4:31b-cloud".into(), local: false },
            ModelInfo { name: "nomic-embed-text".into(), local: true },
            ModelInfo { name: "qwen2.5:7b".into(), local: true },
        ];
        assert_eq!(pick_model(&models).as_deref(), Some("qwen2.5:7b"));
        assert_eq!(pick_model(&models[..2]), None);
    }

    #[test]
    fn ollama_requests_carry_the_context_images_and_tool_calls() {
        let call = ToolCall { id: "call_0".into(), name: "animate".into(), arguments: json!({ "clip": "wave" }) };
        let messages = [
            Msg::System("be nice".into()),
            Msg::User { text: "look".into(), images: vec!["QUJD".into()] },
            Msg::Assistant { text: String::new(), calls: vec![call] },
            Msg::Tool { id: "call_0".into(), name: "animate".into(), content: "waving".into() },
        ];
        let tools = tool_specs(false);
        let body = ollama_request("qwen2.5:7b", &messages, Some(&tools), 8192);
        assert_eq!(body["options"]["num_ctx"], 8192);
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][1]["images"][0], "QUJD");
        assert_eq!(body["messages"][2]["tool_calls"][0]["function"]["arguments"]["clip"], "wave");
        assert_eq!(body["messages"][3]["role"], "tool");
        assert_eq!(body["messages"][3]["tool_name"], "animate");
        assert!(body["tools"].as_array().is_some_and(|t| !t.is_empty()));
        assert!(ollama_request("m", &messages, None, 4096).get("tools").is_none());
    }

    #[test]
    fn openai_requests_use_data_urls_and_json_text_arguments() {
        let call = ToolCall { id: "abc".into(), name: "walk_to".into(), arguments: json!({ "position": 0.5 }) };
        let messages = [
            Msg::User { text: "look".into(), images: vec!["QUJD".into()] },
            Msg::Assistant { text: String::new(), calls: vec![call] },
            Msg::Tool { id: "abc".into(), name: "walk_to".into(), content: "ok".into() },
        ];
        let tools = tool_specs(true);
        let body = openai_request("local-model", &messages, Some(&tools));
        assert_eq!(body["messages"][0]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,QUJD");
        assert_eq!(body["messages"][1]["content"], Value::Null);
        assert_eq!(body["messages"][1]["tool_calls"][0]["function"]["arguments"], "{\"position\":0.5}");
        assert_eq!(body["messages"][2]["tool_call_id"], "abc");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn replies_are_read_in_both_formats() {
        let ollama = json!({ "message": { "role": "assistant", "content": "", "tool_calls": [
            { "function": { "name": "express", "arguments": { "emotion": "happy" } } }
        ] } });
        let (text, calls) = parse_ollama(&ollama).unwrap();
        assert!(text.is_empty());
        assert_eq!(calls[0].name, "express");
        assert_eq!(calls[0].arguments["emotion"], "happy");
        assert_eq!(calls[0].id, "call_0");

        let openai = json!({ "choices": [ { "message": { "content": "Sure!", "tool_calls": [
            { "id": "t1", "type": "function", "function": { "name": "animate", "arguments": "{\"clip\":\"wave\"}" } }
        ] } } ] });
        let (text, calls) = parse_openai(&openai).unwrap();
        assert_eq!(text, "Sure!");
        assert_eq!((calls[0].id.as_str(), calls[0].arguments["clip"].as_str()), ("t1", Some("wave")));

        assert!(parse_ollama(&json!({ "error": "model not found" })).is_err());
        assert!(parse_openai(&json!({ "error": { "message": "bad request" } })).is_err());
    }

    #[test]
    fn thinking_stays_out_of_the_reply() {
        assert_eq!(without_thinking("<think>hmm, a wave?</think>Hi there!"), "Hi there!");
        assert_eq!(without_thinking("planning…</think>Okay!"), "Okay!");
        assert_eq!(without_thinking("No notes."), "No notes.");
    }

    #[test]
    fn tool_calls_written_into_the_text_are_understood() {
        let text = "Sure! <tool_call>{\"name\": \"animate\", \"arguments\": {\"clip\": \"wave\"}}</tool_call>";
        let (rest, calls) = tidy(text, Vec::new(), &known());
        assert_eq!(rest, "Sure!");
        assert_eq!(calls[0].name, "animate");
        assert_eq!(calls[0].arguments["clip"], "wave");

        let bare = "{\"name\": \"walk_to\", \"parameters\": {\"position\": 0.2}}";
        let (rest, calls) = tidy(bare, Vec::new(), &known());
        assert!(rest.is_empty());
        assert_eq!(calls[0].arguments["position"], 0.2);

        let (rest, calls) = tidy("{\"name\": \"rm_rf\"}", Vec::new(), &known());
        assert!(calls.is_empty(), "only known tools");
        assert_eq!(rest, "{\"name\": \"rm_rf\"}");
    }

    #[test]
    fn screen_tools_are_offered_only_when_allowed() {
        let with = tool_names(&tool_specs(true));
        let without = tool_names(&tool_specs(false));
        for tool in ["look_at_screen", "find_on_screen"] {
            assert!(with.iter().any(|n| n == tool));
            assert!(!without.iter().any(|n| n == tool));
        }
        let mut unique = with.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), with.len(), "tool names are unique");
        for tool in tool_specs(true).as_array().unwrap() {
            assert_eq!(tool["type"], "function");
            assert_eq!(tool["function"]["parameters"]["type"], "object", "{}", tool["function"]["name"]);
            assert!(!tool["function"]["description"].as_str().unwrap_or("").is_empty());
        }
    }

    #[test]
    fn errors_are_cut_short() {
        assert_eq!(truncate_err(&"é".repeat(1000)).chars().count(), 400);
        assert_eq!(truncate_err("short"), "short");
    }

    /// A stand-in model server: answers each request with the next canned
    /// reply for its path, and keeps the requests' bodies.
    async fn fake_server(replies: Vec<(&'static str, Value)>) -> (String, Arc<std::sync::Mutex<Vec<Value>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            let mut replies = replies;
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut request = Vec::new();
                let mut buf = [0u8; 65536];
                // Read the head, then as much body as it announces.
                loop {
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&request);
                    if let Some(head_end) = text.find("\r\n\r\n") {
                        let length = text[..head_end]
                            .lines()
                            .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                            .unwrap_or(0);
                        if request.len() >= head_end + 4 + length {
                            break;
                        }
                    }
                }
                let text = String::from_utf8_lossy(&request).to_string();
                let path = text.split_whitespace().nth(1).unwrap_or("").to_string();
                if let Some(body) = text.split("\r\n\r\n").nth(1) {
                    if let Ok(json) = serde_json::from_str::<Value>(body) {
                        log.lock().unwrap().push(json);
                    }
                }
                let index = replies.iter().position(|(p, _)| path.ends_with(p));
                let body = match index {
                    Some(i) if replies.iter().filter(|(p, _)| path.ends_with(p)).count() > 1 => replies.remove(i).1,
                    Some(i) => replies[i].1.clone(),
                    None => json!({ "error": "not found" }),
                };
                let body = body.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        (url, seen)
    }

    fn client(url: &str, provider: LlmProvider) -> AiClient {
        let tmp = std::env::temp_dir().join(format!("tomo-ai-test-{}", std::process::id()));
        let mut cfg = Config::for_tests(&tmp);
        cfg.llm_provider = provider;
        cfg.llm_url = url.to_string();
        let executor = Executor::new(false, tmp.join("audit.log"), Vec::new());
        AiClient::new(
            cfg,
            Db::open_in_memory().unwrap(),
            executor,
            Arc::new(RwLock::new(SystemCatalog::default())),
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[tokio::test]
    async fn a_turn_with_ollama_runs_the_tools_then_answers() {
        let (url, seen) = fake_server(vec![
            ("/api/tags", json!({ "models": [
                { "name": "gemma4:31b-cloud", "remote_host": "https://ollama.com" },
                { "name": "qwen2.5:0.5b" }
            ] })),
            ("/api/chat", json!({ "message": { "role": "assistant", "content": "", "tool_calls": [
                { "function": { "name": "animate", "arguments": { "clip": "wave" } } }
            ] }, "done": true })),
            ("/api/chat", json!({ "message": { "role": "assistant", "content": "<think>ok</think>Hi! *waves*" }, "done": true })),
        ])
        .await;
        let ai = client(&url, LlmProvider::Ollama);
        let (ui, mut from_brain) = tokio::sync::mpsc::unbounded_channel();
        let reply = ai.respond("wave at me", &ui).await.unwrap();
        assert_eq!(reply, "Hi! *waves*");
        let mut waved = false;
        while let Ok(event) = from_brain.try_recv() {
            waved |= matches!(event, BrainToUi::Animate(ref clip) if clip == "wave");
        }
        assert!(waved, "the body got the gesture");
        let requests = seen.lock().unwrap().clone();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["model"], "qwen2.5:0.5b", "the local model, not the cloud one");
        assert_eq!(requests[1]["messages"].as_array().unwrap().last().unwrap()["role"], "tool");
    }

    #[tokio::test]
    async fn a_turn_with_lm_studio_uses_the_openai_api() {
        let (url, seen) = fake_server(vec![
            ("/v1/models", json!({ "data": [ { "id": "text-embedding-nomic" }, { "id": "qwen2.5-7b-instruct" } ] })),
            ("/v1/chat/completions", json!({ "choices": [ { "message": { "role": "assistant", "content": "Hello from LM Studio." } } ] })),
        ])
        .await;
        let ai = client(&url, LlmProvider::LmStudio);
        let (ui, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(ai.respond("hi", &ui).await.unwrap(), "Hello from LM Studio.");
        let requests = seen.lock().unwrap().clone();
        assert_eq!(requests[0]["model"], "qwen2.5-7b-instruct");
        assert_eq!(requests[0]["messages"][0]["role"], "system");
    }

    #[tokio::test]
    async fn with_no_server_tomo_says_how_to_get_one() {
        // Nothing listens on this port.
        let ai = client("http://127.0.0.1:9", LlmProvider::Ollama);
        let (ui, _rx) = tokio::sync::mpsc::unbounded_channel();
        let reply = ai.respond("hi", &ui).await.unwrap();
        assert!(reply.contains("ollama pull"), "{reply}");
    }
}
