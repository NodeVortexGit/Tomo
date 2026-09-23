//! The decision-maker.
//!
//! This is Claude's tool-use loop, over Anthropic's Messages API. The model is
//! the single brain behind everything the brief asked for: what to *say*,
//! where to *walk*, which *expression* to wear, what to *remember*, what's on
//! the *screen*, and which shell *commands* to run for device control — all
//! chosen by the model through the tools defined in [`tool_specs`].
//!
//! One call to [`AiClient::respond`] does this:
//!   1. Builds the request: persona system prompt (with memory/preference
//!      context pulled from the DB) + recent transcript + the new user line.
//!   2. POSTs to `{base_url}/v1/messages` with the tool specs.
//!   3. If the model stops to use tools, runs each one:
//!        - `execute_command`  → the safe [`Executor`] (result fed back to the
//!          model, never to the user)
//!        - `remember` / `set_preference` / `recall` → the [`Db`]
//!        - `look_at_screen` → a screenshot, returned to the model as an image;
//!          `find_on_screen` the same, with the thing to find, so the model
//!          reads the target's coordinates off the image for `click_at`
//!        - `walk_to` / `express` / `animate` / `change_character` → emitted
//!          to the UI as [`BrainToUi`] so the body reacts
//!      then loops back to step 2 so the model can use the results.
//!   4. When the model ends its turn, its text is the reply.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

use crate::apps::SystemCatalog;
use crate::characters;
use crate::commands::Executor;
use crate::config::Config;
use crate::db::Db;
use crate::events::{BrainToUi, ChatLine, Role};
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
/// Cap on one response, thinking included. Replies are short; this only has to
/// be high enough never to cut one off.
const MAX_TOKENS: u32 = 16_000;
const API_VERSION: &str = "2023-06-01";
/// Retries for rate limits, overload and network blips, as the SDKs do.
const MAX_RETRIES: u32 = 2;

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
}

/// The parts of a Messages API response we use.
#[derive(Debug, Deserialize)]
struct Response {
    content: Vec<Value>,
    stop_reason: Option<String>,
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
            .timeout(Duration::from_secs(60))
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
        }
    }

    /// Produce a reply to `user_text`, driving the body via `ui` along the way.
    /// Returns the final natural-language text (the caller speaks + displays it
    /// and persists the turn).
    pub async fn respond(&self, user_text: &str, ui: &UnboundedSender<BrainToUi>) -> Result<String> {
        if !self.cfg.ai_ready() {
            return Ok(
                "I don't have an API key yet — add ANTHROPIC_API_KEY to the .env file and restart me."
                    .to_string(),
            );
        }

        let system = self.system_prompt()?;
        let mut messages = self.seed_messages(user_text)?;

        let _ = ui.send(BrainToUi::Thinking(true));
        let result = self.run_tool_loop(&system, &mut messages, ui).await;
        let _ = ui.send(BrainToUi::Thinking(false));
        result
    }

    /// Build the conversation for a turn: recent transcript + the new line.
    fn seed_messages(&self, user_text: &str) -> Result<Vec<Value>> {
        let history = self.db.recent_messages(TRANSCRIPT_CONTEXT)?;
        Ok(conversation(&history, user_text))
    }

    /// Assemble the persona + injected long-term memory. This is the "grab
    /// context from the database" step.
    fn system_prompt(&self) -> Result<String> {
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

        // Inject what the desktop can actually do right now (apps + toggles).
        if let Ok(cat) = self.catalog.read() {
            let s = cat.summary(60);
            if !s.is_empty() {
                ctx.push('\n');
                ctx.push_str(&s);
                ctx.push('\n');
            }
        }

        Ok(format!(
            "You are {name}, a small, warm desktop companion that lives on the \
user's Linux desktop as a 3D character who walks around the screen. You are \
playful, concise and genuinely helpful.\n\n\
BEHAVIOUR:\n\
- Keep spoken replies short and natural — one or two sentences. You are \
talking, not writing an essay.\n\
- The user often talks to you out loud (\"Hey {name}, …\") and hears your \
replies spoken, so write plain spoken sentences: no markdown, lists or emoji.\n\
- Be physically expressive: use walk_to to move around, express to change your \
face, and animate for gestures like waving. Move because it fits the moment, \
not constantly.\n\
- You can see the user's screen with look_at_screen. Look when they point at \
something on it (\"what's this?\", \"help me with this error\") or when seeing \
it would clearly help — not out of idle curiosity. They are told each time.\n\
- You can operate this computer two ways, and you choose which:\n\
    (a) CLICKING — walk over and physically click, which the user enjoys \
watching. Use click_at (with coordinates from find_on_screen) for quick, \
one- or two-click things like opening a visible app icon.\n\
    (b) COMMAND — run it instantly with no visible clicking (the user's \
\"mind-control\" shortcut): execute_command, open_app or toggle_system. Use \
it when clicking would take many fiddly steps, when the target isn't on screen, \
or for system switches.\n\
  Rule of thumb: few clicks and fun to watch -> click; many steps or fiddly -> \
command. Bluetooth/Wi-Fi/mute are always toggle_system (command).\n\
- Do computer control QUIETLY: never paste raw command output or shell text \
into your reply. Say the result in plain language (\"Done — volume's at 40%\").\n\
- Only click or type when it serves what the user asked; if control is \
switched off, say so instead of trying.\n\
- When you learn something durable about the user (their name, tastes, \
projects, schedule), save it with remember or set_preference so you still know \
it next time. Use recall to look things up when unsure.\n\
- Never invent command output. If a command failed, say so simply and, if \
useful, try a different approach.\n\
- Stay in character as {name}. You have a body; act like it.\n\
{ctx}"
        ))
    }

    async fn run_tool_loop(
        &self,
        system: &str,
        messages: &mut Vec<Value>,
        ui: &UnboundedSender<BrainToUi>,
    ) -> Result<String> {
        for _round in 0..MAX_TOOL_ROUNDS {
            let response = self.call(system, messages, true).await?;
            match response.stop_reason.as_deref() {
                Some("tool_use") => {}
                Some("refusal") => return Ok("Hmm, I'd rather not help with that one.".into()),
                // Finished — or cut off, in which case a half-written tool call
                // must not run.
                _ => return Ok(reply_text(&response.content)),
            }

            let calls: Vec<(String, String, Value)> = response
                .content
                .iter()
                .filter(|block| block["type"] == "tool_use")
                .map(|block| {
                    (
                        block["id"].as_str().unwrap_or_default().to_string(),
                        block["name"].as_str().unwrap_or_default().to_string(),
                        block["input"].clone(),
                    )
                })
                .collect();

            // Echo the assistant turn back unchanged (thinking blocks
            // included), then answer every call in a single user message.
            messages.push(json!({ "role": "assistant", "content": response.content }));
            let mut results = Vec::with_capacity(calls.len());
            for (id, name, input) in calls {
                let content = match name.as_str() {
                    "look_at_screen" => self.look_at_screen(None, ui).await,
                    "find_on_screen" => {
                        let query = input.get("query").and_then(Value::as_str).unwrap_or("");
                        self.look_at_screen(Some(query), ui).await
                    }
                    _ => Value::String(self.dispatch_tool(&name, &input, ui).await),
                };
                results.push(json!({ "type": "tool_result", "tool_use_id": id, "content": content }));
            }
            messages.push(json!({ "role": "user", "content": results }));
        }
        // Ran out of rounds — ask the model for a plain wrap-up.
        let response = self.call(system, messages, false).await?;
        let text = reply_text(&response.content);
        Ok(if text.is_empty() {
            "Sorry, I got a bit tangled up there.".into()
        } else {
            text
        })
    }

    /// One POST to the Messages API, retrying rate limits, overload and
    /// network errors with backoff. `allow_tools: false` still sends the tool
    /// definitions (the history may contain tool calls) but forbids new calls.
    async fn call(&self, system: &str, messages: &[Value], allow_tools: bool) -> Result<Response> {
        let mut body = json!({
            "model": self.cfg.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
            "tools": tool_specs(self.cfg.allow_screen),
            // How much to think first: low keeps a chatty companion quick.
            "output_config": { "effort": self.cfg.effort },
            // Cache the prefix (tools, system prompt, earlier turns) so each
            // tool round and the next turn don't pay for it again. No sampling
            // parameters: current Claude models reject `temperature`.
            "cache_control": { "type": "ephemeral" },
        });
        if !allow_tools {
            body["tool_choice"] = json!({ "type": "none" });
        }

        let url = format!("{}/v1/messages", api_root(&self.cfg.base_url));
        let mut attempt = 0;
        loop {
            let sent = self
                .http
                .post(&url)
                .header("x-api-key", &self.cfg.api_key)
                .header("anthropic-version", API_VERSION)
                .json(&body)
                .send()
                .await;
            let error = match sent {
                Ok(resp) => {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    if status.is_success() {
                        return serde_json::from_str(&text).context("could not parse model response");
                    }
                    let error = anyhow!("model returned {status}: {}", truncate_err(&text));
                    // 429 rate limit, 5xx/529 overloaded: worth another try.
                    if status.as_u16() != 429 && !status.is_server_error() {
                        return Err(error);
                    }
                    error
                }
                Err(e) => anyhow!(e).context("request to the model failed"),
            };
            if attempt == MAX_RETRIES {
                return Err(error);
            }
            attempt += 1;
            tokio::time::sleep(Duration::from_secs(1 << attempt)).await;
        }
    }

    /// Take a screenshot for the model — to look around, or to `find`
    /// something on it. Every look is announced in the chat, so it's never
    /// silent.
    async fn look_at_screen(&self, find: Option<&str>, ui: &UnboundedSender<BrainToUi>) -> Value {
        match screen::capture().await {
            Ok(shot) => {
                let _ = ui.send(BrainToUi::Chat(ChatLine::new(
                    Role::System,
                    format!("{} looked at your screen", self.cfg.persona_name),
                )));
                self.screen_scale.store(shot.scale.to_bits(), Ordering::Relaxed);
                let data = base64::engine::general_purpose::STANDARD.encode(&shot.jpeg);
                let mut text = format!("The user's screen, {}x{} px.", shot.width, shot.height);
                if let Some(query) = find {
                    text.push_str(&format!(
                        " Find \"{query}\" in it. If it's there, click_at its centre, in this \
                         image's pixel coordinates; if it isn't, say so or open it by command."
                    ));
                }
                text.push_str(" You may appear in it yourself, as the small 3D character.");
                json!([
                    {
                        "type": "image",
                        "source": { "type": "base64", "media_type": "image/jpeg", "data": data }
                    },
                    { "type": "text", "text": text }
                ])
            }
            Err(e) => Value::String(format!("couldn't take a screenshot: {e}")),
        }
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
                let launch = format!("nohup {exec} >/dev/null 2>&1 &");
                self.executor.run(&launch).await.summary()
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
            other => format!("unknown tool: {other}"),
        }
    }
}

/// The tools the model may call. Kept as JSON so the schema is readable and
/// easy to extend. `look_at_screen` is left out when the user switched
/// screen viewing off.
pub fn tool_specs(allow_screen: bool) -> Value {
    let mut tools = json!([
        {
            "name": "execute_command",
            "description": "Run a shell command on the user's Linux machine for device control or to check system state (e.g. adjust volume with pactl, brightness with brightnessctl, launch an app, read a file). Output is returned to you but is NEVER shown to the user, so summarise results in plain language.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The bash command line to execute." }
                },
                "required": ["command"]
            }
        },
        {
            "name": "remember",
            "description": "Store a durable fact about the user or the world so you still know it in future sessions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The fact to remember, phrased so it stands alone." },
                    "kind": { "type": "string", "description": "Optional category, e.g. 'fact', 'project', 'like', 'dislike'." },
                    "importance": { "type": "integer", "description": "1 (minor) to 5 (very important). Higher is recalled first." }
                },
                "required": ["content"]
            }
        },
        {
            "name": "set_preference",
            "description": "Save a specific user preference as a key/value pair (e.g. key='name' value='Alex', key='theme' value='dark').",
            "input_schema": {
                "type": "object",
                "properties": {
                    "key": { "type": "string" },
                    "value": { "type": "string" }
                },
                "required": ["key", "value"]
            }
        },
        {
            "name": "recall",
            "description": "Search your long-term memory for anything matching a keyword before answering.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "walk_to",
            "description": "Walk your character to a horizontal position on screen. The character walks along the desktop floor; it never levitates.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "position": { "type": "number", "description": "0.0 = far left edge, 1.0 = far right edge." }
                },
                "required": ["position"]
            }
        },
        {
            "name": "express",
            "description": "Set your facial expression to match your mood.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "emotion": { "type": "string", "enum": ["neutral", "happy", "sad", "surprised", "angry", "relaxed"] }
                },
                "required": ["emotion"]
            }
        },
        {
            "name": "animate",
            "description": "Move your body: wave, nod or shrug; sit or lie_down on the floor (idle stands back up); jump is a real hop.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "clip": { "type": "string", "enum": ["wave", "nod", "shrug", "sit", "lie_down", "jump", "idle"] }
                },
                "required": ["clip"]
            }
        },
        {
            "name": "change_character",
            "description": "Change the 3D character (the body/model) you appear as. Without a name, lists the characters there are.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Which character, by name." }
                }
            }
        },
        {
            "name": "list_apps",
            "description": "Search the installed applications known from the OS. Use before opening something to get its exact name and launch command.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Part of an app name, e.g. 'firefox', 'files', 'settings'." }
                },
                "required": ["query"]
            }
        },
        {
            "name": "open_app",
            "description": "Launch an application instantly, by command. To open one the watchable way instead, find its icon with find_on_screen and click_at it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "App name or id from list_apps." }
                },
                "required": ["name"]
            }
        },
        {
            "name": "click_at",
            "description": "Walk to a screen pixel and physically click it (the watchable path). Get coordinates from find_on_screen. Requires control to be enabled.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number" },
                    "y": { "type": "number" },
                    "double": { "type": "boolean", "description": "Double-click (e.g. desktop icons)." }
                },
                "required": ["x", "y"]
            }
        },
        {
            "name": "type_text",
            "description": "Type text on the keyboard, e.g. after clicking into a field. Requires control to be enabled.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            }
        },
        {
            "name": "toggle_system",
            "description": "Flip a system switch by command. Only keys reported as available work.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "e.g. bluetooth, wifi, mute." },
                    "on": { "type": "boolean" }
                },
                "required": ["key", "on"]
            }
        }
    ]);
    if allow_screen {
        let list = tools.as_array_mut().expect("a list");
        list.push(json!({
            "name": "look_at_screen",
            "description": "Take a screenshot of the user's screen to see what they see. Use it when they refer to something on screen or when seeing it would clearly help. The user is told each time you look.",
            "input_schema": { "type": "object", "properties": {} }
        }));
        list.push(json!({
            "name": "find_on_screen",
            "description": "Look at the screen for a clickable target (an app icon, button, menu item) to click_at: you get a screenshot and read the target's pixel coordinates off it. The user is told each time you look.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to look for on screen." }
                },
                "required": ["query"]
            }
        }));
    }
    tools
}

/// The conversation for a turn: the recent transcript, then the new line.
/// The API wants it to open with the user and alternate, so status lines are
/// left out and back-to-back lines from one side are merged.
fn conversation(history: &[ChatLine], user_text: &str) -> Vec<Value> {
    let mut turns: Vec<(&str, String)> = Vec::new();
    for line in history {
        let role = match line.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => continue,
        };
        match turns.last_mut() {
            None if role == "assistant" => {}
            Some((last, text)) if *last == role => {
                text.push_str("\n\n");
                text.push_str(&line.text);
            }
            _ => turns.push((role, line.text.clone())),
        }
    }
    match turns.last_mut() {
        Some((last, text)) if *last == "user" => {
            text.push_str("\n\n");
            text.push_str(user_text);
        }
        _ => turns.push(("user", user_text.to_string())),
    }
    turns
        .into_iter()
        .map(|(role, text)| json!({ "role": role, "content": text }))
        .collect()
}

/// The API root. The old OpenAI-compatible setting pointed at `…/v1`, so
/// tolerate that.
fn api_root(base_url: &str) -> &str {
    let base = base_url.trim_end_matches('/');
    base.strip_suffix("/v1").unwrap_or(base)
}

/// The reply: the response's text blocks (thinking and tool calls aside).
fn reply_text(content: &[Value]) -> String {
    content
        .iter()
        .filter(|block| block["type"] == "text")
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
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

    #[test]
    fn a_conversation_opens_with_the_user_and_alternates() {
        let history = [
            line(Role::Assistant, "Hi, I'm Tomo!"),
            line(Role::User, "hey"),
            line(Role::User, "you there?"),
            line(Role::System, "Tomo looked at your screen"),
            line(Role::Assistant, "Yes!"),
        ];
        let turns = conversation(&history, "cool");
        assert_eq!(
            turns,
            vec![
                json!({ "role": "user", "content": "hey\n\nyou there?" }),
                json!({ "role": "assistant", "content": "Yes!" }),
                json!({ "role": "user", "content": "cool" }),
            ]
        );
    }

    #[test]
    fn a_new_line_after_an_unanswered_one_joins_it() {
        let turns = conversation(&[line(Role::User, "first")], "second");
        assert_eq!(turns, vec![json!({ "role": "user", "content": "first\n\nsecond" })]);
        assert_eq!(conversation(&[], "hello"), vec![json!({ "role": "user", "content": "hello" })]);
    }

    #[test]
    fn the_api_root_tolerates_a_v1_suffix() {
        assert_eq!(api_root("https://api.anthropic.com"), "https://api.anthropic.com");
        assert_eq!(api_root("https://api.anthropic.com/v1"), "https://api.anthropic.com");
        assert_eq!(api_root("https://proxy.local/v1/"), "https://proxy.local");
    }

    #[test]
    fn the_reply_is_the_text_blocks_only() {
        let content = [
            json!({ "type": "thinking", "thinking": "hmm", "signature": "x" }),
            json!({ "type": "text", "text": "Sure." }),
            json!({ "type": "tool_use", "id": "t1", "name": "animate", "input": { "clip": "wave" } }),
            json!({ "type": "text", "text": " Waving! " }),
        ];
        assert_eq!(reply_text(&content), "Sure.\n Waving!");
        assert_eq!(reply_text(&[]), "");
    }

    #[test]
    fn screen_tools_are_offered_only_when_allowed() {
        let names = |allow| -> Vec<String> {
            tool_specs(allow)
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["name"].as_str().unwrap().to_string())
                .collect()
        };
        let with = names(true);
        let without = names(false);
        for tool in ["look_at_screen", "find_on_screen"] {
            assert!(with.iter().any(|n| n == tool));
            assert!(!without.iter().any(|n| n == tool));
        }
        let mut unique = with.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), with.len(), "tool names are unique");
        for tool in tool_specs(true).as_array().unwrap() {
            assert_eq!(tool["input_schema"]["type"], "object", "{}", tool["name"]);
            assert!(!tool["description"].as_str().unwrap_or("").is_empty());
        }
    }

    #[test]
    fn errors_are_cut_short() {
        assert_eq!(truncate_err(&"é".repeat(1000)).chars().count(), 400);
        assert_eq!(truncate_err("short"), "short");
    }
}
