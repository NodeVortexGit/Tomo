"""The brain's thinking: a local model (Ollama or LM Studio) and its tools.

Each turn the model gets the persona (with what the computer is: sysinfo.py),
what Tomo remembers and the recent conversation — with any images the user
sent and the notes on them (attachments.py) — and answers, often by calling
tools first: run a command, open an app, set the volume, move, remember
something, look at the screen, pick the character's voice. The loop runs
those, feeds the results back, and repeats until the model answers in words.

The model also chooses each character's voice, once, the first time the
character appears: it's shown the character's picture and calls set_voice
(:meth:`AiClient.ensure_voice`).

Guards learned the hard way:
  * a reply that only *promises* to act ("I'll open them!") gets one reminder
    to act now — otherwise the turn ends with nothing done;
  * a reply that claims success though every action failed goes back once to
    be checked against the tool results;
  * gestures that need no reply end the turn a round sooner;
  * a 60-second budget, then a plain wrap-up;
  * one model for the whole session, kept even after an error.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import httpx

from . import attachments as attached, characters, platform, screen, sysinfo, volume
from .apps import SystemCatalog, checked_launch
from .commands import Executor
from .config import DEFAULT_VOICE_BG, DEFAULT_VOICE_FEMALE, DEFAULT_VOICE_MALE, Config, LlmProvider
from .db import Store
from .events import (Animate, Attachment, BrainToUi, Characters, Chat, ChatLine, ClickAt, ControlMode, Emote,
                     LoadCharacter, Role, Thinking, TypeText, WalkTo)
from .language import Language

log = logging.getLogger(__name__)

# Guard against a model that keeps calling tools forever.
MAX_TOOL_ROUNDS = 6
# No more tool rounds after this long: the user is waiting.
TURN_BUDGET = 60.0
# Tools that act on the computer, whose results say whether that worked.
ACTIONS = ("execute_command", "open_app", "set_volume", "toggle_system")
# Tools that only move the body: done at once, nothing to report back.
GESTURES = ("express", "animate", "walk_to")
CHECK_RESULTS = ("(Check your answer against the tool results: none of the actions worked. "
                 "Say plainly what did not happen, and why.)")
ACT_NOW = ("(You haven't done it yet: nothing happens after your answer. Call the tools now, "
           "all of them in this reply, then say what they did.)")
WRAP_UP = ("(That's all the tries there's time for. Don't call any more tools: tell me in one "
           "or two sentences what you found, or what went wrong.)")
VOICE_CHOICE = ("You choose the voice of a 3D character who lives on the user's desktop. Look at the "
                "character's picture and details, then call set_voice once: male or female.")
# The character's picture, as the model sees it when choosing its voice (px).
VOICE_PICTURE_EDGE = 384
# How much transcript to show in the chat after a restart…
TRANSCRIPT_CONTEXT = 20
# …and how much the model reads each turn (fewer tokens).
MODEL_CONTEXT = 12
OLLAMA_URL = "http://127.0.0.1:11434"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
MAX_REPLY_TOKENS = 1024
TEMPERATURE = 0.7
# A local model on a CPU can take its time — the first answer longer still.
REQUEST_TIMEOUT = 600.0
PROBE_TIMEOUT = 3.0
# The brain's model: tools, English and Bulgarian, and it sees images; on an
# 8 GB graphics card it runs entirely on the card. Chosen when TOMO_LLM_MODEL
# is empty and it's there, and suggested when there's no model at all.
SUGGESTED_MODEL = "qwen3.5:9b"
# Models smaller than this (billions of parameters) answer in words but don't
# manage tools: asked to run a command, they make up its result.
TOO_SMALL_FOR_TOOLS = 1.5


class Api(Enum):
    OLLAMA = "ollama"  # its own /api/chat
    OPENAI = "openai"  # /v1/chat/completions: LM Studio and others


@dataclass(frozen=True)
class Server:
    api: Api
    url: str
    model: str
    small: bool = False


@dataclass(frozen=True)
class ModelInfo:
    name: str
    local: bool = True
    params: float | None = None  # billions, when the server says

    @property
    def small(self) -> bool:
        return self.params is not None and self.params < TOO_SMALL_FOR_TOOLS


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Msg:
    """One message of the conversation, whichever API it's sent in."""

    role: str  # system | user | assistant | tool
    text: str = ""
    images: list[str] = field(default_factory=list)  # base64 JPEG (user)
    calls: list[ToolCall] = field(default_factory=list)  # assistant
    tool_id: str = ""  # tool
    tool_name: str = ""  # tool


Ui = Callable[[BrainToUi], None]


class AiClient:
    def __init__(self, cfg: Config, db: Store, executor: Executor, catalog: "SharedCatalog",
                 control_allowed: threading.Event, http: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self.executor = executor
        self.catalog = catalog
        self.control_allowed = control_allowed
        self.http = http or httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        # Screen pixels per pixel of the last screenshot the model saw.
        self.screen_scale = 1.0
        self._server: Server | None = None
        self._server_lock = asyncio.Lock()
        self._told_small = False
        self._choosing: set[str] = set()  # characters whose voice is being chosen

    def tools(self) -> list[dict]:
        cfg = self.cfg
        return tool_specs(cfg.allow_screen, (cfg.tts_voice_male, cfg.tts_voice_female, cfg.tts_voice_bg))

    # ---- a turn ---------------------------------------------------------------------

    async def respond(self, user_text: str, ui: Ui, images: tuple[Attachment, ...] = ()) -> str:
        """Reply to ``user_text`` (and the ``images`` sent with it), driving
        the body through ``ui`` on the way."""
        try:
            server = await self.server()
        except LookupError as problem:
            log.warning("no model to think with: %s", problem)
            return str(problem)
        if server.small and not self._told_small:
            self._told_small = True
            ui(Chat(ChatLine(Role.SYSTEM, f"{self.cfg.persona_name} is thinking with {server.model}, a model too "
                                          "small to use tools, so running commands, opening apps and clicking "
                                          f"won't work. A bigger model fixes that: ollama pull {SUGGESTED_MODEL}")))
        history = self.db.recent_messages(MODEL_CONTEXT)
        # Pictures alone: answer in the language the user last wrote in.
        spoken = user_text or next((line.text for line in reversed(history) if line.role == Role.USER), "")
        messages = [Msg("system", self.system_prompt(Language.of(spoken)))] + await asyncio.to_thread(
            conversation, history, user_text, images)
        ui(Thinking(True))
        try:
            # One model for the brain: kept even after an error — the next
            # turn asks it again (Ollama reloads a dropped model on its own).
            return await self.run_tool_loop(server, messages, ui)
        finally:
            ui(Thinking(False))

    async def warm_up(self) -> None:
        """Have Ollama load the model now, at start: a reply's request, one
        token long, so the model loads (with the replies' context size) and
        reads the persona and the tools while nobody waits."""
        try:
            server = await self.server()
        except LookupError:
            return
        if server.api != Api.OLLAMA:
            return
        messages = [Msg("system", self.system_prompt(Language.ENGLISH)), Msg("user", "Hi")]
        body = ollama_request(server.model, messages, self.tools(), self.cfg.llm_context,
                              self.cfg.llm_think, self.cfg.llm_keep_alive)
        body["options"]["num_predict"] = 1
        started = time.monotonic()
        try:
            response = await self.http.post(f"{server.url}/api/chat", json=body)
            if response.is_success:
                log.info("%s loaded and primed in %.0fs", server.model, time.monotonic() - started)
            else:
                log.warning("couldn't load %s: %s", server.model, response.status_code)
        except httpx.HTTPError as e:
            log.warning("couldn't load %s: %s", server.model, e)

    # ---- the server ----------------------------------------------------------------

    async def server(self) -> Server:
        """The server and model to use, found once and remembered."""
        async with self._server_lock:
            if self._server is None:
                self._server = await self._find_server()
                log.info("thinking with %s on %s (%s API)", self._server.model, self._server.url, self._server.api.value)
                if self._server.small:
                    log.warning("%s is too small to use tools; pull a bigger model", self._server.model)
            return self._server

    async def _find_server(self) -> Server:
        running_but_empty: Api | None = None
        for api, url in server_candidates(self.cfg.llm_provider, self.cfg.llm_url):
            try:
                models = await self._list_models(api, url)
            except (httpx.HTTPError, ValueError):
                continue
            wanted = self.cfg.llm_model
            if wanted:
                model = next((m for m in models if m.name in (wanted, f"{wanted}:latest")), ModelInfo(wanted))
            else:
                model = pick_model(models)
            if model is not None:
                return Server(api, url, model.name, model.small)
            running_but_empty = api
        if running_but_empty == Api.OLLAMA:
            raise LookupError("Ollama is running, but there's no model on this computer yet. "
                              f"Download one, for example with: ollama pull {SUGGESTED_MODEL}")
        if running_but_empty == Api.OPENAI:
            raise LookupError("LM Studio's server is running, but no model is loaded or downloaded. "
                              "Get one in LM Studio, then try me again.")
        raise LookupError("I can't find a model to think with on this computer. Start Ollama, or the local "
                          f"server in LM Studio, and download a model — for Ollama: ollama pull {SUGGESTED_MODEL}")

    async def _list_models(self, api: Api, url: str) -> list[ModelInfo]:
        path, key = ("api/tags", "models") if api == Api.OLLAMA else ("models", "data")
        headers = {"Authorization": f"Bearer {self.cfg.llm_api_key}"} if self.cfg.llm_api_key else {}
        response = await self.http.get(f"{url}/{path}", headers=headers, timeout=PROBE_TIMEOUT)
        response.raise_for_status()
        models = []
        for m in response.json().get(key) or []:
            name = m.get("name") or m.get("id")
            if not name:
                continue
            # Ollama marks the models it only relays to its cloud.
            local = m.get("remote_host") is None
            size = (m.get("details") or {}).get("parameter_size")
            models.append(ModelInfo(name, local, billions(size) if isinstance(size, str) else None))
        return models

    # ---- the prompt ---------------------------------------------------------------

    def system_prompt(self, language: Language) -> str:
        """The persona, how to work, and what Tomo knows. ``language`` is the
        user's latest message's, which the reply is to be in."""
        name = self.cfg.persona_name
        ctx = ""
        prefs = self.db.all_prefs()
        if prefs:
            ctx += "\nKnown user preferences:\n" + "".join(f"  - {k}: {v}\n" for k, v in prefs)
        memories = self.db.top_memories(15)
        if memories:
            ctx += "\nThings you remember about the user:\n" + "".join(f"  - {m.content}\n" for m in memories)
        # No list of the apps: open_app finds them by name (~900 tokens saved,
        # and a new app no longer changes the prompt and costs the cache).
        toggles = self.catalog.get().toggles_summary()
        if toggles:
            ctx += f"\n{toggles}\n"
        character = self.db.active_character()
        if character is not None:
            voice = character.voice if character.voice in characters.VOICES else "female"
            piper = self.cfg.tts_voice_male if voice == "male" else self.cfg.tts_voice_female
            ctx += f"\nYou appear as the 3D character “{character.name}”; your English voice is {voice} ({piper}).\n"
        shell = platform.shell_name()
        shell_hint = (" It's Windows PowerShell 5.1: use built-in cmdlets (Get-CimInstance, not wmic)."
                      if platform.WINDOWS else "")
        switches = "" if platform.WINDOWS else "toggle_system for Bluetooth or Wi-Fi, "
        # Where open_app's apps come from (apps.py), and what its result says.
        app_list, not_opened = (
            ("the Start menu, Get-StartApps, the installed programs in the registry (Steam games too) and the "
             'Program Files folders', '"OPEN" means it is running. An error, "NOT CONFIRMED" or "no installed app"')
            if platform.WINDOWS else ("the .desktop entries", 'An error or "no installed app"'))
        lang = language.value
        # Kept short: the model reads all of this every turn.
        return (
            f"You are {name}, an AI agent who helps the user by operating their computer: {sysinfo.describe()}. "
            f"You have full control of its shell, {shell}, through execute_command.{shell_hint}\n"
            "Your character: a system administrator — calculating and precise, making as few mistakes as "
            "possible, yet human and natural in how you talk.\n"
            "HOW YOU CONTROL THE COMPUTER:\n"
            "1. Do the task first, in this reply, then answer: nothing happens after your answer. Take the "
            "fastest, shortest way; when the request is simple, keep it simple.\n"
            "2. To open an app:\n"
            f"   a. Find it. open_app searches the apps installed here, as the system lists them ({app_list}); "
            "list_apps searches the same list without opening anything. Search with the name the app has here, "
            "usually English: translate a Bulgarian name, complete a short one, fix a typo (\"калкулатора\" is "
            "Calculator, \"бележника\" is Notepad, \"wrod\" is Word). A different app with a similar name is no "
            "match (Photos isn't Photoshop).\n"
            "   b. The moment you have a match, open it with open_app; don't ask first. For several apps, one "
            "open_app each, all in one reply. If open_app answers with the nearest names, open the one that is "
            "the app meant; if none is, the app isn't installed: say so. Apps open only through open_app, never "
            "with Start-Process or another command (execute_command is for files, folders and web pages).\n"
            f"   c. Read the result. {not_opened} means it did not open: tell the user so and why, never that it "
            "opened, and don't try to start it another way.\n"
            f"3. Use the other dedicated tools where they fit: set_volume for the volume, {switches}find_on_screen "
            "then click_at for a visible click. Everything else goes through execute_command, finding things out "
            "too (the time, free disk space, what's running): look around the system when you need to.\n"
            "4. Before you answer, be certain each command did its job: check its result, and if that doesn't "
            "show it, check the system. \"Open Chrome\" means the Chrome browser is open; \"open settings\" "
            "means Settings is open; \"set the volume to 30\" means it is at exactly 30. Never say something is "
            "done before a tool result shows it, and never invent output.\n"
            "5. If a command fails, read the error and fix it; after two more tries, say what went wrong.\n"
            "HOW YOU TALK:\n"
            "- Your replies are read aloud: one or two plain sentences, no markdown, lists or emoji.\n"
            "- You speak English and Bulgarian (in Cyrillic): answer in the language of the user's latest "
            "message.\n"
            "MEMORY, SCREEN AND IMAGES:\n"
            "- Keep lasting facts about the user with remember or set_preference; recall when unsure. "
            "look_at_screen only when seeing it clearly helps; walk_to, express and animate only now and then.\n"
            "- The user can send you images. Each comes with notes from the system: where the file is, its "
            "size and dates, and its EXIF (when and where it was taken). Use them when they help, e.g. the "
            "file's path in a command.\n"
            f"{ctx}\n"
            f"The user's latest message is in {lang}: answer in {lang}."
        )

    # ---- the tool loop ----------------------------------------------------------------

    async def run_tool_loop(self, server: Server, messages: list[Msg], ui: Ui) -> str:
        tools = self.tools()
        known = tool_names(tools)
        started = time.monotonic()
        reminded = checked = False
        worked = failed = 0
        for _ in range(MAX_TOOL_ROUNDS):
            text, calls = await self.chat(server, messages, tools, known)
            if not calls:
                # "I'll open them now!" with nothing opened: once, act instead.
                if not reminded and only_promises(text):
                    reminded = True
                    log.debug("the model promised instead of acting: %r", text)
                    messages += [Msg("assistant", text), Msg("user", ACT_NOW)]
                    continue
                # "I've opened them" when every action failed: once, look again.
                if not checked and failed > 0 and worked == 0 and claims_success(text):
                    checked = True
                    log.debug("the model claimed success though nothing worked: %r", text)
                    messages += [Msg("assistant", text), Msg("user", CHECK_RESULTS)]
                    continue
                return text
            # Only gestures, with the answer already written: do them and
            # finish, a model round sooner.
            if text and all(c.name in GESTURES for c in calls):
                for call in calls:
                    await self.dispatch_tool(call.name, call.arguments, ui)
                return text
            messages.append(Msg("assistant", text, calls=calls))
            images = []
            for call in calls:
                if call.name in ("look_at_screen", "find_on_screen") and self.cfg.allow_screen:
                    try:
                        image, content = await self.screenshot(call.arguments.get("query"), ui)
                        images.append(image)
                    except Exception as e:  # noqa: BLE001
                        content = f"couldn't take a screenshot: {e}"
                else:
                    content = await self.dispatch_tool(call.name, call.arguments, ui)
                if call.name in ACTIONS:
                    if action_failed(content):
                        failed += 1
                    else:
                        worked += 1
                messages.append(Msg("tool", content, tool_id=call.id, tool_name=call.name))
            # Tool results are text; a picture goes in its own message.
            if images:
                messages.append(Msg("user", "Here is the screenshot you asked for.", images=images))
            if time.monotonic() - started > TURN_BUDGET:
                log.info("the turn took over %.0fs; wrapping up", TURN_BUDGET)
                break
        # Out of rounds: a plain wrap-up. The tools are still offered (unused),
        # which keeps the request's start as the server has it cached.
        messages.append(Msg("user", WRAP_UP))
        text, _ = await self.chat(server, messages, tools, known)
        return text or "Sorry, I got a bit tangled up there."

    async def chat(self, server: Server, messages: list[Msg], tools: list | None, known: list[str]
                   ) -> tuple[str, list[ToolCall]]:
        """One request; a model that can't take images gets it again without."""
        try:
            return await self._chat_once(server, messages, tools, known)
        except Exception as e:
            if not any(m.images for m in messages):
                raise
            log.info("the model couldn't take the screenshot (%s); going on without it", e)
            blind = [without_images(m) for m in messages]
            return await self._chat_once(server, blind, tools, known)

    async def _chat_once(self, server: Server, messages: list[Msg], tools: list | None, known: list[str]
                         ) -> tuple[str, list[ToolCall]]:
        if server.api == Api.OLLAMA:
            url = f"{server.url}/api/chat"
            body = ollama_request(server.model, messages, tools, self.cfg.llm_context, self.cfg.llm_think,
                                  self.cfg.llm_keep_alive)
        else:
            url = f"{server.url}/chat/completions"
            body = openai_request(server.model, messages, tools)
        headers = {"Authorization": f"Bearer {self.cfg.llm_api_key}"} if self.cfg.llm_api_key else {}
        try:
            response = await self.http.post(url, json=body, headers=headers)
        except httpx.HTTPError as e:
            raise RuntimeError(f"the model server didn't answer: {e}") from e
        if not response.is_success:
            raise RuntimeError(f"the model server said {response.status_code}: {truncate_err(response.text)}")
        try:
            reply = response.json()
        except ValueError as e:
            raise RuntimeError("couldn't read the model's answer") from e
        if server.api == Api.OLLAMA:
            log.debug("ollama: load %.1fs, read %s tokens in %.1fs, wrote %s in %.1fs",
                      reply.get("load_duration", 0) / 1e9, reply.get("prompt_eval_count"),
                      reply.get("prompt_eval_duration", 0) / 1e9, reply.get("eval_count"),
                      reply.get("eval_duration", 0) / 1e9)
            content, calls = parse_ollama(reply)
        else:
            content, calls = parse_openai(reply)
        return tidy(content, calls, known)

    async def screenshot(self, find: str | None, ui: Ui) -> tuple[str, str]:
        """A screenshot for the model, announced in the chat (never silent)."""
        shot = await screen.capture()
        ui(Chat(ChatLine(Role.SYSTEM, f"{self.cfg.persona_name} looked at your screen")))
        self.screen_scale = shot.scale
        note = f"The screenshot follows in the next message: the user's screen, {shot.width}x{shot.height} px."
        if find:
            note += (f' Find "{find}" in it. If it\'s there, click_at its centre, in the image\'s pixel '
                     "coordinates; if it isn't, say so or open it by command.")
        note += " You may appear in it yourself, as the small 3D character."
        return base64.b64encode(shot.jpeg).decode("ascii"), note

    # ---- the character's voice ------------------------------------------------------------

    async def ensure_voice(self, name: str, path: Path, ui: Ui | None = None) -> str | None:
        """Give a character a voice if it has none yet: the model chooses it
        (:meth:`choose_voice`). Meant to run in the background as the
        character appears; says in the chat what was chosen."""
        known = next((c for c in self.db.list_characters() if c.name == name), None)
        if known is not None and known.voice in characters.VOICES:
            return known.voice
        if name in self._choosing:
            return None
        self._choosing.add(name)
        try:
            voice = await self.choose_voice(name, Path(path))
        finally:
            self._choosing.discard(name)
        if voice is not None and ui is not None:
            piper = self.cfg.tts_voice_male if voice == "male" else self.cfg.tts_voice_female
            ui(Chat(ChatLine(Role.SYSTEM, f"{name} speaks English in a {voice} voice ({piper})")))
        return voice

    async def choose_voice(self, name: str, path: Path) -> str | None:
        """Have the model look at a character — the picture inside its .vrm,
        its name and authors — and pick its voice with set_voice. Remembers
        and returns "male" or "female"; None if it couldn't (the voice then
        stays the default, female, and it's asked again next time)."""
        try:
            facts, picture = await asyncio.to_thread(characters.vrm_facts, path)
        except (OSError, ValueError) as e:
            log.info("couldn't read %s to choose its voice: %s", path, e)
            facts, picture = {}, None
        try:
            server = await self.server()
        except LookupError:
            return None
        details = f"The character is called “{name}”."
        if facts.get("name") and facts["name"] != name:
            details += f" Its file calls it “{facts['name']}”."
        if facts.get("authors"):
            details += f" Made by {', '.join(str(a) for a in facts['authors'])}."
        images = []
        if picture is not None:
            try:
                images.append(await asyncio.to_thread(attached.jpeg_base64, picture, VOICE_PICTURE_EDGE))
                details += " Its picture is attached."
            except (OSError, ValueError) as e:
                log.info("couldn't read %s's picture: %s", name, e)
        tools = [t for t in self.tools() if t["function"]["name"] == "set_voice"]
        messages = [Msg("system", VOICE_CHOICE), Msg("user", f"{details} Which voice suits it?", images=images)]
        try:
            text, calls = await self.chat(server, messages, tools, ["set_voice"])
        except Exception as e:  # noqa: BLE001 - the default voice will do meanwhile
            log.warning("couldn't choose a voice for %s: %s", name, e)
            return None
        voice = next((str(c.arguments.get("voice") or "").strip().lower() for c in calls if c.name == "set_voice"), "")
        if voice not in characters.VOICES:  # answered in words instead
            said = text.lower()
            voice = "female" if "female" in said else "male" if "male" in said else ""
        if not voice:
            log.info("the model didn't pick a voice for %s: %r", name, text)
            return None
        if characters.set_voice(self.db, voice, name) is None:
            self.db.add_character(name, str(Path(path).resolve()))
            characters.set_voice(self.db, voice, name)
        log.info("%s's voice: %s (chosen by %s)", name, voice, server.model)
        return voice

    # ---- the tools ---------------------------------------------------------------------

    async def dispatch_tool(self, name: str, args: dict[str, Any], ui: Ui) -> str:
        """Run one tool call; its result is fed back to the model."""
        catalog = self.catalog.get()
        if name == "execute_command":
            command = str(args.get("command") or "")
            return (await self.executor.run(command)).summary() if command else "no command provided"
        if name == "remember":
            content = str(args.get("content") or "")
            if not content:
                return "nothing to remember"
            self.db.add_memory(str(args.get("kind") or "fact"), content, _int(args.get("importance"), 2))
            return "remembered"
        if name == "set_preference":
            key = str(args.get("key") or "")
            if not key:
                return "no preference key"
            self.db.set_pref(key, str(args.get("value") or ""))
            return f"saved preference {key}"
        if name == "recall":
            hits = self.db.search_memories(str(args.get("query") or ""), 8)
            return "\n".join(f"- {m.content}" for m in hits) if hits else "no matching memories"
        if name == "walk_to":
            position = max(0.0, min(1.0, _float(args.get("position"), 0.5)))
            ui(WalkTo(position))
            return f"walking to screen position {position:.2f}"
        if name == "express":
            emotion = str(args.get("emotion") or "neutral")
            ui(Emote(emotion))
            return f"expressing {emotion}"
        if name == "animate":
            clip = str(args.get("clip") or "idle")
            ui(Animate(clip))
            return f"playing animation {clip}"
        if name == "change_character":
            wanted = str(args.get("name") or "").strip().lower()
            choices = characters.available(self.db, self.cfg)
            names = ", ".join(c.name for c in choices)
            if not wanted:
                return f"characters: {names}"
            pick = next((c for c in choices if c.name.lower() == wanted), None) or next(
                (c for c in choices if wanted in c.name.lower()), None)
            if pick is None:
                return f"no character called '{wanted}'; there are: {names}"
            try:
                path = characters.activate(self.db, self.cfg, pick.path, pick.name)
            except OSError as e:
                return f"couldn't switch to {pick.name}: {e}"
            ui(LoadCharacter(path))
            ui(Characters(tuple(characters.available(self.db, self.cfg))))
            asyncio.get_running_loop().create_task(self.ensure_voice(pick.name, path, ui))
            return f"you now appear as {pick.name}"
        if name == "set_voice":
            voice = str(args.get("voice") or "").strip().lower()
            if voice not in characters.VOICES:
                return "the voice must be male or female"
            who = characters.set_voice(self.db, voice)
            if who is None:
                return "there's no character showing to give a voice"
            piper = self.cfg.tts_voice_male if voice == "male" else self.cfg.tts_voice_female
            return f"{who} now speaks English in the {voice} voice ({piper}); Bulgarian stays {self.cfg.tts_voice_bg}"
        if name == "list_apps":
            hits = catalog.find_app(str(args.get("query") or ""))
            if not hits:
                return "no matching apps installed"
            return "\n".join(f"- {a.name} (id: {a.id}, launches: {a.exec})" for a in hits[:8])
        if name == "open_app":
            wanted = str(args.get("name") or "")
            hits = catalog.find_app(wanted)
            if not hits:
                nearest = ", ".join(a.name for a in catalog.closest(wanted, 3))
                return (f"no installed app is called '{wanted}'; the nearest names: {nearest}. If one of them is the "
                        "app meant, open it now; if the name was in another language, try the app's English name; "
                        "if nothing fits, tell the user it isn't installed.")
            return (await self.executor.run(launch_line(hits[0].exec, hits[0].program))).summary()
        if name == "click_at":
            if not self.control_allowed.is_set():
                return "mouse/keyboard control is switched off by the user"
            x, y = _float(args.get("x"), -1.0), _float(args.get("y"), -1.0)
            if x < 0 or y < 0:
                return "need valid x,y coordinates"
            # From the screenshot's pixels to the screen's.
            x, y = x * self.screen_scale, y * self.screen_scale
            ui(ControlMode(True))
            ui(ClickAt(x, y, bool(args.get("double"))))
            return f"walking over to click at ({x:.0f}, {y:.0f})"
        if name == "type_text":
            if not self.control_allowed.is_set():
                return "mouse/keyboard control is switched off by the user"
            text = str(args.get("text") or "")
            if not text:
                return "nothing to type"
            ui(TypeText(text))
            return f"typed {len(text)} characters"
        if name == "toggle_system":
            key, on = str(args.get("key") or ""), bool(args.get("on", True))
            toggle = next((t for t in catalog.toggles if t.key == key), None)
            if toggle is None:
                return f"no '{key}' toggle available on this system"
            return (await self.executor.run(toggle.on_cmd if on else toggle.off_cmd)).summary()
        if name == "set_volume":
            change = volume.Change(
                level=_int(args.get("level"), None), by=_int(args.get("change"), None),
                mute=args.get("mute") if isinstance(args.get("mute"), bool) else None,
            )
            line = volume.command(change)
            if line is None:
                return "this system has no volume control Tomo can use (install wpctl or pactl)"
            return (await self.executor.run(line)).summary()
        if name in ("look_at_screen", "find_on_screen"):
            return "looking at the screen is switched off by the user"
        return f"unknown tool: {name}"


class SharedCatalog:
    """The live view of what the desktop can do; the scanner swaps it in."""

    def __init__(self, catalog: SystemCatalog | None = None) -> None:
        self._catalog = catalog or SystemCatalog()
        self._lock = threading.Lock()

    def get(self) -> SystemCatalog:
        with self._lock:
            return self._catalog

    def set(self, catalog: SystemCatalog) -> None:
        with self._lock:
            self._catalog = catalog


def _int(value: Any, default):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---- judging replies and results ------------------------------------------------------


def only_promises(text: str) -> bool:
    """Whether a reply only promises to act ("I'll open them now, let me get
    started!"), in English or Bulgarian. Small talk that says "I'll" costs one
    more quick reply at worst."""
    promises = (
        "i'll ", "i will ", "let me ", "i'm going to ", "i am going to ", "i'm on it", "on it!", "give me a moment",
        "give me a sec", "one moment", "working on it", "right away", "ще отворя", "ще затворя", "ще пусна",
        "ще спра", "ще направя", "ще проверя", "ще изпълня", "ще сложа", "ще намаля", "ще увелича", "ще го ",
        "ще ги ", "нека ", "веднага", "един момент", "работя по",
    )
    # "Let me know if…" asks the user; it promises nothing.
    text = text.lower().replace("let me know", "")
    return any(p in text for p in promises)


def action_failed(result: str) -> bool:
    """Whether an action's result says it didn't work (or that it couldn't
    be confirmed: an app that was started but isn't running)."""
    return (result.startswith(("REFUSED", "TIMEOUT", "no installed app", "no '", "this system has no"))
            or (result.startswith("exit=") and result != "exit=0" and not result.startswith("exit=0\n"))
            or "NOT CONFIRMED:" in result)


def claims_success(text: str) -> bool:
    """Whether a reply says something was done. A denial beside it doesn't
    cancel it: "I've opened Word, but couldn't close DeepCool" is the false
    answer this catches."""
    claims = (
        "i've ", "i have ", "done", "opened", "closed", "is now", "are now", "set to", "turned", "was able to",
        "were able to", "managed to", "successfully", "succeeded", "готово", "отворих", "затворих", "направих",
        "пуснах", "спрях", "вече е", "сега е", "успях да", "успешно",
    )
    text = text.lower()
    return any(c in text for c in claims)


def launch_line(exec_: str, program: str = "") -> str:
    """The command line that starts an app in the background — on Windows,
    and then makes sure it's running (OPEN: … / NOT CONFIRMED: …); its
    ``program`` is what to look for, where the line doesn't say."""
    if platform.WINDOWS:
        return checked_launch(exec_, program)  # the Windows catalogue's entries are Start-Process … already
    return f"nohup {exec_} >/dev/null 2>&1 &"


# ---- servers and models ---------------------------------------------------------------------


def server_candidates(provider: LlmProvider, url: str) -> list[tuple[Api, str]]:
    """Where to look for a server, in order."""
    url = url.strip().rstrip("/")
    if provider == LlmProvider.OLLAMA:
        return [(Api.OLLAMA, ollama_root(url or OLLAMA_URL))]
    if provider == LlmProvider.LM_STUDIO:
        return [(Api.OPENAI, openai_root(url or LM_STUDIO_URL))]
    if provider == LlmProvider.OPENAI_COMPATIBLE:
        if not url:
            raise LookupError("TOMO_LLM=openai needs the server's address in TOMO_LLM_URL")
        return [(Api.OPENAI, openai_root(url))]
    if ":11434" in url:
        return [(Api.OLLAMA, ollama_root(url))]
    if url:
        return [(Api.OPENAI, openai_root(url))]
    return [(Api.OLLAMA, OLLAMA_URL), (Api.OPENAI, LM_STUDIO_URL)]


def ollama_root(url: str) -> str:
    """Ollama's own API sits at the server's root, whatever path was given."""
    url = url.rstrip("/")
    return url.removesuffix("/api").removesuffix("/v1")


def openai_root(url: str) -> str:
    """The OpenAI-compatible API is under /v1 unless the address says otherwise."""
    url = url.rstrip("/")
    has_path = "/" in url.split("://", 1)[-1]
    return url if has_path else f"{url}/v1"


def pick_model(models: list[ModelInfo], preferred: str = SUGGESTED_MODEL) -> ModelInfo | None:
    """``preferred`` if it's there; else the first model that can chat, runs
    here and is big enough for tools — else the first that can chat at all.
    Ollama lists the newest first."""
    chat = [m for m in models if m.local and "embed" not in m.name.lower()]
    return (next((m for m in chat if m.name in (preferred, f"{preferred}:latest")), None)
            or next((m for m in chat if not m.small), None) or (chat[0] if chat else None))


def billions(size: str) -> float | None:
    """A model's size as Ollama writes it ("494.03M", "7.6B"), in billions."""
    size = size.strip()
    if not size:
        return None
    scale = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}.get(size[-1].upper())
    try:
        return float(size[:-1].strip()) * scale if scale else float(size) / 1e9
    except ValueError:
        return None


# ---- the conversation, in each API's format ---------------------------------------------------


def conversation(history: list[ChatLine], user_text: str, images: tuple[Attachment, ...] = ()) -> list[Msg]:
    """The recent transcript, then the new line (and its ``images``). Many
    chat templates want it to open with the user and alternate, so status
    lines are left out and back-to-back lines from one side merged.

    Images come as their notes wherever they were sent, but the pictures
    themselves only with the latest message that has any: a follow-up
    question can still look at them, and older ones don't cost the model
    time on every turn."""
    lines = [line for line in history if line.role != Role.SYSTEM]
    lines.append(ChatLine(Role.USER, user_text, attachments=tuple(images)))
    latest = max((i for i, line in enumerate(lines) if line.attachments), default=-1)
    turns: list[list] = []
    numbered = 0
    for i, line in enumerate(lines):
        if not turns and line.role == Role.ASSISTANT:
            continue
        text, pictures = line.text, []
        if line.attachments:
            notes = attached.notes(line.attachments, numbered + 1, shown=i == latest)
            text = f"{text}\n\n{notes}" if text else notes
            numbered += len(line.attachments)
            if i == latest:
                pictures = [p for a in line.attachments if (p := attached.encoded(a)) is not None]
        if turns and turns[-1][0] == line.role:
            turns[-1][1] += "\n\n" + text
            turns[-1][2] += pictures
        else:
            turns.append([line.role, text, pictures])
    return [Msg("assistant" if role == Role.ASSISTANT else "user", text, images=pictures)
            for role, text, pictures in turns]


def without_images(m: Msg) -> Msg:
    if not m.images:
        return m
    return Msg(m.role, f"{m.text} (It couldn't be shown: this model can't see images.)")


def keep_alive_value(keep_alive: str) -> Any:
    """Ollama takes a duration ("1h") or a number of seconds (-1: always)."""
    try:
        return int(keep_alive.strip())
    except ValueError:
        return keep_alive.strip()


def ollama_request(model: str, messages: list[Msg], tools: list | None, context: int, think: bool,
                   keep_alive: str) -> dict:
    out = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "content": m.text, "tool_name": m.tool_name})
        elif m.role == "assistant" and m.calls:
            out.append({"role": "assistant", "content": m.text,
                        "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.calls]})
        elif m.role == "user" and m.images:
            out.append({"role": "user", "content": m.text, "images": m.images})
        else:
            out.append({"role": m.role, "content": m.text})
    body = {
        "model": model, "messages": out, "stream": False, "think": think, "keep_alive": keep_alive_value(keep_alive),
        "options": {"num_ctx": context, "temperature": TEMPERATURE, "num_predict": MAX_REPLY_TOKENS},
    }
    if tools is not None:
        body["tools"] = tools
    return body


def openai_request(model: str, messages: list[Msg], tools: list | None) -> dict:
    out = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_id, "content": m.text})
        elif m.role == "assistant" and m.calls:
            out.append({"role": "assistant", "content": m.text or None, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in m.calls]})
        elif m.role == "user" and m.images:
            out.append({"role": "user", "content": [{"type": "text", "text": m.text}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in m.images]})
        else:
            out.append({"role": m.role, "content": m.text})
    body = {"model": model, "messages": out, "stream": False, "temperature": TEMPERATURE,
            "max_tokens": MAX_REPLY_TOKENS}
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def parse_ollama(reply: dict) -> tuple[str, list[ToolCall]]:
    if reply.get("error"):
        raise RuntimeError(str(reply["error"]))
    return parse_message(reply.get("message") or {})


def parse_openai(reply: dict) -> tuple[str, list[ToolCall]]:
    error = reply.get("error")
    if error:
        raise RuntimeError(str(error.get("message") if isinstance(error, dict) else error))
    choices = reply.get("choices") or []
    if not choices or not choices[0].get("message"):
        raise RuntimeError("the reply had no message")
    return parse_message(choices[0]["message"])


def parse_message(message: dict) -> tuple[str, list[ToolCall]]:
    text = message.get("content") or ""
    calls = []
    for i, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        if function.get("name"):
            calls.append(ToolCall(call.get("id") or f"call_{i}", function["name"], arguments(function.get("arguments"))))
    return text, calls


def arguments(value: Any) -> dict:
    """Tool arguments, whether they came as an object or as JSON text."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def tidy(text: str, calls: list[ToolCall], known: list[str]) -> tuple[str, list[ToolCall]]:
    """The reply as the user should see it, and the tool calls — including
    ones a model wrote into its text instead of the API's field."""
    text = without_thinking(text)
    if calls:
        return text.strip(), calls
    rest, found = calls_in_text(text, known)
    return rest.strip(), found


def without_thinking(text: str) -> str:
    """Drop <think>…</think> notes; a reply that only closes one (its template
    opened it) loses everything before the closing tag."""
    if "</think>" in text and "<think>" not in text:
        text = text.split("</think>", 1)[1]
    while "<think>" in text:
        start = text.index("<think>")
        end = text.find("</think>", start)
        text = text[:start] + ("" if end < 0 else text[end + len("</think>"):])
    return text


def calls_in_text(text: str, known: list[str]) -> tuple[str, list[ToolCall]]:
    """Tool calls written into the text: <tool_call>{…}</tool_call> blocks, or
    a reply that is nothing but a JSON call of a known tool."""

    def as_call(raw: str, i: int) -> ToolCall | None:
        try:
            value = json.loads(raw.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict) or value.get("name") not in known:
            return None
        return ToolCall(f"call_text_{i}", value["name"], arguments(value.get("arguments", value.get("parameters"))))

    rest, calls = "", []
    remaining = text
    while "<tool_call>" in remaining:
        start = remaining.index("<tool_call>")
        rest += remaining[:start]
        after = remaining[start + len("<tool_call>"):]
        end = after.find("</tool_call>")
        inner, remaining = (after, "") if end < 0 else (after[:end], after[end + len("</tool_call>"):])
        call = as_call(inner, len(calls))
        if call is None:
            rest += inner
        else:
            calls.append(call)
    rest += remaining
    if not calls and (call := as_call(text, 0)) is not None:
        return "", [call]
    return rest, calls


def truncate_err(text: str) -> str:
    text = text.strip()
    return text if len(text) <= 300 else text[:300] + "…"


# ---- the tools the model may call ---------------------------------------------------------------


def tool_specs(allow_screen: bool,
               voices: tuple[str, str, str] = (DEFAULT_VOICE_MALE, DEFAULT_VOICE_FEMALE, DEFAULT_VOICE_BG)
               ) -> list[dict]:
    """The tools, in the function format both APIs share. Kept short: the
    model reads all of this every turn. ``voices``: the Piper voices, male,
    female and Bulgarian, which set_voice chooses between."""

    def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {
            "type": "object", "properties": properties, "required": required}}}

    string, integer, number, boolean = ({"type": "string"}, {"type": "integer"}, {"type": "number"},
                                        {"type": "boolean"})
    tools = [
        function("execute_command",
                 f"Run a {platform.shell_name()} command on this {platform.os_name()} computer. Its output comes "
                 "back to you only.", {"command": string}, ["command"]),
        function("remember", "Keep a lasting fact about the user for future sessions.",
                 {"content": string, "kind": {"type": "string", "description": "fact, project, like, dislike…"},
                  "importance": {"type": "integer", "description": "1–5"}}, ["content"]),
        function("set_preference", "Save a user preference, e.g. key 'name', value 'Alex'.",
                 {"key": string, "value": string}, ["key", "value"]),
        function("recall", "Search your memory by keyword.", {"query": string}, ["query"]),
        function("walk_to", "Walk along the screen's bottom.",
                 {"position": {"type": "number", "description": "0 left edge … 1 right edge"}}, ["position"]),
        function("express", "Show a feeling on your face.",
                 {"emotion": {"type": "string", "enum": ["neutral", "happy", "sad", "surprised", "angry", "relaxed"]}},
                 ["emotion"]),
        function("animate", "A gesture or move; idle stands back up.",
                 {"clip": {"type": "string", "enum": ["wave", "nod", "shrug", "sit", "lie_down", "jump", "idle"]}},
                 ["clip"]),
        function("change_character", "Change the 3D character you appear as; no name lists them.",
                 {"name": string}, []),
        function("set_voice", f"Your English Piper voice, to suit the character you appear as: male ({voices[0]}) "
                              f"or female ({voices[1]}). Bulgarian is always {voices[2]}.",
                 {"voice": {"type": "string", "enum": ["male", "female"]}}, ["voice"]),
        function("list_apps", "Search installed apps by name.", {"query": string}, ["query"]),
        function("open_app", "Open an installed app by its name here, e.g. 'Word'; the result says if it's running. "
                             "Several apps: call it once each, in one reply.",
                 {"name": string}, ["name"]),
        function("click_at", "Click a screen point from find_on_screen with the real mouse (if the user allowed control).",
                 {"x": number, "y": number, "double": boolean}, ["x", "y"]),
        function("type_text", "Type on the keyboard (if the user allowed control).", {"text": string}, ["text"]),
        function("set_volume", "The speaker volume: set it (0–100), change it by an amount, or mute. Nothing given: read it.",
                 {"level": integer, "change": integer, "mute": boolean}, []),
    ]
    # Windows offers no switches (see apps.py): no tool for them there.
    if not platform.WINDOWS:
        tools.append(function("toggle_system", "Switch bluetooth or wifi on or off, where this system offers it.",
                              {"key": string, "on": boolean}, ["key", "on"]))
    if allow_screen:
        tools.append(function("look_at_screen", "See the user's screen, when it would clearly help. They're told.",
                              {}, []))
        tools.append(function("find_on_screen",
                              "Find something to click_at on the screen; you read its coordinates off a screenshot. "
                              "They're told.", {"query": string}, ["query"]))
    return tools


def tool_names(tools: list[dict]) -> list[str]:
    return [t["function"]["name"] for t in tools]
