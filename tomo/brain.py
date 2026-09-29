"""The brain's event loop — the seam between the async world (the model,
memory, speech, commands) and the body's render loop.

:func:`Brain.start` runs :func:`run_loop` on its own thread with its own
asyncio loop and returns a :class:`BrainHandle`: ``send`` pushes a
:class:`~tomo.events.UiToBrain` in, ``poll`` drains the
:class:`~tomo.events.BrainToUi` messages out (the body calls it once a
frame). Nothing the body does ever waits on the model.

Single source of truth: the body does NOT add the user's own line to the
chat. It sends the message; the brain echoes it back, then follows with the
reply — so the chat, the memory and the model's context stay in step.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from . import attachments as attached, characters, health, speech as speech_mod, wake as wake_mod
from .ai import TRANSCRIPT_CONTEXT, AiClient, SharedCatalog
from .apps import SystemCatalog
from .commands import Executor
from .config import Config
from .db import Store
from .events import (Attachment, BrainToUi, Characters, Chat, ChatLine, ControlMode, ImportCharacter, LoadCharacter,
                     PanicStop, Role, SetControlAllowed, SetVoice, Shutdown, Speaking, StartVoiceInput, Status,
                     UiToBrain, UserMessage)
from .language import Language

log = logging.getLogger(__name__)

RESCAN_INTERVAL = 120.0  # how often to re-scan the OS for apps


@dataclass
class BrainHandle:
    _to_brain: asyncio.Queue
    _loop: asyncio.AbstractEventLoop
    _from_brain: queue.SimpleQueue
    thread: threading.Thread

    def send(self, message: UiToBrain) -> None:
        """Send a user action to the brain (from any thread)."""
        try:
            self._loop.call_soon_threadsafe(self._to_brain.put_nowait, message)
        except RuntimeError:
            log.warning("the brain has stopped")

    def poll(self) -> list[BrainToUi]:
        """Everything the brain has said since the last call."""
        out = []
        while True:
            try:
                out.append(self._from_brain.get_nowait())
            except queue.Empty:
                return out


class Brain:
    @staticmethod
    def start(cfg: Config) -> BrainHandle:
        """Start the brain on its own thread. Returns at once."""
        loop = asyncio.new_event_loop()
        to_brain: asyncio.Queue = asyncio.Queue()
        from_brain: queue.SimpleQueue = queue.SimpleQueue()

        def main() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_loop(cfg, to_brain, from_brain.put))
            except Exception:  # noqa: BLE001
                log.exception("the brain stopped with an error")

        thread = threading.Thread(target=main, name="tomo-brain", daemon=True)
        handle = BrainHandle(to_brain, loop, from_brain, thread)
        thread.start()
        return handle


async def run_loop(cfg: Config, inbox: asyncio.Queue, ui) -> None:
    log.info("brain starting: %s", cfg.redacted())
    db = await asyncio.to_thread(Store, cfg.chroma_dir, cfg.embeddings)
    await asyncio.to_thread(db.migrate_from_sqlite, cfg.legacy_sqlite)
    executor = Executor(cfg.allow_command_execution, cfg.audit_log_path, cfg.extra_allowed_commands)
    # The English voice follows the character showing (speech.voice_for).
    speech = speech_mod.Speech(cfg, gender=lambda: characters.voice_of(db))

    def to_brain(message: UiToBrain) -> None:
        inbox.put_nowait(message)

    wake = wake_mod.spawn(cfg, to_brain, ui)
    # The exercise programme: on whenever there's a camera, with no switch.
    # It counts in the language the user last spoke.
    speaks_bulgarian = [False]
    health.spawn(cfg, speech, lambda: speaks_bulgarian[0], ui)
    voice_replies = True

    catalog = SharedCatalog(load_cached_catalog(db))
    control_allowed = threading.Event()
    if cfg.allow_command_execution:
        control_allowed.set()
    ai = AiClient(cfg, db, executor, catalog, control_allowed)
    # Load the model and the voices while the character appears, not on the
    # first question.
    asyncio.get_running_loop().create_task(ai.warm_up())
    asyncio.get_running_loop().create_task(speech.warm_up())
    asyncio.get_running_loop().create_task(scan_catalog(db, catalog, ui))

    start = characters.startup(db, cfg)
    if start is not None:
        ui(LoadCharacter(start))
        # The one showing is the active one (the voice goes by it), and the
        # model gives it a voice if it has none yet.
        name = characters.name_of(db, start)
        try:
            start = characters.activate(db, cfg, start, name)
        except OSError as e:
            log.warning("couldn't remember %s as the character: %s", start, e)
        asyncio.get_running_loop().create_task(ai.ensure_voice(name, start, ui))
    else:
        log.warning("no character to show: put a .vrm at %s or set TOMO_CHARACTER in .env", cfg.character_path)
    ui(Characters(tuple(characters.available(db, cfg))))
    # Pick the conversation up where it left off.
    for line in db.recent_messages(TRANSCRIPT_CONTEXT):
        ui(Chat(line))
    ui(Status("ready"))

    while True:
        message = await inbox.get()
        if isinstance(message, UserMessage):
            if message.text.strip():
                speaks_bulgarian[0] = Language.of(message.text) == Language.BULGARIAN
            images = await attach(cfg, message.images, ui) if message.images else ()
            if message.text.strip() or images:
                await handle_user_text(db, ai, speech if voice_replies else None, wake, ui, message.text.strip(),
                                       images)
        elif isinstance(message, StartVoiceInput):
            if wake is not None:
                wake.listen()  # push-to-talk through the offline listener
            else:
                ui(Chat(ChatLine(Role.SYSTEM, "Voice input isn't set up: the speech models are missing "
                                              "(setup_models.py downloads them).")))
        elif isinstance(message, ImportCharacter):
            try:
                path = characters.activate(db, cfg, message.path, message.name)
                ui(LoadCharacter(path))
                ui(Characters(tuple(characters.available(db, cfg))))
                asyncio.get_running_loop().create_task(ai.ensure_voice(message.name, path, ui))
            except OSError as e:
                log.warning("couldn't switch to %s: %s", message.path, e)
                ui(Chat(ChatLine(Role.SYSTEM, f"Couldn't load “{message.name}”: {e}")))
        elif isinstance(message, PanicStop):
            control_allowed.clear()
            ui(ControlMode(False))
            ui(Status("control released (panic hotkey)"))
            log.warning("panic stop: mouse/keyboard control disabled")
        elif isinstance(message, SetVoice):
            voice_replies = message.on
            log.info("voice replies %s", "on" if message.on else "off")
        elif isinstance(message, SetControlAllowed):
            if message.on:
                control_allowed.set()
            else:
                control_allowed.clear()
        elif isinstance(message, Shutdown):
            log.info("brain shutting down")
            return


async def attach(cfg: Config, paths: tuple[Path, ...], ui) -> tuple[Attachment, ...]:
    """Tomo's copies of the images sent, with their notes (attachments.py).
    One that can't be read is left out, and the chat says so."""
    out = []
    for path in paths[:attached.MAX_PER_MESSAGE]:
        try:
            out.append(await asyncio.to_thread(attached.prepare, Path(path), cfg.data_dir / "attachments"))
        except (OSError, ValueError) as e:
            log.warning("couldn't attach %s: %s", path, e)
            ui(Chat(ChatLine(Role.SYSTEM, f"Couldn't attach {Path(path).name}: {e}")))
    return tuple(out)


async def handle_user_text(db: Store, ai: AiClient, speech, wake, ui, text: str,
                           images: tuple[Attachment, ...] = ()) -> None:
    """The core turn: echo the user's line (and images), get a reply, keep
    both, show and speak the reply (``speech`` None: the voice is off)."""
    # Echo now, but keep it only after the reply: the AI rebuilds its context
    # from the stored transcript and adds the current message itself.
    user_line = ChatLine(Role.USER, text, attachments=images)
    ui(Chat(user_line))
    try:
        reply = (await ai.respond(text, ui, images)).strip() or "…"
    except Exception as e:  # noqa: BLE001
        log.warning("ai error: %s", e)
        reply = f"Sorry — I hit a snag reaching my brain ({e})."
    reply_line = ChatLine(Role.ASSISTANT, reply)
    db.add_message(user_line)
    db.add_message(reply_line)
    ui(Chat(reply_line))
    # Speak in the background, with the wake word muted so Tomo's own voice
    # can't trigger it. Nothing to say (only emoji) isn't a broken voice.
    if speech is None or not speech_mod.speakable(reply):
        return

    async def say() -> None:
        try:
            audio = await speech.synthesize(reply)
        except Exception as e:  # noqa: BLE001
            log.warning("tts failed: %s", e)
            if (note := speech.first_failure_note(e)) is not None:
                ui(Chat(ChatLine(Role.SYSTEM, note)))
            return
        if wake is not None:
            wake.mute(True)
        ui(Speaking(True))
        try:
            await speech.play(audio)
        except Exception as e:  # noqa: BLE001
            log.debug("tts playback failed: %s", e)
        finally:
            ui(Speaking(False))
            if wake is not None:
                wake.mute(False)

    asyncio.get_running_loop().create_task(say())


def load_cached_catalog(db: Store) -> SystemCatalog:
    """The last scan, so the AI has something at once, before the first fresh one."""
    cached = db.get_snapshot("catalog")
    if cached is None:
        return SystemCatalog()
    try:
        return SystemCatalog.from_json(cached[1])
    except (ValueError, TypeError, KeyError):
        return SystemCatalog()


async def scan_catalog(db: Store, catalog: SharedCatalog, ui) -> None:
    """Scan the OS now, then every couple of minutes, keeping the result
    (and the memory's copy) only when something changed."""
    while True:
        try:
            fresh = await asyncio.to_thread(SystemCatalog.scan)
            fingerprint = fresh.fingerprint()
            cached = db.get_snapshot("catalog")
            if cached is None or cached[0] != fingerprint:
                db.set_snapshot("catalog", fingerprint, fresh.to_json())
                catalog.set(fresh)
                log.info("catalog refreshed: %d apps", len(fresh.apps))
                ui(Status(f"catalog refreshed: {len(fresh.apps)} apps"))
            elif not catalog.get().apps:
                catalog.set(fresh)
        except Exception as e:  # noqa: BLE001
            log.warning("couldn't scan the apps: %s", e)
        await asyncio.sleep(RESCAN_INTERVAL)
