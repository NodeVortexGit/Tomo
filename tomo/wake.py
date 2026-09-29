"""Voice input, all on this computer: "Hey Tomo", and the Talk button.

Runs ``scripts/wake_word.py`` as a child process — Vosk spots the phrase,
Parakeet transcribes the request, in English or Bulgarian — and turns its
JSON-lines output into brain traffic: a spoken request becomes a
:class:`~tomo.events.UserMessage`, exactly as if typed, and the body is told
while Tomo is listening. With the wake word switched off it still runs, for
the Talk button, but only opens the microphone when that's pressed. The
models live in ``<data dir>/models``; without them there's no voice input.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Callable

from . import platform
from .config import Config
from .events import BrainToUi, Listening, UiToBrain, UserMessage

log = logging.getLogger(__name__)

VOSK_MODEL = "vosk-model-small-en-us-0.15"
# Parakeet TDT 0.6B v3 (int8 ONNX): speech to text in 25 European languages,
# English and Bulgarian among them.
STT_MODEL = "parakeet-tdt-0.6b-v3"


class Wake:
    """Steers the running listener."""

    def __init__(self) -> None:
        self._commands: asyncio.Queue[str] = asyncio.Queue()

    def listen(self) -> None:
        """Skip the wake phrase and take a request now (push-to-talk)."""
        self._commands.put_nowait("listen")

    def mute(self, muted: bool) -> None:
        """Ignore the microphone while Tomo speaks, so she can't wake herself."""
        self._commands.put_nowait("mute" if muted else "unmute")


def spawn(cfg: Config, to_brain: Callable[[UiToBrain], None], ui: Callable[[BrainToUi], None],
          python: str | None = None) -> Wake | None:
    """Start the listener, if the speech models are installed."""
    models = cfg.data_dir / "models"
    vosk, stt = models / VOSK_MODEL, models / STT_MODEL
    if not vosk.is_dir() or not (stt / "encoder-model.int8.onnx").is_file():
        log.warning("voice input is off: speech models missing in %s (setup_models.py fetches them)", models)
        return None
    wake = Wake()
    script = cfg.scripts_dir / "wake_word.py"
    python = python or platform.helper_python()

    async def keep_listening() -> None:
        # Restart after a crash, but give up on one that keeps failing fast
        # (no microphone, a broken install).
        quick_failures = 0
        while True:
            started = time.monotonic()
            try:
                await _listen(python, script, vosk, stt, cfg.wake_word, wake, to_brain, ui)
                return
            except Exception as e:  # noqa: BLE001 - any failure restarts it
                log.warning("wake-word listener stopped: %s", e)
            ui(Listening(False))
            quick_failures = 1 if time.monotonic() - started > 60 else quick_failures + 1
            if quick_failures == 3:
                log.warning('"Hey Tomo" is off after repeated failures')
                return
            await asyncio.sleep(10)

    asyncio.get_running_loop().create_task(keep_listening())
    return wake


async def _listen(python: str, script: Path, vosk: Path, stt: Path, wake_word: bool, wake: Wake,
                  to_brain: Callable[[UiToBrain], None], ui: Callable[[BrainToUi], None]) -> None:
    args = [python, str(script), "--vosk", str(vosk), "--stt", str(stt)]
    if not wake_word:
        args.append("--push-to-talk")
    process = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        creationflags=platform.quiet_flags(),
    )

    async def forward_commands() -> None:
        while True:
            command = await wake._commands.get()
            process.stdin.write(f"{command}\n".encode())
            await process.stdin.drain()

    forwarder = asyncio.get_running_loop().create_task(forward_commands())
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                raise RuntimeError("it exited")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("event")
            if kind == "ready":
                log.info('listening for "Hey Tomo"' if wake_word else "voice input ready (Talk button)")
            elif kind == "wake":
                ui(Listening(True))
            elif kind == "heard":
                ui(Listening(False))
                to_brain(UserMessage(event.get("text", "")))
            elif kind == "idle":
                ui(Listening(False))
            elif kind == "error":
                raise RuntimeError(event.get("message", "error"))
    finally:
        forwarder.cancel()
        if process.returncode is None:
            process.kill()
