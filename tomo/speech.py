"""Speech out, on this computer: Piper turns Tomo's replies into a voice.

Piper runs in a helper, ``scripts/tts_piper.py``, started once and kept
running: it loads the three voices once, then turns each line it's sent into
a WAV file, in the voice :meth:`Speech.voice_for` picks — Bulgarian: Dimitar;
English: the showing character's own, Alan for a male character, Cori for a
female one (the model chooses which a character is: see characters.py).
Playback uses what the system has: Windows' own sound player, PipeWire /
PulseAudio / ALSA players on Linux, ``afplay`` on macOS. (Speech *in* is the
listener in wake.py.)
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable

from . import platform
from .config import Config
from .language import Language

log = logging.getLogger(__name__)

START_TIMEOUT = 180.0  # loading a voice the first time can mean downloading it
SPEAK_TIMEOUT = 60.0


class Speech:
    def __init__(self, cfg: Config, python: str | None = None, gender: Callable[[], str] | None = None) -> None:
        self.python = python or platform.helper_python()
        self.script = cfg.scripts_dir / "tts_piper.py"
        self.voices_dir = cfg.data_dir / "voices"
        self.cache_dir = cfg.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice_male = cfg.tts_voice_male
        self.voice_female = cfg.tts_voice_female
        self.voice_bg = cfg.tts_voice_bg
        # The showing character's voice, "male" or "female" (characters.voice_of).
        self.gender = gender or (lambda: "")
        self.speed = cfg.tts_speed
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._count = itertools.count()
        self._told_broken = False

    def voice_for(self, text: str) -> str:
        """The Piper voice for a line: Dimitar for Bulgarian; for English, the
        character's — Alan if it's male, Cori if it's female (and while its
        voice isn't chosen yet)."""
        if Language.of(text) == Language.BULGARIAN:
            return self.voice_bg
        elif self.gender() == "male":
            return self.voice_male
        else:
            return self.voice_female

    async def warm_up(self) -> None:
        """Start the helper now, as Tomo appears: loading the voices takes a
        few seconds, which the first reply would otherwise wait for."""
        async with self._lock:
            if self._process is None or self._process.returncode is not None:
                try:
                    self._process = await self._start_helper()
                except Exception as e:  # noqa: BLE001 - the first reply tries again, and says what's wrong
                    log.info("the voice isn't ready yet: %s", e)

    async def synthesize(self, text: str) -> Path:
        """Turn ``text`` into a WAV file for :meth:`play`. Markdown and emoji
        are left out of the voice (see :func:`speakable`)."""
        text = speakable(text)
        if not text:
            raise RuntimeError("nothing to say out loud")
        out = self.cache_dir / f"speech-{next(self._count)}.wav"
        async with self._lock:
            if self._process is None or self._process.returncode is not None:
                self._process = await self._start_helper()
            request = json.dumps({"text": text, "out": str(out), "speed": self.speed, "voice": self.voice_for(text)})
            try:
                self._process.stdin.write((request + "\n").encode("utf-8"))
                await self._process.stdin.drain()
                line = await asyncio.wait_for(self._process.stdout.readline(), SPEAK_TIMEOUT)
                if not line:
                    raise RuntimeError("the voice helper stopped")
                _helper_reply(line)
            except Exception:
                self._process = None  # start afresh next time
                raise
        return out

    async def play(self, audio: Path) -> None:
        """Play synthesized speech, then delete it. Returns once it ends."""
        try:
            await asyncio.to_thread(play_wav, audio)
        finally:
            try:
                os.remove(audio)
            except OSError:
                pass

    def first_failure_note(self, error: Exception) -> str | None:
        """What to tell the user the first time speaking fails (None after):
        otherwise Tomo just stays silent and nobody knows why."""
        if self._told_broken:
            return None
        self._told_broken = True
        how = ("Start menu → Tomo → Set up Tomo's voice" if platform.WINDOWS
               else "run ./install.sh again")
        return f"I can't speak out loud ({error}). The voice needs setting up: {how}."

    async def _start_helper(self) -> asyncio.subprocess.Process:
        # All three at once: switching characters, or languages, never waits
        # for a voice to load.
        voices = [self.voice_female, self.voice_male, self.voice_bg]
        try:
            process = await asyncio.create_subprocess_exec(
                self.python, str(self.script), *[arg for v in voices for arg in ("--voice", v)],
                "--voices-dir", str(self.voices_dir),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                creationflags=platform.quiet_flags(),
            )
        except OSError as e:
            raise RuntimeError(f"couldn't start the voice helper (tts_piper.py): {e}") from e
        line = await asyncio.wait_for(process.stdout.readline(), START_TIMEOUT)
        if not line:
            raise RuntimeError("the voice helper exited")
        _helper_reply(line)
        log.info("voices ready: %s", ", ".join(voices))
        return process


def _helper_reply(line: bytes) -> None:
    """The helper answers each line with {"ok": true} or {"error": "…"}."""
    try:
        reply = json.loads(line)
    except json.JSONDecodeError as e:
        raise RuntimeError("the voice helper said something odd") from e
    if reply.get("error"):
        raise RuntimeError(reply["error"])


def play_wav(path: Path) -> None:
    """Play a WAV file with what the system has, and wait for it to end."""
    if platform.WINDOWS:
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    import subprocess

    players = [("afplay", []), ("pw-play", []), ("paplay", []), ("aplay", ["-q"]),
               ("mpv", ["--really-quiet", "--no-video"]), ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"])]
    for player, args in players:
        program = platform.which(player)
        if program is None:
            continue
        result = subprocess.run([program, *args, str(path)], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return
    raise RuntimeError("no audio player found (install pipewire, pulseaudio-utils or alsa-utils)")


# ---- what a voice should say --------------------------------------------------------


def speakable(text: str) -> str:
    """A reply as it should be *heard*: markdown markers and emoji dropped, a
    link's words kept but not its address, and every line ending on a pause
    so list items don't run together."""
    lines = []
    for line in text.splitlines():
        line = link_words(line).lstrip().lstrip("#>").lstrip()
        for bullet in ("- ", "* ", "+ ", "• "):
            if line.startswith(bullet):
                line = line[len(bullet):]
                break
        cleaned = "".join(" " if c == "_" else "" if c in "*`~|" or is_emoji(c) else c for c in line)
        words = ""
        for word in cleaned.split():
            if words and not word.startswith((".", ",", "!", "?", ";", ":", "…", ")")):
                words += " "
            words += word
        if not any(c.isalnum() for c in words):
            continue  # blank, a rule like "---", or nothing but emoji
        lines.append(words if words.endswith((".", "!", "?", "…", ":", ";", ",")) else words + ".")
    return "\n".join(lines)


def link_words(line: str) -> str:
    """``[words](address)`` → ``words``, anywhere in the line."""
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)


def is_emoji(c: str) -> bool:
    """Emoji and pictographic symbols (plus the joiners and variation
    selectors that glue them), none of which a voice should read out."""
    o = ord(c)
    return (0x1F000 <= o <= 0x1FAFF or 0x2190 <= o <= 0x21FF or 0x2300 <= o <= 0x23FF or 0x2600 <= o <= 0x27BF
            or 0x2B00 <= o <= 0x2BFF or 0xFE00 <= o <= 0xFE0F or o in (0x200D, 0x20E3) or 0xE0000 <= o <= 0xE007F)
