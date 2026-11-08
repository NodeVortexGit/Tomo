"""Speech out, on this computer: Tomo's replies become a voice.

Two helpers, started once and kept running, each load their voices once and
then turn each line they're sent into a WAV file: ``scripts/tts_piper.py``
(Piper) for English and ``scripts/tts_bgtts.py`` (BgTTS-38M-V2) for Bulgarian.
:meth:`Speech.voice_for` picks the voice by the reply's language and by the
showing character: Alan or the BgTTS male voice for a male character, Cori or
the BgTTS female voice for a female one (the model chooses which a character
is: see characters.py). Playback uses what the system has: Windows' own sound player, PipeWire /
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
BG_SPEAK_TIMEOUT = 180.0  # BgTTS takes about a third of the time the speech lasts, more while the model thinks


class Helper:
    """A voice helper: started when first needed, one request at a time."""

    def __init__(self, python: str, script: Path, args: list[str], timeout: float) -> None:
        self.python = python
        self.script = script
        self.args = args
        self.timeout = timeout
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def warm_up(self) -> None:
        async with self._lock:
            if self._process is None or self._process.returncode is not None:
                try:
                    self._process = await self._start()
                except Exception as e:  # noqa: BLE001 - the first reply tries again, and says what's wrong
                    log.info("the voice isn't ready yet (%s): %s", self.script.name, e)

    async def ask(self, request: dict) -> None:
        async with self._lock:
            if self._process is None or self._process.returncode is not None:
                self._process = await self._start()
            try:
                self._process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
                await self._process.stdin.drain()
                line = await asyncio.wait_for(self._process.stdout.readline(), self.timeout)
            except Exception:
                self._stop()
                raise
            if not line:
                self._stop()
                raise RuntimeError(f"the voice helper stopped ({self.script.name})")
            _helper_reply(line)

    def _stop(self) -> None:
        process, self._process = self._process, None  # start afresh next time
        try:
            process.kill()
        except (AttributeError, OSError):
            pass

    async def _start(self) -> asyncio.subprocess.Process:
        try:
            process = await asyncio.create_subprocess_exec(
                self.python, str(self.script), *self.args,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                creationflags=platform.quiet_flags(),
            )
        except OSError as e:
            raise RuntimeError(f"couldn't start the voice helper ({self.script.name}): {e}") from e
        line = await asyncio.wait_for(process.stdout.readline(), START_TIMEOUT)
        if not line:
            raise RuntimeError(f"the voice helper exited ({self.script.name})")
        _helper_reply(line)
        log.info("voices ready: %s", self.script.name)
        return process


class Speech:
    def __init__(self, cfg: Config, python: str | None = None, gender: Callable[[], str] | None = None) -> None:
        self.python = python or platform.helper_python()
        self.cache_dir = cfg.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice_male = cfg.tts_voice_male
        self.voice_female = cfg.tts_voice_female
        self.voice_bg_male = cfg.tts_voice_bg_male
        self.voice_bg_female = cfg.tts_voice_bg_female
        # The showing character's voice, "male" or "female" (characters.voice_of).
        self.gender = gender or (lambda: "")
        self.speed = cfg.tts_speed
        # All the voices at once: switching characters, or languages, never
        # waits for a voice to load.
        self.piper = Helper(self.python, cfg.scripts_dir / "tts_piper.py",
                            ["--voice", self.voice_female, "--voice", self.voice_male,
                             "--voices-dir", str(cfg.data_dir / "voices")], SPEAK_TIMEOUT)
        self.bgtts = Helper(self.python, cfg.scripts_dir / "tts_bgtts.py",
                            ["--voice", self.voice_bg_female, "--voice", self.voice_bg_male,
                             "--models-dir", str(cfg.data_dir / "models")], BG_SPEAK_TIMEOUT)
        self._count = itertools.count()
        self._told_broken = False

    def voice_for(self, text: str) -> str:
        """The voice for a line, by its language and the character: male
        (BgTTS or Alan) if the character's male, otherwise female (BgTTS or
        Cori), also while its voice isn't chosen yet."""
        if Language.of(text) == Language.BULGARIAN:
            if self.gender() == "male":
                return self.voice_bg_male
            else:
                return self.voice_bg_female
        elif self.gender() == "male":
            return self.voice_male
        else:
            return self.voice_female

    async def warm_up(self) -> None:
        """Start the helpers now, as Tomo appears: loading the voices takes
        a few seconds, which the first reply would otherwise wait for."""
        await asyncio.gather(self.piper.warm_up(), self.bgtts.warm_up())

    async def synthesize(self, text: str) -> Path:
        """Turn ``text`` into a WAV file for :meth:`play`. Markdown and emoji
        are left out of the voice (see :func:`speakable`)."""
        text = speakable(text)
        if not text:
            raise RuntimeError("nothing to say out loud")
        out = self.cache_dir / f"speech-{next(self._count)}.wav"
        voice = self.voice_for(text)
        if voice in (self.voice_bg_male, self.voice_bg_female):
            # BgTTS has no speed setting.
            await self.bgtts.ask({"text": numbers_in_bulgarian(text), "out": str(out), "voice": voice})
        else:
            await self.piper.ask({"text": text, "out": str(out), "speed": self.speed, "voice": voice})
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


# ---- numbers, which BgTTS can't read ----------------------------------------------------

UNITS = "нула едно две три четири пет шест седем осем девет".split()
TEENS = ("десет единадесет дванадесет тринадесет четиринадесет петнадесет шестнадесет седемнадесет "
         "осемнадесет деветнадесет").split()
TENS = "_ _ двадесет тридесет четиридесет петдесет шестдесет седемдесет осемдесет деветдесет".split()
HUNDREDS = "_ сто двеста триста четиристотин петстотин шестстотин седемстотин осемстотин деветстотин".split()
NUMBER = re.compile(r"(?<![\w.,:])(\d{1,2}):(\d{2})(?!\d)"
                    r"|(?<![\w.,:])(\d+)(?:[.,](\d+)(?![.,]?\d))?(?!\w)(\s*%)?(\s+[а-яА-Я]+)?")


def number_words(n: int, noun: str = "") -> str:
    """A whole number below a billion, in Bulgarian. ``noun`` is the word it
    counts, for the forms of 1 and 2 that go with it: един час, два часа, една
    минута, две минути (едно, две without one)."""
    parts = []  # with "и" before the last of them, as Bulgarian has it
    if n >= 1_000_000:
        millions = n // 1_000_000
        parts.append("един милион" if millions == 1 else f"{number_words(millions, 'милиона')} милиона")
        n %= 1_000_000
    if n >= 1000:
        thousands = n // 1000
        parts.append("хиляда" if thousands == 1 else f"{number_words(thousands, 'минути')} хиляди")
        n %= 1000
    if n >= 100:
        parts.append(HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10])
        n %= 10
    if 10 <= n < 20:
        parts.append(TEENS[n - 10])
    elif n or not parts:
        parts.append(_counted(n, noun.lower(), alone=not parts))
    if len(parts) == 1:
        return parts[0]
    return " ".join(parts[:-1]) + " и " + parts[-1]


def _counted(unit: int, noun: str, alone: bool) -> str:
    """1 or 2 agrees with the noun: a lone 1 with its singular (час, минута,
    приложение), anything else with its plural: masculine nouns count in -а/-я
    (часа, файла), feminine in -и (минути), neuter in -ия (приложения)."""
    if unit == 1 and noun and alone:
        return "една" if noun.endswith(("а", "я")) else "едно" if noun.endswith(("о", "е")) else "един"
    if unit == 1 and noun:
        return "една" if noun.endswith("и") else "един" if noun.endswith(("а", "я")) and not noun.endswith("ия") else "едно"
    if unit == 2 and noun.endswith(("а", "я")) and not noun.endswith("ия"):
        return "два"
    return UNITS[unit]


def numbers_in_bulgarian(text: str) -> str:
    """Digits as Bulgarian words: BgTTS has no sound for them. A number that
    doesn't read as one (a code, 007, ten digits and more) goes digit by digit."""

    def spoken(match: re.Match) -> str:
        hours, minutes, whole, part, percent, noun = match.groups()
        if hours is not None:
            hour = number_words(int(hours), "час" if int(hours) == 1 else "часа")
            if minutes == "00":
                return f"{hour} час" if int(hours) == 1 else f"{hour} часа"
            return f"{hour} и {_as_number(minutes)}"
        if part is not None:
            said = f"{_as_number(whole, 'цяло')} цяло и {_as_number(part)}"
        elif percent:
            said = _as_number(whole, "процент" if whole == "1" else "процента")
        else:
            said = _as_number(whole, (noun or "").strip())
        if percent:
            said += " процент" if whole == "1" and part is None else " процента"
        return said + (noun or "")

    # Whatever's left: v2, the parts of a date.
    text = re.sub(r"\d+", lambda m: f" {_as_number(m.group())} ", NUMBER.sub(spoken, text))
    return re.sub(r" +([.,!?;:)])", r"\1", re.sub(r" {2,}", " ", text)).strip()


def _as_number(digits: str, noun: str = "") -> str:
    if len(digits) > 9 or (len(digits) > 1 and digits.startswith("0")):
        return " ".join(UNITS[int(d)] for d in digits)
    return number_words(int(digits), noun)
