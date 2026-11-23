"""Speech out, on this computer: Tomo's replies become a voice.

Two helpers, started once and kept running, each load their voices once and
then turn each line they're sent into a WAV file: ``scripts/tts_piper.py``
(Piper) for English and ``scripts/tts_bgtts.py`` (BgTTS-38M-V2) for Bulgarian.
:meth:`Speech.voice_for` picks the voice by the reply's language and by the
showing character: Alan or the BgTTS male voice for a male character, Cori or
the BgTTS female voice for a female one (the model chooses which a character
is: see characters.py). A reply is spoken as it's written (:class:`SpeechStream`):
a sentence at a time, the next one made while one plays, and played without
a gap between them. Playback uses what the system has: Windows' own sound player, PipeWire /
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
import wave
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
        self._stream: SpeechStream | None = None

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

    async def synthesize(self, text: str, voice: str | None = None) -> Path:
        """Turn ``text`` into a WAV file for :meth:`play`, in ``voice`` (None:
        :meth:`voice_for` picks it). Markdown and emoji are left out of the
        voice (see :func:`speakable`)."""
        text = speakable(text)
        if not text:
            raise RuntimeError("nothing to say out loud")
        out = self.cache_dir / f"speech-{next(self._count)}.wav"
        voice = voice or self.voice_for(text)
        if voice in (self.voice_bg_male, self.voice_bg_female):
            # BgTTS has no speed setting.
            await self.bgtts.ask({"text": numbers_in_bulgarian(text), "out": str(out), "voice": voice})
        else:
            await self.piper.ask({"text": text, "out": str(out), "speed": self.speed, "voice": voice})
        return out

    def stream(self, speaking: Callable[[bool], None],
               heard: Callable[[object, float], None] | None = None) -> SpeechStream:
        """A reply's speech, a sentence at a time (see :class:`SpeechStream`).
        A new one stops the one before, if it's still talking."""
        if self._stream is not None:
            self._stream.cancel()
        self._stream = SpeechStream(self, speaking, heard=heard)
        return self._stream

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


class SpeechStream:
    """One reply's speech while it's being written: :meth:`add` each piece
    (see :func:`speech_ends`) as it's ready, then :meth:`close`. The voice is
    the one for the first piece, for the whole reply. Pieces are made in
    order while the one before plays, and played back to back without a gap
    (:class:`Player`). ``speaking(True)`` as the first starts, ``(False)``
    after the last — or when it's cancelled. ``heard(tag, seconds)`` as each
    piece starts to sound, with the ``tag`` it was added with and how long it
    lasts (0: a piece that couldn't be made, at the moment it would have
    played)."""

    def __init__(self, speech: Speech, speaking: Callable[[bool], None], player: Callable[[], Player] | None = None,
                 heard: Callable[[object, float], None] | None = None) -> None:
        self.speech, self.speaking = speech, speaking
        self.heard = heard or (lambda tag, seconds: None)
        self.voice: str | None = None
        self.player = (player or Player)()
        self.texts: asyncio.Queue = asyncio.Queue()
        self.sounds: asyncio.Queue = asyncio.Queue()
        self.failed: Exception | None = None  # the first piece that couldn't be made
        loop = asyncio.get_running_loop()
        self.tasks = [loop.create_task(self._make()), loop.create_task(self._play())]

    def add(self, text: str, tag: object = None) -> None:
        text = speakable(text)
        if not text:
            return
        if self.voice is None:
            self.voice = self.speech.voice_for(text)
        self.texts.put_nowait((text, tag))

    def close(self) -> None:
        """Nothing more to say: it ends once what's queued has been played."""
        self.texts.put_nowait(None)

    def cancel(self) -> None:
        self.player.abort()
        for task in self.tasks:
            task.cancel()

    async def done(self) -> None:
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _make(self) -> None:
        while (item := await self.texts.get()) is not None:
            text, tag = item
            try:
                self.sounds.put_nowait((await self.speech.synthesize(text, self.voice), tag))
            except Exception as e:  # noqa: BLE001 - the rest may still work
                log.warning("tts failed: %s", e)
                self.failed = self.failed or e
                self.sounds.put_nowait((None, tag))
        self.sounds.put_nowait(None)

    async def _play(self) -> None:
        started = False
        try:
            while (item := await self.sounds.get()) is not None:
                audio, tag = item
                if audio is None:  # couldn't be made: its words needn't wait for it
                    self.heard(tag, 0.0)
                    continue
                if not started:
                    started = True
                    self.speaking(True)
                self.heard(tag, wav_seconds(audio))
                try:
                    await asyncio.to_thread(self.player.play, audio)
                except Exception as e:  # noqa: BLE001
                    log.debug("tts playback failed: %s", e)
                finally:
                    try:
                        os.remove(audio)
                    except OSError:
                        pass
            await asyncio.to_thread(self.player.close)
        finally:
            if started:
                self.speaking(False)


class Player:
    """Plays one WAV after another without a gap: where ``sounddevice`` is
    (Windows, macOS), into one output stream that stays open between them;
    elsewhere, each file on its own (:func:`play_wav`)."""

    def __init__(self) -> None:
        self.output = None
        self.rate = self.channels = 0
        self.stopped = False

    def play(self, path: Path) -> None:
        if self.stopped:
            return
        try:
            import numpy as np
            import sounddevice
        except ImportError:
            return play_wav(path)
        with wave.open(str(path)) as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if width != 2:
            return play_wav(path)
        samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
        if self.output is None or (rate, channels) != (self.rate, self.channels):
            self.close()
            self.output = sounddevice.OutputStream(samplerate=rate, channels=channels, dtype="int16")
            self.output.start()
            self.rate, self.channels = rate, channels
        self.output.write(samples)  # returns once it's in the stream's buffer: the next follows on at once

    def close(self) -> None:
        """Let what's written play out, then release the device."""
        output, self.output = self.output, None
        if output is not None:
            output.stop()
            output.close()

    def abort(self) -> None:
        """Stop at once (a new reply's speech takes over)."""
        self.stopped = True
        output = self.output
        if output is not None:
            try:
                output.abort()
            except Exception:  # noqa: BLE001 - it's going anyway
                pass


def wav_seconds(path: Path) -> float:
    """How long a WAV file lasts (0 if it can't be read)."""
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return 0.0


SENTENCE_END = re.compile(r"[.!?…]+[\"'»“”)]*(?=\s)|\n")
FIRST_WORDS = 6  # the first piece may end at a comma after this many words, so the voice (and the words) start sooner…
LONG_WORDS = 25  # …and a long sentence at one after this many, so a piece is never long to make


def speech_ends(text: str, start: int = 0, first: bool = False) -> list[int]:
    """Where the next pieces to speak end in ``text`` (after ``start``): at
    the end of each sentence that's complete — followed by a space, so "3.5"
    and a sentence still being written don't count — or at a line break; and
    at a comma once a piece is long (FIRST_WORDS for the reply's first piece,
    LONG_WORDS after), so speaking needn't wait for its end."""
    ends, at = [], start
    while True:
        limit = FIRST_WORDS if first and not ends else LONG_WORDS
        match = SENTENCE_END.search(text, at)
        stop = match.end() if match else len(text)
        piece = text[at:stop]
        comma = None
        if len(piece.split()) > limit:
            # The first comma after enough words, with more of the sentence written after it.
            for found in re.finditer(r",(?=\s)", piece):
                if len(piece[:found.end()].split()) >= limit and piece[found.end():].strip():
                    comma = at + found.end()
                    break
        if comma is not None:
            ends.append(comma)
            at = comma
            continue
        if match is None:
            return ends
        if text[at:match.end()].strip():
            ends.append(match.end())
        at = match.end()


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
