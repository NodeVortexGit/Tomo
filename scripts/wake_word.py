#!/usr/bin/env python3
"""Voice input for Tomo — "Hey Tomo" and push-to-talk — fully offline.

Runs as a long-lived child of Tomo's brain (see tomo/wake.py).
It reads the microphone as 16 kHz mono PCM (parecord / pw-record / arecord on
Linux, PortAudio through `sounddevice` on Windows and macOS) and works in two
stages:

  1. Vosk, restricted to a tiny grammar, spots the wake phrase for very little
     CPU;
  2. Parakeet (NVIDIA's parakeet-tdt-0.6b-v3, int8, on the CPU) confirms it
     (weeding out Vosk's false alarms) and transcribes what follows — said in
     one breath ("Hey Tomo, what's on my screen?") or after a pause ("Hey
     Tomo" … "open Firefox"), in English or Bulgarian ("Хей, Томо, колко е
     часът?"): it tells the languages apart by itself, and writes each in its
     own script.

No audio leaves the machine. Output is one JSON object per line on stdout:

  {"event": "ready"}                   listening
  {"event": "wake"}                    wake phrase confirmed; a request follows
  {"event": "heard", "text": "...", "language": "en" | "bg"}
                                       the request, transcribed
  {"event": "idle"}                    woken, but no request came
  {"event": "error", "message": "..."}

Lines on stdin steer it: "listen" (push-to-talk: skip the wake phrase),
"mute" / "unmute" (e.g. while Tomo speaks, so she can't wake herself).

With --push-to-talk there's no wake phrase: the microphone opens only for a
"listen", and closes again once the request is in.

Test without a microphone:  wake_word.py --vosk DIR --stt DIR --file clip.wav
"""

import argparse
import collections
import json
import os
import queue
import re
import subprocess
import sys
import threading
import wave

import numpy as np

RATE = 16000
CHUNK = RATE // 4  # samples per read: 0.25 s
# Speech recognition threads: enough for a request in a fraction of a second,
# while leaving the rest of the CPU to Piper and the model.
STT_THREADS = 4
# Audio kept from before the wake phrase was spotted, so Parakeet hears all of it.
PRE_ROLL_CHUNKS = 10  # 2.5 s
# After spotting the phrase: how long to wait for more words before settling
# for "just the wake phrase", and the longest request we record.
CONTINUE_SECONDS = 1.5
REQUEST_START_SECONDS = 7
MAX_RECORD_SECONDS = 15

# Vosk's model is English, but "Хей, Томо" sounds the same to it.
WAKE_GRAMMAR = ["hey tomo", "hi tomo", "hey tamo", "hey toma", "tomo", "[unk]"]
WAKE_SPOTTED = re.compile(r"\bt[ao]m[oa]\b")
# How the recogniser may write the name, in either language; everything after
# it is the request.
WAKE_PHRASE = re.compile(
    r"(?<!\w)(?:hey|hi|hay|ok(?:ay)?|хей|хай|ей|здравей)?[\s,]*(?<!\w)"
    r"(?:tomo|tommo|tomoe|tomu|tamo|toma|томо|томмо|тому|тамо|тома)(?!\w)[\s,.!?]*",
    re.I,
)
# In Bulgarian the recogniser may drop the name, which it doesn't know
# ("Хей, Томо, колко е часът?" → "Hej, колко часа!"): after Vosk heard it,
# a greeting at the start is enough.
GREETING = re.compile(r"^\s*(?:hey|hej|hi|hay|хей|хай|ей)(?!\w)[\s,.!?]*", re.I)
# What a recogniser tends to "hear" in near-silence.
HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "bye", "ъ", "м", "мм"}
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")


def language_of(text):
    """"bg" when most letters are Cyrillic, else "en" (as Tomo's brain decides)."""
    return "bg" if len(CYRILLIC.findall(text)) > len(LATIN.findall(text)) else "en"

RECORDERS = [
    ["parecord", "--raw", f"--rate={RATE}", "--channels=1", "--format=s16le"],
    ["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"],
    ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(RATE), "-c", "1"],
]


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def microphone():
    """Yield 0.25 s chunks of 16-bit audio: from the first recorder program
    that works on Linux, else through PortAudio. Closing the generator stops
    the recording."""
    if sys.platform.startswith("linux"):
        for command in RECORDERS:
            try:
                recorder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                continue
            try:
                chunk = recorder.stdout.read(CHUNK * 2)
                if len(chunk) < CHUNK * 2:
                    continue
                while len(chunk) == CHUNK * 2:
                    yield chunk
                    chunk = recorder.stdout.read(CHUNK * 2)
                raise RuntimeError(f"{command[0]} stopped recording")
            finally:
                recorder.kill()
    try:
        import sounddevice
    except ImportError:
        raise RuntimeError("no microphone recorder worked (tried parecord, pw-record, arecord)")
    chunks = queue.Queue()
    with sounddevice.RawInputStream(
        samplerate=RATE, blocksize=CHUNK, channels=1, dtype="int16",
        callback=lambda data, frames, time, status: chunks.put(bytes(data)),
    ):
        while True:
            yield chunks.get()


def audio_file(path):
    """Yield a WAV file's audio (16-bit, mono) as microphone-like chunks, then
    2 s of silence."""
    with wave.open(path) as clip:
        rate = clip.getframerate()
        samples = np.frombuffer(clip.readframes(clip.getnframes()), dtype=np.int16).astype(np.float32)
    if rate != RATE:
        count = int(len(samples) * RATE / rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples)
    samples = np.concatenate([samples.astype(np.int16), np.zeros(RATE * 2, dtype=np.int16)])
    for start in range(0, len(samples) - CHUNK + 1, CHUNK):
        yield samples[start : start + CHUNK].tobytes()


class Listener:
    def __init__(self, vosk_dir, stt_dir):
        import onnx_asr
        import onnxruntime
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        self.KaldiRecognizer = KaldiRecognizer
        self.vosk = Model(vosk_dir)
        self.spotter = KaldiRecognizer(self.vosk, RATE, json.dumps(WAKE_GRAMMAR))
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = STT_THREADS
        options.log_severity_level = 3  # errors only: stdout/stderr aren't for chatter
        self.stt = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", stt_dir, quantization="int8",
            sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.pre_roll = collections.deque(maxlen=PRE_ROLL_CHUNKS)
        self.commands = queue.Queue()
        self.muted = False

    def run(self, chunks):
        chunks = iter(chunks)
        emit("ready")
        for chunk in chunks:
            if self.steer() == "listen":
                emit("wake")
                self.take_request(chunks)
                continue
            self.pre_roll.append(chunk)
            if self.muted:
                continue
            if self.spotter.AcceptWaveform(chunk):
                spotted = json.loads(self.spotter.Result())["text"]
            else:
                spotted = json.loads(self.spotter.PartialResult())["partial"]
            if WAKE_SPOTTED.search(spotted):
                self.spotter.Reset()
                self.confirm_wake(chunks)
                self.pre_roll.clear()

    def push_to_talk(self):
        """No wake phrase: open the microphone only when asked, for one
        request, then close it again."""
        emit("ready")
        while True:
            if self.commands.get() != "listen":
                continue
            emit("wake")
            chunks = microphone()
            try:
                self.take_request(chunks)
            finally:
                chunks.close()

    def steer(self):
        """Apply pending stdin commands; return "listen" if one asks for it."""
        wanted = None
        while not self.commands.empty():
            command = self.commands.get_nowait()
            if command in ("mute", "unmute"):
                self.muted = command == "mute"
                self.spotter.Reset()
            elif command == "listen":
                wanted = command
        return wanted

    def confirm_wake(self, chunks):
        """Stage 2: record to the end of the utterance and let Parakeet decide."""
        more, _ = self.record(chunks, MAX_RECORD_SECONDS, start_within=CONTINUE_SECONDS)
        text = self.transcribe(list(self.pre_roll) + more)
        phrase = WAKE_PHRASE.search(text) or GREETING.match(text)
        if not phrase:
            return  # Vosk's false alarm
        emit("wake")
        request = text[phrase.end():].strip(" ,.!?")
        if request:
            emit("heard", text=request, language=language_of(request))
        else:
            self.take_request(chunks)

    def take_request(self, chunks):
        audio, spoke = self.record(chunks, MAX_RECORD_SECONDS, start_within=REQUEST_START_SECONDS)
        text = self.transcribe(audio) if spoke else ""
        if text.lower().strip(" .!?") in HALLUCINATIONS:
            emit("idle")
        else:
            emit("heard", text=text, language=language_of(text))

    def record(self, chunks, max_seconds, start_within):
        """Record until the speaker finishes (Vosk's end-of-utterance
        detection), for at most `max_seconds`; give up if nobody speaks within
        `start_within`. Returns (chunks, whether anyone spoke)."""
        endpoint = self.KaldiRecognizer(self.vosk, RATE)
        audio, spoke = [], False
        for chunk in chunks:
            audio.append(chunk)
            if endpoint.AcceptWaveform(chunk):
                if json.loads(endpoint.Result())["text"]:
                    return audio, True
            elif json.loads(endpoint.PartialResult())["partial"]:
                spoke = True
            seconds = len(audio) * CHUNK / RATE
            if seconds >= max_seconds or (not spoke and seconds >= start_within):
                break
        return audio, spoke

    def transcribe(self, chunks):
        """What was said, in the language it was said in (no need to name it)."""
        samples = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768
        return self.stt.recognize(samples, sample_rate=RATE).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vosk", required=True, help="Vosk model directory")
    parser.add_argument("--stt", required=True, help="Parakeet model directory (parakeet-tdt-0.6b-v3, int8 ONNX)")
    parser.add_argument("--file", help="read audio from this file instead of the microphone")
    parser.add_argument("--push-to-talk", action="store_true", help="no wake phrase; listen only when asked")
    args = parser.parse_args()
    # UTF-8 both ways, whatever the system's code page (Windows).
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        listener = Listener(args.vosk, args.stt)
        if not args.file:

            def read_commands():
                for line in sys.stdin:
                    listener.commands.put(line.strip())
                os._exit(0)  # stdin closed: Tomo is gone, stop using the mic

            threading.Thread(target=read_commands, daemon=True).start()
        if args.push_to_talk and not args.file:
            listener.push_to_talk()
        else:
            listener.run(audio_file(args.file) if args.file else microphone())
    except Exception as error:  # report, don't traceback: the brain reads stdout
        emit("error", message=str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
