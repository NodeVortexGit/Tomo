#!/usr/bin/env python3
"""Download the speech models Tomo uses. They all run on this computer:

  Vosk, small English    spots "Hey Tomo"          <data>/models/vosk-model-small-en-us-0.15
  Whisper, base.en       turns speech into text    <data>/models/whisper-base.en
  a Piper voice          Tomo's own voice          <data>/voices/<voice>.onnx

<data> is Tomo's data folder — the same one the app uses: ~/.local/share/tomo
on Linux, %APPDATA%\\tomo\\tomo\\data on Windows — unless --data-dir or
TOMO_DATA_DIR says otherwise. Models already there are skipped, so running
this again is quick. Needs the internet only while downloading.
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

VOSK = "vosk-model-small-en-us-0.15"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK}.zip"
WHISPER = "base.en"
DEFAULT_VOICE = "en_US-lessac-medium"


def data_dir():
    """Tomo's data folder, as the app finds it (the Rust `directories` crate)."""
    if os.environ.get("TOMO_DATA_DIR"):
        return Path(os.environ["TOMO_DATA_DIR"])
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "tomo" / "tomo" / "data"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "dev.tomo.tomo"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "tomo"


def say(message):
    print(message, flush=True)


def fetch_vosk(models):
    if (models / VOSK).is_dir():
        return say(f"  Vosk is already there ({VOSK})")
    say(f"  Downloading Vosk ({VOSK}, about 40 MB)…")
    with urllib.request.urlopen(VOSK_URL) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(models)


def fetch_whisper(models):
    target = models / f"whisper-{WHISPER}"
    if (target / "model.bin").exists():
        return say(f"  Whisper is already there ({WHISPER})")
    say(f"  Downloading Whisper ({WHISPER}, about 150 MB)…")
    from faster_whisper import download_model

    download_model(WHISPER, output_dir=str(target))


def fetch_voice(voices, voice):
    if (voices / f"{voice}.onnx").exists():
        return say(f"  The voice is already there ({voice})")
    say(f"  Downloading Tomo's voice ({voice}, about 60 MB)…")
    from piper.download_voices import download_voice

    voices.mkdir(parents=True, exist_ok=True)
    download_voice(voice, voices)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="Tomo's data folder")
    parser.add_argument("--voice", default=os.environ.get("TOMO_TTS_VOICE") or DEFAULT_VOICE)
    args = parser.parse_args()
    data = args.data_dir or data_dir()
    models = data / "models"
    models.mkdir(parents=True, exist_ok=True)
    say(f"Speech models for Tomo, in {data}")
    failed = False
    for step in (lambda: fetch_vosk(models), lambda: fetch_whisper(models), lambda: fetch_voice(data / "voices", args.voice)):
        try:
            step()
        except Exception as error:
            failed = True
            say(f"  … that didn't work: {error}")
    say("Some downloads failed; run this again to retry." if failed else "Done.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
