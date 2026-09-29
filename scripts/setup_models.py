#!/usr/bin/env python3
"""Download the speech models Tomo uses. They all run on this computer:

  Vosk, small English    spots "Hey Tomo"          <data>/models/vosk-model-small-en-us-0.15
  Parakeet TDT 0.6B v3   turns speech into text,   <data>/models/parakeet-tdt-0.6b-v3
                         English or Bulgarian
  three Piper voices     Tomo's own voice: male    <data>/voices/<voice>.onnx
                         and female English, and
                         Bulgarian
  MediaPipe Pose Lite    the joints, for the       <data>/models/pose_landmarker_lite.task
                         health programme

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
# NVIDIA's multilingual speech recogniser (25 European languages, Bulgarian
# among them; CC-BY-4.0), as int8 ONNX: about 670 MB, fast on a CPU.
STT = "parakeet-tdt-0.6b-v3"
STT_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
STT_FILES = ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]
POSE = "pose_landmarker_lite.task"
POSE_URL = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/{POSE}"
DEFAULT_VOICE_MALE = "en_GB-alan-medium"
DEFAULT_VOICE_FEMALE = "en_GB-cori-medium"
DEFAULT_VOICE_BG = "bg_BG-dimitar-medium"


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


def fetch_stt(models):
    target = models / STT
    if all((target / name).exists() for name in STT_FILES):
        return say(f"  Parakeet is already there ({STT})")
    say(f"  Downloading Parakeet ({STT}, about 670 MB)…")
    from huggingface_hub import snapshot_download

    snapshot_download(STT_REPO, local_dir=str(target), allow_patterns=STT_FILES)


def fetch_pose(models):
    target = models / POSE
    if target.exists():
        return say(f"  The pose model is already there ({POSE})")
    say(f"  Downloading the pose model ({POSE}, about 6 MB)…")
    with urllib.request.urlopen(POSE_URL) as response:
        target.write_bytes(response.read())


def fetch_voice(voices, voice):
    if (voices / f"{voice}.onnx").exists():
        return say(f"  The voice is already there ({voice})")
    say(f"  Downloading a voice for Tomo ({voice}, about 60 MB)…")
    from piper.download_voices import download_voice

    voices.mkdir(parents=True, exist_ok=True)
    download_voice(voice, voices)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="Tomo's data folder")
    parser.add_argument("--voice-male", default=os.environ.get("TOMO_TTS_VOICE_MALE") or DEFAULT_VOICE_MALE)
    parser.add_argument("--voice-female", default=os.environ.get("TOMO_TTS_VOICE_FEMALE") or DEFAULT_VOICE_FEMALE)
    parser.add_argument("--voice-bg", default=os.environ.get("TOMO_TTS_VOICE_BG") or DEFAULT_VOICE_BG)
    args = parser.parse_args()
    data = args.data_dir or data_dir()
    models = data / "models"
    models.mkdir(parents=True, exist_ok=True)
    say(f"Speech models for Tomo, in {data}")
    failed = False
    steps = (
        lambda: fetch_vosk(models),
        lambda: fetch_stt(models),
        lambda: fetch_pose(models),
        lambda: fetch_voice(data / "voices", args.voice_male),
        lambda: fetch_voice(data / "voices", args.voice_female),
        lambda: fetch_voice(data / "voices", args.voice_bg),
    )
    for step in steps:
        try:
            step()
        except Exception as error:
            failed = True
            say(f"  … that didn't work: {error}")
    say("Some downloads failed; run this again to retry." if failed else "Done.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
