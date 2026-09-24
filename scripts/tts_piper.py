#!/usr/bin/env python3
"""Tomo's voice: Piper neural text-to-speech, on this computer.

Runs as a long-lived child of Tomo's brain (crates/tomo-core/src/speech.rs).
It loads the voice once — downloading it into --voices-dir the first time —
then answers each JSON line on stdin,

  {"text": "Hello!", "out": "/path/to/file.wav", "speed": 1.0}

by writing the WAV file and printing {"ok": true}, or {"error": "..."}.
The first line it prints, once the voice is loaded, is
{"ok": true, "event": "ready"}.

Try it:
  echo '{"text": "Hello!", "out": "hi.wav"}' | tts_piper.py --voice en_US-lessac-medium --voices-dir voices
"""

import argparse
import json
import sys
import wave
from pathlib import Path


def reply(**fields):
    print(json.dumps(fields), flush=True)


def load_voice(name, voices_dir):
    from piper import PiperVoice

    voices_dir.mkdir(parents=True, exist_ok=True)
    model = voices_dir / f"{name}.onnx"
    if not model.exists() or not Path(f"{model}.json").exists():
        from piper.download_voices import download_voice

        download_voice(name, voices_dir)
    return PiperVoice.load(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voice", required=True, help="Piper voice, e.g. en_US-lessac-medium")
    parser.add_argument("--voices-dir", required=True, type=Path, help="where voices are kept")
    args = parser.parse_args()
    try:
        from piper import SynthesisConfig

        voice = load_voice(args.voice, args.voices_dir)
    except Exception as error:  # report, don't traceback: the brain reads stdout
        reply(error=f"couldn't load the voice {args.voice}: {error}")
        return 1
    reply(ok=True, event="ready")

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            speed = float(request.get("speed") or 1.0)
            # Faster speech is shorter phonemes; 1.0 keeps the voice's own pace.
            length = None if speed == 1.0 else voice.config.length_scale / max(speed, 0.1)
            with wave.open(str(request["out"]), "wb") as wav:
                voice.synthesize_wav(request["text"], wav, syn_config=SynthesisConfig(length_scale=length))
            reply(ok=True)
        except Exception as error:
            reply(error=str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
