#!/usr/bin/env python3
"""Tomo's voice: Piper neural text-to-speech, on this computer.

Runs as a long-lived child of Tomo's brain (tomo/speech.py).
It loads its voices once — each --voice, downloading it into --voices-dir the
first time — then answers each JSON line on stdin,

  {"text": "Hello!", "out": "/path/to/file.wav", "speed": 1.0, "voice": "en_GB-cori-medium"}

by writing the WAV file and printing {"ok": true}, or {"error": "..."}.
"voice" picks one of the loaded voices (Tomo sends the Bulgarian one for a
reply in Bulgarian, and the character's own — male or female — for English);
without it, or for one that isn't loaded, the first --voice speaks. The first
line it prints, once the voices are loaded, is {"ok": true, "event": "ready"}.

Try it:
  echo '{"text": "Здравей!", "out": "hi.wav", "voice": "bg_BG-dimitar-medium"}' |
    tts_piper.py --voice en_GB-cori-medium --voice en_GB-alan-medium --voice bg_BG-dimitar-medium --voices-dir voices
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
    parser.add_argument(
        "--voice", required=True, action="append",
        help="Piper voice, e.g. en_GB-cori-medium; repeat for more (the first is the default)",
    )
    parser.add_argument("--voices-dir", required=True, type=Path, help="where voices are kept")
    args = parser.parse_args()
    # Tomo writes UTF-8; Windows would read it in its ANSI code page, turning
    # Bulgarian into gibberish.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    voices = {}
    try:
        from piper import SynthesisConfig

        voices[args.voice[0]] = load_voice(args.voice[0], args.voices_dir)
    except Exception as error:  # report, don't traceback: the brain reads stdout
        reply(error=f"couldn't load the voice {args.voice[0]}: {error}")
        return 1
    # The others are a bonus: without one, its lines get the default voice.
    for name in args.voice[1:]:
        try:
            voices[name] = load_voice(name, args.voices_dir)
        except Exception as error:
            print(f"couldn't load the voice {name}: {error}", file=sys.stderr, flush=True)
    reply(ok=True, event="ready")

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            voice = voices.get(request.get("voice")) or voices[args.voice[0]]
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
