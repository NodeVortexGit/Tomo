#!/usr/bin/env python3
"""Edge-TTS helper — synthesize speech to an mp3 file.

Called by tomo-core's speech bridge:

    python3 tts_edge.py --voice en-US-AriaNeural --rate +0% \
        --out /path/last_tts.mp3 --text "Hello there"

Uses Microsoft Edge's free neural voices via the `edge-tts` package. The Rust
side handles playback, so this script only writes the file and exits.
"""
import argparse
import asyncio
import sys


async def synth(text: str, voice: str, rate: str, out: str) -> None:
    try:
        import edge_tts
    except ImportError:
        sys.stderr.write(
            "edge-tts is not installed. Run install.sh, or:\n"
            "  pip install edge-tts\n"
        )
        raise

    # rate must look like "+0%" / "-10%"; normalise a bare number just in case.
    if rate and rate[0] not in "+-":
        rate = "+" + rate
    if rate and not rate.endswith("%"):
        rate = rate + "%"

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate or "+0%")
    await communicate.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-US-AriaNeural")
    ap.add_argument("--rate", default="+0%")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    if not args.text.strip():
        return 0
    try:
        asyncio.run(synth(args.text, args.voice, args.rate, args.out))
    except Exception as exc:  # noqa: BLE001 — surface any failure to the caller
        sys.stderr.write(f"tts failed: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
