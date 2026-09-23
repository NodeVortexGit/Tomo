#!/usr/bin/env python3
"""Google Speech-to-Text helper — transcribe a WAV file to text on stdout.

Called by tomo-core's speech bridge:

    python3 stt_google.py --key $GOOGLE_STT_API_KEY --lang en-US \
        --audio /path/mic.wav

It POSTs to the Speech-to-Text v1 REST endpoint using an API key, so it needs
no service-account JSON — only the key from .env. Uses the standard library
only, so there is nothing extra to install for STT.

Note: the input WAV must be LINEAR16 mono at 16 kHz — which is exactly what the
Rust side records with parecord/arecord.
"""
import argparse
import base64
import json
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize?key={key}"


def transcribe(key: str, lang: str, audio_path: str) -> str:
    with open(audio_path, "rb") as fh:
        raw = fh.read()
    # Skip nothing — send the whole WAV; the API reads the header.
    content = base64.b64encode(raw).decode("ascii")

    payload = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "languageCode": lang,
            "enableAutomaticPunctuation": True,
        },
        "audio": {"content": content},
    }
    req = urllib.request.Request(
        ENDPOINT.format(key=key),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    # Concatenate the top alternative of each result segment.
    parts = []
    for result in body.get("results", []):
        alts = result.get("alternatives", [])
        if alts:
            parts.append(alts[0].get("transcript", "").strip())
    return " ".join(p for p in parts if p).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    try:
        text = transcribe(args.key, args.lang, args.audio)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        sys.stderr.write(f"google stt HTTP {exc.code}: {detail}\n")
        return 1
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"stt failed: {exc}\n")
        return 1

    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
