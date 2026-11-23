#!/usr/bin/env python3
"""Tomo's Bulgarian voice: BgTTS-38M-V2 with MioCodec, on this computer's CPU.

Runs as a long-lived child of Tomo's brain (tomo/speech.py), the way
tts_piper.py does for English. It answers each JSON line on stdin,

  {"text": "Здравей!", "out": "/path/to/file.wav", "voice": "bgtts-female"}

by writing the WAV file and printing {"ok": true}, or {"error": "..."}. The
first line it prints, once it's loaded, is {"ok": true, "event": "ready"}.

The model clones a voice from recordings and needs no training: a voice
"bgtts-female" is the model's own samples/sample_female_bg*.wav, reduced once
to a speaker embedding (kept beside them, in voices/). scripts/setup_models.py
downloads the files this reads.

Try it:
  echo '{"text": "Здравей!", "out": "hi.wav", "voice": "bgtts-male"}' |
    tts_bgtts.py --voice bgtts-female --voice bgtts-male --models-dir <data>/models
"""

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_FOLDER = "bgtts-38m-v2"
CODEC_FOLDER = "miocodec-25hz-24khz"
TORCH_FOLDER = "torch"  # MioCodec also builds on torchaudio's WavLM, which torch keeps here
VOICE_PREFIX = "bgtts-"
LOUDEST = 0.9  # the model speaks quietly and each voice differently; Piper is about this loud
# The model's own recommendations.
TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY = 0.3, 250, 0.95, 1.1
MAX_CHUNK = 250  # characters the text encoder takes at once (its limit is 256, with two markers)

reply_to = sys.stdout


def reply(**fields):
    print(json.dumps(fields), file=reply_to, flush=True)


def load_model(folder):
    """Not the model's own load_for_inference: that unpickles the checkpoint
    with weights_only=False, which would run any code in the file."""
    import torch
    from model import TTSEncoderDecoder, V6Config

    checkpoint = torch.load(folder / "checkpoint_inference.pt", map_location="cpu", weights_only=True)
    model = TTSEncoderDecoder(V6Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def load_codec(folder):
    from miocodec import MioCodecModel

    codec = MioCodecModel.from_pretrained(config_path=str(folder / "config.yaml"),
                                          weights_path=str(folder / "model.safetensors"))
    return codec.eval()


def voice_embedding(name, folder, codec):
    """The speaker embedding of a voice: the mean over its Bulgarian samples."""
    import numpy as np
    import soundfile
    import torch
    import torchaudio

    who = name.removeprefix(VOICE_PREFIX)
    saved = folder / "voices" / f"{who}.npy"
    if saved.exists():
        return torch.from_numpy(np.load(saved, allow_pickle=False))
    samples = sorted((folder / "samples").glob(f"sample_{who}_bg*.wav"))
    if not samples:
        raise FileNotFoundError(f"no samples for the voice {name} in {folder}")
    each = []
    for sample in samples:
        audio, rate = soundfile.read(str(sample), dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio).mean(dim=1)
        if rate != codec.config.sample_rate:
            audio = torchaudio.functional.resample(audio, rate, codec.config.sample_rate)
        each.append(codec.encode(audio, return_content=False).global_embedding.reshape(-1))
    embedding = torch.stack(each).mean(dim=0)
    saved.parent.mkdir(exist_ok=True)
    np.save(saved, embedding.numpy())
    return embedding


def synthesize(text, embedding, model, tokenizer, codec):
    """``text`` as a waveform, the loudest moment at LOUDEST."""
    import torch
    from inference import _split_text, generate

    codes = []
    for chunk in _split_text(text, tokenizer, max_len=MAX_CHUNK):
        if not any(c.isalnum() for c in chunk):
            continue  # nothing to say, and the model refuses text it has no letters for
        part = generate(model, tokenizer, chunk, embedding, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
                        rep_penalty=REPETITION_PENALTY, device="cpu")
        if part is None or not len(part):
            raise RuntimeError(f"no sound for: {chunk[:60]}")
        codes.append(part)
    if not codes:
        raise RuntimeError("nothing to say out loud")
    wave = codec.decode(global_embedding=embedding, content_token_indices=torch.cat(codes)).reshape(-1)
    return wave * (LOUDEST / max(float(wave.abs().max()), 1e-6))


def main():
    global reply_to
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voice", required=True, action="append", help="bgtts-female, bgtts-male or bgtts-male2")
    parser.add_argument("--models-dir", required=True, type=Path, help="Tomo's models folder")
    args = parser.parse_args()
    # Tomo writes UTF-8; Windows would read it in its ANSI code page, turning
    # Bulgarian into gibberish.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    reply_to, sys.stdout = sys.stdout, sys.stderr  # the libraries print; the brain reads only replies
    os.environ["TORCH_HOME"] = str(args.models_dir / TORCH_FOLDER)
    folder = args.models_dir / MODEL_FOLDER
    try:
        import miocodec  # noqa: F401 - before the model's folder is on the path: it has a config.py, a model.py…
        import soundfile
        import torch

        sys.path.insert(0, str(folder))
        from tokenizer import TTSTokenizer

        model = load_model(folder)
        codec = load_codec(args.models_dir / CODEC_FOLDER)
        tokenizer = TTSTokenizer()
        voices = {name: voice_embedding(name, folder, codec) for name in args.voice}
    except Exception as error:  # report, don't traceback: the brain reads stdout
        reply(error=f"couldn't load the Bulgarian voice: {error}")
        return 1
    reply(ok=True, event="ready")

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            embedding = voices.get(request.get("voice"), voices[args.voice[0]])
            with torch.inference_mode():
                wave = synthesize(request["text"], embedding, model, tokenizer, codec)
            soundfile.write(str(request["out"]), wave.numpy(), codec.config.sample_rate, subtype="PCM_16")
            reply(ok=True)
        except Exception as error:
            reply(error=str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
