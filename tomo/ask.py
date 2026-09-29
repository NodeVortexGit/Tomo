"""Ask Tomo's brain one thing from the terminal, without the window — handy
for checking that the local model (Ollama or LM Studio) answers, and that the
tools work::

    python -m tomo ask "what's on my screen?"
    python -m tomo ask --speak "say hi"            (and hear it)
    python -m tomo ask --run "how much disk is free?"
    python -m tomo ask --image photo.jpg "when was this taken?"

Commands only run with ``--run`` (otherwise they're refused; either way
they're written to ``tomo-ask-audit.log`` in the temp folder), and nothing is
saved to Tomo's memory. ``--image`` (up to four) sends a picture as the
chat's image button does; its copy goes to the temp folder.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import attachments
from .ai import AiClient, SharedCatalog
from .apps import SystemCatalog
from .commands import Executor
from .config import Config
from .db import Store

USAGE = "usage: tomo ask [--speak] [--run] [--image FILE]… <question>"


def main(cfg: Config, args: list[str]) -> int:
    speak = run = False
    images: list[Path] = []
    words = []
    rest = iter(args)
    for arg in rest:
        if arg == "--speak":
            speak = True
        elif arg == "--run":
            run = True
        elif arg == "--image":
            image = next(rest, None)
            if image is None:
                print(USAGE, file=sys.stderr)
                return 2
            images.append(Path(image))
        else:
            words.append(arg)
    question = " ".join(words).strip()
    if not question and not images:
        print(USAGE, file=sys.stderr)
        return 2
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    return asyncio.run(ask(cfg, question, run, speak, images))


async def ask(cfg: Config, question: str, run: bool, speak: bool, images: list[Path] | None = None) -> int:
    executor = Executor(run, Path(tempfile.gettempdir()) / "tomo-ask-audit.log", cfg.extra_allowed_commands)
    # The installed apps, as the app has them, so open_app works here too.
    catalog = SharedCatalog(await asyncio.to_thread(SystemCatalog.scan))
    # A memory of its own, in memory only: nothing is kept.
    ai = AiClient(cfg, Store(None), executor, catalog, threading.Event())
    sent = []
    for image in (images or [])[:attachments.MAX_PER_MESSAGE]:
        try:
            sent.append(attachments.prepare(image, Path(tempfile.gettempdir()) / "tomo-ask-images"))
        except (OSError, ValueError) as e:
            print(f"couldn't attach {image}: {e}", file=sys.stderr)
            return 1
    for picture in sent:
        print(f"[image] {picture.source}\n{picture.about}", file=sys.stderr)
    # As the app does at start: load the model and have it read the persona.
    await ai.warm_up()
    body = []
    asked = time.monotonic()
    reply = await ai.respond(question, body.append, tuple(sent))
    print(f"(answered in {time.monotonic() - asked:.1f} s)", file=sys.stderr)
    for event in body:
        print(f"[body] {event}")
    print(reply)
    if speak:
        from .speech import Speech

        speech = Speech(cfg)
        await speech.play(await speech.synthesize(reply))
    return 0
