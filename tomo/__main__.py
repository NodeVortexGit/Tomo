"""Starting Tomo: ``python -m tomo`` (or the ``tomo`` command).

* ``tomo`` — the desktop companion: the brain on its own thread, the body
  (window, character, chat) on this one.
* ``tomo ask [--run] [--speak] <question>`` — one question to the brain from
  the terminal, without the window (see :mod:`tomo.ask`).

Where ``.env`` and ``scripts/`` are: ``TOMO_ROOT`` if set, else the current
folder when it has them (running from the source tree), else the folder the
program was installed to. Only one Tomo runs at a time.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from . import platform
from .config import Config

log = logging.getLogger("tomo")


def default_root() -> Path:
    if env := os.environ.get("TOMO_ROOT"):
        return Path(env)
    cwd = Path.cwd()
    if (cwd / "scripts").is_dir() or (cwd / ".env").is_file():
        return cwd
    here = Path(__file__).resolve().parent.parent  # the source tree, or the install folder
    if (here / "scripts").is_dir():
        return here
    return Path(sys.executable).resolve().parent


def setup_logging(data_dir: Path) -> None:
    """``tomo.log`` in the data folder (and the console, if there is one).
    ``TOMO_LOG=debug`` shows more."""
    level = getattr(logging, os.environ.get("TOMO_LOG", "info").upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    try:
        file = logging.handlers.RotatingFileHandler(data_dir / "tomo.log", maxBytes=1_000_000, backupCount=2,
                                                    encoding="utf-8")
        file.setFormatter(fmt)
        root.addHandler(file)
    except OSError:
        pass
    if sys.stderr is not None:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def single_instance(data_dir: Path):
    """A lock only one Tomo can hold (the OS drops it on any exit). None:
    another Tomo has it."""
    handle = open(data_dir / "tomo.lock", "a+")  # noqa: SIM115 - kept open for the process's life
    try:
        handle.seek(0)
        if platform.WINDOWS:
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = default_root()
    cfg = Config.load(root)
    if argv[:1] == ["ask"]:
        from .ask import main as ask

        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        return ask(cfg, argv[1:])
    setup_logging(cfg.data_dir)
    lock = single_instance(cfg.data_dir)
    if lock is None:
        log.info("Tomo is already running")
        return 0
    log.info("%s", cfg.redacted())
    from .brain import Brain

    brain = Brain.start(cfg)
    try:
        from .body.app import App

        App(brain).run()
    except Exception:
        log.exception("the window stopped with an error")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
