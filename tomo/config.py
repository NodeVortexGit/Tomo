"""Settings: the ``.env`` file, the environment and the defaults.

Precedence, highest first: the real process environment, the ``.env`` file
in the project folder (or the current folder), then the defaults below.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_PERSONA = "Tomo"
# The English voices: the character's own, male or female (see speech.py).
DEFAULT_VOICE_MALE = "en_GB-alan-medium"
DEFAULT_VOICE_FEMALE = "en_GB-cori-medium"
# Piper's only Bulgarian voice so far, for every character.
DEFAULT_VOICE_BG = "bg_BG-dimitar-medium"
# Enough for the persona, the tools and a recent stretch of conversation.
DEFAULT_CONTEXT = 8192
# How long Ollama keeps the model loaded between replies.
DEFAULT_KEEP_ALIVE = "1h"


class LlmProvider(Enum):
    AUTO = "auto"  # Ollama if it's running, else LM Studio
    OLLAMA = "ollama"
    LM_STUDIO = "lmstudio"
    OPENAI_COMPATIBLE = "openai"  # any other OpenAI-compatible server, at llm_url

    @classmethod
    def parse(cls, value: str) -> "LlmProvider | None":
        v = value.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        return {
            "": cls.AUTO,
            "auto": cls.AUTO,
            "ollama": cls.OLLAMA,
            "lmstudio": cls.LM_STUDIO,
            "openai": cls.OPENAI_COMPATIBLE,
            "openaicompatible": cls.OPENAI_COMPATIBLE,
        }.get(v)


@dataclass
class Config:
    # ---- the model: a local server --------------------------------------------
    llm_provider: LlmProvider = LlmProvider.AUTO
    llm_url: str = ""  # empty: the provider's usual address on this machine
    llm_model: str = ""  # empty: the first local chat model the server has
    llm_api_key: str = ""  # only for a server that asks for one; never logged
    llm_context: int = DEFAULT_CONTEXT
    # Let a thinking model reason before it answers (Ollama). Off: with it on,
    # a 35B model took a minute or more a reply against a second or two.
    llm_think: bool = False
    llm_keep_alive: str = DEFAULT_KEEP_ALIVE
    # ---- speech (all on this machine) ------------------------------------------
    tts_voice_male: str = DEFAULT_VOICE_MALE
    tts_voice_female: str = DEFAULT_VOICE_FEMALE
    tts_voice_bg: str = DEFAULT_VOICE_BG
    tts_speed: float = 1.0
    # ---- character / persona -------------------------------------------------
    persona_name: str = DEFAULT_PERSONA
    character_path: Path = Path()
    # ---- paths ------------------------------------------------------------------
    data_dir: Path = Path()
    audit_log_path: Path = Path()
    scripts_dir: Path = Path()
    assets_dir: Path = Path()
    # ---- behaviour switches ---------------------------------------------------
    allow_command_execution: bool = True
    extra_allowed_commands: list[str] = field(default_factory=list)
    allow_screen: bool = True
    wake_word: bool = True
    # How memories are embedded for search: "ngram" (built in, offline,
    # language-independent) or "minilm" (ChromaDB's stock English model).
    embeddings: str = "ngram"

    @property
    def chroma_dir(self) -> Path:
        """Where ChromaDB keeps the memory."""
        return self.data_dir / "chroma"

    @property
    def legacy_sqlite(self) -> Path:
        """The Rust version's SQLite memory, migrated on first start."""
        return self.data_dir / "memory.sqlite3"

    @classmethod
    def load(cls, project_root: Path) -> "Config":
        """Load ``.env`` (if present), then build the settings."""
        try:
            from dotenv import load_dotenv

            for candidate in (project_root / ".env", Path.cwd() / ".env"):
                if candidate.is_file():
                    load_dotenv(candidate, override=False)
        except ImportError:  # pragma: no cover - a broken install
            log.warning("python-dotenv is missing: .env was not read")

        data_dir = default_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir = env_path("TOMO_SCRIPTS_DIR") or project_root / "scripts"
        # A relative character path is from the project, not from wherever
        # Tomo happened to be started.
        character = env_path("TOMO_CHARACTER")
        if character is not None and not character.is_absolute():
            character = project_root / character
        provider = LlmProvider.AUTO
        if (value := env_str("TOMO_LLM")) is not None:
            parsed = LlmProvider.parse(value)
            if parsed is None:
                log.warning("TOMO_LLM=%r isn't one of auto, ollama, lmstudio, openai; using auto", value)
            else:
                provider = parsed
        speed = env_float("TOMO_TTS_SPEED", 1.0)
        return cls(
            llm_provider=provider,
            llm_url=env_str("TOMO_LLM_URL") or "",
            llm_model=env_str("TOMO_LLM_MODEL") or "",
            llm_api_key=env_str("TOMO_LLM_API_KEY") or "",
            llm_context=int(env_float("TOMO_LLM_CONTEXT", DEFAULT_CONTEXT)),
            llm_think=env_bool("TOMO_LLM_THINK", False),
            llm_keep_alive=env_str("TOMO_LLM_KEEP_ALIVE") or DEFAULT_KEEP_ALIVE,
            tts_voice_male=env_str("TOMO_TTS_VOICE_MALE") or DEFAULT_VOICE_MALE,
            tts_voice_female=env_str("TOMO_TTS_VOICE_FEMALE") or DEFAULT_VOICE_FEMALE,
            tts_voice_bg=env_str("TOMO_TTS_VOICE_BG") or DEFAULT_VOICE_BG,
            tts_speed=speed if 0.1 < speed < 5.0 else 1.0,
            persona_name=env_str("TOMO_PERSONA") or DEFAULT_PERSONA,
            character_path=character or data_dir / "characters" / "default.vrm",
            data_dir=data_dir,
            audit_log_path=data_dir / "command-audit.log",
            scripts_dir=scripts_dir,
            assets_dir=project_root / "assets",
            allow_command_execution=env_bool("TOMO_ALLOW_COMMANDS", True),
            extra_allowed_commands=[
                p.strip() for p in (env_str("TOMO_EXTRA_ALLOWED") or "").split(",") if p.strip()
            ],
            allow_screen=env_bool("TOMO_ALLOW_SCREEN", True),
            wake_word=env_bool("TOMO_WAKE_WORD", True),
            embeddings=(env_str("TOMO_EMBEDDINGS") or "ngram").lower(),
        )

    @classmethod
    def for_tests(cls, root: Path) -> "Config":
        """Defaults with everything under ``root``, nothing read from the
        environment: for tests that must not touch the real setup."""
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            character_path=data_dir / "characters" / "default.vrm",
            data_dir=data_dir,
            audit_log_path=data_dir / "command-audit.log",
            scripts_dir=root / "scripts",
            assets_dir=root / "assets",
            allow_command_execution=False,
            allow_screen=False,
            wake_word=False,
        )

    def redacted(self) -> str:
        """A view safe to print in logs (no key)."""
        auto = lambda s: s or "auto"  # noqa: E731
        return (
            f"Config {{ llm: {self.llm_provider.value} (url: {auto(self.llm_url)}, model: {auto(self.llm_model)}), "
            f"voices: {self.tts_voice_male} / {self.tts_voice_female} / {self.tts_voice_bg}, "
            f"persona: {self.persona_name}, "
            f"wake_word: {self.wake_word}, data_dir: {self.data_dir} }}"
        )


def default_data_dir() -> Path:
    """Tomo's data folder: TOMO_DATA_DIR, else the platform's usual place —
    ``%APPDATA%\\tomo\\tomo\\data`` on Windows (as the Rust version used),
    ``~/.local/share/tomo`` on Linux."""
    if (custom := env_path("TOMO_DATA_DIR")) is not None:
        return custom
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "tomo" / "tomo" / "data"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "dev.tomo.tomo"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "tomo"


def env_str(key: str) -> str | None:
    value = os.environ.get(key)
    return value if value is not None and value.strip() else None


def env_path(key: str) -> Path | None:
    value = env_str(key)
    return Path(value) if value is not None else None


def env_bool(key: str, default: bool) -> bool:
    value = env_str(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_float(key: str, default: float) -> float:
    value = env_str(key)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default
