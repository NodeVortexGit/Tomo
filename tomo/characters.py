"""Which characters (``.vrm`` models) Tomo can appear as, switching, and
each one's voice.

The choices are every ``.vrm`` in the configured character's folder, the data
folder's ``characters/`` and the bundled ``assets/characters/``, plus any
imported file the memory knows. Switching — from the chat's character menu,
or by asking Tomo (the ``change_character`` tool) — is remembered for the
next start. A file from anywhere else is copied into the data folder first,
so it keeps working if the original is moved or deleted.

Each character has a voice, male or female, which the model chooses: the
first time a character appears, it's shown the character's picture (the
thumbnail inside the .vrm) and name and calls ``set_voice`` (see ai.py);
the user can have it changed any time by asking. speech.py turns the choice
into a Piper voice.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

from .config import Config
from .db import Store
from .events import CharacterChoice

VOICES = ("male", "female")


def startup(db: Store, cfg: Config) -> Path | None:
    """The character to show at start: the last one chosen, else the
    configured one, else the first on offer — whichever exists."""
    active = db.active_character()
    for candidate in ([Path(active.path)] if active else []) + [cfg.character_path]:
        if candidate.is_file():
            return candidate.resolve()
    choices = available(db, cfg)
    return choices[0].path if choices else None


def available(db: Store, cfg: Config) -> list[CharacterChoice]:
    """Every character on offer, sorted by name."""
    files: list[tuple[str | None, Path]] = [(c.name, Path(c.path)) for c in db.list_characters()]
    for folder in folders(cfg):
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            continue
        files += [(None, p) for p in entries if p.suffix.lower() == ".vrm"]
    seen: set[Path] = set()
    choices = []
    for name, path in files:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        choices.append(CharacterChoice(name or stem(resolved), resolved))
    choices.sort(key=lambda c: c.name.lower())
    return choices


def activate(db: Store, cfg: Config, path: Path, name: str) -> Path:
    """Make ``path`` the character, remembered for next time. A file from
    outside the character folders is copied in first. Returns the path to load."""
    path = Path(path).resolve(strict=True)
    in_a_folder = any(path.parent == folder.resolve() for folder in folders(cfg) if folder.exists())
    if not in_a_folder:
        target = cfg.data_dir / "characters"
        target.mkdir(parents=True, exist_ok=True)
        dest = target / (path.name or "character.vrm")
        shutil.copyfile(path, dest)
        path = dest.resolve()
    db.add_character(name, str(path))
    db.set_active_character(name)
    return path


def name_of(db: Store, path: Path) -> str:
    """The name a character file goes by: the memory's, else the file's."""
    resolved = Path(path).resolve()
    for c in db.list_characters():
        try:
            if Path(c.path).resolve() == resolved:
                return c.name
        except OSError:
            continue
    return stem(resolved)


def voice_of(db: Store) -> str:
    """The showing character's voice: "male", "female", or "" (not chosen yet)."""
    active = db.active_character()
    return active.voice if active is not None and active.voice in VOICES else ""


def set_voice(db: Store, voice: str, name: str | None = None) -> str | None:
    """Give a character (the showing one, by default) a voice. Returns the
    character's name, or None if there's no such character or voice."""
    if voice not in VOICES:
        return None
    if name is None:
        active = db.active_character()
        if active is None:
            return None
        name = active.name
    return name if db.set_character_voice(name, voice) else None


def vrm_facts(path: Path) -> tuple[dict, bytes | None]:
    """What a .vrm says about itself — its name, authors, version — and its
    thumbnail picture (PNG or JPEG bytes; None if it has none). Reads only
    the file's header and the picture, not the model."""
    with open(path, "rb") as f:
        magic, _version, _length = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            raise ValueError(f"{Path(path).name} isn't a .vrm (glTF binary) file")
        json_length, json_type = struct.unpack("<I4s", f.read(8))
        if json_type != b"JSON":
            raise ValueError(f"{Path(path).name} has no glTF JSON chunk")
        gltf = json.loads(f.read(json_length))
        bin_start = 12 + 8 + json_length + 8  # after the JSON chunk and the BIN chunk's header
        ext = gltf.get("extensions") or {}
        meta = (ext.get("VRMC_vrm") or {}).get("meta") or (ext.get("VRM") or {}).get("meta") or {}
        facts = {
            "name": meta.get("name") or meta.get("title") or "",
            "authors": meta.get("authors") or ([meta["author"]] if meta.get("author") else []),
            "version": meta.get("version") or "",
        }
        # VRM 1.0 names the thumbnail's image; VRM 0.x its texture.
        image = meta.get("thumbnailImage")
        if image is None and meta.get("texture") is not None:
            try:
                image = gltf["textures"][meta["texture"]]["source"]
            except (KeyError, IndexError, TypeError):
                image = None
        try:
            view = gltf["bufferViews"][gltf["images"][image]["bufferView"]]
        except (KeyError, IndexError, TypeError):
            return facts, None
        f.seek(bin_start + int(view.get("byteOffset", 0)))
        return facts, f.read(int(view["byteLength"]))


def stem(path: Path) -> str:
    """A readable name for a model file: its name without the extension."""
    return Path(path).stem


def folders(cfg: Config) -> list[Path]:
    """Where characters live."""
    dirs = [cfg.data_dir / "characters", cfg.assets_dir / "characters"]
    if cfg.character_path.parent != Path():
        dirs.append(cfg.character_path.parent)
    return dirs
