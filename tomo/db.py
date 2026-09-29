"""Long-term memory, in ChromaDB.

What lets the assistant remember preferences and facts across restarts, and
pull them back out before answering. Five collections:

  preferences  key → value the user set or the AI inferred ("name" → "Alex")
  memories     free-form facts worth keeping, each with an importance, and an
               embedding so ``recall`` finds them by meaning as well as words
  messages     the rolling chat transcript, in order, with the images sent
  characters   the VRM models on offer, which one is showing, and each
               one's voice (male or female, as the model chose it)
  snapshots    cached OS scans (the app catalogue), with a fingerprint

Memories are embedded with a small built-in character-n-gram embedding:
offline, instant, and it works for Bulgarian as well as English (word forms
share their n-grams). ``TOMO_EMBEDDINGS=minilm`` uses ChromaDB's stock
English model instead (a one-time ~80 MB download). The other collections
are looked up by id or by fields, never by meaning, and get a constant
placeholder vector.

The Rust version kept all this in SQLite (``memory.sqlite3``): the first
start of this version moves it over, once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .events import Attachment, ChatLine, Role, now_ms

log = logging.getLogger(__name__)

NGRAM_DIMENSIONS = 384
# Where ChromaDB's search stops counting a memory as related (cosine distance).
RELATED = 0.85
PLACEHOLDER = [1.0]


@dataclass
class Memory:
    id: str
    kind: str
    content: str
    importance: int
    created_at: int


@dataclass
class Character:
    name: str
    path: str
    is_active: bool
    voice: str = ""  # "male", "female", or "" before one is chosen


def ngram_embedding(text: str, dimensions: int = NGRAM_DIMENSIONS) -> list[float]:
    """A language-independent embedding: the text's character 3- and
    4-grams (within words, with the word boundaries), hashed into
    ``dimensions`` signed buckets, then normalised. Texts that share words or
    word stems come out close."""
    vector = [0.0] * dimensions
    for word in re.findall(r"\w+", text.lower()):
        padded = f" {word} "
        grams = [padded[i : i + n] for n in (3, 4) for i in range(max(1, len(padded) - n + 1))]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dimensions
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector
    return [v / norm for v in vector]


class Store:
    """The memory. Thread-safe: every call takes a lock (they're quick)."""

    def __init__(self, path: Path | None, embeddings: str = "ngram") -> None:
        import chromadb
        from chromadb.config import Settings

        # Nothing leaves the machine: no telemetry.
        settings = Settings(anonymized_telemetry=False)
        if path is None:
            self._client = chromadb.EphemeralClient(settings=settings)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path), settings=settings)
        self._lock = threading.RLock()

        def collection(name: str, **meta: str):
            return self._client.get_or_create_collection(
                name, embedding_function=None, metadata={"hnsw:space": "cosine", **meta}
            )

        self.preferences = collection("preferences")
        self.messages = collection("messages")
        self.characters = collection("characters")
        self.snapshots = collection("snapshots")
        self.memories = collection("memories", embedding=embeddings)
        # A collection keeps the embedding it was made with.
        self._embedding = (self.memories.metadata or {}).get("embedding", "ngram")
        if self._embedding != embeddings:
            log.info("memories keep their %s embedding (asked for %s)", self._embedding, embeddings)
        self._minilm = None

    def _embed(self, text: str) -> list[float]:
        if self._embedding == "minilm":
            if self._minilm is None:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

                self._minilm = DefaultEmbeddingFunction()
            return [float(x) for x in self._minilm([text])[0]]
        return ngram_embedding(text)

    # ---- preferences -----------------------------------------------------------

    def set_pref(self, key: str, value: str) -> None:
        with self._lock:
            self.preferences.upsert(ids=[key], documents=[value], embeddings=[PLACEHOLDER],
                                    metadatas=[{"updated_at": now_ms()}])

    def get_pref(self, key: str) -> str | None:
        with self._lock:
            got = self.preferences.get(ids=[key])
        return got["documents"][0] if got["ids"] else None

    def all_prefs(self) -> list[tuple[str, str]]:
        with self._lock:
            got = self.preferences.get()
        return sorted(zip(got["ids"], got["documents"]))

    # ---- memories --------------------------------------------------------------

    def add_memory(self, kind: str, content: str, importance: int) -> str:
        created = now_ms()
        memory_id = f"mem-{created}-{hashlib.blake2b(content.encode(), digest_size=4).hexdigest()}"
        with self._lock:
            self.memories.add(ids=[memory_id], documents=[content], embeddings=[self._embed(content)],
                              metadatas=[{"kind": kind or "fact", "importance": int(importance), "created_at": created}])
        return memory_id

    def _all_memories(self) -> list[Memory]:
        with self._lock:
            got = self.memories.get()
        return [
            Memory(i, m.get("kind", "fact"), d, int(m.get("importance", 1)), int(m.get("created_at", 0)))
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
        ]

    def top_memories(self, limit: int) -> list[Memory]:
        """The most important, then most recent, memories — the default context."""
        memories = self._all_memories()
        memories.sort(key=lambda m: (-m.importance, -m.created_at))
        return memories[:limit]

    def search_memories(self, query: str, limit: int) -> list[Memory]:
        """Memories that contain ``query`` or are close to it in meaning."""
        with self._lock:
            count = self.memories.count()
            if count == 0:
                return []
            words = self.memories.get(where_document={"$contains": query}) if query.strip() else {"ids": []}
            near = self.memories.query(query_embeddings=[self._embed(query)], n_results=min(limit, count))
        found: dict[str, Memory] = {}
        for i, d, m in zip(words["ids"], words.get("documents") or [], words.get("metadatas") or []):
            found[i] = Memory(i, m.get("kind", "fact"), d, int(m.get("importance", 1)), int(m.get("created_at", 0)))
        for i, d, m, distance in zip(near["ids"][0], near["documents"][0], near["metadatas"][0], near["distances"][0]):
            if distance <= RELATED:
                found.setdefault(i, Memory(i, m.get("kind", "fact"), d, int(m.get("importance", 1)), int(m.get("created_at", 0))))
        ranked = sorted(found.values(), key=lambda m: (-m.importance, -m.created_at))
        return ranked[:limit]

    # ---- the transcript ---------------------------------------------------------

    def add_message(self, line: ChatLine) -> None:
        meta = {"seq": 0, "role": line.role.value, "at_ms": line.at_ms}
        if line.attachments:
            # Metadata holds only plain values: the images go in as JSON.
            meta["attachments"] = json.dumps([asdict(a) for a in line.attachments], ensure_ascii=False)
        with self._lock:
            meta["seq"] = seq = self.messages.count()
            self.messages.add(ids=[f"msg-{seq:09d}"], documents=[line.text], embeddings=[PLACEHOLDER],
                              metadatas=[meta])

    def recent_messages(self, limit: int) -> list[ChatLine]:
        """The ``limit`` most recent lines, oldest first."""
        with self._lock:
            total = self.messages.count()
            if total == 0 or limit <= 0:
                return []
            got = self.messages.get(where={"seq": {"$gte": max(0, total - limit)}})
        rows = sorted(zip(got["metadatas"], got["documents"]), key=lambda r: r[0]["seq"])
        return [ChatLine(Role(m["role"]), d or "", int(m["at_ms"]), _attachments(m.get("attachments")))
                for m, d in rows]

    # ---- characters -------------------------------------------------------------

    def add_character(self, name: str, path: str) -> None:
        """Register (or move) a character; it isn't made active here. A known
        one keeps whether it's active, and its voice."""
        with self._lock:
            got = self.characters.get(ids=[name])
            old = got["metadatas"][0] if got["ids"] else {}
            meta = {"added_at": now_ms(), "is_active": bool(old.get("is_active"))}
            if old.get("voice"):
                meta["voice"] = old["voice"]
            self.characters.upsert(ids=[name], documents=[path], embeddings=[PLACEHOLDER], metadatas=[meta])

    def set_active_character(self, name: str) -> None:
        with self._lock:
            got = self.characters.get()
            for i, path, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                self.characters.update(ids=[i], metadatas=[{**meta, "is_active": i == name}])

    def set_character_voice(self, name: str, voice: str) -> bool:
        """Remember a character's voice ("male" or "female"). False if there's
        no character by that name."""
        with self._lock:
            got = self.characters.get(ids=[name])
            if not got["ids"]:
                return False
            self.characters.update(ids=[name], metadatas=[{**got["metadatas"][0], "voice": voice}])
            return True

    def active_character(self) -> Character | None:
        with self._lock:
            got = self.characters.get(where={"is_active": True})
        if not got["ids"]:
            return None
        return Character(got["ids"][0], got["documents"][0], True, str(got["metadatas"][0].get("voice") or ""))

    def list_characters(self) -> list[Character]:
        with self._lock:
            got = self.characters.get()
        return sorted(
            (Character(i, d, bool(m.get("is_active")), str(m.get("voice") or ""))
             for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])),
            key=lambda c: c.name.lower(),
        )

    # ---- snapshots --------------------------------------------------------------

    def set_snapshot(self, key: str, fingerprint: str, text: str) -> None:
        with self._lock:
            self.snapshots.upsert(ids=[key], documents=[text], embeddings=[PLACEHOLDER],
                                  metadatas=[{"fingerprint": fingerprint, "updated_at": now_ms()}])

    def get_snapshot(self, key: str) -> tuple[str, str] | None:
        with self._lock:
            got = self.snapshots.get(ids=[key])
        if not got["ids"]:
            return None
        return got["metadatas"][0].get("fingerprint", ""), got["documents"][0]

    # ---- moving the Rust version's SQLite memory over ---------------------------------

    def migrate_from_sqlite(self, path: Path) -> bool:
        """Copy the old ``memory.sqlite3`` in, once (a marker remembers it).
        The old file is left where it is. True if something was copied."""
        if not path.is_file() or self.get_snapshot("migrated-from-sqlite") is not None:
            return False
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error as e:
            log.warning("couldn't open the old memory %s: %s", path, e)
            return False
        copied = {"preferences": 0, "memories": 0, "messages": 0, "characters": 0}
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "preferences" in tables:
                for key, value in con.execute("SELECT key, value FROM preferences"):
                    self.set_pref(key, value)
                    copied["preferences"] += 1
            if "memories" in tables:
                for kind, content, importance, created in con.execute(
                        "SELECT kind, content, importance, created_at FROM memories ORDER BY id"):
                    with self._lock:
                        self.memories.add(ids=[f"mem-{created}-{copied['memories']}"], documents=[content],
                                          embeddings=[self._embed(content)],
                                          metadatas=[{"kind": kind, "importance": int(importance), "created_at": int(created)}])
                    copied["memories"] += 1
            if "messages" in tables:
                for role, content, at_ms in con.execute("SELECT role, content, at_ms FROM messages ORDER BY at_ms, id"):
                    self.add_message(ChatLine(_role(role), content, int(at_ms)))
                    copied["messages"] += 1
            if "characters" in tables:
                active = None
                for name, char_path, is_active in con.execute("SELECT name, path, is_active FROM characters"):
                    # The Rust version stored Windows' \\?\ long-path prefix.
                    self.add_character(name, char_path.removeprefix("\\\\?\\"))
                    copied["characters"] += 1
                    if is_active:
                        active = name
                if active:
                    self.set_active_character(active)
        except sqlite3.Error as e:
            log.warning("couldn't read all of the old memory %s: %s", path, e)
        finally:
            con.close()
        self.set_snapshot("migrated-from-sqlite", str(path), str(copied))
        log.info("moved the old SQLite memory into ChromaDB: %s", copied)
        return any(copied.values())


def _role(text: str) -> Role:
    try:
        return Role(text.lower())
    except ValueError:
        return Role.SYSTEM


def _attachments(stored) -> tuple[Attachment, ...]:
    """A message's images, back from their JSON (none if it's unreadable)."""
    if not stored:
        return ()
    try:
        return tuple(Attachment(str(a["path"]), str(a.get("source", "")), str(a.get("about", "")))
                     for a in json.loads(stored))
    except (ValueError, TypeError, KeyError):
        return ()
