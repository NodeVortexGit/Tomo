"""Long-term memory, in ChromaDB.

What lets the assistant remember preferences and facts across restarts, and
pull them back out before answering. Six collections:

  preferences  key → value the user set or the AI inferred ("name" → "Alex")
  memories     free-form facts worth keeping, each with an importance, and an
               embedding so ``recall`` finds them by meaning as well as words
  messages     every chat's lines, in order, each with its chat and the
               images sent
  chats        the chats: title (the first words), when started and last
               used, how many lines, and how far the memory has read them
  characters   the VRM models on offer, which one is showing, and each
               one's voice (male or female, as the model chose it)
  snapshots    cached OS scans (the app catalogue), with a fingerprint

Preferences and memories belong to no chat: what Tomo knows about the user
holds in every chat, new ones too.

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

from .events import Attachment, ChatInfo, ChatLine, Role, now_ms

log = logging.getLogger(__name__)

NGRAM_DIMENSIONS = 384
# Where ChromaDB's search stops counting a memory as related (cosine distance)…
RELATED = 0.85
# …and where a new one is the same as one kept already.
SAME_MEMORY = 0.2
PLACEHOLDER = [1.0]
# The old single transcript becomes chats: a new one after a pause this long.
CHAT_GAP_MS = 3 * 3600 * 1000
TITLE_CHARS = 60
BATCH = 500  # records a ChromaDB call changes at once


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
        self.chats = collection("chats")
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

    def knows(self, content: str) -> bool:
        """Whether a memory says (nearly) this already."""
        with self._lock:
            if self.memories.count() == 0:
                return False
            near = self.memories.query(query_embeddings=[self._embed(content)], n_results=1)
        documents, distances = near["documents"][0], near["distances"][0]
        return bool(documents) and (documents[0].strip().lower() == content.strip().lower()
                                    or distances[0] <= SAME_MEMORY)

    # ---- the chats ----------------------------------------------------------------

    def add_message(self, line: ChatLine, chat: str = "") -> None:
        """Keep a line, in ``chat`` (which this starts, if it's new)."""
        meta = {"seq": 0, "role": line.role.value, "at_ms": line.at_ms, "chat": chat}
        if line.attachments:
            # Metadata holds only plain values: the images go in as JSON.
            meta["attachments"] = json.dumps([asdict(a) for a in line.attachments], ensure_ascii=False)
        with self._lock:
            meta["seq"] = seq = self.messages.count()
            self.messages.add(ids=[f"msg-{seq:09d}"], documents=[line.text], embeddings=[PLACEHOLDER],
                              metadatas=[meta])
            if chat:
                got = self.chats.get(ids=[chat])
                old = got["metadatas"][0] if got["ids"] else {"started_at": line.at_ms, "count": 0, "remembered": -1}
                title = got["documents"][0] if got["ids"] else ""
                if not title and line.role == Role.USER:
                    title = chat_title(line)
                self.chats.upsert(ids=[chat], documents=[title], embeddings=[PLACEHOLDER], metadatas=[
                    {**old, "updated_at": line.at_ms, "count": int(old.get("count", 0)) + 1}])

    def recent_messages(self, limit: int, chat: str | None = None) -> list[ChatLine]:
        """The ``limit`` most recent lines of ``chat`` (None: of all of
        them), oldest first."""
        if limit <= 0:
            return []
        if chat is not None:
            return self.chat_lines(chat)[0][-limit:]
        with self._lock:
            total = self.messages.count()
            if total == 0:
                return []
            got = self.messages.get(where={"seq": {"$gte": max(0, total - limit)}})
        rows = sorted(zip(got["metadatas"], got["documents"]), key=lambda r: r[0]["seq"])
        return [_line(m, d) for m, d in rows]

    def chat_lines(self, chat: str, after: int = -1) -> tuple[list[ChatLine], int]:
        """A chat's lines, in order — those after line number ``after`` — and
        the last one's number (-1: none)."""
        with self._lock:
            got = self.messages.get(where={"chat": chat})
        rows = sorted(((m, d) for m, d in zip(got["metadatas"], got["documents"]) if int(m["seq"]) > after),
                      key=lambda r: r[0]["seq"])
        return [_line(m, d) for m, d in rows], (int(rows[-1][0]["seq"]) if rows else after)

    def list_chats(self) -> list[ChatInfo]:
        """Every chat, the most recently used first."""
        with self._lock:
            got = self.chats.get()
        chats = [ChatInfo(i, d or "", int(m.get("started_at", 0)), int(m.get("updated_at", 0)), int(m.get("count", 0)))
                 for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
        return sorted(chats, key=lambda c: (-c.updated_at, c.id))

    def remembered(self, chat: str) -> int:
        """How far the memory has read a chat: its last line number read (-1: none)."""
        with self._lock:
            got = self.chats.get(ids=[chat])
        return int(got["metadatas"][0].get("remembered", -1)) if got["ids"] else -1

    def set_remembered(self, chat: str, seq: int) -> None:
        with self._lock:
            got = self.chats.get(ids=[chat])
            if got["ids"]:
                self.chats.update(ids=[chat], metadatas=[{**got["metadatas"][0], "remembered": seq}])

    def search_messages(self, query: str, limit: int) -> list[tuple[ChatLine, str]]:
        """Lines from any chat that have the query's words in them (word forms
        too: "кучето" finds "куче"), the most words first, then the newest;
        each with its chat."""
        stems = {w[:5] if len(w) > 5 else w for w in re.findall(r"\w+", query.lower()) if len(w) >= 3}
        if not stems:
            return []
        with self._lock:
            got = self.messages.get()
        scored = []
        for meta, text in zip(got["metadatas"], got["documents"]):
            words = (text or "").lower()
            score = sum(stem in words for stem in stems)
            if score:
                scored.append((score, int(meta["seq"]), _line(meta, text), str(meta.get("chat") or "")))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        return [(line, chat) for _, _, line, chat in scored[:limit]]

    def split_into_chats(self, gap_ms: int = CHAT_GAP_MS) -> int:
        """Earlier versions kept one long transcript: give its lines chats —
        a new chat after each pause longer than ``gap_ms``. Once; returns how
        many chats it made."""
        if self.get_snapshot("chats-split") is not None:
            return 0
        with self._lock:
            got = self.messages.get()
        rows = sorted(((i, m, d) for i, m, d in zip(got["ids"], got["metadatas"], got["documents"])
                       if not m.get("chat")), key=lambda r: r[1]["seq"])
        ids, metas, chats = [], [], {}  # chat → [title, metadata]
        chat, last = "", None
        for i, meta, text in rows:
            at = int(meta.get("at_ms", 0))
            if last is None or at - last > gap_ms:
                chat = f"chat-{at}-{len(chats)}"
                chats[chat] = ["", {"started_at": at, "updated_at": at, "count": 0, "remembered": -1}]
            last = at
            ids.append(i)
            metas.append({**meta, "chat": chat})
            line, info = _line(meta, text), chats[chat]
            info[1]["updated_at"], info[1]["count"] = at, info[1]["count"] + 1
            if not info[0] and line.role == Role.USER:
                info[0] = chat_title(line)
        with self._lock:
            for start in range(0, len(ids), BATCH):
                self.messages.update(ids=ids[start:start + BATCH], metadatas=metas[start:start + BATCH])
            if chats:
                self.chats.upsert(ids=list(chats), documents=[c[0] for c in chats.values()],
                                  embeddings=[PLACEHOLDER] * len(chats), metadatas=[c[1] for c in chats.values()])
        self.set_snapshot("chats-split", str(len(chats)), str(len(rows)))
        if chats:
            log.info("the transcript's %d lines are %d chats now", len(rows), len(chats))
        return len(chats)

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


def _line(meta: dict, text: str | None) -> ChatLine:
    return ChatLine(Role(meta["role"]), text or "", int(meta["at_ms"]), _attachments(meta.get("attachments")))


def chat_title(line: ChatLine) -> str:
    """A chat's title: the first thing the user said, on one line."""
    text = " ".join(line.text.split())
    if not text:
        return "A picture" if line.attachments else ""
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1].rstrip() + "…"


def _attachments(stored) -> tuple[Attachment, ...]:
    """A message's images, back from their JSON (none if it's unreadable)."""
    if not stored:
        return ()
    try:
        return tuple(Attachment(str(a["path"]), str(a.get("source", "")), str(a.get("about", "")))
                     for a in json.loads(stored))
    except (ValueError, TypeError, KeyError):
        return ()
