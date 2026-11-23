"""Proofreading Tomo's Bulgarian, a piece at a time, before it's shown or
spoken: Tomo uses what comes back as it is.

Three steps, each only when the one before isn't enough:

1. The rules (assets/grammar/bg_rules.json): fixes that are always right —
   „с“ before с/з is „със“, „в“ before в/ф is „във“, a Russian word with one
   Bulgarian match is replaced — and what's never Bulgarian: ы, э, ё,
   another alphabet, or one word mixing alphabets.
2. The dictionary (Hunspell bg_BG, LibreOffice's; scripts/setup_models.py
   downloads it): a Cyrillic word it doesn't know is suspicious. A piece with
   nothing suspicious is done — in about a millisecond.
3. A small Bulgarian model, BgGPT-Gemma-3-4B (INSAIT), on the CPU — the
   graphics card is the brain's, and Ollama would unload the brain to make
   room — rewrites a suspicious piece, told which words looked wrong. Its
   answer is taken unless it plainly isn't the piece corrected: an
   explanation, much longer or shorter, different numbers or a dropped name.

Without the dictionary or the model, the steps before still apply.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from .language import Language

log = logging.getLogger(__name__)

DICTIONARY = "bg_BG"  # <data>/models/hunspell/bg_BG.aff and .dic
MODEL_TIMEOUT = 20.0  # a piece the model hasn't fixed by then goes as the rules left it
KEEP_ALIVE = "1h"
WORD = re.compile(r"[^\W\d_]+")  # a run of letters, in any alphabet


@dataclass
class Rules:
    fixes: list[tuple[re.Pattern, str]] = field(default_factory=list)
    words: dict[str, str] = field(default_factory=dict)  # lower case → its replacement
    allowed: set[str] = field(default_factory=set)  # lower case
    foreign_letters: str = ""
    prompt: str = ""
    hint: str = "{words}"


def load_rules(path: Path) -> Rules:
    """The rules file; none (or a broken one) is no rules."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("no grammar rules (%s): %s", path, e)
        return Rules()
    return Rules(
        fixes=[(re.compile(f["pattern"]), f["replace"]) for f in data.get("fixes", [])],
        words={k.lower(): v for k, v in data.get("words", {}).items() if k != "about"},
        allowed={w.lower() for w in data.get("allowed", {}).get("words", [])},
        foreign_letters=data.get("foreign_letters", ""),
        prompt=data.get("prompt", ""),
        hint=data.get("hint", "{words}"),
    )


def alphabets(word: str) -> set[str]:
    """The alphabets a word's letters are from: "CYRILLIC", "LATIN", …"""
    return {unicodedata.name(c, "?").split()[0] for c in word}


class Grammar:
    """``server``: the brain's (AiClient.server); the model runs on the same
    Ollama. ``lookup``: whether the dictionary knows a word (for tests;
    otherwise the Hunspell dictionary in ``cfg.data_dir``)."""

    def __init__(self, cfg, server: Callable[[], Awaitable[object]] | None = None,
                 http: httpx.AsyncClient | None = None, lookup: Callable[[str], bool] | None = None,
                 rules: Path | None = None) -> None:
        self.enabled = cfg.grammar
        self.model = cfg.grammar_model
        self.rules = load_rules(rules or cfg.assets_dir / "grammar" / "bg_rules.json")
        self.server = server
        self.http = http or httpx.AsyncClient(timeout=MODEL_TIMEOUT)
        self.folder = cfg.data_dir / "models" / "hunspell"
        self._lookup = lookup
        self._dictionary_loaded = lookup is not None
        self._no_model = not self.model

    # ---- the steps ---------------------------------------------------------------------

    def apply_rules(self, text: str) -> str:
        for pattern, replace in self.rules.fixes:
            text = pattern.sub(replace, text)

        def russian(match: re.Match) -> str:
            word = match.group()
            better = self.rules.words.get(word.lower())
            if better is None:
                return word
            return better[:1].upper() + better[1:] if word[:1].isupper() else better

        return WORD.sub(russian, text)

    def suspicious(self, text: str) -> list[str]:
        """The words that may be wrong, each once."""
        found: list[str] = []
        for word in WORD.findall(text):
            if word in found or word.lower() in self.rules.allowed:
                continue
            kinds = alphabets(word)
            if kinds == {"LATIN"}:
                continue  # a name or an English word: left as written
            if (kinds != {"CYRILLIC"} or any(c in self.rules.foreign_letters for c in word)
                    or not self.known(word)):
                found.append(word)
        return found

    def known(self, word: str) -> bool:
        lookup = self._dictionary()
        if lookup is None:
            return True  # no dictionary: only what the rules see counts
        return lookup(word) or lookup(word.lower()) or lookup(word[:1].upper() + word[1:].lower())

    def _dictionary(self) -> Callable[[str], bool] | None:
        if not self._dictionary_loaded:
            self._dictionary_loaded = True
            try:
                from spylls.hunspell import Dictionary

                self._lookup = Dictionary.from_files(str(self.folder / DICTIONARY)).lookup
            except Exception as e:  # noqa: BLE001 - missing, or unreadable: the rules still work
                log.warning("no Bulgarian dictionary (%s): %s", self.folder, e)
        return self._lookup

    # ---- proofreading ------------------------------------------------------------------

    async def fix(self, piece: str) -> str:
        """``piece`` as it should be (unchanged if it isn't Bulgarian, or is right)."""
        if not self.enabled or Language.of(piece) != Language.BULGARIAN:
            return piece
        lead, core, trail = re.match(r"(\s*)(.*?)(\s*)$", piece, re.S).groups()
        try:
            core = self.apply_rules(core)
            words = self.suspicious(core)
            if words:
                fixed = await self.ask(core, words)
                if fixed is not None:
                    core = self.apply_rules(same_shape(core, fixed))
                elif foreign := [w for w in words if alphabets(w) - {"CYRILLIC", "LATIN"}]:
                    core = drop_foreign(core, foreign)  # the least that can be done without the model
        except Exception as e:  # noqa: BLE001 - a proofreader must never lose the reply
            log.warning("couldn't proofread %r: %s", piece[:60], e)
        return lead + core + trail

    async def ask(self, text: str, words: list[str]) -> str | None:
        """The model's correction of ``text``, or None (no model, no answer,
        or an answer that isn't the text corrected)."""
        if self._no_model or self.server is None:
            return None
        server = await self.server()
        if getattr(getattr(server, "api", None), "value", "") != "ollama":
            return None  # LM Studio and the like: the rules and the dictionary only
        body = {"model": self.model, "stream": False, "keep_alive": KEEP_ALIVE,
                "options": {"temperature": 0, "num_ctx": 1024, "num_gpu": 0, "num_predict": len(text) // 2 + 60},
                "messages": [{"role": "system", "content": self.rules.prompt},
                             {"role": "user", "content": self.rules.hint.format(words=", ".join(words)) + "\n"
                                                         + f"Текст: {text}"}]}
        try:
            response = await self.http.post(f"{server.url}/api/chat", json=body, timeout=MODEL_TIMEOUT)
        except httpx.HTTPError as e:
            log.info("the proofreading model didn't answer: %s", e)
            return None
        if response.status_code == 404:
            self._no_model = True
            log.warning("no proofreading model: ollama pull %s", self.model)
            return None
        if not response.is_success:
            return None
        answer = (response.json().get("message") or {}).get("content") or ""
        return accepted(text, answer)

    async def warm_up(self) -> None:
        """Read the dictionary, and have Ollama load the model (on the CPU)
        now, while nobody waits."""
        if not self.enabled:
            return
        await asyncio.to_thread(self._dictionary)
        if self._no_model or self.server is None:
            return
        try:
            server = await self.server()
            if getattr(getattr(server, "api", None), "value", "") != "ollama":
                return
            response = await self.http.post(f"{server.url}/api/chat", timeout=120.0, json={
                "model": self.model, "stream": False, "keep_alive": KEEP_ALIVE,
                "options": {"num_ctx": 1024, "num_gpu": 0, "num_predict": 1},
                "messages": [{"role": "user", "content": "Здравей"}]})
            if response.status_code == 404:
                self._no_model = True
                log.warning("no proofreading model: ollama pull %s", self.model)
        except Exception as e:  # noqa: BLE001
            log.info("the proofreading model isn't ready yet: %s", e)


def accepted(original: str, answer: str) -> str | None:
    """The model's answer as the corrected text — or None when it plainly
    isn't one: empty, an explanation, much longer or shorter, or with
    numbers or names (Latin words) the original didn't have."""
    text = answer.strip()
    text = re.sub(r"^(?:поправен(?:ият)? текст|текст)\s*:\s*", "", text, flags=re.I)
    text = text.replace("**", "").strip().strip("„“\"'«»").strip()
    if not text or "\n" in text and "\n" not in original:
        return None
    if abs(len(text) - len(original)) > max(12, len(original) * 0.3):
        return None
    if re.findall(r"\d+", text) != re.findall(r"\d+", original):
        return None
    latin = lambda s: {w for w in WORD.findall(s) if alphabets(w) == {"LATIN"}}  # noqa: E731
    if not latin(original) <= latin(text):
        return None
    if difflib.SequenceMatcher(None, original, text).ratio() < 0.6:
        return None
    return text


def same_shape(original: str, fixed: str) -> str:
    """The model's fix shaped like the piece it fixed: a piece cut from the
    middle of a sentence (at a comma) starts and ends as it did — the model
    likes to begin with a capital and end with a full stop."""
    if original[:1].isalpha() and fixed[:1].isalpha() and original[:1].islower() != fixed[:1].islower():
        fixed = (fixed[0].lower() if original[0].islower() else fixed[0].upper()) + fixed[1:]
    ending = re.search(r"[.!?…,:;]*$", original).group()
    fixed_ending = re.search(r"[.!?…,:;]*$", fixed).group()
    if ending != fixed_ending:
        fixed = fixed[:len(fixed) - len(fixed_ending)] + ending
    return fixed


def drop_foreign(text: str, words: list[str]) -> str:
    """Take out letters from alphabets other than Cyrillic and Latin (Hangul,
    Devanagari… that a model slipped in) from ``words``."""
    for word in words:
        kept = "".join(c for c in word if alphabets(c) <= {"CYRILLIC", "LATIN"})
        text = text.replace(word, kept)
    return re.sub(r" {2,}", " ", text)
