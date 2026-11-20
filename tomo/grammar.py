"""Tomo's Bulgarian proofreader — a small helper beside the big model.

The model Tomo thinks with writes Bulgarian with mistakes: the full and short
article mixed up ("компютъра работи"), an adjective that doesn't agree with
its noun, Russian words, misspellings. Like the memory (db.py) hands the model
what it knows about the user, this hands it what's wrong with its reply, and
fixes what it's sure of on its own. Three parts, all offline and small:

  Lexicon   the Bulgarian Hunspell dictionary (bgoffice, ~1.6 MB) and Tomo's
            own words (tomo.dic): which words exist, every form of a word,
            the word a misspelling was meant to be.
  Tagger    a tiny network (~1M weights, numpy only) trained by
            scripts/train_grammar.py: for each word, keep it or change its
            ending ("-а" → "-ът"), from the sentence around it.
  Russian   the Russian words models slip into Bulgarian, with their
            Bulgarian ones.

check() finds the mistakes in a text; fix() applies the sure ones and returns
the rest as notes for the model (ai.py asks it once to rewrite with them).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# A word: letters, maybe joined by a hyphen ("по-добре", "Уест-Енд").
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
CYRILLIC = re.compile(r"[а-яА-ЯѝЍ]")

# What the tagger can say about a word besides an ending to change.
KEEP = "KEEP"
SPELL = "SPELL"  # not a word: the lexicon's nearest word replaces it

# Russian words that slip into a model's Bulgarian, and what Bulgarian says.
# Only words Bulgarian doesn't also have.
RUSSIAN = {
    "является": "е", "являются": "са", "сейчас": "сега", "здесь": "тук", "очень": "много",
    "хорошо": "добре", "конечно": "разбира се", "пожалуйста": "моля", "спасибо": "благодаря",
    "который": "който", "которая": "която", "которое": "което", "которые": "които",
    "тоже": "също", "также": "също така", "сегодня": "днес", "завтра": "утре", "вчера": "вчера",
    "всегда": "винаги", "никогда": "никога", "где": "къде", "куда": "накъде", "почему": "защо",
    "потому": "защото", "если": "ако", "чтобы": "за да", "что": "че", "нужно": "трябва",
    "можно": "може", "нельзя": "не може", "сколько": "колко", "только": "само", "уже": "вече",
    "ещё": "още", "еще": "още", "сделать": "направя", "открыть": "отворя", "закрыть": "затворя",
    "компьютер": "компютър", "программа": "програма", "время": "време", "сейчас-же": "веднага",
    "привет": "здравей", "пока": "докато", "вообще": "изобщо", "немного": "малко",
    "давай": "хайде", "вот": "ето", "здорово": "чудесно", "ничего": "нищо", "чего": "какво",
}

# From this sure, a mistake is fixed at once; from NOTE, the model is told.
SURE = 0.85
NOTE = 0.5
MIN_SPELLING = 4  # shorter unknown words aren't taken for misspellings
# Short words a space is often left out after ("незнам" is "не знам").
JOINED = ("не", "по", "на", "за", "от", "до", "в", "с", "да", "ще")


@dataclass(frozen=True)
class Issue:
    start: int  # where the word is in the text
    end: int
    wrong: str
    fix: str | None  # None: no better word known
    sure: float  # 0..1
    kind: str  # "spelling" | "russian" | "grammar"

    def note(self) -> str:
        if self.fix is None:
            return f"„{self.wrong}“ is not a Bulgarian word"
        return f"„{self.wrong}“ → „{self.fix}“ ({self.kind})"


# ---- the lexicon ---------------------------------------------------------------


class Lexicon:
    """A Hunspell dictionary (.aff + .dic files) with suffix rules only, as
    the Bulgarian one has: stems with flags, each flag a set of endings."""

    def __init__(self, aff: Path, *dics: Path) -> None:
        # flag → [(strip, add, condition)]; add → [(flag, strip, condition)]
        self.rules: dict[str, list[tuple[str, str, re.Pattern | None]]] = {}
        by_ending: dict[str, list[tuple[str, str, re.Pattern | None]]] = {}
        for line in aff.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0] != "SFX":
                continue
            flag, strip, add, cond = parts[1], parts[2], parts[3], parts[4]
            strip = "" if strip == "0" else strip
            add = "" if add == "0" else add.split("/")[0]
            pattern = None if cond == "." else re.compile(f"(?:{cond})$")
            self.rules.setdefault(flag, []).append((strip, add, pattern))
            by_ending.setdefault(add, []).append((flag, strip, pattern))
        self._by_ending = by_ending
        self._longest = max(map(len, by_ending), default=0)
        self.stems: dict[str, str] = {}  # stem → its flags
        for dic in dics:
            # The first line of a .dic is how many words it has.
            for line in dic.read_text(encoding="utf-8").splitlines()[1:]:
                stem, _, flags = line.strip().partition("/")
                if stem:
                    self.stems[stem] = self.stems.get(stem, "") + flags
        self._lock = threading.Lock()
        self._known: dict[str, bool] = {}

    def analyses(self, word: str) -> list[tuple[str, str]]:
        """The (stem, flag) pairs ``word`` comes from; flag "" for the stem
        itself."""
        found = []
        if word in self.stems:
            found.append((word, ""))
        # n = 0: endings that only take letters away ("открия" → "откри").
        for n in range(0, min(self._longest, len(word)) + 1):
            for flag, strip, pattern in self._by_ending.get(word[len(word) - n:], ()):
                stem = word[: len(word) - n] + strip
                flags = self.stems.get(stem)
                if flags is not None and flag in flags and (pattern is None or pattern.search(stem)):
                    found.append((stem, flag))
        return found

    def known(self, word: str) -> bool:
        """Is ``word`` Bulgarian? Capitals at the start of a sentence and
        all-caps words count as their lower case."""
        with self._lock:
            hit = self._known.get(word)
        if hit is None:
            hit = bool(self.analyses(word))
            if not hit and word[:1].isupper():
                hit = bool(self.analyses(word.lower())) or bool(self.analyses(word[0] + word[1:].lower()))
            if not hit and "-" in word:
                hit = all(self.known(part) for part in word.split("-") if part)
            with self._lock:
                if len(self._known) > 50_000:
                    self._known.clear()
                self._known[word] = hit
        return hit

    def paradigm(self, stem: str) -> set[str]:
        """Every form of ``stem``: the stem and its flags' endings."""
        forms = {stem}
        for flag in self.stems.get(stem, ""):
            for strip, add, pattern in self.rules.get(flag, ()):
                if pattern is None or pattern.search(stem):
                    if not strip or stem.endswith(strip):
                        forms.add(stem[: len(stem) - len(strip)] + add)
        return forms

    def forms(self, word: str) -> set[str]:
        """The other forms of the word(s) ``word`` can be ("стола" → "стол",
        "столът", "столове"…), ``word`` itself included."""
        out = {word}
        for stem, _ in self.analyses(word):
            out |= self.paradigm(stem)
        return out

    def suggest(self, word: str, limit: int = 3) -> list[str]:
        """The nearest known words to a misspelled ``word``: one letter
        changed, added, removed or swapped; then two."""
        lower = word.lower()
        letters = "абвгдежзийклмнопрстуфхцчшщъьюя"
        first = [w for w in _edits(lower, letters) if self.known(w)]
        ranked = sorted(set(first), key=lambda w: _rank(lower, w))
        if not ranked and len(lower) > 4:
            seen: set[str] = set()
            for e in _edits(lower, letters):
                for w in _edits(e, letters):
                    if w not in seen:
                        seen.add(w)
                        if self.known(w):
                            ranked.append(w)
                if len(ranked) >= limit * 4:
                    break
            ranked = sorted(set(ranked), key=lambda w: _rank(lower, w))
        if word[:1].isupper():
            ranked = [w[:1].upper() + w[1:] for w in ranked]
        return ranked[:limit]


def spelling_fix(word: str, lexicon: Lexicon) -> tuple[str | None, float]:
    """The word a misspelled ``word`` was meant to be, and how sure that is.
    One typical slip — letters that sound alike, a letter doubled or one of
    a double dropped, a lost "т" or "д", neighbours swapped, a missing space
    — is sure; one weaker or one of several is for the model to judge;
    anything else is likely a word the dictionary lacks, not a slip."""
    lower = word.lower()
    letters = "абвгдежзийклмнопрстуфхцчшщъьюя"
    near = sorted({w for w in _edits(lower, letters) if lexicon.known(w)}, key=lambda w: _rank(lower, w))
    for cut in range(1, 3):
        head, rest = lower[:cut], lower[cut:]
        if head in JOINED and len(rest) >= 3 and lexicon.known(rest):
            near.insert(0, f"{head} {rest}")
            break
    if not near:
        return None, 0.0
    strong = [w for w in near if _slip(lower, w) == 2]
    if len(strong) == 1:
        # A missing space against another fix ("впредвид": "в предвид"? "предвид"?)
        alone = " " not in strong[0] or len(near) == 1
        return _like(word, strong[0]), 0.9 if alone and len(near) <= 6 else 0.6
    weak = strong or [w for w in near if _slip(lower, w) == 1]
    if weak and len(weak) <= 3:
        return _like(word, weak[0]), 0.6
    return None, 0.0


def _slip(wrong: str, right: str) -> int:
    """How typical a slip turns ``right`` into ``wrong``: 2 typical, 1 a
    plain one-letter change, 0 more than that."""
    if " " in right:
        return 2 if right.replace(" ", "") == wrong else 0
    if len(wrong) == len(right):
        diff = [i for i, (a, b) in enumerate(zip(wrong, right)) if a != b]
        if len(diff) == 1:
            return 2 if frozenset((wrong[diff[0]], right[diff[0]])) in _NEAR else 1
        swapped = len(diff) == 2 and diff[1] == diff[0] + 1 and wrong[diff[0]] == right[diff[1]]
        return 2 if swapped else 0
    longer, shorter = (wrong, right) if len(wrong) > len(right) else (right, wrong)
    if len(longer) != len(shorter) + 1:
        return 0
    i = next((i for i, (a, b) in enumerate(zip(longer, shorter)) if a != b), len(shorter))
    if shorter[:i] + longer[i] + shorter[i:] != longer:
        return 0
    doubled = (i > 0 and longer[i] == longer[i - 1]) or (i + 1 < len(longer) and longer[i] == longer[i + 1])
    lost = len(right) > len(wrong) and longer[i] in "тдв"  # "помоща", "чуствам"
    spoken = wrong.endswith("ме") and right == wrong[:-1]  # "правиме" for "правим"
    return 2 if doubled or lost or spoken else 1


def _edits(word: str, letters: str) -> list[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return ([a + b[1:] for a, b in splits if b]
            + [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
            + [a + c + b[1:] for a, b in splits if b for c in letters if c != b[0]]
            + [a + c + b for a, b in splits for c in letters])


# Letters a Bulgarian writer (or a model) mixes up: cheaper to swap.
_NEAR = {frozenset(p) for p in ("ъа", "ея", "еи", "оу", "бп", "вф", "гк", "дт", "жш", "зс", "ьй", "юу", "яа")}


def _rank(wrong: str, right: str) -> tuple[int, int]:
    """Edits whose letters sound alike first, then shorter changes."""
    near = sum(1 for a, b in zip(wrong, right) if a != b and frozenset((a, b)) in _NEAR)
    differ = sum(1 for a, b in zip(wrong, right) if a != b) + abs(len(wrong) - len(right))
    return (differ - near, abs(len(wrong) - len(right)))


# ---- endings: what the tagger changes ---------------------------------------------


def ending_change(wrong: str, right: str) -> str | None:
    """The label that turns ``wrong`` into ``right`` by its ending:
    "ENDING:а>ът" (drop "а", add "ът"); None when more than the ending
    differs (more than 5 letters changed)."""
    if wrong == right:
        return KEEP
    common = 0
    while common < min(len(wrong), len(right)) and wrong[common] == right[common]:
        common += 1
    cut, add = wrong[common:], right[common:]
    if len(cut) > 5 or len(add) > 5 or common < 2:
        return None
    return f"ENDING:{cut}>{add}"


def apply_change(word: str, label: str) -> str | None:
    """``word`` with the ending change ``label`` made; None if it doesn't fit."""
    if not label.startswith("ENDING:"):
        return None
    cut, _, add = label[len("ENDING:"):].partition(">")
    if cut and not word.endswith(cut):
        return None
    return word[: len(word) - len(cut)] + add


# ---- the tagger -------------------------------------------------------------------

# The letters the tagger tells apart; anything else is one "other" letter.
ALPHABET = ("абвгдежзийклмнопрстуфхцчшщъьюяѝ" "abcdefghijklmnopqrstuvwxyz" "0123456789" ".,!?:;-–—\"'„“”()%")
PAD, OTHER, UPPER = 0, 1, 2  # a capital is its lower-case letter after an UPPER mark
CHAR_ID = {c: i + 3 for i, c in enumerate(ALPHABET)}
MAX_CHARS = 20  # the end of a word decides its form: longer words keep their last 20 letters


def tokens(text: str) -> list[tuple[int, int, str]]:
    """The text's words and punctuation marks, as (start, end, text)."""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\W\d_]+(?:-[^\W\d_]+)*|\d+|[^\w\s]", text)]


def char_ids(token: str) -> list[int]:
    ids = []
    for c in token:
        if c.isupper():
            ids.append(UPPER)
            c = c.lower()
        ids.append(CHAR_ID.get(c, OTHER))
    return ids[-MAX_CHARS:]


def features(toks: list[str], lexicon: Lexicon | None) -> tuple[np.ndarray, np.ndarray]:
    """The tagger's input: each token's letters (padded) and two flags —
    is it a known Bulgarian word, is it a word at all."""
    chars = np.zeros((len(toks), MAX_CHARS), dtype=np.int64)
    flags = np.zeros((len(toks), 2), dtype=np.float32)
    for i, t in enumerate(toks):
        ids = char_ids(t)
        chars[i, : len(ids)] = ids
        is_word = bool(WORD.fullmatch(t))
        flags[i, 0] = float(is_word and lexicon is not None and lexicon.known(t))
        flags[i, 1] = float(is_word)
    return chars, flags


class Tagger:
    """The trained network, run with numpy: letters → a vector per token
    (a convolution over its letters) → two bidirectional GRU layers over
    the sentence → a score per label per token. Like a classifier that
    may only answer from the options it's given, a word's probabilities are
    over its own choices: keep it, or one of its forms (:func:`choices`)."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.w = {k: data[k].astype(np.float32) for k in data.files
                      if k not in ("labels", "temperature", "sure", "note")}
            self.labels = [str(x) for x in data["labels"]]
            # What makes its confidence honest (scripts/calibrate_grammar.py).
            self.temperature = float(data["temperature"]) if "temperature" in data.files else 1.0
            # From where its changes are right nine times in ten (sure), half (note).
            self.sure = float(data["sure"]) if "sure" in data.files else SURE
            self.note = float(data["note"]) if "note" in data.files else NOTE
        self.label_id = {label: i for i, label in enumerate(self.labels)}

    def __call__(self, chars: np.ndarray, flags: np.ndarray) -> np.ndarray:
        """Each token's scores (logits) for ``labels``."""
        w = self.w
        x = w["embed"][chars]  # (T, C, E)
        kernel = w["conv.w"]  # (out, E, k)
        k = kernel.shape[2]
        padded = np.pad(x, ((0, 0), (k // 2, k // 2), (0, 0)))
        windows = np.stack([padded[:, i : i + x.shape[1]] for i in range(k)], axis=-1)  # (T, C, E, k)
        conv = np.einsum("tcek,oek->tco", windows, kernel) + w["conv.b"]
        conv = np.where(chars[..., None] > 0, conv, -1e4)
        word = np.maximum(conv.max(axis=1), 0)  # ReLU after max-pool
        h = np.concatenate([word, flags], axis=1)
        for layer in range(2):
            fwd = _gru(h, w[f"gru{layer}.f.wi"], w[f"gru{layer}.f.wh"], w[f"gru{layer}.f.bi"], w[f"gru{layer}.f.bh"])
            bwd = _gru(h[::-1], w[f"gru{layer}.b.wi"], w[f"gru{layer}.b.wh"], w[f"gru{layer}.b.bi"],
                       w[f"gru{layer}.b.bh"])[::-1]
            h = np.concatenate([fwd, bwd], axis=1)
        return (h @ w["out.w"].T + w["out.b"]) / self.temperature


def choices(word: str, lexicon: Lexicon | None, label_id: dict[str, int]) -> list[int]:
    """The labels open to ``word``: keep it; for a word the dictionary
    doesn't know, SPELL; for one it knows, the ending changes into its
    other forms ("компютъра": "компютър", "компютърът", "компютри"…)."""
    allowed = [label_id[KEEP]]
    if lexicon is None or not WORD.fullmatch(word):
        return allowed
    if not lexicon.known(word):
        return allowed + ([label_id[SPELL]] if SPELL in label_id else [])
    lower = word.lower()
    for form in lexicon.forms(lower):
        label = ending_change(lower, form)
        if label is not None and label != KEEP and label in label_id:
            allowed.append(label_id[label])
    return sorted(set(allowed))


def softmax_over(logits: np.ndarray, allowed: list[int]) -> np.ndarray:
    """Probabilities over ``allowed`` only (in that order)."""
    z = logits[allowed] - logits[allowed].max()
    e = np.exp(z)
    return e / e.sum()


def _gru(x: np.ndarray, wi: np.ndarray, wh: np.ndarray, bi: np.ndarray, bh: np.ndarray) -> np.ndarray:
    """PyTorch's GRU, one direction: gates in the order reset, update, new."""
    n = wh.shape[1]
    h = np.zeros(n, dtype=np.float32)
    out = np.empty((len(x), n), dtype=np.float32)
    gi = x @ wi.T + bi
    for t in range(len(x)):
        gh = h @ wh.T + bh
        r = _sigmoid(gi[t, :n] + gh[:n])
        z = _sigmoid(gi[t, n : 2 * n] + gh[n : 2 * n])
        cand = np.tanh(gi[t, 2 * n :] + r * gh[2 * n :])
        h = (1 - z) * cand + z * h
        out[t] = h
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def dictionaries(folder: Path) -> list[Path]:
    """The Bulgarian dictionary, then Tomo's own words (tomo.dic: newer
    words, and common ones bg_BG lacks — "също", "три", "интернет")."""
    return [p for p in (folder / "bg_BG.dic", folder / "tomo.dic") if p.is_file() or p.name == "bg_BG.dic"]


# ---- the proofreader ------------------------------------------------------------------


class Proofreader:
    """Finds the mistakes in Bulgarian text. Loads its files on first use."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._lexicon: Lexicon | None = None
        self._tagger: Tagger | None = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                self._lexicon = Lexicon(self.folder / "bg_BG.aff", *dictionaries(self.folder))
            except OSError as e:
                log.info("no Bulgarian dictionary (%s): spelling isn't checked", e)
            try:
                self._tagger = Tagger(self.folder / "tagger.npz")
            except (OSError, KeyError, ValueError) as e:
                log.info("no grammar tagger (%s): only spelling is checked", e)

    @property
    def lexicon(self) -> Lexicon | None:
        self._load()
        return self._lexicon

    def check(self, text: str) -> list[Issue]:
        """The mistakes in ``text``, in order."""
        self._load()
        lexicon = self._lexicon
        issues: list[Issue] = []
        for start, end in sentences(text):
            toks = tokens(text[start:end])
            if not toks:
                continue
            logits = None
            if self._tagger is not None:
                chars, flags = features([t for _, _, t in toks], lexicon)
                logits = self._tagger(chars, flags)
            for i, (s, e, word) in enumerate(toks):
                if not CYRILLIC.search(word):
                    continue
                issue = self._word(word, start + s, start + e, lexicon, logits[i] if logits is not None else None,
                                   first=i == 0)
                if issue is not None:
                    issues.append(issue)
        return issues

    def _word(self, word: str, start: int, end: int, lexicon: Lexicon | None, logits: np.ndarray | None,
              first: bool) -> Issue | None:
        russian = RUSSIAN.get(word.lower())
        if russian is not None and (lexicon is None or not lexicon.known(word)):
            return Issue(start, end, word, _like(word, russian), 0.95, "russian")
        if lexicon is not None and not lexicon.known(word):
            # A capital mid-sentence is a name the dictionary needn't know;
            # a short word is too easily something else.
            if (word[:1].isupper() and not first) or len(word) < MIN_SPELLING:
                return None
            fix, sure = spelling_fix(word, lexicon)
            return Issue(start, end, word, fix, sure, "spelling") if fix and sure >= NOTE else None
        if logits is None or self._tagger is None:
            return None
        allowed = choices(word, lexicon, self._tagger.label_id)
        if len(allowed) < 2:
            return None
        probs = softmax_over(logits, allowed)
        best = int(np.argmax(probs))
        label = self._tagger.labels[allowed[best]]
        if label in (KEEP, SPELL) or probs[best] < self._tagger.note:
            return None
        fixed = apply_change(word, label)
        if fixed is None or fixed == word:
            return None
        # On the proofreader's one scale: at the tagger's own "sure", SURE.
        sure = SURE + (float(probs[best]) - self._tagger.sure) * (1 - SURE) / max(1 - self._tagger.sure, 1e-6)
        if probs[best] < self._tagger.sure:
            sure = NOTE + (float(probs[best]) - self._tagger.note) * (SURE - NOTE) / max(
                self._tagger.sure - self._tagger.note, 1e-6)
        return Issue(start, end, word, fixed, min(max(sure, NOTE), 1.0), "grammar")

    def fix(self, text: str) -> tuple[str, list[Issue]]:
        """``text`` with the sure mistakes fixed, and the unsure ones left
        for the model."""
        issues = self.check(text)
        out, at, unsure = [], 0, []
        for issue in issues:
            if issue.fix is not None and issue.sure >= SURE:
                out.append(text[at : issue.start] + issue.fix)
                at = issue.end
            else:
                unsure.append(issue)
        out.append(text[at:])
        return "".join(out), unsure


def _like(original: str, word: str) -> str:
    """``word`` with ``original``'s capital."""
    return word[:1].upper() + word[1:] if original[:1].isupper() else word


def sentences(text: str) -> list[tuple[int, int]]:
    """Where each sentence of ``text`` starts and ends."""
    spans, start = [], 0
    for m in re.finditer(r"[.!?…]+[\"“”»)]*\s+|\n+", text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]
