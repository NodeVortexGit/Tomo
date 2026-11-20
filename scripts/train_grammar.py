#!/usr/bin/env python3
"""Train the grammar tagger Tomo proofreads its Bulgarian with (tomo/grammar.py).

The tagger reads a sentence and says, for each word, "keep it" or "change its
ending" ("-а" → "-ът") — choosing only among the word's own forms in the
dictionary, in training as when it runs (grammar.choices). It learns from
correct Bulgarian that this script spoils on purpose, the way a model
writing Bulgarian does:

  - another form of the same word: the short article for the full one
    ("компютъра работи"), an adjective that doesn't agree ("хубав книга"),
    the wrong person or tense;
  - a misspelled word (marked SPELL: the dictionary fixes those).

Correct Bulgarian: Bulgarian Wikipedia (wikimedia/wikipedia, 20231101.bg, one
of its parquet files), Bulgarian film subtitles for how people talk (OPUS
OpenSubtitles 2018, the start of mono/bg.txt.gz; only lines whose every word
the dictionary knows), and the correct sides of thebogko's Bulgarian spelling
and grammar mistakes (Apache-2.0); their wrong sides are the test.

  pip install torch pyarrow
  python scripts/train_grammar.py --data DIR --out assets/grammar/tagger.npz

DIR holds a Wikipedia .parquet, subs.txt (subtitles, a line each),
spell_errors.csv and grammar_errors.csv; the dictionary comes from
assets/grammar. Takes about an hour on four CPU cores.
"""

from __future__ import annotations

import argparse
import collections
import csv
import multiprocessing
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tomo import grammar  # noqa: E402
from tomo.grammar import KEEP, SPELL, WORD, Lexicon  # noqa: E402

IGNORE = -100
MAX_TOKENS = 48
MIN_LABEL_COUNT = 30
# How sentences are spoiled.
CLEAN = 0.3  # left as they are
SPELLING = 0.15  # of the spoiled words: misspelled
LETTERS = "абвгдежзийклмнопрстуфхцчшщъьюя"
# Full ↔ short article: the mistake models make most — planted first in
# this share of the spoiled sentences that have one.
ARTICLE = [("ът", "а"), ("ят", "я"), ("а", "ът"), ("я", "ят")]
ARTICLE_FIRST = 0.35

_lexicon: Lexicon | None = None
ASSETS = Path(__file__).resolve().parent.parent / "assets" / "grammar"
# A preposition before a full article: a common mistake in subtitles.
FULL_AFTER_PREPOSITION = re.compile(r"\b(?:на|в|във|с|със|за|от|до|по|при|към|без|през|над|под)\s+\w+(?:ът|ят)\b", re.I)


def init_worker(folder: str) -> None:
    global _lexicon
    _lexicon = Lexicon(Path(folder) / "bg_BG.aff", *grammar.dictionaries(Path(folder)))


# ---- correct sentences -------------------------------------------------------------


def wiki_sentences(parquet: Path, limit: int, seed: int) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["text"])
    texts = table.column("text").to_pylist()
    random.Random(seed).shuffle(texts)
    out: list[str] = []
    for text in texts:
        for paragraph in text.split("\n"):
            for s in re.split(r"(?<=[.!?])\s+(?=[А-Я„])", paragraph.strip()):
                if good_sentence(s):
                    out.append(s)
        if len(out) >= limit:
            break
    return out[:limit]


def subtitle_sentences(path: Path, limit: int, seed: int) -> list[str]:
    """Spoken Bulgarian: subtitle lines that are whole sentences, every
    word of them known to the dictionary."""
    out, seen = [], set()
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().lstrip("-–— ").strip()
            if s in seen or not good_sentence(s, shortest=12) or FULL_AFTER_PREPOSITION.search(s):
                continue
            words = [t for _, _, t in grammar.tokens(s) if WORD.fullmatch(t)]
            if len(words) < 3 or not all(_lexicon.known(w) for w in words):
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= limit * 2:  # enough to choose from
                break
    random.Random(seed).shuffle(out)
    return out[:limit]


def good_sentence(s: str, shortest: int = 20) -> bool:
    if not (shortest <= len(s) <= 300) or s[-1] not in ".!?" or not s[0].isupper():
        return False
    letters = [c for c in s if c.isalpha()]
    cyr = sum(1 for c in letters if "а" <= c.lower() <= "я")
    return len(letters) > 10 and cyr / len(letters) > 0.9 and "(" not in s and len(grammar.tokens(s)) <= MAX_TOKENS


# ---- spoiling them --------------------------------------------------------------------


def spoil(sentence: str, rng: random.Random) -> tuple[list[str], list[str]]:
    """The sentence's tokens with some words spoiled, and each token's label."""
    toks = [t for _, _, t in grammar.tokens(sentence)]
    labels = [KEEP] * len(toks)
    if rng.random() < CLEAN:
        return toks, labels
    # Lower-case words, and the capitalised first one (often the subject).
    words = [i for i, t in enumerate(toks) if WORD.fullmatch(t) and len(t) > 1
             and (t.islower() or (i == 0 and t[1:].islower()))]
    rng.shuffle(words)
    if rng.random() < ARTICLE_FIRST:
        swappable = [i for i in words if article_swap(toks[i].lower()) is not None]
        if swappable:
            i = swappable[0]
            capital = toks[i][:1].isupper()
            right = toks[i].lower()
            toks[i], labels[i] = _capital(article_swap(right), capital), grammar.ending_change(article_swap(right), right)
            words.remove(i)
    for i in words[: rng.choice((1, 1, 1, 2, 2, 3))]:
        capital = toks[i][:1].isupper()
        right = toks[i].lower()
        if rng.random() < SPELLING:
            wrong = misspell(right, rng)
            if wrong and not _lexicon.known(wrong):
                toks[i], labels[i] = _capital(wrong, capital), SPELL
            continue
        wrong = other_form(right, rng)
        if wrong is None:
            continue
        label = grammar.ending_change(wrong, right)
        if label is None or label == KEEP:
            continue
        toks[i], labels[i] = _capital(wrong, capital), label
    return toks, labels


def _capital(word: str, capital: bool) -> str:
    return word[:1].upper() + word[1:] if capital else word


def article_swap(word: str) -> str | None:
    """A masculine noun with the other article ("столът" ↔ "стола")."""
    for full, short in ARTICLE:
        if word.endswith(full) and len(word) > len(full) + 2:
            swapped = word[: -len(full)] + short
            if swapped in _lexicon.forms(word) and (full in ("ът", "ят") or swapped.endswith(("ът", "ят"))):
                return swapped
    return None


def other_form(word: str, rng: random.Random) -> str | None:
    forms = _lexicon.forms(word)
    if len(forms) < 2:
        return None
    if rng.random() < 0.5:
        for full, short in ARTICLE:
            if word.endswith(full):
                swapped = word[: -len(full)] + short
                if swapped in forms:
                    return swapped
    # Forms with the same beginning: a different ending, not a different word.
    near = [f for f in forms if f != word and f[:2] == word[:2]]
    return rng.choice(near) if near else None


def misspell(word: str, rng: random.Random) -> str | None:
    if len(word) < 4:
        return None
    i = rng.randrange(1, len(word))
    kind = rng.random()
    swaps = {"ъ": "а", "а": "ъ", "о": "у", "у": "о", "е": "и", "и": "е", "я": "а", "т": "д", "д": "т"}
    if kind < 0.4 and word[i] in swaps:
        return word[:i] + swaps[word[i]] + word[i + 1 :]
    if kind < 0.6:
        return word[:i] + word[i + 1 :]
    if kind < 0.8:
        return word[:i] + word[i] + word[i:]
    return word[:i] + rng.choice(LETTERS) + word[i + 1 :]


def encode(args: tuple[str, int]) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    sentence, seed = args
    rng = random.Random(seed)
    toks, labels = spoil(sentence, rng)
    if not toks:
        return None
    chars, flags = grammar.features(toks, _lexicon)
    return chars.astype(np.int16), flags.astype(np.int8), labels, toks


_label_id: dict[str, int] = {}


def init_choices(folder: str, label_id: dict[str, int]) -> None:
    init_worker(folder)
    _label_id.update(label_id)


def word_choices(word: str) -> np.ndarray:
    return np.array(grammar.choices(word, _lexicon, _label_id), dtype=np.int16)


# ---- pairs of wrong and right sentences (the test) ---------------------------------------


def pair_labels(wrong: str, right: str) -> tuple[list[str], list[str]] | None:
    """The wrong sentence's tokens, labelled with how to make them right; None
    when the two don't line up word for word."""
    a = [t for _, _, t in grammar.tokens(wrong)]
    b = [t for _, _, t in grammar.tokens(right)]
    if len(a) != len(b):
        return None
    labels = []
    for x, y in zip(a, b):
        if x == y:
            labels.append(KEEP)
            continue
        if not _lexicon.known(x):
            labels.append(SPELL)
            continue
        label = grammar.ending_change(x.lower(), y.lower())
        if label is None:
            return None
        labels.append(label)
    return a, labels


def read_pairs(path: Path, skip: set[str]) -> list[tuple[str, str, str]]:
    with path.open(encoding="utf-8") as f:
        return [(r["error_type"], r["erroneous"], r["correct"]) for r in csv.DictReader(f) if r["error_type"] not in skip]


# ---- the model --------------------------------------------------------------------------


def build_model(n_labels: int, embed: int, conv: int, hidden: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(len(grammar.CHAR_ID) + 3, embed, padding_idx=0)
            self.conv = nn.Conv1d(embed, conv, 3, padding=1)
            self.gru = nn.GRU(conv + 2, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
            self.drop = nn.Dropout(0.1)
            self.out = nn.Linear(2 * hidden, n_labels)

        def forward(self, chars, flags):  # chars (B, T, C), flags (B, T, 2)
            b, t, c = chars.shape
            x = self.embed(chars.view(b * t, c)).transpose(1, 2)  # (BT, E, C)
            x = self.conv(x).transpose(1, 2)  # (BT, C, conv)
            x = x.masked_fill((chars.view(b * t, c) == 0).unsqueeze(-1), -1e4)
            word = torch.relu(x.max(dim=1).values).view(b, t, -1)
            h, _ = self.gru(self.drop(torch.cat([word, flags], dim=-1)))
            return self.out(self.drop(h))

    return Net()


def export(model, labels: list[str], path: Path) -> None:
    """The weights as numpy float16, named as grammar.Tagger reads them."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    out = {"embed": sd["embed.weight"], "conv.w": sd["conv.weight"], "conv.b": sd["conv.bias"],
           "out.w": sd["out.weight"], "out.b": sd["out.bias"]}
    for layer in range(2):
        for d, suffix in (("f", ""), ("b", "_reverse")):
            out[f"gru{layer}.{d}.wi"] = sd[f"gru.weight_ih_l{layer}{suffix}"]
            out[f"gru{layer}.{d}.wh"] = sd[f"gru.weight_hh_l{layer}{suffix}"]
            out[f"gru{layer}.{d}.bi"] = sd[f"gru.bias_ih_l{layer}{suffix}"]
            out[f"gru{layer}.{d}.bh"] = sd[f"gru.bias_hh_l{layer}{suffix}"]
    out = {k: v.astype(np.float16) for k, v in out.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=np.array(labels), **out)


def batches(items, size: int, rng: random.Random):
    order = list(range(len(items)))
    rng.shuffle(order)
    # Similar lengths together: less padding.
    chunks = [order[i : i + size * 50] for i in range(0, len(order), size * 50)]
    for chunk in chunks:
        chunk.sort(key=lambda i: len(items[i][0]))
        group = [chunk[i : i + size] for i in range(0, len(chunk), size)]
        rng.shuffle(group)
        yield from group


def collate(items, idx, label_id, allowed):
    """A batch: letters, flags, the labels, and which labels each token may
    take (its own forms: the rest are masked out)."""
    import torch

    t = max(len(items[i][0]) for i in idx)
    chars = np.zeros((len(idx), t, grammar.MAX_CHARS), dtype=np.int64)
    flags = np.zeros((len(idx), t, 2), dtype=np.float32)
    y = np.full((len(idx), t), IGNORE, dtype=np.int64)
    mask = np.zeros((len(idx), t, len(label_id)), dtype=bool)
    mask[:, :, label_id[KEEP]] = True
    for row, i in enumerate(idx):
        c, f, labels, toks = items[i]
        chars[row, : len(c)] = c
        flags[row, : len(f)] = f
        for j, (lab, tok) in enumerate(zip(labels, toks)):
            ok = allowed.get(tok)
            if ok is not None:
                mask[row, j, ok] = True
            gold = label_id.get(lab, IGNORE)
            y[row, j] = gold if gold != IGNORE and mask[row, j, gold] else IGNORE
    return torch.from_numpy(chars), torch.from_numpy(flags), torch.from_numpy(y), torch.from_numpy(mask)


def masked(logits, mask):
    return logits.masked_fill(~mask, -1e4)


def evaluate(model, items, label_id, labels, allowed):
    """Over tokens: how many changes it finds (recall), how often a change it
    suggests is right (precision), how often it changes a correct word."""
    import torch

    model.eval()
    found = right = suggested = touched_clean = clean = wanted = 0
    with torch.no_grad():
        for start in range(0, len(items), 256):
            idx = list(range(start, min(start + 256, len(items))))
            chars, flags, y, mask = collate(items, idx, label_id, allowed)
            p = torch.softmax(masked(model(chars, flags), mask), -1)
            conf, pred = p.max(-1)
            for row, i in enumerate(idx):
                for j, gold in enumerate(items[i][2]):
                    guess = labels[pred[row, j]] if conf[row, j] >= grammar.NOTE else KEEP
                    if gold == KEEP:
                        clean += 1
                        touched_clean += guess != KEEP
                    else:
                        wanted += 1
                        found += guess != KEEP
                        right += guess == gold
                    suggested += guess != KEEP
    model.train()
    return {"recall": right / max(wanted, 1), "detected": found / max(wanted, 1),
            "precision": right / max(suggested, 1), "false_alarms": touched_clean / max(clean, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("assets/grammar/tagger.npz"))
    ap.add_argument("--sentences", type=int, default=350_000, help="from Wikipedia")
    ap.add_argument("--spoken", type=int, default=350_000, help="from the subtitles")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--conv", type=int, default=160)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    init_worker(str(ASSETS))
    assert grammar.choices("стола", _lexicon, {KEEP: 0, "ENDING:а>ът": 1}) == [0, 1]

    t0 = time.time()
    parquet = next(args.data.glob("*.parquet"))
    sentences = wiki_sentences(parquet, args.sentences, args.seed)
    if (args.data / "subs.txt").is_file():
        spoken = subtitle_sentences(args.data / "subs.txt", args.spoken, args.seed)
        print(f"{len(spoken)} spoken sentences, e.g. {spoken[:3]}", flush=True)
        sentences += spoken
    # thebogko's pronoun "mistakes" are standard Bulgarian ("към кого"): left out.
    skip = {"pronoun_misuse"}
    grammar_pairs = read_pairs(args.data / "grammar_errors.csv", skip)
    spell_pairs = read_pairs(args.data / "spell_errors.csv", {"semantic"})
    test_pairs = [p for p in grammar_pairs + spell_pairs]
    rng.shuffle(test_pairs)
    # Their correct sides teach too, but not the ones the test spoils.
    held = {right for _, _, right in test_pairs[:4000]}
    sentences += [right for _, _, right in test_pairs[4000:] if good_sentence(right) and right not in held]
    print(f"{len(sentences)} sentences in {time.time() - t0:.0f}s", flush=True)

    with multiprocessing.Pool(4, init_worker, (str(ASSETS),)) as pool:
        items = [x for x in pool.imap(encode, ((s, args.seed * 1_000_003 + i) for i, s in enumerate(sentences)),
                                      chunksize=500) if x is not None]
    print(f"spoiled and encoded in {time.time() - t0:.0f}s", flush=True)

    counts = collections.Counter(lab for _, _, labels, _ in items for lab in labels)
    labels = [KEEP, SPELL] + sorted((lab for lab, n in counts.items()
                                     if lab not in (KEEP, SPELL) and n >= MIN_LABEL_COUNT), key=lambda x: -counts[x])
    label_id = {lab: i for i, lab in enumerate(labels)}
    print(f"{len(labels)} labels; most common changes: {labels[2:12]}", flush=True)

    rng.shuffle(items)
    dev, train = items[:5000], items[5000:]
    tests: dict[str, list] = collections.defaultdict(list)
    for kind, wrong, right in test_pairs[:4000]:
        got = pair_labels(wrong, right)
        if got is not None:
            toks, labs = got
            c, f = grammar.features(toks, _lexicon)
            tests[kind].append((c, f, labs, toks))
    # Each distinct token's choices, once.
    distinct = sorted({tok for group in (items, *tests.values()) for _, _, _, toks in group for tok in toks})
    with multiprocessing.Pool(4, init_choices, (str(ASSETS), label_id)) as pool:
        allowed = dict(zip(distinct, pool.map(word_choices, distinct, chunksize=2000)))
    print(f"choices for {len(allowed)} distinct tokens in {time.time() - t0:.0f}s", flush=True)

    model = build_model(len(labels), args.embed, args.conv, args.hidden)
    print(f"{sum(p.numel() for p in model.parameters()):,} weights", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * (len(train) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    step = 0
    for epoch in range(args.epochs):
        for idx in batches(train, args.batch, rng):
            chars, flags, y, mask = collate(train, idx, label_id, allowed)
            loss = loss_fn(masked(model(chars, flags), mask).flatten(0, 1), y.flatten())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 1000 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step % 5000 == 0:
                export(model, labels, args.out)
        print(f"epoch {epoch} dev {evaluate(model, dev, label_id, labels, allowed)}", flush=True)
        for kind, test in sorted(tests.items()):
            print(f"  {kind:32} {evaluate(model, test, label_id, labels, allowed)}", flush=True)
        export(model, labels, args.out)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
