#!/usr/bin/env python3
"""Make the grammar tagger's confidence honest, and show how honest it is.

The proofreader acts on the tagger's probabilities: at grammar.SURE it fixes
a word by itself, from grammar.NOTE it asks the model. That works only if
"90% sure" is right about nine times in ten. This finds the temperature
(one number dividing the scores) that makes the probabilities fit what
actually happens on sentences the tagger never trained on, prints how
often a change it suggests is right at each level of confidence, and
picks the cut-offs from that: "sure" where its changes are right nine
times in ten, "note" where half are. --write stores them in the tagger
file, which the proofreader reads them from.

Held-out sentences: thebogko's real mistakes (their test part, as
train_grammar.py keeps it out), and subtitle lines from the end of
subs.txt (training reads from the start), spoiled the way training spoils —
and many more left as they are, since most of a model's sentences are
right: how often a correct word is changed counts as much as the fixes.
Misspellings are left out: the dictionary handles those.

  python scripts/calibrate_grammar.py --data DIR --tagger assets/grammar/tagger.npz
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_grammar as tg  # noqa: E402
from tomo import grammar  # noqa: E402
from tomo.grammar import KEEP, Tagger  # noqa: E402

BINS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01)


def held_out_subtitles(path: Path, n: int, seed: int) -> list[str]:
    """Whole, known-word subtitle lines from the last part of the file."""
    with path.open(encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    tail = lines[len(lines) * 4 // 5 :]
    random.Random(seed).shuffle(tail)
    out = []
    for line in tail:
        s = line.strip().lstrip("-–— ").strip()
        if not tg.good_sentence(s, shortest=12) or tg.FULL_AFTER_PREPOSITION.search(s):
            continue
        words = [t for _, _, t in grammar.tokens(s) if grammar.WORD.fullmatch(t)]
        if len(words) >= 3 and all(tg._lexicon.known(w) for w in words):
            out.append(s)
            if len(out) >= n:
                break
    return out


def cases(tagger: Tagger, sentences: list[tuple[list[str], list[str]]]):
    """For every word with a choice: the scores of its choices, which one is
    right, and which one it has now (keep)."""
    out = []
    for toks, labels in sentences:
        chars, flags = grammar.features(toks, tg._lexicon)
        logits = tagger(chars, flags)
        for i, (tok, gold) in enumerate(zip(toks, labels)):
            allowed = grammar.choices(tok, tg._lexicon, tagger.label_id)
            if len(allowed) < 2 or gold not in tagger.label_id or tagger.label_id[gold] not in allowed:
                continue
            spell = tagger.label_id.get(grammar.SPELL)
            if spell in allowed:  # not a word: the dictionary's job
                continue
            kinds = [grammar.kind_of(tagger.labels[a]) for a in allowed]
            out.append((logits[i, allowed], allowed.index(tagger.label_id[gold]), allowed.index(tagger.label_id[KEEP]),
                        kinds))
    return out


def nll(found, temperature: float) -> float:
    total = 0.0
    for scores, gold, *_ in found:
        z = scores / temperature
        z = z - z.max()
        total -= z[gold] - np.log(np.exp(z).sum())
    return total / len(found)


def reliability(found, temperature: float) -> list[tuple[str, int, float]]:
    """Of the changes suggested at each confidence, how many were right."""
    rows = []
    for lo, hi in zip(BINS, BINS[1:]):
        right = total = 0
        for scores, gold, keep, _ in found:
            z = scores / temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            best = int(p.argmax())
            if best != keep and lo <= p[best] < hi:
                total += 1
                right += best == gold
        rows.append((f"{lo:.2f}–{min(hi, 1):.2f}", total, right / total if total else float("nan")))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tagger", type=Path, default=Path("assets/grammar/tagger.npz"))
    ap.add_argument("--spoken", type=int, default=3000, help="subtitle lines spoiled")
    ap.add_argument("--clean", type=int, default=12000, help="subtitle lines left correct")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--write", action="store_true", help="store the temperature and cut-offs in the tagger file")
    args = ap.parse_args()

    tg.init_worker(str(tg.ASSETS))
    tagger = Tagger(args.tagger)
    tagger.temperature = 1.0
    rng = random.Random(args.seed)

    # The test part of thebogko's pairs, as train_grammar.py splits them.
    pairs = tg.read_pairs(args.data / "grammar_errors.csv", {"pronoun_misuse"}) + tg.read_pairs(
        args.data / "spell_errors.csv", {"semantic"})
    random.Random(args.seed).shuffle(pairs)
    real = [got for _, wrong, right in pairs[:4000] if (got := tg.pair_labels(wrong, right)) is not None]
    lines = held_out_subtitles(args.data / "subs.txt", args.spoken + args.clean, args.seed)
    spoken = [tg.spoil(s, rng) for s in lines[: args.spoken]]
    clean = [([t for _, _, t in grammar.tokens(s)], None) for s in lines[args.spoken :]]
    clean = [(toks, [KEEP] * len(toks)) for toks, _ in clean]
    found = cases(tagger, real + spoken + clean)
    print(f"{len(found)} words with a choice, from {len(real)} real mistakes, {len(spoken)} spoiled and "
          f"{len(clean)} correct sentences")
    clean_words = cases(tagger, clean)

    best = min((nll(found, t), t) for t in np.arange(0.5, 3.01, 0.1))[1]
    for t in (1.0, best):
        print(f"\ntemperature {t:.1f}: loss {nll(found, t):.4f}")
        print("  confidence  changes  right")
        for span, n, share in reliability(found, t):
            print(f"  {span:10}  {n:7}  {share:.0%}" if n else f"  {span:10}  {n:7}  –")
    # The cut-offs, for each kind of change, from what its changes turn out
    # to be: fixed by itself where they're right nine times in ten, told
    # to the model where at least half are.
    cuts = {}
    for kind in grammar.KINDS:
        sure = cut_off(found, best, SURE_RIGHT, kind)
        note = min(cut_off(found, best, NOTE_RIGHT, kind), sure)
        cuts[kind] = (note, sure)
        for name, cut in (("note", note), ("sure", sure)):
            changes = [(p, ok) for p, ok in changes_made(found, best, kind) if p >= cut]
            right = sum(ok for _, ok in changes) / max(len(changes), 1)
            touched = sum(1 for p, _ in changes_made(clean_words, best, kind) if p >= cut)
            print(f"{kind} {name} from {cut:.2f}: {len(changes)} changes, {right:.0%} right; correct words "
                  f"changed: {touched} of {len(clean_words)} ({touched / max(len(clean_words), 1):.2%})")
    if args.write:
        with np.load(args.tagger, allow_pickle=False) as data:
            weights = {k: data[k] for k in data.files if k != "temperature" and not k.startswith(("sure", "note"))}
        weights["temperature"] = np.array(best, dtype=np.float32)
        for kind, (note, sure) in cuts.items():
            weights[f"note.{kind}"] = np.array(note, dtype=np.float32)
            weights[f"sure.{kind}"] = np.array(sure, dtype=np.float32)
        np.savez_compressed(args.tagger, **weights)
        print(f"\nstored temperature {best:.1f} and the cut-offs {cuts} in {args.tagger}")

SURE_RIGHT = 0.9
NOTE_RIGHT = 0.5
MIN_CHANGES = 30  # a cut-off needs this many changes above it to be trusted


def changes_made(found, temperature: float, kind: str | None = None) -> list[tuple[float, bool]]:
    """Each change (of ``kind``) it would suggest: its confidence, and
    whether it's right."""
    out = []
    for scores, gold, keep, kinds in found:
        z = scores / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        best = int(p.argmax())
        if best != keep and (kind is None or kinds[best] == kind):
            out.append((float(p[best]), best == gold))
    return out


def cut_off(found, temperature: float, wanted: float, kind: str) -> float:
    """The lowest confidence from which its changes of ``kind`` are right
    ``wanted`` of the time (1.01 if none is: that kind is never used)."""
    changes = sorted(changes_made(found, temperature, kind), reverse=True)
    right = 0
    best = 1.01
    for n, (p, ok) in enumerate(changes, 1):
        right += ok
        if n >= MIN_CHANGES and right / n >= wanted:
            best = p
    return max(round(best, 2), 0.5)

if __name__ == "__main__":
    main()
