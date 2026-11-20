import asyncio
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest

from tomo import grammar
from tomo.ai import AiClient, SharedCatalog
from tomo.commands import Executor
from tomo.config import Config, LlmProvider
from tomo.db import Store
from tomo.grammar import KEEP, Lexicon, Proofreader, apply_change, dictionaries, ending_change, spelling_fix

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "grammar"


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon(ASSETS / "bg_BG.aff", *dictionaries(ASSETS))


@pytest.fixture
def spelling_only(tmp_path):
    """The proofreader with the dictionary and no tagger."""
    for name in ("bg_BG.aff", "bg_BG.dic", "tomo.dic"):
        (tmp_path / name).write_bytes((ASSETS / name).read_bytes())
    return Proofreader(tmp_path)


def test_the_dictionary_knows_bulgarian_words_in_every_form(lexicon):
    for word in ("компютърът", "компютъра", "хубавите", "отворих", "по-добре", "Германия", "Добре"):
        assert lexicon.known(word), word
    for word in ("резолтат", "является", "правиме", "Firefox", "видяли"):
        assert not lexicon.known(word), word
    # Forms made by only taking letters away, and Tomo's own words.
    for word in ("откри", "обеща", "също", "три", "интернет", "гигабайта", "таймерът", "мамо"):
        assert lexicon.known(word), word
    assert {"компютър", "компютъра", "компютърът", "компютри", "компютрите"} <= lexicon.forms("компютъра")


def test_only_a_typical_slip_is_sure(lexicon):
    sure = {"незнам": "не знам", "подяволите": "по дяволите", "чуствам": "чувствам", "помоща": "помощта",
            "обикновенно": "обикновено", "свещенник": "свещеник", "видяли": "видели", "правиме": "правим",
            "резолтат": "резултат", "приложениито": "приложението", "Компутър": "Компютър"}
    for wrong, right in sure.items():
        assert spelling_fix(wrong, lexicon) == (right, 0.9), wrong
    # One fix among others, or a plainer change: for the model to judge.
    for wrong in ("впредвид", "съжелявам", "безопастност"):
        assert spelling_fix(wrong, lexicon)[1] == 0.6, wrong
    assert spelling_fix("бзхтрпщ", lexicon) == (None, 0.0), "no near word: not a slip"


def test_a_misspelling_gets_the_nearest_words(lexicon):
    assert lexicon.suggest("резолтат")[0] == "резултат"
    assert lexicon.suggest("куличествено")[0] == "количествено"
    assert lexicon.suggest("Намиръ")[0] == "Намира", "the capital stays"


def test_endings_are_changed_by_label():
    assert ending_change("компютъра", "компютърът") == "ENDING:а>ът"
    assert ending_change("правиме", "правим") == "ENDING:е>"
    assert ending_change("стол", "стол") == KEEP
    assert ending_change("котка", "куче") is None, "more than the ending"
    assert apply_change("входа", "ENDING:а>ът") == "входът"
    assert apply_change("вход", "ENDING:а>ът") is None, "the ending isn't there"


def test_misspellings_and_russian_words_are_fixed(spelling_only):
    text, left = spelling_only.fix("Този резолтат является добър, сейчас ще отворя Firefox.")
    assert text == "Този резултат е добър, сега ще отворя Firefox."
    assert left == []


def test_names_mid_sentence_and_latin_words_are_left_alone(spelling_only):
    assert spelling_only.check("Отворих Spotify за Мирослава и Гошко.") == []


def test_unknown_words_far_from_any_word_are_left_alone(spelling_only):
    """A word the dictionary lacks, not a misspelling; short ones too."""
    assert spelling_only.check("Това е бзхтрпщ, щрк.") == []


def test_a_slip_the_proofreader_isnt_sure_of_is_a_note(spelling_only):
    text, left = spelling_only.fix("Съжелявам, но незнам.")
    assert text == "Съжелявам, но не знам."
    assert [(i.wrong, i.fix) for i in left] == [("Съжелявам", "Съжалявам")]
    assert left[0].note() == "„Съжелявам“ → „Съжалявам“ (spelling)"


def test_without_its_files_the_proofreader_changes_nothing(tmp_path):
    assert Proofreader(tmp_path).fix("Резолтат.") == ("Резолтат.", [])


def test_the_sentences_of_a_text_are_found():
    text = "Здравей! Как си? Добре.\nА ти"
    assert [text[s:e].strip() for s, e in grammar.sentences(text)] == ["Здравей!", "Как си?", "Добре.", "А ти"]


def test_the_tagger_runs_a_pytorch_gru(tmp_path):
    """The numpy GRU against a hand-written one, gate by gate."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3)).astype(np.float32)
    wi, wh = rng.normal(size=(6, 3)).astype(np.float32), rng.normal(size=(6, 2)).astype(np.float32)
    bi, bh = rng.normal(size=6).astype(np.float32), rng.normal(size=6).astype(np.float32)
    h = np.zeros(2, dtype=np.float32)
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    for t in range(4):
        r = sig(wi[0:2] @ x[t] + bi[0:2] + wh[0:2] @ h + bh[0:2])
        z = sig(wi[2:4] @ x[t] + bi[2:4] + wh[2:4] @ h + bh[2:4])
        n = np.tanh(wi[4:6] @ x[t] + bi[4:6] + r * (wh[4:6] @ h + bh[4:6]))
        h = (1 - z) * n + z * h
    assert np.allclose(grammar._gru(x, wi, wh, bi, bh)[-1], h, atol=1e-5)


@pytest.mark.skipif(not (ASSETS / "tagger.npz").is_file(), reason="no trained tagger")
def test_the_shipped_tagger_finds_the_article_and_keeps_good_bulgarian():
    proof = Proofreader(ASSETS)
    for good in ("Компютърът работи бързо.", "Отворих калкулатора.", "Часът е три и половина.",
                 "Звукът вече е на тридесет.", "Ето какво намерих за времето днес."):
        assert proof.fix(good) == (good, []), good
    # The short article for the subject: at least a note with the full one.
    _, left = proof.fix("Компютъра работи бързо.")
    assert [(i.wrong, i.fix, i.kind) for i in left] == [("Компютъра", "Компютърът", "grammar")]
    # Its cut-offs came with it (scripts/calibrate_grammar.py).
    tagger = grammar.Tagger(ASSETS / "tagger.npz")
    assert tagger.temperature != 1.0 and all(note <= sure for note, sure in tagger.cuts.values())


# ---- in a turn ----------------------------------------------------------------------


def turn(tmp_path, replies, user="Как е компютърът?"):
    """One Bulgarian turn against a fake Ollama answering ``replies`` in order."""
    seen = []

    def server(request):
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": "gemma4:12b-it-qat"}]})
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": replies.pop(0)}, "done": True})

    cfg = Config.for_tests(tmp_path)
    cfg.llm_provider, cfg.llm_url = LlmProvider.OLLAMA, "http://fake:11434"
    for name in ("bg_BG.aff", "bg_BG.dic", "tomo.dic"):
        (cfg.assets_dir / "grammar").mkdir(parents=True, exist_ok=True)
        (cfg.assets_dir / "grammar" / name).write_bytes((ASSETS / name).read_bytes())
    store = Store(tmp_path / "chroma")
    client = AiClient(cfg, store, Executor(False, tmp_path / "audit.log"), SharedCatalog(), threading.Event(),
                      httpx.AsyncClient(transport=httpx.MockTransport(server)))
    reply = asyncio.run(client.respond(user, lambda e: None))
    return reply, seen, store


def test_a_sure_mistake_is_fixed_without_asking_the_model_again(tmp_path):
    reply, seen, store = turn(tmp_path, ["Компютърът е в отличен резолтат сейчас."])
    assert reply == "Компютърът е в отличен резултат сега."
    assert len(seen) == 1
    assert store.mistakes_for("резолтат", 5)


def test_an_unsure_mistake_goes_back_to_the_model_once(tmp_path):
    reply, seen, _ = turn(tmp_path, ["Съжелявам, не мога.", "Съжалявам, не мога."])
    assert reply == "Съжалявам, не мога."
    assert len(seen) == 2 and "Съжелявам" in seen[1]["messages"][-1]["content"]


def test_a_rewrite_that_isnt_better_is_dropped(tmp_path):
    reply, seen, _ = turn(tmp_path, ["Съжелявам, не мога.", "Съжелявам, безопастност."])
    assert reply == "Съжелявам, не мога."


def test_english_replies_are_not_proofread(tmp_path):
    reply, seen, _ = turn(tmp_path, ["All good, the resolt is fine."], user="How is it?")
    assert reply == "All good, the resolt is fine." and len(seen) == 1


def test_past_mistakes_are_named_before_the_model_answers(tmp_path):
    store = Store(tmp_path / "chroma")
    for _ in range(2):
        store.add_mistake("компютъра", "компютърът", "grammar")
    store.add_mistake("сейчас", "сега", "russian")
    assert store.mistakes_for("Какво прави компютърът?", 5)[0] == "„компютъра“ → „компютърът“"
    del store
    _, seen, _ = turn(tmp_path, ["Работи."], user="Какво прави компютърът?")
    assert "компютъра" in seen[0]["messages"][-1]["content"]
