"""Proofreading Tomo's Bulgarian: the rules, the dictionary's suspicions, the
model's answer and when it's taken."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from tomo.config import Config
from tomo.grammar import Grammar, accepted, drop_foreign

RULES = Path(__file__).resolve().parents[1] / "assets" / "grammar" / "bg_rules.json"
KNOWN = {"аз", "съм", "и", "мога", "да", "помагам", "със", "задачите", "ви", "можеш", "изпиеш", "чаша", "вода",
         "тук", "има", "грешка", "това", "е", "добре", "във", "вторник", "ще", "дойда", "сега", "много", "ми",
         "харесва", "регулира", "температурата", "техниката", "на", "запазиш", "фокуса", "си", "за", "отвори"}


class Server:
    api = type("Api", (), {"value": "ollama"})()
    url = "http://fake:11434"


def grammar(tmp_path, reply=None, status=200, seen=None):
    cfg = Config.for_tests(tmp_path)
    cfg.grammar = True

    def answer(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={"error": "model not found"})
        return httpx.Response(200, json={"message": {"content": reply}})

    async def server():
        return Server()

    return Grammar(cfg, server, httpx.AsyncClient(transport=httpx.MockTransport(answer)),
                   lookup=lambda w: w.lower() in KNOWN, rules=RULES)


def test_the_rules_fix_what_is_always_wrong(tmp_path):
    g = grammar(tmp_path)
    assert g.apply_rules("Мога да помагам с задачите ви.") == "Мога да помагам със задачите ви."
    assert g.apply_rules("С здраве! В вторник ще дойда.") == "Със здраве! Във вторник ще дойда."
    assert g.apply_rules("с теб и в Пловдив") == "с теб и в Пловдив", "not before other letters"
    assert g.apply_rules("Это много хорошо сейчас, что ли.") == "Това много добре сега, что ли.", "„что“: two meanings"
    assert g.apply_rules("Это добре") == "Това добре", "a capital stays a capital"
    assert g.apply_rules("кс задачите") == "кс задачите", "only the word „с“"


def test_the_dictionary_and_the_alphabets_say_what_looks_wrong(tmp_path):
    g = grammar(tmp_path)
    assert g.suspicious("Аз съм Томо и мога да помагам.") == [], "Томо is allowed; the rest is known"
    assert g.suspicious("Можеш да изпийш чаша вода.") == ["изпийш"]
    assert g.suspicious("Това е Райлешовscattering и сма자는.") == ["Райлешовscattering", "сма자는"]
    assert g.suspicious("Сыр и Python, и Windows.") == ["Сыр"], "Latin words are names: left alone"


def test_a_piece_with_nothing_wrong_never_reaches_the_model(tmp_path):
    seen = []
    g = grammar(tmp_path, "whatever", seen=seen)
    assert asyncio.run(g.fix(" Тук има грешка. ")) == " Тук има грешка. ", "the spaces around stay"
    assert asyncio.run(g.fix("Open the calculator, please.")) == "Open the calculator, please.", "English: as is"
    assert seen == []


def test_a_suspicious_piece_is_fixed_by_the_model_on_the_cpu(tmp_path):
    seen = []
    g = grammar(tmp_path, "Текст: Можеш да изпиеш чаша вода.", seen=seen)
    assert asyncio.run(g.fix(" Можеш да изпийш чаша вода.")) == " Можеш да изпиеш чаша вода."
    request = seen[0]
    assert request["options"]["num_gpu"] == 0, "the graphics card is the brain's"
    assert request["model"] == g.model and "изпийш" in request["messages"][1]["content"]


def test_without_the_model_the_rules_still_apply_and_other_alphabets_go(tmp_path):
    g = grammar(tmp_path, status=404)
    assert asyncio.run(g.fix("Тя регулира температурата, сма자는 с задачите.")) == \
        "Тя регулира температурата, сма със задачите."
    assert g._no_model, "asked once: a model that isn't there isn't asked again"


@pytest.mark.parametrize("answer, taken", [
    ("Можеш да изпиеш чаша вода.", True),
    ("Текст: Можеш да изпиеш чаша вода.", True),
    ("„Можеш да изпиеш чаша вода.“", True),
    ("Можеш да **изпиеш** чаша вода.", True),
    ("Разбира се, ето текста с поправените думи:\n\nМожеш да изпиеш чаша вода.", False),
    ("Томо.", False),
    ("", False),
])
def test_the_models_answer_is_taken_only_when_its_the_text_corrected(answer, taken):
    original = "Можеш да изпийш чаша вода."
    assert (accepted(original, answer) is not None) == taken


def test_an_answer_that_changes_numbers_or_drops_a_name_isnt_taken():
    assert accepted("Звукът е на 40 процента.", "Звукът е на 30 процента.") is None
    assert accepted("Започни с Python сега.", "Започни със Пайтън сега.") is None
    assert accepted("Започни с Python сега.", "Започни със Python сега.") == "Започни със Python сега."


def test_a_fix_keeps_the_shape_of_a_piece_cut_from_a_sentence(tmp_path):
    from tomo.grammar import same_shape

    assert same_shape("често дъждиво,", "Често дъждовно.") == "често дъждовно,"
    assert same_shape("Можеш да изпийш вода.", "можеш да изпиеш вода.") == "Можеш да изпиеш вода."
    assert same_shape("Тук има грешка", "Тук има поправка.") == "Тук има поправка"
    g = grammar(tmp_path, "Можеш да изпиеш чаша вода.")
    assert asyncio.run(g.fix(" можеш да изпийш чаша вода,")) == " можеш да изпиеш чаша вода,"


def test_letters_from_other_alphabets_are_taken_out():
    assert drop_foreign("Клифърт Чандलर и сма자는 тук", ["Чандलर", "сма자는"]) == "Клифърт Чанд и сма тук"
