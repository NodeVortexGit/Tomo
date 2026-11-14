"""Slash commands: /memory and /timer, run without the model."""

import asyncio

import pytest

from tomo import slash
from tomo.db import Store
from tomo.events import Chat, ChatLine, Role
from tomo.slash import Slash, parse_timer, suggestions


@pytest.fixture
def store():
    return Store(None)


def test_typing_a_slash_offers_the_commands_that_fit():
    assert suggestions("hello") == []
    assert [c.fill for c in suggestions("/")] == [c.fill for c in slash.COMMANDS]
    assert [c.fill for c in suggestions("/mem")] == ["/memory add ", "/memory list", "/memory forget "]
    assert [c.fill for c in suggestions("/TIMER L")] == ["/timer list"]
    # Typing the arguments: the command's own line stays as a reminder.
    assert [c.fill for c in suggestions("/memory add I like tea")] == ["/memory add "]
    assert [c.fill for c in suggestions("/timer 5 tea")] == ["/timer "]
    assert suggestions("/chat") == []


@pytest.mark.parametrize("text, expected", [
    ("5", (5, "")), ("5 tea", (5, "tea")), ("15", (15, "")), ("1 min", (1, "")), ("10 min pasta", (10, "pasta")),
    ("3 мин чай", (3, "чай")), ("5   green   tea", (5, "green tea")), ("2 2 cups", (2, "2 cups")),
])
def test_a_timer_takes_whole_minutes_and_a_label(text, expected):
    assert parse_timer(text) == expected


@pytest.mark.parametrize("text, problem", [
    ("16", "from 1 to 15 minutes"), ("0", "from 1 to 15 minutes"), ("1.5", "whole minutes"), ("5m", "whole minutes"),
    ("90s", "whole minutes"), ("2:30", "whole minutes"), ("-3", "whole minutes"), ("", "whole minutes"),
    ("tea", "whole minutes"),
])
def test_anything_but_whole_minutes_up_to_15_is_turned_down(text, problem):
    assert problem in parse_timer(text)


def test_memory_adds_lists_and_forgets(store):
    s = Slash(store, lambda e: None, lambda w: None)
    assert "don't remember anything" in s.run("/memory list")
    assert "/memory add <fact> remembers, /memory list shows" in s.run("/memory")
    assert s.run("/memory add I work night shifts") == "I'll remember that: I work night shifts"
    assert s.run("/memory add I work night shifts") == "I know that already."
    store.add_memory("fact", "The user's dog is called Sharo", 2)
    store.add_memory("fact", "The user's cat is called Mila", 1)
    store.set_pref("units", "metric")
    listing = s.run("/memory list")
    # The user's own fact first (it's the most important), the preferences last.
    assert listing.splitlines()[1:5] == ["1. I work night shifts", "2. The user's dog is called Sharo",
                                        "3. The user's cat is called Mila", "4. units: metric"]
    assert "Several things mention “called”" in s.run("/memory forget called")
    assert s.run("/memory forget dog") == "Forgotten: The user's dog is called Sharo"
    assert s.run("/memory forget 3") == "Forgotten: units: metric"
    assert store.all_prefs() == []
    assert "no number 9" in s.run("/memory forget 9")
    assert "Nothing I remember mentions “horse”" in s.run("/memory forget horse")
    assert [m.content for m in store.top_memories(10)] == ["I work night shifts", "The user's cat is called Mila"]
    assert "There's no /chat command" in s.run("/chat list")


def test_timers_run_list_cancel_and_ring(store, monkeypatch):
    events, rings = [], []
    real_sleep = asyncio.sleep

    async def run():
        s = Slash(store, events.append, rings.append, lambda: True)
        gate = asyncio.Event()

        async def sleep(seconds):  # time passes when the test says so
            await gate.wait()

        monkeypatch.setattr(slash.asyncio, "sleep", sleep)
        assert s.run("/timer 5 tea").startswith("Timer 1: 5 min (tea), until ")
        assert s.run("/timer 2").startswith("Timer 2: 2 min, until ")
        assert "from 1 to 15 minutes" in s.run("/timer 20")
        assert "No timers" not in (listing := s.run("/timer list"))
        assert "1. tea — 5 min 00 s left" in listing and "2. 2 min — 2 min 00 s left" in listing
        assert "Which one?" in s.run("/timer cancel")
        assert s.run("/timer cancel 2") == "Stopped: timer 2."
        gate.set()
        for _ in range(5):
            await real_sleep(0)
        return s

    s = asyncio.run(run())
    assert s.timers == []
    assert events == [Chat(ChatLine(Role.SYSTEM, "Time's up: tea.", events[0].line.at_ms))]
    assert rings == ["Time's up: tea."], "an English label speaks English"


def test_a_timer_without_a_label_speaks_the_users_language(store, monkeypatch):
    rings = []

    async def run():
        async def sleep(seconds):
            pass

        monkeypatch.setattr(slash.asyncio, "sleep", sleep)
        s = Slash(store, lambda e: None, rings.append, lambda: True)
        s.run("/timer 1")
        s.run("/timer 1 чай")
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert rings == ["Времето изтече.", "Времето изтече: чай."]


def test_no_more_than_five_timers(store):
    async def run():
        s = Slash(store, lambda e: None, lambda w: None)
        for _ in range(5):
            s.run("/timer 10")
        out = s.run("/timer 10")
        s.stop()
        return out

    assert "5 timers are running already" in asyncio.run(run())
