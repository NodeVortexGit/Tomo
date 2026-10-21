"""Tomo's hands (the real mouse, faked here) and finding an app's windows."""

import asyncio
import threading

import pytest

from tomo import desktop, hands
from tomo.desktop import AppWindow
from tomo.events import Carry, ControlMode
from tomo.hands import BodyState, Hands, Stopped, eased, glide_seconds


def allowed() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


def test_the_cursor_eases_off_and_lands_softly_and_never_crawls():
    assert (eased(0.0), eased(0.5), eased(1.0)) == (0.0, 0.5, 1.0)
    assert eased(0.1) < 0.1 and eased(0.9) > 0.9, "slow at both ends"
    assert glide_seconds(0.0) == 0.35 and glide_seconds(10_000.0) == 1.6
    assert glide_seconds(400.0) < glide_seconds(1200.0)


def test_she_picks_the_cursor_up_carries_it_clicks_and_lets_go(mouse):
    body, cues = BodyState(), []

    def ui(event):
        cues.append(event)
        if event == Carry("reach"):  # the body: she stands and raises her arms
            body.hands = (100.0, 700.0)

    asyncio.run(Hands(ui, body, allowed()).click_at(1800.0, 20.0))
    assert cues == [ControlMode(True), Carry("reach"), Carry("hold"), Carry("drop"), ControlMode(False)]
    assert (100, 700) in mouse["moves"], "into her hands first"
    assert mouse["moves"][-1] == (1800, 20) and mouse["clicks"] == 1


def test_the_panic_key_stops_her_mid_glide(mouse):
    body, cues, consent = BodyState(), [], allowed()

    def ui(event):
        cues.append(event)
        if event == Carry("reach"):
            body.hands = (100.0, 700.0)
        if event == Carry("hold"):
            consent.clear()  # Pause, pressed as the cursor starts to fly

    with pytest.raises(Stopped):
        asyncio.run(Hands(ui, body, consent).click_at(1800.0, 20.0))
    assert mouse["clicks"] == 0 and cues[-2:] == [Carry("drop"), ControlMode(False)], "she lets go, the badge goes"


def test_stopped_as_she_reaches_she_puts_her_arms_down(mouse):
    cues, consent = [], allowed()

    def ui(event):
        cues.append(event)
        if event == Carry("reach"):
            consent.clear()

    with pytest.raises(Stopped):
        asyncio.run(Hands(ui, BodyState(), consent).click_at(5.0, 5.0))
    assert cues == [ControlMode(True), Carry("reach"), Carry("drop"), ControlMode(False)]


def test_if_she_cant_reach_the_cursor_goes_alone(mouse, monkeypatch):
    monkeypatch.setattr(hands, "WAIT_FOR_HANDS_SECS", 0.05)
    cues = []
    asyncio.run(Hands(cues.append, BodyState(), allowed()).click_at(50.0, 60.0, double=True))
    assert Carry("hold") not in cues and Carry("drop") in cues, "she put her arms down again"
    assert mouse["moves"][-1] == (50, 60) and mouse["clicks"] == 2
    cues.clear()
    asyncio.run(Hands(cues.append, None, allowed()).click_at(70.0, 80.0))  # no body at all
    assert cues == [ControlMode(True), ControlMode(False)] and mouse["moves"][-1] == (70, 80)


def test_nothing_moves_without_consent(mouse):
    with pytest.raises(Stopped):
        asyncio.run(Hands(lambda e: None, None, threading.Event()).click_at(10.0, 10.0))
    assert mouse["moves"] == [] and mouse["clicks"] == 0


def test_an_apps_windows_are_found_by_program_then_by_name(monkeypatch):
    windows = [AppWindow(1, "New Tab - Google Chrome", "chrome"), AppWindow(2, "Calculator", "CalculatorApp", True),
               AppWindow(3, "notes.txt - Notepad", "Notepad"), AppWindow(4, "Calculator - Search", "chrome")]
    monkeypatch.setattr(desktop, "open_windows", lambda: windows)
    assert [w.handle for w in desktop.app_windows("chrome", "Google Chrome")] == [1, 4]
    assert [w.handle for w in desktop.app_windows("", "Calculator")] == [2], "a Store app by its title, not a tab"
    assert [w.handle for w in desktop.app_windows("", "notepad")] == [3], "by its program's name"
    assert [w.handle for w in desktop.app_windows("winword", "Word")] == []
    assert desktop.program_name("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe") == "chrome"
