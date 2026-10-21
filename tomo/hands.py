"""Tomo's hands: the real mouse and keyboard, used where the user can watch.

When she clicks something for the brain — the Start button to open an app,
a window's close button, a spot she found on the screen — she picks the
cursor up and carries it there: she stops and raises her arms, the cursor
glides into her hands, and she hangs from it as it flies to the spot. It
clicks; she lets go and drops to the floor. The brain moves the cursor
(here, on its own thread); the body only follows it
(:class:`~tomo.events.Carry`) and says where her raised hands are through
the shared :class:`BodyState`.

Consent and safety: only while the user allows control — the panic key
(Pause, or Ctrl+Alt+Esc) takes it back at once, even mid-glide — and with
the red "Tomo is controlling the desktop" badge showing the whole time.
While it shows, her window lets every click through to the apps below.

Windows uses SetCursorPos and SendInput; Linux xdotool (X11) or ydotool
(Wayland), whichever is installed — without either, nothing moves.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable

from . import platform
from .events import BrainToUi, Carry, ControlMode

log = logging.getLogger(__name__)

STEP_SECS = 1 / 60  # how often a glide moves the cursor
REACH_SECS = 0.45  # the cursor gliding into her hands…
TAKE_SECS = 0.15  # …her taking it…
WAIT_FOR_HANDS_SECS = 2.5  # …and the longest she may take to stand and raise her arms
CLICK_PAUSE_SECS = 0.15  # after a click, before she lets go


class Stopped(Exception):
    """The user took control back (the panic key) mid-task."""


@dataclass
class BodyState:
    """What the body tells the brain, every frame (from its own thread).
    ``hands``: where her raised hands are, in screen pixels, while she's
    standing ready for the cursor (else None)."""

    hands: tuple[float, float] | None = None


def eased(t: float) -> float:
    """Ease in and out: slow off, fast between, slow to land."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def glide_seconds(distance: float) -> float:
    """How long the cursor takes over ``distance`` px: longer for further,
    but never a crawl."""
    return max(0.35, min(1.6, 0.25 + distance / 1400.0))


class Hands:
    """The brain's use of the mouse and keyboard. ``ui`` sends the body its
    cues; ``body`` is what it says back (None: no body, as in ``tomo ask``);
    ``allowed`` is the user's consent; ``carry``: whether she picks the
    cursor up (TOMO_USE_MOUSE) or it just goes there."""

    def __init__(self, ui: Callable[[BrainToUi], None], body: BodyState | None, allowed: threading.Event,
                 carry: bool = True) -> None:
        self.ui = ui
        self.body = body
        self.allowed = allowed
        self.carry = carry
        self.holding = False
        self.cued = False  # she's been asked to reach (and must be told to let go)

    def able(self) -> bool:
        """Whether she may and can use the mouse here."""
        return self.allowed.is_set() and can_move()

    def check(self) -> None:
        if not self.allowed.is_set():
            raise Stopped()

    @contextlib.asynccontextmanager
    async def task(self):
        """A task with the mouse and keyboard: the badge shows (and her
        window lets clicks through) from start to end; she always lets go."""
        self.ui(ControlMode(True))
        try:
            yield self
        finally:
            await self.let_go()
            self.ui(ControlMode(False))

    async def carry_to(self, x: float, y: float) -> None:
        """Pick the cursor up and carry it to screen pixel (x, y)."""
        self.check()
        hands = await self.reach() if self.carry and self.body is not None and not self.holding else None
        if hands is not None:
            await self.glide(hands, REACH_SECS)
            self.ui(Carry("hold"))
            self.holding = True
            await asyncio.sleep(TAKE_SECS)
        await self.glide((x, y))

    async def reach(self) -> tuple[float, float] | None:
        """Have her stand and raise her arms; where her hands are once she
        has (None: she can't — no character, a health round)."""
        self.body.hands = None
        self.ui(Carry("reach"))
        self.cued = True
        waited = 0.0
        while waited < WAIT_FOR_HANDS_SECS:
            self.check()
            if self.body.hands is not None:
                return self.body.hands
            await asyncio.sleep(0.03)
            waited += 0.03
        await self.let_go()
        return None

    async def glide(self, to: tuple[float, float], seconds: float | None = None) -> None:
        """Move the cursor smoothly to ``to``."""
        start = cursor_pos() or to
        distance = math.dist(start, to)
        seconds = glide_seconds(distance) if seconds is None else seconds
        began = time.monotonic()
        while True:
            self.check()
            t = min(1.0, (time.monotonic() - began) / seconds) if seconds > 0 else 1.0
            k = eased(t)
            move_cursor(start[0] + (to[0] - start[0]) * k, start[1] + (to[1] - start[1]) * k)
            if t >= 1.0:
                return
            await asyncio.sleep(STEP_SECS)

    async def let_go(self) -> None:
        if self.cued:
            self.cued = self.holding = False
            self.ui(Carry("drop"))

    async def click(self, double: bool = False) -> None:
        """Click where the cursor is."""
        self.check()
        click(2 if double else 1)
        await asyncio.sleep(CLICK_PAUSE_SECS)

    async def click_at(self, x: float, y: float, double: bool = False) -> None:
        """Carry the cursor to (x, y), screen pixels, and click there."""
        async with self.task():
            await self.carry_to(x, y)
            await self.click(double)

    async def type(self, text: str) -> None:
        self.check()
        type_text(text)

    async def key(self, name: str) -> None:
        """Press a key: "enter", "escape" or "win"."""
        self.check()
        press(name)


# ---- the real input ---------------------------------------------------------------------------


def can_move() -> bool:
    return platform.WINDOWS or bool(shutil.which("xdotool") or shutil.which("ydotool"))


def screenshot_to_screen(x: float, y: float) -> tuple[float, float]:
    """A point on a screenshot (from the whole desktop's corner) → the
    cursor's screen coordinates (on Windows the desktop's corner can be
    left of or above the main screen's)."""
    if platform.WINDOWS:
        import ctypes

        metrics = ctypes.windll.user32.GetSystemMetrics
        return x + metrics(76), y + metrics(77)  # SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN
    return x, y


def cursor_pos() -> tuple[float, float] | None:
    if platform.WINDOWS:
        import ctypes
        from ctypes import wintypes

        point = wintypes.POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return float(point.x), float(point.y)
        return None
    if shutil.which("xdotool"):
        try:
            out = subprocess.run(["xdotool", "getmouselocation", "--shell"], capture_output=True, text=True,
                                 timeout=2).stdout
            values = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
            return float(values["X"]), float(values["Y"])
        except (OSError, subprocess.TimeoutExpired, KeyError, ValueError):
            return None
    return None


def move_cursor(x: float, y: float) -> None:
    if platform.WINDOWS:
        import ctypes

        ctypes.windll.user32.SetCursorPos(int(round(x)), int(round(y)))
    elif shutil.which("xdotool"):
        _run(["xdotool", "mousemove", str(int(x)), str(int(y))])
    elif shutil.which("ydotool"):
        _run(["ydotool", "mousemove", "--absolute", "-x", str(int(x)), "-y", str(int(y))])


def click(times: int = 1) -> None:
    """Click the left button where the cursor is."""
    if platform.WINDOWS:
        down, up = 0x0002, 0x0004  # MOUSEEVENTF_LEFTDOWN, …LEFTUP
        _send([_mouse(flags) for _ in range(times) for flags in (down, up)])
    elif shutil.which("xdotool"):
        _run(["xdotool", "click", "--repeat", str(times), "1"])
    elif shutil.which("ydotool"):
        for _ in range(times):
            _run(["ydotool", "click", "0xC0"])
    else:
        log.info("no input tool (xdotool or ydotool): would click %d time(s)", times)


def type_text(text: str) -> None:
    """Type through the real keyboard (any script: the characters
    themselves are sent, whatever the keyboard layout)."""
    if platform.WINDOWS:
        unicode, key_up = 0x0004, 0x0002
        units = text.encode("utf-16-le")
        codes = [int.from_bytes(units[i:i + 2], "little") for i in range(0, len(units), 2)]
        _send([_key(0, code, unicode | up) for code in codes for up in (0, key_up)])
    elif shutil.which("xdotool"):
        _run(["xdotool", "type", "--delay", "15", "--", text])
    elif shutil.which("ydotool"):
        _run(["ydotool", "type", "--", text])
    else:
        log.info("no input tool (xdotool or ydotool): would type %r", text)


KEYS = {"enter": (0x0D, "Return"), "escape": (0x1B, "Escape"), "win": (0x5B, "super")}


def press(name: str) -> None:
    """Press and release a key: "enter", "escape" or "win"."""
    vk, xname = KEYS[name]
    if platform.WINDOWS:
        key_up, extended = 0x0002, 0x0001
        flags = extended if name == "win" else 0
        _send([_key(vk, 0, flags), _key(vk, 0, flags | key_up)])
    elif shutil.which("xdotool"):
        _run(["xdotool", "key", xname])
    else:
        log.info("no input tool: would press %s", name)


def _run(argv: list[str]) -> None:
    try:
        subprocess.run(argv, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("%s failed: %s", argv[0], e)


@functools.cache
def _structures():
    import ctypes
    from ctypes import wintypes

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", UNION)]

    return ctypes, INPUT, MOUSEINPUT, KEYBDINPUT


def _mouse(flags: int):
    _, INPUT, MOUSEINPUT, _ = _structures()
    item = INPUT()
    item.type = 0  # INPUT_MOUSE
    item.u.mi = MOUSEINPUT(0, 0, 0, flags, 0, 0)
    return item


def _key(vk: int, scan: int, flags: int):
    _, INPUT, _, KEYBDINPUT = _structures()
    item = INPUT()
    item.type = 1  # INPUT_KEYBOARD
    item.u.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
    return item


def _send(inputs: list) -> None:
    if not inputs:
        return
    ctypes, INPUT, _, _ = _structures()
    ctypes.windll.user32.SendInput(len(inputs), (INPUT * len(inputs))(*inputs), ctypes.sizeof(INPUT))
