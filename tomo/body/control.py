"""Driving the real mouse and keyboard — the "mouse in disguise".

When the brain clicks something on screen (its ``click_at`` tool, only when
the user allows control), it sends :class:`~tomo.events.ClickAt`: the
character walks over to the spot, then the real cursor moves and clicks.
``type_text`` types through the real keyboard. It's all local, and every
move can be watched.

Consent and safety come first:

* a red **"Tomo is controlling the desktop"** badge shows the whole time she
  drives input;
* the **panic key** — Pause, or Ctrl+Alt+Esc — releases control at once and
  tells the brain to stop (on Windows it works from anywhere while she's in
  control, even when another window has the keyboard).

Windows uses ``SendInput``; Linux ``xdotool`` (X11) or ``ydotool`` (Wayland),
whichever is installed — without either, what she would have done is logged.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

from imgui_bundle import imgui

from .. import events, platform
from ..events import PanicStop
from .movement import Locomotion
from .ui import ICON_MOUSE, flags, vec4

log = logging.getLogger(__name__)

VK_PAUSE, VK_ESCAPE, VK_CONTROL, VK_MENU = 0x13, 0x1B, 0x11, 0x12


@dataclass
class Control:
    active: bool = False  # she's driving input: the badge shows
    pending: tuple[float, float, bool] | None = None  # a click waiting for her to walk over

    def receive(self, event, loco: Locomotion | None, to_arena: Callable[[float, float], tuple[float, float]]) -> None:
        """``to_arena``: screen pixels → the window's logical pixels."""
        if isinstance(event, events.ControlMode):
            self.active = event.on
        elif isinstance(event, events.ClickAt):
            self.pending = (event.x, event.y, event.double)
            self.active = True
            if loco is not None:
                x, _ = to_arena(event.x, event.y)
                width = max(loco.arena.x, 1.0)
                loco.walk_to(x / width)
                loco.held = True  # no wandering off mid-task
        elif isinstance(event, events.TypeText):
            self.active = True
            type_text(event.text)
            self.active = False

    def update(self, loco: Locomotion | None) -> None:
        """Once she has walked over, the real click."""
        if self.pending is None or loco is None or not loco.is_idle():
            return
        x, y, double = self.pending
        click(x, y, double)
        self.pending = None
        loco.held = False
        self.active = False

    def panic(self, loco: Locomotion | None, send: Callable) -> None:
        """Release control at once, and tell the brain to stop."""
        self.pending = None
        self.active = False
        if loco is not None:
            loco.held = False
        send(PanicStop())
        log.warning("panic key: mouse/keyboard control released")

    def draw_badge(self, width: float) -> None:
        """The always-visible sign while she drives input."""
        if not self.active:
            return
        imgui.set_next_window_pos(imgui.ImVec2(width * 0.5, 14.0), 0, imgui.ImVec2(0.5, 0.0))
        imgui.push_style_color(imgui.Col_.window_bg.value, vec4(200, 60, 60, 235))
        imgui.push_style_var(imgui.StyleVar_.window_rounding.value, 999.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding.value, imgui.ImVec2(14.0, 7.0))
        w = imgui.WindowFlags_
        imgui.begin("tomo-control-badge", None, flags(w.no_decoration, w.always_auto_resize, w.no_inputs,
                                                      w.no_saved_settings, w.no_focus_on_appearing, w.no_nav))
        imgui.text_colored(vec4(255, 255, 255), f"{ICON_MOUSE}  Tomo is controlling the desktop  ·  "
                                                "press Pause to stop")
        imgui.end()
        imgui.pop_style_var(2)
        imgui.pop_style_color()


def panic_pressed_anywhere() -> bool:
    """Windows: the panic key, pressed wherever the keyboard is."""
    if not platform.WINDOWS:
        return False
    import ctypes

    state = ctypes.windll.user32.GetAsyncKeyState
    pause = state(VK_PAUSE) & 1
    combo = state(VK_ESCAPE) & 0x8000 and state(VK_CONTROL) & 0x8000 and state(VK_MENU) & 0x8000
    return bool(pause or combo)


# ---- the real input ---------------------------------------------------------------------------


def click(x: float, y: float, double: bool) -> None:
    """Move the real cursor to screen pixel (x, y) — the whole desktop's, as
    the screenshots count them — and click."""
    times = 2 if double else 1
    if platform.WINDOWS:
        _windows_click(x, y, times)
    elif shutil.which("xdotool"):
        _run(["xdotool", "mousemove", str(int(x)), str(int(y)), "click", "--repeat", str(times), "1"])
    elif shutil.which("ydotool"):
        _run(["ydotool", "mousemove", "--absolute", "-x", str(int(x)), "-y", str(int(y))])
        for _ in range(times):
            _run(["ydotool", "click", "0xC0"])
    else:
        log.info("no input tool (xdotool or ydotool): would %sclick at (%.0f, %.0f)",
                 "double-" if double else "", x, y)


def type_text(text: str) -> None:
    """Type through the real keyboard."""
    if platform.WINDOWS:
        _windows_type(text)
    elif shutil.which("xdotool"):
        _run(["xdotool", "type", "--delay", "15", "--", text])
    elif shutil.which("ydotool"):
        _run(["ydotool", "type", "--", text])
    else:
        log.info("no input tool (xdotool or ydotool): would type %r", text)


def _run(argv: list[str]) -> None:
    try:
        subprocess.run(argv, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("%s failed: %s", argv[0], e)


def _windows_inputs():
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


def _windows_click(x: float, y: float, times: int) -> None:
    ctypes, INPUT, MOUSEINPUT, _ = _windows_inputs()
    metrics = ctypes.windll.user32.GetSystemMetrics
    width, height = metrics(78), metrics(79)  # the virtual screen: every monitor
    # The screenshot's pixel (0, 0) is the virtual screen's corner.
    nx = int(round(x * 65535 / max(width - 1, 1)))
    ny = int(round(y * 65535 / max(height - 1, 1)))
    MOVE, ABSOLUTE, VIRTUALDESK, DOWN, UP = 0x0001, 0x8000, 0x4000, 0x0002, 0x0004

    def mouse(dx: int, dy: int, flags_: int):
        item = INPUT()
        item.type = 0  # INPUT_MOUSE
        item.u.mi = MOUSEINPUT(dx, dy, 0, flags_, 0, 0)
        return item

    inputs = [mouse(nx, ny, MOVE | ABSOLUTE | VIRTUALDESK)]
    for _ in range(times):
        inputs += [mouse(0, 0, DOWN), mouse(0, 0, UP)]
    ctypes.windll.user32.SendInput(len(inputs), (INPUT * len(inputs))(*inputs), ctypes.sizeof(INPUT))


def _windows_type(text: str) -> None:
    ctypes, INPUT, _, KEYBDINPUT = _windows_inputs()
    UNICODE, KEYUP = 0x0004, 0x0002
    units = text.encode("utf-16-le")
    events_ = []
    for i in range(0, len(units), 2):
        code = int.from_bytes(units[i:i + 2], "little")
        for up in (0, KEYUP):
            item = INPUT()
            item.type = 1  # INPUT_KEYBOARD
            item.u.ki = KEYBDINPUT(0, code, UNICODE | up, 0, 0)
            events_.append(item)
    if events_:
        array = (INPUT * len(events_))(*events_)
        ctypes.windll.user32.SendInput(len(events_), array, ctypes.sizeof(INPUT))
