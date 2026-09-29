"""The health programme in the body: the lock screen, and Tomo as a mirror.

The brain (:mod:`tomo.health`) decides when a round starts and counts the
reps; this side shows it. While a round is on:

* the window covers every screen, drawn on an opaque backdrop (not the colour
  key, so nothing of the desktop shows or takes a click), kept on top, with
  the keyboard held (Windows: a low-level hook swallows every key — Win,
  Alt+Tab, Alt+F4 … — while Ctrl+Alt+Del stays with Windows, as it always
  does);
* Tomo stands in the middle, big, and mirrors the user: the app stops her
  physics and places her (lower as the user's hips come down, see
  :func:`hip_drop`), :mod:`.animation` turns the joints into her pose;
* a card shows the exercise, the count and how to be seen.

On Linux the lock screen shows and takes the pointer, but the desktop's own
shortcuts stay available.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

from imgui_bundle import imgui

from .. import events, platform
from .mathx import Vec3
from .ui import flags, vec4

log = logging.getLogger(__name__)

# Dark, but not the colour key's pure black — that would show the desktop
# through and let clicks through.
BACKDROP = (0.055, 0.075, 0.09)
VISIBLE = 0.5


@dataclass
class Workout:
    """The round under way, as the body knows it."""

    locked: bool = False
    progress: object | None = None  # tomo.health.Progress
    pose: list | None = None  # the user's joints, latest frame (None: not seen)

    def receive(self, event) -> bool:
        """Take a health event. True when the lock went on or off."""
        if isinstance(event, events.HealthLock):
            if event.on != self.locked:
                log.info("health round %s", "started: desktop locked" if event.on else "over: desktop free")
                self.locked = event.on
                if not event.on:
                    self.progress = self.pose = None
                return True
        elif isinstance(event, events.HealthProgress):
            self.progress = event.progress
        elif isinstance(event, events.Pose):
            self.pose = event.joints
        return False

    def draw(self, ui, width: float) -> None:
        """The card at the top: the exercise, the count, how to be seen."""
        if not self.locked:
            return
        p = self.progress
        title, count, target, hint = (p.title, p.count, p.target, p.hint) if p is not None else ("", 0, 10, "")
        imgui.set_next_window_pos(imgui.ImVec2(width * 0.5, 36.0), 0, imgui.ImVec2(0.5, 0.0))
        imgui.push_style_color(imgui.Col_.window_bg.value, vec4(20, 28, 34, 235))
        imgui.push_style_var(imgui.StyleVar_.window_padding.value, imgui.ImVec2(36.0, 18.0))
        imgui.push_style_var(imgui.StyleVar_.window_rounding.value, 18.0)
        w = imgui.WindowFlags_
        imgui.begin("tomo-workout", None, flags(w.no_decoration, w.always_auto_resize, w.no_inputs,
                                                w.no_saved_settings, w.no_focus_on_appearing, w.no_nav))
        centred(title, 30.0, vec4(120, 230, 240))
        centred(f"{count} / {target}", 64.0, vec4(255, 255, 255), ui.bold)
        bar = max(0.0, min(1.0, count / max(target, 1)))
        imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), (imgui.get_window_width() - 320.0) * 0.5))
        imgui.push_style_color(imgui.Col_.plot_histogram.value, vec4(64, 196, 208))
        imgui.progress_bar(bar, imgui.ImVec2(320.0, 10.0), "")
        imgui.pop_style_color()
        imgui.dummy(imgui.ImVec2(0.0, 6.0))
        centred(hint, 16.0, vec4(190, 190, 190))
        imgui.end()
        imgui.pop_style_var(2)
        imgui.pop_style_color()


def centred(text: str, size: float, color, font=None) -> None:
    """A line of text, centred in the window, at ``size`` px."""
    imgui.push_font(font, size)
    width = imgui.calc_text_size(text).x
    imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), (imgui.get_window_width() - width) * 0.5))
    imgui.text_colored(color, text)
    imgui.pop_font()


# ---- the user's joints, mirrored ----------------------------------------------------------------


def mirrored(pose: list, index: int) -> Vec3 | None:
    """A joint in Tomo's model axes, as her mirror image: the user's right
    shows on her left, as in a mirror (+X her left, +Y up, +Z toward the
    viewer). MediaPipe's axes are x right, y down, z away from the camera:
    all three flip — the mirror's reflection, with left and right swapped by
    the caller."""
    if index >= len(pose):
        return None
    j = pose[index]
    if j.v < VISIBLE:
        return None
    return -j.x, -j.y, -j.z


def hip_drop(pose: list) -> float | None:
    """How far the user's hips have come down, 0 (standing) to 1 (on the
    floor). The pose model gives joints relative to the hips, so where the
    body is comes from what's on the floor: the lowest joint — the feet
    standing or squatting, the hands and feet in a plank. The hips are that
    far above the floor; against the legs' length that's how high they are.
    So in a push-up the body comes down to the hands (as the elbows bend, the
    hands come up toward the hips), not the hands up to the body — and Tomo
    with it."""
    from ..health import (LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_WRIST, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE,
                          RIGHT_WRIST)

    def leg(hip: int, knee: int, ankle: int) -> float | None:
        h, k, a = mirrored(pose, hip), mirrored(pose, knee), mirrored(pose, ankle)
        if h is None or k is None or a is None:
            return None
        return math.dist(h, k) + math.dist(k, a)

    left, right = leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), leg(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    if left is not None and right is not None:
        length = (left + right) / 2.0
    else:
        length = left if left is not None else right
    if length is None or length < 0.1:
        return None
    l_hip, r_hip = mirrored(pose, LEFT_HIP), mirrored(pose, RIGHT_HIP)
    if l_hip is not None and r_hip is not None:
        hips_y = (l_hip[1] + r_hip[1]) / 2.0
    else:
        hips_y = (l_hip or r_hip)[1]
    floor = hips_y
    for i in (LEFT_WRIST, RIGHT_WRIST, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
        p = mirrored(pose, i)
        if p is not None:
            floor = min(floor, p[1])
    return 1.0 - max(0.0, min(1.0, (hips_y - floor) / length))


# ---- Windows: the keyboard held ----------------------------------------------------------------


class KeyboardHold:
    """While on, a low-level keyboard hook swallows every key, anywhere.
    (Windows never lets a hook see Ctrl+Alt+Del.) The hook lives on a thread
    of its own with a message loop, as low-level hooks need, installed the
    first time it's switched on."""

    def __init__(self) -> None:
        self.on = False
        self._started = False
        self._proc = None

    def set(self, on: bool) -> None:
        self.on = on
        if on and not self._started and platform.WINDOWS:
            self._started = True
            threading.Thread(target=self._run, name="tomo-keyboard-hold", daemon=True).start()

    def _run(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        lresult = ctypes.c_ssize_t
        hookproc = ctypes.WINFUNCTYPE(lresult, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, hookproc, wintypes.HINSTANCE, wintypes.DWORD]
        user32.SetWindowsHookExW.restype = wintypes.HHOOK
        user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        user32.CallNextHookEx.restype = lresult
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE

        def keyboard(code: int, wparam, lparam):
            if code == 0 and self.on:  # HC_ACTION
                return 1
            return user32.CallNextHookEx(None, code, wparam, lparam)

        self._proc = hookproc(keyboard)  # kept: Windows calls it for as long as Tomo runs
        hook = user32.SetWindowsHookExW(13, self._proc, kernel32.GetModuleHandleW(None), 0)  # WH_KEYBOARD_LL
        if not hook:
            log.warning("couldn't hold the keyboard for the health round (error %d)", ctypes.get_last_error())
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass
