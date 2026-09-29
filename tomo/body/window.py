"""The surface Tomo lives on: a borderless, always-on-top window over the
desktop, see-through everywhere but where she (or the chat) is.

┌────────────────────────────────────────────────────────────────────────┐
│ see-through    Windows: pure black is a colour key — the desktop shows  │
│                through it and clicks go to what's below.                │
│                Linux: a transparent framebuffer (with a compositor).    │
│ click-through  Windows: the colour key.                                 │
│                Linux X11: the window lets the mouse through except over │
│                her and the chat (checked every frame).                  │
│                Linux Wayland: the window catches the mouse everywhere.  │
│ always on top  Windows, X11: yes. GNOME Wayland ignores it.             │
│ no taskbar     Windows: a tool window. Linux: the desktop decides.      │
└────────────────────────────────────────────────────────────────────────┘

On Linux Tomo prefers X11 (XWayland under Wayland), where windows may be
placed, kept on top and made click-through; ``TOMO_GLFW_PLATFORM=wayland``
asks for native Wayland instead.

The window fills the work area (the screen minus the taskbar, which is then
her floor). During a health round it covers every screen (:meth:`Window.cover`).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum

import glfw

from .. import platform

log = logging.getLogger(__name__)

IDLE_FPS = 24  # at rest; while anything moves, the display's rate
WS_EX_LAYERED, WS_EX_TOOLWINDOW, WS_EX_APPWINDOW = 0x00080000, 0x00000080, 0x00040000
GWL_EXSTYLE, LWA_COLORKEY = -20, 0x1
HWND_TOPMOST = -1
SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW, SWP_NOACTIVATE = 0x2, 0x1, 0x40, 0x10


# ---- the desktop session ----------------------------------------------------------------------


class DisplayServer(Enum):
    WAYLAND = "Wayland"
    X11 = "X11"
    WINDOWS = "Windows"
    MACOS = "macOS"
    UNKNOWN = "unknown display server"


class Desktop(Enum):
    KDE = "KDE Plasma"
    GNOME = "GNOME"
    XFCE = "XFCE"
    CINNAMON = "Cinnamon"
    HYPRLAND = "Hyprland"
    I3 = "i3"
    SWAY = "Sway"
    OTHER = "other desktop"


def detect_server(session_type: str | None, wayland_display: str | None, x_display: str | None) -> DisplayServer:
    t = (session_type or "").lower()
    if t == "wayland":
        return DisplayServer.WAYLAND
    if t == "x11":
        return DisplayServer.X11
    if wayland_display:
        return DisplayServer.WAYLAND
    if x_display:
        return DisplayServer.X11
    return DisplayServer.UNKNOWN


def detect_desktop(raw: str) -> Desktop:
    r = raw.lower()  # XDG_CURRENT_DESKTOP can be colon-separated ("ubuntu:GNOME")
    for key, desktop in (("hyprland", Desktop.HYPRLAND), ("sway", Desktop.SWAY), ("kde", Desktop.KDE),
                         ("plasma", Desktop.KDE), ("gnome", Desktop.GNOME), ("xfce", Desktop.XFCE),
                         ("cinnamon", Desktop.CINNAMON), ("i3", Desktop.I3)):
        if key in r:
            return desktop
    return Desktop.OTHER


@dataclass(frozen=True)
class Session:
    server: DisplayServer
    desktop: Desktop
    raw_desktop: str

    @classmethod
    def detect(cls) -> "Session":
        if platform.WINDOWS:
            server = DisplayServer.WINDOWS
        elif sys.platform == "darwin":
            server = DisplayServer.MACOS
        else:
            server = detect_server(os.environ.get("XDG_SESSION_TYPE"), os.environ.get("WAYLAND_DISPLAY"),
                                   os.environ.get("DISPLAY"))
        raw = os.environ.get("XDG_CURRENT_DESKTOP") or os.environ.get("DESKTOP_SESSION") or ""
        return cls(server, detect_desktop(raw), raw)

    def describe(self) -> str:
        if self.server in (DisplayServer.WINDOWS, DisplayServer.MACOS):
            return self.server.value
        if self.desktop == Desktop.OTHER and self.raw_desktop:
            return f"{self.desktop.value} ({self.raw_desktop}) on {self.server.value}"
        return f"{self.desktop.value} on {self.server.value}"

    def log_notes(self) -> None:
        """The caveats for this desktop, so a first run on, say, GNOME
        Wayland explains itself instead of looking broken."""
        log.info("desktop session: %s", self.describe())
        if self.server == DisplayServer.X11:
            log.info("X11: a compositor must be running (picom, kwin, mutter…) or her background shows black")
        elif self.server == DisplayServer.WAYLAND and self.desktop == Desktop.GNOME:
            log.warning("GNOME on Wayland keeps no window on top: Tomo may sit among the other windows")
        elif self.server == DisplayServer.UNKNOWN:
            log.warning("couldn't tell which display server this is")


# ---- the window ----------------------------------------------------------------------------------


class Window:
    def __init__(self, session: Session) -> None:
        self.session = session
        if sys.platform.startswith("linux") and os.environ.get("DISPLAY") and \
                os.environ.get("TOMO_GLFW_PLATFORM", "").lower() != "wayland":
            glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        if not glfw.init():
            raise RuntimeError("couldn't start GLFW (no display?)")
        self.x11 = sys.platform.startswith("linux") and glfw.get_platform() == glfw.PLATFORM_X11
        self.wayland = sys.platform.startswith("linux") and glfw.get_platform() == glfw.PLATFORM_WAYLAND
        self.colour_key = platform.WINDOWS
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, sys.platform == "darwin")
        glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.FLOATING, True)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.FOCUSED, False)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, False)
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.AUTO_ICONIFY, False)
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        # Windows shows the desktop through a colour key instead (see above):
        # a transparent framebuffer there goes through DWM and catches clicks.
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, not self.colour_key)
        glfw.window_hint_string(glfw.X11_CLASS_NAME, "tomo.desktop.companion")
        glfw.window_hint_string(glfw.X11_INSTANCE_NAME, "tomo")
        glfw.window_hint_string(glfw.WAYLAND_APP_ID, "tomo.desktop.companion")
        x, y, w, h = self.work_area()
        self.handle = glfw.create_window(w, h, "Tomo", None, None)
        if not self.handle:
            raise RuntimeError("couldn't open Tomo's window (OpenGL 3.3 is needed)")
        glfw.make_context_current(self.handle)
        glfw.swap_interval(1)
        if not self.wayland:
            glfw.set_window_pos(self.handle, x, y)
        if platform.WINDOWS:
            self._windows_style()
        glfw.show_window(self.handle)
        self.passthrough = False
        self.covering = False
        self.last_push = 0.0
        self.frame_started = time.perf_counter()

    # ---- geometry -------------------------------------------------------------------------

    def work_area(self) -> tuple[int, int, int, int]:
        """The primary screen minus the taskbar/panels: (x, y, width, height)."""
        if platform.WINDOWS:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
                if rect.right > rect.left and rect.bottom > rect.top:
                    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        monitor = glfw.get_primary_monitor()
        if monitor:
            x, y, w, h = glfw.get_monitor_workarea(monitor)
            if w > 0 and h > 0:
                return x, y, w, h
            mode = glfw.get_video_mode(monitor)
            if mode:
                return 0, 0, mode.size.width, mode.size.height
        return 0, 0, 1280, 800

    def whole_desktop(self) -> tuple[int, int, int, int]:
        """Every screen: the lock screen covers them all."""
        if platform.WINDOWS:
            import ctypes

            m = ctypes.windll.user32.GetSystemMetrics
            if m(78) > 0 and m(79) > 0:
                return m(76), m(77), m(78), m(79)  # the virtual screen
        monitor = glfw.get_primary_monitor()
        if monitor:
            x, y = glfw.get_monitor_pos(monitor)
            mode = glfw.get_video_mode(monitor)
            if mode:
                return x, y, mode.size.width, mode.size.height
        return self.work_area()

    def scale(self) -> float:
        """The display's scaling (1.0 at 100 %, 1.5 at 150 %)."""
        sx, _ = glfw.get_window_content_scale(self.handle)
        return sx if sx > 0 else 1.0

    def framebuffer(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self.handle)

    def logical_size(self) -> tuple[float, float]:
        """The window in logical pixels: the world's units."""
        w, h = self.framebuffer()
        s = self.scale()
        return w / s, h / s

    def to_logical(self, x: float, y: float) -> tuple[float, float]:
        """Window coordinates (as GLFW reports the cursor) → logical pixels."""
        ww, _ = glfw.get_window_size(self.handle)
        fw, _ = self.framebuffer()
        k = (fw / ww if ww else 1.0) / self.scale()
        return x * k, y * k

    def cursor(self) -> tuple[float, float]:
        return self.to_logical(*glfw.get_cursor_pos(self.handle))

    def screen_to_logical(self, x: float, y: float) -> tuple[float, float]:
        """A screen pixel (as the screenshots count them: from the whole
        desktop's corner) → the window's logical pixels."""
        dx, dy, _, _ = self.whole_desktop() if platform.WINDOWS else (0, 0, 0, 0)
        wx, wy = glfw.get_window_pos(self.handle)
        s = self.scale()
        return (x + dx - wx) / s, (y + dy - wy) / s

    # ---- behaviour --------------------------------------------------------------------------

    def _windows_style(self) -> None:
        """A colour-keyed layered window: Windows shows the desktop through
        every pure-black pixel — the cleared background — and sends the
        clicks there to the window below. And a tool window: no taskbar
        button, no Alt+Tab entry."""
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = glfw.get_win32_window(self.handle)
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        style = (style | WS_EX_LAYERED | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        if not user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_COLORKEY):
            log.warning("couldn't make the window see-through; Tomo's background will show black")

    def set_mouse_wanted(self, wanted: bool) -> None:
        """X11: let the mouse through the window unless it's over her or the
        chat (Windows' colour key does this by itself; Wayland can't tell
        where the pointer is once it's let through, so there it stays)."""
        if not self.x11 or self.covering:
            wanted = True
        if self.passthrough == (not wanted):
            return
        self.passthrough = not wanted
        glfw.set_window_attrib(self.handle, glfw.MOUSE_PASSTHROUGH, self.passthrough)

    def cover(self, on: bool) -> None:
        """A health round: the window over every screen (then back to the
        work area)."""
        self.covering = on
        if on:
            self.set_mouse_wanted(True)
        x, y, w, h = self.whole_desktop() if on else self.work_area()
        if not self.wayland:
            glfw.set_window_pos(self.handle, x, y)
        glfw.set_window_size(self.handle, w, h)
        if on:
            self.push_front()

    def keep_in_front(self) -> None:
        """While locked: back on top, once a second (another app may push
        forward)."""
        now = time.monotonic()
        if now - self.last_push >= 1.0:
            self.last_push = now
            self.push_front()

    def push_front(self) -> None:
        if platform.WINDOWS:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = glfw.get_win32_window(self.handle)
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            user32.SetForegroundWindow(hwnd)
        else:
            glfw.focus_window(self.handle)

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.handle))

    def finish_frame(self, busy: bool) -> None:
        """Show the frame; at rest, wait so it's no more than 24 a second."""
        glfw.swap_buffers(self.handle)
        if not busy:
            left = 1.0 / IDLE_FPS - (time.perf_counter() - self.frame_started)
            if left > 0:
                glfw.wait_events_timeout(left)
        self.frame_started = time.perf_counter()

    def close(self) -> None:
        glfw.destroy_window(self.handle)
        glfw.terminate()
