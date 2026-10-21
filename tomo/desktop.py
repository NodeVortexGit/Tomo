"""The desktop's windows — to close an app the way a person does, with its
close button — and the Start button, to open one.

Windows: plain Win32, through ctypes.

* An app's windows are the ones the user sees (as Alt+Tab lists them) whose
  program is the app's. A Store app's window belongs to a host
  (ApplicationFrameHost): the program inside it tells, or its title.
* Where the close button is, the window says itself: WM_NCHITTEST answers
  HTCLOSE over it (Windows asks the same as the mouse moves). A window that
  doesn't say (full screen, or a frame drawn without telling) is asked to
  close as its close button would ask it (WM_SYSCOMMAND, SC_CLOSE).
* Windows 11 draws the Start button inside the taskbar, so UI Automation,
  through PowerShell, finds it; on Windows 10 it's a window of its own.

Linux: wmctrl lists and closes windows; there is no Start button.
"""

from __future__ import annotations

import functools
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import platform

log = logging.getLogger(__name__)

HTCLOSE = 20
# Not app windows, though they're visible and titled: the desktop and the taskbar.
SHELL_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
HOST = "applicationframehost"  # a Store app's window frame
# Whose window has the keyboard while the Start menu is open.
START_PROGRAMS = {"startmenuexperiencehost", "searchhost", "searchapp", "searchui", "shellexperiencehost"}


@dataclass(frozen=True)
class AppWindow:
    handle: int
    title: str
    program: str  # what it belongs to, without ".exe" ("chrome", a Store app's "CalculatorApp")
    hosted: bool = False  # a Store app's, framed by a host


def squash(text: str) -> str:
    """Lower case, letters and digits only: "Google Chrome" → "googlechrome"."""
    return re.sub(r"\W+", "", text.lower())


def program_name(path_or_name: str) -> str:
    """A program's name from its path: "C:\\…\\chrome.exe" → "chrome"."""
    return Path(path_or_name.replace("\\", "/")).stem if path_or_name else ""


def app_windows(program: str, name: str) -> list[AppWindow]:
    """An app's open windows, the frontmost first: those of its ``program``
    if that's known and has any; else those whose program is called
    ``name``, or — a Store app's — whose title has the name in it."""
    windows = open_windows()
    if program:
        mine = [w for w in windows if w.program.lower() == program.lower()]
        if mine:
            return mine
    key = squash(name)
    if not key:
        return []
    return [w for w in windows if squash(w.program) == key or (w.hosted and key in squash(w.title))]


# ---- Windows ---------------------------------------------------------------------------------------


@functools.cache
def _user32():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                                           wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
    user32.SendMessageTimeoutW.restype = wintypes.LPARAM
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetWindow.restype = wintypes.HWND
    user32.GetForegroundWindow.restype = wintypes.HWND
    return ctypes, wintypes, user32


def open_windows() -> list[AppWindow]:
    """The windows the user sees, frontmost first."""
    if not platform.WINDOWS:
        return _wmctrl_windows()
    ctypes, wintypes, user32 = _user32()
    handles: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(handle, _):
        handles.append(handle)
        return True

    user32.EnumWindows(each, 0)
    out = []
    for h in handles:
        if not user32.IsWindowVisible(h) or user32.GetWindow(h, 4) or _cloaked(h):  # 4: GW_OWNER
            continue
        if user32.GetWindowLongW(h, -20) & 0x80 or _class(h) in SHELL_CLASSES:  # WS_EX_TOOLWINDOW
            continue
        title = _title(h)
        if not title:
            continue
        program = _program(h)
        hosted = program.lower() == HOST
        if hosted:
            program = _hosted_program(h) or program
        out.append(AppWindow(int(h), title, program, hosted))
    return out


def close_button(window: AppWindow) -> tuple[int, int] | None:
    """Where the window's close button is, in screen pixels, as the window
    itself says — the middle of the spots near its top-right corner where
    it answers HTCLOSE. None if it doesn't say."""
    if not platform.WINDOWS:
        return None
    left, top, right, bottom = frame(window)
    spots = [(x, y) for x in range(right - 2, max(left, right - 200), -4) for y in range(top + 2, min(bottom, top + 64), 4)
             if _hit_test(window.handle, x, y) == HTCLOSE]
    if not spots:
        return None
    return sum(x for x, _ in spots) // len(spots), sum(y for _, y in spots) // len(spots)


def corner(window: AppWindow) -> tuple[int, int]:
    """Where a close button usually is: by the window's top-right corner."""
    left, top, right, bottom = frame(window)
    scale = 1.0
    if platform.WINDOWS:
        _, _, user32 = _user32()
        scale = (user32.GetDpiForWindow(window.handle) or 96) / 96
    return int(right - 23 * scale), int(top + 15 * scale)


def frame(window: AppWindow) -> tuple[int, int, int, int]:
    """The window's edges on screen, as drawn (left, top, right, bottom)."""
    ctypes, wintypes, user32 = _user32()
    rect = wintypes.RECT()
    if ctypes.windll.dwmapi.DwmGetWindowAttribute(window.handle, 9, ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
        user32.GetWindowRect(window.handle, ctypes.byref(rect))  # 9: DWMWA_EXTENDED_FRAME_BOUNDS
    return rect.left, rect.top, rect.right, rect.bottom


def bring_forward(window: AppWindow) -> None:
    """Out of the taskbar if it's minimised, and in front of the others."""
    if not platform.WINDOWS:
        return
    _, _, user32 = _user32()
    if user32.IsIconic(window.handle):
        user32.ShowWindow(window.handle, 9)  # SW_RESTORE
    user32.SetWindowPos(window.handle, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)  # HWND_TOP; no move, size, activate
    user32.SetForegroundWindow(window.handle)


def shows_at(window: AppWindow, x: float, y: float) -> bool:
    """Whether the window is what's on screen at (x, y) — not covered there."""
    if not platform.WINDOWS:
        return False
    _, wintypes, user32 = _user32()
    found = user32.WindowFromPoint(wintypes.POINT(int(x), int(y)))
    return bool(found) and user32.GetAncestor(found, 2) == window.handle  # GA_ROOT


def ask_to_close(window: AppWindow) -> None:
    """Ask the window to close, as its close button does."""
    if platform.WINDOWS:
        _, _, user32 = _user32()
        user32.PostMessageW(window.handle, 0x0112, 0xF060, 0)  # WM_SYSCOMMAND, SC_CLOSE
    elif shutil.which("wmctrl"):
        subprocess.run(["wmctrl", "-ic", hex(window.handle)], capture_output=True, timeout=5)


def is_open(window: AppWindow) -> bool:
    if not platform.WINDOWS:
        return any(w.handle == window.handle for w in _wmctrl_windows())
    _, _, user32 = _user32()
    return bool(user32.IsWindow(window.handle) and user32.IsWindowVisible(window.handle)) and not _cloaked(window.handle)


def dialog_of(window: AppWindow) -> str | None:
    """The title of a window it has opened in front of itself (a "Save
    changes?" dialog), if any."""
    if not platform.WINDOWS:
        return None
    ctypes, wintypes, user32 = _user32()
    found: list[str] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(handle, _):
        if user32.GetWindow(handle, 4) == window.handle and user32.IsWindowVisible(handle):
            found.append(_title(handle) or "a dialog")
        return True

    user32.EnumWindows(each, 0)
    return found[0] if found else None


def foreground_program() -> str:
    """The program whose window has the keyboard, lower case."""
    if not platform.WINDOWS:
        return ""
    _, _, user32 = _user32()
    h = user32.GetForegroundWindow()
    return _program(h).lower() if h else ""


def start_menu_open() -> bool:
    return foreground_program() in START_PROGRAMS


def start_button() -> tuple[int, int] | None:
    """The middle of the Start button, in screen pixels (None: not found)."""
    if not platform.WINDOWS:
        return None
    ctypes, wintypes, user32 = _user32()
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowExW.restype = wintypes.HWND
    tray = user32.FindWindowW("Shell_TrayWnd", None)
    if not tray:
        return None
    button = user32.FindWindowExW(tray, None, "Start", None)  # Windows 10
    if button:
        rect = wintypes.RECT()
        user32.GetWindowRect(button, ctypes.byref(rect))
        return (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2
    script = (  # Windows 11: inside the taskbar's own drawing
        "Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes; "
        f"$tray = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]{int(tray)}); "
        "$id = New-Object System.Windows.Automation.PropertyCondition("
        "[System.Windows.Automation.AutomationElement]::AutomationIdProperty, 'StartButton'); "
        "$b = $tray.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $id); "
        "if ($b) { $r = $b.Current.BoundingRectangle; '{0} {1} {2} {3}' -f $r.X, $r.Y, $r.Width, $r.Height }")
    try:
        out = subprocess.run(platform.shell_argv(script), capture_output=True, timeout=10,
                             creationflags=platform.quiet_flags()).stdout.decode("utf-8", "replace")
        x, y, w, h = (float(v) for v in out.split()[:4])
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        log.info("couldn't find the Start button: %s", e)
        return None
    return int(x + w / 2), int(y + h / 2)


def _title(handle) -> str:
    ctypes, _, user32 = _user32()
    length = user32.GetWindowTextLengthW(handle)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


def _class(handle) -> str:
    ctypes, _, user32 = _user32()
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(handle, buffer, 256)
    return buffer.value


def _cloaked(handle) -> bool:
    """Hidden by the system though "visible": a suspended Store app, a
    window on another virtual desktop."""
    ctypes, _, _ = _user32()
    value = ctypes.c_int()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(handle, 14, ctypes.byref(value), ctypes.sizeof(value))
    return value.value != 0


def _pid(handle) -> int:
    ctypes, wintypes, user32 = _user32()
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
    return pid.value


def _program(handle) -> str:
    ctypes, wintypes, _ = _user32()
    kernel32 = ctypes.windll.kernel32
    process = kernel32.OpenProcess(0x1000, False, _pid(handle))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not process:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if not kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            return ""
        return program_name(buffer.value)
    finally:
        kernel32.CloseHandle(process)


def _hosted_program(frame_handle) -> str:
    """The Store app inside a host's frame: its child window's program."""
    ctypes, wintypes, user32 = _user32()
    host = _pid(frame_handle)
    found: list[str] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(handle, _):
        if _pid(handle) != host:
            found.append(_program(handle))
            return False
        return True

    user32.EnumChildWindows(frame_handle, each, 0)
    return found[0] if found else ""


def _hit_test(handle, x: int, y: int) -> int | None:
    ctypes, _, user32 = _user32()
    result = ctypes.c_size_t()
    point = ((y & 0xFFFF) << 16) | (x & 0xFFFF)
    ok = user32.SendMessageTimeoutW(handle, 0x0084, 0, point, 0x0002, 200, ctypes.byref(result))  # WM_NCHITTEST
    return result.value if ok else None


# ---- Linux ------------------------------------------------------------------------------------------


def _wmctrl_windows() -> list[AppWindow]:
    if not shutil.which("wmctrl"):
        return []
    try:
        out = subprocess.run(["wmctrl", "-lp"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    windows = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        handle, _, pid, _, title = parts
        try:
            program = Path(f"/proc/{int(pid)}/comm").read_text().strip()
        except (OSError, ValueError):
            program = ""
        windows.append(AppWindow(int(handle, 16), title, program))
    return windows
