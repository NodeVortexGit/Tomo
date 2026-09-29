"""Screenshots for the model to look at.

Taken with what the system has — GDI on Windows (a plain screen copy, which,
unlike most capture libraries' default, includes OpenGL windows), grim /
spectacle / gnome-screenshot / maim / scrot / ImageMagick on Linux,
``screencapture`` on macOS — then scaled down and re-encoded as JPEG so a
4K screen doesn't flood the model's context. Nothing is kept on disk.
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import tempfile
from dataclasses import dataclass

from . import platform

# The longest edge sent to the model (px), and the JPEG quality.
MAX_EDGE = 1920
JPEG_QUALITY = 80


@dataclass
class Screenshot:
    jpeg: bytes
    width: int
    height: int
    # Screen pixels per pixel of this image (to turn the model's click
    # coordinates back into the screen's).
    scale: float


async def capture() -> Screenshot:
    raw = await asyncio.to_thread(grab)
    return await asyncio.to_thread(fit_for_model, raw)


def grab() -> bytes:
    """The whole screen, as an encoded image (PNG/JPEG)."""
    if platform.WINDOWS:
        return _windows_grab()
    path = os.path.join(tempfile.gettempdir(), f"tomo-screen-{os.getpid()}.png")
    if platform.MACOS:
        subprocess.run(["screencapture", "-x", "-t", "png", path], check=True, timeout=15)
        return _read_and_remove(path)
    # (program, arguments, whether it writes the image to stdout or the file)
    tools = [
        ("grim", ["-t", "jpeg", "-q", "80", "-"], True),
        ("spectacle", ["-b", "-n", "-f", "-o", path], False),
        ("gnome-screenshot", ["-f", path], False),
        ("maim", [], True),
        ("scrot", ["-o", path], False),
        ("import", ["-window", "root", "png:-"], True),
    ]
    for program, args, to_stdout in tools:
        if not platform.which(program):
            continue
        try:
            result = subprocess.run([program, *args], capture_output=True, timeout=15, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue  # e.g. grim on X11
        image = result.stdout if to_stdout else _read_and_remove(path)
        if image:
            return image
    raise RuntimeError("couldn't take a screenshot: install grim (Wayland) or maim/scrot (X11)")


def _read_and_remove(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return b""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _windows_grab() -> bytes:
    """The virtual screen (every monitor) through GDI, as PNG."""
    import ctypes
    from ctypes import wintypes as wt

    from PIL import Image

    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    try:
        user32.SetProcessDPIAware()  # a scaled display isn't shrunk
    except Exception:  # noqa: BLE001 - older systems
        pass
    left, top = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
    width, height = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)

    class BitmapInfoHeader(ctypes.Structure):
        _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                    ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                    ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                    ("biClrImportant", wt.DWORD)]

    screen = user32.GetDC(None)
    memory = gdi32.CreateCompatibleDC(screen)
    bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
    try:
        gdi32.SelectObject(memory, bitmap)
        gdi32.BitBlt(memory, 0, 0, width, height, screen, left, top, 0x00CC0020)  # SRCCOPY
        info = BitmapInfoHeader(ctypes.sizeof(BitmapInfoHeader), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
        buffer = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(memory, bitmap, 0, height, buffer, ctypes.byref(info), 0)
    finally:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory)
        user32.ReleaseDC(None, screen)
    image = Image.frombuffer("RGBA", (width, height), buffer.raw, "raw", "BGRA", 0, 1).convert("RGB")
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def fit_for_model(raw: bytes) -> Screenshot:
    """Scale to at most MAX_EDGE and encode as JPEG. A JPEG that already fits
    goes through untouched."""
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    width, height = image.size
    if image.format == "JPEG" and max(width, height) <= MAX_EDGE:
        return Screenshot(raw, width, height, 1.0)
    image = image.convert("RGB")
    if max(width, height) > MAX_EDGE:
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.BILINEAR)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=JPEG_QUALITY)
    return Screenshot(out.getvalue(), image.width, image.height, width / image.width)
