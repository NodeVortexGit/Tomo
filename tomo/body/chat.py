"""The chat and its signature animation.

A small state machine, end to end:

1. The user clicks the character (or says "Hey Tomo").
2. She walks to the right side of the screen.        (``Phase.SLIDING``)
3. A "liquid" blob comes out of her and grows.       (``Phase.LIQUID``)
4. Once fully out, the liquid forms the chat window. (``Phase.OPEN``)
5. Closing reverses it: the window melts back in.    (``Phase.CLOSING``)
   (× button, Escape, clicking the character again, or clicking away.)

The transcript, the text box, the Talk (speech) button and the thinking
indicator all live in the window. Commands Tomo runs are never shown here —
only the conversation. The liquid is a cluster of merging circles easing from
the character into a rounded panel.

Pictures can go with a message: the 📎 button (the desktop's file dialog),
dropping image files on the chat or on her (which opens the chat), or
pasting (Ctrl+V: a screenshot, or image files copied in the file manager).
They wait above the text box — click one to take it off — and go with the
next message; the brain adds what the system knows about each (see
tomo/attachments.py). In the transcript, click a picture to open it.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from imgui_bundle import imgui

from .. import attachments, events, platform
from ..events import CharacterChoice, ChatLine, ImportCharacter, Role, SetVoice, Shutdown, StartVoiceInput, UserMessage
from .movement import Locomotion
from .ui import (ICON_CLOSE, ICON_IMAGE, ICON_MIC, ICON_MUTED, ICON_PAPERCLIP, ICON_POWER, ICON_SEND, ICON_USER,
                 ICON_VOLUME, flags, rgba, vec4)

log = logging.getLogger(__name__)

DOCK_FRACTION = 0.86  # where she parks to open the chat, a share of the width
LIQUID_GROW_SECS = 0.55  # for the liquid to come fully out…
LIQUID_MELT_SECS = 0.40  # …and to melt back
PANEL_W, PANEL_H = 340.0, 460.0  # logical px
MARGIN = 24.0  # from the screen's right edge
TRANSCRIPT_LIMIT = 400  # lines kept on screen; the full history is in the memory
ACCENT = (120, 230, 240)
PENDING_EDGE = 44.0  # px: the pictures waiting to be sent
PICTURE_EDGE = 150.0  # px: the pictures in the transcript
NOTICE_SECS = 4.0


class Phase(Enum):
    CLOSED = 0
    SLIDING = 1
    LIQUID = 2
    OPEN = 3
    CLOSING = 4


def panel_rect(width: float, height: float) -> tuple[float, float, float, float]:
    """Where the panel sits: right side, vertically centred (x0, y0, x1, y1)."""
    x1 = width - MARGIN
    y0 = height * 0.5 - PANEL_H * 0.5
    return x1 - PANEL_W, y0, x1, y0 + PANEL_H


def inside(rect, x: float, y: float) -> bool:
    return rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


@dataclass
class ChatState:
    phase: Phase = Phase.CLOSED
    liquid_t: float = 0.0  # 0 fully retracted … 1 the panel formed
    transcript: list[ChatLine] = field(default_factory=list)
    input: str = ""
    thinking: bool = False
    listening: bool = False  # "Hey Tomo" was heard; the spoken request is being taken
    voice_replies: bool = True
    characters: list[CharacterChoice] = field(default_factory=list)
    focus_input: bool = False
    scroll_to_end: bool = False
    at_end: bool = True  # the transcript is scrolled to its end
    attachments: list[Path] = field(default_factory=list)  # pictures waiting to go with the next message
    # Pictures chosen in the file dialog or pasted, handed over from their threads.
    picked: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)
    notice: str = ""
    notice_until: float = 0.0

    # ---- pictures -------------------------------------------------------------------------------

    def add_images(self, paths) -> int:
        """Queue pictures for the next message: files that exist and are
        pictures, not already queued, up to the limit. Returns how many
        were added (and notes in the window why any weren't)."""
        added, skipped = 0, []
        for path in (Path(p) for p in paths):
            if not attachments.is_image(path) or not path.is_file():
                skipped.append("Only pictures can be sent")
            elif path in self.attachments:
                continue
            elif len(self.attachments) >= attachments.MAX_PER_MESSAGE:
                skipped.append(f"Up to {attachments.MAX_PER_MESSAGE} pictures per message")
            else:
                self.attachments.append(path)
                added += 1
        if skipped:
            self.notice, self.notice_until = skipped[-1], time.monotonic() + NOTICE_SECS
        if added:
            self.focus_input = True
        return added

    def take_picked(self) -> None:
        """The pictures the file dialog or a paste came back with."""
        while True:
            try:
                self.add_images(self.picked.get_nowait())
            except queue.Empty:
                return

    def paste(self, grab: Callable | None = None) -> None:
        """Ctrl+V: a picture on the clipboard, or image files copied in the
        file manager, go with the next message (plain text pastes into the
        text box as usual). Read on a thread: the clipboard can be slow."""
        threading.Thread(target=paste_images, args=(self.picked.put, grab), name="tomo-paste", daemon=True).start()

    def send_message(self, send: Callable) -> bool:
        """Send what's typed, with the pictures waiting. False if there's nothing."""
        text = self.input.strip()
        if not text and not self.attachments:
            return False
        # No local echo: the brain echoes the user's line back (one source of
        # truth for the chat, the memory and the model).
        send(UserMessage(text, tuple(self.attachments)))
        self.input = ""
        self.attachments = []
        return True

    # ---- the brain's side -----------------------------------------------------------------

    def receive(self, event, loco: Locomotion | None) -> None:
        if isinstance(event, events.Characters):
            self.characters = list(event.choices)
        elif isinstance(event, events.Listening):
            self.listening = event.on
            # Hearing "Hey Tomo" makes her hop and opens the chat.
            if event.on and loco is not None:
                loco.jump()
                if self.phase == Phase.CLOSED:
                    self.start_opening(loco)
        elif isinstance(event, events.Chat):
            self.transcript.append(event.line)
            del self.transcript[:-TRANSCRIPT_LIMIT]
            self.scroll_to_end = True
        elif isinstance(event, events.Thinking):
            self.thinking = event.on
            self.scroll_to_end = True

    # ---- the user's side ------------------------------------------------------------------------

    def character_clicked(self, loco: Locomotion) -> None:
        """Clicking her opens the chat; clicking her again while it's open
        closes it. (Dragging her is physics, not a click.)"""
        if self.phase == Phase.CLOSED:
            self.start_opening(loco)
        elif self.phase == Phase.OPEN:
            self.phase = Phase.CLOSING

    def start_opening(self, loco: Locomotion) -> None:
        """The first step: walk to the dock (the liquid grows once there)."""
        self.phase = Phase.SLIDING
        loco.held = True
        loco.walk_to(DOCK_FRACTION)

    def dismiss(self, escape: bool, clicked_at: tuple[float, float] | None, character_rect, width: float,
                height: float) -> None:
        """Close on Escape, or on a click away from both the chat and her."""
        if self.phase != Phase.OPEN:
            return
        away = clicked_at is not None and not inside(character_rect, *clicked_at) and not inside(
            panel_rect(width, height), *clicked_at)
        if escape or away:
            self.phase = Phase.CLOSING

    def advance(self, dt: float, loco: Locomotion | None) -> bool:
        """Move the animation on. True while it's moving."""
        # A character swapped in while the chat is out comes over to it too.
        if self.phase != Phase.CLOSED and loco is not None and not loco.held:
            loco.held = True
            loco.walk_to(DOCK_FRACTION)
        if self.phase == Phase.SLIDING:
            if loco is None or loco.is_idle():
                self.phase, self.liquid_t = Phase.LIQUID, 0.0
        elif self.phase == Phase.LIQUID:
            self.liquid_t += dt / LIQUID_GROW_SECS
            if self.liquid_t >= 1.0:
                self.liquid_t, self.phase = 1.0, Phase.OPEN
                self.focus_input = self.scroll_to_end = True
        elif self.phase == Phase.CLOSING:
            self.liquid_t -= dt / LIQUID_MELT_SECS
            if self.liquid_t <= 0.0:
                self.liquid_t, self.phase = 0.0, Phase.CLOSED
                if loco is not None:
                    loco.held = False  # back to wandering
        return self.phase in (Phase.SLIDING, Phase.LIQUID, Phase.CLOSING)

    def shown(self) -> bool:
        """The panel (or the liquid) is on screen, taking the pointer."""
        return self.phase in (Phase.LIQUID, Phase.OPEN, Phase.CLOSING)

    # ---- drawing ------------------------------------------------------------------------------

    def draw(self, ui, width: float, height: float, source: tuple[float, float] | None, showing: Path | None,
             send: Callable) -> bool:
        """The liquid and, once formed, the window. True: the user quit."""
        if self.phase == Phase.CLOSED:
            return False
        t = ease_out_cubic(self.liquid_t)
        start = source or (width * DOCK_FRACTION, height * 0.5)
        draw_liquid(start, t, width, height)
        if self.phase == Phase.OPEN or self.liquid_t > 0.85:
            close, quit_ = self.draw_window(ui, t, width, height, showing, send)
            if close:
                self.phase = Phase.CLOSING
            if quit_:
                send(Shutdown())
                return True
        return False

    def draw_window(self, ui, t: float, width: float, height: float, showing: Path | None,
                    send: Callable) -> tuple[bool, bool]:
        x0, y0, _, _ = panel_rect(width, height)
        imgui.set_next_window_pos(imgui.ImVec2(x0, y0))
        imgui.set_next_window_size(imgui.ImVec2(PANEL_W, PANEL_H))
        imgui.push_style_color(imgui.Col_.window_bg.value, vec4(20, 28, 34, 245 * t))
        imgui.push_style_var(imgui.StyleVar_.window_padding.value, imgui.ImVec2(14.0, 14.0))
        w = imgui.WindowFlags_
        imgui.begin("tomo-chat", None, flags(w.no_decoration, w.no_move, w.no_saved_settings, w.no_scroll_with_mouse))
        close = quit_ = False

        # The header: her name, then (from the right) close, quit, voice, character.
        imgui.push_font(ui.bold, 18.0)
        imgui.text_colored(vec4(*ACCENT), "Tomo")
        imgui.pop_font()
        buttons = 4
        size = imgui.get_frame_height()
        spacing = imgui.get_style().item_spacing.x
        imgui.same_line(imgui.get_window_width() - 14.0 - buttons * size - (buttons - 1) * spacing)
        if icon_button(ICON_USER, "Change character", size):
            imgui.open_popup("characters")
        if imgui.begin_popup("characters"):
            self.character_menu(showing, send)
            imgui.end_popup()
        imgui.same_line()
        voice_icon = ICON_VOLUME if self.voice_replies else ICON_MUTED
        if icon_button(voice_icon, "Speak replies aloud: " + ("on" if self.voice_replies else "off"), size,
                       active=self.voice_replies):
            self.voice_replies = not self.voice_replies
            send(SetVoice(self.voice_replies))
        imgui.same_line()
        quit_ = icon_button(ICON_POWER, "Quit Tomo", size)
        imgui.same_line()
        close = icon_button(ICON_CLOSE, "Close chat", size)
        imgui.separator()

        # The transcript, above the pictures waiting to go and the input row.
        self.take_picked()
        spacing_y = imgui.get_style().item_spacing.y
        noticing = bool(self.notice) and time.monotonic() < self.notice_until
        row = imgui.get_frame_height() + spacing_y * 2 + 2
        if self.attachments:
            row += PENDING_EDGE + spacing_y
        if noticing:
            row += imgui.get_text_line_height() + spacing_y
        imgui.begin_child("transcript", imgui.ImVec2(0.0, -row))
        for line in self.transcript:
            bubble(line, ui.thumbnails)
        if self.listening:
            imgui.text_colored(vec4(*ACCENT), "Listening…")
        if self.thinking:
            imgui.text_colored(vec4(150, 150, 150), "Tomo is thinking…")
        if self.scroll_to_end or self.at_end:
            imgui.set_scroll_here_y(1.0)
            self.scroll_to_end = False
        # Stay at the bottom as lines arrive and pictures load, unless the
        # user has scrolled up to read.
        self.at_end = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 2.0
        imgui.end_child()
        imgui.separator()
        if noticing:
            imgui.text_colored(vec4(150, 150, 150), self.notice)
        if self.attachments:
            self.draw_pending(ui.thumbnails)

        # The input row: pictures, the text box, Send, Talk.
        pad = imgui.get_style().frame_padding.x * 2
        send_w = imgui.calc_text_size(f"{ICON_SEND} Send").x + pad
        talk_w = imgui.calc_text_size(f"{ICON_MIC} Talk").x + pad
        if icon_button(ICON_PAPERCLIP, "Send pictures (or drop them here, or paste one with Ctrl+V)", size):
            pick_images(self.picked.put)
        imgui.same_line()
        imgui.set_next_item_width(imgui.get_content_region_avail().x - send_w - talk_w - 2 * spacing)
        if self.focus_input:
            imgui.set_keyboard_focus_here()
            self.focus_input = False
        hint = "Say something about them…" if self.attachments else "Say something…"
        entered, self.input = imgui.input_text_with_hint("##say", hint, self.input,
                                                         flags(imgui.InputTextFlags_.enter_returns_true))
        imgui.same_line()
        clicked = imgui.button(f"{ICON_SEND} Send")
        if entered or clicked:
            self.send_message(send)
            imgui.set_keyboard_focus_here(-1)
        imgui.same_line()
        if imgui.button(f"{ICON_MIC} Talk"):
            send(StartVoiceInput())
        if imgui.is_item_hovered():
            imgui.set_tooltip('Push to talk (or just say "Hey Tomo")')
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_style_color()
        return close, quit_

    def draw_pending(self, thumbnails) -> None:
        """The pictures waiting to go with the next message, small; clicking
        one takes it off."""
        imgui.push_style_var(imgui.StyleVar_.frame_padding.value, imgui.ImVec2(0.0, 0.0))
        remove = None
        for i, path in enumerate(self.attachments):
            if i:
                imgui.same_line()
            got = thumbnails.get(path)
            if got is not None and got[0] is not None:
                texture, w, h = got
                s = PENDING_EDGE / max(w, h)
                clicked = imgui.image_button(f"pending-{i}", texture, imgui.ImVec2(w * s, h * s))
            else:  # still being read, or unreadable
                clicked = imgui.button(f"{ICON_IMAGE}##pending-{i}", imgui.ImVec2(PENDING_EDGE, PENDING_EDGE))
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"{path.name} — click to take it off")
            if clicked:
                remove = i
        imgui.pop_style_var()
        if remove is not None:
            del self.attachments[remove]

    def character_menu(self, showing: Path | None, send: Callable) -> None:
        """The characters on offer (the one showing ticked), and importing a
        new .vrm."""
        for choice in self.characters:
            current = showing is not None and Path(choice.path) == Path(showing)
            clicked, _ = imgui.selectable(choice.name, current)
            if clicked and not current:
                send(ImportCharacter(Path(choice.path), choice.name))
        if self.characters:
            imgui.separator()
        clicked, _ = imgui.selectable("Import a .vrm…", False)
        if clicked:
            pick_character_file(send)


def icon_button(icon: str, tip: str, size: float, active: bool = True) -> bool:
    if not active:
        imgui.push_style_color(imgui.Col_.text.value, vec4(120, 120, 120))
    clicked = imgui.button(icon, imgui.ImVec2(size, size))
    if not active:
        imgui.pop_style_color()
    if imgui.is_item_hovered():
        imgui.set_tooltip(tip)
    return clicked


BUBBLES = {Role.USER: ((70, 130, 180), "You"), Role.ASSISTANT: ((38, 66, 74), "Tomo"), Role.SYSTEM: ((60, 60, 60), "•")}


def bubble(line: ChatLine, thumbnails=None) -> None:
    """One line of the conversation: the user's on the right, Tomo's on the
    left, notes in the middle — its pictures, if any, above its words."""
    color, who = BUBBLES.get(line.role, BUBBLES[Role.SYSTEM])
    avail = imgui.get_content_region_avail().x
    pad_x, pad_y = 10.0, 6.0

    def place(item_w: float) -> None:
        x = imgui.get_cursor_pos_x()
        if line.role == Role.USER:
            imgui.set_cursor_pos_x(x + avail - item_w)
        elif line.role == Role.SYSTEM:
            imgui.set_cursor_pos_x(x + (avail - item_w) * 0.5)

    if line.attachments and thumbnails is not None:
        pictures(line, thumbnails, place, avail * 0.82)
    if line.text:
        size = imgui.calc_text_size(line.text, None, False, avail * 0.82 - 2 * pad_x)
        w, h = size.x + 2 * pad_x, size.y + 2 * pad_y
        place(w)
        top = imgui.get_cursor_screen_pos()
        imgui.get_window_draw_list().add_rect_filled(top, imgui.ImVec2(top.x + w, top.y + h), rgba(*color), 12.0)
        imgui.set_cursor_screen_pos(imgui.ImVec2(top.x + pad_x, top.y + pad_y))
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + size.x)
        imgui.text_colored(vec4(255, 255, 255), line.text)
        imgui.pop_text_wrap_pos()
        imgui.set_cursor_screen_pos(imgui.ImVec2(top.x, top.y + h + 2.0))
        imgui.dummy(imgui.ImVec2(0.0, 0.0))
    label = imgui.calc_text_size(who).x
    place(label)
    imgui.text_colored(vec4(120, 120, 120), who)
    imgui.dummy(imgui.ImVec2(0.0, 4.0))


def pictures(line: ChatLine, thumbnails, place: Callable[[float], None], widest: float) -> None:
    """A message's pictures in a row (shrunk to fit ``widest``); clicking
    one opens it in the system's viewer."""
    sizes = []
    for attachment in line.attachments:
        got = thumbnails.get(attachment.path)
        if got is None or got[0] is None:  # still being read, or gone
            sizes.append((None, PICTURE_EDGE * 0.6, PICTURE_EDGE * 0.6))
        else:
            texture, w, h = got
            s = min(PICTURE_EDGE / w, PICTURE_EDGE / h)
            sizes.append((texture, w * s, h * s))
    spacing = imgui.get_style().item_spacing.x
    total = sum(w for _, w, _ in sizes) + spacing * (len(sizes) - 1)
    fit = min(1.0, widest / total) if total > 0 else 1.0
    place(total * fit)
    imgui.push_style_var(imgui.StyleVar_.frame_padding.value, imgui.ImVec2(0.0, 0.0))
    imgui.push_id(str(id(line)))
    for i, (attachment, (texture, w, h)) in enumerate(zip(line.attachments, sizes)):
        if i:
            imgui.same_line()
        box = imgui.ImVec2(w * fit, h * fit)
        if texture is None:
            clicked = imgui.button(f"{ICON_IMAGE}##picture-{i}", box)
        else:
            clicked = imgui.image_button(f"picture-{i}", texture, box)
        if imgui.is_item_hovered():
            name = Path(attachment.source).name if attachment.source else "A pasted picture"
            imgui.set_tooltip(f"{name} — click to open")
        if clicked:
            open_file(attachment.path)
    imgui.pop_id()
    imgui.pop_style_var()


def draw_liquid(start: tuple[float, float], t: float, width: float, height: float) -> None:
    """A blob of merging circles that eases from the character into the
    panel's shape."""
    painter = imgui.get_background_draw_list()
    x0, y0, x1, y1 = panel_rect(width, height)
    target = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
    cx, cy = start[0] + (target[0] - start[0]) * t, start[1] + (target[1] - start[1]) * t
    fill = rgba(64, 196, 208, int(220 * t))  # a soft, glossy teal that reads on any wallpaper
    spread, r = 8.0 + 150.0 * t, 14.0 + 60.0 * t
    for i in range(5):
        f = i / 4.0 - 0.5
        painter.add_circle_filled(imgui.ImVec2(cx + f * spread * 0.4, cy + f * spread), r, fill)
    if t > 0.6:  # nearly formed: a rounded rectangle, so the edge is clean under the window
        a = max(0.0, min(1.0, (t - 0.6) / 0.4))
        painter.add_rect_filled(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), rgba(20, 28, 34, int(235 * a)), 24.0)


def pick_character_file(send: Callable) -> None:
    """Ask for a .vrm with the desktop's file dialog and switch to it."""
    pick_files("Choose a character (.vrm)", "VRM models", (".vrm",), False,
               lambda paths: send(ImportCharacter(paths[0], paths[0].stem)))


def pick_images(picked: Callable[[list[Path]], None]) -> None:
    """Ask for pictures to send with the desktop's file dialog."""
    pick_files("Send pictures to Tomo", "Pictures", attachments.IMAGE_TYPES, True, picked)


def pick_files(title: str, kind: str, extensions: tuple[str, ...], multiple: bool,
               picked: Callable[[list[Path]], None]) -> None:
    """The desktop's file dialog, off the drawing thread: ``picked`` gets the
    files chosen (and isn't called if none were)."""

    def ask() -> None:
        if platform.WINDOWS:
            patterns = ";".join(f"*{e}" for e in extensions)
            script = (f"Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms."
                      f"OpenFileDialog; $d.Title = '{title}'; $d.Filter = '{kind}|{patterns}'; "
                      f"$d.Multiselect = ${'true' if multiple else 'false'}; "
                      f"if ($d.ShowDialog() -eq 'OK') {{ $d.FileNames }}")
            dialogs = [["powershell.exe", "-NoProfile", "-STA", "-Command",
                        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; " + script]]
        else:
            patterns = " ".join(f"*{e} *{e.upper()}" for e in extensions)
            zenity = ["zenity", "--file-selection", "--title", title, f"--file-filter={kind} | {patterns}"]
            kdialog = ["kdialog", "--title", title, "--getopenfilename", ".", f"{patterns}|{kind}"]
            if multiple:
                zenity += ["--multiple", "--separator=\n"]
                kdialog += ["--multiple", "--separate-output"]
            dialogs = [zenity, kdialog]
        for argv in dialogs:
            try:
                out = subprocess.run(argv, capture_output=True, creationflags=platform.quiet_flags())
            except OSError:
                continue  # not installed: try the next
            if out.returncode == 0:
                lines = out.stdout.decode("utf-8", "replace").splitlines()
                paths = [Path(line.strip()) for line in lines if line.strip()]
                if chosen := [p for p in paths if p.is_file()]:
                    picked(chosen)
            return
        log.warning("no file dialog to pick files with (on Linux, install zenity or kdialog)")

    threading.Thread(target=ask, name="tomo-pick-files", daemon=True).start()


def paste_images(put: Callable[[list[Path]], None], grab: Callable | None = None) -> None:
    """What's on the clipboard, if it's pictures: a picture itself (kept in
    a temporary file until the brain has its copy) or image files copied in
    the file manager. Nothing happens for anything else."""
    try:
        if grab is None:
            from PIL import ImageGrab

            grab = ImageGrab.grabclipboard
        got = grab()
    except Exception as e:  # noqa: BLE001 - no clipboard tool here (Linux: wl-paste/xclip), or nothing usable
        log.debug("nothing to paste as a picture: %s", e)
        return
    if isinstance(got, (list, tuple)):
        if found := [Path(p) for p in got if attachments.is_image(p)]:
            put(found)
    elif got is not None and hasattr(got, "save"):
        folder = attachments.pasted_dir()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"pasted-{time.time_ns()}.png"
        got.save(path, "PNG")
        put([path])


def open_file(path: str) -> None:
    """Open a file with the system's own app for it (a picture: its viewer)."""
    try:
        if platform.WINDOWS:
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["open" if platform.MACOS else "xdg-open", path], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        log.warning("couldn't open %s: %s", path, e)
