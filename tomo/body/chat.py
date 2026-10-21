"""The chat and its signature animation.

A small state machine, end to end:

1. The user clicks the character (or says "Hey Tomo").
2. She walks right, stopping beside where the        (``Phase.SLIDING``)
   window will be — clear of it, so it never covers her (``dock_x``).
3. A "liquid" blob comes out of her and grows.       (``Phase.LIQUID``)
4. Once fully out, the liquid forms the chat window. (``Phase.OPEN``)
5. Closing reverses it: the window melts back in.    (``Phase.CLOSING``)
   (× button, Escape, clicking the character again, or clicking away.)

The transcript, the text box, the Talk (speech) button and the thinking
indicator all live in the window. Commands Tomo runs are never shown here —
only the conversation. The liquid is a cluster of merging circles easing from
the character into a rounded panel.

Beside her name, the chats: a new one (the brain empties the window — what
Tomo knows about the user carries over), and the earlier ones, the latest
first, to read again or carry on. On the right, by the voice and quit
buttons: hiding her in the corner (which closes the chat too) or bringing
her out.

Pictures can go with a message: the 📎 button (the desktop's file dialog),
dropping image files on the chat or on her (which opens the chat), or
pasting (Ctrl+V: a screenshot, or image files copied in the file manager).
They wait above the text box — click one to take it off — and go with the
next message; the brain adds what the system knows about each (see
tomo/attachments.py). In the transcript, click a picture to open it.

Any of the conversation's text can be selected and copied, as in a browser:
drag across it (from one message into the next, too), double-click for a
word, triple-click for a paragraph, Shift+click to extend, Ctrl+A for all of
it, Ctrl+C to copy. Right-click for the same as a menu (Copy, Copy message,
Select all, Copy the whole chat); and while the pointer is on a message, a
copy button by its name copies that message. The words are laid out here
(:func:`wrap`), not by ImGui, so a point on screen can be turned into a
place in the text.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, NamedTuple

from imgui_bundle import imgui

from .. import attachments, events, platform
from ..events import (CharacterChoice, ChatInfo, ChatLine, ImportCharacter, NewChat, OpenChat, Role, SetHidden, SetVoice,
                      Shutdown, StartVoiceInput, UserMessage)
from .movement import Locomotion, body_size
from .ui import (ICON_CHECK, ICON_CLOSE, ICON_COPY, ICON_HIDE, ICON_HISTORY, ICON_IMAGE, ICON_MIC, ICON_MUTED,
                 ICON_NEW_CHAT, ICON_PAPERCLIP, ICON_POWER, ICON_SEND, ICON_SHOW, ICON_USER, ICON_VOLUME, flags, rgba,
                 vec4)

log = logging.getLogger(__name__)

DOCK_GAP = 24.0  # logical px between her body and the panel while the chat is out
LIQUID_GROW_SECS = 0.55  # for the liquid to come fully out…
LIQUID_MELT_SECS = 0.40  # …and to melt back
PANEL_W, PANEL_H = 340.0, 460.0  # logical px
MARGIN = 24.0  # from the screen's right edge
TRANSCRIPT_LIMIT = 400  # lines kept on screen; the full history is in the memory
ACCENT = (120, 230, 240)
PENDING_EDGE = 44.0  # px: the pictures waiting to be sent
PICTURE_EDGE = 150.0  # px: the pictures in the transcript
NOTICE_SECS = 4.0
SELECTED = (255, 255, 255, 70)  # the selection's highlight, over any bubble's colour
LINE_END_W = 4.0  # px of highlight past a row's end where the selection runs on
COPY_BOX = 16.0  # px: the copy button by a message's name
COPIED_SECS = 1.5  # the "copied" tick or note
SCROLL_SPEED = 10.0  # selecting past the transcript's edge: px a second per px beyond it
CHATS_W = PANEL_W - 44.0  # px: the list of earlier chats…
CHATS_SHOWN = 10  # …and how many rows of it show at once
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


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


def dock_x(width: float, half_body: float) -> float:
    """Where she stands while the chat is out (her feet's x, logical px):
    left of the panel, with her body's half-width and DOCK_GAP between her
    and its edge, so the window never covers her."""
    return max(half_body, panel_rect(width, 0.0)[0] - DOCK_GAP - half_body)


def dock_fraction(loco: Locomotion) -> float:
    """:func:`dock_x` as the share of the width that ``walk_to`` takes."""
    width = loco.arena.x
    if width <= 0.0:  # not on screen yet
        return 0.5
    return dock_x(width, body_size(loco.height).x) / width


def inside(rect, x: float, y: float) -> bool:
    return rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


# ---- the conversation's text: laid out, found under the pointer, selected -------------------------


@dataclass(frozen=True)
class Row:
    """One drawn row of a message: ``text[start:end]``. ``edges`` holds the x
    of each character boundary from the row's left (one more than its
    characters); ``width`` is how far its visible part reaches (spaces at
    the end don't count)."""

    start: int
    end: int
    edges: tuple[float, ...]
    width: float


def wrap(text: str, width: float, advance: Callable[[str], float]) -> list[Row]:
    """Break ``text`` into rows no wider than ``width``: at its line breaks,
    after a space where a row is full, inside a word only when the word alone
    is wider. ``advance`` is a character's width in the font."""
    rows: list[Row] = []
    start = 0
    for paragraph in text.split("\n"):
        stop = start + len(paragraph)
        while True:
            x = visible = 0.0
            edges = [0.0]
            cut = None  # just after the row's last space so far, and its visible width there
            i = start
            while i < stop:
                w = advance(text[i])
                if text[i] != " " and x + w > width and i > start:
                    break
                x += w
                edges.append(x)
                i += 1
                if text[i - 1] == " ":
                    cut = (i, visible)
                else:
                    visible = x
            if i < stop and cut is not None:
                i, visible = cut
                del edges[i - start + 1:]
            rows.append(Row(start, i, tuple(edges), visible))
            if i >= stop:
                break
            start = i
        start = stop + 1
    return rows


def caret_at(row: Row, x: float) -> int:
    """The character boundary on ``row`` nearest ``x`` (from the row's left)."""
    k = bisect.bisect_left(row.edges, x)
    if k == 0:
        return row.start
    if k == len(row.edges):
        return row.end
    return row.start + (k if row.edges[k] - x < x - row.edges[k - 1] else k - 1)


def char_at(row: Row, x: float) -> int:
    """The character on ``row`` under ``x`` (beyond its ends: the first or last one)."""
    k = bisect.bisect_right(row.edges, x) - 1
    return row.start + max(0, min(k, row.end - row.start - 1))


TextPos = tuple[int, int]  # a place in the transcript: (line, character of its text)


class Placed(NamedTuple):
    """A message's words as drawn this frame: its line in the transcript,
    where its first row starts on screen, and its rows."""

    line: int
    x: float
    y: float
    rows: list[Row]


def hit(placed: list[Placed], x: float, y: float, row_h: float) -> tuple[TextPos, int] | None:
    """The place in the words drawn this frame nearest the point (x, y): the
    character boundary, as a position, and the character under the point.
    Above all of them is their start; below, their end."""
    if not placed:
        return None
    first, last = placed[0], placed[-1]
    if y < first.y:
        return (first.line, 0), 0
    if y >= last.y + len(last.rows) * row_h:
        end = last.rows[-1].end
        return (last.line, end), max(0, end - 1)
    rows = ((words, row, words.y + k * row_h) for words in placed for k, row in enumerate(words.rows))
    words, row, _ = min(rows, key=lambda r: max(r[2] - y, y - r[2] - row_h, 0.0))
    return (words.line, caret_at(row, x - words.x)), char_at(row, x - words.x)


def highlight(row: Row, a: int, b: int, more: bool, length: int) -> tuple[float, float] | None:
    """The stretch of ``row`` (x from, to) to paint when characters ``a``–``b``
    of its message (``length`` long) are selected; ``more``: the selection
    goes on into the next message. Where it carries on past the row's end,
    as across a line break, the stretch reaches a little beyond it."""
    lo, hi = max(a, row.start), min(b, row.end)
    on = a <= row.end and (b > row.end or (more and row.end == length))
    if lo > hi or (lo == hi and not on):
        return None
    x0 = min(row.edges[lo - row.start], row.width)
    x1 = min(row.edges[hi - row.start], row.width) + (LINE_END_W if on else 0.0)
    return x0, x1


@dataclass
class Selection:
    """What's selected in the transcript: from ``anchor`` (where the drag
    began) to ``caret`` (where the pointer is now), in either order. A
    double- or triple-click selects a ``unit`` — a word, a paragraph — that a
    drag then grows from, keeping it whole."""

    anchor: TextPos | None = None
    caret: TextPos | None = None
    unit: tuple[TextPos, TextPos] | None = None
    dragging: bool = False

    def span(self) -> tuple[TextPos, TextPos] | None:
        """Its start and end, in order; None while nothing is selected."""
        if self.anchor is None or self.caret is None or self.anchor == self.caret:
            return None
        return min(self.anchor, self.caret), max(self.anchor, self.caret)

    def within(self, line: int, length: int) -> tuple[int, int, bool] | None:
        """The selected characters of ``line`` (``length`` long): from, to,
        and whether the selection goes on into the next line."""
        span = self.span()
        if span is None or not span[0][0] <= line <= span[1][0]:
            return None
        (first, a), (last, b) = span
        return (a if line == first else 0), (min(b, length) if line == last else length), line < last

    def place(self, pos: TextPos) -> None:
        """A plain click: the caret there, nothing selected."""
        self.anchor = self.caret = pos
        self.unit = None

    def select(self, start: TextPos, end: TextPos) -> None:
        """Select from ``start`` to ``end``, as one unit."""
        self.anchor, self.caret, self.unit = start, end, (start, end)

    def drag_to(self, pos: TextPos) -> None:
        """The pointer has moved to ``pos`` with the button held."""
        if self.unit is None:
            self.caret = pos
        else:
            lo, hi = self.unit
            self.anchor, self.caret = (hi, pos) if pos < lo else (lo, max(pos, hi))

    def clear(self) -> None:
        self.anchor = self.caret = self.unit = None
        self.dragging = False

    def lines_dropped(self, count: int) -> None:
        """``count`` lines left the top of the transcript: stay on the same
        words, or let go if they've gone."""

        def up(pos: TextPos | None) -> TextPos | None:
            return None if pos is None else (pos[0] - count, pos[1])

        anchor, caret = up(self.anchor), up(self.caret)
        if (anchor is not None and anchor[0] < 0) or (caret is not None and caret[0] < 0):
            self.clear()
            return
        self.anchor, self.caret = anchor, caret
        self.unit = None if self.unit is None else (up(self.unit[0]), up(self.unit[1]))


def char_kind(text: str, i: int) -> int:
    """0: part of a word (letters, digits, an apostrophe inside one), 1: a
    space, 2: anything else."""
    ch = text[i]
    if ch.isalnum() or ch == "_" or (ch in "'’" and 0 < i < len(text) - 1 and text[i - 1].isalnum()
                                    and text[i + 1].isalnum()):
        return 0
    return 1 if ch.isspace() else 2


def word_at(text: str, i: int) -> tuple[int, int]:
    """The word around character ``i`` ("don't", "Томо", "C3PO") — or the
    run of spaces, or of other marks, there."""
    if not text:
        return 0, 0
    i = max(0, min(i, len(text) - 1))
    kind = char_kind(text, i)
    a, b = i, i + 1
    while a > 0 and char_kind(text, a - 1) == kind:
        a -= 1
    while b < len(text) and char_kind(text, b) == kind:
        b += 1
    return a, b


def paragraph_at(text: str, i: int) -> tuple[int, int]:
    """The paragraph (between line breaks) around character ``i``."""
    end = text.find("\n", i)
    return text.rfind("\n", 0, i) + 1, len(text) if end < 0 else end


def selected_text(transcript: list[ChatLine], start: TextPos, end: TextPos) -> str:
    """The words from ``start`` to ``end``: each message's own line breaks,
    and a blank line between messages."""
    parts = []
    for line in range(start[0], end[0] + 1):
        text = transcript[line].text
        if part := text[start[1] if line == start[0] else 0:end[1] if line == end[0] else len(text)]:
            parts.append(part)
    return "\n\n".join(parts)


def chat_text(transcript: list[ChatLine]) -> str:
    """The whole conversation as plain text, each message after who said it
    (and a picture as its file's name)."""
    names = {Role.USER: "You", Role.ASSISTANT: "Tomo"}
    out = []
    for line in transcript:
        pictures = [f"[picture: {Path(a.source).name if a.source else 'pasted'}]" for a in line.attachments]
        body = "\n".join([*pictures, line.text] if line.text else pictures)
        out.append(f"{names.get(line.role, 'Note')}: {body}")
    return "\n\n".join(out)


@dataclass
class ChatState:
    phase: Phase = Phase.CLOSED
    liquid_t: float = 0.0  # 0 fully retracted … 1 the panel formed
    transcript: list[ChatLine] = field(default_factory=list)
    chats: list[ChatInfo] = field(default_factory=list)  # the history, the latest first
    current_chat: str = ""  # the one showing
    hidden: bool = False  # she's set to hide in the corner
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
    # Selecting and copying the conversation's words.
    selection: Selection = field(default_factory=Selection)
    placed: list[Placed] = field(default_factory=list)  # this frame's words, to find the pointer in
    hovered: int | None = None  # the line under the pointer this frame
    menu_line: ChatLine | None = None  # the line right-clicked for the menu
    copied_line: int | None = None  # whose copy button has just copied (a tick for a moment)
    copied_until: float = 0.0
    layouts: dict = field(default_factory=dict)  # (text, width) → its rows
    advances: dict = field(default_factory=dict)  # character → width, in the font measured with
    font: tuple = ()  # that font: (size, density)

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
            if (extra := len(self.transcript) - TRANSCRIPT_LIMIT) > 0:
                del self.transcript[:extra]
                self.selection.lines_dropped(extra)
            self.scroll_to_end = True
        elif isinstance(event, events.Transcript):  # another chat (a new one: empty)
            self.transcript = list(event.lines[-TRANSCRIPT_LIMIT:])
            self.selection.clear()
            self.copied_line = None
            self.scroll_to_end = True
        elif isinstance(event, events.ChatList):
            self.chats, self.current_chat = list(event.chats), event.current
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
        """The first step: walk to the dock, beside where the panel will be
        (the liquid grows once she's there)."""
        self.phase = Phase.SLIDING
        loco.held = True
        loco.walk_to(dock_fraction(loco))

    def keep_clear(self, position: float, loco: Locomotion) -> float:
        """Where she may walk (a share of the width) when asked to: while the
        chat is out, no further right than the dock, so it never covers her."""
        return position if self.phase == Phase.CLOSED else min(position, dock_fraction(loco))

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
            loco.walk_to(dock_fraction(loco))
        if self.phase != Phase.OPEN:
            self.selection.dragging = False  # a drag doesn't outlive the window
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
        start = source or (panel_rect(width, height)[0] - DOCK_GAP, height * 0.5)
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

        # The header: her name and the chats (a new one, the earlier ones);
        # then (from the right) close, quit, voice, hiding, character.
        imgui.push_font(ui.bold, 18.0)
        imgui.text_colored(vec4(*ACCENT), "Tomo")
        imgui.pop_font()
        size = imgui.get_frame_height()
        spacing = imgui.get_style().item_spacing.x
        imgui.same_line()
        if icon_button(ICON_NEW_CHAT, "New chat", size):
            self.new_chat(send)
        imgui.same_line()
        if icon_button(ICON_HISTORY, "Earlier chats", size):
            imgui.open_popup("chats")
        # Under the header, inside the panel.
        imgui.set_next_window_pos(imgui.ImVec2(x0 + 8.0, imgui.get_item_rect_max().y + 6.0), imgui.Cond_.appearing.value)
        if imgui.begin_popup("chats"):
            self.chat_menu(send)
            imgui.end_popup()
        buttons = 5
        imgui.same_line(imgui.get_window_width() - 14.0 - buttons * size - (buttons - 1) * spacing)
        if icon_button(ICON_USER, "Change character", size):
            imgui.open_popup("characters")
        if imgui.begin_popup("characters"):
            self.character_menu(showing, send)
            imgui.end_popup()
        imgui.same_line()
        if icon_button(ICON_SHOW if self.hidden else ICON_HIDE,
                       "Come out of the corner" if self.hidden else "Hide in the corner", size, key="hide"):
            send(SetHidden(not self.hidden))
            if not self.hidden:
                close = True  # out of the way: the chat too
        imgui.same_line()
        voice_icon = ICON_VOLUME if self.voice_replies else ICON_MUTED
        if icon_button(voice_icon, "Speak replies aloud: " + ("on" if self.voice_replies else "off"), size,
                       active=self.voice_replies, key="voice"):
            self.voice_replies = not self.voice_replies
            send(SetVoice(self.voice_replies))
        imgui.same_line()
        quit_ = icon_button(ICON_POWER, "Quit Tomo", size)
        imgui.same_line()
        close = icon_button(ICON_CLOSE, "Close chat", size) or close
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
        font = imgui.get_font_baked()
        self.placed, self.hovered = [], None
        for index, line in enumerate(self.transcript):
            self.bubble(index, line, ui.thumbnails, font)
        if not self.transcript and not self.thinking:
            new_chat_note()
        if self.listening:
            imgui.text_colored(vec4(*ACCENT), "Listening…")
        if self.thinking:
            imgui.text_colored(vec4(150, 150, 150), "Tomo is thinking…")
        self.select_text()
        # Not while words are being selected: that scrolls by itself.
        if (self.scroll_to_end or self.at_end) and not self.selection.dragging:
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
                clicked = imgui.button(f"##pending-{i}", imgui.ImVec2(PENDING_EDGE, PENDING_EDGE))
                centred_icon(ICON_IMAGE, imgui.get_item_rect_min(), imgui.get_item_rect_max(),
                             imgui.get_color_u32(imgui.Col_.text.value))
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

    # ---- the chats -----------------------------------------------------------------------------

    def new_chat(self, send: Callable) -> None:
        """Start a new chat (the brain empties the window). During a reply,
        once it's in."""
        send(NewChat())
        if self.thinking:
            self.notice, self.notice_until = "A new chat starts once Tomo has answered", time.monotonic() + NOTICE_SECS
        self.focus_input = True

    def chat_menu(self, send: Callable) -> None:
        """The chats there are: a new one, then each one before, the latest
        first, with when it was last used (the one showing ticked)."""
        clicked, _ = imgui.selectable(f"{ICON_NEW_CHAT}  New chat", False)
        if clicked:
            self.new_chat(send)
        imgui.separator()
        if not self.chats:
            imgui.text_colored(vec4(150, 150, 150), "No chats yet")
            return
        row = imgui.get_text_line_height() + imgui.get_style().item_spacing.y
        imgui.begin_child("chat-list", imgui.ImVec2(CHATS_W, row * min(len(self.chats), CHATS_SHOWN)))
        now = time.time()
        for info in self.chats:
            when = when_said(info.updated_at, now)
            when_w = imgui.calc_text_size(when).x
            width = imgui.get_content_region_avail().x
            title = fit(info.title or "(pictures)", width - when_w - 16.0)
            clicked, _ = imgui.selectable(f"{title}##{info.id}", info.id == self.current_chat)
            if clicked:
                if info.id != self.current_chat:
                    send(OpenChat(info.id))
                imgui.close_current_popup()  # inside the list, a pick doesn't close it by itself
            lo = imgui.get_item_rect_min()
            imgui.get_window_draw_list().add_text(imgui.ImVec2(lo.x + width - when_w, lo.y), rgba(150, 150, 150), when)
        imgui.end_child()

    # ---- the conversation ----------------------------------------------------------------------

    def bubble(self, index: int, line: ChatLine, thumbnails, font) -> None:
        """One line of the conversation: the user's on the right, Tomo's on the
        left, notes in the middle — its pictures, if any, above its words,
        and who said it below, with a copy button while the pointer is on it.
        ``font``: ImGui's current one, baked for its size."""
        color, who = BUBBLES.get(line.role, BUBBLES[Role.SYSTEM])
        avail = imgui.get_content_region_avail().x
        pad_x, pad_y = 10.0, 6.0
        row_h = imgui.get_text_line_height()
        top_y = imgui.get_cursor_screen_pos().y

        def place(item_w: float) -> None:
            x = imgui.get_cursor_pos_x()
            if line.role == Role.USER:
                imgui.set_cursor_pos_x(x + avail - item_w)
            elif line.role == Role.SYSTEM:
                imgui.set_cursor_pos_x(x + (avail - item_w) * 0.5)

        if line.attachments and thumbnails is not None:
            pictures(line, thumbnails, place, avail * 0.82)
        if line.text:
            rows = self.layout(line.text, avail * 0.82 - 2 * pad_x, font)
            w, h = math.ceil(max(row.width for row in rows)) + 2 * pad_x, len(rows) * row_h + 2 * pad_y
            place(w)
            top = imgui.get_cursor_screen_pos()
            words = Placed(index, top.x + pad_x, top.y + pad_y, rows)
            self.placed.append(words)
            corner = imgui.ImVec2(top.x + w, top.y + h)
            if imgui.is_rect_visible(top, corner):
                painter = imgui.get_window_draw_list()
                painter.add_rect_filled(top, corner, rgba(*color), 12.0)
                self.paint_words(painter, words, line.text, row_h)
            imgui.invisible_button(f"##words-{index}", imgui.ImVec2(w, h))
            if imgui.is_item_hovered():
                imgui.set_mouse_cursor(imgui.MouseCursor_.text_input.value)
            if imgui.is_item_activated():
                self.press(words, line.text, row_h)
            imgui.set_cursor_screen_pos(imgui.ImVec2(top.x, top.y + h + 2.0))
            imgui.dummy(imgui.ImVec2(0.0, 0.0))
        mouse_y = imgui.get_io().mouse_pos.y
        if imgui.is_window_hovered() and top_y <= mouse_y < imgui.get_cursor_screen_pos().y + row_h:
            self.hovered = index
        # The name where it always was, the copy button on its inner side.
        user = line.role == Role.USER
        button = imgui.get_style().item_spacing.x + COPY_BOX if line.text and user else 0.0
        place(imgui.calc_text_size(who).x + button)
        if line.text and user:
            self.copy_button(index, line.text)
            imgui.same_line()
        imgui.text_colored(vec4(120, 120, 120), who)
        if line.text and not user:
            imgui.same_line()
            self.copy_button(index, line.text)
        imgui.dummy(imgui.ImVec2(0.0, 4.0))

    def layout(self, text: str, width: float, font) -> list[Row]:
        """``text`` in rows for ``width`` in ``font``: measured once, then kept."""
        if (font.size, font.rasterizer_density) != self.font:
            self.font, self.advances, self.layouts = (font.size, font.rasterizer_density), {}, {}
        rows = self.layouts.get((text, width))
        if rows is None:
            if len(self.layouts) > 2 * TRANSCRIPT_LIMIT:
                self.layouts.clear()
            advances = self.advances

            def advance(ch: str) -> float:
                if (got := advances.get(ch)) is None:
                    got = advances[ch] = 0.0 if ch == "\r" else font.get_char_advance(ord(ch))
                return got

            rows = self.layouts[(text, width)] = wrap(text, width, advance)
        return rows

    def paint_words(self, painter, words: Placed, text: str, row_h: float) -> None:
        """A message's rows, white, over the selection's highlight."""
        chosen = self.selection.within(words.line, len(text))
        for k, row in enumerate(words.rows):
            y = words.y + k * row_h
            if chosen is not None and (stretch := highlight(row, *chosen, len(text))) is not None:
                painter.add_rect_filled(imgui.ImVec2(words.x + stretch[0], y),
                                        imgui.ImVec2(words.x + stretch[1], y + row_h), rgba(*SELECTED))
            if row.end > row.start:
                painter.add_text(imgui.ImVec2(words.x, y), rgba(255, 255, 255), text[row.start:row.end])

    def copy_button(self, index: int, text: str) -> None:
        """The copy button by a message's name: showing while the pointer is
        on the message, and a tick for a moment once it has copied."""
        if imgui.invisible_button(f"##copy-{index}", imgui.ImVec2(COPY_BOX, imgui.get_text_line_height())):
            self.copy(text, index)
        done = self.copied_line == index and time.monotonic() < self.copied_until
        over = imgui.is_item_hovered()
        if over and not done:
            imgui.set_tooltip("Copy")
        if self.hovered == index or done:
            colour = rgba(*ACCENT) if done else rgba(230, 230, 230) if over else rgba(120, 120, 120)
            centred_icon(ICON_CHECK if done else ICON_COPY, imgui.get_item_rect_min(), imgui.get_item_rect_max(),
                         colour)

    def press(self, words: Placed, text: str, row_h: float) -> None:
        """The left button went down on a message's words: the caret there
        (with Shift: the selection stretched to it), or a double-click's word,
        or a triple-click's paragraph, selected."""
        io = imgui.get_io()
        pos, under = hit([words], io.mouse_pos.x, io.mouse_pos.y, row_h)
        clicks = imgui.get_mouse_clicked_count(imgui.MouseButton_.left.value)
        s = self.selection
        if io.key_shift and s.anchor is not None:
            s.caret, s.unit = pos, None
        elif clicks >= 2:
            a, b = (word_at if clicks == 2 else paragraph_at)(text, under)
            s.select((words.line, a), (words.line, b))
        else:
            s.place(pos)
        s.dragging = True

    def select_text(self) -> None:
        """The pointer and the keys on the conversation (after its lines are
        drawn): a drag selecting, a click elsewhere in it letting go, the
        right-click menu, Ctrl+C and Ctrl+A."""
        left = imgui.MouseButton_.left.value
        if self.selection.dragging:
            if imgui.is_mouse_down(left):
                self.drag_selection()
            else:
                self.selection.dragging = False
        elif imgui.is_window_hovered() and imgui.is_mouse_clicked(left) and not imgui.is_any_item_hovered() \
                and not imgui.get_io().key_shift:
            self.selection.clear()
        self.text_menu()
        self.text_keys()

    def drag_selection(self) -> None:
        """Follow the pointer while the button is held, scrolling the
        transcript while it's above or below it."""
        imgui.set_mouse_cursor(imgui.MouseCursor_.text_input.value)
        if not imgui.is_mouse_pos_valid():  # off the window
            return
        io = imgui.get_io()
        top = imgui.get_window_pos().y
        bottom = top + imgui.get_window_height()
        x, y = io.mouse_pos.x, io.mouse_pos.y
        if not top <= y <= bottom:
            beyond = y - top if y < top else y - bottom
            imgui.set_scroll_y(imgui.get_scroll_y() + beyond * SCROLL_SPEED * io.delta_time)
        found = hit(self.placed, x, min(max(y, top), bottom - 1.0), imgui.get_text_line_height())
        if found is not None:
            self.selection.drag_to(found[0])

    def text_menu(self) -> None:
        """Right-click on the conversation: copy what's selected, the message
        under the pointer or all of it; or select all."""
        if imgui.is_window_hovered() and imgui.is_mouse_released(imgui.MouseButton_.right.value):
            self.menu_line = self.transcript[self.hovered] if self.hovered is not None else None
            imgui.open_popup("words")
        if not imgui.begin_popup("words"):
            return
        span = self.selection.span()
        if imgui.menu_item("Copy", "Ctrl+C", False, span is not None)[0]:
            self.copy(selected_text(self.transcript, *span))
        message = self.menu_line.text if self.menu_line is not None else ""
        if imgui.menu_item("Copy message", "", False, bool(message))[0]:
            self.copy(message)
        if imgui.menu_item("Select all", "Ctrl+A", False, any(line.text for line in self.transcript))[0]:
            self.select_all()
        imgui.separator()
        if imgui.menu_item("Copy the whole chat", "", False, bool(self.transcript))[0]:
            self.copy(chat_text(self.transcript))
        imgui.end_popup()

    def text_keys(self) -> None:
        """Ctrl+C copies what's selected and Ctrl+A selects everything — while
        the text box hasn't got the keys (it has its own)."""
        if imgui.get_io().want_text_input:
            return
        ctrl = imgui.Key.mod_ctrl.value
        if imgui.is_key_chord_pressed(ctrl | imgui.Key.c.value) and (span := self.selection.span()) is not None:
            self.copy(selected_text(self.transcript, *span))
        elif imgui.is_key_chord_pressed(ctrl | imgui.Key.a.value):
            self.select_all()

    def select_all(self) -> None:
        """Select all the conversation's words."""
        if lines := [i for i, line in enumerate(self.transcript) if line.text]:
            self.selection.select((lines[0], 0), (lines[-1], len(self.transcript[lines[-1]].text)))

    def copy(self, text: str, line: int | None = None) -> None:
        """Put ``text`` on the clipboard and show it's there: a tick on the
        copy button of ``line``, else a note above the text box."""
        imgui.set_clipboard_text(text)
        self.copied_line, self.copied_until = line, time.monotonic() + COPIED_SECS
        if line is None:
            self.notice, self.notice_until = "Copied", self.copied_until


def icon_button(icon: str, tip: str, size: float, active: bool = True, key: str | None = None) -> bool:
    """A square button with its icon in the exact middle. (ImGui places a
    button's label after the frame padding, centred only when it fits: an
    icon wider than the room left starts at the padding instead, off to the
    right. So the button is drawn without a label, and the icon on top of
    it by its own drawn bounds.) ``key`` keeps the button the same while its
    icon or tip changes."""
    clicked = imgui.button(f"##{key or tip}", imgui.ImVec2(size, size))
    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    hovered = imgui.is_item_hovered()
    colour = rgba(120, 120, 120) if not active else imgui.get_color_u32(imgui.Col_.text.value)
    centred_icon(icon, lo, hi, colour)
    if hovered:
        imgui.set_tooltip(tip)
    return clicked


def centred_icon(icon: str, lo, hi, colour: int) -> None:
    """Draw one icon (a glyph of the current font) with its drawn shape in
    the exact middle of the box ``lo``–``hi``."""
    glyph = imgui.get_font_baked().find_glyph(ord(icon))
    x, y = centre_glyph((lo.x, lo.y, hi.x, hi.y), (glyph.x0, glyph.y0, glyph.x1, glyph.y1))
    imgui.get_window_draw_list().add_text(imgui.get_font(), imgui.get_font_size(), imgui.ImVec2(x, y), colour, icon)


def centre_glyph(box, glyph) -> tuple[float, float]:
    """The text position that puts a glyph's drawn bounds (x0, y0, x1, y1,
    relative to that position) in the middle of ``box`` (x0, y0, x1, y1) —
    in whole pixels, as ImGui draws text."""
    x = (box[0] + box[2] - glyph[0] - glyph[2]) * 0.5
    y = (box[1] + box[3] - glyph[1] - glyph[3]) * 0.5
    return float(round(x)), float(round(y))


BUBBLES = {Role.USER: ((70, 130, 180), "You"), Role.ASSISTANT: ((38, 66, 74), "Tomo"), Role.SYSTEM: ((60, 60, 60), "•")}


def new_chat_note() -> None:
    """What an empty chat says, in its middle: that it's new, and that
    what Tomo knows about the user carries over."""
    avail = imgui.get_content_region_avail()
    imgui.dummy(imgui.ImVec2(0.0, max(0.0, avail.y * 0.4 - 20.0)))
    for text in ("A new chat.", "What Tomo knows about you carries over."):
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, (avail.x - imgui.calc_text_size(text).x) * 0.5))
        imgui.text_colored(vec4(130, 130, 130), text)


def when_said(at_ms: int, now: float | None = None) -> str:
    """When a chat was last used, briefly: "14:05" today, "Yesterday",
    "12 Oct" this year, "12 Oct 2025" before."""
    now = time.time() if now is None else now
    t, today, yesterday = time.localtime(at_ms / 1000), time.localtime(now), time.localtime(now - 86400)
    if (t.tm_year, t.tm_yday) == (today.tm_year, today.tm_yday):
        return f"{t.tm_hour:02d}:{t.tm_min:02d}"
    if (t.tm_year, t.tm_yday) == (yesterday.tm_year, yesterday.tm_yday):
        return "Yesterday"
    return f"{t.tm_mday} {MONTHS[t.tm_mon - 1]}" + ("" if t.tm_year == today.tm_year else f" {t.tm_year}")


def fit(text: str, width: float, measure: Callable[[str], float] | None = None) -> str:
    """``text`` cut short with "…" to fit ``width`` px."""
    measure = measure or (lambda s: imgui.calc_text_size(s).x)
    if measure(text) <= width:
        return text
    while text and measure(text + "…") > width:
        text = text[:-1]
    return text.rstrip() + "…"


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
            clicked = imgui.button(f"##picture-{i}", box)
            centred_icon(ICON_IMAGE, imgui.get_item_rect_min(), imgui.get_item_rect_max(),
                         imgui.get_color_u32(imgui.Col_.text.value))
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
