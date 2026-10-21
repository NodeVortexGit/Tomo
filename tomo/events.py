"""What the brain and the body say to each other.

The body (the window, on the main thread) and the brain (its own thread,
running asyncio) share nothing but two queues of these messages:
``UiToBrain`` from the body to the brain, ``BrainToUi`` the other way. The
body drains its queue once a frame; nothing it does ever waits on the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"  # a status line in the chat; never sent to the model


@dataclass(frozen=True)
class Attachment:
    """An image the user sent (see attachments.py)."""

    path: str  # Tomo's copy, in the data folder: what the chat shows and the model sees
    source: str  # the file the user picked ("" when it was pasted)
    about: str  # what the system knows about it, as the model reads it


@dataclass
class ChatLine:
    role: Role
    text: str
    at_ms: int = field(default_factory=now_ms)
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True)
class CharacterChoice:
    name: str
    path: Path


@dataclass(frozen=True)
class ChatInfo:
    """One chat in the history."""

    id: str
    title: str  # its first words
    started_at: int  # ms
    updated_at: int  # ms
    count: int  # lines


# ---- body → brain ---------------------------------------------------------------


@dataclass(frozen=True)
class UserMessage:
    text: str
    images: tuple[Path, ...] = ()  # image files to send with it


@dataclass(frozen=True)
class StartVoiceInput:
    """The Talk button: take a spoken request now."""


@dataclass(frozen=True)
class ImportCharacter:
    path: Path
    name: str


@dataclass(frozen=True)
class PanicStop:
    """The panic hotkey: no more mouse/keyboard control."""


@dataclass(frozen=True)
class SetControlAllowed:
    on: bool


@dataclass(frozen=True)
class SetVoice:
    """The chat's speaker toggle: speak replies aloud or not."""

    on: bool


@dataclass(frozen=True)
class NewChat:
    """Start a new chat (what Tomo knows about the user stays)."""


@dataclass(frozen=True)
class OpenChat:
    """Go back to a chat from the history, to read it or carry on."""

    id: str


@dataclass(frozen=True)
class SetHidden:
    """Hide her in the corner of the screen, or bring her out."""

    on: bool


@dataclass(frozen=True)
class Shutdown:
    pass


UiToBrain = (UserMessage | StartVoiceInput | ImportCharacter | PanicStop | SetControlAllowed | SetVoice | NewChat
             | OpenChat | SetHidden | Shutdown)


# ---- brain → body -------------------------------------------------------------------


@dataclass(frozen=True)
class Chat:
    line: ChatLine


@dataclass(frozen=True)
class Transcript:
    """The chat showing now, from its start (a new chat: empty)."""

    lines: tuple[ChatLine, ...]


@dataclass(frozen=True)
class ChatList:
    """The chats there are, newest first, and which one is showing."""

    chats: tuple[ChatInfo, ...]
    current: str


@dataclass(frozen=True)
class Emote:
    emotion: str


@dataclass(frozen=True)
class WalkTo:
    position: float  # 0 = left edge … 1 = right edge


@dataclass(frozen=True)
class Carry:
    """Her part as she uses the mouse (hands.py): "reach" — stop and raise
    her arms for the cursor; "hold" — hang from it wherever it goes; "drop"
    — let go."""

    stage: str


@dataclass(frozen=True)
class Hide:
    """Hide in the corner (or come out): she peeks over the screen's edge
    until the chat or a task brings her out."""

    on: bool


@dataclass(frozen=True)
class ControlMode:
    on: bool


@dataclass(frozen=True)
class Animate:
    clip: str


@dataclass(frozen=True)
class Thinking:
    on: bool


@dataclass(frozen=True)
class Listening:
    on: bool


@dataclass(frozen=True)
class Speaking:
    on: bool


@dataclass(frozen=True)
class LoadCharacter:
    path: Path


@dataclass(frozen=True)
class Characters:
    choices: tuple[CharacterChoice, ...]


@dataclass(frozen=True)
class Status:
    text: str


@dataclass(frozen=True)
class HealthLock:
    """The health programme locks the desktop for a round (or lets go)."""

    on: bool


@dataclass(frozen=True)
class HealthProgress:
    progress: Any  # tomo.health.Progress


@dataclass(frozen=True)
class Pose:
    """The user's joints this frame, during a round (None: not seen)."""

    joints: Any  # list[tomo.health.Joint] | None


BrainToUi = (
    Chat | Transcript | ChatList | Emote | WalkTo | Carry | Hide | ControlMode | Animate | Thinking | Listening
    | Speaking | LoadCharacter | Characters | Status | HealthLock | HealthProgress | Pose
)
