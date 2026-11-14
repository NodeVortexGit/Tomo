"""Slash commands: typed in the chat, run by Tomo itself — never sent to the
model, so they're instant and do exactly what they say.

  /memory add <fact>         remember a fact about the user
  /memory list               what Tomo remembers, numbered
  /memory forget <n|words>   forget one (by its number, or words in it)
  /timer <minutes> [label]   a timer: a whole number of minutes, 1 to 15
  /timer list                the timers running
  /timer cancel [n|all]      stop one

What a command says goes in the chat as a note, which isn't kept: it's not
part of the conversation. :func:`suggestions` is what the chat window
offers as the user types "/".
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .events import Chat, ChatLine, Role
from .language import Language

if TYPE_CHECKING:
    from .db import Store

log = logging.getLogger(__name__)

MAX_TIMER_MINUTES = 15
MAX_TIMERS = 5


@dataclass(frozen=True)
class Command:
    fill: str  # what choosing it puts in the text box
    args: str  # what goes after, shown as a hint
    about: str


COMMANDS = (
    Command("/memory add ", "<fact>", "Remember a fact about you"),
    Command("/memory list", "", "What Tomo remembers about you"),
    Command("/memory forget ", "<number or words>", "Forget a fact"),
    Command("/timer ", "<minutes> [label]", "A timer, 1 to 15 whole minutes"),
    Command("/timer list", "", "The timers running"),
    Command("/timer cancel", " [number]", "Stop a timer"),
)


def is_command(text: str) -> bool:
    return text.lstrip().startswith("/")


def suggestions(text: str) -> list[Command]:
    """The commands that fit what's typed so far; while the arguments are
    being typed, the command's own line, as a reminder."""
    typed = text.lstrip().lower()
    if not typed.startswith("/"):
        return []
    fitting = [c for c in COMMANDS if c.fill.lower().startswith(typed)]
    if fitting:
        return fitting
    typing_args = [c for c in COMMANDS if c.fill.endswith(" ") and typed.startswith(c.fill.lower())]
    return typing_args[-1:]


@dataclass
class Timer:
    number: int
    minutes: int
    label: str
    ends: float  # time.monotonic()
    task: asyncio.Task | None = field(default=None, repr=False)


class Slash:
    """Runs the commands. ``ring(words)`` is called when a timer ends, with
    what to say out loud; ``bulgarian()`` says whether the user last spoke
    Bulgarian (the language a timer without a label speaks)."""

    def __init__(self, db: Store, ui: Callable, ring: Callable[[str], None],
                 bulgarian: Callable[[], bool] = lambda: False) -> None:
        self.db, self.ui, self.ring, self.bulgarian = db, ui, ring, bulgarian
        self.timers: list[Timer] = []
        self._numbers = 0

    def run(self, text: str) -> str:
        """What the command ``text`` tells the user, once it's done."""
        words = text.strip().split(maxsplit=1)
        name, rest = words[0].lower(), (words[1].strip() if len(words) > 1 else "")
        handlers = {"/memory": self.memory, "/timer": self.timer}
        if name not in handlers:
            return f"There's no {name} command. Try /memory or /timer."
        try:
            return handlers[name](rest)
        except Exception as e:  # noqa: BLE001 - a command mustn't take the brain down
            log.warning("%s failed: %s", text, e)
            return f"That didn't work: {e}"

    # ---- /memory ------------------------------------------------------------------

    def remembered(self) -> list[tuple[str, str, str]]:
        """Everything known about the user, as /memory list numbers it: the
        facts (the most important first, as the model sees them), then the
        preferences. Each is (kind, id, words)."""
        facts = self.db.top_memories(10_000)
        return ([("fact", m.id, m.content) for m in facts]
                + [("preference", key, f"{key}: {value}") for key, value in self.db.all_prefs()])

    def memory(self, rest: str) -> str:
        action, _, arg = rest.partition(" ")
        action, arg = action.lower(), arg.strip()
        if action == "list":
            known = self.remembered()
            if not known:
                return "I don't remember anything about you yet. Teach me with /memory add <fact>."
            lines = "\n".join(f"{i}. {words}" for i, (_, _, words) in enumerate(known, 1))
            return f"What I remember about you:\n{lines}\n/memory forget <number> forgets one."
        if action == "add":
            if not arg:
                return "What should I remember? /memory add <fact>, e.g. /memory add I work night shifts."
            if self.db.knows(arg):
                return "I know that already."
            self.db.add_memory("fact", arg, 4)  # the user said it outright: among what the model always sees
            return f"I'll remember that: {arg}"
        if action == "forget":
            return self.forget(arg)
        return "/memory add <fact> remembers, /memory list shows, /memory forget <number or words> forgets."

    def forget(self, arg: str) -> str:
        if not arg:
            return "What should I forget? /memory forget <number or words> (/memory list shows the numbers)."
        known = self.remembered()
        if arg.isdigit():
            index = int(arg) - 1
            if not 0 <= index < len(known):
                return f"There's no number {arg}: /memory list shows them."
            matches = [known[index]]
        else:
            matches = [k for k in known if arg.lower() in k[2].lower()]
        if not matches:
            return f"Nothing I remember mentions “{arg}”."
        if len(matches) > 1:
            lines = "\n".join(f"{known.index(k) + 1}. {k[2]}" for k in matches)
            return f"Several things mention “{arg}”:\n{lines}\nForget one by its number: /memory forget <number>."
        kind, key, words = matches[0]
        if kind == "fact":
            self.db.forget_memory(key)
        else:
            self.db.forget_pref(key)
        return f"Forgotten: {words}"

    # ---- /timer -------------------------------------------------------------------

    def timer(self, rest: str) -> str:
        action, _, arg = rest.partition(" ")
        if action.lower() == "list":
            if not self.timers:
                return "No timers are running. /timer 5 tea starts one."
            now = time.monotonic()
            return "Timers:\n" + "\n".join(
                f"{t.number}. {t.label or minutes(t.minutes)} — {left(t.ends - now)} left" for t in self.timers)
        if action.lower() == "cancel":
            return self.cancel(arg.strip().lower())
        parsed = parse_timer(rest)
        if isinstance(parsed, str):
            return parsed
        count, label = parsed
        if len(self.timers) >= MAX_TIMERS:
            return f"{MAX_TIMERS} timers are running already; /timer cancel stops one."
        self._numbers += 1
        timer = Timer(self._numbers, count, label, time.monotonic() + count * 60)
        timer.task = asyncio.get_running_loop().create_task(self._wait(timer))
        self.timers.append(timer)
        ends = time.strftime("%H:%M", time.localtime(time.time() + count * 60))
        return f"Timer {timer.number}: {minutes(count)}" + (f" ({label})" if label else "") + f", until {ends}."

    def cancel(self, arg: str) -> str:
        if not self.timers:
            return "No timers are running."
        if arg == "all":
            stopping = list(self.timers)
        elif arg:
            stopping = [t for t in self.timers if str(t.number) == arg]
            if not stopping:
                return f"There's no timer {arg}: /timer list shows them."
        elif len(self.timers) == 1:
            stopping = list(self.timers)
        else:
            return "Which one? /timer cancel <number>, or /timer cancel all (/timer list shows them)."
        for t in stopping:
            t.task.cancel()
            self.timers.remove(t)
        return "Stopped: " + ", ".join(t.label or f"timer {t.number}" for t in stopping) + "."

    async def _wait(self, timer: Timer) -> None:
        await asyncio.sleep(timer.minutes * 60)
        if timer in self.timers:
            self.timers.remove(timer)
        self.ui(Chat(ChatLine(Role.SYSTEM, f"Time's up: {timer.label or minutes(timer.minutes)}.")))
        bulgarian = (Language.of(timer.label) == Language.BULGARIAN if any(c.isalpha() for c in timer.label)
                     else self.bulgarian())
        if bulgarian:
            words = f"Времето изтече: {timer.label}." if timer.label else "Времето изтече."
        else:
            words = f"Time's up: {timer.label}." if timer.label else "Time's up."
        self.ring(words)

    def stop(self) -> None:
        for t in self.timers:
            t.task.cancel()
        self.timers.clear()


# ---- small helpers -------------------------------------------------------------------

USAGE = "/timer takes whole minutes, 1 to 15: /timer 5, or /timer 5 tea."
MINUTE_WORDS = {"m", "min", "mins", "minute", "minutes", "мин", "минута", "минути"}


def parse_timer(text: str) -> tuple[int, str] | str:
    """``5 tea`` → (5, "tea"): a whole number of minutes, then the label
    (a word for minutes after the number is fine: ``5 min tea``); or what's
    wrong with it."""
    first, _, rest = text.strip().partition(" ")
    if not re.fullmatch(r"\d+", first):
        return USAGE
    count = int(first)
    if not 1 <= count <= MAX_TIMER_MINUTES:
        return f"Timers go from 1 to {MAX_TIMER_MINUTES} minutes; {count} doesn't fit."
    word, _, after = rest.strip().partition(" ")
    if word.lower() in MINUTE_WORDS:
        rest = after
    return count, " ".join(rest.split())


def minutes(count: int) -> str:
    return f"{count} min"


def left(seconds: float) -> str:
    whole = max(0, round(seconds))
    return f"{whole // 60} min {whole % 60:02d} s" if whole >= 60 else f"{whole} s"
