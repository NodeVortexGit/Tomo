"""The health programme: a round of exercise — 10 push-ups, 10 sit-ups and
10 squats — that the camera watches and counts, and that locks the desktop
until it's done.

No AI decides anything here. A small pose model (MediaPipe Pose Lite, in
``scripts/pose.py``) turns camera frames into 33 body joints; everything
after that is this module's plain geometry and rules:

* **Reps** — each exercise is a joint angle going down past one threshold and
  back up past another (a gap between them, so a wobble near one doesn't
  count twice): elbows for push-ups, hips for sit-ups, knees for squats.
* **Breaks** — standing up, or being away from the camera, for 30 s in a row
  puts the next round off to 3 hours from then. Without one, the round comes
  an hour after the last (or after Tomo started).
* **Always on, with a camera** — there's no switch. Without a camera the
  programme sleeps, and it wakes up when one appears. A camera lost in the
  middle of a round ends the lock (there's nothing to count with).
* **Can't be skipped by a restart** — when the next round is due, and whether
  one is under way, is kept in ``health.json`` in the data folder.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from . import platform
from .config import Config
from .events import BrainToUi, Chat, ChatLine, HealthLock, HealthProgress, Pose, Role, now_ms
from .language import Language

log = logging.getLogger(__name__)

REPS = 10
EVERY = 60 * 60 * 1000  # a round is due this long after the last one…
AFTER_BREAK = 3 * 60 * 60 * 1000  # …or this long after a break
BREAK = 30 * 1000  # standing or away this long, in a row, is a break
# Seen seated for less than this doesn't end a break: the pose model can
# misjudge a frame or two.
SEATED_GRACE = 3 * 1000
VISIBLE = 0.5  # a joint the model is less sure of is left out
IDLE_FPS = 2  # a few frames to notice breaks…
EXERCISE_FPS = 15  # …more to count reps and mirror the body smoothly
POSE_MODEL = "pose_landmarker_lite.task"


@dataclass(frozen=True)
class Joint:
    """A joint in metres from the middle of the hips (x to the image's right,
    y down, z away from the camera — MediaPipe's world coordinates), and how
    sure the model is that it's visible, 0–1."""

    x: float
    y: float
    z: float
    v: float


# The joints the programme uses, by MediaPipe's numbering.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
JOINT_COUNT = 33


def visible(pose: list[Joint], index: int) -> tuple[float, float, float] | None:
    if index >= len(pose) or pose[index].v < VISIBLE:
        return None
    j = pose[index]
    return j.x, j.y, j.z


def angle(pose: list[Joint], a: int, b: int, c: int) -> float | None:
    """The angle at ``b`` between ``a`` and ``c``, in degrees (180 = straight)."""
    pa, pb, pc = visible(pose, a), visible(pose, b), visible(pose, c)
    if pa is None or pb is None or pc is None:
        return None
    u = [pa[i] - pb[i] for i in range(3)]
    w = [pc[i] - pb[i] for i in range(3)]
    lu, lw = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in w))
    if lu < 1e-4 or lw < 1e-4:
        return None
    cos = sum(u[i] * w[i] for i in range(3)) / (lu * lw)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def both_sides(pose: list[Joint], left: tuple[int, int, int], right: tuple[int, int, int]) -> float | None:
    """The same angle on both sides of the body — whichever the camera sees."""
    l, r = angle(pose, *left), angle(pose, *right)
    if l is not None and r is not None:
        return (l + r) / 2
    return l if l is not None else r


def lean(pose: list[Joint], a: tuple[int, int], b: tuple[int, int]) -> float | None:
    """How far the line from pair ``a`` to pair ``b`` leans from vertical, in
    degrees (0 = upright, 90 = lying flat), from the midpoints of each pair."""

    def mid(pair: tuple[int, int]):
        p, q = visible(pose, pair[0]), visible(pose, pair[1])
        if p is not None and q is not None:
            return tuple((p[i] + q[i]) / 2 for i in range(3))
        return p if p is not None else q

    top, bottom = mid(a), mid(b)
    if top is None or bottom is None:
        return None
    d = [bottom[i] - top[i] for i in range(3)]
    length = math.sqrt(sum(x * x for x in d))
    if length <= 1e-4:
        return None
    return math.degrees(math.acos(max(0.0, min(1.0, abs(d[1]) / length))))


class Exercise(Enum):
    PUSH_UPS = "push-ups"
    SIT_UPS = "sit-ups"
    SQUATS = "squats"

    def name_in(self, language: Language) -> str:
        if language == Language.BULGARIAN:
            return {"push-ups": "лицеви опори", "sit-ups": "коремни преси", "squats": "клякания"}[self.value]
        return self.value

    def hint(self, language: Language) -> str:
        """How to be seen doing it."""
        hints = {
            (Exercise.PUSH_UPS, Language.ENGLISH): "Side-on to the camera, your whole body in view.",
            (Exercise.SIT_UPS, Language.ENGLISH): "Lie side-on to the camera, knees bent.",
            (Exercise.SQUATS, Language.ENGLISH): "Face the camera, from your head to your feet in view.",
            (Exercise.PUSH_UPS, Language.BULGARIAN): "Застанете странично на камерата, цялото тяло да се вижда.",
            (Exercise.SIT_UPS, Language.BULGARIAN): "Легнете странично на камерата, със свити колене.",
            (Exercise.SQUATS, Language.BULGARIAN): "Застанете с лице към камерата, от главата до стъпалата.",
        }
        return hints[(self, language)]

    def measure(self, pose: list[Joint]) -> float | None:
        """The angle that goes down and up with each rep."""
        if self == Exercise.PUSH_UPS:
            # Elbows bend and straighten — with the body level, so bending the
            # arms standing up doesn't count.
            level = lean(pose, (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_ANKLE, RIGHT_ANKLE))
            if level is None:
                level = lean(pose, (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP))
            if level is None or level < 50:
                return None
            return both_sides(pose, (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))
        if self == Exercise.SIT_UPS:
            # The torso curls up toward bent knees, and back down.
            knees = both_sides(pose, (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
            if knees is not None and knees > 150:
                return None  # legs straight: not a sit-up
            return both_sides(pose, (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE), (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE))
        return both_sides(pose, (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))

    def thresholds(self) -> tuple[float, float]:
        """(down, up), degrees: below ``down`` the rep is at its middle,
        above ``up`` back at the start."""
        return {Exercise.PUSH_UPS: (100.0, 150.0), Exercise.SIT_UPS: (80.0, 115.0),
                Exercise.SQUATS: (105.0, 155.0)}[self]


ROUND = (Exercise.PUSH_UPS, Exercise.SIT_UPS, Exercise.SQUATS)


@dataclass
class RepCounter:
    """Counts one exercise's reps from pose frames."""

    exercise: Exercise
    count: int = 0
    # The starting position (the top of a squat, lying back for a sit-up) has
    # been seen: reps count from here.
    ready: bool = False
    halfway: bool = False
    smoothed: float | None = None  # the angle, smoothed against the model's jitter

    def update(self, pose: list[Joint]) -> bool:
        """Take a frame. True when it completed a rep."""
        raw = self.exercise.measure(pose)
        if raw is None:
            return False
        a = raw if self.smoothed is None else self.smoothed + (raw - self.smoothed) * 0.5
        self.smoothed = a
        down, up = self.exercise.thresholds()
        # Not before the starting position: getting up from the last sit-up
        # (knees bent, then straight) isn't a squat.
        if not self.ready:
            self.ready = a > up
            return False
        if not self.halfway and a < down:
            self.halfway = True
        elif self.halfway and a > up:
            self.halfway = False
            self.count += 1
            return True
        return False

    def done(self) -> bool:
        return self.count >= REPS


class Presence(Enum):
    SEATED = "seated"
    STANDING = "standing"
    AWAY = "away"  # nobody in view


def presence(pose: list[Joint] | None) -> Presence:
    """Seated, standing or away. Standing needs straight legs in view; at a
    desk the camera mostly sees the upper body, and a person who gets up
    usually leaves the picture — away."""
    if pose is None:
        return Presence.AWAY
    if all(visible(pose, i) is None for i in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)):
        return Presence.AWAY
    knees = both_sides(pose, (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
    hips = both_sides(pose, (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE), (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE))
    if knees is not None and hips is not None and knees > 150 and hips > 150:
        return Presence.STANDING
    return Presence.SEATED


@dataclass
class Schedule:
    """When the next round is due, and the break under way, if any. Kept in
    ``health.json`` (with ``in_round``) so a restart changes nothing."""

    due_ms: int
    in_round: bool = False
    break_since: int | None = field(default=None, repr=False)
    seated_since: int | None = field(default=None, repr=False)
    counted: bool = field(default=False, repr=False)  # this break put the round off

    @classmethod
    def new(cls, now: int) -> "Schedule":
        return cls(due_ms=now + EVERY)

    def observe(self, now: int, seen: Presence) -> bool:
        """Take what the camera sees. True when that made a break that put
        the round off."""
        if self.in_round:
            return False
        if seen in (Presence.STANDING, Presence.AWAY):
            self.seated_since = None
            if self.break_since is None:
                self.break_since = now
            if not self.counted and now - self.break_since >= BREAK:
                self.counted = True
                self.due_ms = now + AFTER_BREAK
                return True
        else:
            if self.seated_since is None:
                self.seated_since = now
            if now - self.seated_since >= SEATED_GRACE:
                self.break_since = None
                self.counted = False
        return False

    def is_due(self, now: int) -> bool:
        return self.in_round or now >= self.due_ms

    def completed(self, now: int) -> None:
        """A round was done: the next an hour from now."""
        self.in_round = False
        self.due_ms = now + EVERY
        self.break_since = self.seated_since = None
        self.counted = False

    @classmethod
    def load(cls, path: Path, now: int) -> "Schedule":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            saved = cls(due_ms=int(data["due_ms"]), in_round=bool(data.get("in_round", False)))
        except (OSError, ValueError, KeyError, TypeError):
            return cls.new(now)
        if saved.in_round:
            return saved  # a round under way resumes, whatever the time
        if saved.due_ms <= now:
            # Due while Tomo wasn't running (the computer was off): that was a
            # break, so a fresh hour. A round under way can't be escaped this
            # way — it resumes above.
            return cls.new(now)
        return saved

    def save(self, path: Path) -> None:
        try:
            Path(path).write_text(json.dumps({"due_ms": self.due_ms, "in_round": self.in_round}), encoding="utf-8")
        except OSError as e:
            log.warning("couldn't save the health schedule: %s", e)


@dataclass
class Round:
    """The round under way: which exercise, and its count."""

    step: int = 0
    counter: RepCounter = field(default_factory=lambda: RepCounter(ROUND[0]))

    def exercise(self) -> Exercise:
        return self.counter.exercise

    def next(self) -> bool:
        """Move on after a finished exercise. False when the round is over."""
        self.step += 1
        if self.step >= len(ROUND):
            return False
        self.counter = RepCounter(ROUND[self.step])
        return True


@dataclass(frozen=True)
class Progress:
    """What the lock screen shows."""

    title: str
    count: int
    target: int
    hint: str


def number(n: int, language: Language) -> str:
    """Words for counting aloud."""
    en = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
    bg = ["Едно", "Две", "Три", "Четири", "Пет", "Шест", "Седем", "Осем", "Девет", "Десет"]
    words = bg if language == Language.BULGARIAN else en
    return f"{words[n - 1]}!" if 1 <= n <= len(words) else str(n)


# ---- the voice: one line at a time ---------------------------------------------------


@dataclass(frozen=True)
class Line:
    text: str
    count: bool = False  # a rep's count: skipped if a later count is waiting


GAP = 0.45  # quiet after each line, so they don't run into each other…
GAP_BEFORE_NEWS = 0.9  # …and a longer breath before an announcement after a count


def next_line(waiting: list[Line]) -> Line:
    """The next line to say (in order), dropping counts a later count made stale."""
    first = waiting.pop(0)
    if first.count and any(line.count for line in waiting):
        return next_line(waiting)
    return first


def spawn_voice(speech) -> asyncio.Queue:
    """Speak lines one at a time, with a pause between them. If the user is
    quicker than the voice, the counts they've outrun are dropped: she says
    the latest, never lagging behind."""
    queue: asyncio.Queue[Line] = asyncio.Queue()

    async def speak() -> None:
        waiting: list[Line] = []
        last_was_count = False
        while True:
            if not waiting:
                waiting.append(await queue.get())
            while not queue.empty():
                waiting.append(queue.get_nowait())
            line = next_line(waiting)
            if last_was_count and not line.count:
                await asyncio.sleep(GAP_BEFORE_NEWS - GAP)
            try:
                await speech.play(await speech.synthesize(line.text))
            except Exception as e:  # noqa: BLE001 - a missing voice mustn't stop the round
                log.debug("couldn't say %r: %s", line.text, e)
            last_was_count = line.count
            await asyncio.sleep(GAP)

    asyncio.get_running_loop().create_task(speak())
    return queue


# ---- the runner ----------------------------------------------------------------------------


def dev_mode() -> bool:
    """Python's development mode (``python -X dev``): only then do the test
    hooks work (a replay instead of the camera, a round due at once). A
    normal start always uses the camera and the real schedule."""
    return bool(sys.flags.dev_mode)


def spawn(cfg: Config, speech, bulgarian: Callable[[], bool], ui: Callable[[BrainToUi], None],
          python: str | None = None) -> None:
    """Start the programme: it waits for a camera, and runs for as long as
    the brain does. Nothing to start without the pose model."""
    model = cfg.data_dir / "models" / POSE_MODEL
    if not model.is_file():
        log.warning("the health programme is off: no pose model at %s (setup_models.py fetches it)", model)
        return
    script = cfg.scripts_dir / "pose.py"
    state = cfg.data_dir / "health.json"
    replay = os.environ.get("TOMO_POSE_REPLAY") if dev_mode() else None
    python = python or platform.helper_python()

    async def keep_running() -> None:
        while True:
            try:
                await run(python, script, model, replay, state, speech, bulgarian, ui)
            except Exception as e:  # noqa: BLE001
                log.warning("health programme stopped: %s; restarting it", e)
            ui(HealthLock(False))
            await asyncio.sleep(10)

    asyncio.get_running_loop().create_task(keep_running())


async def run(python: str, script: Path, model: Path, replay: str | None, state: Path, speech,
              bulgarian: Callable[[], bool], ui: Callable[[BrainToUi], None]) -> None:
    """One run of the pose helper: until it exits."""
    args = [python, str(script), "--model", str(model), "--fps", str(IDLE_FPS)]
    if replay:
        args += ["--replay", replay]
    process = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        creationflags=platform.quiet_flags(),
    )
    schedule = Schedule.load(state, now_ms())
    if dev_mode() and (due_in := os.environ.get("TOMO_HEALTH_DUE_IN")):
        schedule.due_ms = now_ms() + int(float(due_in) * 1000)
    schedule.save(state)
    camera = False
    current: Round | None = None
    seen_at = 0
    voice = spawn_voice(speech)

    def language() -> Language:
        return Language.BULGARIAN if bulgarian() else Language.ENGLISH

    async def fps(n: int) -> None:
        process.stdin.write(f"fps {n}\n".encode())
        await process.stdin.drain()

    try:
        while True:
            raw = await process.stdout.readline()
            if not raw:
                raise RuntimeError("pose.py exited")
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            now = now_ms()
            kind = event.get("event")
            if kind == "camera" and bool(event.get("ok")) != camera:
                camera = bool(event.get("ok"))
                if camera:
                    log.info("camera found: the health programme is on")
                else:
                    log.info("no camera: the health programme is off until one appears")
                    if current is not None:
                        # Nothing to count with: let the user go (the round
                        # stays owed, and resumes when the camera is back).
                        current = None
                        ui(HealthLock(False))
            elif kind == "error":
                raise RuntimeError(event.get("message", "error"))
            elif kind == "pose" and camera:
                world = event.get("world")
                pose = [Joint(*j) for j in world] if world else None
                if current is None:
                    if schedule.observe(now, presence(pose)):
                        log.info("a break: the next round is in 3 hours")
                        schedule.save(state)
                    if schedule.is_due(now):
                        schedule.in_round = True
                        schedule.save(state)
                        current = Round()
                        lang = language()
                        ui(HealthLock(True))
                        name = current.exercise().name_in(lang)
                        voice.put_nowait(Line(f"Exercise time! {REPS} {name}." if lang == Language.ENGLISH
                                              else f"Време за упражнения! {REPS} {name}."))
                        await fps(EXERCISE_FPS)
                        send_progress(ui, current, lang, True)
                    continue
                lang = language()
                ui(Pose(pose))
                if pose is None:
                    if now - seen_at > 1500:
                        send_progress(ui, current, lang, False)
                    continue
                seen_at = now
                if current.counter.update(pose):
                    voice.put_nowait(Line(number(current.counter.count, lang), count=True))
                    if current.counter.done():
                        if current.next():
                            name = current.exercise().name_in(lang)
                            voice.put_nowait(Line(f"Now {REPS} {name}." if lang == Language.ENGLISH
                                                  else f"Сега {REPS} {name}."))
                        else:
                            current = None
                            schedule.completed(now)
                            schedule.save(state)
                            await fps(IDLE_FPS)
                            ui(Pose(None))
                            ui(HealthLock(False))
                            done = ("All done, well done! The next round is in an hour." if lang == Language.ENGLISH
                                    else "Готово, браво! Следващият кръг е след час.")
                            ui(Chat(ChatLine(Role.SYSTEM, done)))
                            voice.put_nowait(Line(done))
                            continue
                send_progress(ui, current, lang, True)
    finally:
        if process.returncode is None:
            process.kill()


def send_progress(ui: Callable[[BrainToUi], None], current: Round, language: Language, seen: bool) -> None:
    exercise = current.exercise()
    if seen:
        hint = exercise.hint(language)
    elif language == Language.BULGARIAN:
        hint = "Не ви виждам. Отдръпнете се, за да ви вижда камерата цели."
    else:
        hint = "I can't see you. Step back so the camera sees your whole body."
    name = exercise.name_in(language)
    ui(HealthProgress(Progress(name[:1].upper() + name[1:], current.counter.count, REPS, hint)))
