import asyncio
import json
import sys

from bodies import Body, cycle, hidden

from tomo import health
from tomo.events import HealthLock, HealthProgress, Pose
from tomo.health import (LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE,
                         RIGHT_SHOULDER, Exercise, Joint, Line, Presence, RepCounter, Round, Schedule, angle, lean,
                         next_line, number, presence)
from tomo.language import Language

S = 1000
MIN = 60 * S


def reps(exercise: Exercise, frames: list[Body]) -> int:
    counter = RepCounter(exercise)
    for body in frames:
        counter.update(body.pose())
    return counter.count


def test_angles_come_out_as_built():
    knee = angle(Body(knees=90).pose(), LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    assert abs(knee - 90) < 1
    lying = lean(Body(torso=90).pose(), (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP))
    assert abs(lying - 90) < 1


def test_squats_are_counted():
    top, bottom = Body.standing(), Body(knees=80, hips=90)
    assert reps(Exercise.SQUATS, cycle(10, top, bottom)) == 10
    # Half squats that never get low enough don't count.
    assert reps(Exercise.SQUATS, cycle(5, top, Body(knees=130))) == 0
    # Getting up with bent knees (after the sit-ups) isn't a squat.
    assert reps(Exercise.SQUATS, [bottom, bottom, top, top]) == 0


def test_push_ups_are_counted_only_with_the_body_level():
    # Face down, head forward (torso −90°: leaning toward +x).
    def plank(elbows):
        return Body(torso=-90, hips=180, knees=180, elbows=elbows, shoulders=80)

    assert reps(Exercise.PUSH_UPS, cycle(10, plank(170), plank(70))) == 10
    # Bending the arms standing up isn't a push-up.
    assert reps(Exercise.PUSH_UPS, cycle(5, Body(elbows=170), Body(elbows=60))) == 0


def test_sit_ups_are_counted():
    lying = Body(torso=90, hips=130, knees=90, elbows=90, shoulders=150)
    curled = Body(torso=20, hips=60, knees=90, elbows=90, shoulders=150)
    assert reps(Exercise.SIT_UPS, cycle(10, lying, curled)) == 10
    # Straight legs (touching toes, standing) aren't sit-ups.
    assert reps(Exercise.SIT_UPS, cycle(3, Body(hips=170), Body(hips=60))) == 0


def test_unseen_joints_count_nothing():
    pose = [Joint(j.x, j.y, j.z, 0.1) for j in Body(knees=80).pose()]
    assert not RepCounter(Exercise.SQUATS).update(pose)
    assert Exercise.SQUATS.measure(pose) is None


def test_presence_tells_seated_standing_and_away():
    assert presence(None) == Presence.AWAY
    assert presence(Body.standing().pose()) == Presence.STANDING
    sitting = Body(hips=95, knees=90)
    assert presence(sitting.pose()) == Presence.SEATED
    # At a desk the camera sees no legs: seated.
    desk = hidden(sitting.pose(), LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, LEFT_HIP, RIGHT_HIP)
    assert presence(desk) == Presence.SEATED


def test_without_a_break_the_round_comes_after_an_hour():
    s = Schedule.new(0)
    for t in range(0, 60 * MIN, 500):
        assert not s.observe(t, Presence.SEATED)
        assert not s.is_due(t)
    assert s.is_due(60 * MIN)


def test_a_break_of_30_seconds_puts_it_off_3_hours():
    s = Schedule.new(0)
    start = 20 * MIN
    postponed_at = None
    for t in range(start, start + 45 * S, 500):
        if s.observe(t, Presence.STANDING if t < start + 20 * S else Presence.AWAY):
            postponed_at = t
    assert postponed_at is not None, "a 45 s break counts"
    assert start + 30 * S <= postponed_at < start + 31 * S
    assert s.due_ms == postponed_at + 3 * 60 * MIN
    assert not s.is_due(61 * MIN), "not after the first hour any more"


def test_a_short_break_or_a_flicker_changes_nothing():
    s = Schedule.new(0)
    # Up for 20 s, then back: no break.
    for t in range(0, 20 * S, 500):
        assert not s.observe(t, Presence.STANDING)
    for t in range(20 * S, 40 * S, 500):
        assert not s.observe(t, Presence.SEATED)
    assert s.due_ms == 60 * MIN
    # Up for 40 s with a 1 s "seated" misreading in the middle: a break.
    s = Schedule.new(0)
    postponed = False
    for t in range(0, 40 * S, 500):
        postponed |= s.observe(t, Presence.SEATED if 15 * S <= t < 16 * S else Presence.STANDING)
    assert postponed


def test_a_round_under_way_survives_a_restart(tmp_path):
    path = tmp_path / "health.json"
    s = Schedule.new(0)
    s.in_round = True
    s.save(path)
    loaded = Schedule.load(path, 10 * 60 * MIN)
    assert loaded.in_round and loaded.is_due(10 * 60 * MIN)
    # Breaks don't apply mid-round.
    assert not loaded.observe(10 * 60 * MIN, Presence.AWAY)
    # A plain due time from long ago (the computer was off) starts afresh.
    Schedule.new(0).save(path)
    assert Schedule.load(path, 5 * 60 * MIN).due_ms == 6 * 60 * MIN
    # A broken file starts afresh too.
    path.write_text("{", encoding="utf-8")
    assert Schedule.load(path, 7).due_ms == 7 + 60 * MIN


def test_stale_counts_are_skipped_but_news_never_is():
    waiting = [Line("Eight!", True), Line("Nine!", True), Line("Ten!", True), Line("Now ten sit-ups.")]
    assert next_line(waiting) == Line("Ten!", True)
    assert next_line(waiting) == Line("Now ten sit-ups.")
    assert waiting == []
    # One count waiting, then news: both, in order.
    waiting = [Line("Exercise time!"), Line("One!", True)]
    assert next_line(waiting) == Line("Exercise time!")
    assert next_line(waiting) == Line("One!", True)


def test_a_round_is_ten_of_each_in_order():
    r = Round()
    assert r.exercise() == Exercise.PUSH_UPS
    assert r.next() and r.exercise() == Exercise.SIT_UPS
    assert r.next() and r.exercise() == Exercise.SQUATS
    assert not r.next()
    assert number(3, Language.ENGLISH) == "Three!"
    assert number(10, Language.BULGARIAN) == "Десет!"


# ---- a whole round, with a pretend camera ---------------------------------------------------


class Voice:
    """Speech that writes down what it would say."""

    def __init__(self):
        self.said = []

    async def synthesize(self, text):
        return text

    async def play(self, clip):
        self.said.append(clip)


# Plays the frames it's given, then waits for Tomo to slow the camera down
# again (the round is over) — or 10 s — and quits.
PRETEND_HELPER = """import sys, threading
sys.stdout.write(open(sys.argv[-1], encoding="utf-8").read())
sys.stdout.flush()
idle = threading.Event()
def listen():
    for line in sys.stdin:
        if line.strip() == "fps 2":
            idle.set()
threading.Thread(target=listen, daemon=True).start()
idle.wait(10)
"""


def whole_round() -> list[Body]:
    def plank(elbows):
        return Body(torso=-90, elbows=elbows, shoulders=80)

    lying = Body(torso=90, hips=130, knees=90, elbows=90, shoulders=150)
    curled = Body(torso=20, hips=60, knees=90, elbows=90, shoulders=150)
    return (cycle(10, plank(170), plank(70)) + cycle(10, lying, curled)
            + cycle(10, Body.standing(), Body(knees=80, hips=90)))


def test_a_round_locks_counts_aloud_and_lets_go(tmp_path):
    # The helper: a camera, then the whole round, then it quits.
    frames = tmp_path / "frames.jsonl"
    lines = [{"event": "camera", "ok": True}]
    lines += [{"event": "pose", "world": [[j.x, j.y, j.z, j.v] for j in body.pose()]} for body in whole_round()]
    frames.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    helper = tmp_path / "pose.py"
    helper.write_text(PRETEND_HELPER, encoding="utf-8")
    state = tmp_path / "health.json"
    state.write_text(json.dumps({"due_ms": 0, "in_round": True}), encoding="utf-8")  # a round owed

    events, voice = [], Voice()

    async def go():
        try:
            await health.run(sys.executable, helper, tmp_path / "model.task", str(frames), state, voice,
                             lambda: False, events.append)
        except RuntimeError as e:
            assert "exited" in str(e)
        await asyncio.sleep(0.05)  # the voice's last lines

    health.GAP, health.GAP_BEFORE_NEWS = 0.0, 0.0
    try:
        asyncio.run(go())
    finally:
        health.GAP, health.GAP_BEFORE_NEWS = 0.45, 0.9

    assert [e.on for e in events if isinstance(e, HealthLock)] == [True, False], "locked for the round, then let go"
    shown = [(e.progress.title, e.progress.count) for e in events if isinstance(e, HealthProgress)]
    titles = list(dict.fromkeys(title for title, _ in shown))
    assert titles == ["Push-ups", "Sit-ups", "Squats"]
    # The tenth of each moves on (the tenth squat ends the round).
    assert all(max(c for t, c in shown if t == title) == 9 for title in titles)
    assert any(isinstance(e, Pose) and e.joints for e in events), "Tomo mirrors the user"
    assert voice.said[0] == "Exercise time! 10 push-ups."
    assert "Now 10 sit-ups." in voice.said and "Now 10 squats." in voice.said
    assert voice.said[-1].startswith("All done")
    assert json.loads(state.read_text(encoding="utf-8"))["in_round"] is False
