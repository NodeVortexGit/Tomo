"""Locomotion — a small physics engine for the character.

The character is a rigid body in screen space (logical pixels, origin
top-left, +y down) inside the arena: the window. It has a centre of mass, a
velocity, a roll angle (unbounded — it can turn all the way round) and a
spin, and when it isn't on its feet it collides as a capsule with the floor,
walls and ceiling, through impulses with bounce and friction. Every frame, in
small fixed steps, depending on its stance:

* standing, it balances on its feet, walks toward a target the brain (or idle
  wandering) picks, and leans into acceleration;
* it can sit on the floor, or lie down for a nap, and get back up;
* flying, it tumbles freely: a throw can send it spinning, and it lands on its
  feet only if it comes down nearly upright — otherwise it falls over, lies
  there a moment, and gets up;
* carried, it hangs from the point it was grabbed like a pendulum, free to
  swing all the way round; letting go throws it with that swing;
* its limbs ride along (:class:`Limbs`): arms and legs hang like pendulums in
  what they feel — gravity minus the body's own acceleration, plus the air
  pushing back — so they dangle and swing when carried, float up in a fall
  and lag behind a sudden start; the knees give on landing.

:mod:`.animation` only turns all this into bone rotations. The physics is
plain math on :class:`Locomotion`, and unit-tested.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum

from .mathx import Vec2

GRAVITY = 2200.0  # px/s²: at ~320 px for a ~1.9 m model, a bit over real gravity
WALK_SPEED = 200.0  # px/s
WALK_ACCEL = 1200.0  # how fast walking speeds up and slows down, px/s²
SKID_DECEL = 2500.0  # how fast a skid along the floor stops, px/s²
AIR_DRAG = 0.2  # air resistance on moving, 1/s…
SPIN_DRAG = 0.3  # …and on spinning, 1/s
JUMP_SPEED = 750.0  # launch speed of a jump, px/s (≈ 130 px high)
MAX_THROW_SPEED = 3000.0  # the hardest the user can throw, px/s…
MAX_SPIN = 25.0  # …and the fastest spin, rad/s
ARRIVE_EPS = 2.0  # how close (px) to the walk target counts as arrived
MAX_STEP = 1.0 / 240.0  # the longest physics step, s
# Standing, the walls keep this much of the body on screen (a share of the
# height, either side of the feet): the body, not the arms.
BODY_HALF_WIDTH = 0.25
DRAG_THRESHOLD = 6.0  # pointer travel (px) that turns a press into a drag
TURN_RATE = 10.0  # how fast the model turns toward where it's going, 1/s

# ---- the body as a rigid body ----------------------------------------------------------------
COM_HEIGHT = 0.55  # where the weight sits above the feet, a share of the height
BODY_RADIUS = 0.12  # the body as a capsule, feet to head: its radius
INERTIA = 0.09  # moment of inertia about the centre of mass, per unit mass, in height²
FRICTION = 0.5  # contacts: friction coefficient…
BOUNCE = 0.3  # …and the share of speed kept in a bounce
LAND_ANGLE = 0.6  # coming down tilted less than this (rad)…
LAND_SPIN = 6.0  # …and spinning slower than this (rad/s), it lands on its feet…
FALL_OVER_SPEED = 2400.0  # …unless it lands faster than this (px/s)
HANG_DAMPING = 0.9  # swinging from the pointer loses this much spin, 1/s
ROLL_SECONDS = 0.9  # seconds to get up from lying, or to lie down
SEAT_RATE = 3.0  # how fast it sits down or stands back up, 1/s
SIT_DROP = 0.46  # sitting, the hips drop this far (a share of the height)

# ---- limbs --------------------------------------------------------------------------------------
ARM_LENGTH = 0.3  # arms and legs as pendulums: lengths as shares of the height
LEG_LENGTH = 0.45
ARM_REST = (0.3, -0.3)  # screen angles of the relaxed arms (left, right), rad
AIR_PUSH = 0.9  # how hard passing air pushes the limbs, per px/s of speed
# The most acceleration the limbs feel: a collision stops the body in one
# step; that should read as a jolt, not an explosion.
MAX_FELT = 6.0 * GRAVITY
LEAN = 9.0e-5  # leaning into acceleration while standing, rad per px/s²
HEAD_LAG = 4.0e-5  # the head tips back against acceleration, rad per px/s²
CROUCH_KICK = 0.004  # how much a landing bends the knees, per px/s of impact
CROUCH_DROP = 0.12  # how far the hips sink in a full squat, a share of the height


class Stance(Enum):
    STANDING = "standing"  # on its feet: balancing upright, walking
    SITTING = "sitting"  # sitting on the floor
    LYING = "lying"  # on the floor — knocked over, or napping
    ROLLING = "rolling"  # getting up from the floor, or lying down (see ``Rolling``)
    FLYING = "flying"  # free in the air, tumbling as it likes
    CARRIED = "carried"  # hanging from the pointer


@dataclass
class Rolling:
    """Getting up (``to`` = 0) or lying down (``to`` = ±π/2), ``t`` 0 → 1."""

    start: float
    to: float
    t: float = 0.0


@dataclass
class Spring:
    """A damped oscillator's state: position and velocity."""

    x: float = 0.0
    v: float = 0.0

    def push(self, h: float, force: float) -> None:
        """One semi-implicit Euler step under ``force`` (per unit mass)."""
        self.v += force * h
        self.x += self.v * h


class Looseness(Enum):
    """Planted legs on the ground, loose in the air, floppy when carried
    (except the legs it's held by)."""

    STANDING = 0
    FLYING = 1
    CARRIED = 2
    CARRIED_BY_LEGS = 3


# (muscle tone 1/s², damping 1/s) for (arms, legs).
TONE = {
    Looseness.STANDING: ((60.0, 10.0), (400.0, 40.0)),
    Looseness.FLYING: ((12.0, 2.5), (40.0, 6.0)),
    Looseness.CARRIED: ((5.0, 1.2), (5.0, 1.2)),
    Looseness.CARRIED_BY_LEGS: ((5.0, 1.2), (400.0, 40.0)),
}


@dataclass
class Limbs:
    """How the limbs ride along with the motion. Angles are in the screen
    plane, radians, counter-clockwise as the viewer sees it, from the
    relaxed pose."""

    arms: list[Spring] = field(default_factory=lambda: [Spring(), Spring()])  # left, right
    legs: list[Spring] = field(default_factory=lambda: [Spring(), Spring()])
    head: Spring = field(default_factory=Spring)  # tipping back against sudden moves
    crouch: Spring = field(default_factory=Spring)  # knees: 0 straight … 1 a deep squat

    def settling(self) -> bool:
        """Still swinging — worth drawing at the full frame rate."""
        return any(abs(s.v) > 0.05 for s in (self.head, self.crouch, *self.arms, *self.legs))

    def step(self, h: float, acc: Vec2, vel: Vec2, loose: Looseness, height: float, angle: float) -> None:
        # What a limb feels in the body's frame: gravity minus the body's own
        # acceleration (a jolt swings it the other way; free fall is
        # weightless), plus the passing air pushing back.
        felt = (Vec2(0.0, GRAVITY) - acc).clamp_length_max(MAX_FELT) - vel * AIR_PUSH
        (arm_muscle, arm_damping), (leg_muscle, leg_damping) = TONE[loose]
        for arm, rest in zip(self.arms, ARM_REST):
            hang_limb(arm, h, felt, angle, rest, ARM_LENGTH * height, arm_muscle, arm_damping)
        for leg in self.legs:
            hang_limb(leg, h, felt, angle, 0.0, LEG_LENGTH * height, leg_muscle, leg_damping)
        head_target = max(-0.25, min(0.25, acc.x * HEAD_LAG))
        self.head.push(h, 120.0 * (head_target - self.head.x) - 16.0 * self.head.v)
        self.crouch.push(h, -180.0 * self.crouch.x - 18.0 * self.crouch.v)
        if self.crouch.x < 0.0:
            self.crouch = Spring()
        self.crouch.x = min(self.crouch.x, 1.0)


def hang_limb(limb: Spring, h: float, felt: Vec2, angle: float, rest: float, length: float, muscle: float,
              damping: float) -> None:
    """One limb hanging from its joint on a body rolled by ``angle``: pulled
    along ``felt`` like a pendulum, held in its pose by muscle, damped."""
    pull_field = math.atan2(felt.x, felt.y)
    pull = (-(felt.length() / length) * math.sin(angle + rest + limb.x - pull_field)
            # Standing still, the relaxed pose is the balance point.
            + (GRAVITY / length) * math.sin(rest))
    limb.push(h, pull - muscle * limb.x - damping * limb.v)
    # Shoulders and hips reach about overhead, and no further.
    limb.x = max(-3.0, min(3.0, limb.x))


class Locomotion:
    """The character's body."""

    def __init__(self, seed: int | None = None) -> None:
        self.com = Vec2()  # centre of mass, logical px (origin top-left, +y down)
        self.vel = Vec2()  # px/s
        # Roll in the screen plane, rad, counter-clockwise as the viewer sees
        # it; 0 is upright. Unbounded: it can turn all the way round.
        self.angle = 0.0
        self.spin = 0.0  # roll rate, rad/s, counter-clockwise
        self.stance = Stance.FLYING
        self.rolling: Rolling | None = None  # while ROLLING
        self.seat = 0.0  # how far it has sat down: 0 standing … 1 sitting
        self.napping = False  # lying down on purpose (eyes closed), not knocked over
        self.target: float | None = None  # where it's walking to, a share of the arena's width
        self.held = False  # busy (chatting, clicking for the brain): no wandering
        self.idle_timer = 4.0  # seconds to the next idle decision
        self.limbs = Limbs()
        self.sit_side = 1.0  # which way it sits: +1 facing right, -1 left
        # Carried: the grab point from the centre of mass, in the body's own
        # (unrolled) frame; and the pointer's position, velocity, acceleration.
        self.grip: Vec2 | None = None
        self.pointer = (Vec2(), Vec2(), Vec2())
        self.acc = Vec2()  # over the last step, px/s² — what the limbs feel
        self.arena = Vec2()
        self.height = 320.0
        self.placed = False  # it drops in from the top the first time
        # xorshift32 for idle decisions; non-zero, and not the same each run.
        self.rng = (seed if seed is not None else time.time_ns() & 0xFFFFFFFF) | 1

    def next_rand(self) -> float:
        """[0, 1)."""
        x = self.rng
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.rng = x
        return x / 4294967296.0

    # ---- what it is doing -----------------------------------------------------------------

    def feet(self, height: float) -> Vec2:
        """Where the feet are (the model's origin), for a body ``height`` px tall."""
        return self.com + rotate(Vec2(0.0, COM_HEIGHT * height), self.angle)

    def grounded(self) -> bool:
        """On the floor in any way: standing, sitting, lying or getting up."""
        return self.stance not in (Stance.FLYING, Stance.CARRIED)

    def carried(self) -> bool:
        return self.stance == Stance.CARRIED

    def is_idle(self) -> bool:
        """Standing still with nowhere to go."""
        return self.stance == Stance.STANDING and self.seat < 0.01 and self.target is None and self.vel.x == 0.0

    def at_rest(self) -> bool:
        """Nothing moving that needs drawing: settled standing, sitting or lying."""
        if self.stance == Stance.STANDING:
            still = self.is_idle() and abs(self.spin) < 0.02
        elif self.stance == Stance.SITTING:
            still = self.seat > 0.99
        else:
            still = self.stance == Stance.LYING
        return still and not self.limbs.settling()

    def facing(self) -> float:
        """Which way to face: ±1 walking right/left, part-way round when
        sitting (a three-quarter view toward the middle of the screen), 0
        (the viewer) otherwise."""
        if self.stance == Stance.STANDING and abs(self.vel.x) > 20.0:
            return math.copysign(1.0, self.vel.x)
        if self.stance in (Stance.STANDING, Stance.SITTING):
            return self.sit_side * 0.65 * self.seat
        return 0.0

    # ---- what it's asked to do -----------------------------------------------------------------

    def walk_to(self, frac: float) -> None:
        """Walk to a share [0, 1] of the arena's width — getting up first if
        it's sitting or lying."""
        self.target = max(0.0, min(1.0, frac))
        self.stand_up()

    def jump(self) -> None:
        """Hop — from its feet. Sitting or lying, it gets up instead."""
        if self.stance == Stance.STANDING and self.seat < 0.05:
            self.vel = Vec2(self.vel.x, -JUMP_SPEED)
            self.stance = Stance.FLYING
        else:
            self.stand_up()

    def sit(self) -> None:
        """Sit down on the floor (from standing), facing into the screen."""
        if self.stance == Stance.STANDING:
            self.stance = Stance.SITTING
            self.target = None
            self.sit_side = 1.0 if self.com.x < self.arena.x * 0.5 else -1.0
            self.idle_timer = 6.0 + self.next_rand() * 9.0

    def lie_down(self) -> None:
        """Lie down for a nap (from standing)."""
        if self.stance == Stance.STANDING and self.seat < 0.05:
            side = math.pi / 2 if self.next_rand() < 0.5 else -math.pi / 2
            self.stance, self.rolling = Stance.ROLLING, Rolling(self.angle, side)
            self.target = None
            self.napping = True
            self.idle_timer = 10.0 + self.next_rand() * 15.0

    def stand_up(self) -> None:
        """Back on its feet, from sitting or lying."""
        if self.stance == Stance.SITTING:
            self.stance = Stance.STANDING
        elif self.stance == Stance.LYING:
            self.stance, self.rolling = Stance.ROLLING, Rolling(wrap(self.angle), 0.0)

    def drag_to(self, pointer: Vec2, dt: float) -> None:
        """Follow the pointer while carried, hanging from wherever it was
        grabbed. Its speed is tracked, smoothed, so letting go throws it."""
        if self.stance != Stance.CARRIED:
            self.grip = rotate(pointer - self.com, -self.angle)
            self.pointer = (pointer, Vec2(), Vec2())
            self.stance = Stance.CARRIED
            self.target = None
            self.napping = False
            return
        last, last_vel, last_acc = self.pointer
        if dt > 0.0:
            vel = last_vel.lerp((pointer - last) / dt, 0.5)
            acc = last_acc.lerp((vel - last_vel) / dt, 0.5)
            self.pointer = (pointer, vel, acc)

    def release(self) -> None:
        """Let go after a drag: it flies off with the swing it had."""
        if self.stance == Stance.CARRIED:
            self.stance = Stance.FLYING
            self.grip = None
            self.vel = self.vel.clamp_length_max(MAX_THROW_SPEED)
            self.spin = max(-MAX_SPIN, min(MAX_SPIN, self.spin))

    # ---- the simulation -----------------------------------------------------------------------------

    def step(self, dt: float, arena: Vec2, body: Vec2) -> None:
        """Advance by ``dt`` seconds, for a body ``body`` px (standing
        half-width, height) in an arena ``arena`` px."""
        if arena.x <= 0.0 or arena.y <= 0.0:
            return
        self.arena = arena
        self.height = body.y
        if not self.placed:
            # Drop in from the top, centred.
            self.com = Vec2(arena.x * 0.5, (1.0 - COM_HEIGHT) * body.y)
            self.placed = True
        # After a hitch, don't teleport; split the rest into small steps.
        dt = min(dt, 0.1)
        steps = max(1, math.ceil(dt / MAX_STEP))
        h = dt / steps
        for _ in range(steps):
            before = self.vel
            if self.stance in (Stance.STANDING, Stance.SITTING):
                self.stand(h, arena, body)
            elif self.stance == Stance.FLYING:
                self.fly(h, arena, body.y)
            elif self.stance == Stance.CARRIED:
                self.hang(h, arena, body.y)
            elif self.stance == Stance.LYING:
                self.rest(arena, body.y)
            else:
                self.roll(h, arena, body.y)
            self.acc = self.pointer[2] if self.carried() else (self.vel - before) / h
            sitting = 1.0 if self.stance == Stance.SITTING else 0.0
            self.seat += (sitting - self.seat) * min(SEAT_RATE * h, 1.0)
            if self.stance == Stance.FLYING:
                loose = Looseness.FLYING
            elif self.stance == Stance.CARRIED:
                # Held by a leg, the legs are what it hangs from: taut.
                by_legs = self.grip is not None and self.grip.y > 0.1 * body.y
                loose = Looseness.CARRIED_BY_LEGS if by_legs else Looseness.CARRIED
            else:
                loose = Looseness.STANDING
            self.limbs.step(h, self.acc, self.vel, loose, body.y, self.angle)

    def stand(self, h: float, arena: Vec2, body: Vec2) -> None:
        """On its feet (or sitting): feet on the floor, walking, leaning."""
        left, right, _, floor = limits(arena, body)
        feet = self.feet(body.y)
        if feet.y < floor - 0.5:  # the floor moved away under it: fall
            self.stance = Stance.FLYING
            return
        fx = feet.x
        vx = self.vel.x
        before = vx
        skidding = abs(vx) > WALK_SPEED
        walking = self.stance == Stance.STANDING and self.seat < 0.05
        target_x = max(left, min(right, self.target * arena.x)) if self.target is not None else None
        if target_x is not None and walking and not skidding:
            dx = target_x - fx
            if abs(dx) <= ARRIVE_EPS:
                fx, vx, self.target = target_x, 0.0, None
            else:
                # Ease in so it stops on the target (v² = 2ad).
                speed = min(WALK_SPEED, math.sqrt(2.0 * WALK_ACCEL * abs(dx)))
                vx = approach(vx, math.copysign(speed, dx), WALK_ACCEL * h)
        else:
            # Skidding, sitting or nowhere to go: friction stops it.
            vx = approach(vx, 0.0, SKID_DECEL * h)
        fx += vx * h
        if fx < left or fx > right:
            fx = max(left, min(right, fx))
            vx = 0.0
        self.vel = Vec2(vx, 0.0)
        # Balance: lean into acceleration; upright when sitting.
        lean = max(-0.18, min(0.18, -(vx - before) / h * LEAN)) if self.stance == Stance.STANDING else 0.0
        self.spin += (80.0 * (lean - wrap(self.angle)) - 14.0 * self.spin) * h
        self.angle += self.spin * h
        self.com = Vec2(fx, floor) + rotate(Vec2(0.0, -COM_HEIGHT * body.y), self.angle)

    def fly(self, h: float, arena: Vec2, height: float) -> None:
        """Free flight: gravity and air, then collisions. It lands on its feet
        if it comes down upright; otherwise it tumbles until it lies still."""
        self.vel = Vec2(self.vel.x, self.vel.y + GRAVITY * h) * math.exp(-AIR_DRAG * h)
        self.spin *= math.exp(-SPIN_DRAG * h)
        self.com = self.com + self.vel * h
        self.angle += self.spin * h

        radius = BODY_RADIUS * height
        inertia = INERTIA * height * height
        on_floor = False
        for normal, plane in planes(arena):
            for end, is_feet in self.ends(height):
                depth = radius - ((self.com + end).dot(normal) - plane)
                if depth <= 0.0:
                    continue
                floor = normal.y < 0.0
                if floor and is_feet and self.lands_on_feet():
                    self.land(arena.y, height)
                    return
                if floor and is_feet and self.vel.y >= FALL_OVER_SPEED:
                    # Too hard: the knees give way and it topples.
                    self.spin += 3.0 if self.next_rand() < 0.5 else -3.0
                    self.limbs.crouch.v += self.vel.y * CROUCH_KICK
                on_floor |= floor
                self.com = self.com + normal * depth
                self.impulse(normal, end - normal * radius, inertia)

        # Come to rest: on its feet if upright, else lying there.
        if on_floor and self.vel.length() < 40.0 and abs(self.spin) < 0.8:
            if abs(wrap(self.angle)) < 0.35:
                self.land(arena.y, height)
            else:
                self.stance = Stance.LYING
                self.napping = False
                self.vel = Vec2()
                self.spin = 0.0
                self.idle_timer = 1.5 + self.next_rand() * 1.5

    def lands_on_feet(self) -> bool:
        return abs(wrap(self.angle)) < LAND_ANGLE and abs(self.spin) < LAND_SPIN and self.vel.y < FALL_OVER_SPEED

    def land(self, floor: float, height: float) -> None:
        """Touch down on its feet: the knees give with the impact, and
        whatever tilt and spin it had becomes a stumble it balances out of."""
        self.limbs.crouch.v += max(self.vel.y, 0.0) * CROUCH_KICK
        self.angle = wrap(self.angle)
        self.vel = Vec2(self.vel.x, 0.0)
        self.stance = Stance.STANDING
        self.com = Vec2(self.com.x, self.com.y + floor - self.feet(height).y)

    def impulse(self, normal: Vec2, offset: Vec2, inertia: float) -> None:
        """A collision impulse at ``offset`` from the centre of mass along the
        contact normal, with bounce and Coulomb friction; it both stops and
        spins the body."""
        lever = Vec2(offset.y, -offset.x)  # how a point on the body moves as it turns
        approach_speed = (self.vel + lever * self.spin).dot(normal)
        if approach_speed >= 0.0:
            return
        bounce = BOUNCE if approach_speed < -60.0 else 0.0  # slow contacts don't bounce
        turn = lever.dot(normal)
        push = -(1.0 + bounce) * approach_speed / (1.0 + turn * turn / inertia)
        self.vel = self.vel + normal * push
        self.spin += push * turn / inertia

        tangent = Vec2(-normal.y, normal.x)
        slide = (self.vel + lever * self.spin).dot(tangent)
        turn = lever.dot(tangent)
        grip = max(-FRICTION * push, min(FRICTION * push, -slide / (1.0 + turn * turn / inertia)))
        self.vel = self.vel + tangent * grip
        self.spin += grip * turn / inertia

    def hang(self, h: float, arena: Vec2, height: float) -> None:
        """Carried: a pendulum hanging from the grab point, driven by gravity
        and the pointer's own acceleration. Nothing holds it upright."""
        pivot, pivot_vel, pivot_acc = self.pointer
        grip = self.grip or Vec2()
        arm = rotate(-grip, self.angle)  # pivot → centre of mass
        felt = Vec2(0.0, GRAVITY) - pivot_acc.clamp_length_max(MAX_FELT)
        torque = felt.x * arm.y - felt.y * arm.x
        inertia = INERTIA * height * height + arm.length_squared()
        self.spin += (torque / inertia - HANG_DAMPING * self.spin) * h
        self.angle += self.spin * h

        arm = rotate(-grip, self.angle)
        self.com = pivot + arm
        self.vel = pivot_vel + Vec2(arm.y, -arm.x) * self.spin
        # Dragged into the floor or a wall, it slides along rather than through.
        radius = BODY_RADIUS * height
        for normal, plane in planes(arena):
            for end, _ in self.ends(height):
                depth = radius - ((self.com + end).dot(normal) - plane)
                if depth > 0.0:
                    self.com = self.com + normal * depth

    def rest(self, arena: Vec2, height: float) -> None:
        """Lying still on the floor."""
        self.vel = Vec2()
        self.spin = 0.0
        self.com = Vec2(self.com.x, arena.y - self.lowest(height))

    def roll(self, h: float, arena: Vec2, height: float) -> None:
        """Getting up or lying down: turning between the two with the lowest
        point of the body kept on the floor."""
        r = self.rolling
        if r is None:
            self.stance = Stance.STANDING
            return
        r.t = min(r.t + h / ROLL_SECONDS, 1.0)
        eased = r.t * r.t * (3.0 - 2.0 * r.t)
        self.angle = r.start + (r.to - r.start) * eased
        self.vel = Vec2()
        self.spin = 0.0
        self.com = Vec2(self.com.x, arena.y - self.lowest(height))
        if r.t >= 1.0:
            self.stance = Stance.STANDING if r.to == 0.0 else Stance.LYING
            self.napping = self.napping and r.to != 0.0
            self.rolling = None

    def ends(self, height: float) -> tuple[tuple[Vec2, bool], tuple[Vec2, bool]]:
        """The capsule's ends (feet, head) from the centre of mass, each with
        whether it is the feet end."""
        radius = BODY_RADIUS * height
        feet = Vec2(0.0, COM_HEIGHT * height - radius)
        head = Vec2(0.0, -(1.0 - COM_HEIGHT) * height + radius)
        return (rotate(feet, self.angle), True), (rotate(head, self.angle), False)

    def lowest(self, height: float) -> float:
        """How far below the centre of mass the body reaches."""
        (a, _), (b, _) = self.ends(height)
        return max(a.y, b.y) + BODY_RADIUS * height

    def wander(self, dt: float) -> None:
        """Idle behaviour when there's nothing else to do: stroll, sit down,
        take a nap — and get back up after a while."""
        self.idle_timer -= dt
        if self.idle_timer > 0.0:
            return
        if self.stance == Stance.STANDING and self.is_idle() and not self.held:
            choice = self.next_rand()
            if choice < 0.5:
                self.walk_to(self.next_rand())
            elif choice < 0.62:
                self.sit()
            elif choice < 0.7:
                self.lie_down()
            if self.stance == Stance.STANDING:
                self.idle_timer = 4.0 + self.next_rand() * 6.0  # next decision in 4–10 s
        elif self.stance in (Stance.SITTING, Stance.LYING):
            # Knocked over, or done sitting or napping: back up, busy or not.
            self.stand_up()
            self.idle_timer = 4.0 + self.next_rand() * 6.0


def planes(arena: Vec2) -> tuple[tuple[Vec2, float], ...]:
    """The arena's walls as (inward normal, offset) half-planes: floor,
    ceiling, left, right. A point is inside when ``point · normal ≥ offset``."""
    return ((Vec2(0.0, -1.0), -arena.y), (Vec2(0.0, 1.0), 0.0), (Vec2(1.0, 0.0), 0.0), (Vec2(-1.0, 0.0), -arena.x))


def wrap(angle: float) -> float:
    """An angle wrapped into (-π, π]."""
    wrapped = (angle + math.pi) % math.tau - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def rotate(v: Vec2, angle: float) -> Vec2:
    """Turn a screen vector (+y down) counter-clockwise as the viewer sees it
    — the same turn as a world rotation about +Z."""
    s, c = math.sin(angle), math.cos(angle)
    return Vec2(v.x * c + v.y * s, -v.x * s + v.y * c)


def limits(arena: Vec2, body: Vec2) -> tuple[float, float, float, float]:
    """Where the feet may be while standing: (left, right, top, floor). The
    walls keep the body's width on screen, the ceiling its head."""
    left = min(body.x, arena.x * 0.5)
    right = max(arena.x - body.x, left)
    top = min(body.y, arena.y)
    return left, right, top, arena.y


def approach(start: float, to: float, max_delta: float) -> float:
    """Move ``start`` toward ``to`` by at most ``max_delta``."""
    return start + max(-max_delta, min(max_delta, to - start))


def body_size(height_px: float) -> Vec2:
    """The standing body (half-width, height) for a character, px."""
    return Vec2(height_px * BODY_HALF_WIDTH, height_px)
