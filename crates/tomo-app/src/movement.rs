//! Locomotion — a small physics engine for the character.
//!
//! The character is a rigid body in screen space (logical pixels, origin
//! top-left, +y down) inside the [`Arena`]: the overlay or window. It has a
//! centre of mass, a velocity, a roll angle (unbounded — it can turn all the
//! way round) and a spin, and when it isn't on its feet it collides as a
//! capsule with the floor, walls and ceiling, through impulses with bounce and
//! friction. Every frame, in small fixed steps, depending on its [`Stance`]:
//!   • standing, it balances on its feet, walks toward a target the brain (or
//!     idle wandering) picks, and leans into acceleration;
//!   • it can sit on the floor, or lie down for a nap, and get back up;
//!   • flying, it tumbles freely: a throw can send it spinning, and it lands
//!     on its feet only if it comes down nearly upright — otherwise it falls
//!     over, lies there a moment, and gets up;
//!   • carried, it hangs from the point it was grabbed like a pendulum, free to
//!     swing all the way round; letting go throws it with that swing;
//!   • its limbs ride along ([`Limbs`]): arms and legs hang like pendulums in
//!     what they feel — gravity minus the body's own acceleration, plus the air
//!     pushing back — so they dangle and swing when carried, float up in a fall
//!     and lag behind a sudden start; the knees give on landing.
//! animation.rs only turns all this into bone rotations.
//!
//! The physics is plain math on [`Locomotion`] and is unit-tested; only the
//! systems below it touch Bevy. The camera (main.rs) is orthographic with one
//! world unit per logical pixel, so screen ↔ world is just a flip and a shift.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;

use crate::bridge::{AnimateEvent, WalkToEvent};
use crate::character::{box_corners, Character};
use crate::window::{pixel_rect, Busy, InputRegion};

/// Gravity, px/s². At ~320 px for a ~1.9 m model that's a bit over real
/// gravity — weighty rather than floaty at desktop scale.
const GRAVITY: f32 = 2200.0;
/// Walking speed, px/s.
pub const WALK_SPEED: f32 = 200.0;
/// How fast walking speeds up and slows down, px/s².
const WALK_ACCEL: f32 = 1200.0;
/// How fast a skid along the floor stops, px/s².
const SKID_DECEL: f32 = 2500.0;
/// Air resistance on moving (1/s) and on spinning (1/s).
const AIR_DRAG: f32 = 0.2;
const SPIN_DRAG: f32 = 0.3;
/// Launch speed of a jump, px/s (≈ 130 px high).
const JUMP_SPEED: f32 = 750.0;
/// The hardest the user can throw (px/s) and the fastest spin (rad/s).
const MAX_THROW_SPEED: f32 = 3000.0;
const MAX_SPIN: f32 = 25.0;
/// How close (px) to the walk target counts as arrived.
const ARRIVE_EPS: f32 = 2.0;
/// Longest physics step, s; each frame is split into steps no longer than this.
const MAX_STEP: f32 = 1.0 / 240.0;
/// Standing, the walls keep this much of the body on screen (a share of the
/// height, either side of the feet): the body, not the arms.
const BODY_HALF_WIDTH: f32 = 0.25;
/// Pointer travel (px) that turns a press on the character into a drag.
const DRAG_THRESHOLD: f32 = 6.0;
/// How fast the model turns toward where it's going, 1/s.
const TURN_RATE: f32 = 10.0;

// ---- the body as a rigid body ----------------------------------------------

/// Where the body's weight sits above the feet, as a share of its height.
const COM_HEIGHT: f32 = 0.55;
/// The body as a capsule, feet to head: its radius as a share of the height.
const BODY_RADIUS: f32 = 0.12;
/// Moment of inertia about the centre of mass, per unit mass, in height².
const INERTIA: f32 = 0.09;
/// Contacts: friction coefficient, and the share of speed kept in a bounce.
const FRICTION: f32 = 0.5;
const BOUNCE: f32 = 0.3;
/// Coming down tilted less than this (rad) and spinning slower than this
/// (rad/s), it lands on its feet…
const LAND_ANGLE: f32 = 0.6;
const LAND_SPIN: f32 = 6.0;
/// …unless it lands faster than this (px/s): then its knees give way.
const FALL_OVER_SPEED: f32 = 2400.0;
/// Swinging from the pointer loses this much spin, 1/s: lively when shaken,
/// settled within a few swings.
const HANG_DAMPING: f32 = 0.9;
/// Seconds to get up from lying, or to lie down.
const ROLL_SECONDS: f32 = 0.9;
/// How fast it sits down or stands back up, 1/s.
const SEAT_RATE: f32 = 3.0;
/// Sitting, the hips drop this far (a share of the height) to the floor.
const SIT_DROP: f32 = 0.46;

// ---- limbs --------------------------------------------------------------------

/// Arms and legs as pendulums: lengths as shares of the height.
const ARM_LENGTH: f32 = 0.3;
const LEG_LENGTH: f32 = 0.45;
/// Screen angles of the relaxed arms (left, right) against the body, rad.
const ARM_REST: [f32; 2] = [0.3, -0.3];
/// How hard passing air pushes the limbs, per px/s of speed.
const AIR_PUSH: f32 = 0.9;
/// The most acceleration the limbs feel. A collision stops the body in one
/// step; that should read as a jolt, not an explosion.
const MAX_FELT: f32 = 6.0 * GRAVITY;
/// Leaning into acceleration while standing, rad per px/s².
const LEAN: f32 = 9.0e-5;
/// The head tips back against acceleration, rad per px/s².
const HEAD_LAG: f32 = 4.0e-5;
/// How much a landing bends the knees, per px/s of impact.
const CROUCH_KICK: f32 = 0.004;
/// How far the hips sink in a full squat, as a share of the height.
const CROUCH_DROP: f32 = 0.12;

/// The area the character lives in — the overlay or window — in logical
/// pixels. The floor is its bottom edge, the walls its sides.
#[derive(Resource, Clone, Copy, Default)]
pub struct Arena {
    pub size: Vec2,
}

impl Arena {
    /// Screen position → world position on the z = 0 plane.
    fn to_world(self, p: Vec2) -> Vec3 {
        Vec3::new(p.x - self.size.x * 0.5, self.size.y * 0.5 - p.y, 0.0)
    }

    /// World position → screen position.
    fn to_screen(self, p: Vec3) -> Vec2 {
        Vec2::new(p.x + self.size.x * 0.5, self.size.y * 0.5 - p.y)
    }
}

/// What the body is doing, physically.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Stance {
    /// On its feet: balancing upright, walking.
    Standing,
    /// Sitting on the floor.
    Sitting,
    /// Lying on the floor — knocked over, or napping.
    Lying,
    /// Getting up from the floor (`to` = 0) or lying down (`to` = ±π/2),
    /// `t` going 0 → 1.
    Rolling { from: f32, to: f32, t: f32 },
    /// Free in the air, tumbling as it likes.
    Flying,
    /// Hanging from the pointer.
    Carried,
}

/// The character's body.
#[derive(Component)]
pub struct Locomotion {
    /// Centre of mass, logical px (origin top-left, +y down).
    pub com: Vec2,
    /// Its velocity, px/s.
    pub vel: Vec2,
    /// Roll in the screen plane, rad, counter-clockwise as the viewer sees
    /// it; 0 is upright. Unbounded: it can turn all the way round.
    pub angle: f32,
    /// Roll rate, rad/s, counter-clockwise.
    pub spin: f32,
    pub stance: Stance,
    /// How far it has sat down: 0 standing … 1 sitting.
    pub seat: f32,
    /// Lying down on purpose (eyes closed), rather than knocked over.
    pub napping: bool,
    /// Where it's walking to, as a fraction of the arena width [0,1].
    pub target: Option<f32>,
    /// Busy (chatting, or clicking something for the brain): no wandering.
    pub held: bool,
    /// Seconds until the next idle decision (wander, sit, nap, get up).
    pub idle_timer: f32,
    /// Its limbs, riding along with the motion.
    pub limbs: Limbs,
    /// Which way it sits: +1 facing right, -1 left.
    sit_side: f32,
    /// Carried: the grab point relative to the centre of mass, in the body's
    /// own (unrolled) frame.
    grip: Option<Vec2>,
    /// Carried: the pointer's position, velocity (px/s), acceleration (px/s²).
    pointer: (Vec2, Vec2, Vec2),
    /// Acceleration over the last step, px/s² — what the limbs feel.
    acc: Vec2,
    /// The arena and body size (height, px) of the last step.
    arena: Vec2,
    height: f32,
    /// Placed in the arena yet? It drops in from the top the first time.
    placed: bool,
    /// Tiny PRNG state for idle decisions (no external crate needed).
    rng: u32,
}

impl Default for Locomotion {
    fn default() -> Self {
        Self {
            com: Vec2::ZERO,
            vel: Vec2::ZERO,
            angle: 0.0,
            spin: 0.0,
            stance: Stance::Flying,
            seat: 0.0,
            napping: false,
            target: None,
            held: false,
            idle_timer: 4.0,
            limbs: Limbs::default(),
            sit_side: 1.0,
            grip: None,
            pointer: (Vec2::ZERO, Vec2::ZERO, Vec2::ZERO),
            acc: Vec2::ZERO,
            arena: Vec2::ZERO,
            height: 320.0,
            placed: false,
            rng: seed_from_time(),
        }
    }
}

/// A non-zero, time-derived seed for the xorshift RNG, so idle behaviour isn't
/// identical every launch. `| 1` guarantees xorshift's required non-zero state.
fn seed_from_time() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0x2545_F491)
        | 1
}

impl Locomotion {
    /// xorshift32 → f32 in [0,1). Deterministic, dependency-free.
    fn next_rand(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x as f32) / (u32::MAX as f32)
    }

    /// Where the feet are (the model's origin), for a body `height` px tall.
    pub fn feet(&self, height: f32) -> Vec2 {
        self.com + rotate(Vec2::new(0.0, COM_HEIGHT * height), self.angle)
    }

    /// On the floor in any way: standing, sitting, lying or getting up.
    pub fn grounded(&self) -> bool {
        !matches!(self.stance, Stance::Flying | Stance::Carried)
    }

    pub fn carried(&self) -> bool {
        self.stance == Stance::Carried
    }

    /// Walk to a horizontal screen fraction [0,1] — getting up first if it's
    /// sitting or lying.
    pub fn walk_to(&mut self, frac: f32) {
        self.target = Some(frac.clamp(0.0, 1.0));
        self.stand_up();
    }

    /// Standing still with nowhere to go, and not held.
    pub fn is_idle(&self) -> bool {
        self.stance == Stance::Standing
            && self.seat < 0.01
            && self.target.is_none()
            && self.vel.x == 0.0
    }

    /// Nothing moving that needs drawing: settled standing, sitting or lying.
    pub fn at_rest(&self) -> bool {
        let still = match self.stance {
            Stance::Standing => self.is_idle() && self.spin.abs() < 0.02,
            Stance::Sitting => self.seat > 0.99,
            Stance::Lying => true,
            _ => false,
        };
        still && !self.limbs.settling()
    }

    /// Which way to face: ±1 walking right/left, part-way round when sitting
    /// (a three-quarter view toward the middle of the screen), 0 (the
    /// viewer) otherwise.
    pub fn facing(&self) -> f32 {
        match self.stance {
            Stance::Standing if self.vel.x.abs() > 20.0 => self.vel.x.signum(),
            Stance::Standing | Stance::Sitting => self.sit_side * 0.65 * self.seat,
            _ => 0.0,
        }
    }

    /// Hop — from its feet. Sitting or lying, it gets up instead.
    pub fn jump(&mut self) {
        match self.stance {
            Stance::Standing if self.seat < 0.05 => {
                self.vel.y = -JUMP_SPEED;
                self.stance = Stance::Flying;
            }
            _ => self.stand_up(),
        }
    }

    /// Sit down on the floor (from standing), facing into the screen.
    pub fn sit(&mut self) {
        if self.stance == Stance::Standing {
            self.stance = Stance::Sitting;
            self.target = None;
            self.sit_side = if self.com.x < self.arena.x * 0.5 { 1.0 } else { -1.0 };
            self.idle_timer = 6.0 + self.next_rand() * 9.0;
        }
    }

    /// Lie down for a nap (from standing).
    pub fn lie_down(&mut self) {
        if self.stance == Stance::Standing && self.seat < 0.05 {
            let side = if self.next_rand() < 0.5 { FRAC_PI_2 } else { -FRAC_PI_2 };
            self.stance = Stance::Rolling { from: self.angle, to: side, t: 0.0 };
            self.target = None;
            self.napping = true;
            self.idle_timer = 10.0 + self.next_rand() * 15.0;
        }
    }

    /// Get back on its feet, from sitting or lying.
    pub fn stand_up(&mut self) {
        match self.stance {
            Stance::Sitting => self.stance = Stance::Standing,
            Stance::Lying => {
                let from = wrap(self.angle);
                self.stance = Stance::Rolling { from, to: 0.0, t: 0.0 };
            }
            _ => {}
        }
    }

    /// Follow the pointer while carried, hanging from wherever it was grabbed.
    /// Its speed is tracked, smoothed, so letting go throws the character.
    pub fn drag_to(&mut self, pointer: Vec2, dt: f32) {
        if self.stance != Stance::Carried {
            self.grip = Some(rotate(pointer - self.com, -self.angle));
            self.pointer = (pointer, Vec2::ZERO, Vec2::ZERO);
            self.stance = Stance::Carried;
            self.target = None;
            self.napping = false;
            return;
        }
        let (last, last_vel, last_acc) = self.pointer;
        if dt > 0.0 {
            let vel = last_vel.lerp((pointer - last) / dt, 0.5);
            let acc = last_acc.lerp((vel - last_vel) / dt, 0.5);
            self.pointer = (pointer, vel, acc);
        }
    }

    /// Let go after a drag: it flies off with the swing it had.
    pub fn release(&mut self) {
        if self.stance == Stance::Carried {
            self.stance = Stance::Flying;
            self.grip = None;
            self.vel = self.vel.clamp_length_max(MAX_THROW_SPEED);
            self.spin = self.spin.clamp(-MAX_SPIN, MAX_SPIN);
        }
    }

    /// Advance the simulation by `dt` seconds, for a body `body` px (standing
    /// half-width, height) in an arena `arena` px.
    pub fn step(&mut self, dt: f32, arena: Vec2, body: Vec2) {
        if arena.cmple(Vec2::ZERO).any() {
            return;
        }
        self.arena = arena;
        self.height = body.y;
        if !self.placed {
            // Drop in from the top, centred.
            self.com = Vec2::new(arena.x * 0.5, (1.0 - COM_HEIGHT) * body.y);
            self.placed = true;
        }
        // After a hitch, don't teleport; split the rest into small steps.
        let dt = dt.min(0.1);
        let steps = (dt / MAX_STEP).ceil().max(1.0);
        let h = dt / steps;
        for _ in 0..steps as u32 {
            let before = self.vel;
            match self.stance {
                Stance::Standing | Stance::Sitting => self.stand(h, arena, body),
                Stance::Flying => self.fly(h, arena, body.y),
                Stance::Carried => self.hang(h, arena, body.y),
                Stance::Lying => self.rest(arena, body.y),
                Stance::Rolling { .. } => self.roll(h, arena, body.y),
            }
            self.acc = if self.carried() { self.pointer.2 } else { (self.vel - before) / h };
            let sitting = if self.stance == Stance::Sitting { 1.0 } else { 0.0 };
            self.seat += (sitting - self.seat) * (SEAT_RATE * h).min(1.0);
            let loose = match self.stance {
                Stance::Flying => Looseness::Flying,
                // Held by a leg, the legs are what it hangs from: taut.
                Stance::Carried if self.grip.is_some_and(|g| g.y > 0.1 * body.y) => {
                    Looseness::CarriedByLegs
                }
                Stance::Carried => Looseness::Carried,
                _ => Looseness::Standing,
            };
            self.limbs.step(h, self.acc, self.vel, loose, body.y, self.angle);
        }
    }

    /// On its feet (or sitting): feet on the floor, walking, leaning.
    fn stand(&mut self, h: f32, arena: Vec2, body: Vec2) {
        let (left, right, _, floor) = limits(arena, body);
        let mut feet = self.feet(body.y);
        // The floor moved away under it (a bar was hidden): fall.
        if feet.y < floor - 0.5 {
            self.stance = Stance::Flying;
            return;
        }
        feet.y = floor;
        self.vel.y = 0.0;

        let before = self.vel.x;
        let skidding = self.vel.x.abs() > WALK_SPEED;
        let walking = self.stance == Stance::Standing && self.seat < 0.05;
        match self.target.map(|f| (f * arena.x).clamp(left, right)) {
            Some(target_x) if walking && !skidding => {
                let dx = target_x - feet.x;
                if dx.abs() <= ARRIVE_EPS {
                    feet.x = target_x;
                    self.vel.x = 0.0;
                    self.target = None;
                } else {
                    // Ease in so it stops on the target (v² = 2ad).
                    let speed = WALK_SPEED.min((2.0 * WALK_ACCEL * dx.abs()).sqrt());
                    self.vel.x = approach(self.vel.x, dx.signum() * speed, WALK_ACCEL * h);
                }
            }
            // Skidding, sitting or nowhere to go: friction stops it.
            _ => self.vel.x = approach(self.vel.x, 0.0, SKID_DECEL * h),
        }
        feet.x += self.vel.x * h;
        if feet.x < left || feet.x > right {
            feet.x = feet.x.clamp(left, right);
            self.vel.x = 0.0;
        }

        // Balance: lean into acceleration; upright when sitting.
        let lean = if self.stance == Stance::Standing {
            (-(self.vel.x - before) / h * LEAN).clamp(-0.18, 0.18)
        } else {
            0.0
        };
        self.spin += (80.0 * (lean - wrap(self.angle)) - 14.0 * self.spin) * h;
        self.angle += self.spin * h;
        self.com = feet + rotate(Vec2::new(0.0, -COM_HEIGHT * body.y), self.angle);
    }

    /// Free flight: gravity and air, then collisions. It lands on its feet if
    /// it comes down upright; otherwise it tumbles until it lies still.
    fn fly(&mut self, h: f32, arena: Vec2, height: f32) {
        self.vel.y += GRAVITY * h;
        self.vel *= (-AIR_DRAG * h).exp();
        self.spin *= (-SPIN_DRAG * h).exp();
        self.com += self.vel * h;
        self.angle += self.spin * h;

        let radius = BODY_RADIUS * height;
        let inertia = INERTIA * height * height;
        let mut on_floor = false;
        for (normal, plane) in planes(arena) {
            for (end, is_feet) in self.ends(height) {
                let point = self.com + end;
                let depth = radius - (point.dot(normal) - plane);
                if depth <= 0.0 {
                    continue;
                }
                let floor = normal.y < 0.0;
                if floor && is_feet && self.lands_on_feet() {
                    return self.land(arena.y, height);
                }
                if floor && is_feet && self.vel.y >= FALL_OVER_SPEED {
                    // Too hard: the knees give way and it topples.
                    let side = if self.next_rand() < 0.5 { 1.0 } else { -1.0 };
                    self.spin += side * 3.0;
                    self.limbs.crouch.v += self.vel.y * CROUCH_KICK;
                }
                on_floor |= floor;
                self.com += normal * depth;
                self.impulse(normal, end - normal * radius, inertia);
            }
        }

        // Come to rest: on its feet if upright, else lying there.
        if on_floor && self.vel.length() < 40.0 && self.spin.abs() < 0.8 {
            if wrap(self.angle).abs() < 0.35 {
                self.land(arena.y, height);
            } else {
                self.stance = Stance::Lying;
                self.napping = false;
                self.vel = Vec2::ZERO;
                self.spin = 0.0;
                self.idle_timer = 1.5 + self.next_rand() * 1.5;
            }
        }
    }

    fn lands_on_feet(&self) -> bool {
        wrap(self.angle).abs() < LAND_ANGLE
            && self.spin.abs() < LAND_SPIN
            && self.vel.y < FALL_OVER_SPEED
    }

    /// Touch down on its feet: the knees give with the impact, and whatever
    /// tilt and spin it had becomes a stumble it balances out of.
    fn land(&mut self, floor: f32, height: f32) {
        self.limbs.crouch.v += self.vel.y.max(0.0) * CROUCH_KICK;
        self.angle = wrap(self.angle);
        self.vel.y = 0.0;
        self.stance = Stance::Standing;
        self.com.y += floor - self.feet(height).y;
    }

    /// A collision impulse at `offset` from the centre of mass along the
    /// contact normal, with bounce and Coulomb friction; it both stops and
    /// spins the body.
    fn impulse(&mut self, normal: Vec2, offset: Vec2, inertia: f32) {
        // How a point on the body moves as it turns.
        let lever = Vec2::new(offset.y, -offset.x);
        let approach_speed = (self.vel + self.spin * lever).dot(normal);
        if approach_speed >= 0.0 {
            return;
        }
        // Slow contacts don't bounce, or resting would jitter.
        let bounce = if approach_speed < -60.0 { BOUNCE } else { 0.0 };
        let turn = lever.dot(normal);
        let push = -(1.0 + bounce) * approach_speed / (1.0 + turn * turn / inertia);
        self.vel += normal * push;
        self.spin += push * turn / inertia;

        let tangent = Vec2::new(-normal.y, normal.x);
        let slide = (self.vel + self.spin * lever).dot(tangent);
        let turn = lever.dot(tangent);
        let grip = (-slide / (1.0 + turn * turn / inertia)).clamp(-FRICTION * push, FRICTION * push);
        self.vel += tangent * grip;
        self.spin += grip * turn / inertia;
    }

    /// Carried: a pendulum hanging from the grab point, driven by gravity and
    /// the pointer's own acceleration. Nothing holds it upright.
    fn hang(&mut self, h: f32, arena: Vec2, height: f32) {
        let (pivot, pivot_vel, pivot_acc) = self.pointer;
        let grip = self.grip.unwrap_or_default();
        let arm = rotate(-grip, self.angle); // pivot → centre of mass
        let felt = Vec2::new(0.0, GRAVITY) - pivot_acc.clamp_length_max(MAX_FELT);
        let torque = felt.x * arm.y - felt.y * arm.x;
        let inertia = INERTIA * height * height + arm.length_squared();
        self.spin += (torque / inertia - HANG_DAMPING * self.spin) * h;
        self.angle += self.spin * h;

        let arm = rotate(-grip, self.angle);
        self.com = pivot + arm;
        self.vel = pivot_vel + self.spin * Vec2::new(arm.y, -arm.x);
        // Dragged into the floor or a wall, it slides along rather than through.
        let radius = BODY_RADIUS * height;
        for (normal, plane) in planes(arena) {
            for (end, _) in self.ends(height) {
                let depth = radius - ((self.com + end).dot(normal) - plane);
                if depth > 0.0 {
                    self.com += normal * depth;
                }
            }
        }
    }

    /// Lying still on the floor.
    fn rest(&mut self, arena: Vec2, height: f32) {
        self.vel = Vec2::ZERO;
        self.spin = 0.0;
        self.com.y = arena.y - self.lowest(height);
    }

    /// Getting up or lying down: turning between the two with the lowest point
    /// of the body kept on the floor.
    fn roll(&mut self, h: f32, arena: Vec2, height: f32) {
        let Stance::Rolling { from, to, t } = &mut self.stance else {
            return;
        };
        *t = (*t + h / ROLL_SECONDS).min(1.0);
        let (from, to, done) = (*from, *to, *t >= 1.0);
        let eased = t.powi(2) * (3.0 - 2.0 * *t);
        self.angle = from + (to - from) * eased;
        self.vel = Vec2::ZERO;
        self.spin = 0.0;
        self.com.y = arena.y - self.lowest(height);
        if done {
            self.stance = if to == 0.0 { Stance::Standing } else { Stance::Lying };
            self.napping &= to != 0.0;
        }
    }

    /// The body's capsule ends (feet, head) relative to the centre of mass,
    /// each with whether it is the feet end.
    fn ends(&self, height: f32) -> [(Vec2, bool); 2] {
        let radius = BODY_RADIUS * height;
        let feet = Vec2::new(0.0, COM_HEIGHT * height - radius);
        let head = Vec2::new(0.0, -(1.0 - COM_HEIGHT) * height + radius);
        [(rotate(feet, self.angle), true), (rotate(head, self.angle), false)]
    }

    /// How far below the centre of mass the body reaches.
    fn lowest(&self, height: f32) -> f32 {
        let [(a, _), (b, _)] = self.ends(height);
        a.y.max(b.y) + BODY_RADIUS * height
    }

    /// Idle behaviour when there's nothing else to do: stroll, sit down, take
    /// a nap — and get back up after a while.
    fn wander(&mut self, dt: f32) {
        self.idle_timer -= dt;
        if self.idle_timer > 0.0 {
            return;
        }
        match self.stance {
            Stance::Standing if self.is_idle() && !self.held => {
                let choice = self.next_rand();
                if choice < 0.5 {
                    let frac = self.next_rand();
                    self.walk_to(frac);
                } else if choice < 0.62 {
                    self.sit();
                } else if choice < 0.7 {
                    self.lie_down();
                }
                if self.stance == Stance::Standing {
                    // Next decision in 4–10 seconds.
                    self.idle_timer = 4.0 + self.next_rand() * 6.0;
                }
            }
            // Knocked over, or done sitting or napping: back up, busy or not.
            Stance::Sitting | Stance::Lying => {
                self.stand_up();
                self.idle_timer = 4.0 + self.next_rand() * 6.0;
            }
            _ => {}
        }
    }
}

/// The arena's walls as (inward normal, offset) half-planes: floor, ceiling,
/// left, right. A point is inside when `point · normal ≥ offset`.
fn planes(arena: Vec2) -> [(Vec2, f32); 4] {
    [
        (Vec2::new(0.0, -1.0), -arena.y),
        (Vec2::new(0.0, 1.0), 0.0),
        (Vec2::new(1.0, 0.0), 0.0),
        (Vec2::new(-1.0, 0.0), -arena.x),
    ]
}

/// An angle wrapped into (-π, π].
fn wrap(angle: f32) -> f32 {
    let wrapped = (angle + PI).rem_euclid(TAU) - PI;
    if wrapped == -PI {
        PI
    } else {
        wrapped
    }
}

/// How the limbs ride along with the motion — the physical side of the
/// animation; animation.rs turns it into bone rotations. Angles are in the
/// screen plane, radians, counter-clockwise as the viewer sees it, measured
/// from the relaxed pose.
#[derive(Clone, Copy, Default, Debug)]
pub struct Limbs {
    /// Arms (left, right) swinging from the shoulders.
    pub arms: [Spring; 2],
    /// Legs (left, right) swinging from the hips.
    pub legs: [Spring; 2],
    /// The head tipping back against sudden moves.
    pub head: Spring,
    /// Knees giving on landing: 0 straight … 1 a deep squat.
    pub crouch: Spring,
}

/// A damped oscillator's state: position and velocity.
#[derive(Clone, Copy, Default, Debug)]
pub struct Spring {
    pub x: f32,
    pub v: f32,
}

impl Spring {
    /// One semi-implicit Euler step under `force` (per unit mass).
    fn push(&mut self, h: f32, force: f32) {
        self.v += force * h;
        self.x += self.v * h;
    }
}

/// How loose the limbs are: planted legs on the ground, loose in the air,
/// floppy when carried (except the legs it's held by).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Looseness {
    Standing,
    Flying,
    Carried,
    CarriedByLegs,
}

/// How a limb hangs: its relaxed angle, pendulum length (px), muscle tone
/// (1/s²) and damping (1/s).
struct Joint {
    rest: f32,
    length: f32,
    muscle: f32,
    damping: f32,
}

impl Limbs {
    /// Still swinging — worth drawing at the full frame rate.
    pub fn settling(&self) -> bool {
        [self.head, self.crouch]
            .iter()
            .chain(&self.arms)
            .chain(&self.legs)
            .any(|spring| spring.v.abs() > 0.05)
    }

    fn step(&mut self, h: f32, acc: Vec2, vel: Vec2, loose: Looseness, height: f32, angle: f32) {
        // What a limb feels in the body's frame: gravity minus the body's own
        // acceleration (a jolt swings it the other way; free fall is
        // weightless), plus the passing air pushing back.
        let felt = (Vec2::new(0.0, GRAVITY) - acc).clamp_length_max(MAX_FELT) - vel * AIR_PUSH;
        let ((arm_muscle, arm_damping), (leg_muscle, leg_damping)) = match loose {
            Looseness::Standing => ((60.0, 10.0), (400.0, 40.0)),
            Looseness::Flying => ((12.0, 2.5), (40.0, 6.0)),
            Looseness::Carried => ((5.0, 1.2), (5.0, 1.2)),
            Looseness::CarriedByLegs => ((5.0, 1.2), (400.0, 40.0)),
        };
        for (arm, rest) in self.arms.iter_mut().zip(ARM_REST) {
            let joint = Joint { rest, length: ARM_LENGTH * height, muscle: arm_muscle, damping: arm_damping };
            hang_limb(arm, h, felt, angle, joint);
        }
        for leg in &mut self.legs {
            let joint = Joint { rest: 0.0, length: LEG_LENGTH * height, muscle: leg_muscle, damping: leg_damping };
            hang_limb(leg, h, felt, angle, joint);
        }

        let head_target = (acc.x * HEAD_LAG).clamp(-0.25, 0.25);
        self.head.push(h, 120.0 * (head_target - self.head.x) - 16.0 * self.head.v);

        self.crouch.push(h, -180.0 * self.crouch.x - 18.0 * self.crouch.v);
        if self.crouch.x < 0.0 {
            self.crouch = Spring::default();
        }
        self.crouch.x = self.crouch.x.min(1.0);
    }
}

/// One limb hanging from its joint on a body rolled by `angle`: pulled along
/// `felt` like a pendulum, held in its pose by muscle, damped.
fn hang_limb(limb: &mut Spring, h: f32, felt: Vec2, angle: f32, joint: Joint) {
    let field = felt.x.atan2(felt.y);
    let pull = -(felt.length() / joint.length) * (angle + joint.rest + limb.x - field).sin()
        // Standing still, the relaxed pose is the balance point.
        + (GRAVITY / joint.length) * joint.rest.sin();
    limb.push(h, pull - joint.muscle * limb.x - joint.damping * limb.v);
    // Shoulders and hips reach about overhead, and no further.
    limb.x = limb.x.clamp(-3.0, 3.0);
}

/// Rotate a screen-space vector (+y down) counter-clockwise as the viewer
/// sees it — the same turn as a world rotation about +Z.
fn rotate(v: Vec2, angle: f32) -> Vec2 {
    let (sin, cos) = angle.sin_cos();
    Vec2::new(v.x * cos + v.y * sin, -v.x * sin + v.y * cos)
}

/// Where the feet may be while standing: (left, right, top, floor). Walls
/// keep the body's width on screen, the ceiling its head.
fn limits(arena: Vec2, body: Vec2) -> (f32, f32, f32, f32) {
    let left = body.x.min(arena.x * 0.5);
    let right = (arena.x - body.x).max(left);
    let top = body.y.min(arena.y);
    (left, right, top, arena.y)
}

/// Move `from` toward `to` by at most `max_delta`.
fn approach(from: f32, to: f32, max_delta: f32) -> f32 {
    from + (to - from).clamp(-max_delta, max_delta)
}

/// The standing body (half-width, height) for a character, in px.
fn body_size(character: &Character) -> Vec2 {
    Vec2::new(character.height_px * BODY_HALF_WIDTH, character.height_px)
}

/// The user clicked the character (pressed and let go without dragging).
#[derive(Event)]
pub struct CharacterClicked;

/// A press on the character in progress.
#[derive(Resource, Default)]
struct Grab {
    /// Where the press started; `None` when the character isn't pressed.
    pressed_at: Option<Vec2>,
    /// It moved far enough to be a drag rather than a click.
    dragging: bool,
}

/// The physics systems; animation runs after them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct MovementSet;

pub struct MovementPlugin;

impl Plugin for MovementPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Arena>()
            .init_resource::<Grab>()
            .add_event::<CharacterClicked>()
            .add_systems(
                Update,
                (
                    sync_arena,
                    receive_walk_targets,
                    receive_moves,
                    grab_character,
                    step_physics,
                    apply_transform,
                )
                    .chain()
                    .in_set(MovementSet),
            );
    }
}

/// The arena is the primary window's (or overlay's) logical size.
fn sync_arena(windows: Query<&Window, With<PrimaryWindow>>, mut arena: ResMut<Arena>) {
    if let Ok(window) = windows.get_single() {
        let size = Vec2::new(window.width(), window.height());
        if arena.size != size {
            arena.size = size;
        }
    }
}

/// Brain → "walk to fraction X".
fn receive_walk_targets(mut events: EventReader<WalkToEvent>, mut q: Query<&mut Locomotion>) {
    for WalkToEvent(target) in events.read() {
        for mut loco in &mut q {
            loco.walk_to(*target);
            loco.idle_timer = 4.0; // don't immediately wander after a command
        }
    }
}

/// The brain's whole-body moves are physics: a real hop, sitting down, lying
/// down, and getting back up ("idle").
fn receive_moves(mut events: EventReader<AnimateEvent>, mut q: Query<&mut Locomotion>) {
    for AnimateEvent(clip) in events.read() {
        for mut loco in &mut q {
            match clip.as_str() {
                "jump" => loco.jump(),
                "sit" => loco.sit(),
                "lie_down" => loco.lie_down(),
                "idle" => loco.stand_up(),
                _ => {}
            }
        }
    }
}

/// The mouse side of the physics: press on the character and move to pick it
/// up — it hangs from where you grabbed it — and let go to throw it. A press
/// that doesn't move is a click.
fn grab_character(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut egui: EguiContexts,
    time: Res<Time>,
    mut grab: ResMut<Grab>,
    mut clicked: EventWriter<CharacterClicked>,
    mut characters: Query<(&mut Locomotion, &Character)>,
) {
    let Ok((mut loco, character)) = characters.get_single_mut() else {
        return;
    };
    let cursor = windows.get_single().ok().and_then(Window::cursor_position);

    if buttons.just_pressed(MouseButton::Left) {
        let over_chat = egui
            .try_ctx_mut()
            .is_some_and(|ctx| ctx.is_pointer_over_area());
        if let (Some(cursor), Some(rect)) = (cursor, character.screen_rect) {
            if rect.contains(cursor) && !over_chat {
                *grab = Grab {
                    pressed_at: Some(cursor),
                    dragging: false,
                };
            }
        }
    }
    let Some(pressed_at) = grab.pressed_at else {
        return;
    };

    if buttons.pressed(MouseButton::Left) {
        let Some(cursor) = cursor else { return };
        if !grab.dragging && cursor.distance(pressed_at) > DRAG_THRESHOLD {
            grab.dragging = true;
        }
        if grab.dragging {
            loco.drag_to(cursor, time.delta_secs());
        }
    } else {
        if grab.dragging {
            debug!("thrown at {:.0} px/s, spinning {:.1} rad/s", loco.vel, loco.spin);
            loco.release();
        } else {
            debug!("clicked");
            clicked.send(CharacterClicked);
        }
        *grab = Grab::default();
    }
}

/// Advance the physics and idle behaviour, and pick which way to face.
fn step_physics(
    time: Res<Time>,
    arena: Res<Arena>,
    mut busy: ResMut<Busy>,
    mut q: Query<(&mut Locomotion, &mut Character)>,
) {
    let dt = time.delta_secs();
    for (mut loco, mut character) in &mut q {
        // Wait (hidden) until the model is measured, so it drops in at its
        // real size.
        if character.bounds.is_none() {
            continue;
        }
        let before = loco.stance;
        loco.step(dt, arena.size, body_size(&character));
        loco.wander(dt);
        if std::mem::discriminant(&before) != std::mem::discriminant(&loco.stance) {
            let (vel, spin) = (loco.vel, loco.spin);
            debug!("{before:?} → {:?} (vel {vel:.0}, spin {spin:.1})", loco.stance);
        }
        character.facing = loco.facing();
        busy.0 |= !loco.at_rest();
    }
}

/// Put the model where the body is — feet at its feet, rolled by its angle,
/// scaled to its on-screen height, turned toward where it's going, sunk as
/// the knees give or it sits — and record its screen box, which is what the
/// mouse can grab and the overlay's input region.
fn apply_transform(
    time: Res<Time>,
    arena: Res<Arena>,
    mut region: ResMut<InputRegion>,
    mut q: Query<(&Locomotion, &mut Character, &mut Transform)>,
) {
    let dt = time.delta_secs();
    region.character = None;
    for (loco, mut character, mut transform) in &mut q {
        let Some((min, max)) = character.bounds else {
            continue;
        };
        let height = character.height_px;
        let turn = Quat::from_rotation_y(character.facing * FRAC_PI_2);
        character.yaw = character.yaw.slerp(turn, (TURN_RATE * dt).min(1.0));

        let sink = (loco.limbs.crouch.x * CROUCH_DROP + loco.seat * SIT_DROP) * height;
        let feet = loco.feet(height) + Vec2::new(0.0, sink);
        transform.translation = arena.to_world(feet);
        transform.rotation = Quat::from_rotation_z(loco.angle) * character.yaw;
        transform.scale = Vec3::splat(height / (max.y - min.y).max(1e-3));

        let (mut lo, mut hi) = (Vec2::MAX, Vec2::MIN);
        for corner in box_corners(min, max) {
            let p = arena.to_screen(transform.transform_point(corner));
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let rect = Rect::from_corners(lo, hi);
        character.screen_rect = Some(rect);
        region.character = Some(pixel_rect(rect.inflate(4.0)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARENA: Vec2 = Vec2::new(1920.0, 1080.0);
    const BODY: Vec2 = Vec2::new(80.0, 320.0);
    const DT: f32 = 1.0 / 60.0;

    fn run(loco: &mut Locomotion, seconds: f32) {
        for _ in 0..(seconds / DT) as usize {
            loco.step(DT, ARENA, BODY);
        }
    }

    /// A fixed seed, so idle decisions are the same every run.
    const SEED: u32 = 0x9E37_79B9;

    /// Settled on its feet at `x`.
    fn standing(x: f32) -> Locomotion {
        let mut loco = Locomotion {
            placed: true,
            rng: SEED,
            ..Default::default()
        };
        loco.com = Vec2::new(x, ARENA.y - COM_HEIGHT * BODY.y);
        loco.stance = Stance::Standing;
        loco
    }

    /// In the air at `com`, moving and spinning.
    fn flying(com: Vec2, vel: Vec2, spin: f32) -> Locomotion {
        Locomotion {
            com,
            vel,
            spin,
            placed: true,
            rng: SEED,
            ..Default::default()
        }
    }

    /// The body's lowest and outermost points, to check it stays inside.
    fn inside(loco: &Locomotion) -> bool {
        let r = BODY_RADIUS * BODY.y;
        loco.ends(BODY.y).iter().all(|(end, _)| {
            let p = loco.com + *end;
            p.x >= r - 1.0 && p.x <= ARENA.x - r + 1.0 && p.y >= r - 1.0 && p.y <= ARENA.y - r + 1.0
        })
    }

    #[test]
    fn drops_in_and_lands_on_its_feet() {
        let mut loco = Locomotion::default();
        run(&mut loco, 5.0);
        assert_eq!(loco.stance, Stance::Standing);
        assert!((loco.feet(BODY.y).y - ARENA.y).abs() < 0.01, "feet on the floor");
        assert!(loco.is_idle());
    }

    #[test]
    fn walks_to_the_target_and_stops_there() {
        let mut loco = standing(200.0);
        loco.walk_to(0.75);
        for _ in 0..900 {
            loco.step(DT, ARENA, BODY);
            assert!(loco.feet(BODY.y).x <= 0.75 * ARENA.x + 0.01, "never overshoots");
        }
        assert!((loco.feet(BODY.y).x - 0.75 * ARENA.x).abs() < 0.01);
        assert!(loco.is_idle());
    }

    #[test]
    fn a_jump_goes_up_and_comes_back_down() {
        let mut loco = standing(960.0);
        loco.jump();
        let mut peak = ARENA.y;
        for _ in 0..240 {
            loco.step(DT, ARENA, BODY);
            peak = peak.min(loco.feet(BODY.y).y);
        }
        assert!(ARENA.y - peak > 100.0, "jumped {} px", ARENA.y - peak);
        assert_eq!(loco.stance, Stance::Standing);
    }

    #[test]
    fn a_spin_goes_all_the_way_round() {
        let mut loco = flying(Vec2::new(960.0, 300.0), Vec2::new(0.0, -600.0), 15.0);
        let mut turned: f32 = 0.0;
        for _ in 0..30 {
            loco.step(DT, ARENA, BODY);
            turned = turned.max(loco.angle.abs());
        }
        assert!(turned > TAU, "turned {turned} rad");
    }

    #[test]
    fn a_tumbling_landing_knocks_it_down_then_it_gets_up() {
        let mut loco = flying(Vec2::new(960.0, 500.0), Vec2::new(300.0, 0.0), 9.0);
        let (mut lay_down, mut got_up) = (false, false);
        for _ in 0..900 {
            loco.step(DT, ARENA, BODY);
            loco.wander(DT);
            lay_down |= loco.stance == Stance::Lying;
            got_up |= lay_down && loco.stance == Stance::Standing;
            assert!(inside(&loco), "stays in the arena");
        }
        assert!(lay_down, "fell over");
        assert!(got_up, "and got back up");
    }

    #[test]
    fn a_hard_throw_stays_on_screen_and_settles() {
        let mut loco = flying(Vec2::new(960.0, 600.0), Vec2::new(-3000.0, -900.0), 0.0);
        for _ in 0..720 {
            loco.step(DT, ARENA, BODY);
            loco.wander(DT);
            assert!(inside(&loco), "left the arena at {:?}", loco.com);
        }
        assert!(loco.grounded(), "came to rest on the floor");
    }

    #[test]
    fn a_throw_carries_the_drag_speed() {
        let mut loco = standing(500.0);
        let grip = loco.com + Vec2::new(0.0, -100.0); // by the chest
        for i in 0..=10 {
            loco.drag_to(grip + Vec2::new(20.0 * i as f32, 0.0), DT);
            loco.step(DT, ARENA, BODY);
        }
        loco.release();
        assert_eq!(loco.stance, Stance::Flying);
        assert!(loco.vel.x > 800.0, "thrown at {} px/s", loco.vel.x);
    }

    #[test]
    fn carried_it_hangs_back_from_a_sideways_drag() {
        let mut loco = standing(700.0);
        let chest = loco.com + Vec2::new(0.0, -80.0);
        for i in 0..60 {
            // Hold still for half a second, then sweep right.
            let x = (i.max(30) - 30) as f32 * 25.0;
            loco.drag_to(chest + Vec2::new(x, -300.0), DT);
            loco.step(DT, ARENA, BODY);
        }
        // The body swings back to the left (clockwise), and the legs trail.
        assert!(loco.angle < -0.05, "angle {}", loco.angle);
        assert!(loco.limbs.legs[0].x < -0.05, "legs at {}", loco.limbs.legs[0].x);
    }

    #[test]
    fn carried_by_the_feet_it_hangs_upside_down() {
        let mut loco = standing(900.0);
        let feet = loco.feet(BODY.y) + Vec2::new(3.0, -2.0);
        for i in 0..420 {
            loco.drag_to(feet + Vec2::new(0.0, -500.0 * (i.min(10) as f32 / 10.0)), DT);
            loco.step(DT, ARENA, BODY);
            // It tips over, swings, and settles head-down within ±45°.
            if i >= 360 {
                assert!(loco.angle.cos() < -0.7, "angle {}", wrap(loco.angle));
            }
        }
    }

    #[test]
    fn it_sits_and_stands_back_up_to_walk() {
        let mut loco = standing(500.0);
        loco.sit();
        run(&mut loco, 2.0);
        assert_eq!(loco.stance, Stance::Sitting);
        assert!(loco.seat > 0.95);
        loco.walk_to(0.8);
        run(&mut loco, 8.0);
        assert!(loco.seat < 0.01 && loco.is_idle());
        assert!((loco.feet(BODY.y).x - 0.8 * ARENA.x).abs() < 0.01, "walked there");
    }

    #[test]
    fn it_lies_down_and_gets_up() {
        let mut loco = standing(500.0);
        loco.lie_down();
        run(&mut loco, ROLL_SECONDS + 0.1);
        assert_eq!(loco.stance, Stance::Lying);
        assert!((wrap(loco.angle).abs() - FRAC_PI_2).abs() < 0.01, "flat");
        assert!((loco.com.y - (ARENA.y - BODY_RADIUS * BODY.y)).abs() < 0.5, "on the floor");
        loco.stand_up();
        run(&mut loco, ROLL_SECONDS + 0.5);
        assert_eq!(loco.stance, Stance::Standing);
        assert!((loco.feet(BODY.y).y - ARENA.y).abs() < 0.01);
    }

    #[test]
    fn arms_float_up_in_a_fall() {
        let mut loco = flying(Vec2::new(960.0, 200.0), Vec2::new(0.0, 1800.0), 0.0);
        for _ in 0..18 {
            loco.step(DT, ARENA, BODY);
        }
        assert_eq!(loco.stance, Stance::Flying, "still falling");
        assert!(loco.limbs.arms[0].x > 0.25, "left arm at {}", loco.limbs.arms[0].x);
        assert!(loco.limbs.arms[1].x < -0.25, "right arm at {}", loco.limbs.arms[1].x);
    }

    #[test]
    fn landing_bends_the_knees_and_they_spring_back() {
        let mut loco = flying(Vec2::new(960.0, 400.0), Vec2::ZERO, 0.0);
        let mut deepest: f32 = 0.0;
        for _ in 0..300 {
            loco.step(DT, ARENA, BODY);
            deepest = deepest.max(loco.limbs.crouch.x);
        }
        assert!(deepest > 0.1, "squatted to {deepest}");
        assert_eq!(loco.limbs.crouch.x, 0.0);
        assert!(loco.at_rest(), "everything comes to rest");
    }

    #[test]
    fn starting_to_walk_leans_into_it() {
        let mut loco = standing(200.0);
        loco.walk_to(0.9);
        run(&mut loco, 0.1);
        // Accelerating right tips the top of the body right: clockwise.
        assert!(loco.angle < -0.01, "angle {}", loco.angle);
    }

    #[test]
    fn left_alone_it_never_leaves_the_floor() {
        // Idle life — strolling, sitting, napping, getting up — for two
        // minutes, whatever the dice say.
        for seed in 1..40 {
            let mut loco = standing(960.0);
            loco.rng = seed * 0x9E37_79B9 | 1;
            for frame in 0..7200 {
                loco.step(DT, ARENA, BODY);
                loco.wander(DT);
                assert!(
                    loco.grounded(),
                    "seed {seed}: left the floor at frame {frame} as {:?}, vel {:?}, spin {}",
                    loco.stance,
                    loco.vel,
                    loco.spin
                );
            }
        }
    }

    #[test]
    fn rotate_turns_counter_clockwise_on_screen() {
        let right = rotate(Vec2::new(0.0, 1.0), FRAC_PI_2);
        assert!((right - Vec2::new(1.0, 0.0)).length() < 1e-6);
        assert!((wrap(3.0 * PI) - PI).abs() < 1e-5);
    }
}
