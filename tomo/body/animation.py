"""Bringing the character to life: procedural body animation and facial
expressions, driven by what she is doing.

A VRM carries a standard humanoid map (which node is the left upper arm, the
head, …) and named facial expressions (blink, happy, aa, …). :class:`Rig`
finds both in the loaded model; every frame :func:`pose_body` poses the
skeleton — breathing at rest, a walk cycle paced by the distance walked,
gestures the brain asks for, and on top of it all what the physics
(:mod:`.movement`) simulates: limbs dangling and swinging with the motion,
knees giving on landing — and :func:`animate_face` blinks, shows emotions and
moves the mouth while the voice plays. During a health round she mirrors the
user instead (:func:`mirror_pose`).

Poses work like three-vrm's "normalized" rig: each bone gets a rotation in
model axes (+X the model's left, +Y up, +Z toward the viewer) relative to the
T-pose, turned into the bone's own frame as ``P⁻¹·Q·P·L`` (P: the parent's
rest rotation in model space, L: the bone's rest rotation). The same pose
therefore fits any VRM 1.0 model, whatever its bone axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .. import events
from ..events import Role
from . import workout
from .mathx import (IDENTITY, Quat, add, cross, dot, is_zero, normalize_or_zero, qarc, qaxis, qconj, qfrom_axes,
                    qfrom_mat3, qmul, qrot, qslerp, rotation_of, rx, ry, rz, scale, sub)
from .movement import WALK_SPEED, Locomotion, Stance
from .skeleton import Skeleton
from .vrm import Vrm

POSE_RATE = 10.0  # how quickly the body blends toward a new pose, 1/s
MIRROR_RATE = 18.0  # faster mirroring the user: she keeps up, the camera's jitter smoothed away
STRIDE = 0.8  # one walk cycle (two steps) covers this many body heights
EMOTION_SECONDS = 8.0  # emotions fade back to neutral after this long
EMOTIONS = ("happy", "sad", "angry", "surprised", "relaxed")


class Bone(str, Enum):
    """The humanoid bones Tomo poses, by their VRM names."""

    HIPS = "hips"  # only turned while mirroring the user (their torso's lean)
    SPINE = "spine"
    CHEST = "chest"
    NECK = "neck"
    HEAD = "head"
    LEFT_SHOULDER = "leftShoulder"
    RIGHT_SHOULDER = "rightShoulder"
    LEFT_UPPER_ARM = "leftUpperArm"
    RIGHT_UPPER_ARM = "rightUpperArm"
    LEFT_LOWER_ARM = "leftLowerArm"
    RIGHT_LOWER_ARM = "rightLowerArm"
    LEFT_UPPER_LEG = "leftUpperLeg"
    RIGHT_UPPER_LEG = "rightUpperLeg"
    LEFT_LOWER_LEG = "leftLowerLeg"
    RIGHT_LOWER_LEG = "rightLowerLeg"


class Gesture(Enum):
    WAVE = "wave"
    NOD = "nod"
    SHRUG = "shrug"

    @property
    def seconds(self) -> float:
        return {"wave": 2.4, "nod": 1.2, "shrug": 1.4}[self.value]


Pose = dict[Bone, Quat]


@dataclass
class RigBone:
    node: int
    rest: Quat  # L: rest rotation, local to the parent
    parent_rest: Quat  # P: the parent's rest rotation in model space


@dataclass
class Rig:
    """The humanoid bones and expressions found in a model, and the current
    (smoothed) pose."""

    bones: dict[Bone, RigBone]
    expressions: dict[str, list[tuple[int, int, float]]]  # preset → (mesh, blend shape, weight)
    pose: Pose = field(default_factory=dict)

    @classmethod
    def build(cls, model: Vrm, skeleton: Skeleton) -> "Rig":
        bones = {}
        if model.version == "1.0":
            for bone in Bone:
                node = model.humanoid.get(bone.value)
                if node is None or not 0 <= node < skeleton.count:
                    continue
                parent = model.nodes[node].parent
                parent_rest = qfrom_mat3(rotation_of(skeleton.rest_world[parent])) if parent >= 0 else IDENTITY
                bones[bone] = RigBone(node, tuple(skeleton.rest_rotation[node]), parent_rest)
        expressions = {}
        for preset, binds in model.expressions.items():
            expressions[preset] = [(model.nodes[node].mesh, index, weight) for node, index, weight in binds
                                   if 0 <= node < len(model.nodes) and model.nodes[node].mesh is not None]
        return cls(bones, expressions)

    def apply(self, target: Pose, rate: float, dt: float, skeleton: Skeleton) -> None:
        """Blend toward ``target`` and turn the bones."""
        k = 1.0 - math.exp(-rate * dt)
        for bone, rb in self.bones.items():
            current = qslerp(self.pose.get(bone, IDENTITY), target.get(bone, IDENTITY), k)
            self.pose[bone] = current
            p = rb.parent_rest
            skeleton.rotation[rb.node] = qmul(qmul(qmul(qconj(p), current), p), rb.rest)


@dataclass
class Animator:
    """What the character is up to, beyond her physics."""

    clock: float = 0.0
    walk_phase: float = 0.0
    gesture: Gesture | None = None
    gesture_elapsed: float = 0.0
    thinking: bool = False
    listening: bool = False
    emotion: str | None = None
    emotion_age: float = 0.0
    speaking: bool = False
    blink_in: float = 0.0
    blink_t: float | None = None

    def cue(self, event) -> None:
        """Take the brain's cues."""
        if isinstance(event, events.Animate):
            if event.clip in ("wave", "nod", "shrug"):
                self.gesture, self.gesture_elapsed = Gesture(event.clip), 0.0
            elif event.clip == "idle":
                self.gesture = None
            # "jump", "sit" and "lie_down" are physics (movement.py).
        elif isinstance(event, events.Emote):
            if event.emotion in EMOTIONS:
                self.emotion, self.emotion_age = event.emotion, 0.0
            else:
                self.emotion = None
        elif isinstance(event, events.Speaking):
            self.speaking = event.on
        elif isinstance(event, events.Thinking):
            self.thinking = event.on
        elif isinstance(event, events.Listening):
            self.listening = event.on
        elif isinstance(event, events.Chat):
            # A little smile when she answers, unless she's showing a feeling.
            if event.line.role == Role.ASSISTANT and self.emotion is None:
                self.emotion, self.emotion_age = "happy", EMOTION_SECONDS - 3.0

    def busy(self) -> bool:
        return self.speaking or self.gesture is not None


def pose_body(dt: float, rig: Rig, animator: Animator, loco: Locomotion, facing: float, yaw: Quat,
              height: float, skeleton: Skeleton, mirror: list | None) -> None:
    """Pose the skeleton for this frame, blending smoothly. ``mirror``: the
    user's joints during a health round."""
    animator.clock += dt
    speed = min(abs(loco.vel.x) / WALK_SPEED, 1.0)
    walking = loco.stance == Stance.STANDING and loco.seat < 0.05 and speed > 0.05
    if walking:
        animator.walk_phase += abs(loco.vel.x) * dt / (STRIDE * height) * math.tau
    if animator.gesture is not None:
        animator.gesture_elapsed += dt
        if animator.gesture_elapsed > animator.gesture.seconds:
            animator.gesture = None
    if mirror is not None:
        rig.apply(mirror_pose(mirror, rig.pose), MIRROR_RATE, dt, skeleton)
    else:
        rig.apply(target_pose(animator, loco, facing, yaw, walking, speed), POSE_RATE, dt, skeleton)


def target_pose(animator: Animator, loco: Locomotion, facing: float, yaw: Quat, walking: bool,
                speed: float) -> Pose:
    """The pose for this moment, as model-space rotations relative to the
    T-pose: a base pose (breathing, the walk cycle, gestures) with the
    physics' swinging, sinking limbs on top."""
    t = animator.clock
    pose: Pose = {}
    breath = math.sin(t * math.tau * 0.25)

    # At rest: arms relaxed at the sides, breathing, a slow idle sway.
    arms_down = 72.0 + breath * 1.5
    pose[Bone.LEFT_UPPER_ARM] = rz(-arms_down)
    pose[Bone.RIGHT_UPPER_ARM] = rz(arms_down)
    pose[Bone.LEFT_LOWER_ARM] = ry(-12.0)
    pose[Bone.RIGHT_LOWER_ARM] = ry(12.0)
    pose[Bone.CHEST] = rx(-breath * 1.5)
    pose[Bone.HEAD] = rz(math.sin(t * 0.4) * 2.5)

    if walking:
        # Legs swing (knee bent while a leg comes forward), arms swing
        # opposite, the torso twists a touch.
        s, c, a = math.sin(animator.walk_phase), math.cos(animator.walk_phase), speed
        pose[Bone.LEFT_UPPER_LEG] = rx(-s * 28.0 * a)
        pose[Bone.RIGHT_UPPER_LEG] = rx(s * 28.0 * a)
        pose[Bone.LEFT_LOWER_LEG] = rx(max(c, 0.0) * 45.0 * a)
        pose[Bone.RIGHT_LOWER_LEG] = rx(max(-c, 0.0) * 45.0 * a)
        pose[Bone.LEFT_UPPER_ARM] = qmul(rx(s * 20.0 * a), rz(-arms_down))
        pose[Bone.RIGHT_UPPER_ARM] = qmul(rx(-s * 20.0 * a), rz(arms_down))
        pose[Bone.SPINE] = ry(s * 5.0 * a)

    # Head: tilted in thought, cocked while listening.
    if animator.thinking:
        pose[Bone.HEAD] = qmul(rz(9.0), rx(-7.0))
    elif animator.listening:
        pose[Bone.HEAD] = qmul(rz(-7.0), rx(4.0))

    def blend_in(amount: float, poses: list[tuple[Bone, Quat]]) -> None:
        for bone, q in poses:
            pose[bone] = qslerp(pose.get(bone, IDENTITY), q, amount)

    # Sitting on the floor (the body sinks to match): legs out in front,
    # hands on the floor behind, head turned back toward the viewer.
    if loco.seat > 0.001:
        blend_in(loco.seat, [
            (Bone.LEFT_UPPER_LEG, rx(-88.0)), (Bone.RIGHT_UPPER_LEG, rx(-88.0)),
            (Bone.LEFT_LOWER_LEG, rx(12.0)), (Bone.RIGHT_LOWER_LEG, rx(12.0)),
            (Bone.LEFT_UPPER_ARM, qmul(rx(25.0), rz(-75.0))), (Bone.RIGHT_UPPER_ARM, qmul(rx(25.0), rz(75.0))),
            (Bone.SPINE, rx(-6.0)), (Bone.HEAD, ry(-facing * 55.0)),
        ])

    # Lying on the floor, on her side facing the viewer, knees drawn up; and
    # pushing herself up (or letting herself down) in between.
    lying = pushing = 0.0
    if loco.stance == Stance.LYING:
        lying = 1.0
    elif loco.stance == Stance.ROLLING and loco.rolling is not None:
        r = loco.rolling
        lying, pushing = (1.0 - r.t if r.to == 0.0 else r.t), math.sin(r.t * math.pi)
    if lying > 0.0:
        blend_in(lying, [
            (Bone.LEFT_UPPER_LEG, rx(-30.0)), (Bone.RIGHT_UPPER_LEG, rx(-38.0)),
            (Bone.LEFT_LOWER_LEG, rx(45.0)), (Bone.RIGHT_LOWER_LEG, rx(55.0)),
            (Bone.LEFT_UPPER_ARM, qmul(rx(-25.0), rz(-70.0))), (Bone.RIGHT_UPPER_ARM, qmul(rx(-25.0), rz(70.0))),
            (Bone.LEFT_LOWER_ARM, ry(-50.0)), (Bone.RIGHT_LOWER_ARM, ry(50.0)),
        ])
    if pushing > 0.0:
        blend_in(pushing, [
            (Bone.LEFT_UPPER_LEG, rx(-70.0)), (Bone.RIGHT_UPPER_LEG, rx(-70.0)),
            (Bone.LEFT_LOWER_LEG, rx(110.0)), (Bone.RIGHT_LOWER_LEG, rx(110.0)),
            (Bone.LEFT_UPPER_ARM, qmul(rx(-35.0), rz(-40.0))), (Bone.RIGHT_UPPER_ARM, qmul(rx(-35.0), rz(40.0))),
        ])

    # Gestures the brain asks for, eased in and out over their duration.
    on_feet = loco.stance == Stance.STANDING and loco.seat < 0.05
    if animator.gesture is not None and on_feet:
        gesture, elapsed = animator.gesture, animator.gesture_elapsed
        p = max(0.0, min(1.0, elapsed / gesture.seconds))
        envelope = min(math.sin(p * math.pi), 0.35) / 0.35  # quick in, hold, quick out
        if gesture == Gesture.WAVE:
            blend_in(envelope, [(Bone.RIGHT_UPPER_ARM, rz(-65.0)),
                                (Bone.RIGHT_LOWER_ARM, rz(-50.0 + math.sin(elapsed * 12.0) * 25.0))])
        elif gesture == Gesture.NOD:
            blend_in(envelope, [(Bone.HEAD, rx(max(math.sin(p * math.tau * 2.0), 0.0) * 16.0))])
        else:
            blend_in(envelope, [
                (Bone.LEFT_SHOULDER, rz(12.0)), (Bone.RIGHT_SHOULDER, rz(-12.0)),
                (Bone.LEFT_UPPER_ARM, rz(-55.0)), (Bone.RIGHT_UPPER_ARM, rz(55.0)),
                (Bone.LEFT_LOWER_ARM, qmul(ry(-70.0), rz(35.0))), (Bone.RIGHT_LOWER_ARM, qmul(ry(70.0), rz(-35.0))),
                (Bone.HEAD, rz(8.0)),
            ])

    # The physics' limbs. Knees give on landing, while the body sinks to
    # match; then arms, legs and head swing in the screen plane — in model
    # axes, a turn about the axis pointing out of the screen.
    limbs = loco.limbs

    def turn(bone: Bone, by: Quat) -> None:
        pose[bone] = qmul(by, pose.get(bone, IDENTITY))

    knees = limbs.crouch.x
    turn(Bone.LEFT_UPPER_LEG, rx(-50.0 * knees))
    turn(Bone.RIGHT_UPPER_LEG, rx(-50.0 * knees))
    turn(Bone.LEFT_LOWER_LEG, rx(95.0 * knees))
    turn(Bone.RIGHT_LOWER_LEG, rx(95.0 * knees))
    out_of_screen = qrot(qconj(yaw), (0.0, 0.0, 1.0))
    for bone, spring in ((Bone.LEFT_UPPER_ARM, limbs.arms[0]), (Bone.RIGHT_UPPER_ARM, limbs.arms[1]),
                         (Bone.LEFT_UPPER_LEG, limbs.legs[0]), (Bone.RIGHT_UPPER_LEG, limbs.legs[1]),
                         (Bone.HEAD, limbs.head)):
        turn(bone, qaxis(out_of_screen, spring.x))
    return pose


def mirror_pose(user: list, current: Pose) -> Pose:
    """The user's pose as Tomo's, mirrored (see :func:`workout.mirrored`):
    the torso turned to theirs — bent over, lying in a plank — and each limb
    turned from its T-pose direction to theirs, as rotations relative to the
    parent the rig expects. Her left side follows their right, as in a
    mirror. Bones whose joints the camera doesn't see keep the pose they
    have."""
    from ..health import (LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ANKLE,
                          RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST)

    def at(i: int):
        return workout.mirrored(user, i)

    def mid(a: int, b: int):
        p, q = at(a), at(b)
        if p is not None and q is not None:
            return scale(add(p, q), 0.5)
        return p if p is not None else q

    def held(bone: Bone) -> Quat:
        return current.get(bone, IDENTITY)

    pose = dict(current)
    # The torso: up along the spine, across from her right hip to her left
    # (the user's left hip to their right).
    torso = None
    her_left, her_right = at(RIGHT_HIP), at(LEFT_HIP)
    shoulders, hips_mid = mid(LEFT_SHOULDER, RIGHT_SHOULDER), mid(LEFT_HIP, RIGHT_HIP)
    if None not in (her_left, her_right, shoulders, hips_mid):
        up = normalize_or_zero(sub(shoulders, hips_mid))
        across = sub(her_left, her_right)
        x = normalize_or_zero(sub(across, scale(up, dot(across, up))))
        if not is_zero(up) and not is_zero(x):
            torso = qfrom_axes(x, up, cross(x, up))
    hips = torso if torso is not None else held(Bone.HIPS)
    pose[Bone.HIPS] = hips
    for bone in (Bone.SPINE, Bone.CHEST, Bone.NECK, Bone.HEAD, Bone.LEFT_SHOULDER, Bone.RIGHT_SHOULDER):
        pose[bone] = IDENTITY

    def limb(start: int, end: int, rest, parent: Quat):
        """A limb from joint ``start`` to ``end``, pointing ``rest`` in the
        T-pose, under a parent turned ``parent`` (model space): its local
        turn, and its own."""
        a, b = at(start), at(end)
        if a is None or b is None:
            return None
        direction = normalize_or_zero(sub(b, a))
        if is_zero(direction):
            return None
        turned = qarc(rest, direction)
        return qmul(qconj(parent), turned), turned

    def chain(upper: Bone, lower: Bone, rest, joints: tuple[int, int, int]) -> None:
        found = limb(joints[0], joints[1], rest, hips)
        if found is not None:
            pose[upper] = found[0]
            upper_turned = found[1]
        else:
            upper_turned = qmul(hips, held(upper))
        found = limb(joints[1], joints[2], rest, upper_turned)
        if found is not None:
            pose[lower] = found[0]

    chain(Bone.LEFT_UPPER_ARM, Bone.LEFT_LOWER_ARM, (1.0, 0.0, 0.0), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))
    chain(Bone.RIGHT_UPPER_ARM, Bone.RIGHT_LOWER_ARM, (-1.0, 0.0, 0.0), (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST))
    chain(Bone.LEFT_UPPER_LEG, Bone.LEFT_LOWER_LEG, (0.0, -1.0, 0.0), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
    chain(Bone.RIGHT_UPPER_LEG, Bone.RIGHT_LOWER_LEG, (0.0, -1.0, 0.0), (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
    return pose


def animate_face(dt: float, rig: Rig, animator: Animator, loco: Locomotion, skeleton: Skeleton) -> None:
    """Blinking, emotions and a talking mouth, written to the face's blend
    shapes — plus what the physics does to a face: eyes shut lying on the
    floor, startled when flung about."""
    t = animator.clock
    weights: dict[str, float] = {}
    if not loco.grounded():
        weights["surprised"] = min(loco.vel.length() / 2000.0 + abs(loco.spin) / 15.0, 0.8)

    # Blink every 2–6 s, 0.15 s each.
    animator.blink_in -= dt
    if animator.blink_in <= 0.0 and animator.blink_t is None:
        animator.blink_t = 0.0
        animator.blink_in = 2.0 + abs(math.sin(t * 7.13)) * 4.0
    if animator.blink_t is not None:
        animator.blink_t += dt
        weights["blink"] = 1.0 - abs(animator.blink_t / 0.075 - 1.0)
        if animator.blink_t >= 0.15:
            animator.blink_t = None

    # The current emotion, faded in and back out.
    if animator.emotion is not None:
        animator.emotion_age += dt
        fade_in = min(animator.emotion_age / 0.3, 1.0)
        fade_out = max(0.0, min(1.0, (EMOTION_SECONDS - animator.emotion_age) / 0.8))
        weights[animator.emotion] = fade_in * fade_out
        if animator.emotion_age > EMOTION_SECONDS:
            animator.emotion = None

    # Eyes shut while lying down, napping or knocked over.
    if loco.stance in (Stance.LYING, Stance.ROLLING):
        weights["blink"] = max(weights.get("blink", 0.0), 1.0)

    # Mouth shapes flicker between "aa" and "oh" while the voice plays.
    if animator.speaking:
        weights["aa"] = min(max(math.sin(t * 13.0), 0.0) * 0.8, 1.0)
        weights["oh"] = max(math.sin(t * 7.0 + 1.0), 0.0) * 0.35

    # Every driven blend shape is recomputed each frame, so released
    # expressions go back to zero.
    targets: dict[tuple[int, int], float] = {}
    for preset in ("blink", "aa", "oh", *EMOTIONS):
        weight = max(weights.get(preset, 0.0), 0.0)
        for mesh, index, bind in rig.expressions.get(preset, []):
            targets[(mesh, index)] = targets.get((mesh, index), 0.0) + weight * bind
    for (mesh, index), weight in targets.items():
        skeleton.set_morph(mesh, index, min(weight, 1.0))
