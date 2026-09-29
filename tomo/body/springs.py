"""Spring bones: hair, skirts and ribbons that swing with the motion.

A VRM 1.0 file lists them in its ``VRMC_springBone`` extension: chains of
joints, each with a stiffness (how hard it springs back to its styled
shape), a drag and a gravity, plus spheres and capsules on the body that they
can't pass through. This is that spec's simulation — the tip of each bone is a
Verlet particle held at the bone's length.

Each chain moves in the space the file names as its ``center`` (usually the
model's root), so clothes keep to the body however it's thrown about, as
their author set them up. Hair is the exception: it moves in world space, so
the whole body's motion counts — a throw sends it streaming, and hung upside
down it falls the other way.

It runs after the pose is set (animation), and turns the chains' bones
itself: the swing shows this very frame. The joints are stepped a level at a
time — the first joint of every chain together, then the second… — as numpy
arrays: Python would be too slow joint by joint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mathx import compose_batch, mat3_to_quat_batch, quat_to_mat3_batch
from .skeleton import Skeleton

STEP = 1.0 / 60.0  # the simulation's fixed step, s
MAX_STEPS = 4  # at most this many steps a frame; after a hitch the rest is dropped
# Extra pull toward the floor on hair, m/s: models tend to come with none —
# their springiness alone would hold the hair styled "up" even with the
# character upside down; this is just enough to win against it.
HAIR_GRAVITY = 1.0
SETTLED_PX = 0.5  # a tip moving less than this in a step (screen px) has settled
TELEPORT = 1.0  # the body jumping further than this in a frame (m) is a teleport


@dataclass(frozen=True)
class Params:
    hit_radius: float = 0.0
    stiffness: float = 1.0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)  # direction × power
    drag: float = 0.5


@dataclass(frozen=True)
class JointSpec:
    node: int
    params: Params


@dataclass(frozen=True)
class ChainSpec:
    joints: tuple[JointSpec, ...]  # root first; the last only ends the one before it
    center: int | None  # the node whose space it moves in; None: world space
    colliders: tuple[int, ...]  # indices into SpringSpec.colliders
    hair: bool


@dataclass(frozen=True)
class ShapeSpec:
    node: int
    offset: tuple[float, float, float]
    radius: float
    tail: tuple[float, float, float] | None  # a capsule's other end; None: a sphere


@dataclass
class SpringSpec:
    chains: list[ChainSpec] = field(default_factory=list)
    colliders: list[ShapeSpec | None] = field(default_factory=list)

    @classmethod
    def from_gltf(cls, gltf: dict) -> "SpringSpec":
        """Read the ``VRMC_springBone`` extension of a .vrm's glTF (none: empty)."""
        ext = gltf.get("extensions", {}).get("VRMC_springBone")
        if not isinstance(ext, dict):
            return cls()
        nodes = gltf.get("nodes", [])

        def node(value) -> int | None:
            return value if isinstance(value, int) and 0 <= value < len(nodes) else None

        def vec3(value) -> tuple[float, float, float]:
            v = list(value) if isinstance(value, list) else []
            return tuple(float(v[i]) if i < len(v) else 0.0 for i in range(3))

        def num(value, default: float) -> float:
            return float(value) if isinstance(value, (int, float)) else default

        colliders = []
        for c in ext.get("colliders", []):
            shape, n = c.get("shape", {}), node(c.get("node"))
            if n is not None and "sphere" in shape:
                s = shape["sphere"]
                colliders.append(ShapeSpec(n, vec3(s.get("offset")), num(s.get("radius"), 0.0), None))
            elif n is not None and "capsule" in shape:
                s = shape["capsule"]
                colliders.append(ShapeSpec(n, vec3(s.get("offset")), num(s.get("radius"), 0.0), vec3(s.get("tail"))))
            else:
                colliders.append(None)  # unknown shapes are skipped
        groups = [[i for i in g.get("colliders", []) if isinstance(i, int)] for g in ext.get("colliderGroups", [])]

        chains = []
        for spring in ext.get("springs", []):
            joints = []
            for j in spring.get("joints", []):
                n = node(j.get("node"))
                if n is None:
                    joints = []
                    break
                direction = vec3(j["gravityDir"]) if "gravityDir" in j else (0.0, -1.0, 0.0)
                power = num(j.get("gravityPower"), 0.0)
                joints.append(JointSpec(n, Params(
                    hit_radius=num(j.get("hitRadius"), 0.0),
                    stiffness=num(j.get("stiffness"), 1.0),
                    gravity=(direction[0] * power, direction[1] * power, direction[2] * power),
                    drag=max(0.0, min(1.0, num(j.get("dragForce"), 0.5))),
                )))
            if len(joints) < 2:
                continue  # one joint has nothing to swing
            colliders_of = tuple(i for g in spring.get("colliderGroups", []) if isinstance(g, int)
                                 and 0 <= g < len(groups) for i in groups[g])
            first = nodes[joints[0].node].get("name", "")
            hair = "hair" in str(spring.get("name", "")).lower() or "hair" in first.lower()
            chains.append(ChainSpec(tuple(joints), node(spring.get("center")), colliders_of, hair))
        return cls(chains, colliders)


class Springs:
    """The spring bones of a posed model, and where their tips are."""

    def __init__(self, spec: SpringSpec, skeleton: Skeleton) -> None:
        self.skeleton = skeleton
        rest_world = skeleton.rest_world
        self.chains = []
        for chain in spec.chains:
            parent = skeleton.parents[chain.joints[0].node]
            if parent < 0:
                continue
            nodes = [j.node for j in chain.joints]
            axes, lengths, ok = [], [], True
            for a, b in zip(nodes, nodes[1:]):
                t = skeleton.translation[b]
                n = np.linalg.norm(t)
                if n < 1e-9:
                    ok = False
                    break
                axes.append(t / n)
                lengths.append(np.linalg.norm(rest_world[b][:3, 3] - rest_world[a][:3, 3]))
            if ok:
                self.chains.append((chain, int(parent), nodes, axes, lengths))
        self.colliders = spec.colliders
        rest_local = compose_batch(skeleton.translation, skeleton.rest_rotation, skeleton.scale)
        # Level by level: level k holds the k-th joint of each chain long enough.
        depth = max((len(c[2]) - 1 for c in self.chains), default=0)
        self.levels = []
        for k in range(depth):
            members = [ci for ci, c in enumerate(self.chains) if len(c[2]) - 1 > k]
            chains = [self.chains[ci] for ci in members]
            node = np.array([c[2][k] for c in chains])
            params = [c[0].joints[k].params for c in chains]
            self.levels.append({
                "chains": np.array(members),
                "node": node,
                "rest": rest_local[node],  # the joints' rest transforms, local to their parents
                "axis": np.array([c[3][k] for c in chains]),
                "length": np.array([c[4][k] for c in chains]),
                "stiffness": np.array([p.stiffness for p in params]),
                "drag": np.array([p.drag for p in params]),
                "hit": np.array([p.hit_radius for p in params]),
                "gravity": np.array([p.gravity for p in params]) + np.array(
                    [(0.0, -HAIR_GRAVITY, 0.0) if c[0].hair else (0.0, 0.0, 0.0) for c in chains]),
                "tip": np.zeros((len(chains), 3)),
                "last": np.zeros((len(chains), 3)),
            })
        self.ends = np.array([c[2][-1] for c in self.chains], dtype=np.int64)
        count = len(self.chains)
        # Each chain's colliders, padded to the longest list (-1: none).
        width = max((len(c[0].colliders) for c in self.chains), default=0)
        self.chain_colliders = np.full((count, max(width, 1)), -1, dtype=np.int64)
        for ci, c in enumerate(self.chains):
            usable = [i for i in c[0].colliders if 0 <= i < len(self.colliders) and self.colliders[i] is not None]
            self.chain_colliders[ci, :len(usable)] = usable
        self.centers = [c[0].center if not c[0].hair else None for c in self.chains]
        self.carry = 0.0
        self.last_root: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chains)

    def simulate(self, dt: float) -> bool:
        """Step every chain and turn its bones toward its tips; the
        skeleton's world matrices are brought up to date. True while
        something is still swinging."""
        sk = self.skeleton
        if not self.chains:
            return False
        scale = float(np.linalg.norm(sk.root[:3, 0]))
        if scale <= 0.0:
            return False
        here = sk.root[:3, 3] / scale
        reset = self.last_root is None or float(np.linalg.norm(here - self.last_root)) > TELEPORT
        self.last_root = here
        self.carry = min(self.carry + dt, STEP * MAX_STEPS)
        steps = int(self.carry / STEP)
        self.carry -= steps * STEP

        world = sk.world
        count = len(self.chains)
        # World → each chain's space, for points (4×4) and directions (3×3).
        space = np.tile(np.diag([1.0 / scale, 1.0 / scale, 1.0 / scale, 1.0]), (count, 1, 1))
        turn = np.tile(np.eye(3), (count, 1, 1))
        for center in {c for c in self.centers if c is not None}:
            chains = [ci for ci, c in enumerate(self.centers) if c == center]
            space[chains] = np.linalg.inv(world[center])
            turn[chains] = rotation(world[center]).T

        # The colliders where the body is now, in each chain's space.
        shapes_a = np.zeros((len(self.colliders), 3))
        shapes_b = np.zeros((len(self.colliders), 3))
        radius = np.zeros(len(self.colliders))
        for i, c in enumerate(self.colliders):
            if c is None:
                continue
            m = world[c.node]
            shapes_a[i] = m[:3, :3] @ c.offset + m[:3, 3]
            shapes_b[i] = m[:3, :3] @ (c.tail if c.tail is not None else c.offset) + m[:3, 3]
            radius[i] = c.radius
        valid = self.chain_colliders >= 0
        index = np.where(valid, self.chain_colliders, 0)
        a_chain = transform(space[:, None], shapes_a[index])  # (chains, K, 3)
        b_chain = transform(space[:, None], shapes_b[index])
        r_chain = np.where(valid, radius[index], 0.0)

        parent = world[[c[1] for c in self.chains]].copy()  # (chains, 4, 4)
        settled = SETTLED_PX / scale
        moving = False
        for level in self.levels:
            idx = level["chains"]
            p = parent[idx]
            rest_local = level["rest"]
            rest_world = p @ rest_local
            head = transform(space[idx], rest_world[:, :3, 3])
            parent_turn = rotation(p)
            rest_turn = parent_turn @ rotation(rest_local)
            rest_dir = np.einsum("nij,nj->ni", turn[idx] @ rest_turn, level["axis"])
            if reset:
                level["tip"] = head + rest_dir * level["length"][:, None]
                level["last"] = level["tip"].copy()
            gravity = np.einsum("nij,nj->ni", turn[idx], level["gravity"])
            for _ in range(steps):
                moved = self._step(level, head, rest_dir, gravity, a_chain[idx], b_chain[idx], r_chain[idx],
                                   valid[idx])
                moving |= bool(np.any(moved > settled))
            # Turn each bone from its rest direction to its tip (back in world
            # space), under its parent.
            to_tip = level["tip"] - head
            n = np.linalg.norm(to_tip, axis=1, keepdims=True)
            direction = np.where(n > 1e-9, to_tip / np.maximum(n, 1e-12), rest_dir)
            arc = arc_matrices(rest_dir, direction)
            aim = turn[idx].transpose(0, 2, 1) @ arc @ turn[idx]
            local_rot = parent_turn.transpose(0, 2, 1) @ aim @ rest_turn
            quats = mat3_to_quat_batch(local_rot)
            sk.rotation[level["node"]] = quats
            local = rest_local.copy()
            local[:, :3, :3] = quat_to_mat3_batch(quats) * sk.scale[level["node"]][:, None, :]
            parent[idx] = p @ local
        sk.update()
        return moving

    @staticmethod
    def _step(level: dict, head, rest_dir, gravity, a, b, r, valid) -> np.ndarray:
        """One step for the bones' tips: carry on moving (less the drag),
        spring back toward the rest direction, fall, keep the bone's length
        and stay out of the colliders. Returns how far each tip moved."""
        tip, last, length = level["tip"], level["last"], level["length"][:, None]
        inertia = (tip - last) * (1.0 - level["drag"])[:, None]
        nxt = tip + inertia + (rest_dir * level["stiffness"][:, None] + gravity) * STEP
        nxt = head + normalized(nxt - head, rest_dir) * length
        nxt = push_out(nxt, head, length, level["hit"], a, b, r, valid)
        moved = np.linalg.norm(nxt - tip, axis=1)
        level["last"], level["tip"] = tip, nxt
        return moved


def rotation(m: np.ndarray) -> np.ndarray:
    """The rotation part (…, 3, 3) of affine matrices, scale taken out."""
    r = m[..., :3, :3]
    return r / np.linalg.norm(r, axis=-2, keepdims=True)


def transform(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Points ``p`` (…, 3) through affine matrices ``m`` (…, 4, 4)."""
    return np.einsum("...ij,...j->...i", m[..., :3, :3], p) + m[..., :3, 3]


def normalized(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-12, v / np.maximum(n, 1e-12), fallback)


def push_out(tip, head, length, hit, a, b, r, valid) -> np.ndarray:
    """Keep each tip out of its chain's colliders (spheres, or capsules round
    a segment), one after another, then back at the bone's length from its
    head. ``a``/``b``: (n, K, 3) collider ends (the same point for a
    sphere), ``r``: (n, K) radii, ``valid``: (n, K) which are colliders.

    All colliders are checked at once; only the first one some tip is inside
    is applied, then the rest are checked again from there — the same as
    going through them one by one, without the Python loop."""
    up = np.array([0.0, 1.0, 0.0])
    start, count = 0, a.shape[1]
    while start < count:
        aa, ab = a[:, start:], b[:, start:] - a[:, start:]
        along = np.einsum("nki,nki->nk", tip[:, None] - aa, ab) / np.maximum(np.einsum("nki,nki->nk", ab, ab), 1e-12)
        centre = aa + ab * np.clip(along, 0.0, 1.0)[..., None]
        reach = r[:, start:] + hit[:, None]
        away = tip[:, None] - centre
        inside = valid[:, start:] & (np.einsum("nki,nki->nk", away, away) < reach * reach)
        hits = np.flatnonzero(inside.any(axis=0))
        if len(hits) == 0:
            break
        k = hits[0]
        rows = inside[:, k]
        pushed = centre[:, k] + normalized(away[:, k], up) * reach[:, k][:, None]
        pushed = head + normalized(pushed - head, up) * length
        tip = np.where(rows[:, None], pushed, tip)
        start += k + 1
    return tip


def arc_matrices(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(n, 3, 3) shortest turns taking unit vectors ``src`` to ``dst``."""
    d = np.einsum("ni,ni->n", src, dst)
    c = np.cross(src, dst)
    q = np.concatenate([c, (1.0 + d)[:, None]], axis=1)
    opposite = d < -1.0 + 1e-6
    if np.any(opposite):
        axis = np.cross(np.array([1.0, 0.0, 0.0]), src[opposite])
        small = np.einsum("ni,ni->n", axis, axis) < 1e-8
        axis[small] = np.cross(np.array([0.0, 1.0, 0.0]), src[opposite][small])
        q[opposite] = np.concatenate([axis, np.zeros((len(axis), 1))], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return quat_to_mat3_batch(q)
