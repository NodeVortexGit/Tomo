"""The posed model: every node's transform, and where it all is this frame.

A :class:`Skeleton` holds each node's local translation, rotation and scale
(animation and the spring bones turn the bones by changing rotations) and,
after :meth:`Skeleton.update`, every node's world matrix — the character's
placement on the screen (``root``) times its path down the node tree. The
skinned meshes follow their joints' world matrices (:meth:`Skeleton.skin`),
and the blend shapes of the face are the per-mesh ``morph`` weights.
"""

from __future__ import annotations

import numpy as np

from .mathx import compose_batch
from .vrm import Vrm

# The body's half-width against its height, for the grab box: the meshes'
# bounds hold the T-pose, arms out wide, but posed they hang at the sides.
BODY_HALF_WIDTH = 0.2


class Skeleton:
    def __init__(self, model: Vrm) -> None:
        nodes = model.nodes
        self.model = model
        self.count = len(nodes)
        self.parents = np.array([n.parent for n in nodes], dtype=np.int64)
        self.translation = np.array([n.translation for n in nodes], dtype=np.float64).reshape(-1, 3)
        self.rotation = np.array([n.rotation for n in nodes], dtype=np.float64).reshape(-1, 4)
        self.scale = np.array([n.scale for n in nodes], dtype=np.float64).reshape(-1, 3)
        self.rest_rotation = self.rotation.copy()
        # The character's placement: model space → the world (screen pixels).
        self.root = np.eye(4)
        self.world = np.tile(np.eye(4), (self.count, 1, 1))
        self.morph = [np.array(m.weights, dtype=np.float32) for m in model.meshes]
        self.morph_changed = [True] * len(model.meshes)
        # Nodes by depth, so each level's world matrices are one batched
        # product with their parents' (already done) ones.
        depth = np.zeros(self.count, dtype=np.int64)
        for i in range(self.count):
            d, p = 0, self.parents[i]
            while p >= 0 and d <= self.count:
                d, p = d + 1, self.parents[p]
            depth[i] = d
        self.levels = [np.flatnonzero(depth == d) for d in range(int(depth.max(initial=0)) + 1)]
        self.update()
        self.rest_world = self.world.copy()  # at rest, with the root at the origin

    def reset(self) -> None:
        """Back to the rest pose."""
        self.rotation[:] = self.rest_rotation

    def update(self) -> None:
        """Every node's world matrix from the local transforms and the root."""
        local = compose_batch(self.translation, self.rotation, self.scale)
        roots = self.levels[0]
        self.world[roots] = self.root @ local[roots]
        for level in self.levels[1:]:
            self.world[level] = self.world[self.parents[level]] @ local[level]

    def skin(self, index: int) -> np.ndarray:
        """A skin's joint matrices (joints, 4, 4): bind pose → posed world."""
        skin = self.model.skins[index]
        return self.world[skin.joints] @ skin.inverse_bind

    def set_morph(self, mesh: int, index: int, weight: float) -> None:
        weights = self.morph[mesh]
        if index < len(weights) and weights[index] != weight:
            weights[index] = weight
            self.morph_changed[mesh] = True

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The model's box at rest, in model units (min, max) — the body only:
        the arms stretched out in the T-pose are left out (see
        ``BODY_HALF_WIDTH``)."""
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for i, node in enumerate(self.model.nodes):
            if node.mesh is None:
                continue
            m = self.rest_world[i]
            for p in self.model.meshes[node.mesh].primitives:
                pts = p.vertices.positions @ m[:3, :3].T + m[:3, 3]
                lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
        if not np.all(lo <= hi):
            return None
        half = BODY_HALF_WIDTH * (hi[1] - lo[1])
        lo[0], hi[0] = max(lo[0], -half), min(hi[0], half)
        return lo, hi


def blend(positions: np.ndarray, normals: np.ndarray, deltas: np.ndarray, normal_deltas: np.ndarray | None,
          weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices with blend shapes applied: ``deltas`` (targets, n, 3)."""
    active = np.flatnonzero(weights[:len(deltas)])
    if len(active) == 0:
        return positions, normals
    w = weights[active]
    pos = positions + np.tensordot(w, deltas[active], axes=1)
    if normal_deltas is None:
        return pos.astype(np.float32), normals
    nrm = normals + np.tensordot(w, normal_deltas[active], axes=1)
    return pos.astype(np.float32), nrm.astype(np.float32)
