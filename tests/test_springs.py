import time
from pathlib import Path

import numpy as np
import pytest

from tomo.body import vrm
from tomo.body.mathx import compose
from tomo.body.skeleton import Skeleton
from tomo.body.springs import HAIR_GRAVITY, STEP, SpringSpec, Springs, push_out

CHARACTERS = Path(__file__).resolve().parent.parent / "assets" / "characters"


def gltf():
    return {
        "nodes": [{"name": "Head"}, {"name": "Hair1"}, {"name": "Hair2"}, {"name": "Skirt1"}, {"name": "Skirt2"}],
        "extensions": {"VRMC_springBone": {
            "colliders": [
                {"node": 0, "shape": {"sphere": {"offset": [0, 0.1, 0], "radius": 0.1}}},
                {"node": 0, "shape": {"capsule": {"offset": [0, 0, 0], "radius": 0.05, "tail": [0, 0.2, 0]}}},
                {"node": 0, "shape": {"plane": {}}},
            ],
            "colliderGroups": [{"colliders": [0, 1]}],
            "springs": [
                {"name": "Hair", "colliderGroups": [0], "joints": [
                    {"node": 1, "stiffness": 0.8, "dragForce": 0.4, "hitRadius": 0.01}, {"node": 2}]},
                {"name": "Skirt", "joints": [{"node": 3, "gravityPower": 0.5}, {"node": 4}]},
                {"name": "Lonely", "joints": [{"node": 3}]},
            ],
        }},
    }


def test_it_reads_chains_and_colliders():
    spec = SpringSpec.from_gltf(gltf())
    assert len(spec.chains) == 2, "a one-joint chain has nothing to swing"
    hair = spec.chains[0]
    assert hair.hair and not spec.chains[1].hair
    assert hair.joints[0].node == 1
    assert hair.joints[0].params.stiffness == 0.8
    assert hair.joints[1].params.stiffness == 1.0, "the spec's default"
    assert hair.colliders == (0, 1)
    assert spec.chains[1].joints[0].params.gravity == (0.0, -0.5, 0.0)
    assert spec.colliders[0].tail is None and spec.colliders[0].radius == 0.1
    assert spec.colliders[1].tail == (0.0, 0.2, 0.0)
    assert spec.colliders[2] is None, "unknown shapes are skipped"


def test_a_model_without_springs_has_none():
    assert SpringSpec.from_gltf({"nodes": []}).chains == []


def joint(stiffness, drag, tip):
    tip = np.array([tip], dtype=float)
    return {"tip": tip, "last": tip.copy(), "length": np.array([0.1]), "drag": np.array([drag]),
            "stiffness": np.array([stiffness]), "hit": np.array([0.0])}


NONE = (np.zeros((1, 0, 3)), np.zeros((1, 0, 3)), np.zeros((1, 0)), np.zeros((1, 0), dtype=bool))
DOWN, UP = np.array([[0.0, -1.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])
HEAD = np.zeros((1, 3))


def settle(j, rest_dir, gravity, seconds):
    for _ in range(int(seconds / STEP)):
        Springs._step(j, HEAD, rest_dir, gravity, *NONE)


def test_a_tip_at_rest_stays_put():
    j = joint(0.8, 0.4, (0.0, -0.1, 0.0))
    assert Springs._step(j, HEAD, DOWN, np.zeros((1, 3)), *NONE)[0] < 1e-6


def test_a_swung_tip_springs_back_and_settles():
    # Swung out sideways, it swings back down, and comes to rest there.
    j = joint(0.8, 0.4, (0.1, 0.0, 0.0))
    settle(j, DOWN, np.zeros((1, 3)), 0.2)
    assert j["tip"][0, 0] < 0.09, f"it swings back: {j['tip']}"
    settle(j, DOWN, np.zeros((1, 3)), 5.0)
    assert np.linalg.norm(j["tip"][0] - [0.0, -0.1, 0.0]) < 0.002, f"settled at rest: {j['tip']}"
    assert Springs._step(j, HEAD, DOWN, np.zeros((1, 3)), *NONE)[0] < 1e-4
    assert abs(np.linalg.norm(j["tip"][0]) - 0.1) < 1e-5, "the bone keeps its length"


def test_upside_down_hair_falls_the_other_way():
    # Styled to point "down" from a head now upside down: its rest direction
    # points up. Hair gravity beats the spring.
    tilted = np.array([0.01, 0.1, 0.0]) / np.linalg.norm([0.01, 0.1, 0.0]) * 0.1
    j = joint(0.8, 0.4, tilted)
    settle(j, UP, np.array([[0.0, -HAIR_GRAVITY, 0.0]]), 5.0)
    assert j["tip"][0, 1] < -0.05, f"hangs down: {j['tip']}"
    # Without the extra pull, it keeps its style.
    j = joint(0.8, 0.4, tilted)
    settle(j, UP, np.zeros((1, 3)), 5.0)
    assert j["tip"][0, 1] > 0.09, f"stays up: {j['tip']}"


def one(shape_a, shape_b, radius):
    return (np.array([[shape_a]], float), np.array([[shape_b]], float), np.array([[radius]], float),
            np.array([[True]]))


def test_tips_are_pushed_out_of_colliders_and_keep_their_length():
    head = np.array([[0.0, 0.3, 0.0]])
    inside = np.array([[0.05, 0.1, 0.0]])
    out = push_out(inside, head, np.array([[0.2]]), np.array([0.01]), *one((0, 0.1, 0), (0, 0.1, 0), 0.1))
    assert abs(np.linalg.norm(out[0] - head[0]) - 0.2) < 1e-5
    # Out to the surface (holding the length can leave it a hair inside, as in
    # the spec's own single pass).
    assert np.linalg.norm(out[0] - [0.0, 0.1, 0.0]) > 0.095
    clear = np.array([[0.5, 0.1, 0.0]])
    assert np.array_equal(push_out(clear, head, np.array([[0.2]]), np.array([0.01]), *one((0, 0, 0), (0, 0, 0), 0.1)),
                          clear)
    # Beside a capsule's middle, pushed straight out from its axis.
    out = push_out(np.array([[0.02, 0.1, 0.0]]), np.array([[1.0, 0.1, 0.0]]), np.array([[0.95]]), np.array([0.0]),
                   *one((0, 0, 0), (0, 0.2, 0), 0.05))
    assert np.linalg.norm(out[0] - [0.05, 0.1, 0.0]) < 1e-5


# ---- on a real character ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def female():
    path = CHARACTERS / "female model.vrm"
    if not path.is_file():
        pytest.skip("the bundled character isn't there")
    return vrm.load(path)


def placed(model, x=0.0):
    sk = Skeleton(model)
    lo, hi = sk.bounds()
    s = 320.0 / (hi[1] - lo[1])
    sk.root = compose((x, -200.0, 0.0), (0.0, 0.0, 0.0, 1.0), (s, s, s))
    sk.update()
    return sk


def hair_tip_offset(sk, springs):
    """Where the hair's chain ends sit, relative to the model's root, px."""
    return sk.world[springs.ends][:, :3, 3] - sk.root[:3, 3]


def test_standing_still_the_hair_rests_near_its_style(female):
    sk = placed(female)
    springs = Springs(SpringSpec.from_gltf(female.json), sk)
    assert len(springs) == 32
    styled = hair_tip_offset(sk, springs).copy()
    for _ in range(240):
        springs.simulate(1 / 60)
    drift = np.linalg.norm(hair_tip_offset(sk, springs) - styled, axis=1)
    # Gravity and the springs settle close to the styled shape: a few px at
    # most on a 320 px character.
    assert np.max(drift) < 25.0, drift
    assert not springs.simulate(1 / 60), "it has settled"


def test_a_sudden_move_swings_the_hair_and_it_settles_again(female):
    sk = placed(female)
    springs = Springs(SpringSpec.from_gltf(female.json), sk)
    for _ in range(120):
        springs.simulate(1 / 60)
    before = hair_tip_offset(sk, springs).copy()
    swung = 0.0
    started = time.perf_counter()
    for frame in range(60):
        # Carried quickly to the right for a quarter of a second.
        x = min(frame, 15) * 20.0
        s = np.linalg.norm(sk.root[:3, 0])
        sk.root = compose((x, -200.0, 0.0), (0.0, 0.0, 0.0, 1.0), (s, s, s))
        sk.update()
        springs.simulate(1 / 60)
        swung = max(swung, float(np.max(np.linalg.norm(hair_tip_offset(sk, springs) - before, axis=1))))
    per_frame = (time.perf_counter() - started) / 60
    assert swung > 5.0, f"the hair swung {swung:.1f} px"
    for _ in range(600):
        springs.simulate(1 / 60)
    assert np.max(np.linalg.norm(hair_tip_offset(sk, springs) - before, axis=1)) < 25.0
    assert per_frame < 0.02, f"{per_frame * 1000:.1f} ms a frame"
