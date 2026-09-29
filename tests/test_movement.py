import math

from tomo.body.mathx import Vec2
from tomo.body.movement import COM_HEIGHT, BODY_RADIUS, ROLL_SECONDS, Locomotion, Stance, rotate, wrap

ARENA = Vec2(1920.0, 1080.0)
BODY = Vec2(80.0, 320.0)
DT = 1.0 / 60.0
SEED = 0x9E3779B9  # a fixed seed: idle decisions are the same every run


def run(loco: Locomotion, seconds: float) -> None:
    for _ in range(int(seconds / DT)):
        loco.step(DT, ARENA, BODY)


def standing(x: float) -> Locomotion:
    """Settled on its feet at ``x``."""
    loco = Locomotion(seed=SEED)
    loco.placed = True
    loco.com = Vec2(x, ARENA.y - COM_HEIGHT * BODY.y)
    loco.stance = Stance.STANDING
    return loco


def flying(com: Vec2, vel: Vec2, spin: float) -> Locomotion:
    loco = Locomotion(seed=SEED)
    loco.placed = True
    loco.com, loco.vel, loco.spin = com, vel, spin
    return loco


def inside(loco: Locomotion) -> bool:
    r = BODY_RADIUS * BODY.y
    for end, _ in loco.ends(BODY.y):
        p = loco.com + end
        if not (r - 1.0 <= p.x <= ARENA.x - r + 1.0 and r - 1.0 <= p.y <= ARENA.y - r + 1.0):
            return False
    return True


def test_drops_in_and_lands_on_its_feet():
    loco = Locomotion(seed=SEED)
    run(loco, 5.0)
    assert loco.stance == Stance.STANDING
    assert abs(loco.feet(BODY.y).y - ARENA.y) < 0.01, "feet on the floor"
    assert loco.is_idle()


def test_walks_to_the_target_and_stops_there():
    loco = standing(200.0)
    loco.walk_to(0.75)
    for _ in range(900):
        loco.step(DT, ARENA, BODY)
        assert loco.feet(BODY.y).x <= 0.75 * ARENA.x + 0.01, "never overshoots"
    assert abs(loco.feet(BODY.y).x - 0.75 * ARENA.x) < 0.01
    assert loco.is_idle()


def test_a_jump_goes_up_and_comes_back_down():
    loco = standing(960.0)
    loco.jump()
    peak = ARENA.y
    for _ in range(240):
        loco.step(DT, ARENA, BODY)
        peak = min(peak, loco.feet(BODY.y).y)
    assert ARENA.y - peak > 100.0, f"jumped {ARENA.y - peak} px"
    assert loco.stance == Stance.STANDING


def test_a_spin_goes_all_the_way_round():
    loco = flying(Vec2(960.0, 300.0), Vec2(0.0, -600.0), 15.0)
    turned = 0.0
    for _ in range(30):
        loco.step(DT, ARENA, BODY)
        turned = max(turned, abs(loco.angle))
    assert turned > math.tau, f"turned {turned} rad"


def test_a_tumbling_landing_knocks_it_down_then_it_gets_up():
    loco = flying(Vec2(960.0, 500.0), Vec2(300.0, 0.0), 9.0)
    lay_down = got_up = False
    for _ in range(900):
        loco.step(DT, ARENA, BODY)
        loco.wander(DT)
        lay_down |= loco.stance == Stance.LYING
        got_up |= lay_down and loco.stance == Stance.STANDING
        assert inside(loco), "stays in the arena"
    assert lay_down, "fell over"
    assert got_up, "and got back up"


def test_a_hard_throw_stays_on_screen_and_settles():
    loco = flying(Vec2(960.0, 600.0), Vec2(-3000.0, -900.0), 0.0)
    for _ in range(720):
        loco.step(DT, ARENA, BODY)
        loco.wander(DT)
        assert inside(loco), f"left the arena at {loco.com}"
    assert loco.grounded(), "came to rest on the floor"


def test_a_throw_carries_the_drag_speed():
    loco = standing(500.0)
    grip = loco.com + Vec2(0.0, -100.0)  # by the chest
    for i in range(11):
        loco.drag_to(grip + Vec2(20.0 * i, 0.0), DT)
        loco.step(DT, ARENA, BODY)
    loco.release()
    assert loco.stance == Stance.FLYING
    assert loco.vel.x > 800.0, f"thrown at {loco.vel.x} px/s"


def test_carried_it_hangs_back_from_a_sideways_drag():
    loco = standing(700.0)
    chest = loco.com + Vec2(0.0, -80.0)
    for i in range(60):
        # Hold still for half a second, then sweep right.
        x = (max(i, 30) - 30) * 25.0
        loco.drag_to(chest + Vec2(x, -300.0), DT)
        loco.step(DT, ARENA, BODY)
    # The body swings back to the left (clockwise), and the legs trail.
    assert loco.angle < -0.05, f"angle {loco.angle}"
    assert loco.limbs.legs[0].x < -0.05, f"legs at {loco.limbs.legs[0].x}"


def test_carried_by_the_feet_it_hangs_upside_down():
    loco = standing(900.0)
    feet = loco.feet(BODY.y) + Vec2(3.0, -2.0)
    for i in range(420):
        loco.drag_to(feet + Vec2(0.0, -500.0 * (min(i, 10) / 10.0)), DT)
        loco.step(DT, ARENA, BODY)
        # It tips over, swings, and settles head-down within ±45°.
        if i >= 360:
            assert math.cos(loco.angle) < -0.7, f"angle {wrap(loco.angle)}"


def test_it_sits_and_stands_back_up_to_walk():
    loco = standing(500.0)
    loco.sit()
    run(loco, 2.0)
    assert loco.stance == Stance.SITTING
    assert loco.seat > 0.95
    loco.walk_to(0.8)
    run(loco, 8.0)
    assert loco.seat < 0.01 and loco.is_idle()
    assert abs(loco.feet(BODY.y).x - 0.8 * ARENA.x) < 0.01, "walked there"


def test_it_lies_down_and_gets_up():
    loco = standing(500.0)
    loco.lie_down()
    run(loco, ROLL_SECONDS + 0.1)
    assert loco.stance == Stance.LYING
    assert abs(abs(wrap(loco.angle)) - math.pi / 2) < 0.01, "flat"
    assert abs(loco.com.y - (ARENA.y - BODY_RADIUS * BODY.y)) < 0.5, "on the floor"
    loco.stand_up()
    run(loco, ROLL_SECONDS + 0.5)
    assert loco.stance == Stance.STANDING
    assert abs(loco.feet(BODY.y).y - ARENA.y) < 0.01


def test_arms_float_up_in_a_fall():
    loco = flying(Vec2(960.0, 200.0), Vec2(0.0, 1800.0), 0.0)
    for _ in range(18):
        loco.step(DT, ARENA, BODY)
    assert loco.stance == Stance.FLYING, "still falling"
    assert loco.limbs.arms[0].x > 0.25, f"left arm at {loco.limbs.arms[0].x}"
    assert loco.limbs.arms[1].x < -0.25, f"right arm at {loco.limbs.arms[1].x}"


def test_landing_bends_the_knees_and_they_spring_back():
    loco = flying(Vec2(960.0, 400.0), Vec2(), 0.0)
    deepest = 0.0
    for _ in range(300):
        loco.step(DT, ARENA, BODY)
        deepest = max(deepest, loco.limbs.crouch.x)
    assert deepest > 0.1, f"squatted to {deepest}"
    assert loco.limbs.crouch.x == 0.0
    assert loco.at_rest(), "everything comes to rest"


def test_starting_to_walk_leans_into_it():
    loco = standing(200.0)
    loco.walk_to(0.9)
    run(loco, 0.1)
    # Accelerating right tips the top of the body right: clockwise.
    assert loco.angle < -0.01, f"angle {loco.angle}"


def test_left_alone_it_never_leaves_the_floor():
    # Idle life — strolling, sitting, napping, getting up — for two minutes,
    # whatever the dice say.
    for seed in range(1, 12):
        loco = standing(960.0)
        loco.rng = ((seed * 0x9E3779B9) & 0xFFFFFFFF) | 1
        for frame in range(7200):
            loco.step(DT, ARENA, BODY)
            loco.wander(DT)
            assert loco.grounded(), (f"seed {seed}: left the floor at frame {frame} as {loco.stance}, "
                                     f"vel {loco.vel}, spin {loco.spin}")


def test_rotate_turns_counter_clockwise_on_screen():
    right = rotate(Vec2(0.0, 1.0), math.pi / 2)
    assert (right - Vec2(1.0, 0.0)).length() < 1e-6
    assert abs(wrap(3.0 * math.pi) - math.pi) < 1e-5
