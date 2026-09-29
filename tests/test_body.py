"""The body's parts that need no window: the model, the rig and the mirror,
the lock screen's geometry, the chat's state machine, control, the session."""

import math
from pathlib import Path

import numpy as np
import pytest
from bodies import Body

from tomo import events
from tomo.body import vrm
from tomo.body.animation import Animator, Bone, Rig, animate_face, mirror_pose, pose_body, target_pose
from tomo.body.chat import DOCK_FRACTION, ChatState, Phase, panel_rect
from tomo.body.control import Control
from tomo.body.mathx import IDENTITY, Vec2, qmul, qrot
from tomo.body.movement import Locomotion, Stance
from tomo.body.skeleton import Skeleton
from tomo.body.window import Desktop, DisplayServer, detect_desktop, detect_server
from tomo.body.workout import Workout, hip_drop, mirrored
from tomo.events import ChatLine, Role
from tomo.health import (JOINT_COUNT, LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
                         RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST, Joint)

CHARACTERS = Path(__file__).resolve().parent.parent / "assets" / "characters"


def points(*items) -> list[Joint]:
    """A pose with only these joints seen: (index, (x, y, z))."""
    pose = [Joint(0.0, 0.0, 0.0, 0.0)] * JOINT_COUNT
    for i, (x, y, z) in items:
        pose[i] = Joint(x, y, z, 1.0)
    return pose


# ---- the model ----------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["female model.vrm", "Kiyotaka Ayanokōji.vrm"])
def model(request):
    path = CHARACTERS / request.param
    if not path.is_file():
        pytest.skip("the bundled characters aren't there")
    return vrm.load(path)


def test_a_vrm_loads_whole(model):
    assert model.version == "1.0"
    assert len(model.skins) == 3 and all(len(s.joints) == len(s.inverse_bind) for s in model.skins)
    for mesh in model.meshes:
        for p in mesh.primitives:
            v = p.vertices
            assert v.positions.shape == v.normals.shape and v.uvs.shape == (len(v.positions), 2)
            assert p.indices.max() < len(v.positions)
            assert np.allclose(v.weights.sum(axis=1), 1.0, atol=1e-4)
    face = next(m for m in model.meshes if m.primitives[0].vertices.targets)
    assert len(face.primitives[0].vertices.targets) == 57
    assert all(p.vertices is face.primitives[0].vertices for p in face.primitives), "one vertex set, shared"
    assert {"hips", "head", "leftUpperArm", "rightLowerLeg"} <= set(model.humanoid)
    assert {"blink", "aa", "oh", "happy"} <= set(model.expressions)


def test_at_rest_the_skin_is_where_the_mesh_is(model):
    sk = Skeleton(model)
    for i in range(len(model.skins)):
        # Bind pose = rest pose: every joint matrix is the identity (to the
        # exporter's rounding — a few thousandths).
        assert np.allclose(sk.skin(i), np.eye(4), atol=5e-3)


def test_the_body_is_measured_without_the_arms(model):
    sk = Skeleton(model)
    lo, hi = sk.bounds()
    height = hi[1] - lo[1]
    assert 1.4 < height < 2.1, "a person, in metres"
    assert abs(lo[1]) < 0.05, "standing on the origin"
    assert hi[0] - lo[0] <= 0.4 * height + 1e-6, "the T-pose arms are left out"


def test_the_rig_turns_bones_in_model_axes(model):
    sk = Skeleton(model)
    rig = Rig.build(model, sk)
    assert Bone.LEFT_UPPER_ARM in rig.bones and Bone.HIPS in rig.bones
    arm = rig.bones[Bone.LEFT_UPPER_ARM].node
    elbow = model.humanoid["leftLowerArm"]
    sk.update()
    start = sk.world[elbow][:3, 3] - sk.world[arm][:3, 3]
    # The T-pose arm points out to her left (+X), level.
    assert start[0] > 0.1 and abs(start[1]) < 0.05
    # Lowering it 72° (rz(-72)), as she stands at rest.
    rig.apply({Bone.LEFT_UPPER_ARM: qmul(IDENTITY, (0.0, 0.0, math.sin(math.radians(-36)),
                                                    math.cos(math.radians(-36))))}, 1e9, 1.0, sk)
    sk.update()
    end = sk.world[elbow][:3, 3] - sk.world[arm][:3, 3]
    angle = math.degrees(math.atan2(end[1], end[0]))
    assert -76 < angle < -68, f"the arm hangs at {angle:.1f}°"


def test_the_face_blinks_and_talks(model):
    sk = Skeleton(model)
    rig = Rig.build(model, sk)
    animator, loco = Animator(speaking=True), Locomotion(seed=1)
    loco.stance = Stance.STANDING
    blink_mesh, blink_index, _ = rig.expressions["blink"][0]
    seen_blink = seen_mouth = 0.0
    for _ in range(400):
        pose_body(1 / 60, rig, animator, loco, 0.0, IDENTITY, 320.0, sk, None)
        animate_face(1 / 60, rig, animator, loco, sk)
        seen_blink = max(seen_blink, sk.morph[blink_mesh][blink_index])
        aa_mesh, aa_index, _ = rig.expressions["aa"][0]
        seen_mouth = max(seen_mouth, sk.morph[aa_mesh][aa_index])
    # (A 0.15 s blink sampled at 60 fps peaks a little under 1.)
    assert seen_blink > 0.85 and seen_mouth > 0.5
    animator.speaking = False
    for _ in range(30):
        animate_face(1 / 60, rig, animator, loco, sk)
    assert sk.morph[aa_mesh][aa_index] == 0.0, "the mouth closes when the voice stops"


def test_a_wave_raises_the_right_arm():
    animator, loco = Animator(), Locomotion(seed=1)
    loco.stance = Stance.STANDING
    animator.cue(events.Animate("wave"))
    animator.gesture_elapsed = 1.2  # the middle of the wave
    pose = target_pose(animator, loco, 0.0, IDENTITY, False, 0.0)
    arm = qrot(pose[Bone.RIGHT_UPPER_ARM], (-1.0, 0.0, 0.0))  # her right arm points −X in the T-pose
    assert arm[1] > 0.5, f"raised: {arm}"
    rest = qrot(target_pose(Animator(), loco, 0.0, IDENTITY, False, 0.0)[Bone.RIGHT_UPPER_ARM], (-1.0, 0.0, 0.0))
    assert rest[1] < -0.8, "at rest it hangs down"


# ---- the mirror -------------------------------------------------------------------------------


def person(up, left):
    """A person in the camera's coordinates (x right, y down, z away): the
    direction their head is from their hips, and their own left."""
    up, left = np.array(up, float), np.array(left, float)
    return points((LEFT_SHOULDER, up * 0.5 + left * 0.18), (RIGHT_SHOULDER, up * 0.5 - left * 0.18),
                  (LEFT_HIP, left * 0.1), (RIGHT_HIP, -left * 0.1),
                  (LEFT_KNEE, left * 0.1 - up * 0.45), (RIGHT_KNEE, -left * 0.1 - up * 0.45),
                  (LEFT_ANKLE, left * 0.1 - up * 0.9), (RIGHT_ANKLE, -left * 0.1 - up * 0.9))


def belly(user):
    """Where Tomo's belly faces, mirroring ``user``."""
    return qrot(mirror_pose(user, {})[Bone.HIPS], (0.0, 0.0, 1.0))


def test_she_faces_the_way_a_mirror_image_would():
    # Real up is −y in the camera's coordinates. Standing, facing the camera
    # (their left on the image's right): she faces the user.
    assert belly(person((0, -1, 0), (1, 0, 0)))[2] > 0.9
    # A push-up plank, side-on, face down: she's face down too.
    assert belly(person((1, 0, 0), (0, 0, 1)))[1] < -0.9
    # A sit-up, lying on the back: so is she.
    assert belly(person((1, 0, 0), (0, 0, -1)))[1] > 0.9


def test_her_left_arm_follows_the_users_right():
    # Facing the camera, the user raises their right arm straight up.
    user = person((0, -1, 0), (1, 0, 0))
    s = user[RIGHT_SHOULDER]
    user[RIGHT_ELBOW] = Joint(s.x, s.y - 0.3, s.z, 1.0)
    user[RIGHT_WRIST] = Joint(s.x, s.y - 0.58, s.z, 1.0)
    pose = mirror_pose(user, {})
    raised = qrot(qmul(pose[Bone.HIPS], pose[Bone.LEFT_UPPER_ARM]), (1.0, 0.0, 0.0))
    assert raised[1] > 0.9, f"her left arm points up: {raised}"
    assert Bone.RIGHT_UPPER_ARM not in pose, "the user's unseen left arm stays as it was"


def test_the_mirror_image_swaps_sides_and_keeps_up_up():
    # The user's right hand, raised, to the image's left (x < 0).
    p = points((RIGHT_WRIST, (-0.5, -0.6, -0.1)))
    m = mirrored(p, RIGHT_WRIST)
    assert m[0] > 0.0, "on Tomo's left, the mirror side"
    assert m[1] > 0.0, "up stays up"
    assert m[2] > 0.0, "toward the camera is toward the viewer"
    assert mirrored(p, LEFT_WRIST) is None, "unseen joints are left out"


def legs(hip_y, knee, ankle):
    return points((LEFT_HIP, (0.1, hip_y, 0.0)), (RIGHT_HIP, (-0.1, hip_y, 0.0)),
                  (LEFT_KNEE, (0.1 + knee[0], knee[1], 0.0)), (RIGHT_KNEE, (-0.1 + knee[0], knee[1], 0.0)),
                  (LEFT_ANKLE, (0.1 + ankle[0], ankle[1], 0.0)), (RIGHT_ANKLE, (-0.1 + ankle[0], ankle[1], 0.0)))


def test_hips_drop_in_a_squat():
    assert hip_drop(legs(0.0, (0.0, 0.45), (0.0, 0.9))) < 0.05
    assert 0.4 < hip_drop(legs(0.0, (0.4, 0.2), (0.1, 0.45))) < 0.7


def test_a_push_up_moves_the_body_not_the_hands():
    # Side-on plank, hips at the origin, legs out level; the hands on the
    # floor ``reach`` below the hips (the arms' length, less as they bend).
    def plank(reach):
        return points((LEFT_HIP, (0.0, 0.0, 0.1)), (RIGHT_HIP, (0.0, 0.0, -0.1)),
                      (LEFT_KNEE, (-0.45, 0.03, 0.1)), (RIGHT_KNEE, (-0.45, 0.03, -0.1)),
                      (LEFT_ANKLE, (-0.9, 0.06, 0.1)), (RIGHT_ANKLE, (-0.9, 0.06, -0.1)),
                      (LEFT_WRIST, (0.5, reach, 0.2)), (RIGHT_WRIST, (0.5, reach, -0.2)))

    up, down = hip_drop(plank(0.55)), hip_drop(plank(0.2))
    assert 0.3 < up < 0.5, up
    assert down > up + 0.3, f"the body comes down: {up} → {down}"


def test_the_mirror_follows_a_real_squat():
    # From the health tests' synthetic body: standing, then a deep squat.
    user = Body(knees=80.0, hips=90.0).pose()
    pose = mirror_pose(user, {})
    thigh = qrot(qmul(pose[Bone.HIPS], pose[Bone.LEFT_UPPER_LEG]), (0.0, -1.0, 0.0))
    assert abs(thigh[1]) < 0.5, f"the thighs come up toward level: {thigh}"


def test_the_lock_follows_the_brain():
    w = Workout()
    assert w.receive(events.HealthLock(True)) and w.locked
    assert not w.receive(events.HealthLock(True)), "no change, nothing to do"
    w.receive(events.Pose([Joint(0, 0, 0, 1)]))
    assert w.pose is not None
    assert w.receive(events.HealthLock(False)) and w.pose is None and w.progress is None


# ---- the chat ------------------------------------------------------------------------------------


def settled(x=300.0):
    loco = Locomotion(seed=5)
    loco.placed, loco.stance = True, Stance.STANDING
    loco.com = Vec2(x, 1080.0 - 0.55 * 320.0)
    return loco


def test_a_click_walks_her_over_then_the_chat_grows_out_of_her():
    chat, loco = ChatState(), settled()
    chat.character_clicked(loco)
    assert chat.phase == Phase.SLIDING and loco.held and loco.target == DOCK_FRACTION
    for _ in range(600):
        loco.step(1 / 60, Vec2(1920, 1080), Vec2(80, 320))
        chat.advance(1 / 60, loco)
    assert chat.phase == Phase.OPEN and chat.liquid_t == 1.0
    assert abs(loco.feet(320).x - DOCK_FRACTION * 1920) < 1.0
    # Clicking away closes it; she's free to wander again.
    chat.dismiss(False, (100.0, 100.0), (1600, 700, 1700, 1080), 1920, 1080)
    assert chat.phase == Phase.CLOSING
    for _ in range(60):
        chat.advance(1 / 60, loco)
    assert chat.phase == Phase.CLOSED and not loco.held


def test_clicks_on_the_panel_or_her_dont_close_it():
    chat = ChatState(phase=Phase.OPEN, liquid_t=1.0)
    x0, y0, x1, y1 = panel_rect(1920, 1080)
    chat.dismiss(False, ((x0 + x1) / 2, (y0 + y1) / 2), None, 1920, 1080)
    chat.dismiss(False, (1650.0, 900.0), (1600, 700, 1700, 1080), 1920, 1080)
    assert chat.phase == Phase.OPEN
    chat.dismiss(True, None, None, 1920, 1080)
    assert chat.phase == Phase.CLOSING, "Escape closes it"


def test_hey_tomo_hops_and_opens_the_chat():
    chat, loco = ChatState(), settled()
    chat.receive(events.Listening(True), loco)
    assert loco.stance == Stance.FLYING, "a hop"
    assert chat.phase == Phase.SLIDING and chat.listening


def test_the_transcript_keeps_the_last_lines():
    chat = ChatState()
    for i in range(450):
        chat.receive(events.Chat(ChatLine(Role.USER, f"line {i}")), None)
    assert len(chat.transcript) == 400 and chat.transcript[-1].text == "line 449"


def pictures_in(folder, *names):
    from PIL import Image

    paths = []
    for name in names:
        path = folder / name
        if path.suffix.lower() in (".png", ".jpg"):
            Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
        else:
            path.write_text("not a picture")
        paths.append(path)
    return paths


def test_pictures_wait_for_the_next_message_up_to_four(tmp_path):
    chat = ChatState()
    a, b, c, d, e, notes = pictures_in(tmp_path, "a.png", "b.jpg", "c.png", "d.png", "e.png", "notes.txt")
    assert chat.add_images([a, notes, tmp_path / "gone.png"]) == 1
    assert chat.notice == "Only pictures can be sent"
    assert chat.add_images([a, b, c, d, e]) == 3, "one was there already, and four is the most"
    assert chat.attachments == [a, b, c, d] and chat.notice == "Up to 4 pictures per message"
    sent = []
    chat.input = "  what are these?  "
    assert chat.send_message(sent.append)
    assert sent == [events.UserMessage("what are these?", (a, b, c, d))]
    assert chat.attachments == [] and chat.input == ""
    assert not chat.send_message(sent.append), "nothing to send"
    chat.add_images([e])
    assert chat.send_message(sent.append) and sent[-1] == events.UserMessage("", (e,)), "a picture alone goes too"


def test_a_pasted_picture_or_copied_files_are_attached(tmp_path, monkeypatch):
    from PIL import Image

    from tomo import attachments
    from tomo.body.chat import paste_images

    monkeypatch.setattr(attachments.tempfile, "gettempdir", lambda: str(tmp_path))
    got = []
    paste_images(got.append, lambda: Image.new("RGB", (30, 20), (0, 0, 255)))  # a screenshot
    [[pasted]] = got
    assert attachments.was_pasted(pasted) and Image.open(pasted).size == (30, 20)
    a, notes = pictures_in(tmp_path, "a.png", "notes.txt")
    paste_images(got.append, lambda: [str(a), str(notes)])  # files copied in the file manager
    assert got[-1] == [a]
    paste_images(got.append, lambda: None)  # text on the clipboard: nothing to attach

    def no_clipboard_tool():
        raise NotImplementedError("wl-paste or xclip is required")

    paste_images(got.append, no_clipboard_tool)
    assert len(got) == 2
    chat = ChatState()
    chat.picked.put([a])  # as the file dialog's thread hands them over
    chat.take_picked()
    assert chat.attachments == [a]


# ---- control -------------------------------------------------------------------------------------


def test_a_click_waits_until_she_has_walked_over(monkeypatch):
    clicks = []
    monkeypatch.setattr("tomo.body.control.click", lambda x, y, double: clicks.append((x, y, double)))
    control, loco = Control(), settled(300.0)
    loco.arena = Vec2(1920, 1080)
    control.receive(events.ClickAt(1500.0, 400.0, True), loco, lambda x, y: (x, y))
    assert control.active and loco.held and control.pending
    control.update(loco)
    assert clicks == [], "not before she's there"
    for _ in range(900):
        loco.step(1 / 60, Vec2(1920, 1080), Vec2(80, 320))
        control.update(loco)
    assert clicks == [(1500.0, 400.0, True)]
    assert not control.active and not loco.held


def test_the_panic_key_releases_control():
    sent = []
    control, loco = Control(active=True, pending=(1.0, 2.0, False)), settled()
    loco.held = True
    control.panic(loco, sent.append)
    assert not control.active and control.pending is None and not loco.held
    assert isinstance(sent[0], events.PanicStop)


# ---- the desktop session -------------------------------------------------------------------------


def test_the_session_type_wins_then_the_display_variables():
    assert detect_server("wayland", None, ":0") == DisplayServer.WAYLAND
    assert detect_server("x11", "wayland-0", None) == DisplayServer.X11
    assert detect_server(None, "wayland-0", None) == DisplayServer.WAYLAND
    assert detect_server(None, None, ":0") == DisplayServer.X11
    assert detect_server(None, None, None) == DisplayServer.UNKNOWN


def test_desktops_are_recognised():
    assert detect_desktop("ubuntu:GNOME") == Desktop.GNOME
    assert detect_desktop("KDE") == Desktop.KDE
    assert detect_desktop("X-Cinnamon") == Desktop.CINNAMON
    assert detect_desktop("Hyprland") == Desktop.HYPRLAND
    assert detect_desktop("i3") == Desktop.I3
    assert detect_desktop("whatever") == Desktop.OTHER
