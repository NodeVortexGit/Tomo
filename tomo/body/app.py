"""The body's main loop: every frame, the brain's news, the mouse, the
physics, the pose, the drawing.

:class:`App` owns the window and everything in it. A frame:

1. drain the brain's messages and hand each to whoever acts on it;
2. finish loading a character, if one was being read in the background;
3. the mouse: pick her up, throw her, click her (opens the chat);
4. her physics, idle life, the chat's animation, pictures dropped on her or
   the chat (or pasted into it), a pending real click;
5. place her on screen — or, during a health round, big in the middle as
   the user's mirror — then pose her bones and face and swing her hair;
6. draw her, then the interface (chat, lock screen card, control badge);
7. show the frame: at the display's rate while anything moves, 24 a second
   at rest.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui

from .. import events
from ..brain import BrainHandle
from ..events import Shutdown
from . import vrm
from .animation import Animator, Rig, animate_face, pose_body
from .chat import ChatState, Phase
from .control import Control, panic_pressed_anywhere
from .mathx import IDENTITY, Quat, Vec2, compose, qmul, qslerp, qaxis
from .movement import (CROUCH_DROP, DRAG_THRESHOLD, SIT_DROP, TURN_RATE, Locomotion, body_size)
from .renderer import Renderer, View
from .skeleton import Skeleton
from .springs import SpringSpec, Springs
from .ui import Ui
from .window import Session, Window
from .workout import BACKDROP, KeyboardHold, Workout, hip_drop

log = logging.getLogger(__name__)

HEIGHT_PX = 320.0  # her height on screen, logical px
MIRROR_HEIGHT = 0.62  # her height while she mirrors the user, a share of the screen's
LEG_SHARE = 0.48  # the legs' share of her height: how far she comes down with the user's hips
GRAB_MARGIN = 4.0  # px around her box that still catch the mouse

KEYS = {
    glfw.KEY_TAB: imgui.Key.tab, glfw.KEY_LEFT: imgui.Key.left_arrow, glfw.KEY_RIGHT: imgui.Key.right_arrow,
    glfw.KEY_UP: imgui.Key.up_arrow, glfw.KEY_DOWN: imgui.Key.down_arrow, glfw.KEY_PAGE_UP: imgui.Key.page_up,
    glfw.KEY_PAGE_DOWN: imgui.Key.page_down, glfw.KEY_HOME: imgui.Key.home, glfw.KEY_END: imgui.Key.end,
    glfw.KEY_INSERT: imgui.Key.insert, glfw.KEY_DELETE: imgui.Key.delete, glfw.KEY_BACKSPACE: imgui.Key.backspace,
    glfw.KEY_SPACE: imgui.Key.space, glfw.KEY_ENTER: imgui.Key.enter, glfw.KEY_ESCAPE: imgui.Key.escape,
    glfw.KEY_KP_ENTER: imgui.Key.keypad_enter, glfw.KEY_A: imgui.Key.a, glfw.KEY_C: imgui.Key.c,
    glfw.KEY_V: imgui.Key.v, glfw.KEY_X: imgui.Key.x, glfw.KEY_Y: imgui.Key.y, glfw.KEY_Z: imgui.Key.z,
    glfw.KEY_LEFT_CONTROL: imgui.Key.left_ctrl, glfw.KEY_RIGHT_CONTROL: imgui.Key.right_ctrl,
    glfw.KEY_LEFT_SHIFT: imgui.Key.left_shift, glfw.KEY_RIGHT_SHIFT: imgui.Key.right_shift,
    glfw.KEY_LEFT_ALT: imgui.Key.left_alt, glfw.KEY_RIGHT_ALT: imgui.Key.right_alt,
    glfw.KEY_LEFT_SUPER: imgui.Key.left_super, glfw.KEY_RIGHT_SUPER: imgui.Key.right_super,
}


@dataclass
class Character:
    """The character on screen."""

    path: Path
    model: vrm.Vrm
    skeleton: Skeleton
    rig: Rig
    springs: Springs
    gpu: object
    bounds: tuple[np.ndarray, np.ndarray]  # model units, the body only
    facing: float = 0.0  # +1 right, -1 left, 0 the viewer
    yaw: Quat = IDENTITY  # the turn she's showing, easing toward ``facing``
    screen_rect: tuple[float, float, float, float] | None = None  # logical px: what the mouse can grab


@dataclass
class Grab:
    pressed_at: Vec2 | None = None  # where a press on her started
    dragging: bool = False


@dataclass
class Frame:
    """This frame's input, from the window's callbacks."""

    presses: list[Vec2] = field(default_factory=list)  # left button went down here
    escape: bool = False
    panic: bool = False
    dropped: list[str] = field(default_factory=list)  # files dropped on her or the chat
    paste: bool = False  # Ctrl+V


class App:
    def __init__(self, brain: BrainHandle) -> None:
        self.brain = brain
        self.session = Session.detect()
        self.session.log_notes()
        self.window = Window(self.session)
        self.ctx = moderngl.create_context()
        log.info("drawing with %s (OpenGL %s)", self.ctx.info.get("GL_RENDERER"), self.ctx.version_code)
        self.renderer = Renderer(self.ctx, colour_key=self.window.colour_key)
        self.ui = Ui(self.ctx)
        imgui.get_io().config_input_text_enter_keep_active = True
        self._clipboard()
        self.loco = Locomotion()
        self.animator = Animator()
        self.chat = ChatState()
        self.workout = Workout()
        self.control = Control()
        self.keyboard = KeyboardHold()
        self.grab = Grab()
        self.frame_input = Frame()
        self.character: Character | None = None
        self.loading: tuple[Path, threading.Thread, list] | None = None
        self.running = True
        self.clock = 0.0
        self._callbacks()

    # ---- the window's input -----------------------------------------------------------------

    def _callbacks(self) -> None:
        w = self.window.handle

        def on_cursor(_, x, y):
            self.ui.mouse_moved(*self.window.to_logical(x, y))

        def on_enter(_, entered):
            if not entered:
                self.ui.mouse_left()

        def on_button(_, button, action, mods):
            down = action == glfw.PRESS
            self.ui.mouse_button(button, down)
            if button == glfw.MOUSE_BUTTON_LEFT and down:
                self.frame_input.presses.append(Vec2(*self.window.cursor()))

        def on_scroll(_, dx, dy):
            self.ui.scrolled(dx, dy)

        def on_key(_, key, scancode, action, mods):
            down = action != glfw.RELEASE
            self.ui.key(KEYS.get(key), down, bool(mods & glfw.MOD_CONTROL), bool(mods & glfw.MOD_SHIFT),
                        bool(mods & glfw.MOD_ALT), bool(mods & glfw.MOD_SUPER))
            if action == glfw.PRESS:
                ctrl_alt = mods & glfw.MOD_CONTROL and mods & glfw.MOD_ALT
                if key == glfw.KEY_PAUSE or (key == glfw.KEY_ESCAPE and ctrl_alt):
                    self.frame_input.panic = True
                elif key == glfw.KEY_ESCAPE:
                    self.frame_input.escape = True
                elif key == glfw.KEY_V and mods & glfw.MOD_CONTROL:
                    self.frame_input.paste = True

        def on_char(_, codepoint):
            self.ui.typed(codepoint)

        def on_drop(_, paths):
            self.frame_input.dropped.extend(paths)

        glfw.set_cursor_pos_callback(w, on_cursor)
        glfw.set_cursor_enter_callback(w, on_enter)
        glfw.set_mouse_button_callback(w, on_button)
        glfw.set_scroll_callback(w, on_scroll)
        glfw.set_key_callback(w, on_key)
        glfw.set_char_callback(w, on_char)
        glfw.set_drop_callback(w, on_drop)

    def _clipboard(self) -> None:
        w = self.window.handle
        platform_io = imgui.get_platform_io()

        def get(_ctx):
            text = glfw.get_clipboard_string(w)
            return text.decode("utf-8", "replace") if isinstance(text, bytes) else (text or "")

        def put(_ctx, text):
            glfw.set_clipboard_string(w, text)

        platform_io.platform_get_clipboard_text_fn = get
        platform_io.platform_set_clipboard_text_fn = put

    # ---- the brain's news ---------------------------------------------------------------------

    def dispatch(self, event) -> None:
        loco = self.loco if self.character is not None else None
        if isinstance(event, events.LoadCharacter):
            self.load(Path(event.path))
        elif isinstance(event, events.WalkTo):
            self.loco.walk_to(event.position)
            self.loco.idle_timer = 4.0  # don't wander off right after being asked
        elif isinstance(event, events.Animate):
            self.animator.cue(event)
            moves = {"jump": self.loco.jump, "sit": self.loco.sit, "lie_down": self.loco.lie_down,
                     "idle": self.loco.stand_up}
            if event.clip in moves:
                moves[event.clip]()
        elif isinstance(event, (events.Emote, events.Speaking)):
            self.animator.cue(event)
        elif isinstance(event, (events.Thinking, events.Listening, events.Chat)):
            self.animator.cue(event)
            self.chat.receive(event, loco)
        elif isinstance(event, events.Characters):
            self.chat.receive(event, loco)
        elif isinstance(event, (events.ClickAt, events.TypeText, events.ControlMode)):
            self.control.receive(event, loco, self.window.screen_to_logical)
        elif isinstance(event, (events.HealthLock, events.HealthProgress, events.Pose)):
            if self.workout.receive(event):
                self.keyboard.set(self.workout.locked)
                self.window.cover(self.workout.locked)
                self.grab = Grab()
        elif isinstance(event, events.Status):
            log.info("status: %s", event.text)

    # ---- characters ----------------------------------------------------------------------------

    def load(self, path: Path) -> None:
        """Read a .vrm in the background; it replaces the one showing once
        it's ready (see :meth:`finish_loading`)."""
        if self.character is not None and self.character.path == path:
            return
        if self.loading is not None and self.loading[0] == path:
            return
        log.info("loading character: %s", path)
        result: list = []

        def read() -> None:
            try:
                model = vrm.load(path)
                result.append((model, vrm.decode_images(model)))
            except Exception as e:  # noqa: BLE001
                log.warning("couldn't load %s: %s", path, e)
                result.append(None)

        thread = threading.Thread(target=read, name="tomo-load-character", daemon=True)
        thread.start()
        self.loading = (path, thread, result)

    def finish_loading(self) -> None:
        if self.loading is None or self.loading[1].is_alive():
            return
        path, _, result = self.loading
        self.loading = None
        if not result or result[0] is None:
            return
        model, images = result[0]
        skeleton = Skeleton(model)
        bounds = skeleton.bounds()
        if bounds is None:
            log.warning("%s has no meshes to show", path.name)
            return
        gpu = self.renderer.upload(model, images)
        rig = Rig.build(model, skeleton)
        springs = Springs(SpringSpec.from_gltf(model.json), skeleton)
        log.info("%s: %d bones rigged, %d expressions, %d spring chains", path.name, len(rig.bones),
                 len(rig.expressions), len(springs))
        if self.character is not None:
            self.character.gpu.release()
        self.character = Character(path, model, skeleton, rig, springs, gpu, bounds)
        # A new body drops in from the top (and, if the chat is out, walks
        # over to it: see ChatState.advance).
        self.loco = Locomotion()

    # ---- a frame ------------------------------------------------------------------------------

    def run(self) -> None:
        last = time.perf_counter()
        try:
            while self.running and not self.window.should_close():
                glfw.poll_events()
                now = time.perf_counter()
                dt, last = min(now - last, 0.25), now
                self.frame(dt)
        finally:
            self.brain.send(Shutdown())
            self.keyboard.set(False)
            self.window.close()

    def frame(self, dt: float) -> None:
        self.clock += dt
        busy = False
        for event in self.brain.poll():
            self.dispatch(event)
        self.finish_loading()
        width, height = self.window.logical_size()
        arena = Vec2(width, height)
        scale = self.window.scale()
        frame_input, self.frame_input = self.frame_input, Frame()
        locked = self.workout.locked
        c = self.character

        if frame_input.panic or (self.control.active and panic_pressed_anywhere()):
            self.control.panic(self.loco, self.brain.send)
        if locked:
            self.window.keep_in_front()
            busy = True  # mirror at the display's rate
        elif c is not None:
            self.pointer(frame_input, dt)
            self.loco.step(dt, arena, body_size(HEIGHT_PX))
            self.loco.wander(dt)
            c.facing = self.loco.facing()
            busy |= not self.loco.at_rest()
        if not locked:
            click = frame_input.presses[-1] if frame_input.presses else None
            self.chat.dismiss(frame_input.escape, (click.x, click.y) if click else None,
                              c.screen_rect if c else None, width, height)
            self.pictures(frame_input)
            busy |= self.chat.advance(dt, self.loco if c else None)
        self.control.update(self.loco if c else None)

        if c is not None:
            self.place(c, arena, dt)
            mirror = self.workout.pose if locked and self.workout.pose else None
            pose_body(dt, c.rig, self.animator, self.loco, c.facing, c.yaw, HEIGHT_PX, c.skeleton, mirror)
            animate_face(dt, c.rig, self.animator, self.loco, c.skeleton)
            c.skeleton.update()
            busy |= c.springs.simulate(dt)
            busy |= self.animator.busy()

        fb_w, fb_h = self.window.framebuffer()
        self.ctx.viewport = (0, 0, fb_w, fb_h)
        self.ctx.screen.use()
        if locked:
            self.ctx.clear(*BACKDROP, 1.0, depth=1.0)
        else:
            self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        if c is not None:
            self.renderer.draw(c.gpu, c.skeleton, View(width, height), self.clock)

        self.ui.begin(width, height, scale, dt)
        if not locked:
            source = None
            if c is not None and c.screen_rect is not None:
                r = c.screen_rect
                source = ((r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5)
            if self.chat.draw(self.ui, width, height, source, c.path if c else None, self.brain.send):
                self.running = False
        self.workout.draw(self.ui, width)
        self.control.draw_badge(width)
        self.ui.end()

        self.window.set_mouse_wanted(self.wants_mouse(width, height))
        self.window.finish_frame(busy or self.ui.wants_keyboard())

    def pictures(self, frame_input: Frame) -> None:
        """Pictures for the next message: dropped on her or the chat (which
        opens the chat), or pasted into the open chat."""
        if frame_input.dropped and self.chat.add_images(frame_input.dropped):
            if self.chat.phase == Phase.CLOSED and self.character is not None:
                self.chat.start_opening(self.loco)
        if frame_input.paste and self.chat.phase == Phase.OPEN:
            self.chat.paste()

    def pointer(self, frame_input: Frame, dt: float) -> None:
        """Press on her and move to pick her up — she hangs from where she was
        grabbed — and let go to throw her. A press that doesn't move is a
        click: it opens (or closes) the chat."""
        c = self.character
        cursor = Vec2(*self.window.cursor())
        for press in frame_input.presses:
            if inside(c.screen_rect, press) and not self.ui.wants_mouse():
                self.grab = Grab(press, False)
        if self.grab.pressed_at is None:
            return
        if glfw.get_mouse_button(self.window.handle, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            if not self.grab.dragging and cursor.distance(self.grab.pressed_at) > DRAG_THRESHOLD:
                self.grab.dragging = True
            if self.grab.dragging:
                self.loco.drag_to(cursor, dt)
        else:
            if self.grab.dragging:
                log.debug("thrown at %.0f px/s, spinning %.1f rad/s", self.loco.vel.length(), self.loco.spin)
                self.loco.release()
            else:
                self.chat.character_clicked(self.loco)
            self.grab = Grab()

    def place(self, c: Character, arena: Vec2, dt: float) -> None:
        """Put the model where the body is — feet at its feet, rolled by its
        angle, scaled to its on-screen height, turned toward where it's
        going, sunk as the knees give or it sits — and note its screen box,
        which is what the mouse can grab."""
        lo, hi = c.bounds
        model_height = max(float(hi[1] - lo[1]), 1e-3)
        if self.workout.locked:
            # A health round: the user's mirror, big in the middle, facing them,
            # and as low as their hips have come (a squat, a plank).
            height = arena.y * MIRROR_HEIGHT
            drop = (hip_drop(self.workout.pose) if self.workout.pose else None) or 0.0
            feet = Vec2(arena.x * 0.5, arena.y * 0.93 + drop * height * LEG_SHARE)
            c.yaw, c.facing, c.screen_rect = IDENTITY, 0.0, None
            s = height / model_height
            c.skeleton.root = compose(to_world(arena, feet), IDENTITY, (s, s, s))
            return
        turn = qaxis((0.0, 1.0, 0.0), c.facing * math.pi / 2)
        c.yaw = qslerp(c.yaw, turn, min(TURN_RATE * dt, 1.0))
        sink = (self.loco.limbs.crouch.x * CROUCH_DROP + self.loco.seat * SIT_DROP) * HEIGHT_PX
        feet = self.loco.feet(HEIGHT_PX) + Vec2(0.0, sink)
        s = HEIGHT_PX / model_height
        rotation = qmul(qaxis((0.0, 0.0, 1.0), self.loco.angle), c.yaw)
        c.skeleton.root = compose(to_world(arena, feet), rotation, (s, s, s))
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = corners @ c.skeleton.root.T
        xs = world[:, 0] + arena.x * 0.5
        ys = arena.y * 0.5 - world[:, 1]
        c.screen_rect = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    def wants_mouse(self, width: float, height: float) -> bool:
        """Whether the pointer is over something of Tomo's (her, the chat) —
        or she's held, or a round has the screen."""
        if self.workout.locked or self.grab.pressed_at is not None or self.ui.wants_mouse():
            return True
        x, y = self.window.cursor()
        c = self.character
        if c is not None and c.screen_rect is not None:
            r = c.screen_rect
            if r[0] - GRAB_MARGIN <= x <= r[2] + GRAB_MARGIN and r[1] - GRAB_MARGIN <= y <= r[3] + GRAB_MARGIN:
                return True
        if self.chat.shown():
            from .chat import panel_rect

            x0, y0, x1, y1 = panel_rect(width, height)
            return x0 <= x <= x1 and y0 <= y <= y1
        return False


def inside(rect, p: Vec2) -> bool:
    return rect is not None and rect[0] <= p.x <= rect[2] and rect[1] <= p.y <= rect[3]


def to_world(arena: Vec2, p: Vec2) -> tuple[float, float, float]:
    """A screen position → the world (the z = 0 plane, origin in the middle,
    +y up)."""
    return p.x - arena.x * 0.5, arena.y * 0.5 - p.y, 0.0
