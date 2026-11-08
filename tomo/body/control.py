"""Her side of using the mouse — the "mouse in disguise".

The brain moves the real cursor (tomo/hands.py); here she plays her part
(:class:`~tomo.events.Carry`): she stops and raises her arms for the
cursor ("reach") and says where her hands are once she's ready; she hangs
from it wherever it goes ("hold"); she lets go and drops ("drop").

Consent and safety come first:

* she can be watched the whole time — she carries the cursor herself —
  and her window lets every click through meanwhile;
* the **panic key** — Pause, or Ctrl+Alt+Esc — releases control at once and
  tells the brain to stop (on Windows it works from anywhere while she's in
  control, even when another window has the keyboard).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .. import events, platform
from ..events import PanicStop
from .mathx import Vec2
from .movement import Locomotion, Stance

log = logging.getLogger(__name__)

VK_PAUSE, VK_ESCAPE, VK_CONTROL, VK_MENU = 0x13, 0x1B, 0x11, 0x12
# Her raised hands, above her centre of mass, as a share of her height.
HANDS_UP = 0.55
RAISE_SECS = 0.3  # arms up before the cursor comes


@dataclass
class Control:
    active: bool = False  # she's driving input
    carrying: str = ""  # "reach", "hold", or "" when she isn't
    reached_for: float = 0.0  # seconds her arms have been up for it

    def receive(self, event, loco: Locomotion | None) -> None:
        if isinstance(event, events.ControlMode):
            self.active = event.on
        elif isinstance(event, events.Carry):
            if event.stage == "reach":
                self.carrying, self.reached_for = "reach", 0.0
                if loco is not None:
                    loco.held = True  # no wandering off mid-task
                    loco.target = None
                    loco.stand_up()
            elif event.stage == "hold":
                self.carrying = "hold"
            else:
                self.let_go(loco)

    def update(self, loco: Locomotion | None, cursor: Vec2, dt: float) -> None:
        """Holding the cursor, she goes where it goes."""
        if self.carrying == "reach":
            self.reached_for += dt
        elif self.carrying == "hold" and loco is not None:
            loco.drag_to(cursor, dt)

    def hands(self, loco: Locomotion | None) -> Vec2 | None:
        """Where her raised hands are (logical px) once she's standing ready
        for the cursor; else None."""
        if self.carrying != "reach" or loco is None or self.reached_for < RAISE_SECS:
            return None
        if loco.stance != Stance.STANDING or loco.seat > 0.05 or loco.target is not None:
            return None
        return loco.com + Vec2(0.0, -HANDS_UP * loco.height)

    def let_go(self, loco: Locomotion | None) -> None:
        if loco is not None:
            if self.carrying == "hold":
                loco.release()  # she drops with whatever swing she had
            if self.carrying:
                loco.held = False
        self.carrying = ""

    def panic(self, loco: Locomotion | None, send: Callable) -> None:
        """Release control at once, and tell the brain to stop."""
        self.let_go(loco)
        self.active = False
        send(PanicStop())
        log.warning("panic key: mouse/keyboard control released")


def panic_pressed_anywhere() -> bool:
    """Windows: the panic key, pressed wherever the keyboard is."""
    if not platform.WINDOWS:
        return False
    import ctypes

    state = ctypes.windll.user32.GetAsyncKeyState
    pause = state(VK_PAUSE) & 1
    combo = state(VK_ESCAPE) & 0x8000 and state(VK_CONTROL) & 0x8000 and state(VK_MENU) & 0x8000
    return bool(pause or combo)
