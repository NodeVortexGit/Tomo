import sys
from pathlib import Path

import pytest

# The tests import the package from the source tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mouse(monkeypatch):
    """The real cursor, clicks and keys, recorded instead (tomo.hands); glides at once."""
    from tomo import hands

    record = {"at": (0.0, 0.0), "moves": [], "clicks": 0, "typed": [], "keys": []}

    def move(x, y):
        record["at"] = (x, y)
        record["moves"].append((round(x), round(y)))

    monkeypatch.setattr(hands, "cursor_pos", lambda: record["at"])
    monkeypatch.setattr(hands, "move_cursor", move)
    monkeypatch.setattr(hands, "click", lambda times=1: record.__setitem__("clicks", record["clicks"] + times))
    monkeypatch.setattr(hands, "type_text", record["typed"].append)
    monkeypatch.setattr(hands, "press", record["keys"].append)
    monkeypatch.setattr(hands, "can_move", lambda: True)
    monkeypatch.setattr(hands, "glide_seconds", lambda distance: 0.0)
    monkeypatch.setattr(hands, "REACH_SECS", 0.0)
    monkeypatch.setattr(hands, "TAKE_SECS", 0.0)
    monkeypatch.setattr(hands, "CLICK_PAUSE_SECS", 0.0)
    return record
