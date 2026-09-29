"""What the desktop can do, read from the OS.

* **Apps** — every installed launcher. On Linux, parsed from the freedesktop
  ``.desktop`` files in the standard places (system, user, Flatpak); on
  Windows, the Start menu's shortcuts plus what ``Get-StartApps`` lists
  (Store apps like Calculator have no shortcut file).
* **Toggles** — which system switches THIS machine offers (Bluetooth, Wi-Fi,
  mute), decided by probing for the tools that drive them. Windows has none
  of those tools; there the model uses set_volume and PowerShell.

The whole catalogue is fingerprinted, so the brain can scan at start and
notice on a light re-scan when something was installed or removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import platform


@dataclass
class DesktopApp:
    id: str  # the .desktop file's stem, or the lower-case name on Windows
    name: str
    exec: str  # the launch command (field codes stripped), or Start-Process …
    icon: str = ""
    categories: list[str] = field(default_factory=list)
    terminal: bool = False


@dataclass
class Toggle:
    key: str  # bluetooth, wifi, mute
    label: str
    status_cmd: str
    on_cmd: str
    off_cmd: str


@dataclass
class SystemCatalog:
    apps: list[DesktopApp] = field(default_factory=list)
    toggles: list[Toggle] = field(default_factory=list)

    @classmethod
    def scan(cls) -> "SystemCatalog":
        """Scan the OS: its app launchers, and the toggles it has tools for.
        Takes about a second on Windows (Get-StartApps): run it off the loop."""
        if platform.WINDOWS:
            apps = start_menu_apps(start_menu_dirs())
            known = {a.id for a in apps}
            apps += [a for a in parse_start_apps(start_apps_json()) if a.id not in known]
            toggles: list[Toggle] = []
        else:
            apps = parse_all_entries(application_dirs())
            toggles = probe_toggles()
        apps.sort(key=lambda a: a.name.lower())
        return cls(apps, toggles)

    def fingerprint(self) -> str:
        """Equal for catalogues with the same apps and toggles."""
        h = hashlib.sha1()
        for a in self.apps:
            h.update(f"{a.id}\0{a.exec}\0{a.name}\0".encode())
        for t in self.toggles:
            h.update(t.key.encode())
        return h.hexdigest()

    def to_json(self) -> str:
        return json.dumps({"apps": [asdict(a) for a in self.apps], "toggles": [asdict(t) for t in self.toggles]})

    @classmethod
    def from_json(cls, text: str) -> "SystemCatalog":
        data = json.loads(text)
        return cls([DesktopApp(**a) for a in data.get("apps", [])], [Toggle(**t) for t in data.get("toggles", [])])

    def find_app(self, query: str) -> list[DesktopApp]:
        """Matches by name or id, best first: exact, then prefix, then contains."""
        q = query.strip().lower()
        if not q:
            return []
        scored = []
        for a in self.apps:
            name, id_ = a.name.lower(), a.id.lower()
            if q in (name, id_):
                scored.append((0, a))
            elif name.startswith(q) or id_.startswith(q):
                scored.append((1, a))
            elif q in name or q in id_:
                scored.append((2, a))
        scored.sort(key=lambda s: s[0])
        return [a for _, a in scored]

    def closest(self, query: str, n: int) -> list[DesktopApp]:
        """The ``n`` apps whose names are nearest ``query`` (fewest letters to
        change), for "no app called that — did you mean…"."""
        q = query.strip().lower()
        return sorted(self.apps, key=lambda a: edit_distance(q, a.name.lower()))[:n]

    def toggles_summary(self) -> str | None:
        """The system toggles, for the model's context (None without any)."""
        if not self.toggles:
            return None
        return "Available system toggles: " + ", ".join(t.key for t in self.toggles)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the fewest single-letter edits from a to b."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        diagonal, row[0] = row[0], i + 1
        for j, cb in enumerate(b):
            above = row[j + 1]
            row[j + 1] = min(above + 1, row[j] + 1, diagonal + (ca != cb))
            diagonal = above
    return row[len(b)]


# ---- Linux: freedesktop .desktop files ------------------------------------------


def application_dirs() -> list[Path]:
    """The folders that hold .desktop files, the user's first so their
    override wins over the system copy."""
    home = os.environ.get("HOME", "")
    data_home = os.environ.get("XDG_DATA_HOME") or f"{home}/.local/share"
    dirs = [Path(data_home) / "applications", Path(home) / ".local/share/flatpak/exports/share/applications"]
    for base in (os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":"):
        if base:
            dirs.append(Path(base) / "applications")
    dirs.append(Path("/var/lib/flatpak/exports/share/applications"))
    return dirs


def parse_all_entries(dirs: list[Path]) -> list[DesktopApp]:
    seen: dict[str, DesktopApp] = {}  # by id: the first (highest-priority) copy wins
    for folder in dirs:
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            continue
        for path in entries:
            if path.suffix != ".desktop":
                continue
            try:
                app = parse_entry(path.read_text(encoding="utf-8", errors="replace"), path)
            except OSError:
                continue
            if app is not None:
                seen.setdefault(app.id, app)
    return list(seen.values())


def parse_entry(text: str, path: Path) -> DesktopApp | None:
    """A .desktop file's [Desktop Entry] group. None for anything a launcher
    wouldn't show (hidden, no-display, or not an Application)."""
    in_group = False
    kv: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_group = line == "[Desktop Entry]"
            continue
        if not in_group or not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if "[" in key:  # localised keys like Name[de]
            continue
        kv[key] = value.strip()
    if kv.get("Type") != "Application":
        return None
    if _is_true(kv.get("NoDisplay")) or _is_true(kv.get("Hidden")):
        return None
    name = kv.get("Name")
    exec_ = clean_exec(kv.get("Exec", ""))
    if not name or not exec_:
        return None
    return DesktopApp(
        id=path.stem or name,
        name=name,
        exec=exec_,
        icon=kv.get("Icon", ""),
        categories=[c for c in kv.get("Categories", "").split(";") if c],
        terminal=_is_true(kv.get("Terminal")),
    )


def clean_exec(exec_: str) -> str:
    """Strip freedesktop field codes (%U, %f…) and collapse whitespace."""
    out = []
    i = 0
    while i < len(exec_):
        c = exec_[i]
        if c == "%":
            if i + 1 < len(exec_) and exec_[i + 1] == "%":
                out.append("%")
            i += 2
            continue
        out.append(c)
        i += 1
    return " ".join("".join(out).split())


def _is_true(value: str | None) -> bool:
    return (value or "").lower() == "true"


def probe_toggles() -> list[Toggle]:
    """The toggles for whatever control tools are installed."""
    out = []
    if platform.which("rfkill"):
        out.append(Toggle("bluetooth", "Bluetooth", "rfkill list bluetooth", "rfkill unblock bluetooth", "rfkill block bluetooth"))
    elif platform.which("bluetoothctl"):
        out.append(Toggle("bluetooth", "Bluetooth", "bluetoothctl show", "bluetoothctl power on", "bluetoothctl power off"))
    if platform.which("nmcli"):
        out.append(Toggle("wifi", "Wi-Fi", "nmcli radio wifi", "nmcli radio wifi on", "nmcli radio wifi off"))
    if platform.which("wpctl"):
        out.append(Toggle("mute", "Mute", "wpctl get-volume @DEFAULT_AUDIO_SINK@",
                          "wpctl set-mute @DEFAULT_AUDIO_SINK@ 1", "wpctl set-mute @DEFAULT_AUDIO_SINK@ 0"))
    elif platform.which("pactl"):
        out.append(Toggle("mute", "Mute", "pactl get-sink-mute @DEFAULT_SINK@",
                          "pactl set-sink-mute @DEFAULT_SINK@ 1", "pactl set-sink-mute @DEFAULT_SINK@ 0"))
    return out


# ---- Windows: the Start menu --------------------------------------------------


def start_menu_dirs() -> list[Path]:
    """The Start menu's shortcuts: for everyone, and for this user."""
    dirs = []
    for var in ("ProgramData", "APPDATA"):
        if base := os.environ.get(var):
            dirs.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def start_menu_apps(dirs: list[Path]) -> list[DesktopApp]:
    """The apps behind the Start menu's shortcuts (.lnk), minus uninstallers.
    Opening a shortcut starts the app, so that's the launch command."""
    found: list[Path] = []

    def walk(folder: Path, depth: int) -> None:
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            return
        for path in entries:
            if path.is_dir() and depth < 4:
                walk(path, depth + 1)
            elif path.suffix.lower() == ".lnk":
                found.append(path)

    for folder in dirs:
        walk(folder, 0)
    seen: set[str] = set()
    apps = []
    for path in found:
        name = path.stem
        lower = name.lower()
        if "uninstall" in lower or lower in seen:
            continue
        seen.add(lower)
        quoted = str(path).replace("'", "''")
        apps.append(DesktopApp(id=lower, name=name, exec=f"Start-Process -FilePath '{quoted}'"))
    return apps


def start_apps_json() -> str:
    """What the Start menu lists (``Get-StartApps``), as JSON; empty if it
    can't be had. Takes ~0.8 s."""
    if not platform.WINDOWS:
        return ""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "[Console]::OutputEncoding = [Text.Encoding]::UTF8; Get-StartApps | ConvertTo-Json -Compress"],
            capture_output=True, timeout=30, creationflags=platform.quiet_flags(), stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.decode("utf-8", "replace")


def parse_start_apps(text: str) -> list[DesktopApp]:
    """Apps from Get-StartApps' JSON (a list, or one object for one app),
    opened by their AppID through ``shell:AppsFolder``."""
    try:
        value = json.loads(text.strip()) if text.strip() else []
    except json.JSONDecodeError:
        return []
    entries = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
    apps = []
    for entry in entries:
        name = str(entry.get("Name") or "").strip()
        app_id = str(entry.get("AppID") or "").strip()
        if not name or not app_id or "uninstall" in name.lower():
            continue
        quoted = app_id.replace("'", "''")
        apps.append(DesktopApp(id=name.lower(), name=name, exec=f"Start-Process 'shell:AppsFolder\\{quoted}'"))
    return apps
