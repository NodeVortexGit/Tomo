"""What the desktop can do, read from the OS.

* **Apps** — every installed app, not only what the Start menu shows. On
  Linux, parsed from the freedesktop ``.desktop`` files in the standard
  places (system, user, Flatpak, Snap); on Windows, in this order: the Start
  menu's shortcuts, what ``Get-StartApps`` lists (the apps folder: Store apps
  like Calculator have no shortcut file), the programs Windows lists as
  installed (the registry's Uninstall list — Steam games too), the programs
  registered to start by name (App Paths), and the apps in the Program Files
  folders that none of those mention.
* **Toggles** — which system switches THIS machine offers (Bluetooth, Wi-Fi,
  mute), decided by probing for the tools that drive them. Windows has none
  of those tools; there the model uses set_volume and PowerShell.
* **Opening an app, checked** (Windows): :func:`checked_launch` starts it,
  then waits until it's really running and says so — the model answers
  from that, not from a hopeful "no error".

The whole catalogue is fingerprinted, so the brain can scan at start and
notice on a light re-scan when something was installed or removed.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import platform

# Where a Windows app was found, the most telling first: when two match a
# name as well, the earlier wins (what the Start menu shows is what's meant).
SOURCES = ("start menu", "apps folder", "installed", "registered", "program files")


@dataclass
class DesktopApp:
    id: str  # the .desktop file's stem, or the lower-case name on Windows
    name: str
    exec: str  # the launch command (field codes stripped), or Start-Process …
    icon: str = ""
    categories: list[str] = field(default_factory=list)
    terminal: bool = False
    source: str = ""  # Windows: one of SOURCES
    program: str = ""  # the program that runs while it's open, where exec doesn't say (a Steam game's)


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
        """Scan the OS: its apps, and the toggles it has tools for. Takes a
        second or two on Windows (Get-StartApps, the registry, Program
        Files; less again, as what it read from the programs is kept): run it
        off the loop."""
        if platform.WINDOWS:
            apps = start_menu_apps(start_menu_dirs())
            for more in (parse_start_apps(start_apps_json()), installed_apps(registry_entries(UNINSTALL_KEY)),
                         registered_apps(registry_entries(APP_PATHS_KEY)), program_files_apps(program_files_dirs())):
                apps = merged(apps, more)
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
        """Matches by name or id, best first: exact, then prefix, then
        contains; among equals, by where they were found (SOURCES)."""
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
        scored.sort(key=lambda s: (s[0], SOURCES.index(s[1].source) if s[1].source in SOURCES else 0))
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
    dirs.append(Path("/var/lib/snapd/desktop/applications"))
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
        apps.append(DesktopApp(id=lower, name=name, exec=f"Start-Process -FilePath '{quoted}'", source="start menu"))
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
        apps.append(DesktopApp(id=name.lower(), name=name, exec=f"Start-Process 'shell:AppsFolder\\{quoted}'",
                               source="apps folder"))
    return apps


# ---- Windows: everything else that's installed --------------------------------------------

UNINSTALL_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
APP_PATHS_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
# What isn't an app to open, though it's a program or an installed package:
# uninstallers, updaters, helpers, services, runtimes, drivers, SDKs… Matched
# against file names, the programs' own descriptions and installed names.
NOT_AN_APP = re.compile(
    r"unins|setup|install|update|patch|crash|report|helper|service|svc|daemon|agent|broker|elevat|notif|"
    r"telemetry|diag|debug|proxy|stub|squirrel|redist|repair|regist|feedback|sandbox|watchdog|[_-]monitor|tray|"
    r"(?<![a-z])test(?![a-z])|shim|headless|webview|container|click-?to-?run|surrogate|handler|converter|"
    r"component|server|host\b|protocol|import|config|runtime|driver|\bsdk\b|framework|librar|toolchain|"
    r"targeting pack|manifest|intellisense|^python|^pip\b|^java|^node$|^git\b|^sh$|^bash$", re.I)
# Folders that hold no app of their own: shared parts, runtimes, an app's innards.
NOT_APP_FOLDERS = {
    "common files", "windowsapps", "modifiablewindowsapps", "windows defender", "windows nt", "windows mail",
    "windows photo viewer", "windows portable devices", "windows security", "windows sidebar", "internet explorer",
    "reference assemblies", "msbuild", "windows kits", "microsoft sdks", "dotnet", "package cache", "installer",
    "uninstall", "updater", "update", "updates", "redist", "locales", "resources", "node_modules", "plugins",
    "share", "lib", "libexec", "mingw32", "mingw64", "usr", "drivers", "jre", "runtime", "steamapps", "old",
    "backup", "temp", "tmp", "shared",
}
# Descriptions that don't name the app.
GENERIC_NAMES = {"application", "launcher", "electron", "app", "main", "program"}
# Folders inside an app's that say nothing of which app it is.
PLAIN_FOLDERS = {"bin", "bin64", "bin32", "app", "application", "program", "programs", "client", "current", "latest",
                 "release", "x64", "x86", "64bit", "32bit", "win64", "win32"}
# Left out when comparing a program's name with its folder's.
FILLER = re.compile(r"\b(?:x64|x86|win32|win64|64bit|32bit|bit|app|desktop|portable)\b|[^a-z]")
WINDOWED = 2  # a Windows program's subsystem: a window of its own (3: a console tool)


def merged(apps: list[DesktopApp], more: list[DesktopApp]) -> list[DesktopApp]:
    """``apps``, and the ones from ``more`` it doesn't have yet — by name,
    or by the program it starts (an app found in two places is listed once)."""
    names = {a.id for a in apps} | {a.name.lower() for a in apps}
    programs = {p for a in apps if (p := launched_program(a.exec))}
    out = list(apps)
    for app in more:
        program = launched_program(app.exec)
        if app.id in names or app.name.lower() in names or (program and program in programs):
            continue
        out.append(app)
        names |= {app.id, app.name.lower()}
        if program:
            programs.add(program)
    return out


def launched_program(exec_: str) -> str | None:
    """What a launch line starts, to know one app found twice: the .exe (a
    shortcut's target), or the Steam game, lower-cased."""
    if steam := re.search(r"steam://rungameid/(\d+)", exec_):
        return f"steam://rungameid/{steam.group(1)}"
    m = re.fullmatch(r"Start-Process -FilePath '(.*)'", exec_)
    if not m:
        return None
    path = m.group(1).replace("''", "'")
    if path.lower().endswith(".lnk"):
        path = shortcut_target(Path(path)) or ""
    return path.lower() if path.lower().endswith(".exe") else None


def shortcut_target(path: Path) -> str | None:
    """The file a .lnk shortcut opens, from its link info; None when it
    names none (an installer's "advertised" shortcuts don't)."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 0x4C or int.from_bytes(data[:4], "little") != 0x4C:
        return None
    flags = int.from_bytes(data[0x14:0x18], "little")
    at = 0x4C
    if flags & 0x1:  # a list of shell items comes first
        at += 2 + int.from_bytes(data[at:at + 2], "little")
    info = data[at:]
    if not flags & 0x2 or len(info) < 0x1C or not int.from_bytes(info[8:12], "little") & 0x1:
        return None  # no link info, or no local path in it

    def text(offset: int, wide: bool) -> str:
        if wide:
            end = offset
            while end + 1 < len(info) and info[end:end + 2] != b"\0\0":
                end += 2
            return info[offset:end].decode("utf-16-le", "replace")
        end = info.find(b"\0", offset)
        return info[offset:end if end >= 0 else None].decode("mbcs" if platform.WINDOWS else "latin-1", "replace")

    if int.from_bytes(info[4:8], "little") >= 0x24 and (base := int.from_bytes(info[0x1C:0x20], "little")):
        suffix = int.from_bytes(info[0x20:0x24], "little")
        target = text(base, True) + (text(suffix, True) if suffix else "")
    else:
        target = text(int.from_bytes(info[0x10:0x14], "little"), False) + text(
            int.from_bytes(info[0x18:0x1C], "little"), False)
    return target or None


def registry_entries(key: str) -> list[tuple[str, dict]]:
    """The subkeys of ``key`` (under SOFTWARE) with their values — the
    machine's, in its 64- and 32-bit views, then the user's — as
    [(subkey, {value name: value})]. Empty but on Windows."""
    if not platform.WINDOWS:
        return []
    import winreg

    out = []
    places = [(winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY), (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
              (winreg.HKEY_CURRENT_USER, 0)]
    for hive, view in places:
        try:
            root = winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view)
        except OSError:
            continue
        with root:
            for i in itertools.count():
                try:
                    name = winreg.EnumKey(root, i)
                except OSError:
                    break
                values = {}
                try:
                    with winreg.OpenKey(root, name) as sub:
                        for j in itertools.count():
                            try:
                                value_name, value, _ = winreg.EnumValue(sub, j)
                            except OSError:
                                break
                            values[value_name] = value
                except OSError:
                    continue
                out.append((name, values))
    return out


def installed_apps(entries: list[tuple[str, dict]]) -> list[DesktopApp]:
    """The programs Windows lists as installed (Settings → Apps), each by
    the program that opens it: its icon's, else the app's own one in its
    folder. A Steam game opens through Steam. Leaves out system components,
    updates, runtimes and whatever has no program of its own."""
    apps = []
    for key, values in entries:
        name = clean_name(str(values.get("DisplayName") or ""))
        if not name or values.get("SystemComponent") == 1 or values.get("ParentKeyName") or NOT_AN_APP.search(name):
            continue
        if steam := re.fullmatch(r"Steam App (\d+)", key):
            folder = install_folder(values)
            exe = main_program(folder, [name, folder.name]) if folder else None
            apps.append(DesktopApp(id=name.lower(), name=name, exec=f"Start-Process 'steam://rungameid/{steam.group(1)}'",
                                   source="installed", program=exe.stem if exe else ""))
            continue
        exe = existing_exe(values.get("DisplayIcon"))
        if exe is None or NOT_AN_APP.search(exe.stem) or in_system_folders(exe) or not windowed(exe):
            folder = install_folder(values)
            exe = main_program(folder, [name, folder.name]) if folder else None
        if exe is not None:
            apps.append(exe_app(name, exe, "installed"))
    return apps


def registered_apps(entries: list[tuple[str, dict]]) -> list[DesktopApp]:
    """The programs registered to start by name ("App Paths": chrome.exe,
    winword.exe…) — leaving out Windows' own and the Store's, which the
    Start menu has, and what isn't an app."""
    apps = []
    for _, values in entries:
        exe = existing_exe(values.get(""))
        if exe is None or in_system_folders(exe) or NOT_AN_APP.search(exe.stem) or not windowed(exe):
            continue
        if name := program_name(exe):
            apps.append(exe_app(name, exe, "registered"))
    return apps


def program_files_dirs() -> list[Path]:
    """Where programs are installed: Program Files (both) and the user's own."""
    dirs: list[Path] = []
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        if (base := os.environ.get(var)) and Path(base) not in dirs:
            dirs.append(Path(base))
    if base := os.environ.get("LOCALAPPDATA"):
        dirs.append(Path(base) / "Programs")
    return dirs


def program_files_apps(dirs: list[Path]) -> list[DesktopApp]:
    """The apps in the Program Files folders, whatever lists them or not:
    each app folder's own windowed program — named like the folder, as in
    Geekbench 6\\Geekbench 6.exe — or like a folder one down, for a vendor's
    (Oracle\\VirtualBox\\VirtualBox.exe; not like the vendor: its helpers
    are named so too). Below a folder like bin or 1.2.3, still like the
    app's (obs-studio\\bin\\64bit\\obs64.exe)."""
    apps, seen = [], set()
    for root in dirs:
        for top in app_folders(root):
            places = [(top, [top.name], 1)] + [
                (sub, [top.name if plain_folder(sub.name) else sub.name], 2) for sub in app_folders(top)]
            for folder, names, depth in places:
                exe = main_program(folder, names, depth)
                if exe is None or str(exe).lower() in seen:
                    continue
                seen.add(str(exe).lower())
                if name := program_name(exe):
                    apps.append(exe_app(name, exe, "program files"))
    return apps


def plain_folder(name: str) -> bool:
    """A folder named for what's in it, not for an app: bin, x64, 1.2.3…"""
    return name.lower() in PLAIN_FOLDERS or len(FILLER.sub("", name.lower())) < 3


def app_folders(folder: Path) -> list[Path]:
    try:
        return sorted(Path(e.path) for e in os.scandir(folder)
                      if e.is_dir(follow_symlinks=False) and e.name.lower() not in NOT_APP_FOLDERS)
    except OSError:
        return []


def main_program(folder: Path, names: list[str], depth: int = 2) -> Path | None:
    """The app's own windowed program in ``folder``, or up to ``depth``
    folders below: named like the app or its folder, and not an uninstaller,
    updater or helper — the one nearest the top, then the shortest-named."""
    best: tuple[tuple[int, int], Path] | None = None
    pending = [(folder, 0)]
    while pending:
        current, level = pending.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if level < depth and entry.name.lower() not in NOT_APP_FOLDERS:
                        pending.append((Path(entry.path), level + 1))
                    continue
            except OSError:
                continue
            stem, _, ext = entry.name.rpartition(".")
            if ext.lower() != "exe" or NOT_AN_APP.search(stem) or not resembles(stem, names):
                continue
            rank = (level, len(stem))
            if (best is None or rank < best[0]) and windowed(Path(entry.path)):
                best = (rank, Path(entry.path))
    return best[1] if best else None


def resembles(stem: str, names: list[str]) -> bool:
    """Whether a program's file name is its app folder's: one holds the
    other once both are cut to their letters, less filler like x64 or
    "app" — a short one only at the start or as a whole word: obs64 is
    obs-studio's, Code Microsoft VS Code's, "ollama app" Ollama's."""
    mine = FILLER.sub("", stem.lower())
    for name in names:
        theirs = FILLER.sub("", name.lower())
        (short, _), (long, longer) = sorted(((mine, stem), (theirs, name)), key=lambda p: len(p[0]))
        if len(short) < 3:
            continue
        if short in long if len(short) >= 5 else (
                long.startswith(short) or short in re.split(r"[^a-z0-9]+", longer.lower())):
            return True
    return False


def clean_name(text: str) -> str:
    """An app's name without its version or build: "7-Zip 26.01 (x64)" is
    7-Zip, "Ollama version 0.35.1" Ollama."""
    name = re.sub(r"\s*\((?:x64|x86|arm64|64-bit|32-bit|user|remove only)\)", "", text, flags=re.I)
    name = re.sub(r"\s+(?:version\s+)?v?\d+(?:\.\d+)+[\w.+-]*$", "", name, flags=re.I)
    name = " ".join(name.split()).strip(" -")
    return name or " ".join(text.split())


def program_name(exe: Path) -> str | None:
    """What a program calls itself — its description, else its product's
    name, else its file name — tidied; None when its description says it's
    a part of something (a component, a handler…), not an app."""
    info = version_strings(exe)
    description = info.get("FileDescription", "")
    if NOT_AN_APP.search(description):
        return None
    for value in (description, info.get("ProductName", "")):
        if value and value.lower() not in GENERIC_NAMES and len(value) <= 40:
            return clean_name(value)
    return clean_name(exe.stem)


def exe_app(name: str, exe: Path, source: str) -> DesktopApp:
    quoted = str(exe).replace("'", "''")
    return DesktopApp(id=name.lower(), name=name, exec=f"Start-Process -FilePath '{quoted}'", source=source)


def existing_exe(value) -> Path | None:
    """The .exe a registry value names — "C:\\…\\app.exe,0", quoted, with
    %variables% — if it's there."""
    text = os.path.expandvars(str(value or "")).strip()
    m = re.match(r'"([^"]+\.exe)"|([^",]+\.exe)', text, re.I)
    if not m:
        return None
    path = Path(m.group(1) or m.group(2))
    return path if path.is_file() else None


def install_folder(values: dict) -> Path | None:
    """An installed program's folder: where the list says, else its
    uninstaller's — never a whole drive, Program Files itself or Windows'."""
    location = str(values.get("InstallLocation") or "").strip().strip('"')
    folder = Path(location) if location else None
    if folder is None or not folder.is_dir():
        uninstaller = existing_exe(values.get("UninstallString"))
        folder = uninstaller.parent if uninstaller is not None else None
    if folder is None or not folder.is_dir() or folder.parent == folder or in_system_folders(folder) \
            or folder in program_files_dirs() or folder == Path.home():
        return None
    return folder


def in_system_folders(path: Path) -> bool:
    """Inside Windows' own folder, the Store's apps or an installer's cache."""
    lower = str(path).lower()
    system = (os.environ.get("SystemRoot") or "").lower().rstrip("\\/")
    return (bool(system) and (lower == system or lower.startswith(system + os.sep))) or any(
        part in lower for part in ("\\windowsapps\\", "\\package cache\\", "/windowsapps/", "/package cache/"))


def windowed(path: Path) -> bool:
    """Whether a program opens a window of its own (not a console tool) —
    read from its header, and remembered while the file stays the same."""
    try:
        stat = path.stat()
    except OSError:
        return False
    return _windowed(str(path), stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=8192)
def _windowed(path: str, size: int, mtime: int) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
            offset = int.from_bytes(head[60:64], "little") if len(head) >= 64 else 0
            if len(head) < offset + 96 <= 1 << 20:
                f.seek(0)
                head = f.read(offset + 96)
    except OSError:
        return False
    return pe_subsystem(head) == WINDOWED


def pe_subsystem(head: bytes) -> int | None:
    """A Windows program's subsystem, from the start of its file (2: it has
    windows, 3: a console tool); None if it isn't a Windows program."""
    if len(head) < 64 or head[:2] != b"MZ":
        return None
    offset = int.from_bytes(head[60:64], "little")
    if head[offset:offset + 4] != b"PE\0\0" or len(head) < offset + 94:
        return None
    return int.from_bytes(head[offset + 92:offset + 94], "little")


def version_strings(path: Path) -> dict[str, str]:
    """A Windows program's FileDescription and ProductName, as it states
    them ({} elsewhere, or when it states none); remembered per file."""
    if not platform.WINDOWS:
        return {}
    try:
        stat = path.stat()
    except OSError:
        return {}
    return dict(_version_strings(str(path), stat.st_size, stat.st_mtime_ns))


@functools.lru_cache(maxsize=8192)
def _version_strings(path: str, size: int, mtime: int) -> tuple[tuple[str, str], ...]:
    import ctypes
    from ctypes import wintypes

    api = _version_api()
    length = api.GetFileVersionInfoSizeW(path, None)
    if not length:
        return ()
    data = ctypes.create_string_buffer(length)
    if not api.GetFileVersionInfoW(path, 0, length, data):
        return ()
    pointer, size_out = ctypes.c_void_p(), wintypes.UINT()
    languages = []
    if api.VerQueryValueW(data, "\\VarFileInfo\\Translation", ctypes.byref(pointer), ctypes.byref(size_out)) \
            and size_out.value >= 4:
        words = (wintypes.WORD * (size_out.value // 2)).from_address(pointer.value)
        languages = [f"{words[i]:04x}{words[i + 1]:04x}" for i in range(0, len(words) - 1, 2)]
    found = {}
    for key in ("FileDescription", "ProductName"):
        for language in languages + ["040904b0", "040904e4", "000004b0"]:
            if api.VerQueryValueW(data, f"\\StringFileInfo\\{language}\\{key}", ctypes.byref(pointer),
                                  ctypes.byref(size_out)) and size_out.value and pointer.value:
                value = ctypes.wstring_at(pointer.value).strip()  # up to its end, whatever length it claims
                if value and value.isprintable():
                    found[key] = value
                    break
    return tuple(found.items())


@functools.cache
def _version_api():
    import ctypes
    from ctypes import wintypes

    api = ctypes.WinDLL("version")
    api.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
    api.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    api.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    api.GetFileVersionInfoW.restype = wintypes.BOOL
    api.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(wintypes.UINT)]
    api.VerQueryValueW.restype = wintypes.BOOL
    return api


# Opening an app: how long to wait for it to be running before saying it
# can't confirm it, how often to look, and how long it must keep running —
# a program that exits (or crashes) at once was never really open. A Steam
# game gets longer (Steam may have to start first), within the commands' 20 s.
LAUNCH_WAIT_SECS, LAUNCH_LOOK_MS, LAUNCH_STAYS_MS = 8, 250, 1000
GAME_WAIT_SECS = 15


def checked_launch(exec_: str, program: str = "") -> str:
    """Windows: the PowerShell line that starts an app (a catalogue entry's
    ``Start-Process …``), then makes sure it's running, printing ``OPEN: …``
    or ``NOT CONFIRMED: …``.

    "Running" means the app's own program has a process, still there a
    second later: ``program`` if given (a Steam game's), else the program a
    shortcut points to or the line starts, or the one a Store app's manifest
    names (a Store app's window belongs to a host process, so its window
    can't tell). If none can be worked out, a window that has opened since
    the launch counts. An app that was open already counts as open."""
    wait = GAME_WAIT_SECS if "steam://" in exec_ else LAUNCH_WAIT_SECS
    program = "'" + program.replace("'", "''") + "'" if program else program_of(exec_)
    return (
        f"$started = Get-Date; $program = @({program})[0]; {exec_}; "
        "$look = { Get-Process -ErrorAction SilentlyContinue | Where-Object { try { if ($program) { "
        "$_.ProcessName -eq $program } else { $_.MainWindowHandle -ne 0 -and $_.StartTime -ge $started } } "
        "catch { $false } } | Sort-Object { $_.MainWindowHandle -eq 0 } | Select-Object -First 1 }; "
        "do { $found = & $look; if ($found) { "
        f"Start-Sleep -Milliseconds {LAUNCH_STAYS_MS}; $found = & $look; if ($found) {{ break }} }} "
        f"else {{ Start-Sleep -Milliseconds {LAUNCH_LOOK_MS} }} }} "
        f"while ((Get-Date) -lt $started.AddSeconds({wait})); "
        "if ($found) { 'OPEN: ' + $found.ProcessName + ' is running' + "
        "$(if ($found.MainWindowTitle) { ', window: ' + $found.MainWindowTitle } else { '' }) } "
        f"else {{ 'NOT CONFIRMED: it did not open (its program is not running {wait} s after the start)' }}"
    )


def program_of(exec_: str) -> str:
    """The PowerShell expression for the program a launch line starts: a
    shortcut's target, the .exe itself, or a Store app's from its manifest;
    '' when the line doesn't say."""
    if m := re.fullmatch(r"Start-Process -FilePath '(.*)'", exec_):
        path = m.group(1)  # quoted for PowerShell already ('' for ')
        target = (f"(New-Object -ComObject WScript.Shell).CreateShortcut('{path}').TargetPath"
                  if path.lower().endswith(".lnk") else f"'{path}'")
        return f"[IO.Path]::GetFileNameWithoutExtension({target})"
    if m := re.fullmatch(r"Start-Process 'shell:AppsFolder\\(.*)'", exec_):
        app_id = m.group(1)
        if "!" in app_id:  # a Store app: PackageFamilyName!App
            family, app = app_id.split("!", 1)
            return (f"$(foreach ($p in @(Get-AppxPackage -Name '{family.rsplit('_', 1)[0]}' "
                    f"-ErrorAction SilentlyContinue | Where-Object PackageFamilyName -eq '{family}')) "
                    "{ ([xml](Get-Content -LiteralPath (Join-Path $p.InstallLocation 'AppxManifest.xml') -Raw))"
                    f".Package.Applications.Application | Where-Object Id -eq '{app}' | "
                    "ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_.Executable) } })")
        if app_id.lower().endswith(".exe"):  # e.g. {folder id}\…\notepad.exe
            return f"[IO.Path]::GetFileNameWithoutExtension('{app_id}')"
    return "''"
