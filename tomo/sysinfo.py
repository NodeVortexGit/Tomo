"""What the computer is — the system and its version, the machine, who's
using it — in one line for the model. It goes into the persona (ai.py), so
the model knows where it is without running commands to find out, and picks
commands that fit this system."""

from __future__ import annotations

import functools
import getpass
import os
import platform as py_platform

from . import platform

# Windows' edition ids, as its settings name them.
EDITIONS = {
    "Core": "Home", "CoreN": "Home N", "CoreSingleLanguage": "Home Single Language",
    "CoreCountrySpecific": "Home", "Professional": "Pro", "ProfessionalN": "Pro N",
    "ProfessionalWorkstation": "Pro for Workstations", "Education": "Education", "Enterprise": "Enterprise",
    "IoTEnterprise": "IoT Enterprise", "ServerStandard": "Server Standard", "ServerDatacenter": "Server Datacenter",
}
ARCHITECTURES = {"amd64": "x64", "x86_64": "x64", "arm64": "ARM64", "aarch64": "ARM64", "x86": "x86", "i686": "x86"}


@functools.cache
def describe() -> str:
    """E.g. "Windows 11 Home 24H2 (build 26100.9457, x64), computer
    “DESKTOP-1”, user “Ana”". Read once: none of it changes while Tomo runs."""
    system = windows() if platform.WINDOWS else macos() if platform.MACOS else linux()
    about = [system]
    if node := py_platform.node():
        about.append(f"computer “{node}”")
    try:
        about.append(f"user “{getpass.getuser()}”")
    except (OSError, KeyError, ImportError):  # no login name (a service, a container)
        pass
    return ", ".join(about)


def architecture() -> str:
    machine = py_platform.machine()
    return ARCHITECTURES.get(machine.lower(), machine or "unknown")


def windows() -> str:
    # platform.release() says "11" on Windows 11 (Python 3.12+); the registry
    # has the edition and the feature update ("24H2").
    release, version = py_platform.release(), py_platform.version()
    edition = EDITIONS.get(py_platform.win32_edition() or "", py_platform.win32_edition() or "")
    feature = build = ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            feature = str(_value(key, "DisplayVersion") or "")
            current, ubr = _value(key, "CurrentBuild"), _value(key, "UBR")
            build = f"{current}.{ubr}" if current and ubr is not None else str(current or "")
    except OSError:
        pass
    name = " ".join(part for part in ("Windows", release, edition, feature) if part)
    return f"{name} (build {build or version}, {architecture()})"


def _value(key, name: str):
    import winreg

    try:
        return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


def linux() -> str:
    try:
        name = py_platform.freedesktop_os_release().get("PRETTY_NAME") or "Linux"
    except OSError:
        name = "Linux"
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").replace(":", "/")
    session = {"x11": "X11", "wayland": "Wayland"}.get(os.environ.get("XDG_SESSION_TYPE", "").lower(), "")
    where = f"; {desktop} on {session}" if desktop and session else f"; {desktop or session}" if (
        desktop or session) else ""
    return f"{name} (Linux {py_platform.release()}, {architecture()}{where})"


def macos() -> str:
    return f"macOS {py_platform.mac_ver()[0] or ''}".strip() + f" ({architecture()})"
