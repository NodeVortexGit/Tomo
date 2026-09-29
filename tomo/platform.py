"""What differs between the systems Tomo runs on: which shell runs commands,
finding a program, and keeping helper processes from flashing a console
window on Windows."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"

# Put before every PowerShell line. Windows PowerShell 5.1 writes its output
# in the console's code page and reads BOM-less files as ANSI, so names like
# "Ayanokōji" or a README's "—" would reach the model garbled: make both
# UTF-8, which is how the output is read back.
POWERSHELL_UTF8 = (
    "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
    "$OutputEncoding = [Text.Encoding]::UTF8; $PSDefaultParameterValues['*:Encoding'] = 'utf8'; "
)


def os_name() -> str:
    """The system's name, for the model and the logs."""
    return "Windows" if WINDOWS else "macOS" if MACOS else "Linux"


def shell_name() -> str:
    """The shell ``execute_command`` lines run in."""
    return "PowerShell" if WINDOWS else "bash"


def shell_argv(line: str) -> list[str]:
    """The command that runs ``line`` in the system's shell: PowerShell on
    Windows, else a bash login shell (so the user's own PATH applies)."""
    if WINDOWS:
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-Command", POWERSHELL_UTF8 + line]
    return ["bash", "-lc", line]


def quiet_flags() -> int:
    """Creation flags that keep a helper process from opening a console
    window (Windows); nothing to do elsewhere."""
    return subprocess.CREATE_NO_WINDOW if WINDOWS else 0  # type: ignore[attr-defined]


def which(program: str) -> str | None:
    """Where ``program`` is, if it's on the PATH (with Windows' extensions)."""
    return shutil.which(program)


def helper_python() -> str:
    """The Python that runs the helper scripts: the one running Tomo, which
    has their packages (the installers put everything in one environment)."""
    exe = sys.executable
    # pythonw.exe (no console) runs the app; helpers talk over pipes, so the
    # console Python beside it is the right one.
    if WINDOWS and exe.lower().endswith("pythonw.exe"):
        candidate = exe[: -len("pythonw.exe")] + "python.exe"
        if os.path.exists(candidate):
            return candidate
    return exe
