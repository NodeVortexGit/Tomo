"""The safe command executor.

The assistant runs shell commands on the machine (volume, launching apps,
checking things) *quietly* — the user watches the character, not a terminal.
That is a UX choice, not a stealth feature, so this module keeps one firm
principle: the output is hidden from the chat, but NOTHING is hidden from
the machine's owner. Every command — run, refused or failed — is appended to
a plain-text audit log with a timestamp.

Layers of protection, in order:
  1. a master switch that turns the whole capability off (log-only);
  2. a hard deny-list of irreversible, catastrophic patterns, refused even
     with the switch on;
  3. a timeout, so a hung command can't freeze the assistant.

Commands run through ``bash -lc`` on Linux and PowerShell on Windows
(platform.py).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from . import platform
from .events import now_ms

log = logging.getLogger(__name__)

COMMAND_TIMEOUT = 20.0  # seconds
# Enough output for the AI to understand what happened.
MAX_OUTPUT_BYTES = 16 * 1024


@dataclass
class CommandOutcome:
    command: str
    allowed: bool
    blocked_reason: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def summary(self) -> str:
        """What the AI reads back as the tool result."""
        if not self.allowed:
            # Said outright: a model has answered "I've opened them" after
            # every command was refused.
            return (f"REFUSED: {self.blocked_reason or 'not permitted'}. "
                    "Nothing was done: tell the user it did not happen.")
        if self.timed_out:
            return f"TIMEOUT after {int(COMMAND_TIMEOUT)}s"
        code = "?" if self.exit_code is None else str(self.exit_code)
        out = f"exit={code}"
        if self.stdout.strip():
            out += f"\nstdout:\n{self.stdout.strip()}"
        if self.stderr.strip():
            out += f"\nstderr:\n{self.stderr.strip()}"
        # Left to themselves, local models give up after one failure (or make
        # a result up); said right after the error, it gets them to fix it.
        if self.exit_code != 0:
            out += ("\n\nThe command failed. Work out why from the error and run a corrected or "
                    "different command; don't ask the user first. After at most two other "
                    "tries, say plainly what went wrong.")
        elif not self.stdout.strip() and not self.stderr.strip():
            out += "\n(no output)"
        return out


class Executor:
    def __init__(self, enabled: bool, audit_log_path: Path, extra_allowed: list[str] | None = None) -> None:
        self.enabled = enabled
        self.audit_log_path = Path(audit_log_path)
        # Reserved for explicit allow-listing; the hard deny-list is what
        # actually guards the machine.
        self.extra_allowed = extra_allowed or []

    async def run(self, command: str) -> CommandOutcome:
        """Run a command line and return the outcome. Never raises."""
        command = command.strip()
        if not self.enabled:
            outcome = CommandOutcome(command, False, "command execution is disabled in config")
        elif (reason := hard_denied(command)) is not None:
            outcome = CommandOutcome(command, False, reason)
        else:
            outcome = await self._spawn(command)
        self._audit(outcome)
        return outcome

    async def _spawn(self, command: str) -> CommandOutcome:
        try:
            process = await asyncio.create_subprocess_exec(
                *platform.shell_argv(command),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=platform.quiet_flags(),
            )
        except OSError as e:
            return CommandOutcome(command, False, f"failed to start {platform.shell_name()}: {e}")
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), COMMAND_TIMEOUT)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return CommandOutcome(command, True, timed_out=True)
        return CommandOutcome(
            command,
            True,
            exit_code=process.returncode,
            stdout=_truncate(stdout.decode("utf-8", "replace")),
            stderr=_truncate(stderr.decode("utf-8", "replace")),
        )

    def _audit(self, outcome: CommandOutcome) -> None:
        """One line per command. Best-effort: a logging failure must never
        take the assistant down."""
        # All of it, within reason: what a command does can come after a long
        # preamble (set_volume's Windows line is ~2300 characters).
        text = outcome.command.replace("\n", " ")[:8000]
        line = f"{now_ms()}\t{'RUN' if outcome.allowed else 'REFUSED'}\t{text}\n"
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            log.warning("could not write the command audit log: %s", e)


def _truncate(text: str) -> str:
    if len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
        return text.encode("utf-8")[:MAX_OUTPUT_BYTES].decode("utf-8", "ignore") + "\n…[truncated]"
    return text


def hard_denied(command: str) -> str | None:
    """Why ``command`` is refused whatever the settings, or None.

    Deliberately about *irreversible* damage (wiping disks, fork bombs, piping
    the internet into a shell). Power commands like a reboot are recoverable
    device control and are not here. This is a safety net, not a sandbox —
    which is why every command is also in the audit log."""
    c = " ".join(command.lower().split())

    # Wiping the root filesystem.
    for pattern in ("rm -rf /", "rm -fr /", "rm -r -f /", "rm --recursive --force /"):
        if pattern in c and "rm -rf /tmp" not in c and "rm -rf /home/" not in c:
            after = c.split(pattern, 1)[1]
            if after == "" or after.startswith(" ") or after.startswith("*"):
                return "refuses to recursively delete the root filesystem"
    # Formatting / raw-writing block devices.
    if "mkfs" in c:
        return "refuses to format a filesystem"
    if "dd " in c and "of=/dev/" in c:
        return "refuses to raw-write to a block device with dd"
    for dev in ("> /dev/sd", ">/dev/sd", "> /dev/nvme", ">/dev/nvme", "of=/dev/sd", "of=/dev/nvme"):
        if dev in c:
            return "refuses to write directly to a disk device"
    # The classic fork bomb.
    if ":(){:|:&};:" in c.replace(" ", ""):
        return "refuses to run a fork bomb"
    # Piping a download straight into a shell = arbitrary remote code.
    if ("curl " in c or "wget " in c) and any(p in c for p in ("| sh", "|sh", "| bash", "|bash")):
        return "refuses to pipe a network download directly into a shell"
    # Stripping permissions from the whole tree.
    if "chmod -r 000 /" in c or "chmod 000 -r /" in c or ("chown -r" in c and c.endswith(" /")):
        return "refuses to recursively strip permissions from the root tree"
    if "mv " in c and " /dev/null" in c and " ~" in c:
        return "refuses to move the home directory into /dev/null"
    return _windows_denied(c)


_DESTROYERS = ("format-volume", "clear-disk", "initialize-disk", "remove-partition", "diskpart", "bcdedit",
               "vssadmin delete", "wmic shadowcopy delete", "cipher /w")
_SYSTEM_FOLDERS = ("c:\\windows", "$env:systemroot", "$env:windir", "c:\\users", "c:\\program files")


def _windows_denied(c: str) -> str | None:
    """The same for Windows: formatting or wiping drives, partitions and the
    boot setup, deleting a whole drive or system folder, erasing backups, and
    running a download straight away."""
    for tool in _DESTROYERS:
        if tool in c:
            return f"refuses to run {tool}: it can destroy disks, partitions or backups"
    words = [w for w in re.split(r"[ ;|&]", c) if w]

    def is_drive(w: str) -> bool:
        w = w.strip("\"'")
        return len(w) >= 2 and w[0].isascii() and w[0].isalpha() and w[1] == ":" and w[2:].strip("\\/*") == ""

    # Whatever switches come before the drive: "format -f C:", "format.com /q D:".
    if any(w.strip("\"'").rsplit("\\", 1)[-1] in ("format", "format.com", "format.exe")
           and any(is_drive(after) for after in words[i + 1:]) for i, w in enumerate(words)):
        return "refuses to format a drive"

    def precious(w: str) -> bool:
        w = w.strip("\"'").rstrip("\\/*")
        return is_drive(w) or w in _SYSTEM_FOLDERS

    deletes = any(w in ("rd", "rmdir", "del", "erase", "rm", "ri", "remove-item") for w in words)
    recursive = any(w == "/s" or w.startswith("-r") for w in words)
    if deletes and recursive and any(precious(w) for w in words):
        return "refuses to delete a whole drive or system folder"
    if "reg delete hklm" in c or "remove-item hklm:" in c:
        return "refuses to delete machine-wide registry keys"
    runs = any(w == "iex" or w.startswith("iex(") or "invoke-expression" in w for w in words)
    downloads = any(
        w in ("iwr", "irm", "curl", "wget") or any(d in w for d in ("invoke-webrequest", "invoke-restmethod", "downloadstring"))
        for w in words
    )
    if runs and downloads:
        return "refuses to run a download straight away"
    return None
