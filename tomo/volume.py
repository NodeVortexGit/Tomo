"""The speaker volume, for the ``set_volume`` tool.

Left to guess, a model reaches for volume commands that don't exist —
Windows has no volume cmdlet at all — and burns a minute a try. So this
builds a command line that works, for the executor to run (and audit) like
any other: Core Audio through a few lines of C# on Windows, WirePlumber or
PulseAudio on Linux. Every line reports the volume after, as
``volume=40 muted=false``.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import platform


@dataclass(frozen=True)
class Change:
    """What to do. All None just reports the current volume."""

    level: int | None = None  # set it to this, 0–100
    by: int | None = None  # or turn it up (positive) or down by this much
    mute: bool | None = None


def command(change: Change) -> str | None:
    """The command line for ``change`` on this system, or None where there's
    no way to set the volume (Linux without wpctl or pactl)."""
    change = clamped(change)
    if platform.WINDOWS:
        return windows(change)
    if platform.which("wpctl"):
        return wireplumber(change)
    if platform.which("pactl"):
        return pulseaudio(change)
    return None


def clamped(change: Change) -> Change:
    return Change(
        level=None if change.level is None else max(0, min(100, change.level)),
        by=None if change.by is None else max(-100, min(100, change.by)),
        mute=change.mute,
    )


# Core Audio's endpoint volume for the default speakers, as a PowerShell type
# (compiled in ~0.2 s). One line, so it goes through -Command as it is.
WINDOWS_AUDIO = (
    "Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; "
    '[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] '
    "interface IAudioEndpointVolume { int _0(); int _1(); int _2(); int _3(); "
    "int SetMasterVolumeLevelScalar(float level, Guid context); int _5(); "
    "int GetMasterVolumeLevelScalar(out float level); int _7(); int _8(); int _9(); int _10(); "
    "int SetMute([MarshalAs(UnmanagedType.Bool)] bool mute, Guid context); int GetMute(out bool mute); } "
    '[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] '
    "interface IMMDevice { int Activate(ref Guid id, int context, IntPtr parameters, "
    "[MarshalAs(UnmanagedType.IUnknown)] out object endpoint); } "
    '[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] '
    "interface IMMDeviceEnumerator { int _0(); int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice device); } "
    '[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator { } '
    "public static class TomoVolume { "
    "static IAudioEndpointVolume Endpoint() { var devices = (IMMDeviceEnumerator)new MMDeviceEnumerator(); IMMDevice device; "
    "Marshal.ThrowExceptionForHR(devices.GetDefaultAudioEndpoint(0, 1, out device)); Guid id = typeof(IAudioEndpointVolume).GUID; "
    "object endpoint; Marshal.ThrowExceptionForHR(device.Activate(ref id, 23, IntPtr.Zero, out endpoint)); return (IAudioEndpointVolume)endpoint; } "
    "public static int Level { get { float v; Marshal.ThrowExceptionForHR(Endpoint().GetMasterVolumeLevelScalar(out v)); return (int)Math.Round(v * 100); } "
    "set { Marshal.ThrowExceptionForHR(Endpoint().SetMasterVolumeLevelScalar(Math.Max(0, Math.Min(100, value)) / 100f, Guid.Empty)); } } "
    "public static bool Muted { get { bool m; Marshal.ThrowExceptionForHR(Endpoint().GetMute(out m)); return m; } "
    "set { Marshal.ThrowExceptionForHR(Endpoint().SetMute(value, Guid.Empty)); } } }'"
)


def windows(change: Change) -> str:
    line = WINDOWS_AUDIO + "; "
    if change.level is not None:
        line += f"[TomoVolume]::Level = {change.level}; "
    if change.by is not None:
        line += f"[TomoVolume]::Level = [TomoVolume]::Level + ({change.by}); "
    if change.mute is not None:
        line += f"[TomoVolume]::Muted = ${str(change.mute).lower()}; "
    line += '"volume=$([TomoVolume]::Level) muted=$(([TomoVolume]::Muted).ToString().ToLower())"'
    return line


def wireplumber(change: Change) -> str:
    sink = "@DEFAULT_AUDIO_SINK@"
    steps = []
    if change.level is not None:
        steps.append(f"wpctl set-volume {sink} {change.level}%")
    if change.by is not None:
        steps.append(f"wpctl set-volume -l 1.0 {sink} {change.by}%+" if change.by >= 0
                     else f"wpctl set-volume {sink} {-change.by}%-")
    if change.mute is not None:
        steps.append(f"wpctl set-mute {sink} {int(change.mute)}")
    # "Volume: 0.40 [MUTED]" -> "volume=40 muted=true"
    steps.append(f"wpctl get-volume {sink} | awk '{{printf \"volume=%d muted=%s\\n\", $2*100+0.5, "
                 f"($3==\"[MUTED]\")?\"true\":\"false\"}}'")
    return " && ".join(steps)


def pulseaudio(change: Change) -> str:
    sink = "@DEFAULT_SINK@"
    steps = []
    if change.level is not None:
        steps.append(f"pactl set-sink-volume {sink} {change.level}%")
    if change.by is not None:
        steps.append(f"pactl set-sink-volume {sink} {change.by:+d}%")
    if change.mute is not None:
        steps.append(f"pactl set-sink-mute {sink} {int(change.mute)}")
    steps.append(f"echo \"volume=$(pactl get-sink-volume {sink} | grep -o '[0-9]*%' | head -1 | tr -d %) "
                 f"muted=$(pactl get-sink-mute {sink} | grep -q yes && echo true || echo false)\"")
    return " && ".join(steps)
