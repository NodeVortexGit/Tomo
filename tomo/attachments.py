"""Images the user sends in the chat, and what the system knows about them.

The chat takes images three ways: its image button, dropping files on the
chat (or on Tomo), and pasting — a screenshot, or image files copied in the
file manager. For each one the brain:

1. keeps its own copy in ``<data>/attachments``: turned upright (by its EXIF
   orientation), at most ``MAX_EDGE`` px on its long side, as JPEG. The copy
   is what the chat shows and what the model sees, and it stays when the
   original is moved or deleted, so the conversation still makes sense
   after a restart;
2. reads what the system knows about the file: where it is, its size, when
   it was created and changed, the picture's format and size, and its EXIF —
   when it was taken, with what, where (GPS);
3. sends the model the picture with those notes, beside the user's words.
   The notes are context: the model uses them when they help ("when was this
   taken?", "put it on my desktop") and otherwise leaves them be.

Which system the computer runs (sysinfo.py) is in the persona, every turn.
"""

from __future__ import annotations

import base64
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .events import Attachment

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
MAX_EDGE = 1280  # px: text in a screenshot stays readable, and the model is quick
JPEG_QUALITY = 85
MAX_PER_MESSAGE = 4
# The folder (in the temp folder) where the body keeps a pasted picture until
# the brain has made its copy.
PASTED = "tomo-pasted"
LONGEST_TEXT = 160  # characters of an embedded description or comment


def is_image(path) -> bool:
    """Whether a file is a picture, by its extension."""
    return Path(path).suffix.lower() in IMAGE_TYPES


def pasted_dir() -> Path:
    return Path(tempfile.gettempdir()) / PASTED


def was_pasted(path) -> bool:
    return Path(path).parent.name == PASTED


def prepare(source: Path, folder: Path) -> Attachment:
    """Tomo's copy of the picture at ``source`` (in ``folder``) and the notes
    on it. Raises OSError (no such file, unreadable) or ValueError (not a
    picture)."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    source = Path(source)
    stat = source.stat()
    pasted = was_pasted(source)
    try:
        with Image.open(source) as image:
            about = describe(image, None if pasted else stat)
            picture = flatten(ImageOps.exif_transpose(image))
    except (UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ValueError(f"{source.name} isn't a picture I can open") from e
    picture.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{safe_name(source.stem)}"
    copy = folder / f"{stem}.jpg"
    n = 1
    while copy.exists():
        n += 1
        copy = folder / f"{stem}-{n}.jpg"
    picture.save(copy, "JPEG", quality=JPEG_QUALITY)
    if pasted:
        try:
            source.unlink()  # Tomo's own temporary file
        except OSError:
            pass
    return Attachment(str(copy), "" if pasted else str(source.resolve()), about)


def flatten(image):
    """RGB, with anything see-through laid on white (a transparent PNG would
    turn black otherwise)."""
    from PIL import Image

    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def safe_name(stem: str) -> str:
    """A file name's stem, safe on every system (letters of any script kept)."""
    return re.sub(r"[^\w.-]+", "-", stem).strip("-.")[:40] or "image"


def encoded(attachment: Attachment) -> str | None:
    """Tomo's copy of the picture, base64, for the model (None if it's gone)."""
    try:
        return base64.b64encode(Path(attachment.path).read_bytes()).decode("ascii")
    except OSError:
        return None


def jpeg_base64(data: bytes, edge: int) -> str:
    """Any picture's bytes → a JPEG at most ``edge`` px on its long side,
    base64 — for the model. Raises ValueError if it isn't a picture."""
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            picture = flatten(image)
    except UnidentifiedImageError as e:
        raise ValueError("not a picture") from e
    picture.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    picture.save(out, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(out.getvalue()).decode("ascii")


def notes(attachments, first: int = 1, shown: bool = True) -> str:
    """The notes the model reads beside the pictures, numbered through the
    conversation. ``shown`` False: the pictures aren't sent again (an older
    message's), which the model is told."""
    again = "" if shown else ", sent earlier and not shown again"
    return "\n".join(f"[Image {i}: {a.source or 'pasted from the clipboard, no file'}{again}]\n{a.about}"
                     for i, a in enumerate(attachments, first))


# ---- what the system knows about a picture ----------------------------------------------------


def describe(image, stat) -> str:
    """The file's details (``stat`` None: it has no file of its own) and the
    picture's: format, size, EXIF, embedded text. A few short lines."""
    lines = []
    if stat is not None:
        dates = []
        if (born := getattr(stat, "st_birthtime", None)) is not None:
            dates.append(f"created {stamp(born)}")
        dates.append(f"modified {stamp(stat.st_mtime)}")
        lines.append(f"file: {size(stat.st_size)}, " + ", ".join(dates))
    picture = f"{image.format or 'image'} {image.width}×{image.height}"
    if (frames := getattr(image, "n_frames", 1)) > 1:
        picture += f", {frames} frames"
    if isinstance(dpi := image.info.get("dpi"), tuple) and dpi and dpi[0]:
        picture += f", {round(float(dpi[0]))} dpi"
    lines.append(f"picture: {picture}")
    if facts := exif_facts(image):
        lines.append("EXIF: " + "; ".join(facts))
    if texts := embedded_text(image):
        lines.append("embedded text: " + "; ".join(texts))
    return "\n".join(lines)


def exif_facts(image) -> list[str]:
    """When it was taken, with what and how, where — as far as the EXIF says."""
    from PIL.ExifTags import IFD

    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - a broken EXIF block is just no EXIF
        return []
    if not exif:
        return []
    main = dict(exif)
    try:
        sub = dict(exif.get_ifd(IFD.Exif))
    except Exception:  # noqa: BLE001
        sub = {}
    try:
        gps_ifd = dict(exif.get_ifd(IFD.GPSInfo))
    except Exception:  # noqa: BLE001
        gps_ifd = {}
    facts = []
    if taken := text(sub.get(0x9003) or main.get(0x0132)):  # DateTimeOriginal, else DateTime
        facts.append(f"taken {exif_date(taken)}")
    make, model = text(main.get(0x010F)), text(main.get(0x0110))
    camera = model if make and model.lower().startswith(make.lower()) else " ".join(p for p in (make, model) if p)
    if camera:
        facts.append(f"camera {camera}")
    if lens := text(sub.get(0xA434)):
        facts.append(f"lens {lens}")
    exposure = []
    if (f := number(sub.get(0x829D))) is not None:
        exposure.append(f"f/{f:.1f}")
    if (t := number(sub.get(0x829A))) is not None and t > 0:
        exposure.append(f"1/{round(1 / t)} s" if t < 1 else f"{t:g} s")
    if (iso := number(sub.get(0x8827))) is not None:
        exposure.append(f"ISO {iso:g}")
    if (focal := number(sub.get(0x920A))) is not None:
        exposure.append(f"{focal:g} mm")
    if exposure:
        facts.append(", ".join(exposure))
    if (where := gps(gps_ifd)) is not None:
        facts.append(where)
    for tag, label in ((0x010E, "description"), (0x013B, "artist"), (0x8298, "copyright"), (0x0131, "software")):
        if value := text(main.get(tag)):
            facts.append(f"{label} {value}")
    return facts


def gps(ifd: dict) -> str | None:
    """"GPS 42.697700, 23.321900, 560 m" from the GPS block, if it has a position."""
    lat, lon = degrees(ifd.get(2), ifd.get(1)), degrees(ifd.get(4), ifd.get(3))
    if lat is None or lon is None:
        return None
    out = f"GPS {lat:.6f}, {lon:.6f}"
    if (alt := number(ifd.get(6))) is not None:
        below = ifd.get(5) in (1, b"\x01")
        out += f", {-alt if below else alt:.0f} m"
    return out


def degrees(dms, ref) -> float | None:
    """Degrees, minutes, seconds (and N/S/E/W) → signed decimal degrees."""
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    value = d + m / 60 + s / 3600
    return -value if text(ref).upper() in ("S", "W") else value


def embedded_text(image) -> list[str]:
    """Text a picture carries: a PNG's title, description, comment, author,
    the software that made it, an AI image's prompt; a JPEG's comment."""
    found = []
    for key in ("Title", "Description", "Comment", "comment", "Author", "Software", "Source", "parameters", "prompt"):
        if value := text(image.info.get(key)):
            found.append(f"{key.lower()}: {value}")
    return found


def text(value) -> str:
    """An EXIF or PNG value as one short, clean line ("" for nothing)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    value = " ".join(str(value).replace("\x00", " ").split())
    return value if len(value) <= LONGEST_TEXT else value[:LONGEST_TEXT] + "…"


def number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def exif_date(value: str) -> str:
    """EXIF writes dates as "2026:09:20 10:11:05"."""
    return re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", value)


def stamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).strftime("%Y-%m-%d %H:%M")


def size(n: int) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"
