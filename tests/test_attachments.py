"""Pictures sent in the chat: Tomo's copy, what the system knows about them,
and what the model is told — plus the one-line description of the computer."""

import base64
import io
from fractions import Fraction

import pytest
from PIL import Image
from PIL.ExifTags import IFD
from PIL.TiffImagePlugin import IFDRational

from tomo import attachments, sysinfo
from tomo.attachments import MAX_EDGE, encoded, jpeg_base64, notes, prepare
from tomo.events import Attachment


def rational(x):
    f = Fraction(x).limit_denominator(10000)
    return IFDRational(f.numerator, f.denominator)


def photo(path, size=(1600, 1200), orientation=None, gps=True):
    """A JPEG as a camera would write it: EXIF with the time, the camera,
    the exposure and where it was taken."""
    image = Image.new("RGB", size, (135, 206, 235))
    image.paste((30, 110, 200), (0, size[1] // 2, size[0], size[1]))
    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "Canon EOS R6"
    exif[0x0132] = "2025:07:14 16:20:00"
    if orientation is not None:
        exif[0x0112] = orientation
    sub = exif.get_ifd(IFD.Exif)
    sub[0x9003] = "2025:07:14 16:18:42"
    sub[0x829D] = rational(2.8)
    sub[0x829A] = IFDRational(1, 500)
    sub[0x8827] = 100
    sub[0x920A] = rational(35)
    if gps:
        where = exif.get_ifd(IFD.GPSInfo)
        where[1], where[2] = "N", (rational(42), rational(41), rational(51.72))
        where[3], where[4] = "E", (rational(23), rational(19), rational(18.84))
        where[5], where[6] = b"\x00", rational(560)
    image.save(path, "JPEG", quality=90, exif=exif)
    return path


def test_a_photo_is_copied_small_and_described(tmp_path):
    source = photo(tmp_path / "IMG_2041.jpg")
    got = prepare(source, tmp_path / "attachments")
    copy = Image.open(got.path)
    assert copy.format == "JPEG" and max(copy.size) == MAX_EDGE and copy.size == (MAX_EDGE, 960)
    assert got.path.startswith(str(tmp_path / "attachments")) and got.path.endswith("-IMG_2041.jpg")
    assert got.source == str(source.resolve())
    assert "file: " in got.about and "modified 20" in got.about
    assert "picture: JPEG 1600×1200" in got.about
    assert "taken 2025-07-14 16:18:42" in got.about
    assert "camera Canon EOS R6" in got.about, "the make isn't repeated"
    assert "f/2.8, 1/500 s, ISO 100, 35 mm" in got.about
    assert "GPS 42.697700, 23.321900, 560 m" in got.about
    assert source.exists(), "the user's own file is left alone"


def test_a_sideways_photo_is_turned_upright(tmp_path):
    # Orientation 6: the camera was held on its side; the pixels lie 200×100.
    got = prepare(photo(tmp_path / "side.jpg", size=(200, 100), orientation=6, gps=False), tmp_path / "out")
    assert Image.open(got.path).size == (100, 200)
    assert "GPS" not in got.about


def test_see_through_pictures_are_laid_on_white(tmp_path):
    Image.new("RGBA", (40, 30), (255, 0, 0, 0)).save(tmp_path / "clear.png")
    got = prepare(tmp_path / "clear.png", tmp_path / "out")
    assert Image.open(got.path).convert("RGB").getpixel((20, 15))[1] > 240, "white, not black"
    assert "picture: PNG 40×30" in got.about and "EXIF" not in got.about


def test_a_pasted_picture_has_no_file_and_its_temporary_copy_goes(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments.tempfile, "gettempdir", lambda: str(tmp_path))
    folder = attachments.pasted_dir()
    folder.mkdir()
    pasted = folder / "pasted-1.png"
    Image.new("RGB", (64, 48), (0, 128, 0)).save(pasted)
    got = prepare(pasted, tmp_path / "out")
    assert got.source == "" and not pasted.exists()
    assert "file:" not in got.about and "picture: PNG 64×48" in got.about
    assert "pasted from the clipboard" in notes([got])


def test_what_isnt_a_picture_is_refused(tmp_path):
    (tmp_path / "notes.png").write_text("not really a picture")
    with pytest.raises(ValueError):
        prepare(tmp_path / "notes.png", tmp_path / "out")
    with pytest.raises(OSError):
        prepare(tmp_path / "missing.jpg", tmp_path / "out")
    assert attachments.is_image("a.JPG") and attachments.is_image("b.webp") and not attachments.is_image("c.vrm")


def test_copies_never_overwrite_each_other(tmp_path):
    source = photo(tmp_path / "same.jpg", size=(64, 48), gps=False)
    first, second = prepare(source, tmp_path / "out"), prepare(source, tmp_path / "out")
    assert first.path != second.path


def test_the_notes_number_the_pictures_and_say_which_arent_shown():
    a = Attachment("a.jpg", r"C:\Pics\a.jpg", "picture: JPEG 10×10")
    b = Attachment("b.jpg", "", "picture: PNG 5×5")
    assert notes([a, b], 3) == ("[Image 3: C:\\Pics\\a.jpg]\npicture: JPEG 10×10\n"
                                "[Image 4: pasted from the clipboard, no file]\npicture: PNG 5×5")
    assert notes([a], shown=False).startswith("[Image 1: C:\\Pics\\a.jpg, sent earlier and not shown again]")


def test_pictures_reach_the_model_as_base64_jpeg(tmp_path):
    got = prepare(photo(tmp_path / "p.jpg", size=(300, 200), gps=False), tmp_path / "out")
    assert Image.open(io.BytesIO(base64.b64decode(encoded(got)))).format == "JPEG"
    assert encoded(Attachment(str(tmp_path / "gone.jpg"), "", "")) is None
    png = io.BytesIO()
    Image.new("RGBA", (2048, 1024), (10, 20, 30, 255)).save(png, "PNG")
    small = Image.open(io.BytesIO(base64.b64decode(jpeg_base64(png.getvalue(), 384))))
    assert small.format == "JPEG" and small.size == (384, 192)
    with pytest.raises(ValueError):
        jpeg_base64(b"not a picture", 384)


def test_the_computer_is_described_in_one_line():
    about = sysinfo.describe()
    assert about == sysinfo.describe()
    assert "\n" not in about
    assert about.split()[0] in ("Windows", "macOS") or "(Linux " in about
    assert sysinfo.architecture() in about
