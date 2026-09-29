"""Tiny .vrm files for tests: just the header, the metadata and a thumbnail —
enough for what the brain reads (characters.vrm_facts)."""

import io
import json
import struct

from PIL import Image


def png(color=(200, 80, 80), size=(64, 64)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, "PNG")
    return out.getvalue()


def tiny_vrm(path, name="Ayako", authors=("Someone",), thumbnail: bytes | None = None, version=1):
    """A glTF binary with VRM metadata (1.0, or 0.x with ``version=0``) and,
    if given, a thumbnail picture in its binary chunk."""
    gltf = {"asset": {"version": "2.0"}}
    blob = b""
    image_index = None
    if thumbnail is not None:
        blob = thumbnail + b"\0" * (-len(thumbnail) % 4)
        gltf["buffers"] = [{"byteLength": len(blob)}]
        gltf["bufferViews"] = [{"buffer": 0, "byteOffset": 0, "byteLength": len(thumbnail)}]
        gltf["images"] = [{"bufferView": 0, "mimeType": "image/png"}]
        image_index = 0
    if version == 1:
        meta = {"name": name, "authors": list(authors)}
        if image_index is not None:
            meta["thumbnailImage"] = image_index
        gltf["extensions"] = {"VRMC_vrm": {"specVersion": "1.0", "meta": meta}}
    else:
        meta = {"title": name, "author": authors[0] if authors else ""}
        if image_index is not None:
            gltf["textures"] = [{"source": image_index}]
            meta["texture"] = 0
        gltf["extensions"] = {"VRM": {"meta": meta}}
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text
    if blob:
        chunks += struct.pack("<I4s", len(blob), b"BIN\0") + blob
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)
    return path
