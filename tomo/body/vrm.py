"""Reading a ``.vrm`` file.

A VRM is a binary glTF (``.glb``) — meshes, a skeleton, materials and
textures — whose JSON adds the VRM extensions: which node is which humanoid
bone (``VRMC_vrm``), the facial expressions, the spring bones
(``VRMC_springBone``, read by :mod:`.springs`) and the toon shading
(``VRMC_materials_mtoon``, read by :mod:`.mtoon`).

:func:`load` turns the file into plain arrays, ready for the GPU; it touches
no graphics API, so it runs on a worker thread while the window keeps
drawing. Only VRM 1.0 files are rigged and animated (their bone axes are what
:mod:`.animation` expects); a VRM 0.x file still shows, standing still.
"""

from __future__ import annotations

import io
import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
NORMALIZED = {np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0}
TRIANGLES = 4


@dataclass
class Node:
    name: str
    parent: int  # -1: a root
    children: list[int]
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    mesh: int | None
    skin: int | None


@dataclass
class Vertices:
    """A mesh's vertices. The primitives of a mesh (one per material) mostly
    share one set, each drawing its own triangles of it."""

    positions: np.ndarray  # (n, 3) float32
    normals: np.ndarray  # (n, 3) float32
    uvs: np.ndarray  # (n, 2) float32
    joints: np.ndarray | None  # (n, 4) float32 — joint indices, as floats for the GPU
    weights: np.ndarray | None  # (n, 4) float32
    # Blend shapes: each a change of position (and of normal) per vertex.
    targets: list[tuple[np.ndarray, np.ndarray | None]] = field(default_factory=list)


@dataclass
class Primitive:
    vertices: Vertices
    indices: np.ndarray  # (m,) uint32
    material: int | None


@dataclass
class Mesh:
    name: str
    primitives: list[Primitive]
    weights: list[float]  # the blend shapes' starting weights


@dataclass
class Skin:
    joints: list[int]  # node indices
    inverse_bind: np.ndarray  # (joints, 4, 4)


@dataclass
class Sampler:
    mag: int = 9729  # LINEAR
    min: int = 9987  # LINEAR_MIPMAP_LINEAR
    wrap_s: int = 10497  # REPEAT
    wrap_t: int = 10497


@dataclass
class Texture:
    image: int
    sampler: Sampler


@dataclass
class Material:
    name: str
    base_color: tuple[float, float, float, float]
    base_texture: int | None  # texture index
    # KHR_texture_transform of the base colour texture — VRoid gives every
    # texture of a material the same one.
    uv_offset: tuple[float, float]
    uv_scale: tuple[float, float]
    uv_rotation: float
    emissive: tuple[float, float, float]
    emissive_texture: int | None
    normal_texture: int | None
    normal_scale: float
    alpha_mode: str  # OPAQUE, MASK or BLEND
    alpha_cutoff: float
    double_sided: bool
    mtoon: dict | None  # its VRMC_materials_mtoon, as written


@dataclass
class Vrm:
    path: Path
    json: dict
    nodes: list[Node]
    roots: list[int]
    meshes: list[Mesh]
    skins: list[Skin]
    materials: list[Material]
    textures: list[Texture]
    images: list[bytes]  # encoded (PNG/JPEG); see :func:`decode_images`
    version: str  # "1.0", "0.x" or "" (plain glTF)
    humanoid: dict[str, int]  # VRM bone name → node index
    expressions: dict[str, list[tuple[int, int, float]]]  # preset → (node, blend shape, weight)

    def node_named(self, name: str) -> int | None:
        return next((i for i, n in enumerate(self.nodes) if n.name == name), None)


# ---- the file ------------------------------------------------------------------------------


def read_glb(data: bytes) -> tuple[dict, memoryview]:
    """A binary glTF's JSON and its binary chunk."""
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("not a binary glTF (.vrm/.glb) file")
    total = struct.unpack_from("<I", data, 8)[0]
    view = memoryview(data)
    offset, gltf, blob = 12, None, memoryview(b"")
    while offset + 8 <= min(total, len(data)):
        size, kind = struct.unpack_from("<I4s", data, offset)
        chunk = view[offset + 8:offset + 8 + size]
        if kind == b"JSON":
            gltf = json.loads(bytes(chunk).decode("utf-8"))
        elif kind == b"BIN\x00":
            blob = chunk
        offset += 8 + size
    if gltf is None:
        raise ValueError("the file has no glTF JSON")
    return gltf, blob


def gltf_json(path: Path) -> dict | None:
    """Only the JSON of a .vrm (its first chunk): enough for the specs."""
    try:
        with open(path, "rb") as f:
            header = f.read(20)
            if len(header) < 20 or header[:4] != b"glTF" or header[16:20] != b"JSON":
                return None
            size = struct.unpack_from("<I", header, 12)[0]
            return json.loads(f.read(size).decode("utf-8"))
    except (OSError, ValueError):
        return None


def accessor(gltf: dict, blob: memoryview, index: int) -> np.ndarray:
    """An accessor's data: (count, width) — or (count, 4, 4) row-major for
    matrices — with normalized integers turned into floats."""
    a = gltf["accessors"][index]
    dtype = np.dtype(COMPONENT[a["componentType"]])
    width, count = WIDTH[a["type"]], a["count"]
    if "bufferView" in a:
        view = gltf["bufferViews"][a["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise ValueError("external buffers aren't supported")
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride") or dtype.itemsize * width
        out = np.ndarray((count, width), dtype=dtype, buffer=blob, offset=start,
                         strides=(stride, dtype.itemsize)).copy()
    else:
        out = np.zeros((count, width), dtype=dtype)
    if sparse := a.get("sparse"):
        n = sparse["count"]
        ind, val = sparse["indices"], sparse["values"]
        ind_view, val_view = gltf["bufferViews"][ind["bufferView"]], gltf["bufferViews"][val["bufferView"]]
        where = np.frombuffer(blob, COMPONENT[ind["componentType"]], n,
                              ind_view.get("byteOffset", 0) + ind.get("byteOffset", 0))
        values = np.frombuffer(blob, dtype, n * width, val_view.get("byteOffset", 0) + val.get("byteOffset", 0))
        out[where.astype(np.int64)] = values.reshape(n, width)
    if a.get("normalized") and dtype.type in NORMALIZED:
        out = np.maximum(out.astype(np.float32) / NORMALIZED[dtype.type], -1.0)
    if a["type"] == "MAT4":
        return out.reshape(count, 4, 4).transpose(0, 2, 1)  # glTF stores columns
    return out


def load(path: Path) -> Vrm:
    """Read a .vrm (or .glb) file."""
    path = Path(path)
    gltf, blob = read_glb(path.read_bytes())
    get = lambda i: accessor(gltf, blob, i)  # noqa: E731

    nodes = []
    for n in gltf.get("nodes", []):
        t, r, s = n.get("translation", [0, 0, 0]), n.get("rotation", [0, 0, 0, 1]), n.get("scale", [1, 1, 1])
        if "matrix" in n:
            t, r, s = decompose(np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T)
        nodes.append(Node(n.get("name", ""), -1, list(n.get("children", [])), tuple(map(float, t)),
                          tuple(map(float, r)), tuple(map(float, s)), n.get("mesh"), n.get("skin")))
    for i, n in enumerate(nodes):
        for c in n.children:
            nodes[c].parent = i
    scenes = gltf.get("scenes") or [{"nodes": [i for i, n in enumerate(nodes) if n.parent < 0]}]
    roots = list(scenes[gltf.get("scene", 0)].get("nodes", []))

    shared: dict[str, Vertices] = {}  # by the accessors they're read from

    def vertices(p: dict) -> Vertices:
        attrs = p["attributes"]
        key = json.dumps([attrs, p.get("targets", [])], sort_keys=True)
        if key in shared:
            return shared[key]
        positions = get(attrs["POSITION"]).astype(np.float32)
        n = len(positions)
        normals = get(attrs["NORMAL"]).astype(np.float32) if "NORMAL" in attrs else np.tile(
            np.float32([0, 0, 1]), (n, 1))
        uvs = get(attrs["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in attrs else np.zeros((n, 2), np.float32)
        joints = get(attrs["JOINTS_0"]).astype(np.float32) if "JOINTS_0" in attrs else None
        weights = get(attrs["WEIGHTS_0"]).astype(np.float32) if "WEIGHTS_0" in attrs else None
        if weights is not None:
            total = weights.sum(axis=1, keepdims=True)
            weights = np.where(total > 0, weights / np.maximum(total, 1e-8), weights).astype(np.float32)
        targets = [(get(t["POSITION"]).astype(np.float32) if "POSITION" in t else np.zeros((n, 3), np.float32),
                    get(t["NORMAL"]).astype(np.float32) if "NORMAL" in t else None)
                   for t in p.get("targets", [])]
        shared[key] = Vertices(positions, normals, uvs, joints, weights, targets)
        return shared[key]

    meshes = []
    for m in gltf.get("meshes", []):
        primitives = []
        for p in m.get("primitives", []):
            if p.get("mode", TRIANGLES) != TRIANGLES:
                log.warning("skipping a %s primitive that isn't triangles", m.get("name", "mesh"))
                continue
            v = vertices(p)
            indices = (get(p["indices"]).reshape(-1).astype(np.uint32) if "indices" in p
                       else np.arange(len(v.positions), dtype=np.uint32))
            primitives.append(Primitive(v, indices, p.get("material")))
        count = max((len(p.vertices.targets) for p in primitives), default=0)
        weights = list(m.get("weights", [])) or [0.0] * count
        meshes.append(Mesh(m.get("name", ""), primitives, [float(w) for w in weights]))

    skins = []
    for s in gltf.get("skins", []):
        joints = list(s["joints"])
        ibm = (get(s["inverseBindMatrices"]).astype(np.float64) if "inverseBindMatrices" in s
               else np.tile(np.eye(4), (len(joints), 1, 1)))
        skins.append(Skin(joints, ibm))

    samplers = [Sampler(s.get("magFilter", 9729), s.get("minFilter", 9987), s.get("wrapS", 10497),
                        s.get("wrapT", 10497)) for s in gltf.get("samplers", [])]
    textures = [Texture(t.get("source", 0), samplers[t["sampler"]] if "sampler" in t else Sampler())
                for t in gltf.get("textures", [])]

    images = []
    for im in gltf.get("images", []):
        if "bufferView" in im:
            v = gltf["bufferViews"][im["bufferView"]]
            images.append(bytes(blob[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]))
        else:
            log.warning("skipping an image that isn't inside the file")
            images.append(b"")

    materials = [material(m) for m in gltf.get("materials", [])]
    ext = gltf.get("extensions", {})
    version = "1.0" if "VRMC_vrm" in ext else "0.x" if "VRM" in ext else ""
    humanoid, expressions = {}, {}
    if version == "1.0":
        vrm = ext["VRMC_vrm"]
        humanoid = {bone: spec["node"] for bone, spec in vrm.get("humanoid", {}).get("humanBones", {}).items()
                    if isinstance(spec, dict) and "node" in spec}
        for preset, spec in vrm.get("expressions", {}).get("preset", {}).items():
            expressions[preset] = [(b["node"], b["index"], float(b.get("weight", 1.0)))
                                   for b in spec.get("morphTargetBinds", []) if "node" in b and "index" in b]
    elif version == "0.x":
        log.info("%s is a VRM 0.x model: it shows, but stays unanimated", path.name)
    return Vrm(path, gltf, nodes, roots, meshes, skins, materials, textures, images, version, humanoid, expressions)


def material(m: dict) -> Material:
    pbr = m.get("pbrMetallicRoughness", {})
    base = pbr.get("baseColorTexture")
    transform = (base or {}).get("extensions", {}).get("KHR_texture_transform", {})
    normal = m.get("normalTexture")
    emissive = m.get("emissiveTexture")
    return Material(
        name=m.get("name", ""),
        base_color=tuple(map(float, pbr.get("baseColorFactor", [1, 1, 1, 1]))),
        base_texture=base["index"] if base else None,
        uv_offset=tuple(map(float, transform.get("offset", [0, 0]))),
        uv_scale=tuple(map(float, transform.get("scale", [1, 1]))),
        uv_rotation=float(transform.get("rotation", 0.0)),
        emissive=tuple(map(float, m.get("emissiveFactor", [0, 0, 0]))),
        emissive_texture=emissive["index"] if emissive else None,
        normal_texture=normal["index"] if normal else None,
        normal_scale=float((normal or {}).get("scale", 1.0)),
        alpha_mode=m.get("alphaMode", "OPAQUE"),
        alpha_cutoff=float(m.get("alphaCutoff", 0.5)),
        double_sided=bool(m.get("doubleSided", False)),
        mtoon=m.get("extensions", {}).get("VRMC_materials_mtoon"),
    )


def decompose(m: np.ndarray) -> tuple[tuple, tuple, tuple]:
    """A 4×4 T·R·S matrix back into its parts."""
    from .mathx import qfrom_mat3

    t = m[:3, 3]
    s = np.linalg.norm(m[:3, :3], axis=0)
    r = m[:3, :3] / np.where(s > 0, s, 1.0)
    return tuple(t), qfrom_mat3(r), tuple(s)


def decode_images(model: Vrm) -> list[np.ndarray | None]:
    """Every image as RGBA pixels (height, width, 4), uint8; None where it
    can't be read."""
    from PIL import Image

    out = []
    for data in model.images:
        try:
            with Image.open(io.BytesIO(data)) as im:
                out.append(np.asarray(im.convert("RGBA")))
        except Exception as e:  # noqa: BLE001 - a broken texture mustn't stop the character
            log.warning("couldn't read a texture of %s: %s", model.path.name, e)
            out.append(None)
    return out
