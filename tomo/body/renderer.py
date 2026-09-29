"""Drawing the character with OpenGL (moderngl).

:meth:`Renderer.upload` puts a loaded model on the GPU — vertex buffers,
textures, one small uniform block per material — and :meth:`Renderer.draw`
draws it posed, in VRM's order: opaque and cut-out surfaces, their outlines,
then the see-through ones back to front by their render queue.

Skinning is on the GPU: each skin's joint matrices go up once a frame as a
row of float texels, which the vertex shader fetches. Blend shapes (the face)
are on the CPU: only when the face changes are its vertices re-blended and
re-sent.

The camera is orthographic with one world unit per logical pixel and the
origin in the middle of the window (see :class:`View`), so the physics can map
screen positions straight onto the world. Colours are in linear light
inside, sRGB-encoded on the way out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import resources

import moderngl
import numpy as np

from . import mtoon
from .skeleton import Skeleton, blend
from .vrm import Vrm

log = logging.getLogger(__name__)

GL_SRGB8_ALPHA8 = 0x8C43
# The key light: from the upper right, three-quarters on; lit sides show
# their colours as painted. Plus a little light from everywhere.
LIGHT_DIR = np.array([2.0, 4.0, 3.0]) / np.linalg.norm([2.0, 4.0, 3.0])
LIGHT = 1.0
AMBIENT = 0.08
CAMERA_Z = 1000.0  # in front of the screen plane, looking into it (−Z)
DEPTH = 2000.0  # the depth the camera sees


@dataclass(frozen=True)
class View:
    """The window's size in logical pixels (the world's units)."""

    width: float
    height: float

    def matrix(self) -> np.ndarray:
        """World → clip space."""
        near, far = 0.1, DEPTH
        proj = np.array([[2.0 / self.width, 0, 0, 0],
                         [0, 2.0 / self.height, 0, 0],
                         [0, 0, -2.0 / (far - near), -(far + near) / (far - near)],
                         [0, 0, 0, 1]])
        view = np.eye(4)
        view[2, 3] = -CAMERA_Z
        return proj @ view


def shader_source() -> str:
    return resources.files("tomo.body").joinpath("mtoon.glsl").read_text(encoding="utf-8")


@dataclass(eq=False)
class GpuVertices:
    positions: moderngl.Buffer
    normals: moderngl.Buffer
    others: list  # (buffer, format, attribute) for the VAO
    base: tuple[np.ndarray, np.ndarray]  # rest positions and normals, for blend shapes
    deltas: tuple[np.ndarray, np.ndarray | None] | None  # (targets, n, 3) each


@dataclass(eq=False)
class Draw:
    vertices: GpuVertices
    vao: moderngl.VertexArray
    material: mtoon.ShaderMaterial
    block: moderngl.Buffer
    textures: dict[int, moderngl.Texture]
    node: int
    skin: int | None
    mesh: int


@dataclass
class GpuModel:
    opaque: list[Draw] = field(default_factory=list)
    outlines: list[Draw] = field(default_factory=list)
    transparent: list[Draw] = field(default_factory=list)  # back to front
    joints: list[moderngl.Texture] = field(default_factory=list)  # a row per skin
    vertices_by_mesh: dict[int, list[GpuVertices]] = field(default_factory=dict)
    resources: list = field(default_factory=list)

    def release(self) -> None:
        for r in self.resources:
            r.release()
        self.resources.clear()


class Renderer:
    def __init__(self, ctx: moderngl.Context, colour_key: bool = False) -> None:
        self.ctx = ctx
        # On Windows the window shows the desktop through pure black (see
        # window.py): the character keeps just above it.
        self.floor = 1.0 / 255.0 if colour_key else 0.0
        source = shader_source()

        def program(outline: bool) -> moderngl.Program:
            defines = "#define MTOON_OUTLINE\n" if outline else ""
            return ctx.program(
                vertex_shader=f"#version 330 core\n#define VERTEX_SHADER\n{defines}{source}",
                fragment_shader=f"#version 330 core\n#define FRAGMENT_SHADER\n{defines}{source}")

        self.surface = program(False)
        self.outline = program(True)
        for prog in (self.surface, self.outline):
            for slot, name in mtoon.SLOT_NAMES.items():
                if name in prog:
                    prog[name].value = slot
            if "u_joints" in prog:
                prog["u_joints"].value = 7
            prog["Material"].binding = 0
        self.white = ctx.texture((1, 1), 4, b"\xff\xff\xff\xff")

    # ---- upload --------------------------------------------------------------------------

    def upload(self, model: Vrm, images: list[np.ndarray | None]) -> GpuModel:
        ctx, gpu = self.ctx, GpuModel()
        textures: dict[tuple[int, bool], moderngl.Texture] = {}

        def texture(index: int, srgb: bool) -> moderngl.Texture:
            if (index, srgb) in textures:
                return textures[(index, srgb)]
            tex = self.white
            if 0 <= index < len(model.textures):
                t = model.textures[index]
                pixels = images[t.image] if 0 <= t.image < len(images) else None
                if pixels is not None:
                    h, w = pixels.shape[:2]
                    tex = ctx.texture((w, h), 4, np.ascontiguousarray(pixels).tobytes(),
                                      internal_format=GL_SRGB8_ALPHA8 if srgb else None)
                    tex.repeat_x = t.sampler.wrap_s != 33071  # CLAMP_TO_EDGE
                    tex.repeat_y = t.sampler.wrap_t != 33071
                    if t.sampler.min in (9728, 9984, 9986):  # NEAREST (…)
                        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                    else:
                        # Mipmaps even when the file asks for none: the
                        # character is small on screen against her textures.
                        tex.build_mipmaps()
                        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                        tex.anisotropy = min(8.0, ctx.max_anisotropy)
                    gpu.resources.append(tex)
            textures[(index, srgb)] = tex
            return tex

        materials = [mtoon.build(m) for m in model.materials]
        blocks: dict[bytes, moderngl.Buffer] = {}

        def block(material: mtoon.ShaderMaterial) -> moderngl.Buffer:
            if material.block not in blocks:
                blocks[material.block] = ctx.buffer(material.block)
                gpu.resources.append(blocks[material.block])
            return blocks[material.block]

        uploaded: dict[int, GpuVertices] = {}

        def vertices(v) -> GpuVertices:
            if id(v) in uploaded:
                return uploaded[id(v)]
            dynamic = bool(v.targets)
            pos = ctx.buffer(v.positions.astype("f4").tobytes(), dynamic=dynamic)
            nrm = ctx.buffer(v.normals.astype("f4").tobytes(), dynamic=dynamic)
            others = [(ctx.buffer(v.uvs.astype("f4").tobytes()), "2f", "in_uv")]
            n = len(v.positions)
            joints = v.joints if v.joints is not None else np.zeros((n, 4), np.float32)
            weights = v.weights if v.weights is not None else np.zeros((n, 4), np.float32)
            others += [(ctx.buffer(joints.astype("f4").tobytes()), "4f", "in_joints"),
                       (ctx.buffer(weights.astype("f4").tobytes()), "4f", "in_weights")]
            deltas = None
            if dynamic:
                d = np.stack([t[0] for t in v.targets]).astype(np.float32)
                nd = (np.stack([t[1] if t[1] is not None else np.zeros_like(t[0]) for t in v.targets])
                      .astype(np.float32) if any(t[1] is not None for t in v.targets) else None)
                deltas = (d, nd)
            gpu.resources += [pos, nrm, *(b for b, _, _ in others)]
            uploaded[id(v)] = GpuVertices(pos, nrm, others, (v.positions, v.normals), deltas)
            return uploaded[id(v)]

        for node_index, node in enumerate(model.nodes):
            if node.mesh is None:
                continue
            mesh = model.meshes[node.mesh]
            for p in mesh.primitives:
                gv = vertices(p.vertices)
                listed = gpu.vertices_by_mesh.setdefault(node.mesh, [])
                if not any(g is gv for g in listed):
                    listed.append(gv)
                ibo = ctx.buffer(p.indices.astype("u4").tobytes())
                gpu.resources.append(ibo)
                surface, outline = materials[p.material] if p.material is not None else mtoon.build(
                    _default_material())
                for material in (surface, outline):
                    if material is None:
                        continue
                    prog = self.outline if material.outline else self.surface
                    content = [(gv.positions, "3f", "in_position"), (gv.normals, "3f", "in_normal"),
                               *gv.others]
                    content = [c for c in content if c[2] in prog]
                    vao = ctx.vertex_array(prog, content, index_buffer=ibo, index_element_size=4)
                    gpu.resources.append(vao)
                    draw = Draw(gv, vao, material, block(material),
                                {slot: texture(index, mtoon.SRGB[slot]) for slot, index in material.textures.items()},
                                node_index, node.skin, node.mesh)
                    if material.outline:
                        gpu.outlines.append(draw)
                    elif material.alpha_mode == "BLEND":
                        gpu.transparent.append(draw)
                    else:
                        gpu.opaque.append(draw)
        # A transparent material earlier in the queue is further back. The
        # outlines of see-through materials go with them.
        blended = [d for d in gpu.outlines if d.material.alpha_mode == "BLEND"]
        gpu.outlines = [d for d in gpu.outlines if d.material.alpha_mode != "BLEND"]
        gpu.transparent = sorted(gpu.transparent + blended, key=lambda d: d.material.render_queue)
        for skin in model.skins:
            tex = ctx.texture((4 * max(1, len(skin.joints)), 1), 4, dtype="f4")
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            gpu.joints.append(tex)
            gpu.resources.append(tex)
        log.info("uploaded %s: %d draws, %d textures", model.path.name,
                 len(gpu.opaque) + len(gpu.outlines) + len(gpu.transparent), len(textures))
        return gpu

    # ---- a frame ------------------------------------------------------------------------------

    def draw(self, gpu: GpuModel, skeleton: Skeleton, view: View, time: float,
             framebuffer: moderngl.Framebuffer | None = None) -> None:
        ctx = self.ctx
        fb = framebuffer or ctx.screen
        # The frame's joints and face.
        for i, tex in enumerate(gpu.joints):
            mats = skeleton.skin(i)
            tex.write(np.ascontiguousarray(mats.transpose(0, 2, 1), dtype=np.float32).tobytes())
        for mesh, changed in enumerate(skeleton.morph_changed):
            if not changed:
                continue
            skeleton.morph_changed[mesh] = False
            for gv in gpu.vertices_by_mesh.get(mesh, []):
                if gv.deltas is None:
                    continue
                pos, nrm = blend(gv.base[0], gv.base[1], gv.deltas[0], gv.deltas[1], skeleton.morph[mesh])
                gv.positions.write(np.ascontiguousarray(pos, dtype=np.float32).tobytes())
                gv.normals.write(np.ascontiguousarray(nrm, dtype=np.float32).tobytes())

        view_proj = np.ascontiguousarray(view.matrix().T, dtype=np.float32).tobytes()
        frame = {"u_viewport_height": view.height, "u_light_dir": tuple(LIGHT_DIR), "u_light": (LIGHT,) * 3,
                 "u_ambient": (AMBIENT,) * 3, "u_floor": self.floor, "u_time": time}
        for prog in (self.surface, self.outline):
            prog["u_view_proj"].write(view_proj)
            for name, value in frame.items():
                if name in prog:  # the compiler drops what a program doesn't use
                    prog[name].value = value

        fb.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        ctx.disable(moderngl.BLEND)
        fb.depth_mask = True
        for d in gpu.opaque:
            self._draw(d, skeleton, gpu)
        for d in gpu.outlines:
            self._draw(d, skeleton, gpu)
        ctx.enable(moderngl.BLEND)
        # Straight alpha in, premultiplied out: what a see-through window's
        # compositor expects.
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE,
                          moderngl.ONE_MINUS_SRC_ALPHA)
        for d in gpu.transparent:
            fb.depth_mask = d.material.z_write
            self._draw(d, skeleton, gpu)
        fb.depth_mask = True
        ctx.disable(moderngl.CULL_FACE)

    def _draw(self, d: Draw, skeleton: Skeleton, gpu: GpuModel) -> None:
        ctx = self.ctx
        if d.material.outline:
            ctx.enable(moderngl.CULL_FACE)
            ctx.cull_face = "front"  # the outline is the grown mesh's far side
        elif d.material.double_sided:
            ctx.disable(moderngl.CULL_FACE)
        else:
            ctx.enable(moderngl.CULL_FACE)
            ctx.cull_face = "back"
        prog = d.vao.program
        skinned = d.skin is not None and d.skin < len(gpu.joints)
        prog["u_skinned"].value = 1 if skinned else 0
        if skinned:
            gpu.joints[d.skin].use(7)
        else:
            prog["u_model"].write(np.ascontiguousarray(skeleton.world[d.node].T, dtype=np.float32).tobytes())
        d.block.bind_to_uniform_block(0)
        for slot in range(7):
            d.textures.get(slot, self.white).use(slot)
        d.vao.render(moderngl.TRIANGLES)


def _default_material():
    from .vrm import Material

    return Material("default", (1, 1, 1, 1), None, (0, 0), (1, 1), 0.0, (0, 0, 0), None, None, 1.0, "OPAQUE", 0.5,
                    False, None)
