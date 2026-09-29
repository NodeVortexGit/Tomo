"""MToon: VRM's toon shading, the look VRoid models are made for.

A VRM 1.0 file describes it per material in the ``VRMC_materials_mtoon``
extension: a shade colour for the side away from the light and how sharp the
step to it is, a rim, a matcap, emission, an outline. :class:`MToonParams`
reads that (the spec's defaults filling any gaps); :func:`build` turns a
material into what the shader (``mtoon.glsl``) takes — a block of numbers and
a texture per slot — plus a second material for the outline, when the file
asks for one: the same mesh drawn again, grown along its normals with only
its back faces showing.

A material without MToon is drawn flat (unlit), as glTF's
``KHR_materials_unlit`` fallback asks.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from enum import Enum

from .vrm import Material

# Transparent materials are drawn in the order of their render queue offset.
FLAG_NORMAL_MAP = 1
FLAG_MATCAP = 2
FLAG_SHADING_SHIFT_TEXTURE = 4
FLAG_DOUBLE_SIDED = 8
FLAG_ALPHA_MASK = 16
FLAG_ALPHA_BLEND = 32
FLAG_OUTLINE_SCREEN = 64
FLAG_UNLIT = 128

# Texture slots: (unit, whether its colours are sRGB-encoded).
BASE, SHADE, EMISSIVE, NORMAL, MATCAP, RIM, SHIFT_OR_OUTLINE = range(7)
SRGB = {BASE: True, SHADE: True, EMISSIVE: True, NORMAL: False, MATCAP: True, RIM: True, SHIFT_OR_OUTLINE: False}
SLOT_NAMES = {BASE: "u_base", SHADE: "u_shade", EMISSIVE: "u_emissive", NORMAL: "u_normal", MATCAP: "u_matcap",
              RIM: "u_rim", SHIFT_OR_OUTLINE: "u_shift_or_outline"}


class Outline(Enum):
    NONE = "none"
    WORLD = "worldCoordinates"  # width in the model's units (m)
    SCREEN = "screenCoordinates"  # width as a share of the screen's height


@dataclass(frozen=True)
class MToonParams:
    """One material's ``VRMC_materials_mtoon``, with the spec's defaults."""

    shade_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shade_texture: int | None = None
    shading_shift: float = 0.0
    shading_shift_texture: tuple[int, float] | None = None  # (texture, scale)
    shading_toony: float = 0.9
    matcap_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    matcap_texture: int | None = None
    rim_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rim_texture: int | None = None
    rim_lighting_mix: float = 1.0
    rim_fresnel_power: float = 5.0
    rim_lift: float = 0.0
    outline: Outline = Outline.NONE
    outline_width: float = 0.0
    outline_width_texture: int | None = None
    outline_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    outline_lighting_mix: float = 1.0
    uv_scroll: tuple[float, float] = (0.0, 0.0)
    uv_rotation: float = 0.0
    render_queue: int = 0
    transparent_with_z_write: bool = False

    @classmethod
    def from_json(cls, mtoon: dict) -> "MToonParams":
        def num(key, default):
            value = mtoon.get(key)
            return float(value) if isinstance(value, (int, float)) else default

        def color(key, default):
            value = mtoon.get(key)
            if isinstance(value, list) and len(value) >= 3:
                return tuple(float(v) for v in value[:3])
            return default

        def texture(key):
            value = mtoon.get(key)
            return value.get("index") if isinstance(value, dict) else None

        shift = texture("shadingShiftTexture")
        try:
            outline = Outline(mtoon.get("outlineWidthMode", "none"))
        except ValueError:
            outline = Outline.NONE
        return cls(
            shade_color=color("shadeColorFactor", (0.0, 0.0, 0.0)),
            shade_texture=texture("shadeMultiplyTexture"),
            shading_shift=num("shadingShiftFactor", 0.0),
            shading_shift_texture=(shift, float(mtoon["shadingShiftTexture"].get("scale", 1.0)))
            if shift is not None else None,
            shading_toony=num("shadingToonyFactor", 0.9),
            matcap_color=color("matcapFactor", (1.0, 1.0, 1.0)),
            matcap_texture=texture("matcapTexture"),
            rim_color=color("parametricRimColorFactor", (0.0, 0.0, 0.0)),
            rim_texture=texture("rimMultiplyTexture"),
            rim_lighting_mix=num("rimLightingMixFactor", 1.0),
            rim_fresnel_power=num("parametricRimFresnelPowerFactor", 5.0),
            rim_lift=num("parametricRimLiftFactor", 0.0),
            outline=outline,
            outline_width=num("outlineWidthFactor", 0.0),
            outline_width_texture=texture("outlineWidthMultiplyTexture"),
            outline_color=color("outlineColorFactor", (0.0, 0.0, 0.0)),
            outline_lighting_mix=num("outlineLightingMixFactor", 1.0),
            uv_scroll=(num("uvAnimationScrollXSpeedFactor", 0.0), num("uvAnimationScrollYSpeedFactor", 0.0)),
            uv_rotation=num("uvAnimationRotationSpeedFactor", 0.0),
            render_queue=int(mtoon.get("renderQueueOffsetNumber", 0) or 0),
            transparent_with_z_write=bool(mtoon.get("transparentWithZWrite", False)),
        )

    def outlined(self) -> bool:
        return self.outline != Outline.NONE and self.outline_width > 0.0


@dataclass(frozen=True)
class ShaderMaterial:
    """What one draw of a material needs: the shader's numbers, a texture
    (glTF texture index) per slot, and how to blend and cull."""

    block: bytes  # the shader's ``Material`` uniform block (std140)
    textures: dict[int, int]  # slot → texture index
    alpha_mode: str
    double_sided: bool
    outline: bool
    render_queue: int
    z_write: bool

    @property
    def flags(self) -> int:
        return struct.unpack_from("<i", self.block, BLOCK_FLAGS)[0]


BLOCK_SIZE = 192
BLOCK_FLAGS = 176


def pack(base_color, shade, emissive, matcap, rim_color, outline_color, uv_transform, uv_motion, shading, rim,
         outline, flags) -> bytes:
    """The ``Material`` uniform block, std140: eleven vec4s and an ivec4."""
    floats = [*base_color, *shade, 1.0, *emissive, 1.0, *matcap, 1.0, *rim_color, 1.0, *outline_color, 1.0,
              *uv_transform, *uv_motion, *shading, *rim, *outline]
    return struct.pack("<44f4i", *floats, flags, 0, 0, 0)


def build(material: Material) -> tuple[ShaderMaterial, ShaderMaterial | None]:
    """The surface (and, if the file asks for one, the outline) for a glTF
    material."""
    params = MToonParams.from_json(material.mtoon) if material.mtoon is not None else None
    flags = 0
    if material.double_sided:
        flags |= FLAG_DOUBLE_SIDED
    alpha_cutoff = 0.0
    if material.alpha_mode == "MASK":
        flags |= FLAG_ALPHA_MASK
        alpha_cutoff = material.alpha_cutoff
    elif material.alpha_mode == "BLEND":
        flags |= FLAG_ALPHA_BLEND
    textures = {}
    if material.base_texture is not None:
        textures[BASE] = material.base_texture
    if material.emissive_texture is not None:
        textures[EMISSIVE] = material.emissive_texture
    if material.normal_texture is not None:
        textures[NORMAL] = material.normal_texture
        flags |= FLAG_NORMAL_MAP
    if params is None:
        params = MToonParams()
        flags |= FLAG_UNLIT
    else:
        for slot, index in ((SHADE, params.shade_texture), (MATCAP, params.matcap_texture),
                            (RIM, params.rim_texture)):
            if index is not None:
                textures[slot] = index
        if params.matcap_texture is not None:
            flags |= FLAG_MATCAP
        if params.shading_shift_texture is not None:
            flags |= FLAG_SHADING_SHIFT_TEXTURE
            textures[SHIFT_OR_OUTLINE] = params.shading_shift_texture[0]
    if params.outline == Outline.SCREEN:
        flags |= FLAG_OUTLINE_SCREEN

    def block(flags: int) -> bytes:
        shift_scale = params.shading_shift_texture[1] if params.shading_shift_texture else 0.0
        return pack(material.base_color, params.shade_color, material.emissive, params.matcap_color,
                    params.rim_color, params.outline_color,
                    (*material.uv_offset, *material.uv_scale),
                    (*params.uv_scroll, params.uv_rotation, material.uv_rotation),
                    (params.shading_shift, shift_scale, params.shading_toony, alpha_cutoff),
                    (params.rim_fresnel_power, params.rim_lift, params.rim_lighting_mix, params.outline_lighting_mix),
                    (params.outline_width, material.normal_scale, 0.0, 0.0), flags)

    surface = ShaderMaterial(block(flags), textures, material.alpha_mode, material.double_sided, False,
                             params.render_queue, material.alpha_mode != "BLEND" or params.transparent_with_z_write)
    if not params.outlined():
        return surface, None
    # The outline's slot holds its width, not the surface's shading shift.
    outline_textures = {k: v for k, v in textures.items() if k != SHIFT_OR_OUTLINE}
    if params.outline_width_texture is not None:
        outline_textures[SHIFT_OR_OUTLINE] = params.outline_width_texture
    outline = replace(surface, block=block(flags & ~FLAG_SHADING_SHIFT_TEXTURE), textures=outline_textures,
                      outline=True)
    return surface, outline
