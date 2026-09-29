"""The on-screen interface: Dear ImGui (through imgui-bundle), drawn with
moderngl into the same window as the character.

:class:`Ui` owns the ImGui context: the fonts (Roboto, which has Cyrillic,
plus Font Awesome's icons), the colours, the input the window forwards
(pointer, buttons, wheel, keys, typed characters) and the drawing of each
frame's interface. The chat window (:mod:`.chat`), the lock screen
(:mod:`.workout`) and the control badge (:mod:`.control`) are drawn with it.

Everything is laid out in logical pixels; ``scale`` (the display's scaling,
e.g. 1.5 at 150 %) is applied when drawing, and the fonts are rasterized at
that density so text stays sharp.

Pictures in the chat (the images being sent, and those in the transcript)
are textures too: :class:`Thumbnails` reads them in the background and
uploads them between frames.
"""

from __future__ import annotations

import ctypes
import logging
import queue
import threading
from collections import OrderedDict
from importlib import resources
from pathlib import Path

import moderngl
import numpy as np
from imgui_bundle import imgui

log = logging.getLogger(__name__)

FONT_SIZE = 16.0
# Font Awesome 6 icons used in the interface.
ICON_VOLUME = ""
ICON_MUTED = ""
ICON_USER = ""
ICON_MOUSE = ""
ICON_CLOSE = ""
ICON_POWER = ""
ICON_MIC = ""
ICON_SEND = ""
ICON_PAPERCLIP = ""
ICON_IMAGE = ""

VERTEX = """#version 330 core
uniform mat4 u_proj;
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_color;
out vec2 v_uv;
out vec4 v_color;
void main() {
    v_uv = in_uv;
    v_color = in_color;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
"""
FRAGMENT = """#version 330 core
uniform sampler2D u_texture;
in vec2 v_uv;
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color * texture(u_texture, v_uv);
}
"""


def rgba(r: int, g: int, b: int, a: int = 255) -> int:
    """A colour for ImGui's draw lists, from 0–255 channels."""
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(r / 255, g / 255, b / 255, a / 255))


def vec4(r: int, g: int, b: int, a: float = 255) -> imgui.ImVec4:
    return imgui.ImVec4(r / 255, g / 255, b / 255, a / 255)


def flags(*members) -> int:
    """ImGui flags, or-ed together."""
    out = 0
    for m in members:
        out |= m.value if hasattr(m, "value") else int(m)
    return out


def fonts_dir() -> Path:
    return Path(str(resources.files("imgui_bundle").joinpath("assets", "fonts")))


class Ui:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.context = imgui.create_context()
        io = imgui.get_io()
        io.set_ini_filename("")  # nothing to remember between runs
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
        platform = imgui.get_platform_io()
        platform.renderer_texture_max_width = platform.renderer_texture_max_height = 8192
        self.bold = self.icons = None
        self._load_fonts()
        self._style()
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.program["u_texture"].value = 0
        self.vbo = ctx.buffer(reserve=1 << 16, dynamic=True)
        self.ibo = ctx.buffer(reserve=1 << 16, dynamic=True)
        self.vao = ctx.vertex_array(self.program, [(self.vbo, "2f 2f 4f1", "in_pos", "in_uv", "in_color")],
                                    index_buffer=self.ibo, index_element_size=imgui.INDEX_SIZE)
        self.textures: dict[int, moderngl.Texture] = {}
        self.thumbnails = Thumbnails(ctx, self.textures)

    def _load_fonts(self) -> None:
        io = imgui.get_io()
        folder = fonts_dir()
        regular, bold, icons = (folder / "Roboto" / "Roboto-Regular.ttf", folder / "Roboto" / "Roboto-Bold.ttf",
                                folder / "Font_Awesome_6_Free-Solid-900.otf")
        if not regular.is_file():
            log.warning("no Roboto font in imgui-bundle: Cyrillic may not show")
            io.fonts.add_font_default()
            return
        io.fonts.add_font_from_file_ttf(str(regular), FONT_SIZE)
        if icons.is_file():
            config = imgui.ImFontConfig()
            config.merge_mode = True
            io.fonts.add_font_from_file_ttf(str(icons), FONT_SIZE * 0.9, config)
        if bold.is_file():
            self.bold = io.fonts.add_font_from_file_ttf(str(bold), FONT_SIZE)

    def _style(self) -> None:
        style = imgui.get_style()
        imgui.style_colors_dark()
        style.window_rounding = 20.0
        style.frame_rounding = 10.0
        style.popup_rounding = 10.0
        style.scrollbar_rounding = 8.0
        style.grab_rounding = 8.0
        style.child_rounding = 8.0
        style.window_border_size = 0.0
        style.frame_padding = imgui.ImVec2(10.0, 6.0)
        style.item_spacing = imgui.ImVec2(8.0, 6.0)
        colors = {
            imgui.Col_.button: vec4(38, 66, 74), imgui.Col_.button_hovered: vec4(52, 92, 104),
            imgui.Col_.button_active: vec4(64, 196, 208), imgui.Col_.frame_bg: vec4(12, 18, 22),
            imgui.Col_.frame_bg_hovered: vec4(26, 38, 46), imgui.Col_.frame_bg_active: vec4(26, 38, 46),
            imgui.Col_.check_mark: vec4(120, 230, 240), imgui.Col_.header: vec4(38, 66, 74),
            imgui.Col_.header_hovered: vec4(52, 92, 104), imgui.Col_.header_active: vec4(64, 196, 208),
            imgui.Col_.separator: vec4(60, 72, 80), imgui.Col_.popup_bg: vec4(20, 28, 34, 245),
            imgui.Col_.plot_histogram: vec4(64, 196, 208), imgui.Col_.scrollbar_bg: vec4(0, 0, 0, 0),
            imgui.Col_.text_selected_bg: vec4(64, 196, 208, 110),
        }
        for index, color in colors.items():
            style.set_color_(index.value, color)

    # ---- input, forwarded by the window (logical pixels) ------------------------------------------

    def mouse_moved(self, x: float, y: float) -> None:
        imgui.get_io().add_mouse_pos_event(x, y)

    def mouse_left(self) -> None:
        imgui.get_io().add_mouse_pos_event(-3.4e38, -3.4e38)

    def mouse_button(self, button: int, down: bool) -> None:
        if 0 <= button < 5:
            imgui.get_io().add_mouse_button_event(button, down)

    def scrolled(self, dx: float, dy: float) -> None:
        imgui.get_io().add_mouse_wheel_event(dx, dy)

    def key(self, key: "imgui.Key | None", down: bool, ctrl: bool, shift: bool, alt: bool, super_: bool) -> None:
        io = imgui.get_io()
        io.add_key_event(imgui.Key.mod_ctrl, ctrl)
        io.add_key_event(imgui.Key.mod_shift, shift)
        io.add_key_event(imgui.Key.mod_alt, alt)
        io.add_key_event(imgui.Key.mod_super, super_)
        if key is not None:
            io.add_key_event(key, down)

    def typed(self, codepoint: int) -> None:
        if 0 < codepoint < 0x110000:
            imgui.get_io().add_input_character(codepoint)

    def wants_mouse(self) -> bool:
        return imgui.get_io().want_capture_mouse

    def wants_keyboard(self) -> bool:
        return imgui.get_io().want_text_input

    # ---- a frame -----------------------------------------------------------------------------

    def begin(self, width: float, height: float, scale: float, dt: float) -> None:
        self.thumbnails.upload()
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(width, height)
        io.display_framebuffer_scale = imgui.ImVec2(scale, scale)
        io.delta_time = max(dt, 1e-4)
        imgui.new_frame()

    def end(self, framebuffer: moderngl.Framebuffer | None = None) -> None:
        imgui.render()
        self.draw(imgui.get_draw_data(), framebuffer or self.ctx.screen)

    def draw(self, data, fb: moderngl.Framebuffer) -> None:
        self._update_textures()
        width, height = data.display_size.x, data.display_size.y
        sx, sy = data.framebuffer_scale.x, data.framebuffer_scale.y
        fb_w, fb_h = int(width * sx), int(height * sy)
        if fb_w <= 0 or fb_h <= 0:
            return
        ctx = self.ctx
        fb.use()
        ctx.viewport = (0, 0, fb_w, fb_h)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE,
                          moderngl.ONE_MINUS_SRC_ALPHA)
        x0, y0 = data.display_pos.x, data.display_pos.y
        proj = np.array([[2.0 / width, 0, 0, -1.0 - 2.0 * x0 / width],
                         [0, -2.0 / height, 0, 1.0 + 2.0 * y0 / height],
                         [0, 0, -1.0, 0],
                         [0, 0, 0, 1.0]], dtype=np.float32)
        self.program["u_proj"].write(np.ascontiguousarray(proj.T).tobytes())
        for commands in data.cmd_lists:
            vtx_size = commands.vtx_buffer.size() * imgui.VERTEX_SIZE
            idx_size = commands.idx_buffer.size() * imgui.INDEX_SIZE
            if vtx_size == 0 or idx_size == 0:
                continue
            self._fill(self.vbo, ctypes.string_at(commands.vtx_buffer.data_address(), vtx_size))
            self._fill(self.ibo, ctypes.string_at(commands.idx_buffer.data_address(), idx_size))
            for cmd in commands.cmd_buffer:
                if cmd.elem_count == 0:
                    continue
                texture = self.textures.get(cmd.get_tex_id())
                if texture is None:
                    continue
                cx, cy, cz, cw = cmd.clip_rect.x, cmd.clip_rect.y, cmd.clip_rect.z, cmd.clip_rect.w
                left, top = (cx - x0) * sx, (cy - y0) * sy
                right, bottom = (cz - x0) * sx, (cw - y0) * sy
                if right <= left or bottom <= top:
                    continue
                ctx.scissor = (int(left), int(fb_h - bottom), int(right - left), int(bottom - top))
                texture.use(0)
                self.vao.render(moderngl.TRIANGLES, vertices=cmd.elem_count, first=cmd.idx_offset)
        ctx.scissor = None
        ctx.disable(moderngl.BLEND)

    def _fill(self, buffer: moderngl.Buffer, data: bytes) -> None:
        if buffer.size < len(data):
            buffer.orphan(max(len(data), buffer.size * 2))
        buffer.write(data)

    def _update_textures(self) -> None:
        """ImGui's textures (the font atlas, as it grows): made, updated and
        dropped as it asks."""
        status = imgui.ImTextureStatus
        for tex in imgui.get_platform_io().textures:
            if tex.status == status.want_create:
                pixels = tex.get_pixels_array()
                texture = self.ctx.texture((tex.width, tex.height), 4, np.ascontiguousarray(pixels).tobytes())
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                texture.repeat_x = texture.repeat_y = False
                self.textures[texture.glo] = texture
                tex.set_tex_id(texture.glo)
                tex.status = status.ok
            elif tex.status == status.want_updates:
                texture = self.textures.get(tex.tex_id)
                if texture is not None:
                    full = tex.get_pixels_array().reshape(tex.height, tex.width, 4)
                    for r in tex.updates:
                        block = np.ascontiguousarray(full[r.y:r.y + r.h, r.x:r.x + r.w])
                        texture.write(block.tobytes(), viewport=(r.x, r.y, r.w, r.h))
                tex.status = status.ok
            elif tex.status == status.want_destroy and tex.unused_frames > 0:
                texture = self.textures.pop(tex.tex_id, None)
                if texture is not None:
                    texture.release()
                tex.set_tex_id(0)
                tex.status = status.destroyed


class Thumbnails:
    """Pictures for the chat, as textures ImGui can draw: read and shrunk on
    a thread of their own (a photo can take a moment), uploaded between
    frames, and the least recently shown let go when there are many."""

    EDGE = 256  # px: the longest side kept
    KEEP = 64  # textures kept at most

    def __init__(self, ctx: moderngl.Context, textures: dict[int, moderngl.Texture]) -> None:
        self.ctx = ctx
        self.textures = textures  # the Ui's: ImGui's draw lists name them by GL id
        self.shown: OrderedDict[str, tuple[int, int, int]] = OrderedDict()  # path → (GL id, w, h); id 0: unreadable
        self.pending: set[str] = set()
        self.wanted: queue.SimpleQueue = queue.SimpleQueue()
        self.read: queue.SimpleQueue = queue.SimpleQueue()
        self._worker: threading.Thread | None = None

    def get(self, path) -> tuple[imgui.ImTextureRef | None, float, float] | None:
        """The picture at ``path``: (texture, width, height); the texture is
        None if the file can't be read. None while it's still being read."""
        key = str(path)
        if key in self.shown:
            self.shown.move_to_end(key)
            gl, w, h = self.shown[key]
            return (imgui.ImTextureRef(gl) if gl else None), float(w), float(h)
        if key not in self.pending:
            self.pending.add(key)
            self.wanted.put(key)
            if self._worker is None:
                self._worker = threading.Thread(target=self._read_all, name="tomo-thumbnails", daemon=True)
                self._worker.start()
        return None

    def upload(self) -> None:
        """Turn what's been read into textures (on the drawing thread)."""
        while True:
            try:
                key, picture = self.read.get_nowait()
            except queue.Empty:
                break
            self.pending.discard(key)
            if picture is None:
                self.shown[key] = (0, 1, 1)
                continue
            w, h, pixels = picture
            texture = self.ctx.texture((w, h), 4, pixels)
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = False
            self.textures[texture.glo] = texture
            self.shown[key] = (texture.glo, w, h)
        while len(self.shown) > self.KEEP:
            _, (gl, _, _) = self.shown.popitem(last=False)
            texture = self.textures.pop(gl, None) if gl else None
            if texture is not None:
                texture.release()

    def _read_all(self) -> None:
        while True:
            key = self.wanted.get()
            self.read.put((key, read_thumbnail(key, self.EDGE)))


def read_thumbnail(path: str, edge: int) -> tuple[int, int, bytes] | None:
    """A picture, upright and at most ``edge`` px, as RGBA (None: unreadable)."""
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            picture = ImageOps.exif_transpose(image).convert("RGBA")
        picture.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        return picture.width, picture.height, picture.tobytes()
    except Exception as e:  # noqa: BLE001 - gone, or not a picture: shown as a placeholder
        log.debug("couldn't read the picture %s: %s", path, e)
        return None
