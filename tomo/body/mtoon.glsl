// MToon, VRM's toon shading (VRMC_materials_mtoon 1.0) — see mtoon.py.
//
// The lit side of a surface shows its base colour and the side away from
// the light its shade colour; the shading toony factor sets how sharp the
// boundary is and the shading shift moves it. Ambient light, a matcap and a
// fresnel rim are added on top, then emission. With MTOON_OUTLINE, the same
// mesh drawn again, grown along its normals with only its back faces
// showing, is the outline.
//
// renderer.py puts "#version 330 core" and VERTEX_SHADER or FRAGMENT_SHADER
// (and MTOON_OUTLINE for the outline) in front of this file.

layout(std140) uniform Material {
    vec4 base_color;
    vec4 shade_color;
    vec4 emissive;
    vec4 matcap_color;
    vec4 rim_color;
    vec4 outline_color;
    vec4 uv_transform;  // KHR_texture_transform: offset.xy, scale.xy
    vec4 uv_motion;     // scroll speed.xy, rotation speed, KHR rotation
    vec4 shading;       // shift, shift texture scale, toony, alpha cutoff
    vec4 rim;           // fresnel power, lift, lighting mix, outline lighting mix
    vec4 outline;       // width, normal map scale
    ivec4 flags;
} m;

const int FLAG_NORMAL_MAP = 1;
const int FLAG_MATCAP = 2;
const int FLAG_SHADING_SHIFT_TEXTURE = 4;
const int FLAG_DOUBLE_SIDED = 8;
const int FLAG_ALPHA_MASK = 16;
const int FLAG_ALPHA_BLEND = 32;
const int FLAG_OUTLINE_SCREEN = 64;
const int FLAG_UNLIT = 128;

// An outline under a pixel wide doesn't show; world-space widths meant for a
// close-up are kept at least this wide on screen (px).
const float MIN_OUTLINE_PX = 0.8;
const float PI = 3.141592653589793;

bool has(int flag) {
    return (m.flags.x & flag) != 0;
}

// The surface's shading shift (red), or in the outline pass its width (green).
uniform sampler2D u_shift_or_outline;

#ifdef VERTEX_SHADER

uniform mat4 u_view_proj;
uniform mat4 u_model;  // meshes without a skin: their node's world matrix
uniform int u_skinned;
uniform float u_viewport_height;
// Four texels per joint: the columns of its matrix (bind pose → world).
uniform sampler2D u_joints;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec4 in_joints;
in vec4 in_weights;

out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;

mat4 joint_matrix(float joint) {
    int i = int(joint + 0.5) * 4;
    return mat4(texelFetch(u_joints, ivec2(i, 0), 0), texelFetch(u_joints, ivec2(i + 1, 0), 0),
                texelFetch(u_joints, ivec2(i + 2, 0), 0), texelFetch(u_joints, ivec2(i + 3, 0), 0));
}

void main() {
    mat4 world_from_local = u_model;
    if (u_skinned != 0) {
        world_from_local = in_weights.x * joint_matrix(in_joints.x) + in_weights.y * joint_matrix(in_joints.y)
                         + in_weights.z * joint_matrix(in_joints.z) + in_weights.w * joint_matrix(in_joints.w);
    }
    vec4 world = world_from_local * vec4(in_position, 1.0);
    vec3 normal = normalize(transpose(inverse(mat3(world_from_local))) * in_normal);
#ifdef MTOON_OUTLINE
    // How far to grow the vertex, in world units — which are screen pixels.
    float mask = textureLod(u_shift_or_outline, in_uv, 0.0).g;
    float width;
    if (has(FLAG_OUTLINE_SCREEN)) {
        width = m.outline.x * u_viewport_height;  // a share of the screen's height
    } else {
        // Model units (m), scaled like the model.
        width = max(m.outline.x * length(world_from_local[0].xyz), MIN_OUTLINE_PX);
    }
    world.xyz += normal * width * mask;
#endif
    v_world = world.xyz;
    v_normal = normal;
    v_uv = in_uv;
    gl_Position = u_view_proj * world;
}

#endif

#ifdef FRAGMENT_SHADER

uniform float u_time;
uniform vec3 u_light_dir;     // toward the key light
uniform vec3 u_light;         // its radiance: 1 shows the lit side as painted
uniform vec3 u_ambient;
// The darkest a drawn pixel may be: on Windows pure black is the colour key
// that shows the desktop through, so nothing of hers may be that black.
uniform float u_floor;

uniform sampler2D u_base;
uniform sampler2D u_shade;
uniform sampler2D u_emissive;
uniform sampler2D u_normal;
uniform sampler2D u_matcap;
uniform sampler2D u_rim;

in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;

out vec4 frag_color;

float linearstep(float a, float b, float t) {
    return clamp((t - a) / max(b - a, 1e-5), 0.0, 1.0);
}

// KHR_texture_transform (offset, rotation, scale), then MToon's UV
// animation: rotation about the texture's centre, then scrolling.
vec2 texture_uv(vec2 uv) {
    vec2 scaled = uv * m.uv_transform.zw;
    float r = m.uv_motion.w;
    vec2 turned = vec2(cos(r) * scaled.x + sin(r) * scaled.y, -sin(r) * scaled.x + cos(r) * scaled.y);
    uv = turned + m.uv_transform.xy;
    float angle = m.uv_motion.z * u_time;
    vec2 centred = uv - 0.5;
    vec2 spun = vec2(cos(angle) * centred.x - sin(angle) * centred.y,
                     sin(angle) * centred.x + cos(angle) * centred.y) + 0.5;
    return spun + m.uv_motion.xy * u_time;
}

// Surface normals from the screen-space change of position and UV (there
// are no tangents in the file): the frame a normal map is in.
vec3 mapped_normal(vec3 N, vec2 uv) {
    vec3 dp1 = dFdx(v_world);
    vec3 dp2 = dFdy(v_world);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    if (!(invmax < 1e30)) {
        return N;  // no UV change here: nothing to map by
    }
    vec3 t = texture(u_normal, uv).xyz * 2.0 - 1.0;
    t.xy *= m.outline.y;
    return normalize(mat3(T * invmax, B * invmax, N) * t);
}

vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

void main() {
    vec2 uv = texture_uv(v_uv);
    vec4 base = m.base_color * texture(u_base, uv);
    if (has(FLAG_ALPHA_MASK) && base.a < m.shading.w) {
        discard;
    }
    float alpha = has(FLAG_ALPHA_BLEND) ? base.a : 1.0;
    vec3 emissive = m.emissive.rgb * texture(u_emissive, uv).rgb;
    if (has(FLAG_UNLIT)) {
        frag_color = vec4(max(to_srgb(base.rgb + emissive), vec3(u_floor)), alpha);
        return;
    }

    // The surface normal, facing the viewer on a two-sided material's back.
    vec3 N = normalize(v_normal);
    if (has(FLAG_DOUBLE_SIDED) && !gl_FrontFacing) {
        N = -N;
    }
    if (has(FLAG_NORMAL_MAP)) {
        N = mapped_normal(N, uv);
    }
    // The camera is orthographic, looking down −Z: the viewer is +Z everywhere.
    vec3 V = vec3(0.0, 0.0, 1.0);

    float shift = m.shading.x;
    if (has(FLAG_SHADING_SHIFT_TEXTURE)) {
        shift += texture(u_shift_or_outline, uv).r * m.shading.y;
    }
    vec3 shade = m.shade_color.rgb * texture(u_shade, uv).rgb;

    // MToon's lighting: one key light, plus ambient light, the same from
    // every side (so there's nothing for MToon's GI equalization to even out).
    float toony = m.shading.z;
    float n_dot_l = dot(N, u_light_dir);
    float shading = linearstep(-1.0 + toony, 1.0 - toony, n_dot_l + shift);
    vec3 lit = mix(shade, base.rgb, shading) * u_light + base.rgb * u_ambient;
    vec3 light = u_light + u_ambient * PI;

#ifdef MTOON_OUTLINE
    vec3 color = m.outline_color.rgb * mix(vec3(1.0), lit, m.rim.w);
#else
    // The matcap: a small sphere image, looked up by the normal as the viewer
    // sees it; and the rim, brightest where the surface turns away.
    vec3 rim = vec3(0.0);
    if (has(FLAG_MATCAP)) {
        rim += m.matcap_color.rgb * texture(u_matcap, 0.5 + 0.5 * vec2(N.x, -N.y)).rgb;
    }
    rim += m.rim_color.rgb * pow(clamp(1.0 - dot(N, V) + m.rim.y, 0.0, 1.0), m.rim.x);
    rim *= texture(u_rim, uv).rgb * mix(vec3(1.0), light, m.rim.z);
    vec3 color = lit + rim + emissive;
#endif
    frag_color = vec4(max(to_srgb(color), vec3(u_floor)), alpha);
}

#endif
