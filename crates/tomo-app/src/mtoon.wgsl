// MToon, VRM's toon shading (VRMC_materials_mtoon 1.0) — see mtoon.rs.
//
// The lit side of a surface shows its base colour and the side away from
// the light its shade colour; `shading_toony` sets how sharp the boundary
// is and `shading_shift` moves it. Ambient light, a matcap and a fresnel rim
// are added on top, then emission. With MTOON_OUTLINE, the same mesh drawn
// again, grown along its normals with only its back faces showing, is the
// outline.

#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_functions,
    skinning,
    morph::morph,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{view, lights, globals},
}

struct MToon {
    base_color: vec4<f32>,
    shade_color: vec4<f32>,
    emissive: vec4<f32>,
    matcap_color: vec4<f32>,
    rim_color: vec4<f32>,
    outline_color: vec4<f32>,
    uv_scroll: vec2<f32>,
    uv_rotation: f32,
    shading_shift: f32,
    shading_shift_scale: f32,
    shading_toony: f32,
    rim_fresnel_power: f32,
    rim_lift: f32,
    rim_lighting_mix: f32,
    outline_width: f32,
    outline_lighting_mix: f32,
    alpha_cutoff: f32,
    flags: u32,
}

const FLAG_NORMAL_MAP: u32 = 1u;
const FLAG_MATCAP: u32 = 2u;
const FLAG_SHADING_SHIFT_TEXTURE: u32 = 4u;
const FLAG_DOUBLE_SIDED: u32 = 8u;
const FLAG_ALPHA_MASK: u32 = 16u;
const FLAG_ALPHA_BLEND: u32 = 32u;
const FLAG_OUTLINE_SCREEN: u32 = 64u;

// An outline under a pixel wide doesn't show; world-space widths meant for a
// close-up are kept at least this wide on screen (px).
const MIN_OUTLINE_PX: f32 = 0.8;
const PI: f32 = 3.141592653589793;

@group(2) @binding(0) var<uniform> material: MToon;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(2) var texture_sampler: sampler;
@group(2) @binding(3) var shade_multiply_texture: texture_2d<f32>;
@group(2) @binding(4) var emissive_texture: texture_2d<f32>;
@group(2) @binding(5) var normal_texture: texture_2d<f32>;
@group(2) @binding(6) var matcap_texture: texture_2d<f32>;
@group(2) @binding(7) var rim_multiply_texture: texture_2d<f32>;
@group(2) @binding(8) var shading_shift_texture: texture_2d<f32>;
@group(2) @binding(9) var outline_width_texture: texture_2d<f32>;

fn has(flag: u32) -> bool {
    return (material.flags & flag) != 0u;
}

// ---- vertex: Bevy's own mesh vertex stage, plus the outline's growth ------

#ifdef MORPH_TARGETS
fn morph_vertex(vertex_in: Vertex) -> Vertex {
    var vertex = vertex_in;
    let first_vertex = mesh[vertex.instance_index].first_vertex_index;
    let vertex_index = vertex.index - first_vertex;

    let weight_count = bevy_pbr::morph::layer_count();
    for (var i: u32 = 0u; i < weight_count; i ++) {
        let weight = bevy_pbr::morph::weight_at(i);
        if weight == 0.0 {
            continue;
        }
        vertex.position += weight * morph(vertex_index, bevy_pbr::morph::position_offset, i);
#ifdef VERTEX_NORMALS
        vertex.normal += weight * morph(vertex_index, bevy_pbr::morph::normal_offset, i);
#endif
#ifdef VERTEX_TANGENTS
        vertex.tangent += vec4(weight * morph(vertex_index, bevy_pbr::morph::tangent_offset, i), 0.0);
#endif
    }
    return vertex;
}
#endif

@vertex
fn vertex(vertex_no_morph: Vertex) -> VertexOutput {
    var out: VertexOutput;

#ifdef MORPH_TARGETS
    var vertex = morph_vertex(vertex_no_morph);
#else
    var vertex = vertex_no_morph;
#endif

#ifdef SKINNED
    var world_from_local = skinning::skin_model(vertex.joint_indices, vertex.joint_weights);
#else
    var world_from_local = mesh_functions::get_world_from_local(vertex_no_morph.instance_index);
#endif

#ifdef VERTEX_NORMALS
#ifdef SKINNED
    out.world_normal = skinning::skin_normals(world_from_local, vertex.normal);
#else
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex_no_morph.instance_index);
#endif
#endif

#ifdef VERTEX_POSITIONS
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
#ifdef MTOON_OUTLINE
    out.world_position = vec4(out.world_position.xyz + outline_offset(vertex, world_from_local, out.world_normal), 1.0);
#endif
    out.position = position_world_to_clip(out.world_position.xyz);
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex_no_morph.instance_index
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex_no_morph.instance_index;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex_no_morph.instance_index, world_from_local[3]);
#endif

    return out;
}

#ifdef MTOON_OUTLINE
// How far to grow a vertex, in world units — which are screen pixels here.
fn outline_offset(vertex: Vertex, world_from_local: mat4x4<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    var mask = 1.0;
#ifdef VERTEX_UVS_A
    mask = textureSampleLevel(outline_width_texture, texture_sampler, vertex.uv, 0.0).g;
#endif
    var width: f32;
    if has(FLAG_OUTLINE_SCREEN) {
        // A share of the screen's height.
        width = material.outline_width * view.viewport.w;
    } else {
        // Model units (m), scaled like the model.
        let scale = length(world_from_local[0].xyz);
        width = max(material.outline_width * scale, MIN_OUTLINE_PX);
    }
    return normalize(world_normal) * width * mask;
}
#endif

// ---- fragment ----------------------------------------------------------------

fn linearstep(a: f32, b: f32, t: f32) -> f32 {
    return saturate((t - a) / max(b - a, 1e-5));
}

// UV animation: rotation about the texture's centre, then scrolling.
fn animated_uv(uv: vec2<f32>) -> vec2<f32> {
    let t = globals.time;
    let angle = material.uv_rotation * t;
    let c = cos(angle);
    let s = sin(angle);
    let centred = uv - 0.5;
    let turned = vec2(c * centred.x - s * centred.y, s * centred.x + c * centred.y) + 0.5;
    return turned + material.uv_scroll * t;
}

// Toward the viewer. An orthographic view looks the same way everywhere.
fn view_vector(world_position: vec3<f32>) -> vec3<f32> {
    if view.clip_from_view[3].w == 1.0 {
        return normalize(vec3(view.clip_from_world[0].z, view.clip_from_world[1].z, view.clip_from_world[2].z));
    }
    return normalize(view.world_position - world_position);
}

struct Lit {
    // The toon-shaded colour under the direct lights, plus ambient.
    color: vec3<f32>,
    // All the light arriving, for the rim.
    light: vec3<f32>,
}

// MToon's lighting. A light whose illuminance times the camera's exposure is
// π counts as unit light: the lit side then shows exactly the base colour.
fn toon_light(normal: vec3<f32>, lit: vec3<f32>, shade: vec3<f32>, shift: f32) -> Lit {
    var out: Lit;
    out.color = vec3(0.0);
    out.light = vec3(0.0);
    for (var i: u32 = 0u; i < lights.n_directional_lights; i ++) {
        let light = lights.directional_lights[i];
        let radiance = light.color.rgb * view.exposure / PI;
        let n_dot_l = dot(normal, light.direction_to_light);
        let shading = linearstep(-1.0 + material.shading_toony, 1.0 - material.shading_toony, n_dot_l + shift);
        out.color += mix(shade, lit, shading) * radiance;
        out.light += radiance;
    }
    // Ambient light is the same from every side, so there's nothing for
    // MToon's GI equalization to even out.
    let ambient = lights.ambient_color.rgb * view.exposure;
    out.color += lit * ambient;
    out.light += ambient * PI;
    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    // Every texture is read up front, in uniform control flow.
#ifdef VERTEX_UVS_A
    let uv = animated_uv(in.uv);
#else
    let uv = vec2(0.0);
#endif
    let base = material.base_color * textureSample(base_color_texture, texture_sampler, uv);
    let shade = material.shade_color.rgb * textureSample(shade_multiply_texture, texture_sampler, uv).rgb;
    let emissive = material.emissive.rgb * textureSample(emissive_texture, texture_sampler, uv).rgb;
    let normal_sample = textureSample(normal_texture, texture_sampler, uv).rgb;
    let rim_multiply = textureSample(rim_multiply_texture, texture_sampler, uv).rgb;
    let shift_sample = textureSample(shading_shift_texture, texture_sampler, uv).r;

    // The surface normal, facing the viewer on a two-sided material's back.
    var N = normalize(in.world_normal);
    let flip = has(FLAG_DOUBLE_SIDED) && !is_front;
#ifdef VERTEX_TANGENTS
    if has(FLAG_NORMAL_MAP) {
        let T = normalize(in.world_tangent.xyz - N * dot(in.world_tangent.xyz, N));
        let B = cross(N, T) * in.world_tangent.w;
        let tangent_normal = normal_sample * 2.0 - 1.0;
        N = normalize(tangent_normal.x * T + tangent_normal.y * B + tangent_normal.z * N);
    }
#endif
    if flip {
        N = -N;
    }

    var shift = material.shading_shift;
    if has(FLAG_SHADING_SHIFT_TEXTURE) {
        shift += shift_sample * material.shading_shift_scale;
    }
    let V = view_vector(in.world_position.xyz);

    // The matcap: a small sphere image, looked up by the normal as the
    // viewer sees it.
    let n_view = normalize((view.view_from_world * vec4(N, 0.0)).xyz);
    let v_view = normalize((view.view_from_world * vec4(V, 0.0)).xyz);
    let x = normalize(vec3(v_view.z, 0.0, -v_view.x));
    let y = cross(v_view, x);
    let matcap_uv = 0.5 + 0.5 * vec2(dot(x, n_view), -dot(y, n_view));
    let matcap = textureSample(matcap_texture, texture_sampler, matcap_uv).rgb;

    if has(FLAG_ALPHA_MASK) && base.a < material.alpha_cutoff {
        discard;
    }
    let alpha = select(1.0, base.a, has(FLAG_ALPHA_BLEND));
    let lit = toon_light(N, base.rgb, shade, shift);

#ifdef MTOON_OUTLINE
    let outline = material.outline_color.rgb * mix(vec3(1.0), lit.color, material.outline_lighting_mix);
    return vec4(outline, alpha);
#else
    var rim = vec3(0.0);
    if has(FLAG_MATCAP) {
        rim += material.matcap_color.rgb * matcap;
    }
    rim += material.rim_color.rgb
        * pow(saturate(1.0 - dot(N, V) + material.rim_lift), material.rim_fresnel_power);
    rim *= rim_multiply * mix(vec3(1.0), lit.light, material.rim_lighting_mix);

    return vec4(lit.color + rim + emissive, alpha);
#endif
}
