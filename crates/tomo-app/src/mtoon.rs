//! MToon: VRM's toon shading, the look VRoid models are made for.
//!
//! A VRM 1.0 file describes it per material in the `VRMC_materials_mtoon`
//! extension — a shade colour for the side away from the light and how sharp
//! the step to it is, a rim, a matcap, emission, an outline — and marks the
//! same materials `KHR_materials_unlit` as a fallback, which is all Bevy's
//! glTF loader sees: flat, unlit colour. Once the scene has spawned,
//! [`apply_mtoon`] swaps each of those materials for an [`MToonMaterial`]
//! built from the loaded glTF material plus the extension, and gives the
//! outlined ones a second draw of their mesh for the outline. The shading
//! itself is mtoon.wgsl.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::asset::load_internal_asset;
use bevy::gltf::GltfAssetLabel;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::morph::MeshMorphWeights;
use bevy::render::mesh::skinning::SkinnedMesh;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::primitives::Aabb;
use bevy::render::render_resource::{
    AsBindGroup, Face, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError,
};
use bevy::render::view::NoFrustumCulling;
use serde_json::Value;

use crate::character::Character;

const SHADER: Handle<Shader> = Handle::weak_from_u128(0x7a1e_40b3_9c0d_4f55_8e21_6b3a_0f9d_2c71);

/// Transparent materials are drawn in the order of their render queue
/// offset: each step moves them this far (px) in the back-to-front sort.
const QUEUE_STEP: f32 = 100.0;

/// What a .vrm says about its materials' MToon, by glTF material index.
#[derive(Component, Default)]
pub struct MToonSpec {
    /// The .vrm, to find its textures by label.
    path: PathBuf,
    materials: Vec<Option<MToonParams>>,
}

/// One material's `VRMC_materials_mtoon`, with the spec's defaults filled in.
#[derive(Clone, Debug, PartialEq)]
struct MToonParams {
    shade_color: Vec3,
    shade_texture: Option<usize>,
    shading_shift: f32,
    shading_shift_texture: Option<(usize, f32)>,
    shading_toony: f32,
    matcap_color: Vec3,
    matcap_texture: Option<usize>,
    rim_color: Vec3,
    rim_texture: Option<usize>,
    rim_lighting_mix: f32,
    rim_fresnel_power: f32,
    rim_lift: f32,
    outline: Outline,
    outline_width: f32,
    outline_width_texture: Option<usize>,
    outline_color: Vec3,
    outline_lighting_mix: f32,
    uv_scroll: Vec2,
    uv_rotation: f32,
    render_queue: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outline {
    None,
    /// Width in the model's units (m).
    World,
    /// Width as a share of the screen's height.
    Screen,
}

impl MToonSpec {
    /// Read the MToon of every material in a .vrm's glTF (`path` is the file).
    pub fn from_gltf(gltf: &Value, path: &Path) -> Self {
        let materials = gltf["materials"]
            .as_array()
            .map(|list| {
                list.iter()
                    .map(|m| m.pointer("/extensions/VRMC_materials_mtoon").map(MToonParams::from_json))
                    .collect()
            })
            .unwrap_or_default();
        Self { path: path.to_path_buf(), materials }
    }

    fn has_any(&self) -> bool {
        self.materials.iter().any(Option::is_some)
    }
}

impl MToonParams {
    fn from_json(mtoon: &Value) -> Self {
        let num = |key: &str, default: f32| mtoon[key].as_f64().map_or(default, |n| n as f32);
        let color = |key: &str, default: Vec3| {
            mtoon[key].as_array().filter(|a| a.len() >= 3).map_or(default, |a| {
                Vec3::from_array([0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
            })
        };
        let texture = |key: &str| mtoon[key]["index"].as_u64().map(|i| i as usize);
        let outline = match mtoon["outlineWidthMode"].as_str() {
            Some("worldCoordinates") => Outline::World,
            Some("screenCoordinates") => Outline::Screen,
            _ => Outline::None,
        };
        Self {
            shade_color: color("shadeColorFactor", Vec3::ZERO),
            shade_texture: texture("shadeMultiplyTexture"),
            shading_shift: num("shadingShiftFactor", 0.0),
            shading_shift_texture: texture("shadingShiftTexture")
                .map(|i| (i, mtoon["shadingShiftTexture"]["scale"].as_f64().unwrap_or(1.0) as f32)),
            shading_toony: num("shadingToonyFactor", 0.9),
            matcap_color: color("matcapFactor", Vec3::ONE),
            matcap_texture: texture("matcapTexture"),
            rim_color: color("parametricRimColorFactor", Vec3::ZERO),
            rim_texture: texture("rimMultiplyTexture"),
            rim_lighting_mix: num("rimLightingMixFactor", 1.0),
            rim_fresnel_power: num("parametricRimFresnelPowerFactor", 5.0),
            rim_lift: num("parametricRimLiftFactor", 0.0),
            outline,
            outline_width: num("outlineWidthFactor", 0.0),
            outline_width_texture: texture("outlineWidthMultiplyTexture"),
            outline_color: color("outlineColorFactor", Vec3::ZERO),
            outline_lighting_mix: num("outlineLightingMixFactor", 1.0),
            uv_scroll: Vec2::new(num("uvAnimationScrollXSpeedFactor", 0.0), num("uvAnimationScrollYSpeedFactor", 0.0)),
            uv_rotation: num("uvAnimationRotationSpeedFactor", 0.0),
            render_queue: mtoon["renderQueueOffsetNumber"].as_i64().unwrap_or(0) as i32,
        }
    }

    fn outlined(&self) -> bool {
        self.outline != Outline::None && self.outline_width > 0.0
    }
}

/// The GPU side of one MToon material (mtoon.wgsl's `MToon`).
mod gpu {
    // The derive checks each field with a function nothing calls.
    #![allow(dead_code)]

    use bevy::math::{Vec2, Vec4};
    use bevy::render::render_resource::ShaderType;

    #[derive(Clone, Copy, Debug, Default, ShaderType)]
    pub struct MToonUniform {
        pub base_color: Vec4,
        pub shade_color: Vec4,
        pub emissive: Vec4,
        pub matcap_color: Vec4,
        pub rim_color: Vec4,
        pub outline_color: Vec4,
        pub uv_scroll: Vec2,
        pub uv_rotation: f32,
        pub shading_shift: f32,
        pub shading_shift_scale: f32,
        pub shading_toony: f32,
        pub rim_fresnel_power: f32,
        pub rim_lift: f32,
        pub rim_lighting_mix: f32,
        pub outline_width: f32,
        pub outline_lighting_mix: f32,
        pub alpha_cutoff: f32,
        pub flags: u32,
    }
}
use gpu::MToonUniform;

const FLAG_NORMAL_MAP: u32 = 1;
const FLAG_MATCAP: u32 = 2;
const FLAG_SHADING_SHIFT_TEXTURE: u32 = 4;
const FLAG_DOUBLE_SIDED: u32 = 8;
const FLAG_ALPHA_MASK: u32 = 16;
const FLAG_ALPHA_BLEND: u32 = 32;
const FLAG_OUTLINE_SCREEN: u32 = 64;

/// A material shaded with MToon: either the surface, or (`outline`) its
/// outline.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
#[bind_group_data(MToonKey)]
pub struct MToonMaterial {
    #[uniform(0)]
    uniform: MToonUniform,
    #[texture(1)]
    #[sampler(2)]
    base_color_texture: Option<Handle<Image>>,
    #[texture(3)]
    shade_multiply_texture: Option<Handle<Image>>,
    #[texture(4)]
    emissive_texture: Option<Handle<Image>>,
    #[texture(5)]
    normal_texture: Option<Handle<Image>>,
    #[texture(6)]
    matcap_texture: Option<Handle<Image>>,
    #[texture(7)]
    rim_multiply_texture: Option<Handle<Image>>,
    #[texture(8)]
    shading_shift_texture: Option<Handle<Image>>,
    #[texture(9)]
    outline_width_texture: Option<Handle<Image>>,
    alpha_mode: AlphaMode,
    double_sided: bool,
    outline: bool,
    render_queue: i32,
}

/// What the render pipeline depends on.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MToonKey {
    double_sided: bool,
    outline: bool,
}

impl From<&MToonMaterial> for MToonKey {
    fn from(material: &MToonMaterial) -> Self {
        Self { double_sided: material.double_sided, outline: material.outline }
    }
}

impl Material for MToonMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    fn depth_bias(&self) -> f32 {
        self.render_queue as f32 * QUEUE_STEP
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let MToonKey { double_sided, outline } = key.bind_group_data;
        descriptor.primitive.cull_mode = match (outline, double_sided) {
            // The outline is the grown mesh's far side.
            (true, _) => Some(Face::Front),
            (false, true) => None,
            (false, false) => Some(Face::Back),
        };
        if outline {
            descriptor.vertex.shader_defs.push("MTOON_OUTLINE".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("MTOON_OUTLINE".into());
            }
        }
        Ok(())
    }
}

impl MToonMaterial {
    /// The surface (and, if the file asks for one, the outline) for a glTF
    /// material, from what Bevy loaded of it plus its MToon.
    fn build(params: &MToonParams, standard: &StandardMaterial, texture: impl Fn(usize) -> Handle<Image>) -> (Self, Option<Self>) {
        let mut flags = 0;
        if standard.normal_map_texture.is_some() {
            flags |= FLAG_NORMAL_MAP;
        }
        if params.matcap_texture.is_some() {
            flags |= FLAG_MATCAP;
        }
        if params.shading_shift_texture.is_some() {
            flags |= FLAG_SHADING_SHIFT_TEXTURE;
        }
        if standard.double_sided {
            flags |= FLAG_DOUBLE_SIDED;
        }
        let alpha_cutoff = match standard.alpha_mode {
            AlphaMode::Mask(cutoff) => {
                flags |= FLAG_ALPHA_MASK;
                cutoff
            }
            AlphaMode::Opaque => 0.0,
            _ => {
                flags |= FLAG_ALPHA_BLEND;
                0.0
            }
        };
        if params.outline == Outline::Screen {
            flags |= FLAG_OUTLINE_SCREEN;
        }
        let uniform = MToonUniform {
            base_color: standard.base_color.to_linear().to_vec4(),
            shade_color: params.shade_color.extend(1.0),
            emissive: standard.emissive.to_vec3().extend(1.0),
            matcap_color: params.matcap_color.extend(1.0),
            rim_color: params.rim_color.extend(1.0),
            outline_color: params.outline_color.extend(1.0),
            uv_scroll: params.uv_scroll,
            uv_rotation: params.uv_rotation,
            shading_shift: params.shading_shift,
            shading_shift_scale: params.shading_shift_texture.map_or(0.0, |(_, scale)| scale),
            shading_toony: params.shading_toony,
            rim_fresnel_power: params.rim_fresnel_power,
            rim_lift: params.rim_lift,
            rim_lighting_mix: params.rim_lighting_mix,
            outline_width: params.outline_width,
            outline_lighting_mix: params.outline_lighting_mix,
            alpha_cutoff,
            flags,
        };
        let surface = Self {
            uniform,
            base_color_texture: standard.base_color_texture.clone(),
            shade_multiply_texture: params.shade_texture.map(&texture),
            emissive_texture: standard.emissive_texture.clone(),
            normal_texture: standard.normal_map_texture.clone(),
            matcap_texture: params.matcap_texture.map(&texture),
            rim_multiply_texture: params.rim_texture.map(&texture),
            shading_shift_texture: params.shading_shift_texture.map(|(i, _)| texture(i)),
            outline_width_texture: params.outline_width_texture.map(&texture),
            alpha_mode: standard.alpha_mode,
            double_sided: standard.double_sided,
            outline: false,
            render_queue: params.render_queue,
        };
        let outline = params.outlined().then(|| Self { outline: true, ..surface.clone() });
        (surface, outline)
    }
}

/// Marks a character whose materials have been switched to MToon.
#[derive(Component)]
struct MToonApplied;

pub struct MToonPlugin;

impl Plugin for MToonPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER, "mtoon.wgsl", Shader::from_wgsl);
        app.add_plugins(MaterialPlugin::<MToonMaterial> {
            prepass_enabled: false,
            shadows_enabled: false,
            ..default()
        })
        .add_systems(Update, apply_mtoon.after(crate::character::measure_character));
    }
}

/// Once a character's scene has spawned, shade its MToon materials with MToon.
#[allow(clippy::type_complexity)]
fn apply_mtoon(
    mut commands: Commands,
    characters: Query<(Entity, &Character, &MToonSpec), Without<MToonApplied>>,
    children: Query<&Children>,
    primitives: Query<(
        &MeshMaterial3d<StandardMaterial>,
        &Mesh3d,
        &Transform,
        &Parent,
        Option<&SkinnedMesh>,
        Option<&MeshMorphWeights>,
        Option<&Aabb>,
        Has<NoFrustumCulling>,
    )>,
    standard: Res<Assets<StandardMaterial>>,
    mut mtoon: ResMut<Assets<MToonMaterial>>,
    asset_server: Res<AssetServer>,
) {
    for (root, character, spec) in &characters {
        // Measured means the scene's meshes are there.
        if character.bounds.is_none() {
            continue;
        }
        commands.entity(root).insert(MToonApplied);
        if !spec.has_any() {
            continue;
        }
        let texture = |index: usize| asset_server.load(GltfAssetLabel::Texture(index).from_asset(spec.path.clone()));
        let mut made: HashMap<usize, (Handle<MToonMaterial>, Option<Handle<MToonMaterial>>)> = HashMap::new();
        for entity in children.iter_descendants(root) {
            let Ok((material, mesh, transform, parent, skin, morphs, aabb, no_culling)) = primitives.get(entity) else {
                continue;
            };
            let Some(index) = material_index(&material.0) else { continue };
            let Some(params) = spec.materials.get(index).and_then(Option::as_ref) else { continue };
            let Some(loaded) = standard.get(&material.0) else { continue };
            let (surface, outline) = made
                .entry(index)
                .or_insert_with(|| {
                    let (surface, outline) = MToonMaterial::build(params, loaded, texture);
                    (mtoon.add(surface), outline.map(|o| mtoon.add(o)))
                })
                .clone();
            commands
                .entity(entity)
                .remove::<MeshMaterial3d<StandardMaterial>>()
                .insert(MeshMaterial3d(surface));
            if let Some(outline) = outline {
                // A second draw of the same mesh — skinned, with the same
                // blend shapes — for the outline.
                let mut copy = commands.spawn((Mesh3d(mesh.0.clone()), MeshMaterial3d(outline), *transform));
                if let Some(skin) = skin {
                    copy.insert(skin.clone());
                }
                if let Some(morphs) = morphs {
                    copy.insert(morphs.clone());
                }
                if let Some(aabb) = aabb {
                    copy.insert(*aabb);
                }
                if no_culling {
                    copy.insert(NoFrustumCulling);
                }
                copy.set_parent(parent.get());
            }
        }
        info!("MToon on {} materials", made.len());
    }
}

/// The glTF material a loaded material came from: its label is
/// `Material{index}` (or `Material{index} (inverted)`).
fn material_index(handle: &Handle<StandardMaterial>) -> Option<usize> {
    let path = handle.path()?;
    let digits: String = path.label()?.strip_prefix("Material")?.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec() -> MToonSpec {
        let gltf = json!({ "materials": [
            { "name": "plain" },
            { "name": "hair", "extensions": { "VRMC_materials_mtoon": {
                "shadeColorFactor": [0.5, 0.25, 1.0],
                "shadeMultiplyTexture": { "index": 4 },
                "shadingToonyFactor": 0.8,
                "shadingShiftTexture": { "index": 2, "scale": 0.5 },
                "outlineWidthMode": "worldCoordinates",
                "outlineWidthFactor": 0.001,
                "renderQueueOffsetNumber": -2
            } } },
            { "name": "defaults", "extensions": { "VRMC_materials_mtoon": {} } }
        ] });
        MToonSpec::from_gltf(&gltf, Path::new("/m.vrm"))
    }

    #[test]
    fn it_reads_each_materials_mtoon() {
        let spec = spec();
        assert!(spec.has_any());
        assert!(spec.materials[0].is_none(), "a material without MToon keeps Bevy's");
        let hair = spec.materials[1].as_ref().unwrap();
        assert_eq!(hair.shade_color, Vec3::new(0.5, 0.25, 1.0));
        assert_eq!(hair.shade_texture, Some(4));
        assert_eq!(hair.shading_toony, 0.8);
        assert_eq!(hair.shading_shift_texture, Some((2, 0.5)));
        assert_eq!(hair.outline, Outline::World);
        assert!(hair.outlined());
        assert_eq!(hair.render_queue, -2);
    }

    #[test]
    fn missing_values_take_the_specs_defaults() {
        let plain = spec().materials[2].clone().unwrap();
        assert_eq!(plain.shading_toony, 0.9);
        assert_eq!(plain.matcap_color, Vec3::ONE);
        assert_eq!(plain.rim_fresnel_power, 5.0);
        assert_eq!(plain.outline, Outline::None);
        assert!(!plain.outlined());
        assert_eq!(plain.render_queue, 0);
    }

    #[test]
    fn materials_are_built_from_what_bevy_loaded() {
        let hair = spec().materials[1].clone().unwrap();
        let loaded = StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.0),
            alpha_mode: AlphaMode::Mask(0.5),
            double_sided: true,
            ..default()
        };
        let (surface, outline) = MToonMaterial::build(&hair, &loaded, |_| Handle::default());
        assert_eq!(surface.uniform.alpha_cutoff, 0.5);
        let flags = surface.uniform.flags;
        assert!(flags & FLAG_ALPHA_MASK != 0 && flags & FLAG_DOUBLE_SIDED != 0 && flags & FLAG_SHADING_SHIFT_TEXTURE != 0);
        assert!(flags & FLAG_NORMAL_MAP == 0 && flags & FLAG_MATCAP == 0);
        assert!(surface.shade_multiply_texture.is_some() && surface.matcap_texture.is_none());
        assert!(!surface.outline);
        let outline = outline.expect("an outlined material");
        assert!(outline.outline);
        assert!(MToonKey::from(&outline).outline);
        // A transparent material earlier in the queue sorts further back.
        assert!(surface.depth_bias() < 0.0);
    }
}
