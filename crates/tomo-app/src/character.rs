//! The character: loading a VRoid Studio `.vrm`, measuring it and standing it
//! on the floor. Its motion and expressions live in animation.rs.
//!
//! VRM is a humanoid glTF profile. `bevy_vrm` registers an asset loader for the
//! `.vrm` extension and adds humanoid bones, MToon materials and spring-bone
//! physics on top of `bevy_gltf`. All of that is deliberately hidden behind
//! [`spawn_vrm`] — the ONE function that touches the VRM crate — so if the crate
//! moves or you prefer another loader, this is the only place to edit.

use std::path::PathBuf;

use bevy::asset::io::Reader;
use bevy::asset::{AssetLoader, LoadContext};
use bevy::gltf::{Gltf, GltfError, GltfLoader, GltfLoaderSettings};
use bevy::image::CompressedImageFormats;
use bevy::prelude::*;
use bevy::render::primitives::Aabb;
use bevy::render::renderer::RenderDevice;

use crate::animation::{Animator, VrmSpec};
use crate::bridge::LoadCharacterEvent;

/// Half the body's width as a share of its height (for the grab box).
const BODY_HALF_WIDTH: f32 = 0.2;

/// Marks the character entity. `movement.rs` owns its position; this module
/// owns its appearance.
#[derive(Component)]
pub struct Character {
    /// On-screen height in logical pixels; the model is scaled to it.
    pub height_px: f32,
    /// Where it looks: +1.0 right, -1.0 left, 0.0 at the viewer.
    pub facing: f32,
    /// The turn it's actually showing, easing toward `facing`.
    pub yaw: Quat,
    /// Model-space bounds (min, max) of its meshes, measured once the scene
    /// has spawned. `None` until then, and the character stays hidden.
    pub bounds: Option<(Vec3, Vec3)>,
    /// This frame's on-screen box (logical px, origin top-left): what the
    /// mouse can grab, and the overlay's input region.
    pub screen_rect: Option<Rect>,
}

impl Default for Character {
    fn default() -> Self {
        Self {
            height_px: 320.0,
            facing: 0.0,
            yaw: Quat::IDENTITY,
            bounds: None,
            screen_rect: None,
        }
    }
}

/// The eight corners of the box spanning `min`..`max`.
pub fn box_corners(min: Vec3, max: Vec3) -> [Vec3; 8] {
    std::array::from_fn(|i| {
        Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        )
    })
}

/// Tracks the currently spawned character so we can despawn it on a swap.
#[derive(Resource, Default)]
pub struct ActiveCharacter {
    pub entity: Option<Entity>,
    pub path: Option<PathBuf>,
}

pub struct CharacterPlugin;

impl Plugin for CharacterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveCharacter>()
            // bevy_vrm's plugin registers the `.vrm` loader. Kept here so the
            // dependency stays contained to this module.
            // .add_plugins(bevy_vrm::VrmPlugin)   // TODO(vrm): enable once the
            //   bevy_vrm version is aligned with your bevy version (see Cargo.toml).
            .add_systems(
                Update,
                (on_load_character, measure_character),
            );
    }

    fn finish(&self, app: &mut App) {
        // Mirrors GltfPlugin::finish: texture-compression support is only known
        // once the render device exists.
        let supported_compressed_formats = app
            .world()
            .get_resource::<RenderDevice>()
            .map(|device| CompressedImageFormats::from_features(device.features()))
            .unwrap_or(CompressedImageFormats::NONE);
        app.register_asset_loader(VrmAsGltfLoader(GltfLoader {
            supported_compressed_formats,
            custom_vertex_attributes: default(),
        }));
    }
}

/// A `.vrm` is a binary glTF (`.glb`) with extra VRM extensions, but bevy_gltf
/// only claims the `gltf`/`glb` extensions, so without this nothing can load a
/// `.vrm`. This lends the stock glTF loader the `vrm` extension: meshes, skins
/// and textures load; the VRM-only extras (MToon, spring bones, expressions)
/// are ignored until bevy_vrm replaces it (TODO(vrm)).
struct VrmAsGltfLoader(GltfLoader);

impl AssetLoader for VrmAsGltfLoader {
    type Asset = Gltf;
    type Settings = GltfLoaderSettings;
    type Error = GltfError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &GltfLoaderSettings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Gltf, GltfError> {
        self.0.load(reader, settings, load_context).await
    }

    fn extensions(&self) -> &[&str] {
        &["vrm"]
    }
}

/// React to a request (from the brain) to load/swap the VRM model.
fn on_load_character(
    mut events: EventReader<LoadCharacterEvent>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut active: ResMut<ActiveCharacter>,
) {
    for LoadCharacterEvent(path) in events.read() {
        if active.path.as_ref() == Some(path) {
            continue; // already showing this one
        }
        if let Some(old) = active.entity.take() {
            commands.entity(old).despawn_recursive();
        }
        info!("loading character: {}", path.display());
        let entity = spawn_vrm(&mut commands, &asset_server, path);
        active.entity = Some(entity);
        active.path = Some(path.clone());
    }
}

/// The single VRM seam. Everything VRM-specific lives here.
///
/// Returns the root entity of the spawned character (with a [`Character`] and a
/// [`Transform`] the movement system can drive).
fn spawn_vrm(commands: &mut Commands, asset_server: &AssetServer, path: &std::path::Path) -> Entity {
    // `asset_server.load` on a `.vrm` goes through bevy_vrm's loader once its
    // plugin is added. We load it as a Scene and parent it under our own
    // transform so movement/scaling stays independent of the model's internals.
    //
    // TODO(vrm): with bevy_vrm enabled, prefer its bundle so humanoid bones and
    // spring-bone physics are set up:
    //     commands.spawn((
    //         bevy_vrm::VrmBundle {
    //             vrm: asset_server.load(path.to_path_buf()),
    //             scene_bundle: SceneBundle { transform, ..default() },
    //             ..default()
    //         },
    //         Character::default(),
    //     ))
    //
    // The Scene-only path below keeps this file compiling and rendering a model
    // even before the VRM plugin is wired, using bevy_gltf's glTF pipeline.
    let scene: Handle<Scene> = asset_server.load(
        // glTF scenes are addressed with the `#Scene0` label.
        format!("{}#Scene0", path.to_string_lossy()),
    );

    commands
        .spawn((
            Character::default(),
            // Locomotion MUST be here — the movement, click and chat systems
            // all query `&Locomotion, With<Character>`; without it the
            // character would never move.
            crate::movement::Locomotion::default(),
            // The humanoid rig and expressions, read from the file itself.
            VrmSpec::read(path).unwrap_or_default(),
            Animator::default(),
            SceneRoot(scene),
            Transform::default(),
            // Shown once measured (see measure_character); movement.rs then
            // drops it in from the top of the screen.
            Visibility::Hidden,
            Name::new("tomo-character"),
        ))
        .id()
}

/// Measure the model once its meshes exist (a frame after the scene spawns,
/// when their bounds are computed), then reveal it — so it's never shown at
/// the wrong size.
fn measure_character(
    mut characters: Query<(Entity, &mut Character, &GlobalTransform, &mut Visibility)>,
    children: Query<&Children>,
    meshes: Query<(&Aabb, &GlobalTransform)>,
) {
    for (root, mut character, root_transform, mut visibility) in &mut characters {
        if character.bounds.is_some() {
            continue;
        }
        let to_model = root_transform.affine().inverse();
        let (mut min, mut max) = (Vec3::MAX, Vec3::MIN);
        for entity in children.iter_descendants(root) {
            let Ok((aabb, transform)) = meshes.get(entity) else {
                continue;
            };
            let to_model = to_model * transform.affine();
            for corner in box_corners(aabb.min().into(), aabb.max().into()) {
                let p = to_model.transform_point3(corner);
                min = min.min(p);
                max = max.max(p);
            }
        }
        if min.cmple(max).all() {
            // The meshes' bounds hold the T-pose, arms out wide; posed
            // (animation.rs) they hang at the sides, so keep the box to the
            // body or it would catch clicks meant for the desktop beside her.
            let half_width = BODY_HALF_WIDTH * (max.y - min.y);
            min.x = min.x.max(-half_width);
            max.x = max.x.min(half_width);
            character.bounds = Some((min, max));
            *visibility = Visibility::Inherited;
        }
    }
}

