//! Bringing the character to life: procedural body animation and facial
//! expressions, driven by what it is doing.
//!
//! A VRM carries a standard humanoid map (which node is the left upper arm,
//! the head, …) and named facial expressions (blink, happy, aa, …). This module
//! reads both from the file's glTF JSON, finds the matching entities once the
//! scene has spawned, then every frame poses the skeleton — breathing at rest,
//! a walk cycle paced by the distance walked, gestures the brain asks for,
//! and on top of it all what the physics engine (movement.rs) simulates: limbs
//! dangling and swinging with the motion, knees giving on landing — and
//! drives blinking, emotions and a mouth that moves while the voice plays.
//!
//! Poses work like three-vrm's "normalized" rig: each bone gets a rotation in
//! model axes (+X the model's left, +Y up, +Z toward the viewer) relative to
//! the T-pose, turned into the bone's own frame as `P⁻¹·Q·P·L` (P: the
//! parent's rest rotation in model space, L: the bone's rest rotation). The
//! same pose therefore fits any VRM 1.0 model, whatever its bone axes. (VRM 0.x
//! files have a different layout and stay unanimated for now.)

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::io::Read;
use std::path::Path;

use bevy::prelude::*;
use bevy::render::mesh::morph::MorphWeights;
use serde_json::Value;

use crate::bridge::{
    AnimateEvent, ChatAppendEvent, EmoteEvent, ListeningEvent, SpeakingEvent, ThinkingEvent,
};
use crate::character::Character;
use crate::movement::{Locomotion, Stance, WALK_SPEED};
use crate::window::Busy;
use tomo_core::events::Role;

/// How quickly the body blends toward a new pose, 1/s.
const POSE_RATE: f32 = 10.0;
/// One walk cycle (two steps) covers this many body heights.
const STRIDE: f32 = 0.8;
/// Emotions fade back to neutral after this many seconds.
const EMOTION_SECONDS: f32 = 8.0;

/// The humanoid bones Tomo poses, by their VRM names.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Bone {
    Spine,
    Chest,
    Neck,
    Head,
    LeftShoulder,
    RightShoulder,
    LeftUpperArm,
    RightUpperArm,
    LeftLowerArm,
    RightLowerArm,
    LeftUpperLeg,
    RightUpperLeg,
    LeftLowerLeg,
    RightLowerLeg,
}

const BONES: [(Bone, &str); 14] = [
    (Bone::Spine, "spine"),
    (Bone::Chest, "chest"),
    (Bone::Neck, "neck"),
    (Bone::Head, "head"),
    (Bone::LeftShoulder, "leftShoulder"),
    (Bone::RightShoulder, "rightShoulder"),
    (Bone::LeftUpperArm, "leftUpperArm"),
    (Bone::RightUpperArm, "rightUpperArm"),
    (Bone::LeftLowerArm, "leftLowerArm"),
    (Bone::RightLowerArm, "rightLowerArm"),
    (Bone::LeftUpperLeg, "leftUpperLeg"),
    (Bone::RightUpperLeg, "rightUpperLeg"),
    (Bone::LeftLowerLeg, "leftLowerLeg"),
    (Bone::RightLowerLeg, "rightLowerLeg"),
];

/// Expression presets Tomo drives.
const EMOTIONS: [&str; 5] = ["happy", "sad", "angry", "surprised", "relaxed"];

/// What the .vrm says about its rig: humanoid bone → node name, and each
/// expression preset → (node name, morph target, weight) binds.
#[derive(Component, Default)]
pub struct VrmSpec {
    bones: HashMap<String, String>,
    expressions: HashMap<String, Vec<(String, usize, f32)>>,
}

impl VrmSpec {
    /// Read the rig description from a .vrm (binary glTF): the JSON chunk's
    /// `VRMC_vrm` extension. `None` if the file isn't VRM 1.0.
    pub fn read(path: &Path) -> Option<Self> {
        let mut file = std::fs::File::open(path).ok()?;
        let mut header = [0u8; 20];
        file.read_exact(&mut header).ok()?;
        if &header[0..4] != b"glTF" || &header[16..20] != b"JSON" {
            return None;
        }
        let mut json = vec![0; u32::from_le_bytes(header[12..16].try_into().ok()?) as usize];
        file.read_exact(&mut json).ok()?;
        let gltf: Value = serde_json::from_slice(&json).ok()?;

        let vrm = gltf.pointer("/extensions/VRMC_vrm")?;
        let nodes = gltf["nodes"].as_array()?;
        let node_name = |index: &Value| -> Option<String> {
            Some(nodes.get(index.as_u64()? as usize)?["name"].as_str()?.to_string())
        };
        let bones = vrm["humanoid"]["humanBones"]
            .as_object()?
            .iter()
            .filter_map(|(bone, spec)| Some((bone.clone(), node_name(&spec["node"])?)))
            .collect();
        let expressions = vrm["expressions"]["preset"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(preset, spec)| {
                let binds = spec["morphTargetBinds"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|bind| {
                        let weight = bind["weight"].as_f64().unwrap_or(1.0) as f32;
                        Some((node_name(&bind["node"])?, bind["index"].as_u64()? as usize, weight))
                    })
                    .collect();
                (preset.clone(), binds)
            })
            .collect();
        Some(Self { bones, expressions })
    }
}

/// The rig resolved to entities, plus the current (smoothed) pose.
#[derive(Component)]
struct Rig {
    bones: HashMap<Bone, RigBone>,
    expressions: HashMap<String, Vec<(Entity, usize, f32)>>,
    pose: HashMap<Bone, Quat>,
}

struct RigBone {
    entity: Entity,
    /// L: rest rotation, local to the parent.
    rest: Quat,
    /// P: the parent's rest rotation in model space.
    parent_rest: Quat,
}

/// What the character is up to, beyond its physics.
#[derive(Component, Default)]
pub struct Animator {
    clock: f32,
    walk_phase: f32,
    gesture: Option<(Gesture, f32)>,
    thinking: bool,
    listening: bool,
    emotion: Option<(String, f32)>,
    speaking: bool,
    blink_in: f32,
    blink_t: Option<f32>,
}

#[derive(Clone, Copy)]
enum Gesture {
    Wave,
    Nod,
    Shrug,
}

impl Gesture {
    fn seconds(self) -> f32 {
        match self {
            Gesture::Wave => 2.4,
            Gesture::Nod => 1.2,
            Gesture::Shrug => 1.4,
        }
    }
}

pub struct AnimationPlugin;

impl Plugin for AnimationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (build_rig, receive_cues, pose_body, animate_face)
                .chain()
                .after(crate::movement::MovementSet),
        );
    }
}

/// Once the scene has spawned (the character has been measured), find the
/// bone and face entities by node name and record the rest pose.
fn build_rig(
    mut commands: Commands,
    characters: Query<(Entity, &Character, &VrmSpec, &GlobalTransform), Without<Rig>>,
    children: Query<&Children>,
    nodes: Query<(&Name, &Transform, Option<&Parent>)>,
    globals: Query<&GlobalTransform>,
) {
    for (root, character, spec, root_global) in &characters {
        if character.bounds.is_none() {
            continue;
        }
        let by_name: HashMap<&str, Entity> = children
            .iter_descendants(root)
            .filter_map(|entity| Some((nodes.get(entity).ok()?.0.as_str(), entity)))
            .collect();
        let to_model = root_global.affine().inverse();

        let mut bones = HashMap::new();
        for (bone, vrm_name) in BONES {
            let Some(&entity) = spec.bones.get(vrm_name).and_then(|n| by_name.get(n.as_str())) else {
                continue;
            };
            let Ok((_, transform, parent)) = nodes.get(entity) else { continue };
            let parent_rest = parent
                .and_then(|p| globals.get(p.get()).ok())
                .map(|g| (to_model * g.affine()).to_scale_rotation_translation().1)
                .unwrap_or(Quat::IDENTITY);
            let rest = transform.rotation;
            bones.insert(bone, RigBone { entity, rest, parent_rest });
        }
        let expressions = spec
            .expressions
            .iter()
            .map(|(preset, binds)| {
                let binds = binds
                    .iter()
                    .filter_map(|(node, index, weight)| Some((*by_name.get(node.as_str())?, *index, *weight)))
                    .collect();
                (preset.clone(), binds)
            })
            .collect();
        info!("rigged {} bones and {} expressions", bones.len(), spec.expressions.len());
        commands.entity(root).insert(Rig {
            bones,
            expressions,
            pose: HashMap::new(),
        });
    }
}

/// Turn the brain's cues into animation state.
#[allow(clippy::too_many_arguments)]
fn receive_cues(
    mut animate: EventReader<AnimateEvent>,
    mut emote: EventReader<EmoteEvent>,
    mut speaking: EventReader<SpeakingEvent>,
    mut thinking: EventReader<ThinkingEvent>,
    mut listening: EventReader<ListeningEvent>,
    mut chat: EventReader<ChatAppendEvent>,
    mut animators: Query<&mut Animator>,
) {
    for mut animator in &mut animators {
        for AnimateEvent(clip) in animate.read() {
            animator.gesture = match clip.as_str() {
                "wave" => Some((Gesture::Wave, 0.0)),
                "nod" => Some((Gesture::Nod, 0.0)),
                "shrug" => Some((Gesture::Shrug, 0.0)),
                // "jump" is physics (movement.rs); "sit" has no pose yet.
                "idle" => None,
                _ => animator.gesture,
            };
        }
        for EmoteEvent(emotion) in emote.read() {
            animator.emotion = EMOTIONS
                .contains(&emotion.as_str())
                .then(|| (emotion.clone(), 0.0));
        }
        for SpeakingEvent(on) in speaking.read() {
            animator.speaking = *on;
        }
        for ThinkingEvent(on) in thinking.read() {
            animator.thinking = *on;
        }
        for ListeningEvent(on) in listening.read() {
            animator.listening = *on;
        }
        for ChatAppendEvent(line) in chat.read() {
            // A little smile when she answers, unless she's showing a feeling.
            if line.role == Role::Assistant && animator.emotion.is_none() {
                animator.emotion = Some(("happy".into(), EMOTION_SECONDS - 3.0));
            }
        }
    }
}

/// Pose the skeleton from the character's state, blending smoothly.
fn pose_body(
    time: Res<Time>,
    mut busy: ResMut<Busy>,
    mut rigs: Query<(&mut Rig, &mut Animator, &Locomotion, &Character)>,
    mut transforms: Query<&mut Transform>,
) {
    let dt = time.delta_secs();
    for (mut rig, mut animator, loco, character) in &mut rigs {
        animator.clock += dt;
        busy.0 |= animator.speaking || animator.gesture.is_some();
        let speed = (loco.vel.x.abs() / WALK_SPEED).min(1.0);
        let walking = loco.stance == Stance::Standing && loco.seat < 0.05 && speed > 0.05;
        if walking {
            animator.walk_phase += loco.vel.x.abs() * dt / (STRIDE * character.height_px) * TAU;
        }
        if let Some((gesture, elapsed)) = &mut animator.gesture {
            *elapsed += dt;
            if *elapsed > gesture.seconds() {
                animator.gesture = None;
            }
        }

        let target = target_pose(&animator, loco, character, walking, speed);
        let k = 1.0 - (-POSE_RATE * dt).exp();
        let Rig { bones, pose, .. } = &mut *rig;
        for (bone, rig_bone) in bones.iter() {
            let goal = target.get(bone).copied().unwrap_or(Quat::IDENTITY);
            let current = pose.entry(*bone).or_insert(Quat::IDENTITY);
            *current = current.slerp(goal, k);
            if let Ok(mut transform) = transforms.get_mut(rig_bone.entity) {
                let p = rig_bone.parent_rest;
                transform.rotation = p.inverse() * *current * p * rig_bone.rest;
            }
        }
    }
}

/// The pose for this moment, as model-space rotations relative to the T-pose:
/// a base pose (breathing, the walk cycle, gestures) with the physics'
/// swinging, sinking limbs on top.
fn target_pose(
    animator: &Animator,
    loco: &Locomotion,
    character: &Character,
    walking: bool,
    speed: f32,
) -> HashMap<Bone, Quat> {
    let t = animator.clock;
    let mut pose = HashMap::new();
    let breath = (t * TAU * 0.25).sin();

    // At rest: arms relaxed at the sides, breathing, a slow idle sway.
    let arms_down = 72.0 + breath * 1.5;
    pose.insert(Bone::LeftUpperArm, rz(-arms_down));
    pose.insert(Bone::RightUpperArm, rz(arms_down));
    pose.insert(Bone::LeftLowerArm, ry(-12.0));
    pose.insert(Bone::RightLowerArm, ry(12.0));
    pose.insert(Bone::Chest, rx(-breath * 1.5));
    pose.insert(Bone::Head, rz((t * 0.4).sin() * 2.5));

    if walking {
        // Legs swing (knee bent while a leg comes forward), arms swing
        // opposite, the torso twists a touch.
        let (s, c) = animator.walk_phase.sin_cos();
        let a = speed;
        pose.insert(Bone::LeftUpperLeg, rx(-s * 28.0 * a));
        pose.insert(Bone::RightUpperLeg, rx(s * 28.0 * a));
        pose.insert(Bone::LeftLowerLeg, rx(c.max(0.0) * 45.0 * a));
        pose.insert(Bone::RightLowerLeg, rx((-c).max(0.0) * 45.0 * a));
        pose.insert(Bone::LeftUpperArm, rx(s * 20.0 * a) * rz(-arms_down));
        pose.insert(Bone::RightUpperArm, rx(-s * 20.0 * a) * rz(arms_down));
        pose.insert(Bone::Spine, ry(s * 5.0 * a));
    }

    // Head: tilted in thought, cocked while listening.
    if animator.thinking {
        pose.insert(Bone::Head, rz(9.0) * rx(-7.0));
    } else if animator.listening {
        pose.insert(Bone::Head, rz(-7.0) * rx(4.0));
    }

    // Sitting on the floor (the body sinks to match, movement.rs): legs out in
    // front, hands on the floor behind, head turned back toward the viewer.
    let blend_in = |pose: &mut HashMap<Bone, Quat>, amount: f32, poses: &[(Bone, Quat)]| {
        for &(bone, q) in poses {
            let base = pose.get(&bone).copied().unwrap_or(Quat::IDENTITY);
            pose.insert(bone, base.slerp(q, amount));
        }
    };
    if loco.seat > 0.001 {
        blend_in(&mut pose, loco.seat, &[
            (Bone::LeftUpperLeg, rx(-88.0)),
            (Bone::RightUpperLeg, rx(-88.0)),
            (Bone::LeftLowerLeg, rx(12.0)),
            (Bone::RightLowerLeg, rx(12.0)),
            (Bone::LeftUpperArm, rx(25.0) * rz(-75.0)),
            (Bone::RightUpperArm, rx(25.0) * rz(75.0)),
            (Bone::Spine, rx(-6.0)),
            (Bone::Head, ry(-character.facing * 55.0)),
        ]);
    }

    // Lying on the floor, on its side facing the viewer, knees drawn up; and
    // pushing itself up (or letting itself down) in between.
    let (lying, pushing) = match loco.stance {
        Stance::Lying => (1.0, 0.0),
        Stance::Rolling { to, t, .. } => (if to == 0.0 { 1.0 - t } else { t }, (t * PI).sin()),
        _ => (0.0, 0.0),
    };
    if lying > 0.0 {
        blend_in(&mut pose, lying, &[
            (Bone::LeftUpperLeg, rx(-30.0)),
            (Bone::RightUpperLeg, rx(-38.0)),
            (Bone::LeftLowerLeg, rx(45.0)),
            (Bone::RightLowerLeg, rx(55.0)),
            (Bone::LeftUpperArm, rx(-25.0) * rz(-70.0)),
            (Bone::RightUpperArm, rx(-25.0) * rz(70.0)),
            (Bone::LeftLowerArm, ry(-50.0)),
            (Bone::RightLowerArm, ry(50.0)),
        ]);
    }
    if pushing > 0.0 {
        blend_in(&mut pose, pushing, &[
            (Bone::LeftUpperLeg, rx(-70.0)),
            (Bone::RightUpperLeg, rx(-70.0)),
            (Bone::LeftLowerLeg, rx(110.0)),
            (Bone::RightLowerLeg, rx(110.0)),
            (Bone::LeftUpperArm, rx(-35.0) * rz(-40.0)),
            (Bone::RightUpperArm, rx(-35.0) * rz(40.0)),
        ]);
    }

    // Gestures the brain asks for, eased in and out over their duration.
    let on_feet = loco.stance == Stance::Standing && loco.seat < 0.05;
    if let (Some((gesture, elapsed)), true) = (animator.gesture, on_feet) {
        let p = (elapsed / gesture.seconds()).clamp(0.0, 1.0);
        let envelope = (p * PI).sin().min(0.35) / 0.35; // quick in, hold, quick out
        let blend = |pose: &mut HashMap<Bone, Quat>, bone: Bone, q: Quat| {
            let base = pose.get(&bone).copied().unwrap_or(Quat::IDENTITY);
            pose.insert(bone, base.slerp(q, envelope));
        };
        match gesture {
            Gesture::Wave => {
                blend(&mut pose, Bone::RightUpperArm, rz(-65.0));
                blend(&mut pose, Bone::RightLowerArm, rz(-50.0 + (elapsed * 12.0).sin() * 25.0));
            }
            Gesture::Nod => {
                blend(&mut pose, Bone::Head, rx((p * TAU * 2.0).sin().max(0.0) * 16.0));
            }
            Gesture::Shrug => {
                blend(&mut pose, Bone::LeftShoulder, rz(12.0));
                blend(&mut pose, Bone::RightShoulder, rz(-12.0));
                blend(&mut pose, Bone::LeftUpperArm, rz(-55.0));
                blend(&mut pose, Bone::RightUpperArm, rz(55.0));
                blend(&mut pose, Bone::LeftLowerArm, ry(-70.0) * rz(35.0));
                blend(&mut pose, Bone::RightLowerArm, ry(70.0) * rz(-35.0));
                blend(&mut pose, Bone::Head, rz(8.0));
            }
        }
    }

    // The physics' limbs (movement.rs). Knees give on landing, while the body
    // sinks to match; then arms, legs and head swing in the screen plane —
    // in model axes, a turn about the axis pointing out of the screen.
    let limbs = &loco.limbs;
    let knees = limbs.crouch.x;
    let mut turn = |bone: Bone, by: Quat| {
        let base = pose.get(&bone).copied().unwrap_or(Quat::IDENTITY);
        pose.insert(bone, by * base);
    };
    turn(Bone::LeftUpperLeg, rx(-50.0 * knees));
    turn(Bone::RightUpperLeg, rx(-50.0 * knees));
    turn(Bone::LeftLowerLeg, rx(95.0 * knees));
    turn(Bone::RightLowerLeg, rx(95.0 * knees));
    let out_of_screen = character.yaw.inverse() * Vec3::Z;
    for (bone, spring) in [
        (Bone::LeftUpperArm, limbs.arms[0]),
        (Bone::RightUpperArm, limbs.arms[1]),
        (Bone::LeftUpperLeg, limbs.legs[0]),
        (Bone::RightUpperLeg, limbs.legs[1]),
        (Bone::Head, limbs.head),
    ] {
        turn(bone, Quat::from_axis_angle(out_of_screen, spring.x));
    }
    pose
}

/// Blinking, emotions and a talking mouth, written to the face's morph targets
/// — plus what the physics does to a face: eyes shut lying on the floor,
/// startled when flung about.
fn animate_face(
    time: Res<Time>,
    mut rigs: Query<(&Rig, &mut Animator, &Locomotion)>,
    mut morphs: Query<&mut MorphWeights>,
) {
    let dt = time.delta_secs();
    for (rig, mut animator, loco) in &mut rigs {
        let t = animator.clock;
        let mut weights: HashMap<&'static str, f32> = HashMap::new();
        if !loco.grounded() {
            let startled = (loco.vel.length() / 2000.0 + loco.spin.abs() / 15.0).min(0.8);
            weights.insert("surprised", startled);
        }

        // Blink every 2–6 s, 0.15 s each.
        animator.blink_in -= dt;
        if animator.blink_in <= 0.0 && animator.blink_t.is_none() {
            animator.blink_t = Some(0.0);
            animator.blink_in = 2.0 + (t * 7.13).sin().abs() * 4.0;
        }
        if let Some(blink_t) = &mut animator.blink_t {
            *blink_t += dt;
            weights.insert("blink", 1.0 - ((*blink_t / 0.075) - 1.0).abs());
            if *blink_t >= 0.15 {
                animator.blink_t = None;
            }
        }

        // The current emotion, faded in and back out.
        if let Some((emotion, age)) = &mut animator.emotion {
            *age += dt;
            let fade_in = (*age / 0.3).min(1.0);
            let fade_out = ((EMOTION_SECONDS - *age) / 0.8).clamp(0.0, 1.0);
            if let Some(preset) = EMOTIONS.iter().find(|preset| **preset == emotion.as_str()) {
                weights.insert(preset, fade_in * fade_out);
            }
        }
        if animator.emotion.as_ref().is_some_and(|(_, age)| *age > EMOTION_SECONDS) {
            animator.emotion = None;
        }

        // Eyes shut while lying down, napping or knocked over.
        if matches!(loco.stance, Stance::Lying | Stance::Rolling { .. }) {
            let shut = weights.get("blink").copied().unwrap_or(0.0).max(1.0);
            weights.insert("blink", shut);
        }

        // Mouth shapes flicker between "aa" and "oh" while the voice plays.
        if animator.speaking {
            weights.insert("aa", ((t * 13.0).sin().max(0.0) * 0.8).min(1.0));
            weights.insert("oh", (t * 7.0 + 1.0).sin().max(0.0) * 0.35);
        }

        // Every driven morph target is recomputed each frame, so released
        // expressions go back to zero. Only changed ones are written: a write
        // re-uploads the face's weights.
        let mut targets: HashMap<(Entity, usize), f32> = HashMap::new();
        for preset in ["blink", "aa", "oh"].into_iter().chain(EMOTIONS) {
            let weight = weights.get(preset).copied().unwrap_or(0.0).max(0.0);
            for &(entity, index, bind) in rig.expressions.get(preset).into_iter().flatten() {
                *targets.entry((entity, index)).or_default() += weight * bind;
            }
        }
        for ((entity, index), weight) in targets {
            let Ok(mut morph) = morphs.get_mut(entity) else { continue };
            let weight = weight.min(1.0);
            if morph.weights().get(index).is_some_and(|w| *w != weight) {
                morph.weights_mut()[index] = weight;
            }
        }
    }
}

fn rx(degrees: f32) -> Quat {
    Quat::from_rotation_x(degrees.to_radians())
}

fn ry(degrees: f32) -> Quat {
    Quat::from_rotation_y(degrees.to_radians())
}

fn rz(degrees: f32) -> Quat {
    Quat::from_rotation_z(degrees.to_radians())
}
