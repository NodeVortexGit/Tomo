//! Spring bones: hair, skirts and ribbons that swing with the motion.
//!
//! A VRM 1.0 file lists them in its `VRMC_springBone` extension: chains of
//! joints, each with a stiffness (how hard it springs back to its styled
//! shape), a drag and a gravity, plus spheres and capsules on the body that
//! they can't pass through. This is that spec's simulation — the tip of each
//! bone is a Verlet particle held at the bone's length.
//!
//! Each chain moves in the space the file names as its `center` (usually the
//! model's root), so clothes keep to the body however it's thrown about, as
//! their author set them up. Hair is the exception: it moves in world space,
//! so the whole body's motion counts — a throw sends it streaming, and hung
//! upside down it falls the other way.
//!
//! It runs after Bevy has propagated this frame's transforms, so it starts
//! from the pose animation.rs set, and writes the chains' global transforms
//! itself: the swing shows this very frame rather than the next.

use std::collections::HashMap;

use bevy::math::Affine3A;
use bevy::prelude::*;
use bevy::transform::TransformSystem;
use serde_json::Value;

use crate::animation::node_name;
use crate::character::Character;
use crate::window::Busy;

/// The simulation's fixed step, s.
const STEP: f32 = 1.0 / 60.0;
/// At most this many steps a frame; after a hitch the rest is dropped.
const MAX_STEPS: u32 = 4;
/// Extra pull toward the floor on hair, m/s. Models tend to come with none —
/// their springiness alone would hold the hair styled "up" even with the
/// character upside down; this is just enough to win against it.
const HAIR_GRAVITY: f32 = 1.0;
/// A tip moving less than this in a step (screen px) has settled: tips at
/// rest against a collider keep trembling a little, but not visibly.
const SETTLED_PX: f32 = 0.5;
/// The body jumping further than this in a frame (m) is a teleport, not a
/// motion: the chains are put back at rest instead of whipped.
const TELEPORT: f32 = 1.0;

/// The spring bones as the file describes them.
#[derive(Component, Default)]
pub struct SpringSpec {
    chains: Vec<ChainSpec>,
    colliders: Vec<Option<ShapeSpec>>,
}

struct ChainSpec {
    /// Root first; the last joint is only the end of the one before it.
    joints: Vec<JointSpec>,
    /// The node whose space it moves in; `None`: world space.
    center: Option<String>,
    /// Indices into [`SpringSpec::colliders`].
    colliders: Vec<usize>,
    hair: bool,
}

#[derive(Clone)]
struct JointSpec {
    node: String,
    params: Params,
}

#[derive(Clone, Copy)]
struct Params {
    hit_radius: f32,
    stiffness: f32,
    gravity: Vec3,
    drag: f32,
}

struct ShapeSpec {
    node: String,
    offset: Vec3,
    radius: f32,
    /// A capsule's other end; `None` for a sphere.
    tail: Option<Vec3>,
}

impl SpringSpec {
    /// Read the `VRMC_springBone` extension of a .vrm's glTF (none: empty).
    pub fn from_gltf(gltf: &Value) -> Self {
        let Some(ext) = gltf.pointer("/extensions/VRMC_springBone") else {
            return Self::default();
        };
        let list = |v: &Value| v.as_array().cloned().unwrap_or_default();
        let colliders = list(&ext["colliders"])
            .iter()
            .map(|c| {
                let node = node_name(gltf, &c["node"])?;
                let (shape, tail) = match (c["shape"].get("sphere"), c["shape"].get("capsule")) {
                    (Some(sphere), _) => (sphere, None),
                    (None, Some(capsule)) => (capsule, Some(vec3(&capsule["tail"]))),
                    _ => return None,
                };
                Some(ShapeSpec {
                    node,
                    offset: vec3(&shape["offset"]),
                    radius: shape["radius"].as_f64().unwrap_or(0.0) as f32,
                    tail,
                })
            })
            .collect();
        let groups: Vec<Vec<usize>> = list(&ext["colliderGroups"])
            .iter()
            .map(|g| list(&g["colliders"]).iter().filter_map(|i| Some(i.as_u64()? as usize)).collect())
            .collect();
        let chains = list(&ext["springs"])
            .iter()
            .filter_map(|spring| {
                let joints: Vec<JointSpec> = list(&spring["joints"])
                    .iter()
                    .map(|j| {
                        Some(JointSpec {
                            node: node_name(gltf, &j["node"])?,
                            // Defaults as in the spec.
                            params: Params {
                                hit_radius: num(&j["hitRadius"], 0.0),
                                stiffness: num(&j["stiffness"], 1.0),
                                gravity: j.get("gravityDir").map_or(Vec3::NEG_Y, vec3)
                                    * num(&j["gravityPower"], 0.0),
                                drag: num(&j["dragForce"], 0.5).clamp(0.0, 1.0),
                            },
                        })
                    })
                    .collect::<Option<_>>()?;
                let colliders = list(&spring["colliderGroups"])
                    .iter()
                    .filter_map(|g| groups.get(g.as_u64()? as usize))
                    .flatten()
                    .copied()
                    .collect();
                let hair = spring["name"].as_str().unwrap_or_default().to_lowercase().contains("hair")
                    || joints[0].node.to_lowercase().contains("hair");
                let center = spring.get("center").and_then(|c| node_name(gltf, c));
                (joints.len() >= 2).then_some(ChainSpec { joints, center, colliders, hair })
            })
            .collect();
        Self { chains, colliders }
    }
}

fn num(v: &Value, default: f32) -> f32 {
    v.as_f64().map_or(default, |n| n as f32)
}

fn vec3(v: &Value) -> Vec3 {
    let n = |i: usize| v.get(i).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    Vec3::new(n(0), n(1), n(2))
}

/// The spring bones found in the spawned scene, and where their tips are.
#[derive(Component)]
struct SpringRig {
    chains: Vec<Chain>,
    colliders: Vec<Option<Collider>>,
    /// Time not simulated yet, s.
    carry: f32,
    /// Where the body was last frame, to spot teleports; `None` at first.
    last_root: Option<Vec3>,
}

struct Chain {
    /// The body bone the chain hangs from.
    parent: Entity,
    /// What it moves in: a node's space, or (`None`) world space shrunk to
    /// the model's scale. Either way, in the model's units (m).
    center: Option<Entity>,
    joints: Vec<Joint>,
    /// The last joint: carried along, not simulated.
    end: (Entity, Transform),
    colliders: Vec<usize>,
    hair: bool,
}

struct Joint {
    entity: Entity,
    rest: Transform,
    /// Toward the next joint, in this joint's own frame.
    axis: Vec3,
    length: f32,
    params: Params,
    /// The bone's tip now and a step ago, in the chain's space.
    tip: Vec3,
    last_tip: Vec3,
}

struct Collider {
    entity: Entity,
    offset: Vec3,
    radius: f32,
    tail: Option<Vec3>,
}

pub struct SpringPlugin;

impl Plugin for SpringPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, build_springs).add_systems(
            PostUpdate,
            simulate_springs.after(TransformSystem::TransformPropagate),
        );
    }
}

/// Once the scene has spawned (the character has been measured), find the
/// joints and colliders by node name and record the rest shape.
fn build_springs(
    mut commands: Commands,
    characters: Query<(Entity, &Character, &SpringSpec, &GlobalTransform), Without<SpringRig>>,
    children: Query<&Children>,
    nodes: Query<(&Name, &Transform, &GlobalTransform, Option<&Parent>)>,
) {
    for (root, character, spec, root_global) in &characters {
        if character.bounds.is_none() {
            continue;
        }
        let by_name: HashMap<&str, Entity> = children
            .iter_descendants(root)
            .filter_map(|entity| Some((nodes.get(entity).ok()?.0.as_str(), entity)))
            .collect();
        let scale = uniform_scale(root_global);
        let chains = spec
            .chains
            .iter()
            .filter_map(|chain| {
                let entities: Vec<Entity> =
                    chain.joints.iter().map(|j| by_name.get(j.node.as_str()).copied()).collect::<Option<_>>()?;
                let parent = nodes.get(entities[0]).ok()?.3?.get();
                let joints = chain
                    .joints
                    .iter()
                    .zip(entities.windows(2))
                    .map(|(spec, pair)| {
                        let (_, rest, global, _) = nodes.get(pair[0]).ok()?;
                        let (_, next, next_global, _) = nodes.get(pair[1]).ok()?;
                        let length = global.translation().distance(next_global.translation()) / scale;
                        Some(Joint {
                            entity: pair[0],
                            rest: *rest,
                            axis: next.translation.try_normalize()?,
                            length,
                            params: spec.params,
                            tip: Vec3::ZERO,
                            last_tip: Vec3::ZERO,
                        })
                    })
                    .collect::<Option<Vec<_>>>()?;
                let end = *entities.last()?;
                // Hair swings in world space (see the module docs).
                let center = match &chain.center {
                    Some(node) if !chain.hair => Some(*by_name.get(node.as_str())?),
                    _ => None,
                };
                Some(Chain {
                    parent,
                    center,
                    joints,
                    end: (end, *nodes.get(end).ok()?.1),
                    colliders: chain.colliders.clone(),
                    hair: chain.hair,
                })
            })
            .collect::<Vec<_>>();
        let colliders = spec
            .colliders
            .iter()
            .map(|shape| {
                let shape = shape.as_ref()?;
                Some(Collider {
                    entity: *by_name.get(shape.node.as_str())?,
                    offset: shape.offset,
                    radius: shape.radius,
                    tail: shape.tail,
                })
            })
            .collect();
        if !chains.is_empty() {
            info!("{} spring chains", chains.len());
        }
        commands.entity(root).insert(SpringRig { chains, colliders, carry: 0.0, last_root: None });
    }
}

/// Step every chain and pose its bones toward their tips.
fn simulate_springs(
    time: Res<Time>,
    mut busy: ResMut<Busy>,
    mut rigs: Query<(&mut SpringRig, &GlobalTransform)>,
    mut bones: Query<(&mut Transform, &mut GlobalTransform), Without<SpringRig>>,
) {
    for (mut rig, root) in &mut rigs {
        let scale = uniform_scale(root);
        if scale <= 0.0 {
            continue;
        }
        let rig = &mut *rig;
        let here = root.translation() / scale;
        let reset = rig.last_root.is_none_or(|last| last.distance(here) > TELEPORT);
        rig.last_root = Some(here);
        rig.carry = (rig.carry + time.delta_secs()).min(STEP * MAX_STEPS as f32);
        let steps = (rig.carry / STEP) as u32;
        rig.carry -= steps as f32 * STEP;

        // The colliders where the body is now, in world space.
        let shapes: Vec<Option<Shape>> = rig
            .colliders
            .iter()
            .map(|c| {
                let c = c.as_ref()?;
                let global = bones.get(c.entity).ok()?.1;
                Some((global.transform_point(c.offset), c.tail.map(|t| global.transform_point(t)), c.radius))
            })
            .collect();

        let settled = SETTLED_PX / scale;
        let mut moving = false;
        for chain in &mut rig.chains {
            let Ok((_, parent)) = bones.get(chain.parent) else { continue };
            let mut parent = *parent;
            // World → the chain's space, for points and for directions.
            let (space, turn) = match chain.center.and_then(|c| bones.get(c).ok()) {
                Some((_, center)) => {
                    let (_, turn, _) = center.to_scale_rotation_translation();
                    (center.affine().inverse(), turn.inverse())
                }
                None => (Affine3A::from_scale(Vec3::splat(1.0 / scale)), Quat::IDENTITY),
            };
            let shapes: Vec<Shape> = chain
                .colliders
                .iter()
                .filter_map(|&i| shapes.get(i).copied().flatten())
                .map(|(a, b, radius)| {
                    (space.transform_point3(a), b.map(|b| space.transform_point3(b)), radius)
                })
                .collect();
            let gravity_boost = if chain.hair { Vec3::NEG_Y * HAIR_GRAVITY } else { Vec3::ZERO };
            for joint in &mut chain.joints {
                // Where the joint sits, and where its bone would point at rest.
                let (_, parent_turn, _) = parent.to_scale_rotation_translation();
                let rest = parent.mul_transform(joint.rest);
                let head = space.transform_point3(rest.translation());
                let rest_turn = parent_turn * joint.rest.rotation;
                let rest_dir = turn * rest_turn * joint.axis;
                if reset {
                    joint.tip = head + rest_dir * joint.length;
                    joint.last_tip = joint.tip;
                }
                // Gravity is a world direction.
                let gravity = turn * (joint.params.gravity + gravity_boost);
                for _ in 0..steps {
                    moving |= joint.step(head, rest_dir, gravity, &shapes) > settled;
                }
                // Turn the bone from its rest direction to its tip (back in
                // world space).
                let aim = turn.inverse() * Quat::from_rotation_arc(rest_dir, (joint.tip - head).normalize_or(rest_dir)) * turn;
                let local = Transform {
                    rotation: parent_turn.inverse() * aim * rest_turn,
                    ..joint.rest
                };
                let global = parent.mul_transform(local);
                if let Ok((mut transform, mut global_transform)) = bones.get_mut(joint.entity) {
                    transform.rotation = local.rotation;
                    *global_transform = global;
                }
                parent = global;
            }
            let (end, rest) = chain.end;
            if let Ok((_, mut global_transform)) = bones.get_mut(end) {
                *global_transform = parent.mul_transform(rest);
            }
        }
        // Keep drawing at full rate until the swinging has died down.
        busy.0 |= moving;
    }
}

impl Joint {
    /// One step for the bone's tip: carry on moving (less the drag), spring
    /// back toward the rest direction, fall, keep the bone's length and stay
    /// out of the colliders. Returns how far the tip moved.
    fn step(&mut self, head: Vec3, rest_dir: Vec3, gravity: Vec3, shapes: &[Shape]) -> f32 {
        let p = self.params;
        let inertia = (self.tip - self.last_tip) * (1.0 - p.drag);
        let mut next = self.tip + inertia + (rest_dir * p.stiffness + gravity) * STEP;
        next = head + (next - head).normalize_or(rest_dir) * self.length;
        for &shape in shapes {
            next = push_out(next, head, self.length, p.hit_radius, shape);
        }
        self.last_tip = self.tip;
        self.tip = next;
        next.distance(self.last_tip)
    }
}

/// A collider placed for a step: a sphere's centre (or a capsule's two
/// ends) and its radius.
type Shape = (Vec3, Option<Vec3>, f32);

/// Keep a tip out of a collider (a sphere, or a capsule round a segment),
/// then back at the bone's length from its head.
fn push_out(tip: Vec3, head: Vec3, length: f32, hit_radius: f32, (a, b, radius): Shape) -> Vec3 {
    let centre = match b {
        Some(b) => {
            let ab = b - a;
            a + ab * ((tip - a).dot(ab) / ab.length_squared().max(f32::EPSILON)).clamp(0.0, 1.0)
        }
        None => a,
    };
    let reach = radius + hit_radius;
    let away = tip - centre;
    if away.length_squared() >= reach * reach {
        return tip;
    }
    let pushed = centre + away.normalize_or(Vec3::Y) * reach;
    head + (pushed - head).normalize_or(Vec3::Y) * length
}

/// The (uniform) scale of a transform: world units per model unit.
fn uniform_scale(global: &GlobalTransform) -> f32 {
    global.affine().matrix3.x_axis.length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gltf() -> Value {
        json!({
            "nodes": [{ "name": "Head" }, { "name": "Hair1" }, { "name": "Hair2" }, { "name": "Skirt1" }, { "name": "Skirt2" }],
            "extensions": { "VRMC_springBone": {
                "colliders": [
                    { "node": 0, "shape": { "sphere": { "offset": [0, 0.1, 0], "radius": 0.1 } } },
                    { "node": 0, "shape": { "capsule": { "offset": [0, 0, 0], "radius": 0.05, "tail": [0, 0.2, 0] } } },
                    { "node": 0, "shape": { "plane": {} } }
                ],
                "colliderGroups": [{ "colliders": [0, 1] }],
                "springs": [
                    { "name": "Hair", "colliderGroups": [0], "joints": [
                        { "node": 1, "stiffness": 0.8, "dragForce": 0.4, "hitRadius": 0.01 },
                        { "node": 2 }
                    ] },
                    { "name": "Skirt", "joints": [{ "node": 3, "gravityPower": 0.5 }, { "node": 4 }] },
                    { "name": "Lonely", "joints": [{ "node": 3 }] }
                ]
            } }
        })
    }

    #[test]
    fn it_reads_chains_and_colliders() {
        let spec = SpringSpec::from_gltf(&gltf());
        assert_eq!(spec.chains.len(), 2, "a one-joint chain has nothing to swing");
        let hair = &spec.chains[0];
        assert!(hair.hair && !spec.chains[1].hair);
        assert_eq!(hair.joints[0].node, "Hair1");
        assert_eq!(hair.joints[0].params.stiffness, 0.8);
        assert_eq!(hair.joints[1].params.stiffness, 1.0, "the spec's default");
        assert_eq!(hair.colliders, [0, 1]);
        assert_eq!(spec.chains[1].joints[0].params.gravity, Vec3::new(0.0, -0.5, 0.0));
        assert!(spec.colliders[0].as_ref().is_some_and(|c| c.tail.is_none() && c.radius == 0.1));
        assert!(spec.colliders[1].as_ref().is_some_and(|c| c.tail == Some(Vec3::new(0.0, 0.2, 0.0))));
        assert!(spec.colliders[2].is_none(), "unknown shapes are skipped");
    }

    #[test]
    fn a_model_without_springs_has_none() {
        assert!(SpringSpec::from_gltf(&json!({ "nodes": [] })).chains.is_empty());
    }

    fn joint(stiffness: f32, drag: f32, tip: Vec3) -> Joint {
        Joint {
            entity: Entity::PLACEHOLDER,
            rest: Transform::IDENTITY,
            axis: Vec3::NEG_Y,
            length: 0.1,
            params: Params { hit_radius: 0.0, stiffness, gravity: Vec3::ZERO, drag },
            tip,
            last_tip: tip,
        }
    }

    /// Run `seconds` of steps with the head held still.
    fn settle(joint: &mut Joint, rest_dir: Vec3, gravity: Vec3, seconds: f32) {
        for _ in 0..(seconds / STEP) as u32 {
            joint.step(Vec3::ZERO, rest_dir, gravity, &[]);
        }
    }

    #[test]
    fn a_tip_at_rest_stays_put() {
        let mut j = joint(0.8, 0.4, Vec3::NEG_Y * 0.1);
        assert!(j.step(Vec3::ZERO, Vec3::NEG_Y, Vec3::ZERO, &[]) < 1e-6);
    }

    #[test]
    fn a_swung_tip_springs_back_and_settles() {
        // Swung out sideways, it swings back down, and comes to rest there.
        let mut j = joint(0.8, 0.4, Vec3::X * 0.1);
        settle(&mut j, Vec3::NEG_Y, Vec3::ZERO, 0.2);
        assert!(j.tip.x < 0.09, "it swings back: {:?}", j.tip);
        settle(&mut j, Vec3::NEG_Y, Vec3::ZERO, 5.0);
        assert!(j.tip.distance(Vec3::NEG_Y * 0.1) < 0.002, "settled at rest: {:?}", j.tip);
        assert!(j.step(Vec3::ZERO, Vec3::NEG_Y, Vec3::ZERO, &[]) < 1e-4);
        assert!((j.tip.length() - 0.1).abs() < 1e-5, "the bone keeps its length");
    }

    #[test]
    fn upside_down_hair_falls_the_other_way() {
        // Styled to point "down" from a head that is now upside down: its
        // rest direction points up. Hair gravity beats the spring.
        let tilted = Vec3::new(0.01, 0.1, 0.0).normalize() * 0.1;
        let mut j = joint(0.8, 0.4, tilted);
        settle(&mut j, Vec3::Y, Vec3::NEG_Y * HAIR_GRAVITY, 5.0);
        assert!(j.tip.y < -0.05, "hangs down: {:?}", j.tip);
        // Without the extra pull, it keeps its style.
        let mut j = joint(0.8, 0.4, tilted);
        settle(&mut j, Vec3::Y, Vec3::ZERO, 5.0);
        assert!(j.tip.y > 0.09, "stays up: {:?}", j.tip);
    }

    #[test]
    fn tips_are_pushed_out_of_colliders_and_keep_their_length() {
        let head = Vec3::new(0.0, 0.3, 0.0);
        let inside = Vec3::new(0.05, 0.1, 0.0);
        let out = push_out(inside, head, 0.2, 0.01, (Vec3::new(0.0, 0.1, 0.0), None, 0.1));
        assert!((out.distance(head) - 0.2).abs() < 1e-5);
        // Out to the surface (holding the length can leave it a hair inside,
        // as in the spec's own single pass).
        assert!(out.distance(Vec3::new(0.0, 0.1, 0.0)) > 0.095);
        let clear = Vec3::new(0.5, 0.1, 0.0);
        assert_eq!(push_out(clear, head, 0.2, 0.01, (Vec3::ZERO, None, 0.1)), clear);
        // Beside a capsule's middle, pushed straight out from its axis.
        let capsule = (Vec3::ZERO, Some(Vec3::Y * 0.2), 0.05);
        let out = push_out(Vec3::new(0.02, 0.1, 0.0), Vec3::new(1.0, 0.1, 0.0), 0.95, 0.0, capsule);
        assert!(out.distance(Vec3::new(0.05, 0.1, 0.0)) < 1e-5);
    }
}
