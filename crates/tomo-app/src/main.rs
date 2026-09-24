//! Tomo — an AI-driven VRM desktop companion for Linux and Windows.
//!
//! This binary is the "body". It:
//!   • detects the desktop session (Windows; on Linux X11/Wayland, which DE),
//!   • floats above the desktop: a layer-shell overlay where the Wayland
//!     compositor supports it, else a transparent, always-on-top window,
//!   • starts the `tomo-core` brain on its own thread,
//!   • renders the VRM character and runs its physics: walking, falling,
//!     being dragged and thrown around,
//!   • grows the liquid chat window on click.
//!
//! All decisions (speech, movement, expression, memory, commands) come from the
//! brain over channels; this crate only turns those into pixels and motion.

// A Windows release build is a GUI app: no console window next to Tomo.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod animation;
mod bridge;
mod character;
mod chat;
mod input;
mod movement;
mod mtoon;
#[cfg(target_os = "linux")]
mod overlay;
mod session;
mod springs;
mod window;

use bevy::audio::AudioPlugin;
use bevy::core::{TaskPoolOptions, TaskPoolPlugin};
use bevy::ecs::schedule::{ExecutorKind, Schedules};
use bevy::gilrs::GilrsPlugin;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::render::camera::ClearColorConfig;
use bevy::render::RenderApp;
use bevy::app::PluginGroupBuilder;
use bevy_egui::EguiPlugin;

use tomo_core::{Brain, Config};

use crate::animation::AnimationPlugin;
use crate::bridge::{Bridge, BridgePlugin};
use crate::character::CharacterPlugin;
use crate::chat::ChatPlugin;
use crate::input::InputPlugin;
use crate::movement::MovementPlugin;
use crate::mtoon::MToonPlugin;
use crate::session::Session;
use crate::springs::SpringPlugin;
use crate::window::{Busy, InputRegion};

fn main() -> anyhow::Result<()> {
    tomo_core::init_tracing();

    // Where to find `.env` and `scripts/`: TOMO_ROOT (the Linux desktop
    // entry sets it), else the current folder in development, else next to
    // the program (the Windows install).
    let root = std::env::var("TOMO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_root());

    let config = Config::load(&root)?;

    // Which graphics API to draw with, if not the usual: `--renderer gl`, or
    // TOMO_RENDERER in .env. (Some Windows GPU drivers only draw the
    // transparent background right with one of them.)
    let renderer = std::env::args()
        .skip_while(|arg| arg != "--renderer")
        .nth(1)
        .or_else(|| std::env::var("TOMO_RENDERER").ok());
    if let Some(renderer) = renderer.filter(|r| !r.trim().is_empty()) {
        tracing::info!("drawing with {renderer}");
        std::env::set_var("WGPU_BACKEND", renderer.trim());
    }

    // One Tomo at a time (per memory): a second launch, say the autostart
    // entry plus a click in the launcher, would put two on the desktop. The
    // lock lives as long as `instance_lock`; the OS drops it on any exit.
    let instance_lock = std::fs::File::create(config.data_dir.join("tomo.lock"));
    if let Ok(Err(std::fs::TryLockError::WouldBlock)) = instance_lock.as_ref().map(|f| f.try_lock()) {
        tracing::info!("Tomo is already running");
        return Ok(());
    }

    tracing::info!("{}", config.redacted());

    let session = Session::detect();
    window::log_environment_notes(&session);
    std::env::set_var("WINIT_UNIX_BACKEND", window::recommended_backend(&session));

    // Start the brain; hand its channels to Bevy as a resource.
    let brain = Brain::spawn(config)?;
    let bridge = Bridge::new(brain);

    let window_plugin = WindowPlugin {
        primary_window: Some(window::overlay_window(&session)),
        ..default()
    };
    // Gamepads and Bevy's own audio go unused (speech plays through mpv);
    // leaving them out saves their threads. And one small scene doesn't need
    // a compute thread per core: waking them each frame cost more than the
    // work they did. Logging is already set up by `init_tracing` above.
    let plugins = DefaultPlugins
        .set(window_plugin)
        .set(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions::with_num_threads(3),
        })
        .disable::<GilrsPlugin>()
        .disable::<AudioPlugin>()
        .disable::<LogPlugin>();
    let mut app = App::new();
    // On Wayland with layer-shell, not a window at all: overlay.rs drives
    // Bevy in place of winit.
    #[cfg(target_os = "linux")]
    let overlay = match session.server {
        session::DisplayServer::Wayland => overlay::Overlay::connect(),
        _ => None,
    };
    #[cfg(target_os = "linux")]
    if let Some(overlay) = overlay {
        info!("floating above all windows (layer-shell overlay)");
        app.add_plugins(plugins.disable::<bevy::winit::WinitPlugin>())
            .set_runner(move |app| overlay.run(app));
    } else {
        in_a_window(&mut app, plugins);
    }
    #[cfg(not(target_os = "linux"))]
    in_a_window(&mut app, plugins);
    app.add_plugins(EguiPlugin)
        .insert_resource(ClearColor(Color::NONE)) // transparent desktop
        .insert_resource(bridge)
        .init_resource::<InputRegion>()
        .init_resource::<Busy>()
        .add_systems(First, reset_busy)
        .add_plugins((
            BridgePlugin,
            CharacterPlugin,
            MToonPlugin,
            AnimationPlugin,
            MovementPlugin,
            SpringPlugin,
            ChatPlugin,
            InputPlugin,
        ))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, forward_shutdown);
    run_systems_in_line(&mut app);
    app.run();

    Ok(())
}

/// Tomo in a regular window: sized to the screen, lazy at rest, and on
/// Windows letting clicks through around the character.
fn in_a_window(app: &mut App, plugins: PluginGroupBuilder) {
    app.add_plugins(plugins).add_systems(Last, window::pace_frames);
    #[cfg(windows)]
    app.add_systems(Update, (window::windows::fit_to_work_area, window::windows::click_through));
    #[cfg(not(windows))]
    app.add_systems(Update, fit_once);
}

/// Where `.env` and `scripts/` are when TOMO_ROOT doesn't say: the current
/// folder when running from the source tree, else the program's own folder.
fn default_root() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join("scripts").is_dir() || cwd.join(".env").is_file() {
        return cwd;
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
        .unwrap_or(cwd)
}

/// Run each schedule's systems one after another on the thread running it,
/// rather than handing them out to the task pool. One character's worth of
/// work is far smaller than the cost of waking pool threads for it: this
/// about halved the CPU use, at rest (≈23% → 13% of a core) and moving.
fn run_systems_in_line(app: &mut App) {
    fn in_line(world: &mut World) {
        if let Some(mut schedules) = world.get_resource_mut::<Schedules>() {
            for (_, schedule) in schedules.iter_mut() {
                schedule.set_executor_kind(ExecutorKind::SingleThreaded);
            }
        }
    }
    in_line(app.world_mut());
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        in_line(render.world_mut());
    }
}

/// See `setup_scene`: π over the default camera exposure.
const KEY_LIGHT_LUX: f32 = std::f32::consts::PI * 1.2 * 831.8; // 2^9.7 = 831.8

/// Each frame starts idle; whatever moves this frame marks it busy.
fn reset_busy(mut busy: ResMut<Busy>) {
    busy.0 = false;
}

/// Camera + key light. The camera is orthographic with one world unit per
/// logical pixel, so movement.rs can map screen positions straight onto the
/// world, and clears to transparent so only the character (and chat) show.
fn setup_scene(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        Projection::Orthographic(OrthographicProjection::default_3d()),
        // Centred on the screen plane, looking into it (-Z).
        Transform::from_xyz(0.0, 0.0, 500.0),
    ));

    // A three-quarter key light, from the upper right. MToon (mtoon.rs)
    // counts illuminance × exposure = π as full light, the lit side showing
    // its colours as painted: with the camera's default exposure, ~3100 lux.
    commands.spawn((
        DirectionalLight {
            illuminance: KEY_LIGHT_LUX,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Regular-window mode: resize/anchor the window to the primary monitor exactly
/// once, after winit has reported monitor geometry (not available on the first
/// frame).
#[cfg(not(windows))]
fn fit_once(
    mut done: Local<bool>,
    windows: Query<&mut Window, With<bevy::window::PrimaryWindow>>,
    monitors: Query<&bevy::window::Monitor>,
) {
    if *done {
        return;
    }
    if monitors.iter().next().is_some() {
        window::fit_to_primary_monitor(windows, monitors);
        *done = true;
    }
}

/// Tell the brain to shut down cleanly when the app window is closed.
fn forward_shutdown(
    mut closed: EventReader<bevy::window::WindowCloseRequested>,
    bridge: Res<Bridge>,
) {
    if closed.read().next().is_some() {
        bridge.send(tomo_core::UiToBrain::Shutdown);
    }
}
