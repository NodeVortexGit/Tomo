//! Tomo — an AI-driven VRM desktop companion for Linux.
//!
//! This binary is the "body". It:
//!   • detects the desktop session (X11/Wayland, which DE),
//!   • floats above the desktop: a layer-shell overlay where the compositor
//!     supports it, else a transparent, always-on-top window,
//!   • starts the `tomo-core` brain on its own thread,
//!   • renders the VRM character and runs its physics: walking, falling,
//!     being dragged and thrown around,
//!   • grows the liquid chat window on click.
//!
//! All decisions (speech, movement, expression, memory, commands) come from the
//! brain over channels; this crate only turns those into pixels and motion.

mod animation;
mod bridge;
mod character;
mod chat;
mod input;
mod movement;
mod overlay;
mod session;
mod window;

use bevy::audio::AudioPlugin;
use bevy::core::{TaskPoolOptions, TaskPoolPlugin};
use bevy::gilrs::GilrsPlugin;
use bevy::prelude::*;
use bevy::render::camera::ClearColorConfig;
use bevy::window::PrimaryWindow;
use bevy::winit::WinitPlugin;
use bevy_egui::EguiPlugin;

use tomo_core::{Brain, Config};

use crate::animation::AnimationPlugin;
use crate::bridge::{Bridge, BridgePlugin};
use crate::character::CharacterPlugin;
use crate::chat::ChatPlugin;
use crate::input::InputPlugin;
use crate::movement::MovementPlugin;
use crate::overlay::Overlay;
use crate::session::{DisplayServer, Session};
use crate::window::{Busy, InputRegion};

fn main() -> anyhow::Result<()> {
    tomo_core::init_tracing();

    // Where to find `.env` and `scripts/`. Overridable so the installed
    // desktop entry can point at the install prefix.
    let root = std::env::var("TOMO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default());

    let config = Config::load(&root)?;
    tracing::info!("{}", config.redacted());
    if !config.ai_ready() {
        tracing::warn!("no ANTHROPIC_API_KEY set — Tomo will run but can't think. Edit .env.");
    }

    let session = Session::detect();
    window::log_environment_notes(&session);
    std::env::set_var("WINIT_UNIX_BACKEND", window::recommended_backend(&session));

    // Start the brain; hand its channels to Bevy as a resource.
    let brain = Brain::spawn(config)?;
    let bridge = Bridge::new(brain);

    // Absolute-path VRMs imported by the user are copied into the data dir by
    // the brain; TODO(assets) in character.rs explains registering that dir as
    // an asset source for clean loading.
    let window_plugin = WindowPlugin {
        primary_window: Some(window::overlay_window(&session)),
        ..default()
    };
    // Gamepads and Bevy's own audio go unused (speech plays through mpv);
    // leaving them out saves their threads. And one small scene doesn't need
    // a compute thread per core: waking them each frame cost more than the
    // work they did.
    let plugins = DefaultPlugins
        .set(window_plugin)
        .set(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions::with_num_threads(3),
        })
        .disable::<GilrsPlugin>()
        .disable::<AudioPlugin>();
    let overlay = match session.server {
        DisplayServer::Wayland => Overlay::connect(),
        _ => None,
    };
    let mut app = App::new();
    match overlay {
        // Not a window at all: overlay.rs drives Bevy in place of winit.
        Some(overlay) => {
            info!("floating above all windows (layer-shell overlay)");
            app.add_plugins(plugins.disable::<WinitPlugin>())
                .set_runner(move |app| overlay.run(app));
        }
        None => {
            app.add_plugins(plugins).add_systems(Update, fit_once);
        }
    }
    app.add_plugins(EguiPlugin)
        .insert_resource(ClearColor(Color::NONE)) // transparent desktop
        .insert_resource(bridge)
        .init_resource::<InputRegion>()
        .init_resource::<Busy>()
        .add_systems(First, reset_busy)
        .add_plugins((
            BridgePlugin,
            CharacterPlugin,
            AnimationPlugin,
            MovementPlugin,
            ChatPlugin,
            InputPlugin,
        ))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, forward_shutdown)
        .run();

    Ok(())
}

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

    // Soft three-quarter key light so the model isn't flat.
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Regular-window mode: resize/anchor the window to the primary monitor exactly
/// once, after winit has reported monitor geometry (not available on the first
/// frame).
fn fit_once(
    mut done: Local<bool>,
    windows: Query<&mut Window, With<PrimaryWindow>>,
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
