//! The desktop overlay — how Tomo floats above everything without being a
//! window.
//!
//! A regular window is still a *window*: the compositor tiles or floats it,
//! frames it, blurs what's behind it and stacks it among the others. Desktop
//! pets need the `wlr-layer-shell` protocol instead (Hyprland, Sway, KDE Plasma
//! and other wlroots compositors — not GNOME): one surface covering the screen
//! on the compositor's *overlay* layer, above every window, with no
//! decorations, and an input region so clicks fall through to the desktop
//! everywhere except on the character (and the chat, while it's open).
//!
//! winit can't create layer surfaces, so when one is available Bevy runs
//! without its winit plugin: this module creates the surface with
//! smithay-client-toolkit, hands its raw handles to the primary [`Window`] (the
//! renderer then draws into it like into any window), and replaces the winit
//! runner with a loop that turns Wayland pointer and keyboard events into
//! Bevy's usual input events — so bevy_egui and `ButtonInput` work unchanged.
//! What the app wants back (input region, keyboard focus) comes through the
//! [`InputRegion`] resource.
//!
//! Known gap: HiDPI. The surface is rendered at its logical size and the
//! compositor upscales it on scaled outputs (soft, but correct).

use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use bevy::app::PluginsState;
use bevy::input::keyboard::{Key, KeyboardFocusLost, KeyboardInput, NativeKey, NativeKeyCode};
use bevy::input::mouse::{MouseButtonInput, MouseScrollUnit, MouseWheel};
use bevy::input::ButtonState;
use bevy::prelude::*;
use bevy::window::{
    CursorEntered, CursorLeft, CursorMoved, PrimaryWindow, RawHandleWrapper, WindowResized,
    WindowWrapper,
};
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle, WindowHandle,
};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, Region},
    delegate_compositor, delegate_keyboard, delegate_layer, delegate_output, delegate_pointer,
    delegate_registry, delegate_seat, delegate_shm,
    output::{OutputHandler, OutputState},
    reexports::{
        calloop::{EventLoop, LoopHandle},
        calloop_wayland_source::WaylandSource,
        client::{
            globals::registry_queue_init,
            protocol::{wl_keyboard, wl_output, wl_pointer, wl_seat, wl_surface},
            Connection, Proxy, QueueHandle,
        },
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers},
        pointer::{
            CursorIcon, PointerEvent, PointerEventKind, PointerHandler, ThemeSpec, ThemedPointer,
        },
        Capability, SeatHandler, SeatState,
    },
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
    shm::{Shm, ShmHandler},
};

use crate::window::{Busy, InputRegion};

/// A connected overlay surface, ready to become Bevy's runner.
pub struct Overlay {
    event_loop: EventLoop<'static, State>,
    state: State,
}

impl Overlay {
    /// Connect to the compositor and create the overlay surface. `None` when
    /// there's no Wayland compositor or it has no layer-shell (e.g. GNOME) —
    /// the caller then falls back to a regular window.
    pub fn connect() -> Option<Self> {
        let conn = Connection::connect_to_env().ok()?;
        let (globals, queue) = registry_queue_init::<State>(&conn).ok()?;
        let qh = queue.handle();
        let Ok(layer_shell) = LayerShell::bind(&globals, &qh) else {
            info!("this compositor has no layer-shell; using a regular window");
            return None;
        };
        let compositor = CompositorState::bind(&globals, &qh).ok()?;
        let shm = Shm::bind(&globals, &qh).ok()?;
        let event_loop = EventLoop::<State>::try_new().ok()?;
        WaylandSource::new(conn.clone(), queue)
            .insert(event_loop.handle())
            .ok()?;

        let surface = compositor.create_surface(&qh);
        let layer =
            layer_shell.create_layer_surface(&qh, surface, Layer::Overlay, Some("tomo"), None);
        // Cover the whole output. An exclusive zone of 0 keeps panels' space
        // free, so the floor is the top of a bottom bar, not hidden behind it.
        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer.set_exclusive_zone(0);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        // Catch no clicks until the character has a shape (see apply_requests).
        let empty = Region::new(&compositor).ok()?;
        layer.wl_surface().set_input_region(Some(empty.wl_region()));
        // The first commit carries no buffer; it asks for a configure (size).
        layer.commit();

        let state = State {
            registry: RegistryState::new(&globals),
            seats: SeatState::new(&globals, &qh),
            outputs: OutputState::new(&globals, &qh),
            loop_handle: event_loop.handle(),
            conn,
            compositor,
            shm,
            layer,
            pointer: None,
            keyboard: None,
            size: None,
            closed: false,
            inputs: Vec::new(),
            applied: InputRegion::default(),
        };
        Some(Overlay { event_loop, state })
    }

    /// Bevy runner: feed Wayland input into the app, update it, repeat. The
    /// renderer's vsync paces the loop.
    pub fn run(mut self, mut app: App) -> AppExit {
        // Finish plugin setup, as Bevy's own runners do.
        if app.plugins_state() != PluginsState::Cleaned {
            while app.plugins_state() == PluginsState::Adding {
                bevy::tasks::tick_global_task_pools_on_main_thread();
            }
            app.finish();
            app.cleanup();
        }

        // Nothing can be drawn before the first configure gives us a size.
        while self.state.size.is_none() {
            if self.state.closed || self.event_loop.dispatch(None, &mut self.state).is_err() {
                error!("the compositor closed the overlay before it was shown");
                return AppExit::error();
            }
        }

        let mut windows = app
            .world_mut()
            .query_filtered::<Entity, With<PrimaryWindow>>();
        let Ok(window) = windows.get_single(app.world()) else {
            error!("no primary window to draw the overlay into");
            return AppExit::error();
        };
        let handle = WindowWrapper::new(SurfaceHandle {
            conn: self.state.conn.clone(),
            layer: self.state.layer.clone(),
        });
        match RawHandleWrapper::new(&handle) {
            Ok(raw) => {
                app.world_mut().entity_mut(window).insert(raw);
            }
            Err(e) => {
                error!("overlay surface has no usable handle: {e}");
                return AppExit::error();
            }
        }

        let mut busy = true;
        let mut frame_start = Instant::now();
        // Frame-rate report for tuning (RUST_LOG=tomo::overlay=debug).
        let (mut report_start, mut frames, mut busy_frames) = (Instant::now(), 0u32, 0u32);
        loop {
            // While something moves, frames follow the display (vsync paces
            // the renderer). At rest, wait out the idle frame interval — but
            // inside the event dispatch, so input still wakes us at once. The
            // socket also wakes for the renderer's own Wayland traffic every
            // frame; that's no reason to draw, so keep waiting then.
            let deadline = frame_start + if busy { Duration::ZERO } else { IDLE_FRAME };
            loop {
                let wait = deadline.saturating_duration_since(Instant::now());
                if let Err(e) = self.event_loop.dispatch(Some(wait), &mut self.state) {
                    error!("lost the Wayland connection: {e}");
                    return AppExit::error();
                }
                if wait.is_zero() || !self.state.inputs.is_empty() || self.state.closed {
                    break;
                }
            }
            frame_start = Instant::now();
            self.state.forward_input(app.world_mut(), window);
            app.update();
            if let Some(exit) = app.should_exit() {
                return exit;
            }
            if self.state.closed {
                return AppExit::Success;
            }
            busy = app.world().get_resource::<Busy>().is_none_or(|b| b.0);
            self.state.apply_requests(app.world());

            frames += 1;
            busy_frames += busy as u32;
            let elapsed = report_start.elapsed();
            if elapsed >= Duration::from_secs(5) {
                let fps = frames as f32 / elapsed.as_secs_f32();
                debug!("{fps:.0} fps, {}% of frames busy", busy_frames * 100 / frames);
                (report_start, frames, busy_frames) = (Instant::now(), 0, 0);
            }
        }
    }
}

/// Frame interval while nothing moves: breathing and blinking still read well
/// at 24 fps, for well under half the power of the display's full rate.
const IDLE_FRAME: Duration = Duration::from_micros(41_667);

/// Wayland input, queued by the handlers and turned into Bevy events by the
/// runner (the handlers can't reach the Bevy world themselves).
enum Input {
    Resized,
    PointerEntered(Vec2),
    PointerMoved(Vec2),
    PointerLeft,
    Button(MouseButton, ButtonState),
    Scroll(Vec2),
    Key {
        key_code: KeyCode,
        logical_key: Key,
        state: ButtonState,
        repeat: bool,
    },
    KeyboardLeft,
}

struct State {
    conn: Connection,
    registry: RegistryState,
    seats: SeatState,
    outputs: OutputState,
    compositor: CompositorState,
    shm: Shm,
    layer: LayerSurface,
    loop_handle: LoopHandle<'static, State>,
    pointer: Option<ThemedPointer>,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    /// Logical size from the compositor's latest configure.
    size: Option<UVec2>,
    /// The compositor closed the surface (e.g. its output went away).
    closed: bool,
    inputs: Vec<Input>,
    /// The input region / keyboard mode last sent to the compositor.
    applied: InputRegion,
}

impl State {
    /// Turn queued Wayland input into Bevy events on `window`.
    fn forward_input(&mut self, world: &mut World, window: Entity) {
        for input in self.inputs.drain(..) {
            match input {
                Input::Resized => {
                    let Some(size) = self.size else { continue };
                    if let Some(mut w) = world.get_mut::<Window>(window) {
                        w.resolution.set_physical_resolution(size.x, size.y);
                    }
                    world.send_event(WindowResized {
                        window,
                        width: size.x as f32,
                        height: size.y as f32,
                    });
                }
                Input::PointerEntered(position) => {
                    world.send_event(CursorEntered { window });
                    move_cursor(world, window, position);
                }
                Input::PointerMoved(position) => move_cursor(world, window, position),
                Input::PointerLeft => {
                    if let Some(mut w) = world.get_mut::<Window>(window) {
                        w.set_cursor_position(None);
                    }
                    world.send_event(CursorLeft { window });
                }
                Input::Button(button, state) => {
                    world.send_event(MouseButtonInput {
                        button,
                        state,
                        window,
                    });
                }
                Input::Scroll(delta) => {
                    world.send_event(MouseWheel {
                        unit: MouseScrollUnit::Pixel,
                        x: delta.x,
                        y: delta.y,
                        window,
                    });
                }
                Input::Key {
                    key_code,
                    logical_key,
                    state,
                    repeat,
                } => {
                    world.send_event(KeyboardInput {
                        key_code,
                        logical_key,
                        state,
                        repeat,
                        window,
                    });
                }
                Input::KeyboardLeft => {
                    world.send_event(KeyboardFocusLost);
                }
            }
        }
    }

    /// Send what the app asked for in [`InputRegion`] to the compositor. Both
    /// are latched by the surface's next commit, which the renderer makes when
    /// it presents the next frame.
    fn apply_requests(&mut self, world: &World) {
        let Some(wanted) = world.get_resource::<InputRegion>() else {
            return;
        };
        if *wanted == self.applied {
            return;
        }
        if (wanted.character, wanted.chat) != (self.applied.character, self.applied.chat) {
            match Region::new(&self.compositor) {
                Ok(region) => {
                    for rect in [wanted.character, wanted.chat].into_iter().flatten() {
                        let size = rect.size();
                        region.add(rect.min.x, rect.min.y, size.x, size.y);
                    }
                    self.layer
                        .wl_surface()
                        .set_input_region(Some(region.wl_region()));
                }
                Err(e) => warn!("couldn't update the overlay's input region: {e}"),
            }
        }
        if wanted.keyboard != self.applied.keyboard {
            // OnDemand: the compositor focuses Tomo when the chat is clicked.
            self.layer.set_keyboard_interactivity(if wanted.keyboard {
                KeyboardInteractivity::OnDemand
            } else {
                KeyboardInteractivity::None
            });
        }
        self.applied = wanted.clone();
        let _ = self.conn.flush();
    }

    fn key(&mut self, event: KeyEvent, state: ButtonState, repeat: bool) {
        self.inputs.push(Input::Key {
            key_code: physical_key(event.raw_code),
            logical_key: logical_key(&event),
            state,
            repeat,
        });
    }
}

fn move_cursor(world: &mut World, window: Entity, position: Vec2) {
    if let Some(mut w) = world.get_mut::<Window>(window) {
        w.set_cursor_position(Some(position));
    }
    world.send_event(CursorMoved {
        window,
        position,
        delta: None,
    });
}

/// Raw handles for the renderer. It keeps a clone of the layer surface, so the
/// `wl_surface` lives at least as long as the GPU surface drawn into it.
struct SurfaceHandle {
    conn: Connection,
    layer: LayerSurface,
}

impl HasWindowHandle for SurfaceHandle {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let surface = NonNull::new(self.layer.wl_surface().id().as_ptr().cast::<c_void>())
            .ok_or(HandleError::Unavailable)?;
        let raw = RawWindowHandle::Wayland(WaylandWindowHandle::new(surface));
        // SAFETY: the surface lives as long as `self.layer` does.
        Ok(unsafe { WindowHandle::borrow_raw(raw) })
    }
}

impl HasDisplayHandle for SurfaceHandle {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        let display = NonNull::new(self.conn.backend().display_ptr().cast::<c_void>())
            .ok_or(HandleError::Unavailable)?;
        let raw = RawDisplayHandle::Wayland(WaylandDisplayHandle::new(display));
        // SAFETY: the display lives as long as `self.conn` does.
        Ok(unsafe { DisplayHandle::borrow_raw(raw) })
    }
}

// ---- Wayland event handlers ------------------------------------------------

impl LayerShellHandler for State {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        self.closed = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _: u32,
    ) {
        // Anchored to all four edges, the compositor always picks the size.
        let (w, h) = configure.new_size;
        let size = UVec2::new(w.max(1), h.max(1));
        if self.size != Some(size) {
            self.size = Some(size);
            self.inputs.push(Input::Resized);
        }
    }
}

impl PointerHandler for State {
    fn pointer_frame(
        &mut self,
        conn: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            if &event.surface != self.layer.wl_surface() {
                continue;
            }
            let position = Vec2::new(event.position.0 as f32, event.position.1 as f32);
            let input = match event.kind {
                PointerEventKind::Enter { .. } => {
                    // A surface must set its own cursor, or it's undefined.
                    if let Some(pointer) = &self.pointer {
                        let _ = pointer.set_cursor(conn, CursorIcon::Default);
                    }
                    Input::PointerEntered(position)
                }
                PointerEventKind::Leave { .. } => Input::PointerLeft,
                PointerEventKind::Motion { .. } => Input::PointerMoved(position),
                PointerEventKind::Press { button, .. } => {
                    Input::Button(mouse_button(button), ButtonState::Pressed)
                }
                PointerEventKind::Release { button, .. } => {
                    Input::Button(mouse_button(button), ButtonState::Released)
                }
                // Wayland scrolls positive down/right; Bevy (like winit) up/left.
                PointerEventKind::Axis {
                    horizontal,
                    vertical,
                    ..
                } => Input::Scroll(Vec2::new(
                    -horizontal.absolute as f32,
                    -vertical.absolute as f32,
                )),
            };
            self.inputs.push(input);
        }
    }
}

impl KeyboardHandler for State {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
        if surface == self.layer.wl_surface() {
            debug!("overlay has keyboard focus");
        }
    }

    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
    ) {
        if surface == self.layer.wl_surface() {
            debug!("overlay lost keyboard focus");
            self.inputs.push(Input::KeyboardLeft);
        }
    }

    fn press_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        self.key(event, ButtonState::Pressed, false);
    }

    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        self.key(event, ButtonState::Released, false);
    }

    // Bevy tracks modifiers from the modifier keys' own press/release events.
    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: Modifiers,
        _: u32,
    ) {
    }
}

impl SeatHandler for State {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seats
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        match capability {
            Capability::Keyboard if self.keyboard.is_none() => {
                // Held keys repeat on the event loop's timer.
                let loop_handle = self.loop_handle.clone();
                let repeat = Box::new(|state: &mut State, _: &wl_keyboard::WlKeyboard, event| {
                    state.key(event, ButtonState::Pressed, true);
                });
                match self
                    .seats
                    .get_keyboard_with_repeat(qh, &seat, None, loop_handle, repeat)
                {
                    Ok(keyboard) => self.keyboard = Some(keyboard),
                    Err(e) => warn!("no keyboard for the overlay: {e}"),
                }
            }
            Capability::Pointer if self.pointer.is_none() => {
                let cursor_surface = self.compositor.create_surface(qh);
                match self.seats.get_pointer_with_theme(
                    qh,
                    &seat,
                    self.shm.wl_shm(),
                    cursor_surface,
                    ThemeSpec::default(),
                ) {
                    Ok(pointer) => self.pointer = Some(pointer),
                    Err(e) => warn!("no pointer for the overlay: {e}"),
                }
            }
            _ => {}
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        match capability {
            Capability::Keyboard => {
                if let Some(keyboard) = self.keyboard.take() {
                    keyboard.release();
                }
            }
            Capability::Pointer => {
                if let Some(pointer) = self.pointer.take() {
                    pointer.pointer().release();
                }
            }
            _ => {}
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

// Required by smithay-client-toolkit; the overlay doesn't need these events.

impl CompositorHandler for State {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }

    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {}

    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for State {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.outputs
    }

    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}

    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}

    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
    }
}

impl ShmHandler for State {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for State {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry
    }
    registry_handlers![OutputState, SeatState];
}

delegate_compositor!(State);
delegate_output!(State);
delegate_shm!(State);
delegate_seat!(State);
delegate_keyboard!(State);
delegate_pointer!(State);
delegate_layer!(State);
delegate_registry!(State);

// ---- Wayland → Bevy input codes --------------------------------------------

/// Linux `BTN_*` codes (linux/input-event-codes.h) → Bevy buttons.
fn mouse_button(code: u32) -> MouseButton {
    match code {
        0x110 => MouseButton::Left,
        0x111 => MouseButton::Right,
        0x112 => MouseButton::Middle,
        0x113 => MouseButton::Back,
        0x114 => MouseButton::Forward,
        other => MouseButton::Other(other as u16),
    }
}

/// Linux `KEY_*` codes → Bevy's physical keys, for the keys Tomo and egui look
/// at (Escape/Pause for the panic hotkey, editing keys for the chat box).
/// Anything else stays identifiable as its raw XKB code.
fn physical_key(code: u32) -> KeyCode {
    match code {
        1 => KeyCode::Escape,
        14 => KeyCode::Backspace,
        15 => KeyCode::Tab,
        28 => KeyCode::Enter,
        29 => KeyCode::ControlLeft,
        42 => KeyCode::ShiftLeft,
        54 => KeyCode::ShiftRight,
        56 => KeyCode::AltLeft,
        57 => KeyCode::Space,
        96 => KeyCode::NumpadEnter,
        97 => KeyCode::ControlRight,
        100 => KeyCode::AltRight,
        102 => KeyCode::Home,
        103 => KeyCode::ArrowUp,
        104 => KeyCode::PageUp,
        105 => KeyCode::ArrowLeft,
        106 => KeyCode::ArrowRight,
        107 => KeyCode::End,
        108 => KeyCode::ArrowDown,
        109 => KeyCode::PageDown,
        110 => KeyCode::Insert,
        111 => KeyCode::Delete,
        119 => KeyCode::Pause,
        125 => KeyCode::SuperLeft,
        126 => KeyCode::SuperRight,
        // XKB keycodes are the evdev code + 8.
        other => KeyCode::Unidentified(NativeKeyCode::Xkb(other + 8)),
    }
}

/// What the key means in the current layout: a named key, or the text it
/// types. With Ctrl held the text is a control character, so fall back to the
/// key's symbol — that's what makes Ctrl+C/V/X reach egui as shortcuts.
fn logical_key(event: &KeyEvent) -> Key {
    match event.keysym {
        Keysym::Return | Keysym::KP_Enter => Key::Enter,
        Keysym::BackSpace => Key::Backspace,
        Keysym::Tab | Keysym::ISO_Left_Tab => Key::Tab,
        Keysym::Escape => Key::Escape,
        Keysym::Delete | Keysym::KP_Delete => Key::Delete,
        Keysym::Insert => Key::Insert,
        Keysym::Home => Key::Home,
        Keysym::End => Key::End,
        Keysym::Page_Up => Key::PageUp,
        Keysym::Page_Down => Key::PageDown,
        Keysym::Left => Key::ArrowLeft,
        Keysym::Right => Key::ArrowRight,
        Keysym::Up => Key::ArrowUp,
        Keysym::Down => Key::ArrowDown,
        Keysym::Shift_L | Keysym::Shift_R => Key::Shift,
        Keysym::Control_L | Keysym::Control_R => Key::Control,
        Keysym::Alt_L | Keysym::Alt_R => Key::Alt,
        Keysym::Super_L | Keysym::Super_R => Key::Super,
        Keysym::Pause => Key::Pause,
        Keysym::space => Key::Space,
        symbol => match &event.utf8 {
            Some(text) if !text.is_empty() && !text.chars().any(char::is_control) => {
                Key::Character(text.as_str().into())
            }
            _ => match symbol.key_char() {
                Some(c) if !c.is_control() => Key::Character(c.to_string().into()),
                _ => Key::Unidentified(NativeKey::Xkb(symbol.raw())),
            },
        },
    }
}
