# Tomo — an AI-driven VRM desktop companion for Linux

📘 **Документация на български:** [docs/bg/README.md](docs/bg/README.md)

Tomo is a Desktop-Mate-style companion: a VRoid Studio (`.vrm`) character that
walks around on your Linux desktop, talks with you, and is driven end-to-end by
an LLM. The AI decides what Tomo says, where it walks, how it emotes, what it
remembers, and which shell commands it runs for you — quietly — in the
background.

Written in **Rust** (with a little **Bash** and **Python** for device control
and speech), split into a "brain" (AI, memory, speech) and a "body" (the Bevy
app that renders and moves the character).

> **Where it runs.** Tomo is developed and tested on Arch Linux with Hyprland
> (Wayland). The other desktops it knows about — KDE, Sway, GNOME, XFCE,
> Cinnamon, i3, X11 in general — are handled in code but untested. See
> **[What's done vs. what needs work](#status)**.

---

## What it does

- 🧍 **VRM character** exported from VRoid Studio, floating above everything on
  your desktop — not a window (a layer-shell overlay on Hyprland/Sway/KDE), and
  clicks pass straight through everywhere except on the character.
- 🚶 **A small rigid-body physics engine** — she walks on the floor and never
  levitates; pick her up by any point (by the feet, she hangs upside down),
  swing her, throw her spinning all the way round. Come down wrong and she
  falls over, lies there a moment, and gets back up. She also sits down or
  takes a nap on her own now and then.
- 🤸 **Animation driven by the physics**: limbs dangle and swing when she's
  carried, float up in a fall, knees give on landing, she leans into a start;
  plus breathing, a walk cycle, blinking, emotions, gestures and a talking
  mouth, all from the model's own VRM rig.
- 💇 **Spring bones**: hair streams behind her when she's thrown and falls
  the other way when she hangs upside down; skirts and ribbons sway with her
  (the model's own `VRMC_springBone` setup, colliders included).
- 🎨 **MToon toon shading**, as the model was made to look: shade colours,
  toon-sharp light and shadow, rim light, matcap, glowing hair highlights and
  outlines (`VRMC_materials_mtoon`).
- 🧠 **Claude-powered decisions** via tool use (Anthropic Messages API):
  speech, movement, emotion, memory, and command execution are all model
  choices.
- 👀 **Sees your screen** when it helps ("what's this error?") — the chat
  notes every look.
- 🗣 **"Hey Tomo"** — an always-on wake word, recognised offline (Vosk), with
  your request transcribed offline too (Whisper). No audio leaves the machine.
- 💬 **Liquid chat window**: click Tomo → it walks to the right → a liquid blob
  grows out of it → the blob forms the chat window, with your recent
  conversation in it.
- 🔊 **Text-to-speech** with Edge-TTS (free neural voices) — markdown and
  emoji are left out of the voice; the 🔊 toggle in the chat mutes it.
- 🗃 **Long-term memory** in SQLite — preferences and facts the AI recalls and
  reuses as context.
- 🖥 **Silent device control**: the AI runs shell commands for you; you never
  see a terminal — but every command is written to an audit log.
- 📦 **One `install.sh`** and all secrets in a `.env` file.
- 🎭 **Your own characters** — pick one from the chat's 👤 menu, import any
  `.vrm` from there, or just ask Tomo to change into another one. The choice
  sticks.

---

## Architecture

Two crates, split so the brain builds fast and is fully testable without a GPU:

```
tomo/
├── install.sh                # one-shot installer (distro-aware)
├── .env.example              # copy to .env, add your keys
├── crates/
│   ├── tomo-core/            # THE BRAIN — no graphics deps, unit-tested
│   │   └── src/
│   │       ├── config.rs     #  loads .env + XDG paths
│   │       ├── db.rs         #  SQLite memory (prefs, memories, chat, chars)
│   │       ├── characters.rs #  the .vrm models on offer; switching
│   │       ├── apps.rs       #  installed apps + system toggles, from the OS
│   │       ├── commands.rs   #  audited, deny-listed shell executor
│   │       ├── ai.rs         #  Claude tool-use loop (Messages API)
│   │       ├── screen.rs     #  screenshots for the model to look at
│   │       ├── wake.rs       #  runs the offline "Hey Tomo" listener
│   │       ├── speech.rs     #  Edge-TTS + Google-STT bridge
│   │       ├── events.rs     #  messages exchanged with the body
│   │       └── brain.rs      #  async event loop + channels
│   └── tomo-app/             # THE BODY — Bevy app
│       └── src/
│           ├── main.rs       #  wires the App together
│           ├── session.rs    #  X11/Wayland + DE detection (done, tested)
│           ├── overlay.rs    #  layer-shell overlay above everything (Wayland)
│           ├── window.rs     #  regular-window fallback (X11, GNOME)
│           ├── character.rs  #  VRM loading + measuring
│           ├── movement.rs   #  physics: body, limbs, drag & throw (tested)
│           ├── animation.rs  #  poses the VRM rig from the physics; face
│           ├── springs.rs    #  VRM spring bones: hair, skirt (tested)
│           ├── mtoon.rs      #  VRM's MToon toon shading (+ mtoon.wgsl)
│           ├── chat.rs       #  click→walk→liquid→chat sequence (egui)
│           ├── input.rs      #  watchable clicks/typing (feature-gated)
│           └── bridge.rs     #  pumps brain⇄Bevy over channels
└── scripts/
    ├── tts_edge.py           # text → mp3 (Edge neural TTS)
    ├── wake_word.py          # "Hey Tomo" + request → text (Vosk + Whisper)
    ├── stt_google.py         # wav → text (Google STT, REST + API key)
    └── requirements.txt
```

**Threading model.** The Bevy render loop owns the main thread. The brain runs
on its own thread with a Tokio runtime. They only ever exchange the messages in
`events.rs` over channels, so the render loop never blocks on the network, the
model, the database, or TTS.

```
        UiToBrain (user typed / mic / import / shutdown)
  ┌──────────────────────────────────────────────────────────┐
  │                                                            ▼
BODY (Bevy, main thread) ◀───────── BrainToUi ───────── BRAIN (Tokio thread)
  render, walk, chat UI     (chat, walk, emote,          AI loop, DB,
                             animate, thinking,           commands, speech
                             load-character)
```

---

## Install

```bash
git clone <your-repo> tomo && cd tomo
./install.sh
```

`install.sh` detects your package manager (apt / dnf / pacman / zypper),
installs the Bevy graphics/audio system libraries, installs Rust via rustup if
needed, creates the Python speech venv, downloads the speech models (~190 MB),
builds the release binary, seeds `.env`, and adds a desktop entry.

Then:

1. Edit `.env` and add your `ANTHROPIC_API_KEY`.
2. Put a `.vrm` at `~/.local/share/tomo/characters/default.vrm` (or set
   `TOMO_CHARACTER`, or import one from the chat's 👤 menu).
3. Launch **Tomo** from your app menu. Launching it again while it runs does
   nothing: there's only ever one Tomo.

Installer options: `TOMO_SKIP_SYSDEPS=1`, `TOMO_NO_BUILD=1`, `TOMO_PREFIX=...`.

---

## Configuration

Everything lives in `.env` (see `.env.example` for the annotated list). Keys:
`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL`, `TOMO_EFFORT`,
`EDGE_TTS_VOICE`, `EDGE_TTS_RATE`, `TOMO_WAKE_WORD`, `GOOGLE_STT_API_KEY`,
`STT_LANGUAGE`, `TOMO_PERSONA`, `TOMO_CHARACTER`, `TOMO_OUTPUT` (which
monitor), `TOMO_ALLOW_COMMANDS`, `TOMO_ALLOW_SCREEN`.

**Tomo uses Claude Sonnet 5** through Anthropic's Messages API, at low effort
by default so replies come quickly (`TOMO_EFFORT` raises it). Older `.env`
files with `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` keep working.
To check the key, model and tools without starting the app:

```bash
cargo run -p tomo-core --example ask -- "what's on my screen?"
```

---

## Safety model for command execution

The brief asks for the AI to run commands silently. "Silent" here means *no
terminal in your face* — **not** hidden from you, the machine's owner:

- **Master switch** — `TOMO_ALLOW_COMMANDS=false` turns execution into
  log-only (records what it *would* have run).
- **Hard deny-list** — irreversible disasters (`rm -rf /`, `mkfs`, `dd` to a
  block device, fork bombs, piping the internet into a shell) are refused even
  when the switch is on, and can't be re-enabled from `.env`.
- **Full audit log** — every command (run, refused, or errored) is appended to
  `~/.local/share/tomo/command-audit.log` with a timestamp. `tail -f` it any
  time.
- **Timeout** — a hung command can't freeze the assistant.

This is a safety net, not a sandbox. If you want strong isolation, run Tomo
under a dedicated user or inside a container.

---

## Operating your desktop (apps, clicks, toggles)

Tomo reads what your desktop can do straight from the OS and can act on it two
ways, with the model choosing between them:

- **What it knows** — `apps.rs` parses the freedesktop `.desktop` files in the
  standard XDG locations (system, user, Flatpak) into a searchable app list,
  and probes for the tools that drive system switches (`rfkill`, `nmcli`,
  `pactl`/`wpctl`) to build the available **toggles** (Bluetooth, Wi-Fi, mute).
  It scans once at boot and re-scans on a light interval, refreshing only when
  the fingerprint changes — so newly installed/removed apps are picked up.

- **Clicking (watchable)** — the character walks to a screen spot and drives
  the real cursor/keyboard via `enigo` (same idea as `xdotool`/`ydotool`). Good
  for quick, one- or two-click actions that are fun to watch.

- **Command ("mind control")** — it runs the action instantly with no visible
  clicking. Good when clicking would be many fiddly steps, or for toggles.

The model picks per the policy in `ai.rs`: *few clicks and fun to watch → click;
many steps or fiddly → command; system toggles → command.*

**Consent is built in.** Real input synthesis is behind the `control` cargo
feature (off by default — without it the app logs what it *would* do). While the
character is driving input a **red "Tomo is controlling the desktop" badge** is
shown, and a **panic hotkey (Ctrl+Alt+Esc, or Pause)** instantly releases
control and tells the brain to stop. `TOMO_ALLOW_COMMANDS=false` disables both
commands and input from the start.

Enable real control:

```bash
cargo run --release --features control     # X11: also install `xdotool`
                                           # Wayland: run `ydotoold`
```

To find where to click, `find_on_screen` shows the model a screenshot (noted
in the chat, like every look) and it reads the target's coordinates off it.
It's only offered while screen viewing is allowed (`TOMO_ALLOW_SCREEN`).

---

## <a name="status"></a>What's done vs. what needs work

**Done and unit-tested (`cargo test`):**

- ✅ Config / `.env` loading
- ✅ SQLite memory (prefs, memories, transcript, characters)
- ✅ Claude tool-use loop over the Messages API (execute / remember / recall /
  walk / express / animate / change character / look at or find on the
  screen), with prompt caching and retries
- ✅ Screenshots for the model (grim / spectacle / gnome-screenshot / maim /
  scrot / import), scaled to 1080p-class JPEG
- ✅ Offline "Hey Tomo": Vosk wake phrase + Whisper transcription, muted while
  Tomo speaks; the Talk button is push-to-talk through it
- ✅ Audited, deny-listed command executor
- ✅ Edge-TTS + Google-STT bridge (Rust side + Python helpers); replies are
  cleaned of markdown and emoji before they're spoken
- ✅ Session detection (X11/Wayland, KDE/GNOME/XFCE/Cinnamon/Hyprland/i3/Sway)
- ✅ Rigid-body physics: free rotation, capsule collisions with bounce and
  friction, hanging from the grab point, throws with spin, landing on its feet
  or falling over; standing, walking, sitting, lying, getting up; idle
  strolling, sitting and napping; limbs riding along (dangling, swinging,
  landing squat, lean)
- ✅ Procedural animation of the VRM 1.0 rig from the physics, plus blinking,
  emotions, gestures (wave / nod / shrug) and a mouth synced to the voice
- ✅ VRM 1.0 spring bones (sphere and capsule colliders), hair in world space
- ✅ MToon 1.0 shading: shade colour and texture, shading shift and toony,
  ambient, matcap, parametric rim, emission, UV animation, render-queue order
  for transparent parts, and outlines (world- or screen-sized, kept at least
  about a pixel wide at desktop-pet size)
- ✅ Characters: the models on offer, switching from the chat or by asking,
  importing a `.vrm` through the desktop's file dialog; remembered
- ✅ Adaptive frame rate (full rate while anything moves, 24 fps at rest) and
  a lean scheduler: about 13% of one core at rest (was ~67% at all times)
- ✅ One instance at a time; the chat reopens on the recent conversation
- ✅ Layer-shell overlay (Hyprland, Sway, KDE): above all windows, click-through
  except on the character and the open chat
- ✅ Liquid-chat state machine and egui UI
- ✅ OS catalog: `.desktop` app scan + toggle detection, cached & diffed (scan
  at boot + on change)
- ✅ Model-driven click-vs-command policy; input synthesis (feature-gated) with
  a visible control badge and a panic hotkey
- ✅ `install.sh`

**Needs iteration on real hardware (marked with `TODO(...)` in code):**

- 🔧 `TODO(input-region)` — click-through for the regular-window fallback (X11,
  GNOME Wayland): winit only offers whole-window hit-test, so this needs an X11
  XShape input region or `wl_surface.set_input_region` on winit's surface. The
  layer-shell overlay already does it. Isolated to `window.rs`.
- 🔧 `TODO(fluid)` — the liquid is currently merging circles (reads well, no
  shader). Drop in an SDF/metaball shader for a glossier effect.
- 🔧 VRM 0.x models load and render but aren't animated (their rig is laid out
  differently; `animation.rs` reads VRM 1.0), and face away from the viewer.
- 🔧 HiDPI (scaled outputs) is untested; screen coordinates assume scale 1.
- ⚠️ **GNOME Wayland** ignores always-on-top and has no `wlr-layer-shell`;
  wlroots compositors (Hyprland, Sway) and all X11 setups behave correctly.
  `window.rs` logs a clear note at startup.

### Tests

```bash
cargo test -p tomo-core             # the brain
cargo test --release -p tomo-app    # physics, spring bones, MToon, session detection
```

---

## License

MIT — see `LICENSE`.
