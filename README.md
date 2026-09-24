# Tomo — a local AI VRM desktop companion for Linux and Windows

📘 **Документация на български:** [docs/bg/README.md](docs/bg/README.md)

Tomo is a Desktop-Mate-style companion: a VRoid Studio (`.vrm`) character that
walks around on your desktop, talks with you, and is driven end-to-end by a
language model. The model decides what Tomo says, where it walks, how it
emotes, what it remembers, and which commands it runs for you — quietly — in
the background.

**Everything runs on your computer.** The model is served by
[Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai); Tomo's voice
is [Piper](https://github.com/OHF-Voice/piper1-gpl); "Hey Tomo" and what you
say are recognised by Vosk and Whisper. No cloud services, no API keys, and
nothing you say or show her leaves the machine.

Written in **Rust** (with a little **Python** for speech, **Bash** and
**PowerShell** for installing), split into a "brain" (the model's tool use,
memory, speech) and a "body" (the Bevy app that renders and moves the
character).

> **Where it runs.** Linux — developed and tested on Arch Linux with Hyprland
> (Wayland); other desktops are handled in code but untested — and Windows
> 10/11, which is built and unit-tested on Windows by the project's CI but
> still needs trying on real PCs: see
> **[What's done vs. what needs work](#status)**.

---

## What it does

- 🧍 **VRM character** exported from VRoid Studio, floating above everything on
  your desktop, with clicks passing straight through everywhere except on the
  character (a layer-shell overlay on Hyprland/Sway/KDE; a transparent,
  click-through window on Windows).
- 🚶 **A small rigid-body physics engine** — she walks on the floor (on
  Windows, the taskbar) and never levitates; pick her up by any point (by the
  feet, she hangs upside down), swing her, throw her spinning all the way
  round. Come down wrong and she falls over, lies there a moment, and gets back
  up. She also sits down or takes a nap on her own now and then.
- 🤸 **Animation driven by the physics**: limbs dangle and swing when she's
  carried, float up in a fall, knees give on landing, she leans into a start;
  plus breathing, a walk cycle, blinking, emotions, gestures and a talking
  mouth, all from the model's own VRM rig.
- 💇 **Spring bones**: hair streams behind her when she's thrown and falls
  the other way when she hangs upside down; skirts and ribbons sway with her.
- 🎨 **MToon toon shading**, as the model was made to look: shade colours,
  toon-sharp light and shadow, rim light, matcap, glowing hair highlights and
  outlines.
- 🧠 **A local model decides**, through tool use: speech, movement, emotion,
  memory and command execution are all its choices. Ollama or LM Studio, found
  automatically.
- 👀 **Sees your screen** when it helps ("what's this error?"), with a model
  that can see images — the chat notes every look.
- 🗣 **"Hey Tomo"** — a wake word recognised on the computer (Vosk), with
  your request transcribed there too (Whisper). Or press Talk.
- 🔊 **Her own voice**, Piper neural text-to-speech, on the computer.
  Markdown and emoji are left out of the voice; the 🔊 toggle in the chat
  mutes it.
- 💬 **Liquid chat window**: click Tomo → she walks to the right → a liquid blob
  grows out of her → the blob forms the chat window, with your recent
  conversation in it.
- 🗃 **Long-term memory** in SQLite — preferences and facts the model recalls
  and reuses as context.
- 🖥 **Silent device control**: the model runs commands for you (bash on
  Linux, PowerShell on Windows); you never see a terminal — but every command
  is written to an audit log.
- 🎭 **Your own characters** — pick one from the chat's 👤 menu, import any
  `.vrm` from there, or just ask Tomo to change into another one.
- 📦 **Installers**: `install.sh` on Linux, `Tomo-Setup-<version>.exe` on
  Windows.

---

## Install

### What Tomo thinks with

Tomo needs a model server on the computer. Either:

- **[Ollama](https://ollama.com/download)**, then download a model:
  `ollama pull qwen2.5:7b`. Tomo finds Ollama by itself.
- **[LM Studio](https://lmstudio.ai)**: download a model there and start the
  local server (Developer → Start Server). Tomo finds it too.

Which model: one good at **tool use**. With an 8 GB graphics card (e.g. an
RTX 3070), `qwen2.5:7b` or `llama3.1:8b`; on smaller machines `qwen2.5:3b` or
`llama3.2:3b` (smaller models make more mistakes). For Tomo to look at your
screen, use one that also sees images, such as `qwen2.5vl:7b` or `gemma3`.
Set `TOMO_LLM_MODEL` in `.env` to pick one; otherwise Tomo uses the first local
model the server has.

### Windows

1. Install [Ollama](https://ollama.com/download) (or LM Studio).
2. Run **`Tomo-Setup-<version>.exe`** — from the
   [Releases](https://github.com/NodeVortexGit/Tomo/releases) page, or the
   `Tomo-Setup-windows` artifact of the latest
   [Build](https://github.com/NodeVortexGit/Tomo/actions) run. It installs for
   your user only (no admin rights), and can:
   - set up the voice and "Hey Tomo" — a private Python with Piper, Vosk and
     Whisper, and their models (about 1 GB, downloaded once);
   - download `qwen2.5:7b` for Ollama (about 4.7 GB).
3. Start **Tomo** from the Start menu. Settings: Start menu → **Tomo settings**.

If her background shows black instead of your desktop, start **Tomo
(compatibility)** instead, which draws with OpenGL, or set `TOMO_RENDERER=gl`
(or `dx12`) in the settings.

### Linux

```bash
git clone https://github.com/NodeVortexGit/Tomo.git tomo && cd tomo
./install.sh
```

`install.sh` detects your package manager (apt / dnf / pacman / zypper),
installs the Bevy graphics/audio system libraries, installs Rust via rustup if
needed, creates the Python speech venv and downloads the speech models
(~250 MB), builds the release binary, seeds `.env`, adds a desktop entry, and
checks for Ollama or LM Studio — offering to download a model for Ollama.
Then launch **Tomo** from your app menu. Launching it again while it runs does
nothing: there's only ever one Tomo.

Installer options: `TOMO_SKIP_SYSDEPS=1`, `TOMO_NO_BUILD=1`, `TOMO_PREFIX=...`,
`TOMO_MODEL=...` (the Ollama model to offer).

To check the model and the voice without starting the app:

```bash
cargo run -p tomo-core --example ask -- --speak "say hi"
```

---

## Configuration

Everything lives in `.env` (see [`.env.example`](.env.example) for the
annotated list); the defaults work as they are.

| Key | Default | What |
|---|---|---|
| `TOMO_LLM` | `auto` | `auto` (Ollama, else LM Studio), `ollama`, `lmstudio`, or `openai` (any OpenAI-compatible server) |
| `TOMO_LLM_URL` | the server's usual | e.g. `http://127.0.0.1:11434` or `http://127.0.0.1:1234/v1` |
| `TOMO_LLM_MODEL` | first local model | e.g. `qwen2.5:7b` |
| `TOMO_LLM_CONTEXT` | `8192` | how much the model reads at once, tokens (Ollama) |
| `TOMO_LLM_API_KEY` | — | only if your server asks for one |
| `TOMO_TTS_VOICE` | `en_US-lessac-medium` | any [Piper voice](https://rhasspy.github.io/piper-samples/) |
| `TOMO_TTS_SPEED` | `1.0` | speaking speed |
| `TOMO_WAKE_WORD` | `true` | listen for "Hey Tomo" (off: the mic is used only for Talk) |
| `TOMO_WHISPER_DEVICE` | `cpu` | `cuda` for an NVIDIA card with CUDA 12 + cuDNN 9 |
| `TOMO_PERSONA` | `Tomo` | her name |
| `TOMO_CHARACTER` | the bundled one | a `.vrm` to show |
| `TOMO_OUTPUT` | — | which monitor (Linux, Wayland) |
| `TOMO_RENDERER` | best available | `vulkan`, `dx12` or `gl` |
| `TOMO_ALLOW_COMMANDS` | `true` | the master switch for commands and control |
| `TOMO_ALLOW_SCREEN` | `true` | may she look at the screen |

---

## Architecture

Two crates, split so the brain builds fast and is fully testable without a GPU:

```
tomo/
├── install.sh                # Linux installer (distro-aware)
├── .env.example              # settings, with explanations
├── packaging/windows/        # the Windows installer (Inno Setup) + its scripts
├── .github/workflows/        # CI: tests on Linux and Windows, the installer
├── crates/
│   ├── tomo-core/            # THE BRAIN — no graphics deps, unit-tested
│   │   └── src/
│   │       ├── config.rs     #  loads .env + the platform's folders
│   │       ├── db.rs         #  SQLite memory (prefs, memories, chat, chars)
│   │       ├── characters.rs #  the .vrm models on offer; switching
│   │       ├── apps.rs       #  installed apps (+ Linux system toggles)
│   │       ├── commands.rs   #  audited, deny-listed command executor
│   │       ├── ai.rs         #  the local model's tool-use loop
│   │       ├── screen.rs     #  screenshots for the model to look at
│   │       ├── wake.rs       #  voice input: "Hey Tomo" and Talk
│   │       ├── speech.rs     #  Tomo's voice (Piper)
│   │       ├── platform.rs   #  what differs between Linux and Windows
│   │       ├── events.rs     #  messages exchanged with the body
│   │       └── brain.rs      #  async event loop + channels
│   └── tomo-app/             # THE BODY — Bevy app
│       └── src/
│           ├── main.rs       #  wires the App together
│           ├── session.rs    #  Windows / X11 / Wayland + desktop detection
│           ├── overlay.rs    #  layer-shell overlay above everything (Wayland)
│           ├── window.rs     #  the window elsewhere (Windows, X11, GNOME)
│           ├── character.rs  #  VRM loading + measuring
│           ├── movement.rs   #  physics: body, limbs, drag & throw (tested)
│           ├── animation.rs  #  poses the VRM rig from the physics; face
│           ├── springs.rs    #  VRM spring bones: hair, skirt (tested)
│           ├── mtoon.rs      #  VRM's MToon toon shading (+ mtoon.wgsl)
│           ├── chat.rs       #  click→walk→liquid→chat sequence (egui)
│           ├── input.rs      #  watchable clicks/typing (feature-gated)
│           └── bridge.rs     #  pumps brain⇄Bevy over channels
└── scripts/
    ├── tts_piper.py          # Tomo's voice: text → WAV (Piper)
    ├── wake_word.py          # "Hey Tomo" + request → text (Vosk + Whisper)
    ├── setup_models.py       # downloads the speech models
    └── requirements.txt
```

**Threading model.** The Bevy render loop owns the main thread. The brain runs
on its own thread with a Tokio runtime. They only ever exchange the messages in
`events.rs` over channels, so the render loop never waits on the model, the
database or speech. The speech helpers are separate Python processes, started
once and kept running.

```
        UiToBrain (user typed / talk / import / shutdown)
  ┌──────────────────────────────────────────────────────────┐
  │                                                            ▼
BODY (Bevy, main thread) ◀───────── BrainToUi ───────── BRAIN (Tokio thread)
  render, walk, chat UI     (chat, walk, emote,          model loop, DB,
                             animate, thinking,           commands, speech
                             load-character)
```

---

## Privacy

- **The model runs on your computer** (Ollama / LM Studio). Your messages,
  what Tomo remembers about you and any screenshot she takes go only to it.
- **Voice in and out stay on the computer**: Vosk, Whisper and Piper.
- **Screenshots** are taken only while `TOMO_ALLOW_SCREEN` is on, and every
  look is noted in the chat.
- The only downloads are at install time (the speech models, a model for
  Ollama) and a Piper voice the first time you pick a new one.

---

## Safety model for command execution

The model may run commands silently. "Silent" here means *no terminal in your
face* — **not** hidden from you, the machine's owner:

- **Master switch** — `TOMO_ALLOW_COMMANDS=false` turns execution into
  log-only (records what it *would* have run).
- **Hard deny-list** — irreversible disasters are refused even when the switch
  is on, and can't be re-enabled from `.env`: on Linux `rm -rf /`, `mkfs`,
  `dd` to a block device, fork bombs, piping the internet into a shell; on
  Windows formatting or wiping drives and partitions, `diskpart`, `bcdedit`,
  deleting a whole drive or the Windows folder, deleting shadow copies, and
  running a download straight away (`iwr … | iex`).
- **Full audit log** — every command (run, refused, or errored) is appended to
  `command-audit.log` in Tomo's data folder (`~/.local/share/tomo` on Linux,
  `%APPDATA%\tomo\tomo\data` on Windows), with a timestamp.
- **Timeout** — a hung command can't freeze the assistant.

This is a safety net, not a sandbox. If you want strong isolation, run Tomo
under a dedicated user or inside a container.

---

## Operating your desktop (apps, clicks, toggles)

Tomo reads what your desktop can do straight from the OS and can act on it two
ways, with the model choosing between them:

- **What it knows** — `apps.rs` finds the installed apps: on Linux the
  freedesktop `.desktop` files (system, user, Flatpak), on Windows the Start
  Menu's shortcuts. On Linux it also probes for the tools that drive system
  switches (`rfkill`, `nmcli`, `pactl`/`wpctl`) to offer **toggles**
  (Bluetooth, Wi-Fi, mute); on Windows the model uses PowerShell for those.
  It scans at boot and re-scans on a light interval, refreshing only on change.

- **Clicking (watchable)** — the character walks to a screen spot and drives
  the real cursor/keyboard via `enigo`. Good for quick, one- or two-click
  actions that are fun to watch.

- **Command ("mind control")** — it runs the action instantly with no visible
  clicking.

**Consent is built in.** Real input synthesis is behind the `control` cargo
feature (off by default — without it the app logs what it *would* do). While the
character is driving input a **red "Tomo is controlling the desktop" badge** is
shown, and a **panic hotkey (Ctrl+Alt+Esc, or Pause)** instantly releases
control and tells the brain to stop.

```bash
cargo run --release --features control     # X11: also install `xdotool`
                                           # Wayland: run `ydotoold`
```

To find where to click, `find_on_screen` shows the model a screenshot (noted
in the chat, like every look) and it reads the target's coordinates off it.

---

## <a name="status"></a>What's done vs. what needs work

**Done and unit-tested (`cargo test --workspace`, 72 tests, on Linux and
Windows in CI):**

- ✅ A local model's tool-use loop over Ollama's API or the OpenAI-compatible
  one (LM Studio), finding the server and a local model by itself; tool calls
  written into the text and `<think>` notes are handled; a model that can't
  see images still gets the rest of the conversation
- ✅ Piper voice in a helper that loads it once; replies cleaned of markdown
  and emoji first
- ✅ "Hey Tomo" (Vosk + Whisper) and push-to-talk; the microphone through
  PipeWire/PulseAudio/ALSA on Linux and PortAudio on Windows
- ✅ SQLite memory; audited, deny-listed command executor (Linux and Windows
  rules); screenshots on Linux, Windows and macOS
- ✅ Rigid-body physics, physics-driven animation of the VRM 1.0 rig, spring
  bones, MToon shading
- ✅ Characters: switching from the chat or by asking, importing a `.vrm`
- ✅ Layer-shell overlay on Wayland (Hyprland, Sway, KDE): above all windows,
  click-through except on the character and the open chat
- ✅ Adaptive frame rate: about 13% of one core at rest on Linux
- ✅ Windows: builds, passes the tests, and packs into an installer in CI

**Needs trying on real hardware:**

- 🔧 **Windows desktop behaviour** — transparency, click-through, the window
  over the work area, and the voice setup have been written for Windows but
  not yet seen running on a Windows PC. Transparency depends on the graphics
  driver; if the background shows black, try the compatibility shortcut.
- 🔧 `TODO(input-region)` — click-through for the regular window on X11 and
  GNOME Wayland (the Wayland overlay and Windows already do it).
- 🔧 `TODO(fluid)` — the liquid is merging circles; an SDF/metaball shader
  would be glossier.
- 🔧 VRM 0.x models load but aren't animated; HiDPI (scaled screens) is
  untested; "Hey Tomo" understands English only.
- ⚠️ **GNOME Wayland** ignores always-on-top and has no `wlr-layer-shell`.

### Tests and the Windows installer

```bash
cargo test --workspace              # 72 tests: brain, physics, springs, MToon…
```

Every push runs [`.github/workflows/build.yml`](.github/workflows/build.yml):
clippy and the tests on Linux and Windows, then the Windows release build and
the installer (`packaging/windows/tomo.iss`, Inno Setup), uploaded as the
`Tomo-Setup-windows` artifact. Pushing a tag like `v0.2.0` also attaches the
installer to a GitHub release.

---

## License

MIT — see `LICENSE`.
