#!/usr/bin/env bash
# =============================================================================
#  Tomo — one-shot installer
#
#  Installs everything Tomo needs and builds it:
#    • system libraries for Bevy (graphics/X11/Wayland/Vulkan) + audio I/O
#    • the Rust toolchain (via rustup) if it's missing
#    • a Python virtualenv with edge-tts for text-to-speech, and Vosk + Whisper
#      (plus their models) for the offline "Hey Tomo" wake word
#    • the release binary, a .env from the template, and a desktop entry
#
#  Supported package managers: apt, dnf, pacman, zypper (Debian/Ubuntu/Mint/Pop,
#  Fedora/RHEL, Arch/Manjaro/EndeavourOS, openSUSE). Other distros: install the
#  equivalents of the package list printed at the top and re-run with
#  TOMO_SKIP_SYSDEPS=1.
#
#  Usage:   ./install.sh
#  Options (env vars):
#     TOMO_SKIP_SYSDEPS=1   don't touch system packages (you installed them)
#     TOMO_NO_BUILD=1       set up deps + venv but skip `cargo build`
#     TOMO_PREFIX=~/.local  where to install the binary (default ~/.local)
# =============================================================================
set -euo pipefail

# ---- pretty output ----------------------------------------------------------
c_reset='\033[0m'; c_bold='\033[1m'; c_grn='\033[32m'; c_yel='\033[33m'; c_red='\033[31m'; c_cya='\033[36m'
say()  { printf "${c_cya}▸${c_reset} %s\n" "$*"; }
ok()   { printf "${c_grn}✓${c_reset} %s\n" "$*"; }
warn() { printf "${c_yel}!${c_reset} %s\n" "$*"; }
die()  { printf "${c_red}✗ %s${c_reset}\n" "$*" >&2; exit 1; }
hr()   { printf "${c_bold}%s${c_reset}\n" "────────────────────────────────────────────────────────"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${TOMO_PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
APPS_DIR="$HOME/.local/share/applications"
DATA_DIR="${TOMO_DATA_DIR:-$HOME/.local/share/tomo}"

hr
printf "${c_bold}  Tomo installer${c_reset}  —  AI VRM desktop companion\n"
hr

# ---- 0. .env FIRST ----------------------------------------------------------
# Done before anything that could fail, so you always end up with a .env to
# edit even if a later step (packages, rustup, pip) errors out.
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    ok "Created .env from template → $SCRIPT_DIR/.env"
    warn "Edit it and add your ANTHROPIC_API_KEY before launching."
else
    ok ".env already exists (left untouched)."
fi

# ---- 1. detect the package manager -----------------------------------------
PM=""
for candidate in apt-get dnf pacman zypper; do
    if command -v "$candidate" >/dev/null 2>&1; then PM="$candidate"; break; fi
done
[ -n "$PM" ] || warn "No supported package manager found; will skip system deps."
[ -n "$PM" ] && say "Package manager: $PM"

# ---- 2. system dependencies -------------------------------------------------
install_sysdeps() {
    [ -n "${TOMO_SKIP_SYSDEPS:-}" ] && { warn "Skipping system deps (TOMO_SKIP_SYSDEPS set)."; return; }
    [ -n "$PM" ] || return

    say "Installing system dependencies (may prompt for sudo)…"
    case "$PM" in
      apt-get)
        sudo apt-get update -y
        sudo apt-get install -y \
          build-essential pkg-config curl clang \
          libasound2-dev libudev-dev \
          libx11-dev libxcursor-dev libxrandr-dev libxi-dev libxkbcommon-dev \
          libwayland-dev \
          libvulkan1 mesa-vulkan-drivers vulkan-tools \
          python3 python3-venv python3-pip \
          mpv pulseaudio-utils alsa-utils grim
        ;;
      dnf)
        sudo dnf install -y \
          gcc gcc-c++ pkgconf-pkg-config curl clang \
          alsa-lib-devel systemd-devel \
          libX11-devel libXcursor-devel libXrandr-devel libXi-devel libxkbcommon-devel \
          wayland-devel \
          vulkan-loader mesa-vulkan-drivers vulkan-tools \
          python3 python3-pip \
          mpv pulseaudio-utils alsa-utils grim
        ;;
      pacman)
        sudo pacman -Sy --needed --noconfirm \
          base-devel pkgconf curl clang \
          alsa-lib systemd-libs \
          libx11 libxcursor libxrandr libxi libxkbcommon \
          wayland \
          vulkan-icd-loader vulkan-tools \
          python python-pip \
          mpv libpulse alsa-utils grim
        ;;
      zypper)
        sudo zypper --non-interactive install -y \
          gcc gcc-c++ pkg-config curl clang \
          alsa-devel systemd-devel \
          libX11-devel libXcursor-devel libXrandr-devel libXi-devel libxkbcommon-devel \
          wayland-devel \
          vulkan-loader vulkan-tools \
          python3 python3-pip \
          mpv pulseaudio-utils alsa-utils grim
        ;;
    esac
    ok "System dependencies installed."
}
install_sysdeps || warn "Some system packages failed to install — continuing anyway; if the build later complains about a missing library, install it from the list above and re-run."

# ---- 3. Rust toolchain ------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
    say "Rust not found — installing via rustup…"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
command -v cargo >/dev/null 2>&1 || die "cargo still not on PATH; open a new shell and re-run."
ok "Rust toolchain: $(cargo --version)"

# ---- 4. Python venv + speech deps ------------------------------------------
say "Setting up Python virtualenv for speech…"
VENV="$SCRIPT_DIR/scripts/.venv"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
"$VENV/bin/pip" install --upgrade pip >/dev/null
"$VENV/bin/pip" install -r "$SCRIPT_DIR/scripts/requirements.txt"
ok "Speech venv ready at scripts/.venv"

# ---- 5. speech models for "Hey Tomo" (offline) ------------------------------
MODELS="$DATA_DIR/models"
mkdir -p "$MODELS"
VOSK="vosk-model-small-en-us-0.15"
if [ ! -d "$MODELS/$VOSK" ]; then
    say "Downloading the wake-word model (Vosk, ~40 MB)…"
    curl -fL --progress-bar -o "$MODELS/$VOSK.zip" "https://alphacephei.com/vosk/models/$VOSK.zip"
    # Python's zipfile rather than `unzip`, which many systems lack.
    "$VENV/bin/python" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$MODELS/$VOSK.zip" "$MODELS"
    rm -f "$MODELS/$VOSK.zip"
fi
if [ ! -d "$MODELS/whisper-base.en" ]; then
    say "Downloading the transcription model (Whisper base.en, ~145 MB)…"
    "$VENV/bin/python" -c "import sys; from faster_whisper import download_model; download_model('base.en', output_dir=sys.argv[1])" \
        "$MODELS/whisper-base.en"
fi
ok "Speech models ready in $MODELS"

# (.env was already created at step 0, before anything that could fail.)

# ---- 6. build ---------------------------------------------------------------
if [ -n "${TOMO_NO_BUILD:-}" ]; then
    warn "Skipping build (TOMO_NO_BUILD set)."
else
    say "Building Tomo in release mode (first build downloads Bevy — grab a coffee)…"
    ( cd "$SCRIPT_DIR" && cargo build --release )
    ok "Build complete."

    # ---- 7. install binary + desktop entry ---------------------------------
    mkdir -p "$BIN_DIR" "$APPS_DIR" "$DATA_DIR/characters"
    install -m 0755 "$SCRIPT_DIR/target/release/tomo" "$BIN_DIR/tomo"
    ok "Installed binary → $BIN_DIR/tomo"

    cat > "$APPS_DIR/tomo.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Tomo
Comment=AI-driven VRM desktop companion
Exec=env TOMO_ROOT=$SCRIPT_DIR $BIN_DIR/tomo
Icon=$SCRIPT_DIR/assets/icon.png
Terminal=false
Categories=Utility;
X-GNOME-Autostart-enabled=true
EOF
    ok "Installed desktop entry → $APPS_DIR/tomo.desktop"
fi

# ---- done -------------------------------------------------------------------
hr
ok "Tomo is set up."
echo
echo -e "${c_bold}Next steps:${c_reset}"
echo "  1. Edit ${c_cya}$SCRIPT_DIR/.env${c_reset} and add your ANTHROPIC_API_KEY."
echo "  2. Drop a VRoid .vrm at ${c_cya}$DATA_DIR/characters/default.vrm${c_reset}"
echo "     (or set TOMO_CHARACTER in .env, or import one from the chat window)."
echo "  3. Launch from your app menu (\"Tomo\"), or run:"
echo -e "        ${c_cya}TOMO_ROOT=$SCRIPT_DIR $BIN_DIR/tomo${c_reset}"
echo
echo "  Then just say \"Hey Tomo\" — or click her to chat."
echo "  Tip: on GNOME Wayland, always-on-top is limited — see README.md."
hr
