#!/usr/bin/env bash
# =============================================================================
#  Tomo — one-shot installer
#
#  Installs everything Tomo needs and builds it. Tomo runs entirely on this
#  computer: no cloud services, no API keys.
#    • system libraries for Bevy (graphics/X11/Wayland/Vulkan) + audio I/O
#    • the Rust toolchain (via rustup) if it's missing
#    • a Python virtualenv with Piper (Tomo's voice) and Vosk + Whisper
#      ("Hey Tomo" and the Talk button), plus their models
#    • the release binary, a .env from the template, and a desktop entry
#    • a check for a local model server (Ollama or LM Studio), offering to
#      download a model for Ollama
#
#  (Windows has its own installer: Tomo-Setup-<version>.exe, built by the
#  project's GitHub Actions from packaging/windows/.)
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
#     TOMO_MODEL=qwen2.5:7b the Ollama model to offer (default qwen2.5:7b)
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
MODEL="${TOMO_MODEL:-qwen2.5:7b}"

hr
printf "${c_bold}  Tomo installer${c_reset}  —  AI VRM desktop companion\n"
hr

# ---- 0. .env FIRST ----------------------------------------------------------
# Done before anything that could fail, so you always end up with a .env to
# edit even if a later step (packages, rustup, pip) errors out.
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    ok "Created .env from template → $SCRIPT_DIR/.env (the defaults work as they are)"
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
          pulseaudio-utils alsa-utils grim zenity
        ;;
      dnf)
        sudo dnf install -y \
          gcc gcc-c++ pkgconf-pkg-config curl clang \
          alsa-lib-devel systemd-devel \
          libX11-devel libXcursor-devel libXrandr-devel libXi-devel libxkbcommon-devel \
          wayland-devel \
          vulkan-loader mesa-vulkan-drivers vulkan-tools \
          python3 python3-pip \
          pulseaudio-utils alsa-utils grim zenity
        ;;
      pacman)
        sudo pacman -Sy --needed --noconfirm \
          base-devel pkgconf curl clang \
          alsa-lib systemd-libs \
          libx11 libxcursor libxrandr libxi libxkbcommon \
          wayland \
          vulkan-icd-loader vulkan-tools \
          python python-pip \
          libpulse alsa-utils grim zenity
        ;;
      zypper)
        sudo zypper --non-interactive install -y \
          gcc gcc-c++ pkg-config curl clang \
          alsa-devel systemd-devel \
          libX11-devel libXcursor-devel libXrandr-devel libXi-devel libxkbcommon-devel \
          wayland-devel \
          vulkan-loader vulkan-tools \
          python3 python3-pip \
          pulseaudio-utils alsa-utils grim zenity
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
say "Setting up the Python virtualenv for speech (Piper, Vosk, Whisper)…"
VENV="$SCRIPT_DIR/scripts/.venv"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
"$VENV/bin/pip" install --upgrade pip >/dev/null
"$VENV/bin/pip" install -r "$SCRIPT_DIR/scripts/requirements.txt"
ok "Speech venv ready at scripts/.venv"

# ---- 5. speech models (they run on this computer) ---------------------------
say "Downloading the speech models (Vosk, Whisper and Tomo's voice, ~250 MB)…"
if TOMO_DATA_DIR="$DATA_DIR" "$VENV/bin/python" "$SCRIPT_DIR/scripts/setup_models.py"; then
    ok "Speech models ready in $DATA_DIR"
else
    warn "Some speech models didn't download; run scripts/setup_models.py again later."
fi

# ---- 5b. a local model to think with ---------------------------------------
if command -v ollama >/dev/null 2>&1; then
    ok "Ollama found."
    if ! ollama list 2>/dev/null | awk 'NR > 1 && $3 != "-" { found = 1 } END { exit !found }'; then
        warn "Ollama has no local model yet; Tomo needs one to think."
        if [ -t 0 ]; then
            read -r -p "  Download $MODEL now (a few GB)? [y/N] " answer
            case "$answer" in
                [yY]*) ollama pull "$MODEL" && ok "Model ready: $MODEL" ;;
                *) warn "Later: ollama pull $MODEL" ;;
            esac
        else
            warn "Later: ollama pull $MODEL"
        fi
    fi
elif command -v lms >/dev/null 2>&1; then
    ok "LM Studio found: start its local server and load a model."
else
    warn "No local model server found. Install Ollama (https://ollama.com/download)"
    warn "or LM Studio (https://lmstudio.ai), then get a model: ollama pull $MODEL"
fi

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
echo "  1. Make sure Ollama (or LM Studio's server) is running with a model:"
echo -e "        ${c_cya}ollama pull $MODEL${c_reset}"
echo "  2. Optional: settings live in ${c_cya}$SCRIPT_DIR/.env${c_reset}; other characters"
echo "     can go in ${c_cya}$DATA_DIR/characters/${c_reset} or be imported from the chat."
echo "  3. Launch from your app menu (\"Tomo\"), or run:"
echo -e "        ${c_cya}TOMO_ROOT=$SCRIPT_DIR $BIN_DIR/tomo${c_reset}"
echo
echo "  Then just say \"Hey Tomo\" — or click her to chat."
echo "  Tip: on GNOME Wayland, always-on-top is limited — see README.md."
hr
