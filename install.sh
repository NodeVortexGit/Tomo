#!/usr/bin/env bash
# =============================================================================
#  Tomo — the Linux installer
#
#  Sets up everything Tomo needs. Tomo runs entirely on this computer: no
#  cloud services, no API keys.
#    • system packages: Python, OpenGL, audio tools, a screenshot tool, a file
#      dialog, clipboard tools (to paste pictures into the chat), xdotool (for
#      the mouse/keyboard control)
#    • Tomo's Python environment (.venv, with uv if it's there, else venv+pip)
#    • the speech and camera models (Piper voices, Vosk, Parakeet, Pose Lite)
#    • a .env from the template, and an entry in the app menu
#    • a check for a local model server (Ollama or LM Studio), offering to
#      download a model for Ollama
#
#  (Windows has its own installer: Tomo-Setup-<version>.exe, built by the
#  project's GitHub Actions from packaging/windows/.)
#
#  Package managers: apt, dnf, pacman, zypper. Other distributions: install
#  the equivalents of the lists below and run with TOMO_SKIP_SYSDEPS=1.
#
#  Usage:   ./install.sh
#  Options (environment variables):
#     TOMO_SKIP_SYSDEPS=1   don't touch system packages
#     TOMO_SKIP_MODELS=1    don't download the speech and camera models
#     TOMO_MODEL=qwen3.5:9b the Ollama model to offer
# =============================================================================
set -euo pipefail

c_reset='\033[0m'; c_bold='\033[1m'; c_grn='\033[32m'; c_yel='\033[33m'; c_red='\033[31m'; c_cya='\033[36m'
say()  { printf "${c_cya}▸${c_reset} %s\n" "$*"; }
ok()   { printf "${c_grn}✓${c_reset} %s\n" "$*"; }
warn() { printf "${c_yel}!${c_reset} %s\n" "$*"; }
die()  { printf "${c_red}✗ %s${c_reset}\n" "$*" >&2; exit 1; }
hr()   { printf "${c_bold}%s${c_reset}\n" "────────────────────────────────────────────────────────"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS_DIR="$HOME/.local/share/applications"
DATA_DIR="${TOMO_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/tomo}"
MODEL="${TOMO_MODEL:-qwen3.5:9b}"
VENV="$ROOT/.venv"

hr
printf "${c_bold}  Tomo installer${c_reset}  —  a local AI VRM desktop companion\n"
hr

# ---- 0. .env first: you always end up with settings to edit ---------------------
if [ ! -f "$ROOT/.env" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    ok "Created .env from the template → $ROOT/.env (the defaults work as they are)"
else
    ok ".env already exists (left as it is)."
fi

# ---- 1. system packages -------------------------------------------------------------------
PM=""
for candidate in apt-get dnf pacman zypper; do
    if command -v "$candidate" >/dev/null 2>&1; then PM="$candidate"; break; fi
done

install_sysdeps() {
    [ -n "${TOMO_SKIP_SYSDEPS:-}" ] && { warn "Skipping system packages (TOMO_SKIP_SYSDEPS)."; return; }
    [ -n "$PM" ] || { warn "No supported package manager: skipping system packages."; return; }
    say "Installing system packages with $PM (may ask for sudo)…"
    case "$PM" in
      apt-get)
        sudo apt-get update -y
        sudo apt-get install -y python3 python3-venv python3-pip curl \
          libgl1 libegl1 libglib2.0-0 \
          pulseaudio-utils alsa-utils grim zenity xdotool wl-clipboard xclip ;;
      dnf)
        sudo dnf install -y python3 python3-pip curl \
          mesa-libGL mesa-libEGL glib2 \
          pulseaudio-utils alsa-utils grim zenity xdotool wl-clipboard xclip ;;
      pacman)
        sudo pacman -Sy --needed --noconfirm python python-pip curl \
          mesa glib2 \
          libpulse alsa-utils grim zenity xdotool wl-clipboard xclip ;;
      zypper)
        sudo zypper --non-interactive install -y python3 python3-pip curl \
          Mesa-libGL1 Mesa-libEGL1 glib2 \
          pulseaudio-utils alsa-utils grim zenity xdotool wl-clipboard xclip ;;
    esac
    ok "System packages installed."
}
install_sysdeps || warn "Some system packages didn't install — carrying on; install them by hand if something is missing."

# ---- 2. Tomo's Python environment ---------------------------------------------------------
say "Setting up Tomo's Python environment in .venv…"
if command -v uv >/dev/null 2>&1; then
    [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" -r "$ROOT/pyproject.toml" --extra all
else
    command -v python3 >/dev/null 2>&1 || die "Python 3 is needed (3.11 or newer)."
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip >/dev/null
    "$VENV/bin/pip" install "$ROOT[all]"
fi
ok "Python environment ready: $VENV"

# ---- 3. the speech and camera models (they run on this computer) ------------------------
if [ -n "${TOMO_SKIP_MODELS:-}" ]; then
    warn "Skipping the models (TOMO_SKIP_MODELS); later: .venv/bin/python scripts/setup_models.py"
else
    say "Downloading the speech and camera models (about 1 GB, once)…"
    if TOMO_DATA_DIR="$DATA_DIR" "$VENV/bin/python" "$ROOT/scripts/setup_models.py"; then
        ok "Models ready in $DATA_DIR"
    else
        warn "Some models didn't download; run .venv/bin/python scripts/setup_models.py again later."
    fi
fi

# ---- 4. a local model to think with ---------------------------------------------------------
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

# ---- 5. the app menu -------------------------------------------------------------------------
mkdir -p "$APPS_DIR" "$DATA_DIR/characters"
cat > "$APPS_DIR/tomo.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Tomo
Comment=A local AI VRM desktop companion
Exec=env TOMO_ROOT=$ROOT $VENV/bin/python -m tomo
Path=$ROOT
Terminal=false
Categories=Utility;
StartupWMClass=tomo.desktop.companion
EOF
ok "App menu entry → $APPS_DIR/tomo.desktop"

# ---- done ------------------------------------------------------------------------------------
hr
ok "Tomo is set up."
echo
echo -e "${c_bold}Next:${c_reset}"
echo "  1. Make sure Ollama (or LM Studio's server) runs with a model:"
echo -e "        ${c_cya}ollama pull $MODEL${c_reset}"
echo "  2. Settings are in $ROOT/.env; more characters can go in $DATA_DIR/characters/"
echo "     or be imported from the chat."
echo "  3. Start Tomo from the app menu, or:"
echo -e "        ${c_cya}$VENV/bin/python -m tomo${c_reset}"
echo
echo "  Then say \"Hey Tomo\" — or click her to chat."
echo "  A compositor is needed for the see-through window (on X11: picom, kwin, mutter…)."
hr
