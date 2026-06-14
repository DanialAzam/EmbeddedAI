#!/usr/bin/env bash
# ============================================================================
#  Launch the control-panel GUI on Linux / Raspberry Pi.   Run:  bash gui.sh
#  Needs a desktop session (VNC or HDMI) and tkinter (python3-tk).
#  On a Pi the GUI auto-hides the Train/Optimize tabs (inference + benchmark only).
# ============================================================================
set -e
cd "$(dirname "$0")"

# Prefer the venv python (consistent with the subprocesses it launches),
# fall back to the system python3.
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
  echo "[gui] tkinter not found. Install it once with:"
  echo "      sudo apt install -y python3-tk"
  exit 1
fi

if [ -z "${DISPLAY:-}" ]; then
  echo "[gui] No DISPLAY detected. The GUI needs a desktop (open it from your VNC"
  echo "      session, not a plain SSH terminal). For headless detection use:"
  echo "      bash run_on_pi.sh"
  exit 1
fi

exec "$PY" gui.py "$@"
