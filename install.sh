#!/usr/bin/env bash
# ============================================================================
#  Semester Project installer (Linux / Raspberry Pi 5 / macOS)
#
#  Full PC install (training + optimization):   bash install.sh
#  Raspberry Pi inference-only install:         bash install.sh --edge
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

REQS="requirements.txt"
if [[ "${1:-}" == "--edge" ]]; then
  REQS="requirements_edge.txt"
  echo "[mode] EDGE install - inference-only (Raspberry Pi)"
else
  echo "[mode] FULL install - training + optimization"
fi

echo "[1/3] Checking Python..."
command -v python3 >/dev/null || { echo "python3 not found - install it first"; exit 1; }
python3 --version

echo "[2/3] Creating virtual environment (.venv)..."
[[ -d .venv ]] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "[3/3] Installing libraries from $REQS ..."
pip install -r "$REQS"

echo
echo "Done. Activate with:  source .venv/bin/activate"
if [[ "$REQS" == "requirements_edge.txt" ]]; then
  echo "Run detection:  python detect.py --variant pruned_quantized --source rtsp://<pc-ip>:8554/cam1"
else
  echo "Start with:     python download.py"
fi
