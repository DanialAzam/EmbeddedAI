#!/usr/bin/env bash
# ============================================================================
#  Run vehicle detection on the Raspberry Pi using a BUNDLED DEMO VIDEO.
#  No camera, no LAN, no ONVIF - everything is local to this folder.
#
#    bash run_on_pi.sh                                    # defaults below
#    bash run_on_pi.sh quantized_static.onnx              # pick a model
#    bash run_on_pi.sh pruned.onnx highway-busy.mp4       # model + video
#
#  Over SSH with no screen? It auto-switches to headless (prints stats only).
# ============================================================================
set -e
cd "$(dirname "$0")"

MODEL="${1:-models/quantized_static.onnx}"   # deployable INT8 by default
VIDEO="${2:-dataset/demo/highway-busy.mp4}"   # busiest demo clip by default

# Accept bare names ("pruned.onnx" / "highway-busy.mp4") too.
[ -f "$MODEL" ] || MODEL="models/$MODEL"
[ -f "$VIDEO" ] || VIDEO="dataset/demo/$VIDEO"

if [ ! -d ".venv" ]; then
  echo "[run_on_pi] no .venv yet - installing inference dependencies first..."
  python3 install.py --edge --no-pause
fi

if [ ! -f "$MODEL" ]; then
  echo "[run_on_pi] model not found: $MODEL"
  echo "available models:"; ls -1 models/*.onnx 2>/dev/null || echo "  (none - copy them from the PC)"
  exit 1
fi
if [ ! -f "$VIDEO" ]; then
  echo "[run_on_pi] video not found: $VIDEO"
  echo "available demo videos:"; ls -1 dataset/demo/*.mp4 2>/dev/null || echo "  (none)"
  exit 1
fi

# Headless if no display is attached (e.g. plain SSH).
DISP=""
[ -z "${DISPLAY:-}" ] && DISP="--no-display" && echo "[run_on_pi] no DISPLAY - running headless (stats only)."

source .venv/bin/activate
echo "[run_on_pi] model=$MODEL  video=$VIDEO"
python detect.py --model "$MODEL" --source "$VIDEO" $DISP
